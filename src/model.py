"""Reward model architectures: BT and Mean-Var (Gaussian/Thurstonian)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoModel, PreTrainedModel,AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutputWithPast


class BTRewardModel(PreTrainedModel):
    """Bradley-Terry reward model with a single scalar reward head."""

    supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)
        self.backbone = AutoModel.from_config(config, attn_implementation="sdpa")
        self.reward_head = nn.Linear(config.hidden_size, 1, bias=False)
        self.post_init()

    def get_last_hidden_state(self, input_ids, attention_mask):
        """Extract last non-padding token hidden state."""
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                                use_cache=False)
        hidden = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)
        # Get index of last non-padding token
        last_token_idx = attention_mask.sum(dim=1) - 1  # (batch,)
        last_hidden = hidden[torch.arange(hidden.size(0), device=hidden.device), last_token_idx]
        return last_hidden

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        last_hidden = self.get_last_hidden_state(input_ids, attention_mask)
        reward = self.reward_head(last_hidden).squeeze(-1)  # (batch,)
        return reward

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.backbone.gradient_checkpointing_disable()


class MeanVarRewardModel(PreTrainedModel):
    """Mean-Variance (Gaussian/Thurstonian) reward model with mean and variance heads."""

    supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)
        self.backbone = AutoModel.from_config(config, attn_implementation="sdpa")
        self.mean_head = nn.Linear(config.hidden_size, 1, bias=False)
        self.var_head = nn.Linear(config.hidden_size, 1, bias=False)
        self.post_init()

    def get_last_hidden_state(self, input_ids, attention_mask):
        """Extract last non-padding token hidden state."""
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                                use_cache=False)
        hidden = outputs.last_hidden_state
        last_token_idx = attention_mask.sum(dim=1) - 1
        last_hidden = hidden[torch.arange(hidden.size(0), device=hidden.device), last_token_idx]
        return last_hidden

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        last_hidden = self.get_last_hidden_state(input_ids, attention_mask)
        mean = self.mean_head(last_hidden).squeeze(-1)          # (batch,)
        var = F.softplus(self.var_head(last_hidden)).squeeze(-1) + 1e-6  # (batch,), variance > 0
        return mean, var

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.backbone.gradient_checkpointing_disable()


def create_reward_model(model_name_or_path, model_type="bt", torch_dtype=torch.bfloat16):
    """Create a reward model from a pretrained LLM.

    Args:
        model_name_or_path: HuggingFace model name or path.
        model_type: "bt" or "meanvar".
        torch_dtype: Model precision.

    Returns:
        Reward model instance.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name_or_path)
    config.num_labels = 1  # required by TRL RewardTrainer


    # Create model shell on meta device (no actual memory allocated)
    with torch.device("meta"):
        if model_type == "bt":
            model = BTRewardModel(config)
        elif model_type == "meanvar":
            model = MeanVarRewardModel(config)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    # Load pretrained backbone (only copy in memory)
    model.backbone = AutoModel.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation="sdpa",
    )

    # Materialize reward heads from meta to real device
    for name, module in model.named_children():
        if name == "backbone":
            continue
        module.to_empty(device="cpu")
        module.to(torch_dtype)
        # Re-initialize head weights
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()

    return model