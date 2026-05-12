"""Shared gold reward-model loading and scoring utilities."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    LlamaConfig,
    LlamaModel,
    PreTrainedModel,
)


ULTRARM_TEMPLATE_NAME = "ultrarm_human_assistant"
SEQCLS_TEMPLATE_NAME = "chat_template"


class LlamaRewardModel(PreTrainedModel):
    """UltraRM reward model architecture from the official model card."""

    config_class = LlamaConfig
    _tied_weights_keys = []
    all_tied_weights_keys = {}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.regression_head = nn.Linear(self.config.hidden_size, 1, bias=False)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        del labels, use_cache, output_attentions, output_hidden_states, return_dict
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = transformer_outputs[0]
        rewards = self.regression_head(hidden_states).squeeze(-1)

        if attention_mask is None:
            ends = torch.full(
                (rewards.shape[0], 1),
                rewards.shape[1] - 1,
                dtype=torch.long,
                device=rewards.device,
            )
        else:
            ends = attention_mask.cumsum(dim=1).argmax(dim=1).view(-1, 1)
        return torch.gather(rewards, 1, ends).squeeze(-1)


def is_ultrarm_model(model_name_or_path: str, config=None) -> bool:
    """Return whether a model should use the UltraRM reward architecture."""
    name = str(model_name_or_path).rstrip("/").lower()
    if name == "openbmb/ultrarm-13b" or name.endswith("/ultrarm-13b"):
        return True

    architectures = getattr(config, "architectures", None) or []
    return any(str(arch).lower() == "llamarewardmodel" for arch in architectures)


def gold_scorer_metadata_for_model(model_name_or_path: str) -> dict:
    """Return scorer/template metadata without loading full model weights."""
    config = None
    config_path = os.path.join(str(model_name_or_path), "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = SimpleConfig(json.load(f))

    if is_ultrarm_model(model_name_or_path, config):
        return {
            "gold_scorer_type": "ultrarm",
            "gold_prompt_template": ULTRARM_TEMPLATE_NAME,
        }
    return {
        "gold_scorer_type": "seqcls",
        "gold_prompt_template": SEQCLS_TEMPLATE_NAME,
    }


class SimpleConfig:
    def __init__(self, data: dict):
        self.architectures = data.get("architectures")


def _model_device(model) -> torch.device:
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


@dataclass
class GoldRewardScorer:
    """Thin wrapper over UltraRM or sequence-classification gold RMs."""

    model: torch.nn.Module
    tokenizer: object
    scorer_type: str
    prompt_template: str

    @property
    def is_ultrarm(self) -> bool:
        return self.scorer_type == "ultrarm"

    @property
    def cache_metadata(self) -> dict:
        return {
            "gold_scorer_type": self.scorer_type,
            "gold_prompt_template": self.prompt_template,
        }

    def format_pairs(
        self,
        prompts: Sequence[str],
        responses: Sequence[str],
    ) -> List[str]:
        if self.is_ultrarm:
            return [
                f"Human: {prompt}\n\nAssistant: {response}"
                for prompt, response in zip(prompts, responses)
            ]

        texts: List[str] = []
        for prompt, response in zip(prompts, responses):
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            texts.append(
                self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            )
        return texts

    @torch.no_grad()
    def score_input_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.is_ultrarm:
            return self.model(input_ids=input_ids, attention_mask=attention_mask)

        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        return logits.squeeze(-1)

    @torch.no_grad()
    def score_texts(
        self,
        texts: Sequence[str],
        batch_size: int,
        max_length: int,
        desc: str,
    ) -> List[float]:
        scores: List[float] = []
        device = _model_device(self.model)
        for start in tqdm(range(0, len(texts), batch_size), desc=desc):
            batch = list(texts[start : start + batch_size])
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            batch_scores = self.score_input_ids(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
            )
            scores.extend(batch_scores.float().cpu().tolist())
        return scores


def load_gold_reward_scorer(
    model_name_or_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    use_cache: bool = False,
) -> Tuple[GoldRewardScorer, object]:
    """Load a gold RM and return a scorer plus its tokenizer."""
    config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if is_ultrarm_model(model_name_or_path, config):
        model = LlamaRewardModel.from_pretrained(
            model_name_or_path,
            dtype=dtype,
        )
        model.config.use_cache = use_cache
        scorer_type = "ultrarm"
        prompt_template = ULTRARM_TEMPLATE_NAME
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            dtype=dtype,
            num_labels=1,
            use_cache=use_cache,
        )
        scorer_type = "seqcls"
        prompt_template = SEQCLS_TEMPLATE_NAME

    model = model.to(device)
    if hasattr(model, "config"):
        model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    scorer = GoldRewardScorer(
        model=model,
        tokenizer=tokenizer,
        scorer_type=scorer_type,
        prompt_template=prompt_template,
    )
    return scorer, tokenizer
