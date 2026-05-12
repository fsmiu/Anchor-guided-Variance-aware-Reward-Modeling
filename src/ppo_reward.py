"""Proxy reward wrapper for PPO training.

Wraps a trained reward-model checkpoint (BT or MeanVar) and exposes two
scoring interfaces:

- ``score_for_ppo``: the PPO reward signal. For MeanVar RMs, applies the
  selected quantile correction (q5/q10/.../q95 in 5% steps with scalar
  Phi^{-1}, or ``random`` which draws a fresh quantile per sample).
- ``score_for_eval``: point estimate used only for evaluation. Always
  returns mu (for MV) or r (for BT), regardless of ``reward_mode``.

RM weights are loaded via ``src.evaluate.load_model`` to reuse the
existing safetensors / bin / sharded fallback logic.
"""

from __future__ import annotations

import torch
from torch.distributions import Normal
from transformers import AutoModelForSequenceClassification

from src.evaluate import load_model


VALID_REWARD_MODES = (
    "q5", "q10", "q15", "q20", "q25", "q30", "q35", "q40", "q45",
    "q50",
    "q55", "q60", "q65", "q70", "q75", "q80", "q85", "q90", "q95",
    "random",
)

# Phi^{-1}(q) for the fixed quantile modes. Using high-precision literals so
# we do not pay the cost of calling Normal(0,1).icdf for every forward pass.
_PHI_INV_SCALAR = {
    "q5":  -1.6448536269514729,
    "q10": -1.2815515655446004,
    "q15": -1.0364333894937898,
    "q20": -0.8416212335729143,
    "q25": -0.6744897501960817,
    "q30": -0.5244005127080409,
    "q35": -0.3853204664075677,
    "q40": -0.2533471031357997,
    "q45": -0.12566134685507402,
    "q50":  0.0,
    "q55":  0.12566134685507402,
    "q60":  0.2533471031357997,
    "q65":  0.3853204664075677,
    "q70":  0.5244005127080409,
    "q75":  0.6744897501960817,
    "q80":  0.8416212335729143,
    "q85":  1.0364333894937898,
    "q90":  1.2815515655446004,
    "q95":  1.6448536269514729,
}


class ProxyRewardWrapper:
    """Proxy RM with mode-dependent PPO scoring and mode-independent eval scoring."""

    def __init__(
        self,
        checkpoint_dir: str,
        rm_type: str,
        reward_mode: str,
        quantile_range: tuple = (0.05, 0.95),
        device: str = "cuda",
    ):
        if rm_type not in ("bt", "meanvar", "anchor"):
            raise ValueError(
                f"rm_type must be 'bt', 'meanvar', or 'anchor', got {rm_type!r}"
            )
        if reward_mode not in VALID_REWARD_MODES:
            raise ValueError(
                f"reward_mode must be one of {VALID_REWARD_MODES}, got {reward_mode!r}"
            )
        if rm_type == "bt" and reward_mode != "q50":
            raise ValueError(
                "BT reward models only support reward_mode='q50' (no sigma)."
            )
        if rm_type == "anchor" and reward_mode != "q50":
            raise ValueError(
                "Anchor (external seq-classification) RMs only support "
                "reward_mode='q50' (no sigma)."
            )
        lo, hi = quantile_range
        if not (0.0 < lo < hi < 1.0):
            raise ValueError(f"quantile_range must satisfy 0<lo<hi<1, got {quantile_range}")

        self.checkpoint_dir = checkpoint_dir
        self.rm_type = rm_type
        self.reward_mode = reward_mode
        self.quantile_range = quantile_range
        self.device = device

        if rm_type == "anchor":
            # External AutoModelForSequenceClassification reward model (e.g. Skywork).
            # Load directly from HF Hub id or local path; tokenizer source is the
            # same identifier.
            model = AutoModelForSequenceClassification.from_pretrained(
                checkpoint_dir,
                num_labels=1,
                dtype=torch.bfloat16,
            )
            model = model.to(device)
            model.eval()
            self.model = model
            self.base_model_name = checkpoint_dir
        else:
            # Delegate weight loading (safetensors / bin / sharded fallback handled there).
            self.model, self.base_model_name = load_model(
                checkpoint_dir, model_type=rm_type, device=device
            )
            self.model.eval()

    @torch.no_grad()
    def _forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Run the RM forward pass.

        Returns
        -------
        (mu, sigma) for MV: sigma is standard deviation (sqrt of the head's variance).
        (r, None)  for BT or anchor (external seq-classification RM).
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        if self.rm_type == "anchor":
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return out.logits.squeeze(-1), None
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        if self.rm_type == "bt":
            return out, None
        mean, var = out
        # MeanVarRewardModel.forward returns variance (via softplus); take sqrt so
        # downstream code works with std dev (matches the paper's sigma notation).
        sigma = torch.sqrt(var.clamp_min(1e-12))
        return mean, sigma

    @torch.no_grad()
    def score_for_ppo(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return the per-sample reward signal used by PPO.

        - BT:                       r
        - MV + q5/q10/.../q95:      mu + Phi^{-1}(q) * sigma        (scalar Phi^{-1}, 5% grid)
        - MV + random:              mu + Phi^{-1}(q_i) * sigma_i,   q_i ~ U(lo,hi) per sample
        """
        mu, sigma = self._forward(input_ids, attention_mask)
        if self.rm_type in ("bt", "anchor"):
            return mu  # (batch,), scalar r

        if self.reward_mode in _PHI_INV_SCALAR:
            return mu + _PHI_INV_SCALAR[self.reward_mode] * sigma

        # "random" — independent q per sample.
        lo, hi = self.quantile_range
        # Draw q in fp32 so icdf has enough precision; cast back afterwards.
        q = torch.empty_like(mu, dtype=torch.float32).uniform_(lo, hi)
        normal = Normal(
            torch.tensor(0.0, device=q.device),
            torch.tensor(1.0, device=q.device),
        )
        phi_inv_q = normal.icdf(q).to(mu.dtype)
        return mu + phi_inv_q * sigma

    @torch.no_grad()
    def score_for_eval(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return mu (MV) or r (BT).

        Invariant: this method ignores ``self.reward_mode`` — all runs produce
        comparable proxy-eval scores under the RM's best point estimate. The
        invariant is guaranteed by construction (we never read reward_mode here);
        the ``reward_mode`` attribute is still exposed so callers can assert it.
        """
        mu, _ = self._forward(input_ids, attention_mask)
        return mu