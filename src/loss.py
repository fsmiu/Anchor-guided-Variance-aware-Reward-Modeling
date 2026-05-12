"""Loss functions for BT and Mean-Var reward models, and anchor losses."""

import torch
import torch.nn.functional as F
from torch.distributions import Normal


def bt_loss(r_w, r_l, soft_label=None):
    """Bradley-Terry loss with optional soft labels.

    Hard label (soft_label=None): -log(sigmoid(r_w - r_l))
    Soft label: -[y * log(sigmoid(m)) + (1-y) * log(sigmoid(-m))]

    Args:
        r_w: Reward for chosen/winning response. Shape: (batch,)
        r_l: Reward for rejected/losing response. Shape: (batch,)
        soft_label: Soft label in [0.5, 1.0]. Shape: (batch,) or None.

    Returns:
        Scalar loss.
    """
    margin = r_w - r_l
    if soft_label is None:
        return -torch.nn.functional.logsigmoid(margin).mean()
    y = soft_label
    log_pos = torch.nn.functional.logsigmoid(margin)
    log_neg = torch.nn.functional.logsigmoid(-margin)
    return -(y * log_pos + (1 - y) * log_neg).mean()


def mean_var_loss(r_w, r_l, var_w, var_l, soft_label=None):
    """Thurstonian/Gaussian reward model loss with optional soft labels.

    Args:
        r_w: Mean reward for chosen. Shape: (batch,)
        r_l: Mean reward for rejected. Shape: (batch,)
        var_w: Variance for chosen. Shape: (batch,)
        var_l: Variance for rejected. Shape: (batch,)
        soft_label: Soft label in (0.5, 1.0]. Shape: (batch,) or None.
    """
    r_w, r_l = r_w.float(), r_l.float()
    var_w, var_l = var_w.float(), var_l.float()
    diff = r_w - r_l
    std = torch.sqrt(var_w + var_l)
    z = (diff / std).clamp(min=-6, max=6)  # Avoid extreme values for numerical stability
    normal = Normal(0, 1)
    if soft_label is None:
        return -normal.cdf(z).clamp(min=1e-7).log().mean()
    y = soft_label
    log_pos = normal.cdf(z).clamp(min=1e-7).log()
    log_neg = normal.cdf(-z).clamp(min=1e-7).log()
    return -(y * log_pos + (1 - y) * log_neg).mean()


# ---------------------------------------------------------------------------
# Anchor losses (applied independently to each response, not to pairs)
# ---------------------------------------------------------------------------

def weighted_mean(loss, sample_weight=None):
    """Mean loss, optionally over samples with positive weight."""
    if sample_weight is None:
        return loss.mean()

    weight = sample_weight.to(device=loss.device, dtype=loss.dtype)
    denom = weight.sum()
    denom = torch.where(denom > 0, denom, torch.ones_like(denom))
    return (loss * weight).sum() / denom


def anchor_1_loss(r_hat, var, r_tilde, tau, eps=1e-4, sample_weight=None):
    """Single-threshold binary anchor loss.

    Args:
        r_hat: Predicted mean reward. Shape: (batch,)
        var: Predicted variance. Shape: (batch,)
        r_tilde: External RM score. Shape: (batch,)
        tau: Threshold scalar.
    """
    r_hat = r_hat.float()
    var = var.float().clamp(min=1e-6, max=1e6)
    std = torch.sqrt(var)
    r_tilde = r_tilde.float()

    A_tau = (r_tilde >= float(tau)).float()
    normal = Normal(
        torch.tensor(0.0, device=r_hat.device, dtype=torch.float32),
        torch.tensor(1.0, device=r_hat.device, dtype=torch.float32),
    )
    z = ((r_hat - float(tau)) / std).clamp(min=-6, max=6)
    q = normal.cdf(z).clamp(min=eps, max=1.0 - eps)
    loss = -(A_tau * q.log() + (1 - A_tau) * (1 - q).log())
    return weighted_mean(loss, sample_weight=sample_weight)


def anchor_2_loss(r_hat, var, r_tilde, tau1, tau2, eps=1e-7, sample_weight=None):
    """Two-threshold ordinal anchor loss (Eq. 7).

    Multinomial cross-entropy over the three threshold-induced classes
    {(0,0), (1,0), (1,1)} derived from a shared latent utility with
    tau1 < tau2. Let p_k = Phi((r_hat - tau_k)/std) be the model's
    probability of exceeding tau_k. Then
        pi_0 = 1 - p_1,  pi_1 = p_1 - p_2,  pi_2 = p_2,
    and with A_k = 1{r_tilde >= tau_k},
        L = -E[(1 - A_1) log pi_0 + (A_1 - A_2) log pi_1 + A_2 log pi_2].
    """
    if not (float(tau1) < float(tau2)):
        raise ValueError(f"anchor_2_loss requires tau1 < tau2, got {tau1}, {tau2}")

    r_hat = r_hat.float()
    var = var.float().clamp(min=1e-6, max=1e6)
    std = torch.sqrt(var)
    r_tilde = r_tilde.float()

    normal = Normal(
        torch.tensor(0.0, device=r_hat.device, dtype=torch.float32),
        torch.tensor(1.0, device=r_hat.device, dtype=torch.float32),
    )
    z1 = ((r_hat - float(tau1)) / std).clamp(min=-6, max=6)
    z2 = ((r_hat - float(tau2)) / std).clamp(min=-6, max=6)
    p1 = normal.cdf(z1)
    p2 = normal.cdf(z2)

    pi0 = (1.0 - p1).clamp(min=eps, max=1.0 - eps)
    pi1 = (p1 - p2).clamp(min=eps, max=1.0 - eps)
    pi2 = p2.clamp(min=eps, max=1.0 - eps)

    A1 = (r_tilde >= float(tau1)).float()
    A2 = (r_tilde >= float(tau2)).float()

    log_lik = (1.0 - A1) * pi0.log() + (A1 - A2) * pi1.log() + A2 * pi2.log()
    return weighted_mean(-log_lik, sample_weight=sample_weight)
