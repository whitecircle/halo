"""The offline-GRPO per-token objective — one definition for the non-PP loss and its pipeline twin.

Pure tensor math over one batch, or one pipeline microbatch, of completion log-probs: the policy
term, the capped k3 KL against the reference, and the per-token quantities both paths buffer as
diagnostics. What stays with the caller is what genuinely differs — the ``min_log_prob`` clamp, which
each path applies where it computes its log-probs, and the reduction: a rank-local quotient off PP, a
microbatch sum under it (the runtime divides by the full-batch normalizer).
"""

from __future__ import annotations

import torch

from src.trainers.grpo.objective.logratio import clamp_ref_logps


def offline_token_objective(
    token_logps: torch.Tensor,
    token_logps_unclamped: torch.Tensor,
    advantages: torch.Tensor,
    *,
    policy_gradient_formulation: str,
    beta: float = 0.0,
    ref_logps: torch.Tensor | None = None,
    ref_logps_unclamped: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """The per-token loss — unmasked and unweighted — plus the per-token diagnostics to buffer.

    ``prob_weighted`` L = -(π·A) weights high-prob tokens more; ``reinforce`` L = -(log π·A) is
    uniform. At ``beta != 0`` the capped k3 KL ``exp(Δ) - Δ - 1`` is added on top of the reward
    term, which the diagnostics therefore capture before the KL lands.

    ``clamp_ref_logps`` is fed the DETACHED policy log-probs: the ceiling is ``policy + the clamp``,
    so a grad-carrying policy tensor would make ``ref_clamped - logp`` that constant on every
    clamped token — zeroing the KL gradient there instead of bounding it. Its clamped fraction
    is dropped rather than logged: this runs per gradient-accumulation microbatch, where the
    ``.item()`` to read it is the host sync the buffered diagnostics exist to avoid.
    """
    weights = torch.exp(token_logps) if policy_gradient_formulation == "prob_weighted" else token_logps
    per_token_loss = -(weights * advantages.unsqueeze(1))
    sample_values = {
        "logps": token_logps,
        "logps_unclamped": token_logps_unclamped,
        "rewards": -per_token_loss,
    }
    if beta != 0.0:
        ref_logps, _ = clamp_ref_logps(ref_logps, token_logps.detach())
        per_token_kl = torch.exp(ref_logps - token_logps) - (ref_logps - token_logps) - 1
        per_token_loss = per_token_loss + beta * per_token_kl
        sample_values |= {"kl": per_token_kl, "ref_logps": ref_logps, "ref_logps_unclamped": ref_logps_unclamped}
    return per_token_loss, sample_values
