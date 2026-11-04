#!/usr/bin/env python
"""SMPO percentile clipping.
The bound carries no gradient, or the quantile element would take ``clamped_count`` times the gradient."""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from halo.trainers.preference.smpo_objective import SMPOObjective, cp_quantile
from tests.common.gloo import run_gloo_ranks

CP_WORLD_SIZE = 2


def _clipper(lower=None, upper=None, min_log_prob=None) -> SMPOObjective:
    return SMPOObjective(
        beta=1.0,
        loss_type="sigmoid",
        chosen_sft_ratio=1.0,
        lower_clip_percentile=lower,
        upper_clip_percentile=upper,
        min_log_prob=min_log_prob,
    )


def _rejected_logps(rows=1, seq=600, seed=0):
    """Log-probs of a well-SFT'd policy: mostly near 0 with a thin low tail (Exponential(3))."""
    generator = torch.Generator().manual_seed(seed)
    uniform = torch.rand(rows, seq, generator=generator).clamp(min=1e-9)
    return uniform.log() / 3.0


def test_percentile_clip_bound_carries_no_gradient():
    """A correct floor gives each token a gradient of exactly 1 (unclamped) or 0 (clamped)."""
    logps = _rejected_logps().requires_grad_(True)
    loss_mask = torch.ones_like(logps, dtype=torch.bool)
    is_chosen = torch.zeros(logps.size(0), dtype=torch.bool).unsqueeze(1)

    clipper = _clipper(lower=0.02)
    clipped = clipper.clip_log_probs(logps, loss_mask, is_chosen, cp_config=None)
    clipped.sum().backward()

    max_grad = logps.grad.abs().max().item()
    assert max_grad <= 1.0 + 1e-6, (
        f"a single token received {max_grad:.1f}x gradient through the percentile clip — the "
        f"quantile bound is differentiable, so every clamped token back-propagates into it"
    )


def test_padding_free_percentile_clip_bound_carries_no_gradient():
    flat = _rejected_logps(rows=1, seq=800).reshape(-1).requires_grad_(True)
    loss_mask = torch.ones_like(flat, dtype=torch.bool)
    is_chosen = torch.zeros_like(flat, dtype=torch.bool)

    clipper = _clipper(lower=0.02, upper=0.98)
    clipped = clipper.clip_log_probs(flat, loss_mask, is_chosen)
    clipped.sum().backward()

    assert flat.grad.abs().max().item() <= 1.0 + 1e-6


def test_clip_still_clamps_the_low_tail():
    logps = _rejected_logps()
    loss_mask = torch.ones_like(logps, dtype=torch.bool)
    is_chosen = torch.zeros(logps.size(0), dtype=torch.bool).unsqueeze(1)
    clipper = _clipper(lower=0.02)
    clipped = clipper.clip_log_probs(logps, loss_mask, is_chosen, cp_config=None)

    assert clipped.min().item() > logps.min().item(), "the low tail was not raised at all"
    bound = torch.quantile(logps.float(), 0.02)
    assert clipped.min().item() == pytest.approx(bound.item(), abs=1e-5)


def _row_split_clip(clipper, per_token_logps, loss_mask, num_chosen, cp_config=None):
    """Padded spelling: row-slice selectors, ``torch.where`` writes."""
    per_token_logps = per_token_logps.clone()
    if clipper.lower_clip_percentile is not None:
        rejected_logps = per_token_logps[num_chosen:][loss_mask[num_chosen:]]
        lower_bound = cp_quantile(rejected_logps, clipper.lower_clip_percentile, cp_config)
        if lower_bound is not None:
            per_token_logps[num_chosen:] = torch.where(
                loss_mask[num_chosen:] & (per_token_logps[num_chosen:] < lower_bound),
                lower_bound.to(per_token_logps.dtype),
                per_token_logps[num_chosen:],
            )
    if clipper.upper_clip_percentile is not None:
        chosen_logps = per_token_logps[:num_chosen][loss_mask[:num_chosen]]
        upper_bound = cp_quantile(chosen_logps, clipper.upper_clip_percentile, cp_config)
        if upper_bound is not None:
            per_token_logps[:num_chosen] = torch.where(
                loss_mask[:num_chosen] & (per_token_logps[:num_chosen] > upper_bound),
                upper_bound.to(per_token_logps.dtype),
                per_token_logps[:num_chosen],
            )
    if clipper.min_log_prob is not None:
        per_token_logps[num_chosen:] = torch.where(
            loss_mask[num_chosen:] & (per_token_logps[num_chosen:] < clipper.min_log_prob),
            clipper.min_log_prob,
            per_token_logps[num_chosen:],
        )
    return per_token_logps


def _boolean_mask_clip(clipper, per_token_logps, loss_mask, is_chosen):
    """Padding-free spelling: boolean-mask selectors, ``clamp`` writes."""
    per_token_logps = per_token_logps.clone()
    rejected_valid = loss_mask & ~is_chosen
    chosen_valid = loss_mask & is_chosen
    if clipper.lower_clip_percentile is not None:
        lower_bound = cp_quantile(per_token_logps[rejected_valid], clipper.lower_clip_percentile, None)
        if lower_bound is not None:
            per_token_logps[rejected_valid] = per_token_logps[rejected_valid].clamp(
                min=lower_bound.to(per_token_logps.dtype)
            )
    if clipper.upper_clip_percentile is not None:
        upper_bound = cp_quantile(per_token_logps[chosen_valid], clipper.upper_clip_percentile, None)
        if upper_bound is not None:
            per_token_logps[chosen_valid] = per_token_logps[chosen_valid].clamp(
                max=upper_bound.to(per_token_logps.dtype)
            )
    if clipper.min_log_prob is not None:
        per_token_logps[rejected_valid] = per_token_logps[rejected_valid].clamp(min=clipper.min_log_prob)
    return per_token_logps


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_the_clip_reproduces_both_selector_spellings(dtype):
    """A change to the rule (an order swap, a masked position touched, a dtype cast moved) breaks one comparison."""
    num_chosen, seq = 3, 400
    logps = _rejected_logps(rows=2 * num_chosen, seq=seq, seed=7).to(dtype)
    loss_mask = torch.ones_like(logps, dtype=torch.bool)
    loss_mask[:, -17:] = False  # padded tail: must stay untouched by every branch
    clipper = _clipper(lower=0.05, upper=0.9, min_log_prob=-2.0)

    row_is_chosen = (torch.arange(2 * num_chosen) < num_chosen).unsqueeze(1)
    clipped = clipper.clip_log_probs(logps, loss_mask, row_is_chosen, cp_config=None)

    reference = _row_split_clip(clipper, logps, loss_mask, num_chosen)
    assert torch.equal(clipped, reference), "the clip disagrees with the padded spelling"

    flat_logps = logps.reshape(-1)
    flat_mask = loss_mask.reshape(-1)
    flat_is_chosen = row_is_chosen.expand_as(logps).reshape(-1)
    clipped_flat = clipper.clip_log_probs(flat_logps, flat_mask, flat_is_chosen)
    assert torch.equal(clipped_flat, _boolean_mask_clip(clipper, flat_logps, flat_mask, flat_is_chosen)), (
        "the clip disagrees with the padding-free spelling"
    )
    assert torch.equal(clipped_flat.view_as(logps), clipped), (
        "padded and padding-free batches of the same data clip differently — the selector, not the "
        "rule, is supposed to be the only difference"
    )
    assert torch.equal(clipped[~loss_mask], logps[~loss_mask]), "a masked (padding) position was clipped"


def _cp_clip_worker(rank: int, out_path: str) -> None:
    """Rank-local slice + a real gloo CP group must clip exactly like one process holding it all."""
    num_chosen, seq = 2, 512
    logps = _rejected_logps(rows=2 * num_chosen, seq=seq, seed=11)
    loss_mask = torch.ones_like(logps, dtype=torch.bool)
    row_is_chosen = (torch.arange(2 * num_chosen) < num_chosen).unsqueeze(1)
    clipper = _clipper(lower=0.05, upper=0.9, min_log_prob=-2.0)

    whole = clipper.clip_log_probs(logps, loss_mask, row_is_chosen, cp_config=None)

    chunk = seq // CP_WORLD_SIZE
    sl = slice(rank * chunk, (rank + 1) * chunk)
    local = clipper.clip_log_probs(
        logps[:, sl],
        loss_mask[:, sl],
        row_is_chosen,
        cp_config=SimpleNamespace(process_group=dist.group.WORLD),
    )
    ok = torch.equal(local, whole[:, sl])
    if rank == 0:
        with open(out_path, "w") as fh:
            fh.write("PASS" if ok else "FAIL: CP-sliced clip differs from the whole-sequence clip")


def test_cp_sliced_clip_matches_the_whole_sequence_clip(tmp_path):
    """``cp_quantile``'s gather keeps the clip independent of ``cp_size``; a local quantile would differ per rank."""
    out = str(tmp_path / "result.txt")
    run_gloo_ranks(_cp_clip_worker, CP_WORLD_SIZE, out)
    with open(out) as fh:
        assert fh.read() == "PASS"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
