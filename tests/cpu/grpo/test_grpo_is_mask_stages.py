"""CPU tests for the IS mask/veto stages (ISMaskConfig: trajectory geometric band, catastrophic veto,
OPSM). They must be exact no-ops at their defaults — the default path is the running baseline.

    python tests/cpu/grpo/test_grpo_is_mask_stages.py
"""

import pytest
import torch

from src.trainers.grpo.objective.logratio import ISMaskConfig, apply_is_masks, apply_opsm, compute_is_ratio


def _ratio_setup(logdiffs: torch.Tensor, clip_max: float = 3.0):
    """Build (ratio, logps_diff, corrected_mask) from a [rows, T] log-diff tensor, all tokens corrected.

    The sampling logprobs sit at -1 rather than 0: a sampling logprob of exactly 0 is the engine's
    "emitted with probability 1" marker, which ``compute_is_ratio`` leaves uncorrected.
    """
    rows, t = logdiffs.shape
    completion_mask = torch.ones(rows, t, dtype=torch.long)
    sampling = torch.full_like(logdiffs, -1.0)
    ratio, diff, corrected = compute_is_ratio(
        logdiffs + sampling, sampling, completion_mask, torch.ones(rows, dtype=torch.bool), clip_max
    )
    return ratio, diff, corrected


def test_forced_token_does_not_trip_the_veto():
    """A budget-forced ``</think>`` (sampling logprob 0, trainer logprob −12) must not veto the trajectory:
    the engine made no sampling choice there, so it is not evidence of a broken sync."""
    recompute = torch.tensor([[-0.5, -12.0, -0.7]])
    sampling = torch.tensor([[-0.5, 0.0, -0.7]])
    ratio, diff, corrected = compute_is_ratio(
        recompute, sampling, torch.ones(1, 3, dtype=torch.long), torch.ones(1, dtype=torch.bool), 3.0
    )
    out, stats = apply_is_masks(ratio, diff, corrected, torch.arange(1), ISMaskConfig(veto_min=1e-4))
    assert stats["sampling/is_veto_masked_frac"] == (0, 1)
    assert torch.equal(out, torch.ones(1, 3))
    # The same disagreement on a token the engine actually sampled is exactly what the veto is for.
    ratio, diff, corrected = compute_is_ratio(
        recompute,
        torch.tensor([[-0.5, -1e-3, -0.7]]),
        torch.ones(1, 3, dtype=torch.long),
        torch.ones(1, dtype=torch.bool),
        3.0,
    )
    out, stats = apply_is_masks(ratio, diff, corrected, torch.arange(1), ISMaskConfig(veto_min=1e-4))
    assert stats["sampling/is_veto_masked_frac"] == (1, 1)


def test_ismask_defaults_are_inert():
    cfg = ISMaskConfig()
    assert not cfg.any_mask_active
    ratio, diff, corrected = _ratio_setup(torch.randn(4, 5) * 0.01)
    out, stats = apply_is_masks(ratio, diff, corrected, torch.arange(4), cfg)
    assert out is ratio and stats == {}


def test_ismask_validation():
    with pytest.raises(ValueError, match="together"):
        ISMaskConfig(geo_band_min=0.99)
    with pytest.raises(ValueError, match="bounds"):
        ISMaskConfig(geo_band_min=1.2, geo_band_max=2.0)
    with pytest.raises(ValueError, match="veto"):
        ISMaskConfig(veto_min=2.0)
    with pytest.raises(ValueError, match="opsm"):
        ISMaskConfig(opsm_delta=-1.0)


def test_geo_band_masks_whole_trajectory_across_turn_rows():
    # Rows are pooled per trajectory: row 1 is clean but belongs to drifted trajectory 0, so it masks too.
    diff = torch.zeros(3, 4)
    diff[0] = 0.1
    diff[1] = 0.0
    ratio, d, corrected = _ratio_setup(diff)
    traj_ids = torch.tensor([0, 0, 1])
    out, stats = apply_is_masks(ratio, d, corrected, traj_ids, ISMaskConfig(geo_band_min=0.99, geo_band_max=1.01))
    assert (out[0] == 0).all() and (out[1] == 0).all()
    assert (out[2] == 1).all()
    assert stats["sampling/is_geo_band_masked_frac"] == (1, 2)


def test_veto_masks_trajectory_with_catastrophic_token():
    diff = torch.zeros(2, 4)
    diff[0, 3] = torch.log(torch.tensor(1e-5))
    ratio, d, corrected = _ratio_setup(diff)
    out, stats = apply_is_masks(ratio, d, corrected, torch.arange(2), ISMaskConfig(veto_min=1e-4))
    assert (out[0] == 0).all()
    assert (out[1] == 1).all()
    assert stats["sampling/is_veto_masked_frac"] == (1, 2)


def test_dummy_rows_never_masked_by_trajectory_stages():
    diff = torch.zeros(2, 4)
    diff[1] = torch.log(torch.tensor(1e-5))
    ratio, d, corrected = _ratio_setup(diff)
    out, _ = apply_is_masks(
        ratio, d, corrected, torch.tensor([0, -1]), ISMaskConfig(veto_min=1e-4, geo_band_min=0.99, geo_band_max=1.01)
    )
    assert (out[0] == 1).all()
    # Garbage in a traj_id=-1 dummy row must pass through untouched (its loss is masked elsewhere).
    assert (out[1] == ratio[1]).all()
    all_dummy, stats = apply_is_masks(ratio, d, corrected, torch.tensor([-1, -1]), ISMaskConfig(veto_min=1e-4))
    assert torch.equal(all_dummy, ratio) and stats == {}


def test_opsm_masks_only_drifted_negative_trajectories():
    diff = torch.zeros(3, 4)
    diff[0] = 0.5
    diff[1] = 0.5
    ratio, d, corrected = _ratio_setup(diff)
    traj_ids = torch.arange(3)
    # traj 0 negative+drifted (masked), 1 positive+drifted, 2 negative+clean — only 0 qualifies.
    advantages = torch.tensor([-1.0, 1.0, -1.0])
    out, masked = apply_opsm(ratio, d, corrected, traj_ids, advantages, delta=0.2)
    assert (out[0] == 0).all()
    assert (out[1] > 0).all()
    assert (out[2] > 0).all()
    assert masked.tolist() == [True, False, False]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
