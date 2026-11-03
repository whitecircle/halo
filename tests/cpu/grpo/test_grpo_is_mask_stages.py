"""The IS mask and veto stages (ISRMaskConfig: trajectory geometric band, catastrophic veto, OPSM). They must be exact
no-ops at their defaults."""

import pytest
import torch

from halo.configs.grpo_environmental import ISRMaskConfig
from halo.trainers.grpo.objective.logratio import apply_is_masks, apply_opsm, compute_is_ratio


def _counts(stats: dict, key: str) -> tuple[float, float]:
    masked, total = stats[key]
    assert isinstance(masked, torch.Tensor), "the masked count must stay on device for the caller's one read"
    return float(masked), float(total)


def _ratio_setup(logdiffs: torch.Tensor, clip_max: float = 3.0):
    """Sampling logprobs sit at -1 because exactly 0 is the engine's probability-1 marker, which ``compute_is_ratio``
    leaves uncorrected."""
    rows, t = logdiffs.shape
    completion_mask = torch.ones(rows, t, dtype=torch.long)
    sampling = torch.full_like(logdiffs, -1.0)
    ratio, diff, corrected = compute_is_ratio(
        logdiffs + sampling, sampling, completion_mask, torch.ones(rows, dtype=torch.bool), clip_max
    )
    return ratio, diff, corrected


def test_forced_token_does_not_trip_the_veto():
    """The engine made no sampling choice at a budget-forced ``</think>``, so it is no evidence of a broken sync."""
    recompute = torch.tensor([[-0.5, -12.0, -0.7]])
    sampling = torch.tensor([[-0.5, 0.0, -0.7]])
    ratio, diff, corrected = compute_is_ratio(
        recompute, sampling, torch.ones(1, 3, dtype=torch.long), torch.ones(1, dtype=torch.bool), 3.0
    )
    out, stats = apply_is_masks(ratio, diff, corrected, torch.arange(1), ISRMaskConfig(veto_min=1e-4))
    assert _counts(stats, "sampling/is_veto_masked_frac") == (0, 1)
    assert torch.equal(out, torch.ones(1, 3))
    # The same disagreement on a token the engine sampled is what the veto is for.
    ratio, diff, corrected = compute_is_ratio(
        recompute,
        torch.tensor([[-0.5, -1e-3, -0.7]]),
        torch.ones(1, 3, dtype=torch.long),
        torch.ones(1, dtype=torch.bool),
        3.0,
    )
    out, stats = apply_is_masks(ratio, diff, corrected, torch.arange(1), ISRMaskConfig(veto_min=1e-4))
    assert _counts(stats, "sampling/is_veto_masked_frac") == (1, 1)


def test_ismask_defaults_are_inert():
    cfg = ISRMaskConfig()
    assert not cfg.any_mask_active
    ratio, diff, corrected = _ratio_setup(torch.randn(4, 5) * 0.01)
    out, stats = apply_is_masks(ratio, diff, corrected, torch.arange(4), cfg)
    assert out is ratio and stats == {}


def test_any_stage_active_counts_opsm_beside_the_mask_stages():
    """OPSM runs apart from ``apply_is_masks`` (it needs the advantages) but is a stage all the same."""
    assert not ISRMaskConfig().any_stage_active
    opsm_only = ISRMaskConfig(opsm_delta=0.2)
    assert opsm_only.any_stage_active and not opsm_only.any_mask_active
    assert ISRMaskConfig(veto_min=1e-4).any_stage_active
    assert ISRMaskConfig(geo_band_min=0.99, geo_band_max=1.01).any_stage_active


def test_ismask_validation():
    with pytest.raises(ValueError, match="together"):
        ISRMaskConfig(geo_band_min=0.99)
    with pytest.raises(ValueError, match=r"isr_geo_band_min must be in \(0, 1\)"):
        ISRMaskConfig(geo_band_min=1.2, geo_band_max=2.0)
    with pytest.raises(ValueError, match="veto"):
        ISRMaskConfig(veto_min=2.0)
    with pytest.raises(ValueError, match="opsm"):
        ISRMaskConfig(opsm_delta=-1.0)


def test_geo_band_masks_whole_trajectory_across_turn_rows():
    # Rows are pooled per trajectory: row 1 is clean but belongs to drifted trajectory 0, so it masks too.
    diff = torch.zeros(3, 4)
    diff[0] = 0.1
    diff[1] = 0.0
    ratio, d, corrected = _ratio_setup(diff)
    traj_ids = torch.tensor([0, 0, 1])
    out, stats = apply_is_masks(ratio, d, corrected, traj_ids, ISRMaskConfig(geo_band_min=0.99, geo_band_max=1.01))
    assert (out[0] == 0).all() and (out[1] == 0).all()
    assert (out[2] == 1).all()
    assert _counts(stats, "sampling/is_geo_band_masked_frac") == (1, 2)


def test_veto_masks_trajectory_with_catastrophic_token():
    diff = torch.zeros(2, 4)
    diff[0, 3] = torch.log(torch.tensor(1e-5))
    ratio, d, corrected = _ratio_setup(diff)
    out, stats = apply_is_masks(ratio, d, corrected, torch.arange(2), ISRMaskConfig(veto_min=1e-4))
    assert (out[0] == 0).all()
    assert (out[1] == 1).all()
    assert _counts(stats, "sampling/is_veto_masked_frac") == (1, 2)


def test_dummy_rows_never_masked_by_trajectory_stages():
    diff = torch.zeros(2, 4)
    diff[1] = torch.log(torch.tensor(1e-5))
    ratio, d, corrected = _ratio_setup(diff)
    out, _ = apply_is_masks(
        ratio, d, corrected, torch.tensor([0, -1]), ISRMaskConfig(veto_min=1e-4, geo_band_min=0.99, geo_band_max=1.01)
    )
    assert (out[0] == 1).all()
    # Garbage in a traj_id=-1 dummy row must pass through untouched (its loss is masked elsewhere).
    assert (out[1] == ratio[1]).all()
    all_dummy, stats = apply_is_masks(ratio, d, corrected, torch.tensor([-1, -1]), ISRMaskConfig(veto_min=1e-4))
    assert torch.equal(all_dummy, ratio) and stats == {}


def test_opsm_masks_only_drifted_negative_trajectories():
    diff = torch.zeros(3, 4)
    diff[0] = 0.5
    diff[1] = 0.5
    ratio, d, corrected = _ratio_setup(diff)
    traj_ids = torch.arange(3)
    # traj 0 negative+drifted, 1 positive+drifted, 2 negative+clean; only 0 qualifies.
    advantages = torch.tensor([-1.0, 1.0, -1.0])
    out, masked = apply_opsm(ratio, d, corrected, traj_ids, advantages, delta=0.2)
    assert (out[0] == 0).all()
    assert (out[1] > 0).all()
    assert (out[2] > 0).all()
    assert masked.tolist() == [True, False, False]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
