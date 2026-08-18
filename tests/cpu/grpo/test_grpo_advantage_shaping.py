"""CPU tests for the WS-1 advantage surgery (AdvantageShaping: qae / asymmetric / neg_mask_hard)
and the WS-4 IS mask/veto stages (ISMaskConfig: token band, trajectory geometric band, catastrophic
veto, OPSM). Both must be exact no-ops at their defaults — the default path is the running baseline.

    python tests/cpu/grpo/test_grpo_advantage_shaping.py
"""

import sys

import pytest
import torch

from src.args.mixins import AdvantageShaping
from src.trainers.grpo.objective.advantages import group_relative_advantages
from src.trainers.grpo.objective.logratio import ISMaskConfig, apply_is_masks, apply_opsm, compute_is_ratio

G = 4  # num_generations


# --- AdvantageShaping (WS-1) ---


def test_shaping_none_and_mean_are_bit_identical():
    rewards = torch.tensor([0.0, 0.1, 1.0, 0.05, 0.3, 0.3, 0.2, 0.9])
    base = group_relative_advantages(rewards, G, "batch")
    shaped = group_relative_advantages(rewards, G, "batch", shaping=AdvantageShaping(mode="mean"))
    assert torch.equal(base, shaped)


def test_shaping_validation():
    with pytest.raises(ValueError, match="mode"):
        AdvantageShaping(mode="quantile")
    with pytest.raises(ValueError, match="quantile"):
        AdvantageShaping(mode="qae", quantile=1.5)
    with pytest.raises(ValueError, match="scale"):
        AdvantageShaping(mode="asymmetric", neg_scale=-0.1)


def test_qae_hard_group_zeroes_failures_trains_success():
    # The 0.4-quantile of {0,0,0,1} sits at the failure reward, so only the rare success trains.
    rewards = torch.tensor([0.0, 0.0, 0.0, 1.0])
    adv = group_relative_advantages(rewards, G, "none", shaping=AdvantageShaping(mode="qae", quantile=0.4))
    assert torch.allclose(adv[:3], torch.zeros(3), atol=1e-6)
    assert adv[3] == pytest.approx(1.0)
    # The mean baseline would instead give every failure a negative advantage.
    mean_adv = group_relative_advantages(rewards, G, "none")
    assert (mean_adv[:3] < 0).all()


def test_qae_easy_group_trains_residual_failures():
    rewards = torch.tensor([1.0, 1.0, 1.0, 0.0])
    adv = group_relative_advantages(rewards, G, "none", shaping=AdvantageShaping(mode="qae", quantile=0.4))
    assert torch.allclose(adv[:3], torch.zeros(3), atol=1e-6)
    assert adv[3] == pytest.approx(-1.0)


def test_qae_respects_valid_mask():
    # Masking the invalid placeholder leaves {0,1,1}, whose interpolated 0.4-quantile is 0.8 —
    # above where the unmasked quantile would sit, so a dropped valid_mask changes every number.
    rewards = torch.tensor([0.0, 0.0, 1.0, 1.0])
    valid = torch.tensor([False, True, True, True])
    adv = group_relative_advantages(
        rewards, G, "none", valid_mask=valid, shaping=AdvantageShaping(mode="qae", quantile=0.4)
    )
    assert adv[1] == pytest.approx(-0.8)
    assert adv[2] == pytest.approx(0.2) and adv[3] == pytest.approx(0.2)


def test_asymmetric_scales_signs_independently():
    rewards = torch.tensor([0.0, 0.0, 1.0, 1.0])
    base = group_relative_advantages(rewards, G, "none")
    shaped = group_relative_advantages(
        rewards, G, "none", shaping=AdvantageShaping(mode="asymmetric", pos_scale=0.9, neg_scale=0.4)
    )
    assert torch.allclose(shaped[base > 0], base[base > 0] * 0.9)
    assert torch.allclose(shaped[base < 0], base[base < 0] * 0.4)


def test_neg_mask_hard_zeroes_negatives_only_in_hard_groups():
    # Group 1 is hard (best 0.3 < threshold) so its negatives are zeroed; group 2 has a solve.
    rewards = torch.tensor([0.0, 0.1, 0.3, 0.2, 0.0, 0.1, 1.0, 0.2])
    gate = rewards.clone()
    shaped = group_relative_advantages(
        rewards, G, "none", gate_rewards=gate, shaping=AdvantageShaping(mode="neg_mask_hard", hard_group_threshold=0.5)
    )
    base = group_relative_advantages(rewards, G, "none")
    hard, easy = slice(0, 4), slice(4, 8)
    assert (shaped[hard][base[hard] < 0] == 0).all()
    assert torch.allclose(shaped[hard][base[hard] > 0], base[hard][base[hard] > 0])
    assert torch.equal(shaped[easy], base[easy])


def test_neg_mask_hard_gate_uses_objective_not_shaped_total():
    # Totals clear the threshold on shaping rungs alone; gating on them would wrongly call the group solved.
    totals = torch.tensor([0.55, 0.6, 0.65, 0.7])
    objective = torch.zeros(G)
    shaped = group_relative_advantages(
        totals,
        G,
        "none",
        gate_rewards=objective,
        shaping=AdvantageShaping(mode="neg_mask_hard", hard_group_threshold=0.5),
    )
    base = group_relative_advantages(totals, G, "none")
    assert (shaped[base < 0] == 0).all()


# --- ISMaskConfig stages (WS-4) ---


def _ratio_setup(logdiffs: torch.Tensor, clip_max: float = 3.0):
    """Build (ratio, logps_diff, corrected_mask) from a [rows, T] log-diff tensor, all tokens corrected."""
    rows, t = logdiffs.shape
    completion_mask = torch.ones(rows, t, dtype=torch.long)
    ratio, diff, corrected = compute_is_ratio(
        logdiffs, torch.zeros_like(logdiffs), completion_mask, torch.ones(rows, dtype=torch.bool), clip_max
    )
    return ratio, diff, corrected


def test_ismask_defaults_are_inert():
    cfg = ISMaskConfig()
    assert not cfg.any_mask_active
    ratio, diff, corrected = _ratio_setup(torch.randn(4, 5) * 0.01)
    out, stats = apply_is_masks(ratio, diff, corrected, torch.arange(4), cfg)
    assert out is ratio and stats == {}


def test_ismask_validation():
    with pytest.raises(ValueError, match="together"):
        ISMaskConfig(band_min=0.5)
    with pytest.raises(ValueError, match="bounds"):
        ISMaskConfig(band_min=1.2, band_max=2.0)
    with pytest.raises(ValueError, match="veto"):
        ISMaskConfig(veto_min=2.0)
    with pytest.raises(ValueError, match="opsm"):
        ISMaskConfig(opsm_delta=-1.0)


def test_token_band_masks_out_of_band_tokens():
    diff = torch.zeros(2, 4)
    diff[0, 1] = torch.log(torch.tensor(4.0))
    diff[1, 2] = torch.log(torch.tensor(0.2))
    ratio, d, corrected = _ratio_setup(diff)
    out, stats = apply_is_masks(ratio, d, corrected, torch.arange(2), ISMaskConfig(band_min=0.5, band_max=2.0))
    assert out[0, 1] == 0 and out[1, 2] == 0
    assert out[0, 0] == 1 and out[1, 0] == 1
    assert stats["sampling/is_token_band_masked_frac"] == pytest.approx(2 / 8)


def test_token_band_uses_raw_ratio_not_truncated():
    # Truncation pulls ratio 4 down to 3; the band test must still see the raw 4 and mask it.
    diff = torch.full((1, 2), 0.0)
    diff[0, 0] = torch.log(torch.tensor(4.0))
    ratio, d, corrected = _ratio_setup(diff, clip_max=3.0)
    assert ratio[0, 0] == pytest.approx(3.0)
    out, _ = apply_is_masks(ratio, d, corrected, torch.arange(1), ISMaskConfig(band_min=0.5, band_max=3.5))
    assert out[0, 0] == 0


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
    assert stats["sampling/is_geo_band_masked_frac"] == pytest.approx(0.5)


def test_veto_masks_trajectory_with_catastrophic_token():
    diff = torch.zeros(2, 4)
    diff[0, 3] = torch.log(torch.tensor(1e-5))
    ratio, d, corrected = _ratio_setup(diff)
    out, stats = apply_is_masks(ratio, d, corrected, torch.arange(2), ISMaskConfig(veto_min=1e-4))
    assert (out[0] == 0).all()
    assert (out[1] == 1).all()
    assert stats["sampling/is_veto_masked_frac"] == pytest.approx(0.5)


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


def test_opsm_masks_only_drifted_negative_trajectories():
    diff = torch.zeros(3, 4)
    diff[0] = 0.5
    diff[1] = 0.5
    ratio, d, corrected = _ratio_setup(diff)
    traj_ids = torch.arange(3)
    # traj 0 negative+drifted (masked), 1 positive+drifted, 2 negative+clean — only 0 qualifies.
    advantages = torch.tensor([-1.0, 1.0, -1.0])
    out, frac = apply_opsm(ratio, d, corrected, traj_ids, advantages, delta=0.2)
    assert (out[0] == 0).all()
    assert (out[1] > 0).all()
    assert (out[2] > 0).all()
    assert frac == pytest.approx(1 / 3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
