#!/usr/bin/env python
"""The SMPO, DPO and KTO construction refusals: each fires on its own condition, named by its mechanism, and the
supported run passes them all.

Run: python tests/cpu/trainers/test_preference_rules.py
"""

from types import SimpleNamespace

import pytest

from halo.trainers.preference.dpo import _PP_PAIR_LOSSES
from halo.trainers.preference.preference_rules import (
    reject_dpo_pp_objective,
    reject_kto_pp_loss_type,
    reject_policy_from_checkpoint_without_resume,
    reject_smpo_options,
    reject_smpo_pp_options,
)

_PP = SimpleNamespace(is_pp_mode=True, cp_size=1)
_CP = SimpleNamespace(is_pp_mode=False, cp_size=2)


def _smpo_args(**overrides) -> SimpleNamespace:
    return SimpleNamespace(
        **{"padding_free": False, "lower_clip_percentile": None, "upper_clip_percentile": None, **overrides}
    )


def _dpo_args(**overrides) -> SimpleNamespace:
    return SimpleNamespace(
        **{
            "loss_type": "sigmoid",
            "f_divergence_type": "reverse_kl",
            "use_weighting": False,
            "ld_alpha": None,
            **overrides,
        }
    )


def test_the_supported_run_passes_every_refusal():
    reject_smpo_options(_smpo_args(padding_free=True), SimpleNamespace(cp_size=1))
    reject_smpo_options(_smpo_args(), _CP)
    reject_smpo_pp_options(_smpo_args(), _PP, None, is_vlm=False)
    reject_dpo_pp_objective(_dpo_args(loss_type=["sigmoid", "ipo"]), supported=_PP_PAIR_LOSSES)
    reject_kto_pp_loss_type(SimpleNamespace(loss_type="apo_zero_unpaired"))
    reject_policy_from_checkpoint_without_resume({"policy_from_checkpoint": True, "resume_checkpoint": "ckpt"})
    reject_policy_from_checkpoint_without_resume({})


def test_the_pp_options_are_inert_off_pipeline_parallelism():
    """Every SMPO PP refusal sits behind ``is_pp_mode``: a non-PP run keeps padding_free and the clip bounds."""
    off = SimpleNamespace(is_pp_mode=False, cp_size=1)
    reject_smpo_pp_options(_smpo_args(padding_free=True, lower_clip_percentile=0.1), off, object(), is_vlm=True)


@pytest.mark.parametrize(
    ("refusal", "match"),
    [
        (lambda: reject_smpo_options(_smpo_args(padding_free=True), _CP), r"padding_free mode .* Context Parallelism"),
        (
            lambda: reject_smpo_pp_options(_smpo_args(padding_free=True), _PP, None, is_vlm=False),
            r"padding_free is not supported under pipeline parallelism: it flattens each batch",
        ),
        (
            lambda: reject_smpo_pp_options(_smpo_args(lower_clip_percentile=0.1), _PP, None, is_vlm=False),
            r"clip bound is a token quantile over the WHOLE batch half",
        ),
        (
            lambda: reject_smpo_pp_options(_smpo_args(upper_clip_percentile=0.9), _PP, None, is_vlm=False),
            r"clip bound is a token quantile over the WHOLE batch half",
        ),
        (
            lambda: reject_dpo_pp_objective(_dpo_args(loss_type="aot"), supported=_PP_PAIR_LOSSES),
            r"loss_type 'aot' .* sorts the paired log-ratios across the whole batch",
        ),
        (
            lambda: reject_dpo_pp_objective(_dpo_args(loss_type=["sigmoid", "sft"]), supported=_PP_PAIR_LOSSES),
            r"loss_type 'sft' .* batch-global token denominator",
        ),
        (
            lambda: reject_dpo_pp_objective(_dpo_args(loss_type="robust"), supported=_PP_PAIR_LOSSES),
            r"loss_type 'robust' .* not re-implemented in the PP last-stage loss",
        ),
        (
            lambda: reject_dpo_pp_objective(_dpo_args(f_divergence_type="js_divergence"), supported=_PP_PAIR_LOSSES),
            r"f_divergence_type='js_divergence' .* reverse-KL scores only",
        ),
        (
            lambda: reject_dpo_pp_objective(_dpo_args(use_weighting=True), supported=_PP_PAIR_LOSSES),
            r"use_weighting \(WPO\) .* full-vocab logsumexp",
        ),
        (
            lambda: reject_dpo_pp_objective(_dpo_args(ld_alpha=0.5), supported=_PP_PAIR_LOSSES),
            r"ld_alpha \(length-desensitized DPO\) .* reshapes the sequence log-probs",
        ),
        (
            lambda: reject_kto_pp_loss_type(SimpleNamespace(loss_type="kto")),
            r"loss_type 'kto' .* world-global detached KL baseline",
        ),
        (
            lambda: reject_policy_from_checkpoint_without_resume({"policy_from_checkpoint": True}),
            r"policy_from_checkpoint=True needs the resume_checkpoint",
        ),
    ],
    ids=[
        "smpo-padding-free-cp",
        "smpo-padding-free-pp",
        "smpo-lower-clip-pp",
        "smpo-upper-clip-pp",
        "dpo-aot-pp",
        "dpo-sft-pp",
        "dpo-unaudited-pp",
        "dpo-f-divergence-pp",
        "dpo-wpo-pp",
        "dpo-ld-alpha-pp",
        "kto-kl-loss-pp",
        "policy-from-checkpoint",
    ],
)
def test_each_refusal_names_its_mechanism(refusal, match):
    with pytest.raises(ValueError, match=match):
        refusal()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
