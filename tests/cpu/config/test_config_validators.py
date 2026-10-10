#!/usr/bin/env python
"""Validation tests for ClassificationConfig, EmbeddingConfig,
DistillScriptArguments and RLVROnlineGRPOScriptArguments ``__post_init__`` (a CLI override meets
the same guards), plus the NaN refusal on every float those configs and SMPO range-check.

Each boundary the validator rejects is exercised (raise expected) alongside a valid
neighbor (no raise), mirroring the SMPO validator-test style in test_config_dataclasses.py.

Both configs derive from a TrainingArguments-like base whose tail __post_init__ rejects
bf16 on a GPU-less host; valid configs pass bf16=False + use_cpu=True to reach that tail
cleanly. The validation-branch tests raise BEFORE the tail, so they need only output_dir.

Run: pytest tests/cpu/config/test_config_validators.py
"""

import math
from typing import get_args, get_type_hints

import pytest

from src.args.distill_args import DistillScriptArguments
from src.args.rlvr_online_grpo_args import RLVROnlineGRPOScriptArguments
from src.configs.classification_config import ClassificationConfig
from src.configs.distillation_config import TOPK_DISTILL_LOSSES, DistillationConfig
from src.configs.embedding_config import EmbeddingConfig
from src.configs.smpo_config import SmoothMarginPOConfig
from src.training.parser import H4ArgumentParser

OUTPUT_DIR = "/tmp/test_config_validators"
# Clears the bf16/GPU tail check for configs expected to construct successfully.
_CPU_OK = {"output_dir": OUTPUT_DIR, "bf16": False, "use_cpu": True}


# ClassificationConfig.__post_init__


def test_classification_defaults_valid():
    cfg = ClassificationConfig(**_CPU_OK)
    assert cfg.focal_gamma == 2.0
    assert cfg.label_smoothing == 0.0
    assert cfg.multi_label_threshold == 0.5


def test_classification_negative_focal_gamma_raises():
    with pytest.raises(ValueError, match="focal_gamma"):
        ClassificationConfig(output_dir=OUTPUT_DIR, focal_gamma=-0.1)


def test_classification_focal_gamma_zero_ok():
    """focal_gamma == 0 is the boundary (>= 0) and must be accepted."""
    cfg = ClassificationConfig(focal_gamma=0.0, **_CPU_OK)
    assert cfg.focal_gamma == 0.0


def test_classification_label_smoothing_one_raises():
    """label_smoothing must be in [0, 1); 1.0 is out of range."""
    with pytest.raises(ValueError, match="label_smoothing"):
        ClassificationConfig(output_dir=OUTPUT_DIR, label_smoothing=1.0)


def test_classification_label_smoothing_negative_raises():
    with pytest.raises(ValueError, match="label_smoothing"):
        ClassificationConfig(output_dir=OUTPUT_DIR, label_smoothing=-0.01)


def test_classification_label_smoothing_high_interior_ok():
    """0.999 is interior to [0, 1) and must be accepted (guards an off-by-one on the bound)."""
    cfg = ClassificationConfig(label_smoothing=0.999, **_CPU_OK)
    assert cfg.label_smoothing == 0.999


def test_classification_multi_label_threshold_zero_raises():
    """threshold must be in the OPEN interval (0, 1); 0.0 is excluded."""
    with pytest.raises(ValueError, match="multi_label_threshold"):
        ClassificationConfig(output_dir=OUTPUT_DIR, multi_label_threshold=0.0)


def test_classification_multi_label_threshold_one_raises():
    with pytest.raises(ValueError, match="multi_label_threshold"):
        ClassificationConfig(output_dir=OUTPUT_DIR, multi_label_threshold=1.0)


def test_classification_multi_label_threshold_interior_ok():
    cfg = ClassificationConfig(multi_label_threshold=0.7, **_CPU_OK)
    assert cfg.multi_label_threshold == 0.7


def test_classification_class_weights_and_auto_mutually_exclusive_raises():
    """class_weights and derive_class_weights set the same per-class weight from two sources —
    accepting both silently drops one, so __post_init__ must reject the pair."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        ClassificationConfig(output_dir=OUTPUT_DIR, class_weights=[2.0, 1.0], derive_class_weights=True)


def test_classification_class_weights_only_ok():
    cfg = ClassificationConfig(class_weights=[2.0, 1.0], **_CPU_OK)
    assert cfg.class_weights == [2.0, 1.0]
    assert cfg.derive_class_weights is False


def test_classification_auto_class_weights_only_ok():
    cfg = ClassificationConfig(derive_class_weights=True, **_CPU_OK)
    assert cfg.derive_class_weights is True
    assert cfg.class_weights is None


# EmbeddingConfig.__post_init__


def test_embedding_defaults_valid():
    cfg = EmbeddingConfig(**_CPU_OK)
    assert cfg.loss_scale == 20.0
    assert cfg.matryoshka_dimensions is None


def test_embedding_loss_scale_zero_raises():
    """loss_scale must be > 0; 0.0 is rejected (a zero inverse-temperature is degenerate)."""
    with pytest.raises(ValueError, match="loss_scale"):
        EmbeddingConfig(output_dir=OUTPUT_DIR, loss_scale=0.0)


def test_embedding_loss_scale_negative_raises():
    with pytest.raises(ValueError, match="loss_scale"):
        EmbeddingConfig(output_dir=OUTPUT_DIR, loss_scale=-5.0)


def test_embedding_loss_scale_small_positive_ok():
    cfg = EmbeddingConfig(loss_scale=0.001, **_CPU_OK)
    assert cfg.loss_scale == 0.001


def test_embedding_matryoshka_weights_length_mismatch_raises():
    """matryoshka_weights length must equal matryoshka_dimensions length."""
    with pytest.raises(ValueError, match="matryoshka_weights"):
        EmbeddingConfig(
            output_dir=OUTPUT_DIR,
            matryoshka_dimensions=[256, 128, 64],
            matryoshka_weights=[1.0, 0.5],  # 2 != 3
        )


def test_embedding_empty_matryoshka_dimensions_raises():
    with pytest.raises(ValueError, match="matryoshka_dimensions"):
        EmbeddingConfig(output_dir=OUTPUT_DIR, matryoshka_dimensions=[])


def test_embedding_matching_matryoshka_weights_ok():
    """Equal-length weights/dimensions must be accepted."""
    cfg = EmbeddingConfig(
        matryoshka_dimensions=[256, 128, 64],
        matryoshka_weights=[1.0, 0.5, 0.25],
        **_CPU_OK,
    )
    assert cfg.matryoshka_dimensions == [256, 128, 64]
    assert cfg.matryoshka_weights == [1.0, 0.5, 0.25]


def test_embedding_dimensions_without_weights_ok():
    """matryoshka_dimensions with no weights (uniform) must be accepted."""
    cfg = EmbeddingConfig(matryoshka_dimensions=[512, 256], **_CPU_OK)
    assert cfg.matryoshka_weights is None


def test_embedding_weights_without_dimensions_raises():
    """The mirror case: weights alone never reach a loss — MatryoshkaLoss is built from the
    dimensions, so without them the run trains the plain loss and drops the weights silently."""
    with pytest.raises(ValueError, match="matryoshka_weights"):
        EmbeddingConfig(output_dir=OUTPUT_DIR, matryoshka_weights=[1.0, 0.5])


def test_embedding_weights_without_dimensions_raises_on_cli_override(tmp_path):
    config = tmp_path / "embedding.yaml"
    config.write_text(f"output_dir: {OUTPUT_DIR}\nbf16: false\nuse_cpu: true\n")
    with pytest.raises(ValueError, match="matryoshka_weights"):
        H4ArgumentParser((EmbeddingConfig,)).parse_yaml_and_args(str(config), ["--matryoshka_weights=1.0,0.5"])


# DistillationConfig / DistillScriptArguments / RLVROnlineGRPOScriptArguments range guards


def test_distill_defaults_valid():
    cfg = DistillationConfig(**_CPU_OK)
    assert cfg.distill_temperature == 1.0
    assert cfg.distill_alpha == 1.0
    assert cfg.distill_loss == "kl_divergence"
    assert cfg.apply_hard_labels is False


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"distill_temperature": 0.0}, "distill_temperature"),
        ({"distill_temperature": -1.0}, "distill_temperature"),
        ({"distill_alpha": -0.1}, "distill_alpha"),
        ({"distill_alpha": 1.5}, "distill_alpha"),
    ],
)
def test_distill_out_of_range_raises(kwargs, match):
    with pytest.raises(ValueError, match=match):
        DistillationConfig(**_CPU_OK, **kwargs)


_DISTILL_LOSSES = get_args(get_type_hints(DistillationConfig)["distill_loss"])


@pytest.mark.parametrize("beta", [-0.1, 1.1])
def test_distill_jsd_beta_outside_the_unit_interval_raises(beta):
    """Outside [0, 1] the mixture weight log1p(-β) or log(β) is NaN, so every JSD term would be NaN."""
    with pytest.raises(ValueError, match="distill_jsd_beta must be in"):
        DistillationConfig(**_CPU_OK, distill_loss="jensen_shannon", distill_jsd_beta=beta)


@pytest.mark.parametrize("beta", [True, False, math.nan])
def test_distill_jsd_beta_that_is_not_a_number_raises(beta):
    """A YAML ``true``/``false`` compares as 1/0 and would silently train the reverse/forward KL endpoint."""
    with pytest.raises(ValueError, match="distill_jsd_beta must be a finite number"):
        DistillationConfig(**_CPU_OK, distill_loss="jensen_shannon", distill_jsd_beta=beta)


@pytest.mark.parametrize("loss", [name for name in _DISTILL_LOSSES if name != "jensen_shannon"])
def test_distill_jsd_beta_with_another_loss_raises(loss):
    """Only jensen_shannon reads β, so a set β beside any other loss would be silently ignored."""
    with pytest.raises(ValueError, match="only applies to distill_loss: jensen_shannon"):
        DistillationConfig(**_CPU_OK, distill_loss=loss, distill_jsd_beta=0.3)


@pytest.mark.parametrize("beta", [0.0, 0.3, 1.0])
def test_distill_jsd_beta_with_jensen_shannon_builds(beta):
    cfg = DistillationConfig(**_CPU_OK, distill_loss="jensen_shannon", distill_jsd_beta=beta)
    assert cfg.distill_jsd_beta == beta


def test_distill_jsd_beta_cross_check_sees_a_cli_override(tmp_path):
    """A CLI override of one side of the pair must trip the guard the YAML value would."""
    config = tmp_path / "config.yaml"
    config.write_text(
        f"output_dir: {OUTPUT_DIR}\nbf16: false\nuse_cpu: true\ndistill_loss: jensen_shannon\ndistill_jsd_beta: 0.3\n"
    )
    with pytest.raises(ValueError, match="only applies to distill_loss: jensen_shannon"):
        H4ArgumentParser((DistillationConfig,)).parse_yaml_and_args(str(config), ["--distill_loss=kl_divergence"])


@pytest.mark.parametrize("topk", [0, -1, True, 2.5])
def test_distill_topk_must_be_a_positive_int(topk):
    with pytest.raises(ValueError, match="distill_topk must be an integer >= 1"):
        DistillationConfig(**_CPU_OK, distill_topk=topk)


@pytest.mark.parametrize("loss", [name for name in _DISTILL_LOSSES if name not in TOPK_DISTILL_LOSSES])
def test_distill_topk_with_a_loss_the_teacher_topk_cannot_carry_raises(loss):
    with pytest.raises(ValueError, match="distill_topk applies only to distill_loss in"):
        DistillationConfig(**_CPU_OK, distill_loss=loss, distill_topk=8)


@pytest.mark.parametrize("loss", TOPK_DISTILL_LOSSES)
def test_distill_topk_with_a_teacher_weighted_loss_builds(loss):
    cfg = DistillationConfig(**_CPU_OK, distill_loss=loss, distill_topk=8)
    assert cfg.distill_topk == 8


def test_distill_teacher_model_still_required_on_the_script_args():
    """The knobs moved to the training config; the teacher the SCRIPT loads did not."""
    with pytest.raises(ValueError, match="teacher_model"):
        DistillScriptArguments()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"rlrr_tau": 0.0}, "rlrr_tau"),
        ({"rlrr_lambda": 0.0}, "rlrr_lambda"),
        ({"rlrr_lambda": -1.0}, "rlrr_lambda"),
    ],
)
def test_rlvr_rlrr_out_of_range_raises(kwargs, match):
    with pytest.raises(ValueError, match=match):
        RLVROnlineGRPOScriptArguments(**kwargs)


@pytest.mark.parametrize(
    ("config_cls", "field_name"),
    [
        (ClassificationConfig, "focal_gamma"),
        (DistillationConfig, "distill_temperature"),
        (EmbeddingConfig, "loss_scale"),
        (SmoothMarginPOConfig, "target_margin"),
        (SmoothMarginPOConfig, "initial_margin"),
        (SmoothMarginPOConfig, "min_log_prob"),
    ],
)
def test_a_nan_float_is_refused(config_cls, field_name):
    """NaN passes every ordered comparison, so a bare range check admits it and the loss goes NaN."""
    with pytest.raises(ValueError, match=f"{field_name} must be a finite number"):
        config_cls(output_dir=OUTPUT_DIR, **{field_name: float("nan")})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
