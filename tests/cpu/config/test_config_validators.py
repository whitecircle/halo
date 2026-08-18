#!/usr/bin/env python
"""Validation tests for ClassificationConfig, EmbeddingConfig,
DistillScriptArguments and RLVROnlineGRPOScriptArguments ``__post_init__`` (and the CLI-override
re-run of the same guards), and the ``sync_token_field`` precedence helper.

Each boundary the validator rejects is exercised (raise expected) alongside a valid
neighbor (no raise), mirroring the SMPO validator-test style in test_config_dataclasses.py.

Both configs derive from a TrainingArguments-like base whose tail __post_init__ rejects
bf16 on a GPU-less host; valid configs pass bf16=False + use_cpu=True to reach that tail
cleanly. The validation-branch tests raise BEFORE the tail, so they need only output_dir.

Run: pytest tests/cpu/config/test_config_validators.py
"""

import pytest

from src.args.distill_args import DistillScriptArguments
from src.args.rlvr_online_grpo_args import RLVROnlineGRPOScriptArguments
from src.configs.classification_config import ClassificationConfig
from src.configs.distillation_config import DistillationConfig
from src.configs.embedding_config import EmbeddingConfig
from src.training.script_runner import sync_token_field

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


def test_embedding_weights_without_dimensions_raises_on_cli_override():
    """The guard lives in ``_validate_ranges``, so the setattr override path re-runs it — a
    ``--matryoshka_weights`` CLI override must not slip past ``__post_init__``."""
    cfg = EmbeddingConfig(**_CPU_OK)
    cfg.matryoshka_weights = [1.0, 0.5]
    with pytest.raises(ValueError, match="matryoshka_weights"):
        cfg.__post_override__({"matryoshka_weights"})


# sync_token_field precedence


class _Holder:
    """Minimal stand-in for args / training_config carrying a token field."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_sync_token_field_args_value_used_when_config_none():
    """config=None, args=X → X copied onto BOTH sides."""
    args = _Holder(pad_token="<pad>")
    cfg = _Holder(pad_token=None)
    sync_token_field(args, cfg, "pad_token")
    assert args.pad_token == "<pad>"
    assert cfg.pad_token == "<pad>", "args value must propagate to config when config is None"


def test_sync_token_field_config_wins_when_both_set():
    """config=Y, args=X → config wins, Y on BOTH sides (config takes precedence)."""
    args = _Holder(pad_token="<from_args>")
    cfg = _Holder(pad_token="<from_config>")
    sync_token_field(args, cfg, "pad_token")
    assert cfg.pad_token == "<from_config>"
    assert args.pad_token == "<from_config>", "config value must overwrite args when both are set"


def test_sync_token_field_both_none_is_noop():
    """config=None, args=None → neither side is touched (no spurious value written)."""
    args = _Holder(pad_token=None)
    cfg = _Holder(pad_token=None)
    sync_token_field(args, cfg, "pad_token")
    assert args.pad_token is None
    assert cfg.pad_token is None


# DistillationConfig / DistillScriptArguments / RLVROnlineGRPOScriptArguments range guards


def test_distill_defaults_valid():
    cfg = DistillationConfig(**_CPU_OK)
    assert cfg.distill_temperature == 1.0
    assert cfg.distill_alpha == 1.0
    assert cfg.distill_loss == "kl_divergence"
    assert cfg.use_clm_loss is True
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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
