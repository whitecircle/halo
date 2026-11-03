#!/usr/bin/env python
"""``__post_init__`` guards of ClassificationConfig, EmbeddingConfig, TeacherDistillScriptArguments and
OnlineGRPOScriptArguments, plus the NaN refusal on every float they range-check. Valid configs pass bf16=False +
use_cpu=True to clear TrainingArguments' tail check."""

import pytest

from halo.configs.classification import ClassificationConfig
from halo.configs.distillation import TeacherDistillationConfig, TeacherDistillScriptArguments
from halo.configs.embedding import EmbeddingConfig
from halo.configs.grpo_online import OnlineGRPOScriptArguments
from halo.configs.parser import HaloArgumentParser
from halo.configs.preference import SMPOConfig

OUTPUT_DIR = "/tmp/test_config_validators"
_CPU_OK = {"output_dir": OUTPUT_DIR, "bf16": False, "use_cpu": True}


def test_classification_defaults_valid():
    cfg = ClassificationConfig(**_CPU_OK)
    assert cfg.focal_gamma == 2.0
    assert cfg.label_smoothing == 0.0
    assert cfg.multi_label_threshold == 0.5


def test_classification_negative_focal_gamma_raises():
    with pytest.raises(ValueError, match="focal_gamma"):
        ClassificationConfig(output_dir=OUTPUT_DIR, focal_gamma=-0.1)


def test_classification_focal_gamma_zero_ok():
    cfg = ClassificationConfig(focal_gamma=0.0, **_CPU_OK)
    assert cfg.focal_gamma == 0.0


def test_classification_label_smoothing_one_raises():
    with pytest.raises(ValueError, match="label_smoothing"):
        ClassificationConfig(output_dir=OUTPUT_DIR, label_smoothing=1.0)


def test_classification_label_smoothing_negative_raises():
    with pytest.raises(ValueError, match="label_smoothing"):
        ClassificationConfig(output_dir=OUTPUT_DIR, label_smoothing=-0.01)


def test_classification_label_smoothing_high_interior_ok():
    cfg = ClassificationConfig(label_smoothing=0.999, **_CPU_OK)
    assert cfg.label_smoothing == 0.999


def test_classification_multi_label_threshold_zero_raises():
    with pytest.raises(ValueError, match="multi_label_threshold"):
        ClassificationConfig(output_dir=OUTPUT_DIR, multi_label_threshold=0.0)


def test_classification_multi_label_threshold_one_raises():
    with pytest.raises(ValueError, match="multi_label_threshold"):
        ClassificationConfig(output_dir=OUTPUT_DIR, multi_label_threshold=1.0)


def test_classification_multi_label_threshold_interior_ok():
    cfg = ClassificationConfig(multi_label_threshold=0.7, **_CPU_OK)
    assert cfg.multi_label_threshold == 0.7


def test_classification_class_weights_and_auto_mutually_exclusive_raises():
    """Both set the same per-class weight from two sources; accepting both silently drops one."""
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


def test_embedding_defaults_valid():
    cfg = EmbeddingConfig(**_CPU_OK)
    assert cfg.loss_scale == 20.0
    assert cfg.matryoshka_dimensions is None


def test_embedding_loss_scale_zero_raises():
    with pytest.raises(ValueError, match="loss_scale"):
        EmbeddingConfig(output_dir=OUTPUT_DIR, loss_scale=0.0)


def test_embedding_loss_scale_negative_raises():
    with pytest.raises(ValueError, match="loss_scale"):
        EmbeddingConfig(output_dir=OUTPUT_DIR, loss_scale=-5.0)


def test_embedding_loss_scale_small_positive_ok():
    cfg = EmbeddingConfig(loss_scale=0.001, **_CPU_OK)
    assert cfg.loss_scale == 0.001


def test_embedding_matryoshka_weights_length_mismatch_raises():
    with pytest.raises(ValueError, match="matryoshka_weights"):
        EmbeddingConfig(
            output_dir=OUTPUT_DIR,
            matryoshka_dimensions=[256, 128, 64],
            matryoshka_weights=[1.0, 0.5],
        )


def test_embedding_empty_matryoshka_dimensions_raises():
    with pytest.raises(ValueError, match="matryoshka_dimensions"):
        EmbeddingConfig(output_dir=OUTPUT_DIR, matryoshka_dimensions=[])


def test_embedding_matching_matryoshka_weights_ok():
    cfg = EmbeddingConfig(
        matryoshka_dimensions=[256, 128, 64],
        matryoshka_weights=[1.0, 0.5, 0.25],
        **_CPU_OK,
    )
    assert cfg.matryoshka_dimensions == [256, 128, 64]
    assert cfg.matryoshka_weights == [1.0, 0.5, 0.25]


def test_embedding_dimensions_without_weights_ok():
    cfg = EmbeddingConfig(matryoshka_dimensions=[512, 256], **_CPU_OK)
    assert cfg.matryoshka_weights is None


def test_embedding_weights_without_dimensions_raises():
    """MatryoshkaLoss is built from the dimensions, so without them the run trains the plain loss and silently drops
    the weights."""
    with pytest.raises(ValueError, match="matryoshka_weights"):
        EmbeddingConfig(output_dir=OUTPUT_DIR, matryoshka_weights=[1.0, 0.5])


def test_embedding_weights_without_dimensions_raises_on_cli_override(tmp_path):
    config = tmp_path / "embedding.yaml"
    config.write_text(f"output_dir: {OUTPUT_DIR}\nbf16: false\nuse_cpu: true\n")
    parser = HaloArgumentParser((EmbeddingConfig,))
    with pytest.raises(ValueError, match="matryoshka_weights"):
        parser.parse_dict(parser.load_values([str(config), "--matryoshka_weights=1.0,0.5"]))


def test_distill_defaults_valid():
    cfg = TeacherDistillationConfig(**_CPU_OK)
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
        TeacherDistillationConfig(**_CPU_OK, **kwargs)


def test_distill_teacher_model_still_required_on_the_script_args():
    with pytest.raises(ValueError, match="teacher_model"):
        TeacherDistillScriptArguments()


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
        OnlineGRPOScriptArguments(**kwargs)


@pytest.mark.parametrize(
    ("config_cls", "field_name"),
    [
        (ClassificationConfig, "focal_gamma"),
        (TeacherDistillationConfig, "distill_temperature"),
        (EmbeddingConfig, "loss_scale"),
        (SMPOConfig, "target_margin"),
        (SMPOConfig, "initial_margin"),
        (SMPOConfig, "min_log_prob"),
    ],
)
def test_a_nan_float_is_refused(config_cls, field_name):
    """NaN passes every ordered comparison, so a bare range check admits it and the loss goes NaN."""
    with pytest.raises(ValueError, match=f"{field_name} must be a finite number"):
        config_cls(output_dir=OUTPUT_DIR, **{field_name: float("nan")})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
