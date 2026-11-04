#!/usr/bin/env python
"""``__post_init__`` guards of CommonScriptArguments, ClassificationConfig, EmbeddingConfig,
TeacherDistillScriptArguments and OnlineGRPOScriptArguments, plus the NaN refusal on every float they range-check. Valid
configs pass bf16=False + use_cpu=True to clear TrainingArguments' tail check."""

import math

import pytest

from halo.configs.classification import ClassificationConfig
from halo.configs.distillation import TeacherDistillationConfig, TeacherDistillScriptArguments
from halo.configs.embedding import EmbeddingConfig
from halo.configs.grpo_offline import OfflineGRPOScriptArguments
from halo.configs.grpo_online import OnlineGRPOScriptArguments
from halo.configs.parallelism_arguments import ParallelismArguments
from halo.configs.parser import HaloArgumentParser
from halo.configs.preference import DPOScriptArguments, SMPOConfig, SMPOScriptArguments
from halo.configs.script_arguments import CommonScriptArguments
from halo.configs.sft import SFTScriptArguments

OUTPUT_DIR = "/tmp/test_config_validators"
_CPU_OK = {"output_dir": OUTPUT_DIR, "bf16": False, "use_cpu": True}
_REFUSED = "does not support these config fields"


@pytest.mark.parametrize(
    ("knob", "value"),
    [
        ("profiler_output_dir", "/elsewhere"),
        ("profiler_wait", 1),
        ("profiler_warmup", 0),
        ("profiler_active", 1),
        ("profiler_ranks", "all"),
        ("profiler_record_memory_snapshot", True),
    ],
)
def test_a_profiler_knob_needs_the_profiler(knob, value):
    with pytest.raises(ValueError, match=rf"enable_torch_profiler off {_REFUSED}.*{knob}"):
        CommonScriptArguments(**{knob: value})
    assert getattr(CommonScriptArguments(enable_torch_profiler=True, **{knob: value}), knob) == value


@pytest.mark.parametrize(("knob", "value"), [("num_full_model_params", 1e9), ("report_mfu_diagnostics", True)])
def test_an_efficiency_knob_needs_the_efficiency_callback(knob, value):
    with pytest.raises(ValueError, match=rf"enable_efficiency_metrics off {_REFUSED}.*{knob}"):
        CommonScriptArguments(**{knob: value})
    assert getattr(CommonScriptArguments(enable_efficiency_metrics=True, **{knob: value}), knob) == value


@pytest.mark.parametrize("mode", ["none", "aux_loss"])
def test_a_balancing_rate_is_refused_under_a_mode_that_takes_no_sign_step(mode):
    with pytest.raises(ValueError, match=rf"router_balancing_rate=0.01 has no effect under moe_balancing: {mode}"):
        CommonScriptArguments(moe_balancing=mode, router_balancing_rate=0.01)
    assert CommonScriptArguments(moe_balancing=mode).router_balancing_rate == 1e-3


@pytest.mark.parametrize("mode", ["auto", "bias_update", "bias_update_transient"])
def test_a_balancing_rate_stands_where_a_bias_mode_can_read_it(mode):
    assert CommonScriptArguments(moe_balancing=mode, router_balancing_rate=0.01).router_balancing_rate == 0.01


@pytest.mark.parametrize("size", ["junk", "5", "5GiB", "0GB", "-5GB", "", 5, None])
def test_an_unusable_shard_size_is_refused_before_the_first_save(size):
    with pytest.raises(ValueError, match="save_max_shard_size must be a positive size"):
        CommonScriptArguments(save_max_shard_size=size)


@pytest.mark.parametrize("size", ["5GB", "500MB", "1.5gb", " 2TB "])
def test_a_shard_size_the_writer_parses_passes(size):
    assert CommonScriptArguments(save_max_shard_size=size).save_max_shard_size == size


def test_classification_defaults_valid():
    cfg = ClassificationConfig(**_CPU_OK)
    assert cfg.focal_gamma == 2.0
    assert cfg.label_smoothing == 0.0
    assert cfg.multi_label_threshold == 0.5


def test_classification_negative_focal_gamma_raises():
    with pytest.raises(ValueError, match="focal_gamma"):
        ClassificationConfig(output_dir=OUTPUT_DIR, focal_gamma=-0.1)


def test_classification_focal_gamma_zero_ok():
    cfg = ClassificationConfig(loss_type="focal", focal_gamma=0.0, **_CPU_OK)
    assert cfg.focal_gamma == 0.0


def test_classification_label_smoothing_one_raises():
    with pytest.raises(ValueError, match="label_smoothing"):
        ClassificationConfig(output_dir=OUTPUT_DIR, label_smoothing=1.0)


def test_classification_label_smoothing_negative_raises():
    with pytest.raises(ValueError, match="label_smoothing"):
        ClassificationConfig(output_dir=OUTPUT_DIR, label_smoothing=-0.01)


def test_classification_label_smoothing_high_interior_ok():
    cfg = ClassificationConfig(loss_type="label_smoothing_ce", label_smoothing=0.999, **_CPU_OK)
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


@pytest.mark.parametrize(
    ("loss_type", "knob", "value"),
    [
        ("cross_entropy", "focal_gamma", 3.0),
        ("cross_entropy", "focal_alpha", 0.25),
        ("label_smoothing_ce", "focal_gamma", 3.0),
        ("cross_entropy", "label_smoothing", 0.1),
        ("focal", "label_smoothing", 0.1),
    ],
)
def test_classification_refuses_a_knob_of_another_loss(loss_type, knob, value):
    smoothing = {"label_smoothing": 0.1} if loss_type == "label_smoothing_ce" else {}
    with pytest.raises(ValueError, match=rf"loss_type={loss_type} {_REFUSED}.*{knob}"):
        ClassificationConfig(loss_type=loss_type, **smoothing, **{knob: value}, **_CPU_OK)


@pytest.mark.parametrize(
    ("loss_type", "knob", "value"),
    [("focal", "focal_gamma", 3.0), ("focal", "focal_alpha", 0.25), ("label_smoothing_ce", "label_smoothing", 0.1)],
)
def test_classification_takes_a_knob_under_its_own_loss(loss_type, knob, value):
    assert getattr(ClassificationConfig(loss_type=loss_type, **{knob: value}, **_CPU_OK), knob) == value


def test_label_smoothing_ce_needs_a_smoothing():
    with pytest.raises(ValueError, match="at label_smoothing=0 is plain cross-entropy"):
        ClassificationConfig(loss_type="label_smoothing_ce", **_CPU_OK)


@pytest.mark.parametrize("weight", [math.nan, math.inf, True, -1.0, "1"])
def test_classification_refuses_an_unusable_class_weight_entry(weight):
    with pytest.raises(ValueError, match="class_weights must be"):
        ClassificationConfig(class_weights=[1.0, weight], **_CPU_OK)


def test_classification_takes_a_zero_class_weight():
    assert ClassificationConfig(class_weights=[0.0, 1], **_CPU_OK).class_weights == [0.0, 1]


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


@pytest.mark.parametrize("dimension", [32.5, True, 0, -8, "64"])
def test_embedding_refuses_an_unusable_matryoshka_dimension(dimension):
    with pytest.raises(ValueError, match="matryoshka_dimensions must be an integer"):
        EmbeddingConfig(matryoshka_dimensions=[64, dimension], **_CPU_OK)


@pytest.mark.parametrize("weight", [math.nan, math.inf, True, -1.0, "1"])
def test_embedding_refuses_an_unusable_matryoshka_weight(weight):
    with pytest.raises(ValueError, match="matryoshka_weights must be"):
        EmbeddingConfig(matryoshka_dimensions=[64, 32], matryoshka_weights=[1.0, weight], **_CPU_OK)


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


@pytest.mark.parametrize("layers", [1.5, True, "3"])
def test_a_pipeline_split_entry_must_be_an_integer(layers):
    with pytest.raises(ValueError, match="pipeline_split must be an integer"):
        ParallelismArguments(pipeline_split=[3, layers])


@pytest.mark.parametrize("loss", ["mse", "cosine_similarity"])
def test_distillation_refuses_a_temperature_its_loss_does_not_read(loss):
    with pytest.raises(ValueError, match=rf"distill_loss={loss} {_REFUSED}.*distill_temperature"):
        TeacherDistillationConfig(distill_loss=loss, distill_temperature=2.0, **_CPU_OK)
    assert TeacherDistillationConfig(distill_loss=loss, **_CPU_OK).distill_temperature == 1.0
    assert TeacherDistillationConfig(distill_temperature=2.0, **_CPU_OK).distill_temperature == 2.0


@pytest.mark.parametrize(
    ("config_cls", "knob", "value"),
    [
        (SFTScriptArguments, "num_eval_examples", 7),
        (SMPOScriptArguments, "num_eval_examples", 7),
        (OfflineGRPOScriptArguments, "num_eval_examples", 7),
        (DPOScriptArguments, "num_eval_examples", 7),
        (DPOScriptArguments, "generation_max_prompt_length", 64),
    ],
    ids=lambda v: getattr(v, "__name__", str(v)),
)
def test_a_generation_knob_needs_generation(config_cls, knob, value):
    with pytest.raises(ValueError, match=rf"generate_eval_examples off {_REFUSED}.*{knob}"):
        config_cls(generate_eval_examples=False, **{knob: value})
    assert getattr(config_cls(generate_eval_examples=True, **{knob: value}), knob) == value


_TEACHER = {"teacher_model": "org/teacher"}


@pytest.mark.parametrize(
    ("config_cls", "required"), [(SFTScriptArguments, {}), (TeacherDistillScriptArguments, _TEACHER)]
)
def test_a_response_marker_needs_completion_masking(config_cls, required):
    with pytest.raises(ValueError, match="assistant_message_template marks the completions"):
        config_cls(train_on_completions_only=False, assistant_message_template="<a>", **required)
    assert config_cls(assistant_message_template="<a>", **required).assistant_message_template == "<a>"
    assert config_cls(train_on_completions_only=False, **required).assistant_message_template is None


@pytest.mark.parametrize("value", [None, "", "  "])
@pytest.mark.parametrize(
    ("config_cls", "required"), [(SFTScriptArguments, {}), (TeacherDistillScriptArguments, _TEACHER)]
)
def test_a_rendering_script_needs_a_conversation_column(config_cls, required, value):
    with pytest.raises(ValueError, match="conversation_field must be a non-blank string"):
        config_cls(conversation_field=value, **required)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
