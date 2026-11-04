#!/usr/bin/env python
"""The MTP knobs: SFT-only, off by default with nothing built, and each combination the head cannot honor refused before
the model load, naming what fixes it."""

import logging
import math
from types import SimpleNamespace

import pytest
from trl import SFTConfig

from halo.configs.classification import ClassificationScriptArguments
from halo.configs.distillation import SelfDistillScriptArguments, TeacherDistillScriptArguments
from halo.configs.embedding import EmbeddingScriptArguments
from halo.configs.grpo_environmental import EnvironmentalGRPOScriptArguments
from halo.configs.grpo_offline import OfflineGRPOScriptArguments
from halo.configs.grpo_online import OnlineGRPOScriptArguments
from halo.configs.parallelism_arguments import ParallelismArguments
from halo.configs.preference import DPOScriptArguments, KTOScriptArguments, SMPOScriptArguments
from halo.configs.reward import RewardScriptArguments
from halo.configs.script_arguments import ModelArguments
from halo.configs.sft import MTP_FIELDS, MtpArguments, SFTScriptArguments
from halo.entrypoints import model_setup
from halo.entrypoints.model_setup import load_model_and_tokenizer
from halo.entrypoints.script import ScriptContext
from halo.models.mtp.spec import MtpHeadRequest, MtpObjective
from halo.trainers.sft_rules import (
    reject_unsupported_mtp_options,
    reject_unsupported_vlm_sft_options,
    settle_mtp_balancing,
)
from tests.common.mtp_heads import MTP_TINY_FAMILIES
from tests.common.tiny_models import tiny_family_model


def _args(**fields) -> SFTScriptArguments:
    return SFTScriptArguments(dataset="unused", assistant_message_template="<a>", **fields)


def _sft_config(tmp_path, **fields) -> SFTConfig:
    return SFTConfig(output_dir=str(tmp_path), use_cpu=True, bf16=False, **fields)


def _head_only_clean(**fields) -> dict:
    return {"mtp_loss_weight": 1.0, "mtp_train_scope": "head_only", "mtp_min_acceptance_gain": None, **fields}


@pytest.fixture(scope="module")
def qwen_config():
    return tiny_family_model(MTP_TINY_FAMILIES["qwen3_5_moe_text"]).config


@pytest.fixture(scope="module")
def glm_config():
    return tiny_family_model(MTP_TINY_FAMILIES["glm4_moe_lite"]).config


def _reject(args, sft_config, config, *, model_args=None, dist_args=None) -> None:
    reject_unsupported_mtp_options(
        args,
        sft_config,
        model_args or ModelArguments(model_name_or_path="org/base"),
        dist_args or ParallelismArguments(),
        config,
    )


@pytest.mark.parametrize(
    "script_args",
    [
        ClassificationScriptArguments,
        DPOScriptArguments,
        EmbeddingScriptArguments,
        EnvironmentalGRPOScriptArguments,
        KTOScriptArguments,
        OfflineGRPOScriptArguments,
        OnlineGRPOScriptArguments,
        RewardScriptArguments,
        SMPOScriptArguments,
        SelfDistillScriptArguments,
        TeacherDistillScriptArguments,
    ],
)
def test_only_the_sft_script_carries_the_knobs(script_args):
    """A run that syncs weights, or scores its own logits, cannot reach a head: the parser's unknown-key raise keeps it
    out."""
    assert not set(MTP_FIELDS) & set(script_args.__dataclass_fields__)


def test_the_sft_script_carries_every_knob():
    assert set(MTP_FIELDS) <= set(SFTScriptArguments.__dataclass_fields__)


def test_by_default_no_head_is_requested():
    args = _args()
    assert args.mtp_loss_weight == 0 and args.mtp_head_request is None


def test_a_weight_makes_the_request_carry_the_objective_and_the_source():
    request = _args(
        mtp_loss_weight=0.3, mtp_loss_type="soft", mtp_train_scope="detached", mtp_head_source="org/x"
    ).mtp_head_request
    assert request == MtpHeadRequest(MtpObjective(0.3, "soft", "detached"), "org/x")


@pytest.mark.parametrize("weight", [-0.1, math.inf, math.nan])
def test_a_weight_off_its_domain_is_refused(weight):
    with pytest.raises(ValueError, match="mtp_loss_weight must be"):
        MtpArguments(mtp_loss_weight=weight)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mtp_loss_type", "soft"),
        ("mtp_train_scope", "detached"),
        ("mtp_head_source", "org/x"),
        ("mtp_min_acceptance_gain", 0.01),
        ("mtp_min_acceptance_gain", None),
    ],
)
def test_a_knob_set_with_no_weight_is_refused_as_inert(field, value):
    with pytest.raises(ValueError, match=field):
        MtpArguments(**{field: value})


def test_an_unknown_loss_type_or_scope_is_refused():
    with pytest.raises(ValueError, match="mtp_loss_type"):
        MtpArguments(mtp_loss_weight=1.0, mtp_loss_type="kl")
    with pytest.raises(ValueError, match="mtp_train_scope"):
        MtpArguments(mtp_loss_weight=1.0, mtp_train_scope="frozen")


@pytest.mark.parametrize("gain", [0.01, None])
def test_the_acceptance_gate_is_refused_off_head_only(gain):
    with pytest.raises(ValueError, match="mtp_min_acceptance_gain gates a head_only fit"):
        MtpArguments(mtp_loss_weight=1.0, mtp_train_scope="joint", mtp_min_acceptance_gain=gain)


def test_a_non_finite_gate_is_refused():
    with pytest.raises(ValueError, match="mtp_min_acceptance_gain must be a finite number"):
        MtpArguments(mtp_loss_weight=1.0, mtp_train_scope="head_only", mtp_min_acceptance_gain=math.inf)


def test_a_run_with_no_head_passes_every_rule_untouched(tmp_path):
    """Off means off: label smoothing, TP and an unreadable config are no business of this gate."""
    args = _args(tree_packing=False)
    reject_unsupported_mtp_options(
        args,
        _sft_config(tmp_path, label_smoothing_factor=0.1),
        SimpleNamespace(),
        SimpleNamespace(),
        None,
    )
    before = args.moe_balancing
    settle_mtp_balancing(args)
    assert args.moe_balancing == before


def test_a_clean_joint_run_passes(tmp_path, qwen_config):
    _reject(_args(mtp_loss_weight=0.3), _sft_config(tmp_path), qwen_config)


def test_a_clean_head_only_run_passes(tmp_path, qwen_config):
    _reject(_args(**_head_only_clean()), _sft_config(tmp_path, gradient_checkpointing=False), qwen_config)


@pytest.mark.parametrize(
    ("dist_fields", "named"),
    [
        ({"context_parallel_size": 2}, "context_parallel_size"),
        ({"tensor_parallel_size": 2}, "tensor_parallel_size"),
        ({"pipeline_parallel_size": 2}, "pipeline_parallel_size"),
        ({"use_hsdp": True}, "use_hsdp"),
        ({"lowp_precision": "fp8"}, "lowp_precision"),
    ],
)
def test_a_parallel_axis_the_head_cannot_run_under_is_refused(tmp_path, qwen_config, dist_fields, named):
    with pytest.raises(ValueError, match=named):
        _reject(
            _args(mtp_loss_weight=1.0),
            _sft_config(tmp_path),
            qwen_config,
            dist_args=ParallelismArguments(**dist_fields),
        )


def test_an_accelerate_launch_is_refused(tmp_path, qwen_config, monkeypatch):
    monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", "bf16")
    with pytest.raises(ValueError, match="torchrun"):
        _reject(_args(mtp_loss_weight=1.0), _sft_config(tmp_path), qwen_config)


@pytest.mark.parametrize(
    ("model_fields", "named"),
    [
        ({"use_peft": True}, "use_peft"),
        ({"load_in_4bit": True}, "quantized load"),
    ],
)
def test_an_adapter_or_a_quantized_load_is_refused(tmp_path, qwen_config, model_fields, named):
    with pytest.raises(ValueError, match=named):
        _reject(
            _args(mtp_loss_weight=1.0),
            _sft_config(tmp_path),
            qwen_config,
            model_args=ModelArguments(model_name_or_path="org/base", **model_fields),
        )


@pytest.mark.parametrize(
    ("sft_fields", "named"),
    [
        ({"torch_compile": True}, "torch_compile"),
        ({"label_smoothing_factor": 0.1}, "label_smoothing_factor"),
        ({"loss_type": "dft"}, "loss_type"),
    ],
)
def test_a_training_option_that_takes_the_labels_or_the_hooks_is_refused(tmp_path, qwen_config, sft_fields, named):
    with pytest.raises(ValueError, match=named):
        _reject(_args(mtp_loss_weight=1.0), _sft_config(tmp_path, **sft_fields), qwen_config)


def test_tree_packing_and_image_rows_are_refused(tmp_path, qwen_config):
    with pytest.raises(ValueError, match="tree_packing"):
        _reject(_args(mtp_loss_weight=1.0, tree_packing=True), _sft_config(tmp_path), qwen_config)
    with pytest.raises(ValueError, match="images_field"):
        _reject(_args(mtp_loss_weight=1.0, images_field="image"), _sft_config(tmp_path), qwen_config)


def test_the_vlm_path_refuses_a_head_even_on_a_preprocessed_dataset(tmp_path):
    with pytest.raises(ValueError, match="mtp_loss_weight"):
        reject_unsupported_vlm_sft_options(_args(mtp_loss_weight=1.0), _sft_config(tmp_path), preprocessed=True)


def test_a_sharded_save_is_refused_only_where_the_export_writes_trunk_copies(tmp_path, qwen_config, glm_config):
    args = _args(mtp_loss_weight=1.0, save_sharded_ep=True)
    _reject(args, _sft_config(tmp_path), qwen_config)
    with pytest.raises(ValueError, match="save_sharded_ep"):
        _reject(args, _sft_config(tmp_path), glm_config)


def test_a_source_under_init_from_scratch_is_refused(tmp_path, qwen_config):
    with pytest.raises(ValueError, match="init_from_scratch"):
        _reject(
            _args(mtp_loss_weight=1.0, mtp_head_source="org/x"),
            _sft_config(tmp_path),
            qwen_config,
            dist_args=ParallelismArguments(init_from_scratch=True),
        )


def test_a_family_with_a_named_refusal_is_refused_before_the_load(tmp_path):
    config = SimpleNamespace(model_type="deepseek_v4")
    with pytest.raises(ValueError, match="vendor names"):
        _reject(_args(mtp_loss_weight=1.0), _sft_config(tmp_path), config)


def test_a_family_with_no_tail_is_refused_before_the_load(tmp_path):
    with pytest.raises(ValueError, match="no head to train"):
        _reject(_args(mtp_loss_weight=1.0), _sft_config(tmp_path), SimpleNamespace(model_type="gpt_oss"))


def test_an_unreadable_config_leaves_the_family_to_the_load(tmp_path):
    _reject(_args(mtp_loss_weight=1.0), _sft_config(tmp_path), None)


@pytest.mark.parametrize(
    ("args_fields", "sft_fields", "dist_fields", "named"),
    [
        ({}, {"gradient_checkpointing": True}, {}, "gradient_checkpointing off"),
        ({"gradient_checkpointing_plan": "auto"}, {}, {}, "gradient_checkpointing_plan"),
        ({"freeze_layers_patterns": ["x"]}, {}, {}, "freeze_layers_patterns"),
        ({"unfreeze_layers_patterns": ["x"]}, {}, {}, "unfreeze_layers_patterns"),
        ({"moe_balancing": "bias_update"}, {}, {}, "moe_balancing"),
        ({"moe_balancing": "bias_update_transient"}, {}, {}, "moe_balancing"),
        ({"moe_balancing": "aux_loss"}, {}, {}, "moe_balancing"),
        ({}, {}, {"init_from_scratch": True}, "init_from_scratch"),
        ({}, {}, {"fp32_non_ep_params": True}, "fp32_non_ep_params"),
        ({}, {}, {"fp32_experts": True}, "fp32_experts"),
        ({"mtp_min_acceptance_gain": 0.0}, {"eval_strategy": "no"}, {}, "an eval split"),
    ],
)
def test_head_only_refuses_what_a_frozen_trunk_cannot_honor(
    tmp_path, qwen_config, args_fields, sft_fields, dist_fields, named
):
    sft_config = _sft_config(tmp_path, **{"gradient_checkpointing": False, **sft_fields})
    with pytest.raises(ValueError, match=named):
        _reject(
            _args(**_head_only_clean(**args_fields)),
            sft_config,
            qwen_config,
            dist_args=ParallelismArguments(**dist_fields),
        )


def test_the_same_options_stay_open_on_a_joint_run(tmp_path, qwen_config):
    args = _args(mtp_loss_weight=1.0, moe_balancing="aux_loss", freeze_layers_patterns=["x"])
    _reject(args, _sft_config(tmp_path, gradient_checkpointing=True), qwen_config)


def test_a_gated_head_only_run_with_an_eval_split_passes(tmp_path, qwen_config):
    args = _args(mtp_loss_weight=1.0, mtp_train_scope="head_only")
    assert args.mtp_min_acceptance_gain == 0.0
    sft_config = _sft_config(tmp_path, gradient_checkpointing=False, eval_strategy="steps", eval_steps=10)
    _reject(args, sft_config, qwen_config)


def test_auto_balancing_settles_to_none_on_a_frozen_trunk_only():
    head_only = _args(**_head_only_clean())
    assert head_only.moe_balancing == "auto"
    settle_mtp_balancing(head_only)
    assert head_only.moe_balancing == "none"
    joint = _args(mtp_loss_weight=1.0)
    settle_mtp_balancing(joint)
    assert joint.moe_balancing == "auto"


@pytest.mark.parametrize("request_value", [None, MtpHeadRequest(MtpObjective(1.0))], ids=["off", "on"])
def test_the_request_reaches_the_distributed_load_unchanged(tmp_path, monkeypatch, request_value):
    seen = {}

    def load(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(config=SimpleNamespace()), SimpleNamespace()

    monkeypatch.setattr(model_setup, "load_distributed_model", load)
    training = SimpleNamespace(
        output_dir=str(tmp_path),
        model_init_kwargs=None,
        bf16=True,
        fp16=False,
        use_liger_kernel=False,
        liger_kernel_config=None,
    )
    context = ScriptContext(SimpleNamespace(), 0, None, "org/base", object)
    load_model_and_tokenizer(
        context,
        training,
        ModelArguments(model_name_or_path="org/base"),
        SimpleNamespace(reset_sinks=True, train_sinks=False, text_only_model=False),
        mtp_head=request_value,
    )
    assert seen["mtp_head"] is request_value


def _resume_load(tmp_path, monkeypatch, caplog, *, policy_from_checkpoint: bool):
    monkeypatch.setattr(
        model_setup,
        "load_distributed_model",
        lambda **_kwargs: (SimpleNamespace(config=SimpleNamespace(_name_or_path="x")), SimpleNamespace()),
    )
    training = SimpleNamespace(
        output_dir=str(tmp_path),
        model_init_kwargs=None,
        bf16=True,
        fp16=False,
        use_liger_kernel=False,
        liger_kernel_config=None,
    )
    checkpoint = str(tmp_path / "checkpoint-3")
    source = checkpoint if policy_from_checkpoint else "org/base"
    with caplog.at_level(logging.INFO, logger=model_setup.__name__):
        load_model_and_tokenizer(
            ScriptContext(SimpleNamespace(), 0, checkpoint, source, object),
            training,
            ModelArguments(model_name_or_path="org/base"),
            SimpleNamespace(reset_sinks=True, train_sinks=False, text_only_model=False),
            mtp_head=MtpHeadRequest(MtpObjective(1.0)),
        )
    return [
        record.getMessage() for record in caplog.records if "MTP head read now is overwritten" in record.getMessage()
    ]


def test_a_resume_that_restores_over_the_base_says_the_head_read_is_overwritten(tmp_path, monkeypatch, caplog):
    assert _resume_load(tmp_path, monkeypatch, caplog, policy_from_checkpoint=False)


def test_a_resume_reading_the_checkpoint_itself_says_nothing_of_it(tmp_path, monkeypatch, caplog):
    assert not _resume_load(tmp_path, monkeypatch, caplog, policy_from_checkpoint=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
