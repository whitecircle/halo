#!/usr/bin/env python
"""Load-time refusals in ``model_loading.parallel`` and loader default parity.
EP/CP/TP require torchrun, so the gates reject every ``accelerate launch`` distributed_type, DDP included."""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import inspect
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch.distributed.tensor import DTensor
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
from transformers.distributed.tensor_parallel import apply_tensor_parallelism
from transformers.models.qwen3 import Qwen3Config
from transformers.models.qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_vl import Qwen3VLConfig

from halo.configs.script_arguments import ModelLoadArguments
from halo.distributed.mesh import MeshDim
from halo.distributed.parallelism_config import ParallelismConfig
from halo.distributed.parallelism_rules import reject_accelerate_launch
from halo.distributed.tensor_parallel.hf_plan import plan_shards_param
from halo.kernels.lowp.spec import LowpSpec
from halo.model_loading.dtype import resolve_load_dtype
from halo.model_loading.lazy_safetensors.meta_shell import instantiate_on_meta
from halo.model_loading.parallel import load_rules, whole_model
from halo.model_loading.parallel import loader as model_loading
from halo.model_loading.parallel import tp as tp_loading
from halo.model_loading.parallel.cp import load_cp_model
from halo.model_loading.parallel.ep import load_ep_model_lazy
from halo.model_loading.parallel.frozen_models import load_frozen_auxiliary_model
from halo.model_loading.parallel.load_rules import reject_unsupported_model, reject_unsupported_run
from halo.model_loading.parallel.loader import load_distributed_model
from halo.model_loading.parallel.model_source import ModelSource
from halo.model_loading.parallel.pp import _load_stage_lazy
from halo.models.attention.gpt_oss_sinks import SinksPolicy
from halo.models.attention.selection import resolve_attn_implementation
from halo.models.moe.family_specs import GEMMA4, MoEFamilySpec
from halo.trainers.accelerator import accelerate_manages_ddp
from tests.common.distributed import fake_process_group_mesh
from tests.common.models import TINY_QWEN3_MOE_CONFIG
from tests.common.parallelism import make_parallelism_config

_CP_LOADING = "halo.model_loading.parallel.cp"
_SOURCE = ModelSource("org/x", None, "org/x")

_MULTI_GPU_ENV = {"ACCELERATE_MIXED_PRECISION": "bf16"}
_FSDP_ENV = {"ACCELERATE_MIXED_PRECISION": "bf16", "ACCELERATE_USE_FSDP": "true"}
_ACCELERATE_VARS = ("ACCELERATE_MIXED_PRECISION", "ACCELERATE_USE_FSDP")
# The mesh the fixture shards on is the mesh the load runs on; a width the config does not declare models no rank's
# real load.
_TP_SIZE = 2


def _env(overrides: dict[str, str]) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in _ACCELERATE_VARS}
    env.update(overrides)
    return env


def _pc(**kwargs) -> ParallelismConfig:
    return make_parallelism_config(world_size=2, gpus_per_node=2, **kwargs)


@pytest.mark.parametrize("launch_env", [_MULTI_GPU_ENV, _FSDP_ENV])
def test_loader_gate_rejects_ep_under_any_accelerate_launch(launch_env):
    with patch.dict(os.environ, _env(launch_env), clear=True):
        with pytest.raises(ValueError, match="torchrun"):
            reject_unsupported_run(_pc(ep_size=2), torch.bfloat16)


def test_loader_gate_passes_under_torchrun_and_for_plain_dp():
    with patch.dict(os.environ, _env({}), clear=True):
        reject_unsupported_run(_pc(ep_size=2), torch.bfloat16)
    with patch.dict(os.environ, _env(_MULTI_GPU_ENV), clear=True):
        reject_unsupported_run(_pc(), torch.bfloat16)


@pytest.mark.parametrize("launch_env", [_MULTI_GPU_ENV, _FSDP_ENV])
def test_trainer_gate_rejects_ep_under_any_accelerate_launch(launch_env):
    with patch.dict(os.environ, _env(launch_env), clear=True):
        with pytest.raises(ValueError, match="accelerate launch"):
            reject_accelerate_launch(_pc(ep_size=2))


def test_trainer_gate_passes_under_torchrun():
    with patch.dict(os.environ, _env({}), clear=True):
        reject_accelerate_launch(_pc(ep_size=2))


@pytest.mark.parametrize("launch_env", [_MULTI_GPU_ENV, _FSDP_ENV])
def test_the_grouped_gemm_wrappers_are_refused_under_any_accelerate_launch(launch_env):
    """Plain DP under accelerate would hand the wrapped experts to accelerate's DDP or FSDP, which they are not
    built for."""
    with patch.dict(os.environ, _env(launch_env), clear=True):
        with pytest.raises(ValueError, match=r"Grouped GEMM for MoE experts .* require 'torchrun'") as refusal:
            reject_unsupported_model(
                _pc(),
                Qwen3MoeConfig(**TINY_QWEN3_MOE_CONFIG),
                sinks_policy=SinksPolicy.NEUTRALIZED,
                wraps_experts=True,
            )
    assert "use_grouped_gemm: false" in str(refusal.value)


def test_the_grouped_gemm_wrappers_pass_under_torchrun_and_unwrapped_experts_under_accelerate():
    with patch.dict(os.environ, _env({}), clear=True):
        reject_unsupported_model(
            _pc(), Qwen3MoeConfig(**TINY_QWEN3_MOE_CONFIG), sinks_policy=SinksPolicy.NEUTRALIZED, wraps_experts=True
        )
    with patch.dict(os.environ, _env(_MULTI_GPU_ENV), clear=True):
        reject_unsupported_model(
            _pc(), Qwen3MoeConfig(**TINY_QWEN3_MOE_CONFIG), sinks_policy=SinksPolicy.NEUTRALIZED, wraps_experts=False
        )


def test_pure_etp_is_not_swept_up_by_the_fp32_refusal():
    """``has_ep_group`` is true at ``ep_size=1``, but that dispatcher short-circuits, so keying on it would
    over-refuse."""
    pc = _pc(ep_size=1, expert_tp_size=2)
    assert pc.has_ep_group, "premise: expert-TP folds into has_ep_group, which is why the guard cannot use it"
    with patch.dict(os.environ, _env({}), clear=True), pytest.raises(Exception) as excinfo:
        load_distributed_model("org/x", parallelism_config=pc, dtype=torch.float32, trust_remote_code=True)
    assert "fp32 training" not in str(excinfo.value)


def test_loader_allows_fp32_without_expert_parallelism():
    """DeepEP's dispatch buffer is built for 2-byte tokens; 4-byte ones trip a raw C++ assert at the first dispatch.
    The refusal keys on EP since the dense fp32 path works (measured 699 tok/s on qwen3-8b)."""
    with patch.dict(os.environ, _env({}), clear=True):
        with pytest.raises(ValueError, match="fp32 training"):
            load_distributed_model(
                "org/x", parallelism_config=_pc(ep_size=2), dtype=torch.float32, trust_remote_code=True
            )
        with pytest.raises(Exception) as excinfo:
            load_distributed_model("org/x", parallelism_config=_pc(), dtype=torch.float32, trust_remote_code=True)
        assert "fp32 training" not in str(excinfo.value)


class _FakeDeviceMesh:
    """``apply_fsdp2_for_tp`` reads the stamped mesh's ``mesh_dim_names`` and indexes it, so a bare string would be
    a mesh no code path consumes."""

    mesh_dim_names = (MeshDim.DP, MeshDim.TP)

    def __getitem__(self, dim_name):
        return f"{dim_name}-submesh"


_DP_TP_MESH = _FakeDeviceMesh()


@contextlib.contextmanager
def _load_collaborators(model):
    """``joined_node_load`` is stubbed to a region the weight load must not run inside: the load is never
    rank-serialized, the restore after it may be."""
    throttled = []

    @contextlib.contextmanager
    def _throttle(*_args, **_kwargs):
        throttled.append(True)
        yield
        throttled.pop()

    def _load(*_args, **_kwargs):
        assert not throttled, "load_tp_model must not rank-serialize its load"
        return model

    with (
        patch.object(tp_loading, "build_dp_tp_mesh", return_value=_DP_TP_MESH),
        patch.object(tp_loading, "tp_submesh", return_value="tp-mesh"),
        patch.object(tp_loading, "joined_node_load", _throttle),
        patch.object(tp_loading, "from_pretrained_verified", _load),
        patch("halo.distributed.tensor_parallel.hf_plan.retarget_hf_replicated_grad_hooks", lambda _model: None),
    ):
        yield


def test_tp_load_never_rank_serializes_its_load(tp_materialized_model):
    """``tie_weights`` compares tied DTensors with a mesh-wide collective, so a throttle deadlocks a tied model
    until the store timeout."""
    with _load_collaborators(tp_materialized_model):
        tp_loading.load_tp_model(
            _SOURCE,
            _pc(tp_size=_TP_SIZE),
            AutoModelForCausalLM,
            {"config": tp_materialized_model.config},
            preserve_checkpoint_precision=False,
        )


def test_the_tp_fp32_master_restore_is_node_batched(tp_materialized_model):
    """The restore stages each full FP32 tensor in host memory and runs no collective, so unlike the load it is
    batched per node."""
    events = []

    @contextlib.contextmanager
    def throttle(what, max_concurrent):
        events.append(("batched", max_concurrent))
        yield
        events.append("released")

    with (
        _load_collaborators(tp_materialized_model),
        patch.object(tp_loading, "joined_node_load", throttle),
        patch.object(tp_loading, "restore_fp32_master_parameters", lambda *a, **k: events.append("restore")),
    ):
        tp_loading.load_tp_model(
            _SOURCE,
            _pc(tp_size=_TP_SIZE, max_concurrent_loading=1),
            AutoModelForCausalLM,
            {"config": tp_materialized_model.config},
            preserve_checkpoint_precision=False,
        )
    assert events == [("batched", 1), "restore", "released"], events


@pytest.fixture
def sharding_model():
    """Function-scoped because :func:`tp_materialized_model` shards it in place."""
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    return AutoModelForCausalLM.from_config(config)


@pytest.fixture
def tp_materialized_model(sharding_model):
    """The post-load guard reads the parameters, so an unsharded stand-in would be the materialization regression it
    exists for. Sharded by transformers' own ``apply_tensor_parallelism`` over a fake process group, which slices
    locally with no collective."""
    with fake_process_group_mesh(rank=0, world_size=_TP_SIZE) as mesh:
        yield apply_tensor_parallelism(sharding_model, mesh)


@pytest.fixture(scope="module")
def planless_model():
    """``base_model_tp_plan`` is None on both levels, so ``tp_plan="auto"`` resolves to the wrapper's head entry
    alone and transformers only warns."""
    config = Qwen3VLConfig(
        text_config={
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 64,
        },
        vision_config={"hidden_size": 32, "intermediate_size": 64, "depth": 2, "num_heads": 4, "out_hidden_size": 32},
    )
    return AutoModelForImageTextToText.from_config(config)


def test_the_plan_is_what_separates_a_sharding_architecture_from_a_planless_one(sharding_model, planless_model):
    """If the two collapsed to one answer, the guard could not tell a working TP run from a wasted one."""
    sharded = [
        name for name, _ in sharding_model.named_parameters() if plan_shards_param(name, sharding_model._tp_plan)
    ]
    assert sharded, "premise: qwen3 ships a base_model_tp_plan covering q/k/v/o and the MLP"
    assert any("q_proj" in name for name in sharded) and any("down_proj" in name for name in sharded)

    head = planless_model.get_output_embeddings().weight
    assert set(planless_model._tp_plan or {}) <= {"lm_head"}, "premise: qwen3-vl ships no base_model_tp_plan"
    assert not [
        name
        for name, param in planless_model.named_parameters()
        if param is not head and plan_shards_param(name, planless_model._tp_plan or {})
    ]


def test_tp_load_refuses_an_architecture_whose_plan_shards_nothing(planless_model):
    """Silent otherwise: ``tp_size`` identical replicas at ``1/tp_size`` throughput."""
    with _load_collaborators(planless_model), pytest.raises(ValueError, match="sharded ZERO parameters"):
        tp_loading.load_tp_model(
            _SOURCE,
            _pc(tp_size=_TP_SIZE),
            AutoModelForImageTextToText,
            {"config": planless_model.config},
            preserve_checkpoint_precision=False,
        )


def test_tp_load_accepts_an_architecture_that_does_shard(tp_materialized_model):
    """What transformers stamps on a distributed model is cleared: the toolkit owns the DP wrap, loss scaling and
    checkpoints."""
    tp_materialized_model._is_distributed_loading_by_transformers = True
    tp_materialized_model._tp_size = _TP_SIZE
    with _load_collaborators(tp_materialized_model):
        model = tp_loading.load_tp_model(
            _SOURCE,
            _pc(tp_size=_TP_SIZE),
            AutoModelForCausalLM,
            {"config": tp_materialized_model.config},
            preserve_checkpoint_precision=False,
        )
    assert model is tp_materialized_model
    assert model._device_mesh is _DP_TP_MESH
    assert not model.is_distributed_loading_by_transformers and model.tp_size is None


def test_tp_load_refuses_a_plainly_materialized_load(sharding_model):
    """Every rank would hold a bare slice the trainer averages against its peers' disjoint slices, and a plain
    tensor looks like a genuine replica."""
    assert not any(isinstance(param, DTensor) for param in sharding_model.parameters()), (
        "premise: this fixture is the un-materialized load"
    )
    with _load_collaborators(sharding_model), pytest.raises(ValueError, match="materialized as plain"):
        tp_loading.load_tp_model(
            _SOURCE,
            _pc(tp_size=_TP_SIZE),
            AutoModelForCausalLM,
            {"config": sharding_model.config},
            preserve_checkpoint_precision=False,
        )


def test_tp_load_is_inert_at_tp_size_one(planless_model):
    """Refusing at ``tp_size=1`` would ban the model outright."""
    with _load_collaborators(planless_model):
        loaded = tp_loading.load_tp_model(
            _SOURCE,
            _pc(),
            AutoModelForImageTextToText,
            {"config": planless_model.config},
            preserve_checkpoint_precision=False,
        )
    assert loaded is planless_model


class _ModelConfig:
    """A ``model_type``, plus the composite layout where the family's lives on the text sub-config."""

    def __init__(self, model_type: str, text: _ModelConfig | None = None):
        self.model_type = model_type
        self._text = text

    def get_text_config(self):
        return self._text


def _reject_unsupported_model(pc: ParallelismConfig, model_config) -> None:
    reject_unsupported_model(pc, model_config, sinks_policy=SinksPolicy.NEUTRALIZED, wraps_experts=False)


@pytest.mark.parametrize("model_type", ["gemma4", "gemma4_text"])
def test_fp32_non_ep_params_is_refused_for_a_family_that_declares_it_unsupported(model_type):
    """The alternative is DeepEP's raw C++ assert at the first dispatch, after the whole multi-GPU load."""
    with pytest.raises(ValueError, match="fp32_non_ep_params") as excinfo:
        _reject_unsupported_model(_pc(ep_size=2, fp32_non_ep_params=True), _ModelConfig(model_type))
    assert model_type in str(excinfo.value), "the refusal must name the model_type that triggered it"


def test_the_text_tower_is_consulted_for_a_composite_config():
    """Reading only the wrapper's ``model_type`` would let the refused family through under any VLM checkpoint."""
    composite = _ModelConfig("some_multimodal_wrapper", text=_ModelConfig("gemma4_text"))
    with pytest.raises(ValueError, match="gemma4_text"):
        _reject_unsupported_model(_pc(ep_size=2, fp32_non_ep_params=True), composite)


@pytest.mark.parametrize("model_type", ["qwen3_moe", "qwen3"])
def test_families_that_support_the_upcast_and_dense_models_are_left_alone(model_type):
    """A guard that refused either would ban a validated production shape."""
    _reject_unsupported_model(_pc(ep_size=2, fp32_non_ep_params=True), _ModelConfig(model_type))


@pytest.mark.parametrize("pc_kwargs", [{"ep_size": 1}, {"ep_size": 2, "fp32_non_ep_params": False}])
def test_the_refusal_needs_both_conjuncts(pc_kwargs):
    """No experts distributed means no DeepEP dispatch to break, and without the knob nothing to refuse."""
    kwargs = {"fp32_non_ep_params": True, **pc_kwargs}
    _reject_unsupported_model(_pc(**kwargs), _ModelConfig("gemma4"))


def test_the_class_declaration_is_what_drives_the_refusal(monkeypatch):
    """Anti-vacuity: flip the declaration on the family spec and the same call must pass."""
    assert MoEFamilySpec(model_types=()).supports_fp32_non_ep_params is True, "the default must be permissive"
    with pytest.raises(ValueError):
        _reject_unsupported_model(_pc(ep_size=2, fp32_non_ep_params=True), _ModelConfig("gemma4"))

    permissive = dataclasses.replace(GEMMA4, supports_fp32_non_ep_params=True)
    monkeypatch.setattr(load_rules, "find_moe_family_spec_for_config", lambda _config: permissive)
    _reject_unsupported_model(_pc(ep_size=2, fp32_non_ep_params=True), _ModelConfig("gemma4"))


def test_the_family_gate_is_wired_into_the_loader_right_after_the_config_read():
    """Everything but the download and config read is real, so a dropped call site fails here."""
    config = _ModelConfig("gemma4")
    with (
        patch.dict(os.environ, _env({}), clear=True),
        patch.object(
            model_loading, "resolve_model_source", lambda path, revision, **kwargs: ModelSource(path, revision, path)
        ),
        patch.object(model_loading, "AutoConfig", SimpleNamespace(from_pretrained=lambda *a, **k: config)),
        pytest.raises(ValueError, match="fp32_non_ep_params"),
    ):
        load_distributed_model(
            "org/x", parallelism_config=_pc(ep_size=2, fp32_non_ep_params=True), trust_remote_code=True
        )


_FP8 = LowpSpec(precision="fp8")
_QUANTIZED = SimpleNamespace(load_in_4bit=True)
_LOAD_REFUSALS = {
    "qlora_under_ep": ({"ep_size": 2}, {"quantization_config": _QUANTIZED}, ValueError, "Quantized loading"),
    "qlora_under_tp": ({"tp_size": 2}, {"quantization_config": _QUANTIZED}, ValueError, "Quantized loading"),
    "scratch_under_cp": ({"cp_size": 2}, {"init_from_scratch": True}, NotImplementedError, "dense data-parallel"),
    "scratch_with_qlora": (
        {},
        {"init_from_scratch": True, "quantization_config": _QUANTIZED},
        ValueError,
        "incompatible with quantization",
    ),
    "lowp_with_qlora": ({"lowp": _FP8}, {"quantization_config": _QUANTIZED}, ValueError, "quantization_config"),
    "lowp_under_fp16": ({"lowp": _FP8}, {"dtype": torch.float16}, ValueError, "dtype=float16"),
}


def _reached(stage: str):
    def fail(*args, **kwargs):
        raise AssertionError(f"{stage} ran before the load refusal")

    return fail


@pytest.mark.parametrize(
    ("pc_fields", "load_kwargs", "error", "fragment"), _LOAD_REFUSALS.values(), ids=_LOAD_REFUSALS
)
def test_every_load_refusal_fires_before_attention_the_tokenizer_and_the_weight_read(
    pc_fields, load_kwargs, error, fragment
):
    """An invalid run costs a config read: no attention resolution, tokenizer download, process groups or full load."""
    config = Qwen3Config(
        num_hidden_layers=1, hidden_size=64, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2
    )
    with (
        patch.dict(os.environ, _env({}), clear=True),
        patch.object(
            model_loading, "resolve_model_source", lambda path, revision, **kwargs: ModelSource(path, revision, path)
        ),
        patch.object(model_loading, "AutoConfig", SimpleNamespace(from_pretrained=lambda *a, **k: config)),
        patch.object(model_loading, "_resolve_run_attention", _reached("the attention resolution")),
        patch.object(model_loading, "AutoTokenizer", SimpleNamespace(from_pretrained=_reached("the tokenizer load"))),
        patch.object(model_loading, "_dispatch_model_loading", _reached("the weight read")),
        pytest.raises(error, match=fragment),
    ):
        load_distributed_model("org/x", parallelism_config=_pc(**pc_fields), trust_remote_code=True, **load_kwargs)


def test_qlora_passes_plain_data_parallelism_and_is_refused_under_the_expert_wrappers():
    load_rules.reject_unsupported_load(
        _pc(), torch.bfloat16, quantization_config=_QUANTIZED, init_from_scratch=False, wraps_experts=False
    )
    with pytest.raises(ValueError, match="grouped-GEMM MoE loaders"):
        load_rules.reject_unsupported_load(
            _pc(), torch.bfloat16, quantization_config=_QUANTIZED, init_from_scratch=False, wraps_experts=True
        )


def test_scratch_and_lowp_pass_where_each_is_supported():
    """Anti-vacuity: both are production shapes."""
    load_rules.reject_unsupported_load(
        _pc(), torch.bfloat16, quantization_config=None, init_from_scratch=True, wraps_experts=False
    )
    load_rules.reject_unsupported_load(
        _pc(lowp=_FP8), torch.bfloat16, quantization_config=None, init_from_scratch=False, wraps_experts=False
    )


def test_accelerate_manages_ddp_semantics():
    config = SimpleNamespace(has_custom_parallelism=False)
    with patch.dict(os.environ, _env(_MULTI_GPU_ENV), clear=True):
        assert accelerate_manages_ddp(config) is True
    with patch.dict(os.environ, _env(_FSDP_ENV), clear=True):
        assert accelerate_manages_ddp(config) is False
    with patch.dict(os.environ, _env({}), clear=True):
        assert accelerate_manages_ddp(config) is False


def test_the_run_dtype_reaches_transformers_under_its_own_key(tp_materialized_model):
    """``from_pretrained`` folds the deprecated ``torch_dtype`` alias, so a re-spelled key would look correct while
    depending on it."""
    common = {"config": tp_materialized_model.config, "dtype": torch.bfloat16, "trust_remote_code": True}
    with (
        _load_collaborators(tp_materialized_model),
        patch.object(tp_loading, "from_pretrained_verified", return_value=tp_materialized_model) as loader,
    ):
        tp_loading.load_tp_model(
            _SOURCE, _pc(tp_size=_TP_SIZE), AutoModelForCausalLM, common, preserve_checkpoint_precision=False
        )

    passed = loader.call_args.kwargs
    assert passed["dtype"] is torch.bfloat16
    assert "torch_dtype" not in passed
    assert common["dtype"] is torch.bfloat16, "the caller's dict is shared with the other loaders"


def test_the_cp_loader_cannot_install_an_unvalidated_attention_implementation():
    """An optional config with an unconditional install branch would pass the raw request, training GPT-OSS with
    live sinks on a silently different kernel."""
    seen = {}

    def fake_load_whole_model(model_class, source, pc, **kwargs):
        seen.update(kwargs)
        return torch.nn.Linear(2, 2)

    pc = SimpleNamespace(
        cp_config=lambda: SimpleNamespace(cp_size=2), max_concurrent_loading=None, fp32_non_ep_params=False
    )
    config = Qwen3Config(num_hidden_layers=1, hidden_size=8, intermediate_size=8, num_attention_heads=2)
    with (
        patch(f"{_CP_LOADING}.load_whole_model", fake_load_whole_model),
        patch(f"{_CP_LOADING}.UlyssesModel", lambda model, cp_config: model),
        patch("halo.models.attention.selection.validate_attn_implementation", return_value="sdpa") as validator,
    ):
        with pytest.raises(KeyError, match="config"):
            load_cp_model(
                _SOURCE,
                pc,
                AutoModelForCausalLM,
                {"attn_implementation": "flash_attention_2"},
                preserve_checkpoint_precision=False,
                wraps_experts=False,
            )
        load_cp_model(
            _SOURCE,
            pc,
            AutoModelForCausalLM,
            {"config": config, "attn_implementation": "flash_attention_2"},
            preserve_checkpoint_precision=False,
            wraps_experts=False,
        )

    validator.assert_called_once_with(config, "flash_attention_2")
    assert seen["attn_implementation"] == "sdpa", (
        f"the loader forwarded {seen.get('attn_implementation')!r}, not the validator's verdict"
    )


def test_every_loader_names_the_run_dtype_dtype():
    """A signature drifting back to ``torch_dtype`` is a TypeError only the mode reaching that loader hits."""
    for loader in (
        load_distributed_model,
        load_ep_model_lazy,
        _load_stage_lazy,
        load_frozen_auxiliary_model,
        instantiate_on_meta,
        resolve_load_dtype,
        resolve_attn_implementation,
    ):
        params = inspect.signature(loader).parameters
        assert "dtype" in params, f"{loader.__name__} does not take the run's dtype as `dtype`"
        assert "torch_dtype" not in params, f"{loader.__name__} still spells the run's dtype `torch_dtype`"


def test_no_transformers_call_in_the_loader_passes_the_deprecated_dtype_alias():
    """Covers the two loaders the CPU suite cannot drive (both end in ``.to(cuda)``); the alias's only symptom is a
    log line."""
    boundary = ("from_pretrained_verified", "from_pretrained", "from_config")
    offenders = [
        f"{callee}() at line {node.lineno}"
        for module in (model_loading, tp_loading, whole_model)
        for node in ast.walk(ast.parse(inspect.getsource(module)))
        if isinstance(node, ast.Call)
        for callee in [node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")]
        if callee in boundary
        for keyword in node.keywords
        if keyword.arg == "torch_dtype"
    ]
    assert not offenders, f"pass the run's dtype as `dtype=`: {offenders}"


def test_reset_sinks_defaults_agree_across_loaders():
    entry = next(f.default for f in dataclasses.fields(ModelLoadArguments) if f.name == "reset_sinks")
    direct = inspect.signature(load_distributed_model).parameters["reset_sinks"].default
    # The frozen loader is scored against the policy loaders, so a different default pairs a reset policy with a
    # live-sink reference.
    frozen = inspect.signature(load_frozen_auxiliary_model).parameters["reset_sinks"].default
    assert entry is True and direct is True and frozen is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
