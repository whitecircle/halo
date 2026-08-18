#!/usr/bin/env python
"""CPU tests for the load-time refusals in ``model_loading`` and loader default parity.

EP/CP/TP require torchrun; the gates must reject EVERY ``accelerate launch`` distributed_type —
including MULTI_GPU/DDP, which sets ``ACCELERATE_MIXED_PRECISION`` but not ``ACCELERATE_USE_FSDP``
(rejecting only the FSDP launch lets DDP fall through into custom parallelism). Also pins
``_should_accelerate_manage_ddp`` semantics over the ``src.env`` helpers, and the ``reset_sinks``
default parity between the two model-loading entry points.

Three more refusals raised from the same module are pinned here, each of which replaces a failure
that lands far from its cause: ``_load_tp_model``'s zero-shard check (an architecture with no
``base_model_tp_plan`` silently trains ``tp_size`` full replicas at ``1/tp_size`` throughput), its
plan-vs-materialization check (a plan-sharded param that came back a plain tensor is a bare slice
the gradient sync averages against its peers' disjoint slices), and ``_validate_fp32_non_ep_params``
(otherwise a raw DeepEP C++ assert at the first dispatch, naming neither fp32 nor the knob).

The TP-load tests run against a model transformers itself materialized on a device mesh: the
materialization check reads the loaded parameters, not only the plan.

Run: ``python tests/cpu/parallelism/test_launch_and_loading_gates.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from accelerate import PartialState
from torch.distributed.tensor import DTensor
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
from transformers.distributed.tensor_parallel import apply_tensor_parallelism
from transformers.models.qwen3 import Qwen3Config
from transformers.models.qwen3_vl import Qwen3VLConfig

from src.distributed.context_parallel.loading import load_model_for_cp, load_model_for_ep_cp

_CP_LOADING = "src.distributed.context_parallel.loading"
from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.expert_parallel.layers.gemma4 import EPGemma4MoELayer
from src.distributed.expert_parallel.lazy_loader import load_ep_model_lazy
from src.distributed.expert_parallel.loading import load_ep_model
from src.distributed.loading import model_loading
from src.distributed.loading.frozen_models import load_frozen_auxiliary_model
from src.distributed.loading.model_loading import (
    _load_tp_model,
    _validate_fp32_non_ep_params,
    _validate_launch_method_for_parallelism,
    load_distributed_model,
)
from src.distributed.loading.vlm_setup import load_model_for_training
from src.distributed.mesh import MeshDim
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.pipeline_parallel.lazy_loader import load_pp_stage_model
from src.distributed.tensor_parallel.state_dict import tp_plan_shards_params
from src.models.loading.lazy_safetensors.meta_shell import instantiate_on_meta
from src.models.loading.lazy_safetensors.weights import resolve_run_dtype
from src.models.patches.attention import resolve_attn_implementation
from src.trainers.mixins.base import DistributedTrainerMixin
from src.trainers.mixins.validation import ParallelismValidationMixin
from tests.common.distributed import fake_process_group_mesh
from tests.common.parallelism import make_parallelism_config

PartialState()  # the loader's accelerate logger refuses to emit without an initialized state

# accelerate MULTI_GPU (DDP) launch: MIXED_PRECISION set, USE_FSDP unset.
_MULTI_GPU_ENV = {"ACCELERATE_MIXED_PRECISION": "bf16"}
_FSDP_ENV = {"ACCELERATE_MIXED_PRECISION": "bf16", "ACCELERATE_USE_FSDP": "true"}
_ACCELERATE_VARS = ("ACCELERATE_MIXED_PRECISION", "ACCELERATE_USE_FSDP")
# One number for the TP-load tests: the mesh the fixture shards on IS the mesh the load runs on, and
# a fixture sharding at a width the config does not declare models no rank's real load.
_TP_SIZE = 2


def _env(overrides: dict[str, str]) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in _ACCELERATE_VARS}
    env.update(overrides)
    return env


def _pc(**kwargs) -> ParallelismConfig:
    return make_parallelism_config(world_size=2, gpus_per_node=2, **kwargs)


class _Validating(ParallelismValidationMixin):
    _supports_tp = True
    _supports_ep = True
    _supports_cp = True

    def __init__(self, pc: ParallelismConfig):
        self.parallelism_config = pc


@pytest.mark.parametrize("launch_env", [_MULTI_GPU_ENV, _FSDP_ENV])
def test_loader_gate_rejects_ep_under_any_accelerate_launch(launch_env):
    with patch.dict(os.environ, _env(launch_env), clear=True):
        with pytest.raises(ValueError, match="torchrun"):
            _validate_launch_method_for_parallelism(_pc(ep_size=2))


def test_loader_gate_passes_under_torchrun_and_for_plain_dp():
    with patch.dict(os.environ, _env({}), clear=True):
        _validate_launch_method_for_parallelism(_pc(ep_size=2))  # torchrun: no accelerate env
    with patch.dict(os.environ, _env(_MULTI_GPU_ENV), clear=True):
        _validate_launch_method_for_parallelism(_pc())  # accelerate DDP without EP/CP/TP is fine


@pytest.mark.parametrize("launch_env", [_MULTI_GPU_ENV, _FSDP_ENV])
def test_trainer_gate_rejects_ep_under_any_accelerate_launch(launch_env):
    with patch.dict(os.environ, _env(launch_env), clear=True):
        with pytest.raises(ValueError, match="accelerate launch"):
            _Validating(_pc(ep_size=2))._validate_parallelism_modes()


def test_trainer_gate_passes_under_torchrun():
    with patch.dict(os.environ, _env({}), clear=True):
        _Validating(_pc(ep_size=2))._validate_parallelism_modes()


def test_pure_etp_is_not_swept_up_by_the_fp32_refusal():
    """``ep_size=1`` + expert-TP never reaches the transport the refusal is about.

    ``is_ep_mode`` is true there — ``ep_group_size = ep_size * expert_tp_size`` — so keying the guard
    on it would refuse a shape whose dispatcher short-circuits at ``ep_size <= 1`` and whose
    ``dispatch_ep_group`` is left None. A guard must refuse what its stated mechanism governs and no
    more; if fp32 breaks pure ETP elsewhere, that needs its own message naming its own cause.
    """
    pc = _pc(ep_size=1, expert_tp_size=2)
    assert pc.is_ep_mode, "premise: expert-TP folds into is_ep_mode, which is why the guard cannot use it"
    with patch.dict(os.environ, _env({}), clear=True), pytest.raises(Exception) as excinfo:
        load_distributed_model("org/x", parallelism_config=pc, dtype=torch.float32)
    assert "fp32 training" not in str(excinfo.value)


def test_loader_allows_fp32_without_expert_parallelism():
    """fp32 + EP must be refused before the model loads, not asserted on by DeepEP mid-step: DeepEP
    sizes its dispatch buffer from the token dtype's element_size and is built for 2-byte tokens;
    4-byte ones trip a raw C++ assertion in ``get_dispatch_buffer_size`` at the first dispatch —
    after the whole model is resident, naming neither fp32 nor the config that caused it.

    Anti-vacuity: the refusal must key on EP, not on fp32 alone — the dense fp32 path works
    (measured 699 tok/s on qwen3-8b), so a blanket fp32 rejection would break it."""
    with patch.dict(os.environ, _env({}), clear=True):
        with pytest.raises(ValueError, match="fp32 training"):
            load_distributed_model("org/x", parallelism_config=_pc(ep_size=2), dtype=torch.float32)
        # Same call without EP must get past the guard (it fails later, on the absent repo).
        with pytest.raises(Exception) as excinfo:
            load_distributed_model("org/x", parallelism_config=_pc(), dtype=torch.float32)
        assert "fp32 training" not in str(excinfo.value)


# ── TP: an architecture with no plan shards nothing ──────────────────────


class _FakeDeviceMesh:
    """Stand-in for the (dp, tp) mesh, shaped like one: ``setup_fsdp2_for_tp`` reads the stamped
    mesh's ``mesh_dim_names`` and indexes it, so a bare string here would be a mesh no code path
    could consume."""

    mesh_dim_names = (MeshDim.DP, MeshDim.TP)

    def __getitem__(self, dim_name):
        return f"{dim_name}-submesh"


_DP_TP_MESH = _FakeDeviceMesh()


@contextlib.contextmanager
def _load_collaborators(model):
    """Stub only what needs CUDA or a process group, so the guard sees a real model.

    The applied ``_tp_plan``, the parameter names and the predicate are all genuine — which is the
    whole point, since the guard's correctness is exactly "does it read what HF's sharder read".

    ``sequential_load_within_node`` is stubbed to RAISE, not to a nullcontext: this load must never
    be rank-serialized (see the test below).
    """

    def _refuse(**_kwargs):
        raise AssertionError("_load_tp_model must not rank-serialize its load")

    with (
        patch.object(model_loading, "create_dp_tp_mesh", return_value=_DP_TP_MESH),
        patch.object(model_loading, "get_tp_submesh", return_value="tp-mesh"),
        patch.object(model_loading, "sequential_load_within_node", _refuse),
        patch.object(model_loading, "from_pretrained_verified", return_value=model),
        patch.object(model_loading, "retarget_hf_replicated_grad_hooks", lambda _model: None),
    ):
        yield


def test_tp_load_never_rank_serializes_its_load(tp_materialized_model):
    """A ``max_concurrent_loading`` throttle here DEADLOCKS a tied model.

    transformers ends the load in ``tie_weights``, which compares the two tied parameters when the
    checkpoint carries both keys (the Qwen3-0.6B/1.7B export shape) — on the DTensors HF-native TP
    produces that comparison is a mesh-wide collective. Inside a rank-serialized region the loading
    rank blocks in it while its peers wait their turn, and the job hangs until the store timeout
    rather than failing. The stub above turns a re-added throttle into this assertion.
    """
    with _load_collaborators(tp_materialized_model):
        _load_tp_model("org/x", _pc(tp_size=_TP_SIZE), AutoModelForCausalLM, {"config": tp_materialized_model.config})


@pytest.fixture
def sharding_model():
    """A real Qwen3 that HF's ``tp_plan="auto"`` shards (attention + MLP projections).

    Function-scoped: :func:`tp_materialized_model` shards it IN PLACE, so a shared instance would
    reach the next test already sharded — and re-running the materialization over DTensors models no
    load at all.
    """
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
    """``sharding_model`` as a real TP load hands it back: every plan-sharded param a DTensor.

    ``_load_tp_model`` reads the loaded PARAMETERS, not just the plan — its post-load guard refuses
    a plan-sharded param that materialized plain, because a bare rank-local slice is
    indistinguishable downstream from a replica and the gradient sync averages it against its peers'
    DIFFERENT slices. So an unsharded stand-in is not a stub for a TP load; it is exactly the
    materialization regression the guard exists for (pinned in
    :func:`test_tp_load_refuses_a_plainly_materialized_load`).

    The sharding is transformers' OWN ``apply_tensor_parallelism`` — the function ``from_pretrained``
    calls under ``tp_plan="auto"`` with a device mesh — over a fake process group, where
    ``distribute_tensor(..., src_data_rank=None)`` slices locally with no collective. The group stays
    up for the test, as it is during a real load.
    """
    with fake_process_group_mesh(rank=0, world_size=_TP_SIZE) as mesh:
        yield apply_tensor_parallelism(sharding_model, mesh)


@pytest.fixture(scope="module")
def planless_model():
    """A real Qwen3-VL: ``base_model_tp_plan`` is None on both the wrapper and the text tower, so
    ``tp_plan="auto"`` resolves to ``{}`` and transformers only warns."""
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
    """The predicate the guard is built on, over real plans and real parameter names.

    ``model._tp_plan`` is the plan transformers actually applied (base plan, prefixed at
    ``post_init``); ``tp_plan_shards_params`` over ``named_parameters()`` is exactly what the guard
    counts. Qwen3 must come out with sharded parameters and Qwen3-VL with none — if those two
    collapsed to the same answer, the guard could not tell a working TP run from a wasted one.
    """
    sharded = [
        name for name, _ in sharding_model.named_parameters() if tp_plan_shards_params(name, sharding_model._tp_plan)
    ]
    assert sharded, "premise: qwen3 ships a base_model_tp_plan covering q/k/v/o and the MLP"
    assert any("q_proj" in name for name in sharded) and any("down_proj" in name for name in sharded)

    assert not (planless_model._tp_plan or {}), "premise: qwen3-vl ships no base_model_tp_plan"
    assert not [
        name
        for name, _ in planless_model.named_parameters()
        if tp_plan_shards_params(name, planless_model._tp_plan or {})
    ]


def test_tp_load_refuses_an_architecture_whose_plan_shards_nothing(planless_model):
    """Silent otherwise: ``tp_size`` identical replicas, at ``1/tp_size`` throughput, with
    ``data_parallel_size`` already divided by ``tp_size`` — a run that trains and looks fine."""
    with _load_collaborators(planless_model), pytest.raises(ValueError, match="sharded ZERO parameters"):
        _load_tp_model("org/x", _pc(tp_size=_TP_SIZE), AutoModelForImageTextToText, {"config": planless_model.config})


def test_tp_load_accepts_an_architecture_that_does_shard(tp_materialized_model):
    """Anti-vacuity: an architecture whose plan does shard, materialized the way HF materializes it,
    must pass every post-load gate and come back stamped with the mesh."""
    with _load_collaborators(tp_materialized_model):
        model = _load_tp_model(
            "org/x", _pc(tp_size=_TP_SIZE), AutoModelForCausalLM, {"config": tp_materialized_model.config}
        )
    assert model is tp_materialized_model
    assert model._device_mesh is _DP_TP_MESH


def test_tp_load_refuses_a_plainly_materialized_load(sharding_model):
    """The same architecture, loaded WITHOUT transformers placing its shards, must be refused.

    This is the transformers materialization regression the post-load guard was added for: the plan
    covers q/k/v/o and the MLP, so every rank holds a bare slice of them that the trainer sorts into
    the replicated AVG bucket and averages against its peers' disjoint slices — silently, since a
    plain tensor is what a genuine replica looks like. Refused at load, or never noticed at all.
    """
    assert not any(isinstance(param, DTensor) for param in sharding_model.parameters()), (
        "premise: this fixture is the un-materialized load"
    )
    with _load_collaborators(sharding_model), pytest.raises(ValueError, match="materialized as plain"):
        _load_tp_model("org/x", _pc(tp_size=_TP_SIZE), AutoModelForCausalLM, {"config": sharding_model.config})


def test_tp_load_is_inert_at_tp_size_one(planless_model):
    """``tp_size=1`` is not a TP run; refusing there would ban the model outright."""
    with _load_collaborators(planless_model):
        loaded = _load_tp_model("org/x", _pc(), AutoModelForImageTextToText, {"config": planless_model.config})
    assert loaded is planless_model


# ── fp32_non_ep_params: family-declared, registry-resolved ───────────────


class _ModelConfig:
    """Stand-in HF config: a ``model_type``, plus the composite layout where the family's real
    ``model_type`` lives on the text sub-config."""

    def __init__(self, model_type: str, text: _ModelConfig | None = None):
        self.model_type = model_type
        self._text = text

    def get_text_config(self):
        return self._text


@pytest.mark.parametrize("model_type", ["gemma4", "gemma4_text"])
def test_fp32_non_ep_params_is_refused_for_a_family_that_declares_it_unsupported(model_type):
    """Both spellings resolve to ``EPGemma4MoELayer``, which declares the opt-out; the alternative
    is DeepEP's raw C++ assert at the first dispatch, after the whole multi-GPU load."""
    with pytest.raises(ValueError, match="fp32_non_ep_params") as excinfo:
        _validate_fp32_non_ep_params(_pc(ep_size=2, fp32_non_ep_params=True), _ModelConfig(model_type))
    assert model_type in str(excinfo.value), "the refusal must name the model_type that triggered it"


def test_the_text_tower_is_consulted_for_a_composite_config():
    """A multimodal wrapper keeps its own ``model_type``; reading only that would let the refused
    family through under any VLM checkpoint."""
    composite = _ModelConfig("some_multimodal_wrapper", text=_ModelConfig("gemma4_text"))
    with pytest.raises(ValueError, match="gemma4_text"):
        _validate_fp32_non_ep_params(_pc(ep_size=2, fp32_non_ep_params=True), composite)


@pytest.mark.parametrize("model_type", ["qwen3_moe", "qwen3"])
def test_families_that_support_the_upcast_and_dense_models_are_left_alone(model_type):
    """``qwen3_moe`` resolves to an EP layer that declares support; ``qwen3`` resolves to nothing at
    all. A guard that refused either would ban a validated production shape."""
    _validate_fp32_non_ep_params(_pc(ep_size=2, fp32_non_ep_params=True), _ModelConfig(model_type))


@pytest.mark.parametrize("pc_kwargs", [{"ep_size": 1}, {"ep_size": 2, "fp32_non_ep_params": False}])
def test_the_refusal_needs_both_conjuncts(pc_kwargs):
    """No experts distributed (``ep_size=1``) means no DeepEP dispatch to break, and without the
    knob there is nothing to refuse — the guard must fire on the pair, not on either half."""
    kwargs = {"fp32_non_ep_params": True, **pc_kwargs}
    _validate_fp32_non_ep_params(_pc(**kwargs), _ModelConfig("gemma4"))


def test_the_class_declaration_is_what_drives_the_refusal(monkeypatch):
    """Anti-vacuity: flip the attribute on the layer class and the same call must pass. This is the
    difference between a family opting out where its layout knowledge lives and a model_type string
    list in the loader — only the former lets a new family declare its own answer."""
    assert EPMoELayerBase._supports_fp32_non_ep_params is True, "the roster default must be permissive"
    with pytest.raises(ValueError):
        _validate_fp32_non_ep_params(_pc(ep_size=2, fp32_non_ep_params=True), _ModelConfig("gemma4"))

    monkeypatch.setattr(EPGemma4MoELayer, "_supports_fp32_non_ep_params", True)
    _validate_fp32_non_ep_params(_pc(ep_size=2, fp32_non_ep_params=True), _ModelConfig("gemma4"))


def test_the_family_gate_is_wired_into_the_loader_right_after_the_config_read():
    """The gate is only worth anything if ``load_distributed_model`` calls it — and calls it while
    the only thing resident is ``config.json``. Everything before that point is real; the download
    and the config read are the sole stand-ins, so a dropped call site fails here."""
    config = _ModelConfig("gemma4")
    with (
        patch.dict(os.environ, _env({}), clear=True),
        patch.object(model_loading, "_ensure_model_downloaded", lambda *args, **kwargs: None),
        patch.object(model_loading, "AutoConfig", SimpleNamespace(from_pretrained=lambda *a, **k: config)),
        pytest.raises(ValueError, match="fp32_non_ep_params"),
    ):
        load_distributed_model("org/x", parallelism_config=_pc(ep_size=2, fp32_non_ep_params=True))


def test_should_accelerate_manage_ddp_semantics():
    trainer = SimpleNamespace(_no_custom_parallelism=lambda: True)
    manage_ddp = DistributedTrainerMixin._should_accelerate_manage_ddp
    with patch.dict(os.environ, _env(_MULTI_GPU_ENV), clear=True):
        assert manage_ddp(trainer) is True
    with patch.dict(os.environ, _env(_FSDP_ENV), clear=True):
        assert manage_ddp(trainer) is False  # FSDP launch is not DDP
    with patch.dict(os.environ, _env({}), clear=True):
        assert manage_ddp(trainer) is False  # torchrun


# ── the run dtype crosses into transformers under transformers' own key ──


def test_the_run_dtype_reaches_transformers_under_its_own_key(tp_materialized_model):
    """``common_kwargs`` carries the run's dtype under transformers' own ``dtype`` key, which is what
    lets every loader forward the dict unchanged. ``from_pretrained`` still accepts ``torch_dtype``
    as a deprecated alias it folds into ``dtype``, so a loader re-spelling the key on the way out
    would look correct while depending on that alias."""
    common = {"config": tp_materialized_model.config, "dtype": torch.bfloat16, "trust_remote_code": True}
    with (
        _load_collaborators(tp_materialized_model),
        patch.object(model_loading, "from_pretrained_verified", return_value=tp_materialized_model) as loader,
    ):
        _load_tp_model("org/x", _pc(tp_size=_TP_SIZE), AutoModelForCausalLM, common)

    passed = loader.call_args.kwargs
    assert passed["dtype"] is torch.bfloat16
    assert "torch_dtype" not in passed
    assert common["dtype"] is torch.bfloat16, "the caller's dict is shared with the other loaders"


def test_the_cp_loader_cannot_install_an_unvalidated_attention_implementation():
    """Every ``attn_implementation`` the CP loader forwards must have been through
    ``validate_attn_implementation``.

    ``config`` is REQUIRED and the install reads only the validated value — both pinned here. An
    optional config with a second, unconditional install branch hands ``from_pretrained`` the RAW
    request whenever the caller omits the config; GPT-OSS with live sinks is exactly the case the
    validator rejects, so the run would train on a silently different attention kernel.
    """
    assert inspect.signature(load_model_for_cp).parameters["config"].default is inspect.Parameter.empty, (
        "config must stay REQUIRED: an optional one is what let an unvalidated impl through"
    )

    seen = {}

    def fake_from_pretrained_verified(model_class, path, **kwargs):
        seen.update(kwargs)
        return torch.nn.Linear(2, 2)

    config = Qwen3Config(num_hidden_layers=1, hidden_size=8, intermediate_size=8, num_attention_heads=2)
    with (
        patch(f"{_CP_LOADING}.from_pretrained_verified", fake_from_pretrained_verified),
        patch(f"{_CP_LOADING}.move_model_to_local_device", lambda model: model),
        patch(f"{_CP_LOADING}.finalize_loaded_model", lambda model: None),
        patch(f"{_CP_LOADING}.patch_model_for_cp", lambda model, cp_config: model),
        patch("src.models.patches.attention.validate_attn_implementation", return_value="sdpa") as validator,
    ):
        load_model_for_cp(
            "org/repo",
            SimpleNamespace(cp_size=2),
            config,
            model_class=AutoModelForCausalLM,
            attn_implementation="flash_attention_2",
        )

    validator.assert_called_once_with(config, "flash_attention_2")
    assert seen["attn_implementation"] == "sdpa", (
        f"the loader forwarded {seen.get('attn_implementation')!r}, not the validator's verdict"
    )


def test_every_loader_names_the_run_dtype_dtype():
    """One spelling from the entry point down to ``from_pretrained``, so nothing translates in
    between. A signature drifting back to ``torch_dtype`` is a TypeError only the parallelism mode
    that reaches that loader hits — every other mode stays green.
    """
    for loader in (
        load_distributed_model,
        load_ep_model,
        load_ep_model_lazy,
        load_pp_stage_model,
        load_model_for_cp,
        load_model_for_ep_cp,
        load_frozen_auxiliary_model,
        instantiate_on_meta,
        resolve_run_dtype,
        resolve_attn_implementation,
    ):
        params = inspect.signature(loader).parameters
        assert "dtype" in params, f"{loader.__name__} does not take the run's dtype as `dtype`"
        assert "torch_dtype" not in params, f"{loader.__name__} still spells the run's dtype `torch_dtype`"


def test_no_transformers_call_in_the_loader_passes_the_deprecated_dtype_alias():
    """Covers the two loaders the CPU suite cannot drive (both end in ``.to(cuda)``). A
    ``from_pretrained`` handed ``torch_dtype=`` still loads: transformers warns once and folds the
    alias into ``dtype``, so the only visible symptom is a log line — until the alias is removed."""
    boundary = ("from_pretrained_verified", "from_pretrained", "from_config")
    offenders = [
        f"{callee}() at line {node.lineno}"
        for node in ast.walk(ast.parse(inspect.getsource(model_loading)))
        if isinstance(node, ast.Call)
        for callee in [node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")]
        if callee in boundary
        for keyword in node.keywords
        if keyword.arg == "torch_dtype"
    ]
    assert not offenders, f"pass the run's dtype as `dtype=`: {offenders}"


def test_reset_sinks_defaults_agree_across_loaders():
    entry = inspect.signature(load_model_for_training).parameters["reset_sinks"].default
    direct = inspect.signature(load_distributed_model).parameters["reset_sinks"].default
    # The frozen reference/teacher loader is scored against the policy loaders, so a different
    # default here pairs a reset policy with a live-sink reference.
    frozen = inspect.signature(load_frozen_auxiliary_model).parameters["reset_sinks"].default
    assert entry is True and direct is True and frozen is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
