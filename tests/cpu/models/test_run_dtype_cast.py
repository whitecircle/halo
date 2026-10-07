#!/usr/bin/env python
"""``cast_parameters_to_run_dtype``: the loaded-parameter cast every training and scoring loader applies.

transformers keeps a family's ``_keep_in_fp32_modules_strict`` parameters in fp32 (DeepSeek-V4 norms
and hyper-connections, GLM-5 Next KDA state, Inkling short convolutions). The EP loaders cast them to
the run dtype; a loader that does not leaves a mixed-dtype model that FSDP2 refuses to shard and whose
fp32 DeepSeek-V4 norms feed fp32 activations into bf16 projections. These tests pin (1) the helper's
contract, fp8, quantized storage and fp32-master handling included, (2) that the unsharded loaders a
CPU run reaches hand back a uniform run-dtype model for every pinned family, (3) that under fp32
masters the lazy loaders preserve every configured non-EP master (including stored pins), while
without the flag they load all bf16, and (4) that every model build in
``src/`` and ``scripts/training/`` either casts and finalizes or is pinned as no training load, so a
new loader that forgets either fails here rather than on its first GPU step.

Run: python tests/cpu/models/test_run_dtype_cast.py  (or pytest)
"""

import functools
import types
from unittest import mock

import pytest
import torch
import torch.nn as nn
from accelerate import PartialState
from bitsandbytes.nn import Linear4bit, Params4bit
from safetensors.torch import save_file
from torch.distributed.tensor import Shard, distribute_tensor
from transformers import AutoConfig
from trl import ModelConfig

import scripts.training.embedding as embedding_script
import src.distributed.expert_parallel.lazy_loader as ep_lazy_loader
import src.distributed.pipeline_parallel.lazy_loader as pp_lazy_loader
from src.configs.embedding_config import EmbeddingConfig
from src.distributed.expert_parallel.config import EPConfig
from src.distributed.expert_parallel.fp32_masters import fp32_master_param_keys
from src.distributed.expert_parallel.lazy_loader import load_ep_model_lazy
from src.distributed.expert_parallel.loading import cast_loaded_parameters
from src.distributed.expert_parallel.patching import ep_claimed_blocks
from src.distributed.loading.frozen_models import load_frozen_auxiliary_model
from src.distributed.loading.model_loading import load_model_from_pretrained
from src.distributed.pipeline_parallel.lazy_loader import PPWeightPlanner, load_pp_stage_model
from src.distributed.pipeline_parallel.stage import PP_STAGE_PARTITION_ATTR, resolve_layer_root
from src.models.loading.dtype import cast_parameters_to_run_dtype
from src.models.loading.lazy_safetensors.meta_shell import instantiate_on_meta
from src.models.loading.lazy_safetensors.weights import SafetensorsWeightLoader, WeightAction, WeightPlan
from src.models.patches.buffer_fixes import finalize_loaded_model
from src.models.structure import fp32_pinned_param_names
from tests.common.distributed import fake_process_group_mesh
from tests.common.models import QWEN3_0_6B
from tests.common.pinned_params import pins_off_stored, stored_fp32_pins
from tests.common.source_sweep import builds_a_model, functions_calling
from tests.common.tiny_models import PINNED_FP32_FAMILIES, TINY_MOE_FAMILIES, build_tiny_family_checkpoint
from tests.common.tokenizers import load_cached_tokenizer
from tests.common.utils import params_off_dtype, safetensors_state_dict

# The loaders log through accelerate's logger, which requires an initialized state.
PartialState()

# The run-dtype cast, directly or through its EP-aware wrapper.
CASTS = frozenset({cast_parameters_to_run_dtype.__name__, cast_loaded_parameters.__name__})
FINALIZE = finalize_loaded_model.__name__

# Where a training or scoring load lives; every model build there inherits the fp32 pins.
# ``src/models/loading/`` defines the entry points and is left out: conversion tools load through it
# too and must keep the pins, so the cast belongs to each training loader rather than to the shared core.
SWEPT_ROOTS = ("src", "scripts/training")
LOAD_CORE = "src/models/loading/"

# Every function the sweep below finds. Pinned exactly so a renamed or new loader is a deliberate
# edit rather than a silently shrunk sweep.
EAGER_LOADERS = frozenset(
    {
        ("scripts/training/embedding.py", "build_sentence_transformer"),
        ("src/distributed/expert_parallel/loading.py", "load_through_cpu"),
        ("src/distributed/loading/frozen_models.py", "load_frozen_auxiliary_model"),
        ("src/distributed/loading/model_loading.py", "_from_pretrained_on_local_gpu"),
        ("src/distributed/loading/model_loading.py", "load_model_from_pretrained"),
    }
)

# The dense TP load materializes straight into DTensors, which the cast cannot re-dtype (it skips one
# already at the run dtype and raises on any other). No dense family pins a parameter, and a mixed load
# there would fail FSDP2's wrap loudly.
UNCAST_LOADERS = frozenset({("src/distributed/loading/model_loading.py", "_load_tp_model")})

# Model builds that are no training or scoring load: the PEFT merge folds an adapter into the base a
# conversion tool loaded (conversions keep the pins), and the head-transform probe builds an fp32 meta
# shell that holds no weights.
NON_TRAINING_BUILDS = frozenset(
    {
        ("src/checkpoint/adapters.py", "merge_adapter_into_base"),
        ("src/models/head_transform.py", "_probe_logits"),
    }
)

# Functions whose ``from_config`` builds the toolkit's own config-driven objects (the example-generation
# callback, the reward terms), not a model.
NON_MODEL_BUILDERS = frozenset(
    {
        ("scripts/training/offline_grpo.py", "main"),
        ("scripts/training/sft.py", "main"),
        ("src/rewards/terms.py", "from_config"),
        ("src/rewards/terms.py", "parse_reward_terms"),
        ("src/training/script_runner.py", "prepare_script_preference_data"),
    }
)

# The fp32-masters loader cases. A lazy loader plans against ``from_pretrained``'s meta shell, or the
# config-built one for a ``config_shell`` case; the PP cases load the single stage of a one-stage split.
FP32_MASTER_LOADERS = ("path_string", "ep_lazy", "ep_lazy_config_shell", "pp_stage", "pp_stage_config_shell")


class _Mixed(nn.Module):
    """Every parameter and buffer kind the cast must tell apart."""

    def __init__(self):
        super().__init__()
        self.pinned = nn.Linear(4, 4).float()
        self.run = nn.Linear(4, 4).to(torch.bfloat16)
        self.fp16 = nn.Linear(4, 4).to(torch.float16)
        self.head = nn.Linear(4, 4, bias=False).to(torch.bfloat16)
        self.head.weight = self.run.weight
        self.index = nn.Parameter(torch.arange(4), requires_grad=False)
        self.register_buffer("bias_state", torch.zeros(4, dtype=torch.float32))


def test_the_cast_moves_floating_parameters_only():
    model = _Mixed()
    pinned_weight = model.pinned.weight

    cast_parameters_to_run_dtype(model, torch.bfloat16)

    assert params_off_dtype(model, torch.bfloat16) == []
    # In place: an optimizer or hook holding the Parameter keeps holding the model's parameter.
    assert model.pinned.weight is pinned_weight
    assert model.head.weight is model.run.weight
    assert model.index.dtype == torch.int64
    assert model.bias_state.dtype == torch.float32


def test_the_cast_keeps_values_to_the_target_precision():
    model = _Mixed()
    expected = model.pinned.weight.detach().to(torch.bfloat16)

    cast_parameters_to_run_dtype(model, torch.bfloat16)

    assert torch.equal(model.pinned.weight.detach(), expected)


def test_fp32_masters_keep_fp32_parameters_as_stored():
    """Under ``fp32_non_ep_params`` the run upcasts to fp32 anyway; a bf16 round trip first would lose
    the stored precision. Other off-dtype parameters still take the run dtype."""
    model = _Mixed()
    stored = model.pinned.weight.detach().clone()

    cast_parameters_to_run_dtype(model, torch.bfloat16, keep_fp32=True)

    assert model.pinned.weight.dtype == torch.float32
    assert torch.equal(model.pinned.weight.detach(), stored)
    assert model.fp16.weight.dtype == torch.bfloat16


def test_an_fp8_parameter_refuses_the_cast():
    """A plain cast of an fp8 weight drops its block scales, and the fp8 linear then runs unscaled."""
    model = _Mixed()
    model.pinned.weight = nn.Parameter(model.pinned.weight.detach().to(torch.float8_e4m3fn), requires_grad=False)

    with pytest.raises(ValueError, match=r"'pinned\.weight'.*Dequantize it to bf16"):
        cast_parameters_to_run_dtype(model, torch.bfloat16)


@pytest.mark.parametrize("run_dtype", [torch.float16, torch.float32])
def test_a_quantized_parameter_keeps_its_float_storage(run_dtype):
    """A QLoRA base stored as ``bnb_4bit_quant_storage: bfloat16`` holds packed 4-bit codes in a bf16
    tensor; casting that tensor to another run dtype would rewrite the codes as if they were values.
    Compared as bytes: a code pair can read as a bf16 NaN."""
    model = _Mixed()
    model.quantized = Linear4bit(64, 64, bias=False, quant_storage=torch.bfloat16, quant_type="nf4")
    model.quantized.weight = Params4bit(
        torch.randn(64, 64, dtype=torch.bfloat16), requires_grad=False, quant_storage=torch.bfloat16, quant_type="nf4"
    )
    model.quantized.to("cpu")
    weight = model.quantized.weight
    packed = weight.data.view(torch.uint8).clone()

    cast_parameters_to_run_dtype(model, run_dtype)

    assert weight.bnb_quantized and weight.dtype == torch.bfloat16
    assert model.quantized.weight is weight and torch.equal(weight.data.view(torch.uint8), packed)
    assert "quantized.weight" not in fp32_master_param_keys(model, keep_non_ep=True)
    assert model.pinned.weight.dtype == run_dtype


class _TaggedParameter(nn.Parameter):
    """A ``Parameter`` subclass that is not bnb's 4-bit storage."""


def test_any_other_parameter_subclass_refuses_the_cast():
    """Only bnb's 4-bit storage is known to hold packed codes; another subclass off the run dtype raises
    rather than being cast or silently left mixed."""
    model = _Mixed()
    model.pinned.weight = _TaggedParameter(model.pinned.weight.detach())

    with pytest.raises(TypeError, match=r"'pinned\.weight', a _TaggedParameter"):
        cast_parameters_to_run_dtype(model, torch.bfloat16)


def test_an_unresolved_dtype_name_is_refused():
    """A name reaching the cast unresolved would otherwise leave the pins mixed in silently."""
    with pytest.raises(TypeError, match="'bfloat16'"):
        cast_parameters_to_run_dtype(_Mixed(), "bfloat16")


def test_a_sharded_parameter_refuses_the_cast():
    """A DTensor's ``.data`` rebind keeps its local shard in the old dtype, so a cast of one off the run
    dtype must raise."""
    with fake_process_group_mesh(rank=0, world_size=1) as mesh:
        model = nn.Linear(4, 4)
        model.weight = nn.Parameter(distribute_tensor(model.weight.detach(), mesh, [Shard(0)], src_data_rank=None))

        with pytest.raises(TypeError, match="'weight'"):
            cast_parameters_to_run_dtype(model, torch.bfloat16)


@pytest.mark.parametrize("dtype", ["auto", None])
def test_a_request_that_is_not_a_dtype_leaves_the_model_as_loaded(dtype):
    model = _Mixed()

    cast_parameters_to_run_dtype(model, dtype)

    assert model.pinned.weight.dtype == torch.float32
    assert model.run.weight.dtype == torch.bfloat16


def _tiny_checkpoints(tmp_path_factory, name: str, families, *, fp32_pins: bool = False) -> dict[str, str]:
    """``families``' tiny checkpoints under a fresh ``name`` directory, keyed by family."""
    root = tmp_path_factory.mktemp(name)
    checkpoints = {family: str(root / family) for family in families}
    for family, path in checkpoints.items():
        build_tiny_family_checkpoint(TINY_MOE_FAMILIES[family], path, fp32_pins=fp32_pins)
    return checkpoints


@pytest.fixture(scope="module")
def roster_checkpoints(tmp_path_factory) -> dict[str, str]:
    return _tiny_checkpoints(tmp_path_factory, "roster", TINY_MOE_FAMILIES)


@pytest.fixture(scope="module")
def pinned_checkpoints(roster_checkpoints) -> dict[str, str]:
    return {family: roster_checkpoints[family] for family in PINNED_FP32_FAMILIES}


@pytest.mark.parametrize("family", PINNED_FP32_FAMILIES)
def test_a_bf16_from_pretrained_keeps_the_family_pins(pinned_checkpoints, family):
    """The premise the loader tests rest on: without the cast, these checkpoints load mixed."""
    model = TINY_MOE_FAMILIES[family].load_class.from_pretrained(pinned_checkpoints[family], dtype=torch.bfloat16)

    assert params_off_dtype(model, torch.bfloat16)


@pytest.mark.parametrize("family", PINNED_FP32_FAMILIES)
def test_the_frozen_loader_scores_in_the_run_dtype(pinned_checkpoints, family):
    model = load_frozen_auxiliary_model(pinned_checkpoints[family], dtype=torch.bfloat16, device_map="cpu")

    assert params_off_dtype(model, torch.bfloat16) == []


@pytest.mark.parametrize("model_init_kwargs", [{"dtype": "bfloat16"}, {"dtype": "auto"}, {}], ids=repr)
def test_the_path_string_loader_trains_in_one_dtype(pinned_checkpoints, model_init_kwargs):
    """An explicit dtype is the target; an unset or "auto" one loads at the checkpoint's own (bf16) dtype
    and the pins are unified to it."""
    args = types.SimpleNamespace(model_init_kwargs=dict(model_init_kwargs), gradient_checkpointing=False)

    model, _ = load_model_from_pretrained(pinned_checkpoints["deepseek_v4"], args, keep_fp32=False)

    assert params_off_dtype(model, torch.bfloat16) == []


@pytest.mark.parametrize("ep_wrapped", [True, False])
def test_fp32_masters_keep_stored_values_where_the_upcast_reaches(pinned_checkpoints, ep_wrapped):
    """``fp32_non_ep_params`` upcasts non-EP parameters alone, so a parameter inside a block that gets EP
    wrappers trains at the run dtype even when it loaded fp32; without wrappers it is upcast too, and
    keeps its stored value like the pins."""
    model = TINY_MOE_FAMILIES["inkling_text"].load_class.from_pretrained(
        pinned_checkpoints["inkling_text"], dtype=torch.bfloat16
    )
    blocks = ep_claimed_blocks(model)
    in_block = next(param for _path, block in blocks for param in block.parameters())
    in_block.data = in_block.data.float()
    stored = {name: param.detach().clone() for name, param in model.named_parameters() if param.dtype == torch.float32}

    cast_loaded_parameters(model, torch.bfloat16, keep_fp32=True, ep_wrapped=ep_wrapped)

    assert blocks and in_block.dtype == (torch.bfloat16 if ep_wrapped else torch.float32)
    kept = {name: param for name, param in model.named_parameters() if name in stored and param is not in_block}
    assert kept and all(torch.equal(param.detach(), stored[name]) for name, param in kept.items())


@pytest.mark.parametrize("family", sorted(TINY_MOE_FAMILIES))
def test_exactly_the_pinned_families_load_fp32_parameters(roster_checkpoints, family):
    """Holds ``PINNED_FP32_FAMILIES`` to the roster: a family whose class starts pinning parameters, or
    stops, fails here. None pins one inside a MoE block, the premise of the EP loaders' block cast."""
    tiny = TINY_MOE_FAMILIES[family]
    model = tiny.load_class.from_pretrained(
        roster_checkpoints[family], dtype=torch.bfloat16, trust_remote_code=tiny.trust_remote_code
    )
    fp32 = set(params_off_dtype(model, torch.bfloat16))
    in_blocks = {f"{path}.{name}" for path, block in ep_claimed_blocks(model) for name, _ in block.named_parameters()}

    assert bool(fp32) == (family in PINNED_FP32_FAMILIES)
    assert not fp32 & in_blocks


@pytest.mark.parametrize("family", sorted(TINY_MOE_FAMILIES))
def test_a_config_built_shell_selects_the_same_masters_as_pretrained(roster_checkpoints, family):
    """Master ownership depends on the family tree and configured flags, not the meta shell dtype."""
    tiny = TINY_MOE_FAMILIES[family]
    path = roster_checkpoints[family]
    config = AutoConfig.from_pretrained(path, trust_remote_code=tiny.trust_remote_code)
    shells = [
        instantiate_on_meta(
            path,
            tiny.load_class,
            config,
            dtype=torch.bfloat16,
            trust_remote_code=tiny.trust_remote_code,
            config_only=config_only,
        )
        for config_only in (False, True)
    ]

    ep_config = EPConfig(ep_size=1, world_size=1, gpus_per_node=1)
    kept = [fp32_master_param_keys(shell, ep_config, keep_non_ep=True) for shell in shells]
    assert kept[0] == kept[1]
    assert kept[0], "every family has non-EP parameters for configured masters"


@pytest.fixture(scope="module")
def stored_fp32_checkpoints(tmp_path_factory) -> dict[str, str]:
    """The pinned families saved as a release stores them: the pins at full fp32, the rest bf16."""
    return _tiny_checkpoints(tmp_path_factory, "stored_fp32", PINNED_FP32_FAMILIES, fp32_pins=True)


def _config_built_shell(*args, **kwargs):
    return instantiate_on_meta(*args, **{**kwargs, "config_only": True})


def _load_with(
    loader: str, family: str, path: str, keep_fp32: bool, monkeypatch, *, pp_rank: int = 0, pp_size: int = 1
) -> nn.Module:
    """``family`` loaded at bf16 through ``loader``, with ``fp32_non_ep_params`` as ``keep_fp32``.

    The lazy loaders run with EP patching stubbed (it needs DeepEP and process groups) under an ep1
    config, the PP loader as stage ``pp_rank`` of ``pp_size``.
    """
    tiny = TINY_MOE_FAMILIES[family]
    if loader == "path_string":
        args = types.SimpleNamespace(model_init_kwargs={"dtype": torch.bfloat16}, gradient_checkpointing=False)
        model, _ = load_model_from_pretrained(path, args, tiny.load_class, keep_fp32=keep_fp32)
        return model
    for module in (ep_lazy_loader, pp_lazy_loader):
        monkeypatch.setattr(module, "patch_moe_model_for_ep", lambda model, *args, **kwargs: model)
        monkeypatch.setattr(module, "create_ep_buffers", lambda *args, **kwargs: None)
        if loader.endswith("config_shell"):
            monkeypatch.setattr(module, "instantiate_on_meta", _config_built_shell)
    ep_config = EPConfig(ep_size=1, world_size=1, gpus_per_node=1)
    config = AutoConfig.from_pretrained(path)
    common = {
        "dtype": torch.bfloat16,
        "trust_remote_code": False,
        "model_class": tiny.load_class,
        "keep_fp32_params": keep_fp32,
    }
    if loader.startswith("ep_lazy"):
        return load_ep_model_lazy(path, ep_config, config, **common)
    return load_pp_stage_model(path, pp_rank, pp_size, config=config, ep_config=ep_config, **common)


def _outside_moe_blocks(model: nn.Module, *, aliases: bool = False) -> frozenset[str]:
    """Every parameter name outside the family's MoE blocks, read off the blocks' module paths: the
    masters ``fp32_non_ep_params`` configures when no router or expert master is (:func:`_load_with`'s
    ep1 config). ``aliases`` lists a tied parameter under each of its names."""
    blocks = tuple(f"{path}." for path, _block in ep_claimed_blocks(model))
    assert blocks, "the premise is a MoE family whose blocks EP claims"
    return frozenset(
        name for name, _param in model.named_parameters(remove_duplicate=not aliases) if not name.startswith(blocks)
    )


def _assert_keeps_exactly(
    model: nn.Module, pinned: dict[str, str], stored: dict, keep_fp32: bool, *, masters: frozenset[str] | None = None
) -> None:
    """``pinned`` maps each pin's name in ``model`` to its checkpoint name."""
    fp32 = {name for name, param in model.named_parameters() if param.dtype == torch.float32}
    assert set(params_off_dtype(model, torch.bfloat16)) == fp32
    expected_fp32 = set(pinned) if masters is None else set(masters) & set(dict(model.named_parameters()))
    assert fp32 == (expected_fp32 if keep_fp32 else set())
    off = pins_off_stored(model, stored, pinned) if keep_fp32 else []
    assert not off, f"pins off their stored fp32 value: {off}"


@pytest.mark.parametrize("loader", FP32_MASTER_LOADERS)
def test_every_loader_keeps_a_pinned_buffer_fp32(roster_checkpoints, loader, monkeypatch):
    """GLM-5 Next pins its router's ``e_score_correction_bias`` buffer. The tiny checkpoint stores it
    bf16 and ``from_pretrained`` loads it fp32, so every training loader must too: the lazy ones read a
    buffer at its stored dtype unless it is pinned."""
    stored = {
        name: tensor.dtype
        for name, tensor in safetensors_state_dict(roster_checkpoints["glm5_next"]).items()
        if name.endswith("e_score_correction_bias")
    }
    assert stored and set(stored.values()) == {torch.bfloat16}, f"the premise is a bf16-stored bias: {stored}"
    model = _load_with(loader, "glm5_next", roster_checkpoints["glm5_next"], False, monkeypatch)

    dtypes = {name: b.dtype for name, b in model.named_buffers() if name.endswith("e_score_correction_bias")}
    assert dtypes and set(dtypes.values()) == {torch.float32}, dtypes


@pytest.mark.parametrize("keep_fp32", [False, True], ids=["bf16", "fp32_masters"])
@pytest.mark.parametrize("loader", FP32_MASTER_LOADERS)
@pytest.mark.parametrize("family", PINNED_FP32_FAMILIES)
def test_fp32_masters_keep_exactly_the_stored_pins(stored_fp32_checkpoints, family, loader, keep_fp32, monkeypatch):
    """Stored pins remain exact within the configured master set; flags, not shell dtype, select it."""
    path = stored_fp32_checkpoints[family]
    pinned, stored = stored_fp32_pins(family, path)

    model = _load_with(loader, family, path, keep_fp32, monkeypatch)

    masters = None if loader == "path_string" else _outside_moe_blocks(model)
    _assert_keeps_exactly(model, {name: name for name in pinned}, stored, keep_fp32, masters=masters)


@pytest.mark.parametrize("keep_fp32", [False, True], ids=["bf16", "fp32_masters"])
def test_a_later_pp_stage_keeps_the_pins_of_its_own_layers(stored_fp32_checkpoints, keep_fp32, monkeypatch):
    """The second of two stages re-bases its layers to 0, so its masters must be taken in the stage's
    numbering, after the slice. DeepSeek-V4's layer types pin different parameters (only a compressed
    layer pins its compressor's norm), so a keep set in the checkpoint's numbering names other layers'
    parameters (the premise, checked below)."""
    path = stored_fp32_checkpoints["deepseek_v4"]
    pinned, stored = stored_fp32_pins("deepseek_v4", path)

    model = _load_with("pp_stage", "deepseek_v4", path, keep_fp32, monkeypatch, pp_rank=1, pp_size=2)

    lo, hi = getattr(model, PP_STAGE_PARTITION_ATTR)[1]
    planner = PPWeightPlanner(lo, hi, resolve_layer_root(model))
    stage_pins = {planner.stage_key(name): name for name in pinned if planner.owns(name)}
    stage_names = {name for name, _ in model.named_parameters()}
    assert lo > 0 and set(stage_pins) != pinned & stage_names
    _assert_keeps_exactly(model, stage_pins, stored, keep_fp32, masters=_outside_moe_blocks(model))


@pytest.mark.parametrize("ep_wrapped", [True, False])
def test_the_lazy_keep_set_leaves_out_ep_wrapped_blocks(pinned_checkpoints, ep_wrapped):
    """The lazy loaders' twin of :func:`cast_loaded_parameters`: a parameter inside a block EP wraps
    trains at the run dtype, so only an unwrapped load keeps it fp32."""
    tiny = TINY_MOE_FAMILIES["inkling_text"]
    checkpoint = pinned_checkpoints["inkling_text"]
    shell = instantiate_on_meta(
        checkpoint,
        tiny.load_class,
        AutoConfig.from_pretrained(checkpoint),
        dtype=torch.bfloat16,
        trust_remote_code=False,
    )
    pins = set(params_off_dtype(shell, torch.bfloat16))
    in_block = next(
        f"{path}.{name}" for path, block in ep_claimed_blocks(shell) for name, _ in block.named_parameters()
    )
    shell.get_parameter(in_block).data = shell.get_parameter(in_block).data.float()

    ep_config = EPConfig(ep_size=1, world_size=1, gpus_per_node=1) if ep_wrapped else None
    kept = fp32_master_param_keys(shell, ep_config, keep_non_ep=True)
    every_name = frozenset(name for name, _param in shell.named_parameters(remove_duplicate=False))
    expected = _outside_moe_blocks(shell, aliases=True) if ep_wrapped else every_name
    assert pins and kept == expected
    assert (in_block in kept) is (not ep_wrapped)


@pytest.mark.parametrize("keep_fp32", [frozenset(), frozenset({"weight"})], ids=["cast", "kept"])
def test_the_weight_loader_keeps_only_the_named_keys_fp32(tmp_path, keep_fp32):
    stored = {"weight": torch.randn(8, 8), "bias": torch.randn(8)}
    save_file(stored, str(tmp_path / "model.safetensors"))
    plans = [WeightPlan(WeightAction.REPLICATE, "model.safetensors", key, key) for key in stored]
    with torch.device("meta"):
        model = nn.Linear(8, 8)
    loader = SafetensorsWeightLoader(str(tmp_path), ["model.safetensors"], device="cpu")

    loader.load_into_model(model, plans, dtype=torch.bfloat16, keep_fp32=keep_fp32)

    for key, value in stored.items():
        param = model.get_parameter(key)
        expected = value if key in keep_fp32 else value.bfloat16()
        assert param.dtype == expected.dtype and torch.equal(param.detach(), expected)


def test_the_sentence_transformer_backbone_is_cast_and_finalized(tmp_path):
    """The default embedding path loads through SentenceTransformer's own ``from_pretrained``."""
    base = tmp_path / "base"
    build_tiny_family_checkpoint(TINY_MOE_FAMILIES["deepseek_v4"], str(base), load_cached_tokenizer(QWEN3_0_6B))
    runtime = types.SimpleNamespace(
        parallelism_config=types.SimpleNamespace(
            is_ep_mode=False, is_tp_mode=False, fp32_non_ep_params=False, max_concurrent_loading=None
        ),
        model_source=str(base),
    )
    embedding_config = EmbeddingConfig(
        output_dir=str(tmp_path / "run"),
        bf16=True,
        use_cpu=True,
        pooling_mode="mean",
        normalize_embeddings=False,
        max_length=32,
    )

    with mock.patch.object(embedding_script, FINALIZE, wraps=finalize_loaded_model) as finalize:
        st_model = embedding_script.build_sentence_transformer(
            runtime,
            embedding_config,
            ModelConfig(model_name_or_path=str(base)),
            types.SimpleNamespace(reset_sinks=True, train_sinks=False),
        )

    backbone = st_model[0].auto_model
    assert fp32_pinned_param_names(backbone)
    assert params_off_dtype(backbone, torch.bfloat16) == []
    finalize.assert_called_once_with(backbone)


@functools.cache
def _eager_loaders() -> dict[tuple[str, str], frozenset[str]]:
    """(file, function) -> the names it calls, for every function under :data:`SWEPT_ROOTS` that builds a
    model."""
    return {
        loader: calls
        for loader, calls in functions_calling(SWEPT_ROOTS, builds_a_model).items()
        if not loader[0].startswith(LOAD_CORE)
    }


def test_the_eager_load_surface_is_pinned():
    pinned = EAGER_LOADERS | UNCAST_LOADERS | NON_TRAINING_BUILDS | NON_MODEL_BUILDERS
    discovered = set(_eager_loaders())
    assert discovered == pinned, f"the model-build surface changed: {sorted(discovered ^ pinned)}; classify each"


@pytest.mark.parametrize("loader", sorted(EAGER_LOADERS), ids=lambda loader: loader[1])
def test_every_eager_loader_casts_and_finalizes(loader):
    calls = _eager_loaders()[loader]
    assert calls & CASTS, f"{loader} builds a model without calling any of {sorted(CASTS)}"
    assert FINALIZE in calls, f"{loader} builds a model without calling {FINALIZE}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
