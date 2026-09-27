#!/usr/bin/env python
"""``cast_parameters_to_run_dtype``: the loaded-parameter cast every training and scoring loader applies.

transformers keeps a family's ``_keep_in_fp32_modules_strict`` parameters in fp32 (DeepSeek-V4 norms
and hyper-connections, GLM-5 Next KDA state, Inkling short convolutions). The EP loaders cast them to
the run dtype; a loader that does not leaves a mixed-dtype model that FSDP2 refuses to shard and whose
fp32 DeepSeek-V4 norms feed fp32 activations into bf16 projections. These tests pin (1) the helper's
contract, fp8 and fp32-master handling included, (2) that the unsharded loaders a CPU run reaches hand
back a uniform run-dtype model for every pinned family, and (3) that every eager load in ``src/`` and
``scripts/training/`` casts and finalizes, so a new loader that forgets either fails here rather than
on its first GPU step.

Run: python tests/cpu/models/test_run_dtype_cast.py  (or pytest)
"""

import ast
import functools
import types
from unittest import mock

import pytest
import torch
import torch.nn as nn
from accelerate import PartialState
from torch.distributed.tensor import Shard, distribute_tensor
from trl import ModelConfig

import scripts.training.embedding as embedding_script
from src.configs.embedding_config import EmbeddingConfig
from src.distributed.expert_parallel.loading import cast_loaded_parameters
from src.distributed.expert_parallel.patching import ep_claimed_blocks
from src.distributed.loading.frozen_models import load_frozen_auxiliary_model
from src.distributed.loading.model_loading import load_model_from_pretrained
from src.models.loading.dtype import cast_parameters_to_run_dtype
from src.models.patches.buffer_fixes import finalize_loaded_model
from src.models.structure import fp32_pinned_param_names
from tests.common.distributed import fake_process_group_mesh
from tests.common.models import QWEN3_0_6B
from tests.common.tiny_models import PINNED_FP32_FAMILIES, TINY_MOE_FAMILIES, build_tiny_family_checkpoint
from tests.common.tokenizers import load_cached_tokenizer
from tests.common.utils import REPO_ROOT, params_off_dtype

# The loaders log through accelerate's logger, which requires an initialized state.
PartialState()

# The run-dtype cast, directly or through its EP-aware wrapper.
CASTS = frozenset({"cast_parameters_to_run_dtype", "cast_loaded_parameters"})
FINALIZE = "finalize_loaded_model"

# The calls that materialize weights through ``from_pretrained``, and so inherit the fp32 pins: the two
# toolkit entry points, and a SentenceTransformer built from a checkpoint path.
EAGER_LOAD_CALLS = frozenset({"from_pretrained_verified", "auto_load_model", "SentenceTransformer"})

# Where a training or scoring load lives. ``src/models/loading/`` defines the entry points and is left
# out: conversion tools load through it too and must keep the pins, so the cast belongs to each
# training loader rather than to the shared core.
SWEPT_ROOTS = ("src", "scripts/training")
LOAD_CORE = "src/models/loading/"

# Every function the sweep below finds. Pinned exactly so a renamed or new loader is a deliberate
# edit rather than a silently shrunk sweep.
EAGER_LOADERS = frozenset(
    {
        ("scripts/training/embedding.py", "build_sentence_transformer"),
        ("src/distributed/context_parallel/loading.py", "load_model_for_cp"),
        ("src/distributed/expert_parallel/loading.py", "_load_ep_model_huggingface"),
        ("src/distributed/loading/frozen_models.py", "load_frozen_auxiliary_model"),
        ("src/distributed/loading/model_loading.py", "_from_pretrained_on_local_gpu"),
        ("src/distributed/loading/model_loading.py", "_sequential_load_to_cuda"),
        ("src/distributed/loading/model_loading.py", "load_model_from_pretrained"),
    }
)

# The dense TP load materializes straight into DTensors, which the cast refuses. No dense family pins a
# parameter, and a mixed load there would fail FSDP2's wrap loudly.
UNCAST_LOADERS = frozenset({("src/distributed/loading/model_loading.py", "_load_tp_model")})


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

    with pytest.raises(ValueError, match=r"'pinned\.weight'.*convert_\*_bf16\.py"):
        cast_parameters_to_run_dtype(model, torch.bfloat16)


def test_a_sharded_parameter_refuses_the_cast():
    """A DTensor's ``.data`` rebind keeps its local shard in the old dtype, so a cast there must raise."""
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


@pytest.fixture(scope="module")
def pinned_checkpoints(tmp_path_factory) -> dict[str, str]:
    root = tmp_path_factory.mktemp("pinned")
    checkpoints = {family: str(root / family) for family in PINNED_FP32_FAMILIES}
    for family, path in checkpoints.items():
        build_tiny_family_checkpoint(TINY_MOE_FAMILIES[family], path)
    return checkpoints


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

    model, _ = load_model_from_pretrained(pinned_checkpoints["deepseek_v4"], args)

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


@pytest.fixture(scope="module")
def roster_checkpoints(tmp_path_factory) -> dict[str, str]:
    root = tmp_path_factory.mktemp("roster")
    checkpoints = {family: str(root / family) for family in TINY_MOE_FAMILIES}
    for family, path in checkpoints.items():
        build_tiny_family_checkpoint(TINY_MOE_FAMILIES[family], path)
    return checkpoints


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


def test_the_sentence_transformer_backbone_is_cast_and_finalized(tmp_path):
    """The default embedding path loads through SentenceTransformer's own ``from_pretrained``."""
    base = tmp_path / "base"
    build_tiny_family_checkpoint(TINY_MOE_FAMILIES["deepseek_v4"], str(base), load_cached_tokenizer(QWEN3_0_6B))
    runtime = types.SimpleNamespace(
        parallelism_config=types.SimpleNamespace(is_ep_mode=False, is_tp_mode=False, fp32_non_ep_params=False),
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


def _calls(function: ast.FunctionDef) -> set[str]:
    names = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            target = node.func
            names.add(target.id if isinstance(target, ast.Name) else getattr(target, "attr", ""))
    return names


@functools.cache
def _eager_loaders() -> dict[tuple[str, str], frozenset[str]]:
    found = {}
    for root in SWEPT_ROOTS:
        for path in sorted((REPO_ROOT / root).rglob("*.py")):
            rel = str(path.relative_to(REPO_ROOT))
            if rel.startswith(LOAD_CORE):
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.FunctionDef) and (calls := _calls(node)) & EAGER_LOAD_CALLS:
                    found[(rel, node.name)] = frozenset(calls)
    return found


def test_the_eager_load_surface_is_pinned():
    assert set(_eager_loaders()) == EAGER_LOADERS | UNCAST_LOADERS


@pytest.mark.parametrize("loader", sorted(EAGER_LOADERS), ids=lambda loader: loader[1])
def test_every_eager_loader_casts_and_finalizes(loader):
    calls = _eager_loaders()[loader]
    assert calls & CASTS
    assert FINALIZE in calls


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
