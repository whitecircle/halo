#!/usr/bin/env python
"""``cast_parameters_to_run_dtype``: the loaded-parameter cast every training and scoring loader applies.

transformers keeps a family's ``_keep_in_fp32_modules_strict`` parameters in fp32 (DeepSeek-V4 norms
and hyper-connections, GLM-5 Next KDA state, Inkling short convolutions). The EP loaders cast them to
the run dtype; a loader that does not leaves a mixed-dtype model that FSDP2 refuses to shard and whose
fp32 DeepSeek-V4 norms feed fp32 activations into bf16 projections. These tests pin (1) the helper's
contract, (2) that the unsharded loaders a CPU run reaches hand back a uniform run-dtype model for every
pinned family, and (3) that every eager load in ``src/`` applies the cast, so a new loader that forgets
it fails here rather than on its first GPU step.

Run: python tests/cpu/models/test_run_dtype_cast.py  (or pytest)
"""

import ast
import types

import pytest
import torch
import torch.nn as nn
from accelerate import PartialState
from torch.distributed.tensor import Shard, distribute_tensor
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

from src.distributed.loading.frozen_models import load_frozen_auxiliary_model
from src.distributed.loading.model_loading import load_model_from_pretrained
from src.models.loading.dtype import cast_parameters_to_run_dtype
from tests.common.distributed import fake_process_group_mesh
from tests.common.tiny_models import PINNED_FP32_FAMILIES, build_tiny_pinned_checkpoint
from tests.common.utils import REPO_ROOT

# The loaders log through accelerate's logger, which requires an initialized state.
PartialState()

CAST = "cast_parameters_to_run_dtype"

# The eager load entry points: a function calling one of these materializes weights through
# ``from_pretrained`` and so inherits the fp32 pins.
EAGER_LOAD_CALLS = frozenset({"from_pretrained_verified", "auto_load_model"})

# The package defining those entry points. Conversion tools load through it too and must keep the pins,
# so the cast belongs to each training loader rather than to the shared core.
LOAD_CORE = "src/models/loading/"

# Every src function the sweep below finds. Pinned exactly so a renamed or new loader is a deliberate
# edit rather than a silently shrunk sweep.
EAGER_LOADERS = frozenset(
    {
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
        self.head = nn.Linear(4, 4, bias=False).to(torch.bfloat16)
        self.head.weight = self.run.weight
        self.index = nn.Parameter(torch.arange(4), requires_grad=False)
        self.register_buffer("bias_state", torch.zeros(4, dtype=torch.float32))


def _off_dtype(model: nn.Module, dtype: torch.dtype) -> list[str]:
    return [n for n, p in model.named_parameters() if p.is_floating_point() and p.dtype != dtype]


def test_the_cast_moves_floating_parameters_only():
    model = _Mixed()
    pinned_weight = model.pinned.weight

    cast_parameters_to_run_dtype(model, torch.bfloat16)

    assert _off_dtype(model, torch.bfloat16) == []
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
    return {family: build_tiny_pinned_checkpoint(family, str(root / family)) for family in PINNED_FP32_FAMILIES}


def _stock_class(family: str):
    return AutoModelForImageTextToText if family == "glm5_next" else AutoModelForCausalLM


@pytest.mark.parametrize("family", sorted(PINNED_FP32_FAMILIES))
def test_a_bf16_from_pretrained_keeps_the_family_pins(pinned_checkpoints, family):
    """The premise the loader tests rest on: without the cast, these checkpoints load mixed."""
    model = _stock_class(family).from_pretrained(pinned_checkpoints[family], dtype=torch.bfloat16)

    assert _off_dtype(model, torch.bfloat16)


@pytest.mark.parametrize("family", sorted(PINNED_FP32_FAMILIES))
def test_the_frozen_loader_scores_in_the_run_dtype(pinned_checkpoints, family):
    model = load_frozen_auxiliary_model(pinned_checkpoints[family], dtype=torch.bfloat16, device_map="cpu")

    assert _off_dtype(model, torch.bfloat16) == []


def test_the_path_string_loader_trains_in_the_requested_dtype(pinned_checkpoints):
    args = types.SimpleNamespace(model_init_kwargs={"dtype": "bfloat16"}, gradient_checkpointing=False)

    model, _ = load_model_from_pretrained(pinned_checkpoints["deepseek_v4"], args)

    assert _off_dtype(model, torch.bfloat16) == []


def _calls(function: ast.FunctionDef) -> set[str]:
    names = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            target = node.func
            names.add(target.id if isinstance(target, ast.Name) else getattr(target, "attr", ""))
    return names


def _eager_loaders() -> dict[tuple[str, str], set[str]]:
    found = {}
    for path in sorted((REPO_ROOT / "src").rglob("*.py")):
        rel = str(path.relative_to(REPO_ROOT))
        if rel.startswith(LOAD_CORE):
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.FunctionDef) and (calls := _calls(node)) & EAGER_LOAD_CALLS:
                found[(rel, node.name)] = calls
    return found


def test_the_eager_load_surface_is_pinned():
    assert set(_eager_loaders()) == EAGER_LOADERS | UNCAST_LOADERS


@pytest.mark.parametrize("loader", sorted(EAGER_LOADERS), ids=lambda loader: loader[1])
def test_every_eager_loader_casts_to_the_run_dtype(loader):
    assert CAST in _eager_loaders()[loader]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
