"""Loader collective-trace golden: what ``load_distributed_model`` issues on the ranks of a 4-rank job on
two 2-GPU nodes, for every branch of its loader ladder from a local directory and from the Hub, and
what the frozen reference load beside it issues.

Each case runs the real loader on a tiny checkpoint under :func:`recording_rank`, so the trace holds
the process-group builds with their members, the collectives with their operation, payload and group,
the device meshes, the c10d-store operations with their keys, the node-load phases with their labels
and, between them, the model builds :func:`cpu_model_builds` stands in for, one event each.
``max_concurrent_loading`` is 1, so rank 1 waits on rank 0 in every node-throttled load.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import shutil
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from unittest.mock import patch

import torch
from accelerate import PartialState
from huggingface_hub import constants as hub_constants
from huggingface_hub.file_download import repo_folder_name
from transformers import PreTrainedModel
from transformers.distributed.tensor_parallel import apply_tensor_parallelism

from src.checkpoint.reading import load_full_state_dict
from src.distributed import rank_uniform_env
from src.model_loading.parallel.frozen_models import load_frozen_auxiliary_model
from src.model_loading.parallel.loader import load_distributed_model
from src.runtime import load_coordination
from tests.common.collective_recorder import CollectiveTrace, rebind_everywhere, recording_rank
from tests.common.goldens import regen_main
from tests.common.parallelism import make_parallelism_config
from tests.common.pp_fakes import registered_pp_backbones
from tests.common.tiny_models import (
    TINY_DENSE_FAMILY,
    TINY_MOE_FAMILIES,
    TINY_UNWRAPPED_MOE_FAMILY,
    build_tiny_family_checkpoint,
    word_level_tokenizer,
)

GOLDEN_PATH = Path(__file__).with_suffix(".json")
WORLD_SIZE = 4
GPUS_PER_NODE = 2
# Pinned rather than auto-selected off the host's GPU and flash builds; Ulysses CP requires a flash backend.
ATTN_IMPLEMENTATION = "flash_attention_2"
# The namespace of the repos the fake Hub cache holds, one per checkpoint.
HUB_NAMESPACE = "golden"
# What a trace writes in place of the host's temporary checkpoint root.
CHECKPOINTS_PLACEHOLDER = "<checkpoints>"


@dataclasses.dataclass(frozen=True)
class LoadCase:
    """One loader branch: the checkpoint it reads, the ``ParallelismConfig`` fields that route it there
    and the variables it adds to the launcher's environment.

    ``from_hub`` names the checkpoint by its repo id in the fake Hub cache instead of by its directory,
    and ``frozen_reference`` loads it as a run's frozen reference. ``ranks`` are the ranks traced: a
    per-node case adds rank 2, which leads node 1's scope.
    """

    key: str
    checkpoint: str
    parallelism: Mapping[str, object] = dataclasses.field(default_factory=dict)
    env: Mapping[str, str] = dataclasses.field(default_factory=dict)
    init_from_scratch: bool = False
    from_hub: bool = False
    frozen_reference: bool = False
    ranks: tuple[int, ...] = (0, 1)


LOAD_CASES = (
    LoadCase("undistributed dense", "dense"),
    LoadCase("undistributed dense init_from_scratch", "dense", init_from_scratch=True),
    LoadCase("undistributed moe grouped-GEMM", "moe"),
    LoadCase("undistributed moe without an EP class", "moe_no_ep_class"),
    LoadCase("ep lazy", "moe", {"ep_size": 2}),
    LoadCase("ep lazy per-node filesystem", "moe", {"ep_size": 2}, {"DIST_SHARED_FILESYSTEM": "0"}, ranks=(0, 1, 2)),
    LoadCase("ep fallback", "moe_bin", {"ep_size": 2}),
    LoadCase("ep lazy-disabled", "moe", {"ep_size": 2, "ep_lazy_loading": False}),
    LoadCase("ep+etp lazy", "moe", {"ep_size": 2, "expert_tp_size": 2, "ep_scope": "global"}),
    LoadCase("ep+cp lazy", "moe", {"ep_size": 2, "cp_size": 2}),
    LoadCase("ep+tp lazy", "moe", {"ep_size": 4, "tp_size": 2, "ep_scope": "global"}),
    LoadCase("ep+tp fallback", "moe_bin", {"ep_size": 4, "tp_size": 2, "ep_scope": "global"}),
    LoadCase(
        "ep+tp lazy-disabled", "moe", {"ep_size": 4, "tp_size": 2, "ep_scope": "global", "ep_lazy_loading": False}
    ),
    LoadCase("tp dense", "dense", {"tp_size": 2}),
    LoadCase("tp moe", "moe", {"tp_size": 2}),
    LoadCase("cp dense", "dense", {"cp_size": 2}),
    LoadCase("cp moe", "moe", {"cp_size": 2}),
    LoadCase("pp dense", "dense", {"pp_size": 2}),
    LoadCase("pp+ep", "moe", {"pp_size": 2, "ep_size": 2}),
    LoadCase("frozen reference", "dense", frozen_reference=True),
    LoadCase("hub undistributed dense", "dense", from_hub=True),
    LoadCase("hub ep lazy", "moe", {"ep_size": 2}, from_hub=True),
    LoadCase(
        "hub ep lazy per-node filesystem",
        "moe",
        {"ep_size": 2},
        {"DIST_SHARED_FILESYSTEM": "0"},
        from_hub=True,
        ranks=(0, 1, 2),
    ),
    LoadCase("hub ep fallback", "moe_bin", {"ep_size": 2}, from_hub=True),
    LoadCase("hub ep+tp lazy", "moe", {"ep_size": 4, "tp_size": 2, "ep_scope": "global"}, from_hub=True),
    LoadCase("hub pp dense", "dense", {"pp_size": 2}, from_hub=True),
    LoadCase("hub frozen reference", "dense", from_hub=True, frozen_reference=True),
)


@dataclasses.dataclass(frozen=True)
class Checkpoints:
    """The cases' checkpoints: a directory each under ``root``, and the same files as repos of the Hub
    cache under ``root``, the only cache a traced load reads."""

    root: Path

    @property
    def hub_cache(self) -> Path:
        return self.root / "hub"

    def source(self, case: LoadCase) -> str:
        """What ``case`` names its checkpoint by: its repo id when it loads from the Hub, else its directory."""
        return f"{HUB_NAMESPACE}/{case.checkpoint}" if case.from_hub else str(self.root / case.checkpoint)

    def portable(self, events: list[str]) -> list[str]:
        """``events`` with the host's checkpoint root written as :data:`CHECKPOINTS_PLACEHOLDER`."""
        return [event.replace(str(self.root), CHECKPOINTS_PLACEHOLDER) for event in events]

    @contextlib.contextmanager
    def offline_hub(self) -> Iterator[None]:
        """The Hub as nothing but :attr:`hub_cache`, read offline. huggingface_hub reads both settings at
        call time, for transformers as well."""
        with (
            patch.object(hub_constants, "HF_HUB_CACHE", str(self.hub_cache)),
            patch.object(hub_constants, "HF_HUB_OFFLINE", True),
        ):
            yield


def build_checkpoints(root: Path) -> Checkpoints:
    """The checkpoints the cases read, written under ``root``: a tiny Qwen3, a tiny Qwen3-MoE in
    safetensors and as a ``pytorch_model.bin`` no lazy loader reads, and a tiny MoE of a family no EP
    layer class wraps. Each is also its namesake repo's ``main`` in the Hub cache."""
    tokenizer = word_level_tokenizer()
    directories = {name: root / name for name in ("dense", "moe", "moe_bin", "moe_no_ep_class")}
    build_tiny_family_checkpoint(TINY_DENSE_FAMILY, str(directories["dense"]), tokenizer)
    build_tiny_family_checkpoint(TINY_MOE_FAMILIES["qwen3_moe"], str(directories["moe"]), tokenizer)
    shutil.copytree(directories["moe"], directories["moe_bin"], ignore=shutil.ignore_patterns("*.safetensors*"))
    torch.save(load_full_state_dict(str(directories["moe"])), directories["moe_bin"] / "pytorch_model.bin")
    build_tiny_family_checkpoint(TINY_UNWRAPPED_MOE_FAMILY, str(directories["moe_no_ep_class"]), tokenizer)
    checkpoints = Checkpoints(root)
    for name, directory in directories.items():
        repo = checkpoints.hub_cache / repo_folder_name(repo_id=f"{HUB_NAMESPACE}/{name}", repo_type="model")
        commit = hashlib.sha1(name.encode()).hexdigest()
        shutil.copytree(directory, repo / "snapshots" / commit)
        (repo / "refs").mkdir()
        (repo / "refs" / "main").write_text(commit)
    return checkpoints


def _off_gpu(value):
    """``value``, with a CUDA device replaced by the CPU."""
    if isinstance(value, (str, torch.device)) and torch.device(value).type == "cuda":
        return torch.device("cpu")
    return value


@contextlib.contextmanager
def cpu_model_builds(trace: CollectiveTrace) -> Iterator[None]:
    """Every model build as one event, on the CPU whatever device the loader names.

    A load onto a TP device mesh is sharded by transformers' ``apply_tensor_parallelism``, the step
    ``from_pretrained`` takes for one, over the recorder's CPU mesh. A module moved to a GPU stays on
    the CPU, and a flash attention label is accepted without a GPU, since no traced load runs attention.
    """
    real_from_pretrained = PreTrainedModel.from_pretrained.__func__
    real_from_config = PreTrainedModel._from_config.__func__
    real_module_to = torch.nn.Module.to
    real_attention_check = PreTrainedModel._check_and_adjust_attn_implementation

    def from_pretrained(cls, *args, **kwargs):
        device_mesh = kwargs.pop("device_mesh", None)
        kwargs.pop("distributed_config", None)
        device_map = kwargs.get("device_map")
        trace.record("from_pretrained", "device_mesh" if device_mesh is not None else f"device_map={device_map}")
        if device_map not in (None, "meta"):
            kwargs["device_map"] = "cpu"
        with trace.pause():
            loaded = real_from_pretrained(cls, *args, **kwargs)
            if device_mesh is not None:
                apply_tensor_parallelism(loaded[0] if isinstance(loaded, tuple) else loaded, device_mesh)
        return loaded

    def from_config(cls, *args, **kwargs):
        trace.record("from_config")
        with trace.pause():
            return real_from_config(cls, *args, **kwargs)

    def to(module, *args, **kwargs):
        return real_module_to(module, *map(_off_gpu, args), **{key: _off_gpu(value) for key, value in kwargs.items()})

    def attention_check(model, attn_implementation, *args, **kwargs):
        if attn_implementation is not None and attn_implementation.startswith("flash_attention"):
            return attn_implementation
        return real_attention_check(model, attn_implementation, *args, **kwargs)

    with (
        patch.object(PreTrainedModel, "from_pretrained", classmethod(from_pretrained)),
        patch.object(PreTrainedModel, "_from_config", classmethod(from_config)),
        patch.object(torch.nn.Module, "to", to),
        patch.object(PreTrainedModel, "_check_and_adjust_attn_implementation", attention_check),
    ):
        yield


def recorded_load_phases(trace: CollectiveTrace) -> contextlib.AbstractContextManager[None]:
    """Each ``joined_node_load`` as an event carrying its phase label, ahead of the load's own events."""
    real_joined_node_load = load_coordination.joined_node_load

    def joined_node_load(what: str, max_concurrent: int | None):
        trace.record("joined_node_load", what)
        return real_joined_node_load(what, max_concurrent)

    return rebind_everywhere({real_joined_node_load: joined_node_load})


def trace_rank(
    case: LoadCase, rank: int, checkpoints: Checkpoints, rank0: CollectiveTrace | None = None
) -> CollectiveTrace:
    """What ``case``'s load issues as ``rank``, receiving the object broadcasts of ``rank0``, global rank
    0's trace of it.

    The parallelism config is built before the recording starts and the rank-uniform EP environment
    counts as verified, as the entry scripts do both in their distributed setup, before the load.
    """
    # Outside the recording, so accelerate sees one process; the loaders' accelerate loggers need a state.
    PartialState()
    config = make_parallelism_config(
        world_size=WORLD_SIZE, gpus_per_node=GPUS_PER_NODE, rank=rank, max_concurrent_loading=1, **case.parallelism
    )
    source = checkpoints.source(case)
    with (
        recording_rank(rank, WORLD_SIZE, GPUS_PER_NODE, env=case.env, rank0=rank0) as trace,
        checkpoints.offline_hub(),
        cpu_model_builds(trace),
        recorded_load_phases(trace),
        patch.object(rank_uniform_env, "_env_uniformity_verified", True),
        # The traced branches are family-agnostic; the tiny MoE's backbone is not registered for pipeline parallelism.
        registered_pp_backbones("Qwen3MoeModel"),
    ):
        if case.frozen_reference:
            load_frozen_auxiliary_model(
                source,
                dtype=torch.bfloat16,
                attn_implementation=ATTN_IMPLEMENTATION,
                download_tag="reference_model",
                trust_remote_code=False,
            )
        else:
            load_distributed_model(
                source,
                config,
                attn_implementation=ATTN_IMPLEMENTATION,
                init_from_scratch=case.init_from_scratch,
                trust_remote_code=True,
            )
    return trace


def golden_key(rank: int) -> str:
    return f"rank {rank}"


def trace_case(case: LoadCase, checkpoints: Checkpoints) -> dict[str, list[str]]:
    """``case``'s trace on each of its ranks, keyed by :func:`golden_key`. Rank 0 is traced first, and
    every other rank receives its object broadcasts."""
    rank0 = trace_rank(case, 0, checkpoints)
    traces = {rank: rank0 if rank == 0 else trace_rank(case, rank, checkpoints, rank0) for rank in case.ranks}
    return {golden_key(rank): checkpoints.portable(trace.events) for rank, trace in traces.items()}


def render_golden() -> str:
    """The golden file: every case's trace on each of its ranks, one event a line."""
    with tempfile.TemporaryDirectory() as root:
        checkpoints = build_checkpoints(Path(root))
        traces = {case.key: trace_case(case, checkpoints) for case in LOAD_CASES}
    return json.dumps(traces, indent=2) + "\n"


if __name__ == "__main__":
    regen_main(GOLDEN_PATH, render_golden)
