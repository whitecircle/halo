"""A recording ``torch.distributed`` for one simulated rank of a CPU-only job.

:func:`recording_rank` runs its body as global rank ``rank`` of a ``world_size``-rank job on
``gpus_per_node``-GPU nodes: the environment holds the launcher's variables for that rank and none of
the host's, torch's fake backend is the default process group, CUDA reads as absent and the c10d
store is an in-memory one. The process-group builds, collectives and device meshes the body issues,
and its c10d-store operations, are appended to the yielded :class:`CollectiveTrace`, one line an
event, in call order. A ``torch.distributed`` function is replaced by identity, under every name a
loaded module holds it by, so a call through ``distributed_c10d`` or DTensor's collective helpers is
caught as well as one through ``torch.distributed``. A collective or functional collective with no
model here raises instead of running unrecorded.

The simulated ranks run the same code on one filesystem, so a peer's value is taken to be this
rank's own: a tensor broadcast leaves the tensor as it is, a gather fills every slot with this rank's
value, a sum is this rank's input times the group size and a min, max or average the input itself, a
scatter keeps this rank's slot of the list (a rank the list is not sent from keeps its input), and a
store wait on a key a peer writes returns at once. An object broadcast from global rank 0 is where one
rank decides for all, so a rank given rank 0's trace of the same body receives the payloads rank 0
recorded, in order.
"""

from __future__ import annotations

import contextlib
import copy
import inspect
import os
import sys
import types
from collections.abc import Callable, Iterator, Mapping
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch.distributed import device_mesh
from torch.distributed import distributed_c10d as c10d

from src.distributed.runtime import reset_shared_filesystem_consensus
from tests.common.distributed import fake_process_group

_WORLD = "world"
# The ``torch.distributed`` queries and process-group lifecycle calls, which issue no collective. A
# recording leaves them as they are and models or refuses every other function the package exports.
_NON_COLLECTIVES = frozenset(
    {
        "destroy_process_group",
        "get_backend",
        "get_backend_config",
        "get_debug_level",
        "get_default_backend_for_device",
        "get_global_rank",
        "get_group_rank",
        "get_node_local_rank",
        "get_pg_count",
        "get_process_group_ranks",
        "get_rank",
        "get_world_size",
        "init_process_group",
        "is_backend_available",
        "is_gloo_available",
        "is_initialized",
        "is_mpi_available",
        "is_nccl_available",
        "is_torchelastic_launched",
        "is_ucc_available",
        "is_xccl_available",
        "supports_complex",
    }
)
# Environment variables a launcher or the toolkit sets; a recording sees only the ones it sets itself.
_LAUNCH_VARIABLE_PREFIXES = ("ACCELERATE_", "DIST_", "HALO_", "NVLINK_", "SLURM_", "TORCHELASTIC_")
_LAUNCH_VARIABLES = frozenset(
    {
        "GROUP_RANK",
        "GROUP_WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "RANK",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "WORLD_SIZE",
    }
)
# The reductions a rank's own input stands in for: every rank holds the same value.
_IDENTITY_REDUCTIONS = (dist.ReduceOp.MIN, dist.ReduceOp.MAX, dist.ReduceOp.AVG)


class CollectiveTrace:
    """The ordered events of one recorded rank, and the object broadcasts it received from global rank 0."""

    def __init__(self) -> None:
        self.events: list[str] = []
        # Per group label, each payload broadcast from global rank 0, in order.
        self.from_rank0: dict[str, list[list]] = {}
        self._paused = 0

    @property
    def paused(self) -> bool:
        return self._paused > 0

    def record(self, *parts: object) -> None:
        if not self.paused:
            self.events.append(" ".join(str(part) for part in parts))

    @contextlib.contextmanager
    def pause(self) -> Iterator[None]:
        """Leave out what the body issues: the internals of a stand-in that is itself one event.

        A modeled call runs unrecorded, and a refused one runs on the fake backend.
        """
        self._paused += 1
        try:
            yield
        finally:
            self._paused -= 1


class _RecordingStore:
    """The job's c10d store as one rank sees it alone: its own writes land, a key only a peer writes
    reads as empty, and a wait never blocks."""

    def __init__(self, trace: CollectiveTrace) -> None:
        self._trace = trace
        self._kv: dict[str, str | int] = {}

    def add(self, key: str, amount: int) -> int:
        self._trace.record("store add", key)
        self._kv[key] = int(self._kv.get(key, 0)) + amount
        return self._kv[key]

    def set(self, key: str, value) -> None:
        self._trace.record("store set", key)
        self._kv[key] = value

    def get(self, key: str):
        self._trace.record("store get", key)
        return self._kv.get(key, "")

    def multi_get(self, keys) -> list:
        keys = list(keys)
        self._trace.record("store multi_get", _brace_keys(keys))
        return [self._kv.get(key, "") for key in keys]

    def wait(self, keys, timeout=None) -> None:
        del timeout
        self._trace.record("store wait", _brace_keys(list(keys)))

    def delete_key(self, key: str) -> bool:
        self._trace.record("store delete", key)
        return self._kv.pop(key, None) is not None


def _brace_keys(keys: list[str]) -> str:
    """``keys`` with the directory they share written once, as ``dir/{a,b}``."""
    directories = {key.rpartition("/")[0] for key in keys}
    if len(keys) < 2 or len(directories) > 1:
        return " ".join(keys)
    return f"{directories.pop()}/{{{','.join(key.rpartition('/')[2] for key in keys)}}}"


def _group_label(group) -> str:
    return _WORLD if group is None else str(dist.get_process_group_ranks(group))


def _tensor_label(tensor: torch.Tensor) -> str:
    return f"{str(tensor.dtype).removeprefix('torch.')}{list(tensor.shape)}"


def _source_label(src: int | None, group_src: int | None) -> str:
    return f"src={src}" if group_src is None else f"group_src={group_src}"


def _global_source(src: int | None, group, group_src: int | None) -> int | None:
    if group_src is None:
        return src
    return group_src if group is None else dist.get_global_rank(group, group_src)


def _blocking(name: str, async_op: bool) -> None:
    if async_op:
        raise NotImplementedError(f"the collective recorder models a blocking {name} only")


class _ModeledCalls:
    """The stand-ins for the ``torch.distributed`` calls the recorder models, recording into ``trace``.

    ``rank0`` is global rank 0's trace of the same body, whose object broadcasts this rank receives.
    """

    def __init__(self, trace: CollectiveTrace, rank0: CollectiveTrace | None) -> None:
        self._trace = trace
        self._from_rank0 = None if rank0 is None else {group: iter(sent) for group, sent in rank0.from_rank0.items()}
        self._real_new_group = c10d.new_group
        self._real_init_device_mesh = device_mesh.init_device_mesh

    def replacements(self) -> dict[Callable, Callable]:
        """Each modeled function mapped to its stand-in."""
        modeled = {
            c10d.new_group: self.new_group,
            c10d.all_reduce: self.all_reduce,
            c10d.broadcast: self.broadcast,
            c10d.broadcast_object_list: self.broadcast_object_list,
            c10d.all_gather: self.all_gather,
            c10d.all_gather_object: self.all_gather_object,
            c10d.barrier: self.barrier,
            c10d.scatter: self.scatter,
            device_mesh.init_device_mesh: self.init_device_mesh,
        }
        return modeled | {real: self._refusal(name, real) for name, real in _unmodeled(modeled).items()}

    def new_group(self, ranks=None, *args, **kwargs):
        self._trace.record("new_group", _WORLD if ranks is None else list(ranks))
        with self._trace.pause():
            return self._real_new_group(ranks, *args, **kwargs)

    def init_device_mesh(self, device_type, mesh_shape, **kwargs):
        """One event, then the mesh built over the fake process group on the CPU."""
        self._trace.record("init_device_mesh", device_type, tuple(mesh_shape), kwargs.get("mesh_dim_names"))
        with self._trace.pause():
            return self._real_init_device_mesh("cpu", mesh_shape, **kwargs)

    def all_reduce(self, tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
        _blocking("all_reduce", async_op)
        self._trace.record("all_reduce", str(op).rsplit(".", 1)[-1], _tensor_label(tensor), _group_label(group))
        if op == dist.ReduceOp.SUM:
            tensor.mul_(dist.get_world_size(group))
        elif op not in _IDENTITY_REDUCTIONS:
            raise NotImplementedError(f"the collective recorder has no model of an all_reduce with {op}")

    def broadcast(self, tensor, src=None, group=None, async_op=False, group_src=None):
        _blocking("broadcast", async_op)
        self._trace.record("broadcast", _source_label(src, group_src), _tensor_label(tensor), _group_label(group))

    def broadcast_object_list(self, object_list, src=None, group=None, device=None, group_src=None):
        label = _group_label(group)
        if _global_source(src, group, group_src) == 0 and not self._trace.paused:
            if self._from_rank0 is not None:
                object_list[:] = copy.deepcopy(self._next_from_rank0(label))
            self._trace.from_rank0.setdefault(label, []).append(copy.deepcopy(object_list))
        self._trace.record("broadcast_object_list", _source_label(src, group_src), label, repr(object_list))

    def all_gather(self, tensor_list, tensor, group=None, async_op=False):
        _blocking("all_gather", async_op)
        self._trace.record("all_gather", _tensor_label(tensor), _group_label(group))
        for slot in tensor_list:
            slot.copy_(tensor)

    def all_gather_object(self, object_list, obj, group=None):
        self._trace.record("all_gather_object", f"object[{len(object_list)}]", _group_label(group))
        object_list[:] = [obj] * len(object_list)

    def barrier(self, group=None, async_op=False, device_ids=None):
        _blocking("barrier", async_op)
        self._trace.record("barrier", _group_label(group))

    def scatter(self, tensor, scatter_list=None, src=None, group=None, async_op=False, group_src=None):
        _blocking("scatter", async_op)
        self._trace.record("scatter", _source_label(src, group_src), _tensor_label(tensor), _group_label(group))
        if scatter_list is not None:
            tensor.copy_(scatter_list[dist.get_rank(group)])

    def _next_from_rank0(self, group_label: str) -> list:
        try:
            return next(self._from_rank0.get(group_label, iter(())))
        except StopIteration:
            raise RuntimeError(
                f"this rank broadcasts from global rank 0 on group {group_label} more often than rank 0 did, "
                f"which on a real job leaves it waiting for a broadcast rank 0 never sends"
            ) from None

    def _refusal(self, name: str, real: Callable) -> Callable:
        def refuse(*args, **kwargs):
            if self._trace.paused:
                return real(*args, **kwargs)
            raise NotImplementedError(
                f"the collective recorder has no model of {name}; add one to tests/common/collective_recorder.py "
                f"before tracing code that calls it"
            )

        return refuse


def _unmodeled(modeled: Mapping[Callable, Callable]) -> dict[str, Callable]:
    """Every ``torch.distributed`` function outside :data:`_NON_COLLECTIVES` that ``modeled`` does not
    cover, and every functional collective, by qualified name."""
    distributed = {
        f"torch.distributed.{name}": function
        for name in c10d.__all__
        if name not in _NON_COLLECTIVES
        # By type: an attribute read on the deprecated ``reduce_op`` warns.
        and issubclass(type(function := getattr(c10d, name)), (types.FunctionType, types.BuiltinFunctionType))
        and function not in modeled
    }
    functional = {
        f"{funcol.__name__}.{name}": function
        for name, function in vars(funcol).items()
        if inspect.isfunction(function) and function.__module__ == funcol.__name__ and not name.startswith("_")
    }
    return distributed | functional


@contextlib.contextmanager
def rebind_everywhere(replacements: Mapping[object, object]) -> Iterator[None]:
    """Rebind, for the body, every module attribute that is one of ``replacements``' keys to its value.

    By identity, so each name a loaded module holds the object by is rebound: its defining module's,
    a package's re-export and every ``from ... import`` of it.
    """
    by_identity = {id(real): replacement for real, replacement in replacements.items()}
    with contextlib.ExitStack() as stack:
        for module in list(sys.modules.values()):
            namespace = getattr(module, "__dict__", None)
            if not isinstance(namespace, dict):
                continue
            for name, value in list(namespace.items()):
                if id(value) in by_identity:
                    stack.enter_context(patch.object(module, name, by_identity[id(value)]))
        yield


@contextlib.contextmanager
def _launch_environment(rank: int, world_size: int, gpus_per_node: int, env: Mapping[str, str] | None):
    """``os.environ`` as a torchrun launch sets it for ``rank``, plus ``env``, with every launcher and
    toolkit variable of the host removed. The whole environment is restored on exit."""
    with patch.dict(os.environ):
        host = [name for name in os.environ if name in _LAUNCH_VARIABLES or name.startswith(_LAUNCH_VARIABLE_PREFIXES)]
        for name in host:
            del os.environ[name]
        os.environ.update(
            RANK=str(rank),
            WORLD_SIZE=str(world_size),
            LOCAL_RANK=str(rank % gpus_per_node),
            LOCAL_WORLD_SIZE=str(gpus_per_node),
            **(env or {}),
        )
        yield


@contextlib.contextmanager
def recording_rank(
    rank: int,
    world_size: int,
    gpus_per_node: int,
    env: Mapping[str, str] | None = None,
    rank0: CollectiveTrace | None = None,
) -> Iterator[CollectiveTrace]:
    """Run the body as ``rank`` of the simulated job, recording what it issues.

    ``env`` is what the body's environment holds beyond the launcher's variables (a filesystem
    declaration, say). ``rank0`` is global rank 0's trace of the same body, whose object broadcasts
    this rank receives. The shared-filesystem consensus starts and ends unresolved, as in a fresh
    process.
    """
    trace = CollectiveTrace()
    store = _RecordingStore(trace)
    with contextlib.ExitStack() as stack:
        stack.enter_context(_launch_environment(rank, world_size, gpus_per_node, env))
        stack.enter_context(fake_process_group(rank, world_size))
        stack.enter_context(patch.object(torch.cuda, "is_available", lambda: False))
        stack.enter_context(patch.object(c10d, "_get_default_store", lambda: store))
        stack.enter_context(rebind_everywhere(_ModeledCalls(trace, rank0).replacements()))
        reset_shared_filesystem_consensus()
        stack.callback(reset_shared_filesystem_consensus)
        yield trace
