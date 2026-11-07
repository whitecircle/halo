"""How the streamed weight sync cuts its chunks. Snapshots stay off pinned host memory, as a chunk that transits the
host is copied out and back over PCIe, and those two copies would set the rate instead of the fabric."""

from collections.abc import Callable, Iterable

import torch
from torch.distributed.tensor import DTensor

from halo.env import env_positive_int
from halo.kernels.lowp.block_fp8 import SCALE_SUFFIX
from halo.weight_sync.transport.packed_broadcast import PACKED_BUFFER_BYTES

# Read once at import. The forwarding rank holds this plus the largest tensor, which is a chunk of its own; each
# chunk costs one engine round trip, so a larger budget trades memory for fewer.
WEIGHT_SYNC_CHUNK_BYTES = env_positive_int("HALO_WEIGHT_SYNC_CHUNK_MB", PACKED_BUFFER_BYTES >> 20) << 20


def describe_chunk(chunk: list[tuple[str, torch.Tensor]]) -> tuple[list[str], list[str], list[list[int]]]:
    """``(names, dtype names, shapes)`` of one chunk in wire order, as both engines' update requests declare it."""
    names = [name for name, _ in chunk]
    dtypes = [str(param.dtype).split(".")[-1] for _, param in chunk]
    shapes = [list(param.shape) for _, param in chunk]
    return names, dtypes, shapes


# A block-FP8 weight and its scale are one co-load group of two, found by dtype and name, as the engine re-derives a
# layer's kernel tensors only once both have landed.
_FP8_PAIR_GROUP = -1
_FP8_PAIR_SIZE = 2


def _co_load_key(name: str, tensor: torch.Tensor, groups: tuple[tuple[str, ...], ...]) -> tuple[int, str] | None:
    """``(group index, module prefix)`` of a parameter one of ``groups`` claims, else ``None``; a block-FP8 weight and
    its ``_scale_inv`` share the group :data:`_FP8_PAIR_GROUP` keyed by the weight's name."""
    if tensor.dtype == torch.float8_e4m3fn:
        return _FP8_PAIR_GROUP, name
    if name.endswith(".weight" + SCALE_SUFFIX):
        return _FP8_PAIR_GROUP, name.removesuffix(SCALE_SUFFIX)
    for index, suffixes in enumerate(groups):
        for suffix in suffixes:
            if name == suffix or name.endswith("." + suffix):
                return index, name[: len(name) - len(suffix)]
    return None


def split_co_loaded(
    chunk: list[tuple[str, torch.Tensor]], groups: tuple[tuple[str, ...], ...]
) -> tuple[list[tuple[str, torch.Tensor]], list[tuple[str, torch.Tensor]]]:
    """``(sendable, held)``, order kept: a co-load group not yet complete in ``chunk`` waits for its partners."""
    keys = [_co_load_key(name, tensor, groups) for name, tensor in chunk]
    members: dict[tuple[int, str], set[str]] = {}
    for (name, _), key in zip(chunk, keys, strict=True):
        if key is not None:
            members.setdefault(key, set()).add(name)
    sendable, held = [], []
    for item, key in zip(chunk, keys, strict=True):
        size = None if key is None else _FP8_PAIR_SIZE if key[0] == _FP8_PAIR_GROUP else len(groups[key[0]])
        complete = key is None or len(members[key]) == size
        (sendable if complete else held).append(item)
    return sendable, held


def declared_co_load_groups(
    groups: tuple[tuple[str, ...], ...], module_names: Iterable[str]
) -> tuple[tuple[str, ...], ...]:
    """Narrowed to what ``module_names`` carries; an MLA block without ``q_lora_rank`` has no ``q_a_proj``."""
    names = set(module_names)

    def declared(suffix: str) -> bool:
        module = suffix.rsplit(".", 1)[0]
        return any(name == module or name.endswith("." + module) for name in names)

    return tuple(tuple(suffix for suffix in group if declared(suffix)) for group in groups)


def resolve_sync_device(device: torch.device | str | int) -> torch.device:
    """The explicit ``cuda:N`` form of ``device``, as ``torch.device("cuda")`` compares unequal to ``cuda:0``."""
    resolved = device if isinstance(device, torch.device) else torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    return resolved


def snapshot_to_sync_device(
    name: str,
    weights: torch.Tensor,
    *,
    device: torch.device | None,
    buffered_bytes: int,
    budget: int,
    flush: Callable[[], None],
    owned: bool = False,
) -> torch.Tensor:
    """A contiguous copy of ``weights`` on ``device`` (the source's when ``None``), after ``flush`` sent the pending
    chunk if ``weights`` would pass ``budget``. An alias would let the next optimizer step rewrite a buffered weight.
    ``owned`` says the caller made ``weights`` for this push and keeps no other reference, so a contiguous one already
    on ``device`` is its own snapshot. Async, completed by the flush. A param no broadcast can carry raises first, before it can quiesce an engine."""
    if isinstance(weights, DTensor):
        raise TypeError(
            f"Parameter {name!r} is a DTensor: the weight-sync client broadcasts raw local storage but "
            f"describes it with the global shape, so a sharded param would be read past its shard. "
            f"Materialize it first (materialize_dtensor / full_tensor)."
        )
    if weights.ndim == 0:
        raise ValueError(
            f"Parameter {name!r} is 0-dimensional: the broadcast protocols re-view every tensor as bytes, "
            f"which needs a last dimension, and the consumer rebuilds it from the declared shape. "
            f"Give the parameter an explicit shape (e.g. (1,)) in the model."
        )
    if buffered_bytes > 0 and buffered_bytes + weights.nbytes > budget:
        flush()
    if owned and weights.is_contiguous() and (device is None or weights.device == device):
        return weights
    snapshot = torch.empty_like(weights, memory_format=torch.contiguous_format, device=device)
    snapshot.copy_(weights.detach(), non_blocking=True)
    return snapshot
