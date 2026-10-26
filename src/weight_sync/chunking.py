"""How the streamed weight sync cuts its chunks: the chunk budget, the parameter groups an engine loads only
together, and the parameter snapshots on the sync device.

Snapshots are taken on the device, not in pinned host memory: a chunk that transits the host is copied out
and back over PCIe before it reaches the NIC, and those two copies, not the fabric, would set the sync rate.
"""

from collections.abc import Callable, Iterable

import torch
from torch.distributed.tensor import DTensor

from src.env import env_positive_int
from src.weight_sync.transport.packed_broadcast import PACKED_BUFFER_BYTES

# Budget for one streamed chunk on the sync GPU, HALO_WEIGHT_SYNC_CHUNK_MB (default one packed buffer), read
# once at import. The forwarding rank stages this plus the largest single tensor, since a tensor above it is
# a chunk of its own (both wire protocols describe whole tensors), and the SGLang staging arena is at least
# this size. Each chunk costs one engine round trip, so a larger budget trades staged memory for fewer.
WEIGHT_SYNC_CHUNK_BYTES = env_positive_int("HALO_WEIGHT_SYNC_CHUNK_MB", PACKED_BUFFER_BYTES >> 20) << 20


def describe_chunk(chunk: list[tuple[str, torch.Tensor]]) -> tuple[list[str], list[str], list[list[int]]]:
    """``(names, dtype names, shapes)`` of one chunk in wire order: what both engines' update requests
    declare ahead of the broadcasts, so the consumer allocates and unpacks against them."""
    names = [name for name, _ in chunk]
    dtypes = [str(param.dtype).split(".")[-1] for _, param in chunk]
    shapes = [list(param.shape) for _, param in chunk]
    return names, dtypes, shapes


def _co_load_key(name: str, groups: tuple[tuple[str, ...], ...]) -> tuple[int, str] | None:
    """``(group index, module prefix)`` of a parameter one of ``groups`` claims, else ``None``."""
    for index, suffixes in enumerate(groups):
        for suffix in suffixes:
            if name == suffix or name.endswith("." + suffix):
                return index, name[: len(name) - len(suffix)]
    return None


def split_co_loaded(
    chunk: list[tuple[str, torch.Tensor]], groups: tuple[tuple[str, ...], ...]
) -> tuple[list[tuple[str, torch.Tensor]], list[tuple[str, torch.Tensor]]]:
    """``(sendable, held)``: the members of a co-load group not yet complete in ``chunk`` are held.

    An engine whose loader fuses a module's tensors from a cache local to one request drops a member
    that arrives without the rest, so the incomplete set waits for the chunk its partners land in.
    Order is preserved on both sides.
    """
    keys = [_co_load_key(name, groups) for name, _ in chunk]
    members: dict[tuple[int, str], set[str]] = {}
    for (name, _), key in zip(chunk, keys, strict=True):
        if key is not None:
            members.setdefault(key, set()).add(name)
    sendable, held = [], []
    for item, key in zip(chunk, keys, strict=True):
        complete = key is None or len(members[key]) == len(groups[key[0]])
        (sendable if complete else held).append(item)
    return sendable, held


def declared_co_load_groups(
    groups: tuple[tuple[str, ...], ...], module_names: Iterable[str]
) -> tuple[tuple[str, ...], ...]:
    """``groups`` narrowed to the members whose module ``module_names`` carries.

    An engine fuses a group only where the model declares every member (an MLA block without
    ``q_lora_rank`` has no ``q_a_proj``), so a member no module spells is dropped, and a group
    left with one member is complete on its own.
    """
    names = set(module_names)

    def declared(suffix: str) -> bool:
        module = suffix.rsplit(".", 1)[0]
        return any(name == module or name.endswith("." + module) for name in names)

    return tuple(tuple(suffix for suffix in group if declared(suffix)) for group in groups)


def resolve_sync_device(device: torch.device | str | int) -> torch.device:
    """The explicit ``cuda:N`` form of ``device``.

    An index (0 included) and a bare ``"cuda"`` must name the same GPU the transport is formed on
    and the snapshots are staged on, and ``torch.device("cuda")`` compares unequal to ``cuda:0``.
    """
    resolved = device if isinstance(device, torch.device) else torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    return resolved


def _validate_syncable_param(name: str, param: torch.Tensor) -> None:
    """Reject params neither broadcast protocol can carry, before they are staged for a chunk.

    A DTensor reports a global ``numel()``/``shape`` against a local ``data_ptr()``, so a sharded
    param would be broadcast past the end of its shard. A 0-dim tensor has no last dimension to
    re-view as bytes: it raises inside the vLLM producer with the server already paused mid-reload,
    and gives the SGLang consumer a shape it cannot allocate against.
    """
    if isinstance(param, DTensor):
        raise TypeError(
            f"Parameter {name!r} is a DTensor: the weight-sync client broadcasts raw local storage but "
            f"describes it with the global shape, so a sharded param would be read past its shard. "
            f"Materialize it first (materialize_dtensor / full_tensor)."
        )
    if param.ndim == 0:
        raise ValueError(
            f"Parameter {name!r} is 0-dimensional: the broadcast protocols re-view every tensor as bytes, "
            f"which needs a last dimension, and the consumer rebuilds it from the declared shape. "
            f"Give the parameter an explicit shape (e.g. (1,)) in the model."
        )


def _snapshot_param(weights: torch.Tensor, device: torch.device | None) -> torch.Tensor:
    """A contiguous copy of ``weights`` on ``device`` (the source's own when ``None``), never an alias.

    A snapshot: an in-place mutation of the source after buffering (the next optimizer step)
    cannot revert a staged weight before it is sent. One copy even from a strided
    source, since ``copy_`` reads any layout. It is async on the caller's stream; the flush
    completes it before a producer reads the tensor.
    """
    snapshot = torch.empty_like(weights, memory_format=torch.contiguous_format, device=device)
    snapshot.copy_(weights.detach(), non_blocking=True)
    return snapshot


def snapshot_to_sync_device(
    name: str,
    weights: torch.Tensor,
    *,
    device: torch.device | None,
    buffered_bytes: int,
    budget: int,
    flush: Callable[[], None],
) -> torch.Tensor:
    """A snapshot of ``weights`` on ``device`` (the source's own when ``None``), after ``flush`` has sent the
    pending chunk if ``weights`` would take it past ``budget``.

    The param is checked first, so a param neither protocol carries is refused before it can cut a chunk
    and quiesce an engine.
    """
    _validate_syncable_param(name, weights)
    if buffered_bytes > 0 and buffered_bytes + weights.nbytes > budget:
        flush()
    return _snapshot_param(weights, device)
