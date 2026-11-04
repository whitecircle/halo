"""Per-case tensor records of a harness run and their bit-for-bit comparison: dtype, shape and bytes, so ``-0.0``
differs from ``0.0``. A case whose body raised saves what it raised, so a change in what raises is a difference too."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import torch
from torch.distributed.tensor import DTensor

from tests.common.equivalence.revisions import RAISED, compare_outputs, raised_description

RECORD_SUFFIX = ".pt"
GRADIENT_PREFIX = "grad/"

Tensors = dict[str, torch.Tensor]
Orders = dict[str, list[str]]


def _plain(tensor: torch.Tensor) -> torch.Tensor:
    local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
    return local.detach().to("cpu", copy=True).contiguous()


def record_case(output: Path, case: str, rank: int, body: Callable[[], tuple[Tensors, Orders]]) -> None:
    """Save ``body``'s ``(tensors, orders)`` as ``rank``'s record of ``case``, or what it raised."""
    try:
        tensors, orders = body()
        record = {
            "tensors": {name: _plain(tensor) for name, tensor in tensors.items()},
            "orders": {name: list(order) for name, order in orders.items()},
        }
    except Exception as error:
        record = {RAISED: raised_description(error, output)}
    path = output / case / f"rank{rank}{RECORD_SUFFIX}"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(record, path)


def case_of(record: str) -> str:
    """A record path without its ``rank<r>`` leaf."""
    return record.rpartition("/")[0]


def load_record(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def record_paths(root: Path) -> dict[str, Path]:
    return {str(path.relative_to(root).with_suffix("")): path for path in sorted(root.rglob(f"*{RECORD_SUFFIX}"))}


def bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    return torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8))


def max_abs_difference(a: torch.Tensor, b: torch.Tensor) -> float:
    """The largest elementwise ``|a - b|`` in fp64, ``inf`` when shapes differ; NaN propagates."""
    if a.shape != b.shape:
        return float("inf")
    if a.numel() == 0:
        return 0.0
    return (a.double() - b.double()).abs().max().item()


def _order_delta(name: str, base: list[str], head: list[str]) -> str:
    first = next(
        (i for i, (left, right) in enumerate(zip(base, head, strict=False)) if left != right),
        min(len(base), len(head)),
    )
    left = base[first] if first < len(base) else "<end>"
    right = head[first] if first < len(head) else "<end>"
    return f"orders/{name}[{first}]: {left!r} != {right!r} ({len(base)} vs {len(head)} entries)"


def _tensor_delta(name: str, base: torch.Tensor, head: torch.Tensor) -> str:
    if base.dtype != head.dtype or base.shape != head.shape:
        return f"tensors/{name}: {base.dtype}{list(base.shape)} != {head.dtype}{list(head.shape)}"
    if base.is_floating_point():
        return f"tensors/{name}: bits differ, max |delta| {max_abs_difference(base, head):.3e}"
    return f"tensors/{name}: {int((base != head).sum())} of {base.numel()} values differ"


def record_deltas(base: Mapping, head: Mapping) -> list[str]:
    if RAISED in base or RAISED in head:
        return [] if base.get(RAISED) == head.get(RAISED) else [f"{RAISED}: {base.get(RAISED)} != {head.get(RAISED)}"]
    deltas = []
    for kind in ("orders", "tensors"):
        for name in sorted(base[kind].keys() | head[kind].keys()):
            if name not in head[kind]:
                deltas.append(f"{kind}/{name}: only in base")
            elif name not in base[kind]:
                deltas.append(f"{kind}/{name}: only in head")
            elif kind == "orders" and base[kind][name] != head[kind][name]:
                deltas.append(_order_delta(name, base[kind][name], head[kind][name]))
            elif kind == "tensors" and not bitwise_equal(base[kind][name], head[kind][name]):
                deltas.append(_tensor_delta(name, base[kind][name], head[kind][name]))
    return deltas


def compare_records(base_root: Path, head_root: Path) -> list[dict]:
    """:func:`~tests.common.equivalence.revisions.compare_outputs` over two outputs' records, bit for bit."""
    return compare_outputs(
        record_paths(base_root),
        record_paths(head_root),
        lambda base, head: record_deltas(load_record(base), load_record(head)),
    )


def raised_records(root: Path) -> dict[str, dict]:
    raised = {}
    for record, path in record_paths(root).items():
        loaded = load_record(path)
        if RAISED in loaded:
            raised[record] = loaded[RAISED]
    return raised
