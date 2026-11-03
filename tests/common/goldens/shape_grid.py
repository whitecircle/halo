"""The shape grid the process-group goldens run over: every accepted shape up to :data:`STAGE_WIDTH_LIMIT` ranks, at
every NVLink domain. A golden stores each shape's digest, plus a few literal shapes in full so a failure reads as a
diff."""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import itertools
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from types import MappingProxyType

from halo.distributed.parallelism_config import ParallelismConfig
from halo.distributed.parallelism_rules import AXES, SUPPORTED_AXIS_SETS
from tests.common.parallelism import make_parallelism_config

STAGE_WIDTH_LIMIT = 32
PIPELINE_SIZES = (1, 2)
PRODUCTION_WORLDS = (64, 72, 128, 144, 256)
# (NVLink domain, OS-node width): an 8-GPU HGX node, and an NVL72 rack of 4-GPU trays.
PRODUCTION_DOMAINS = ((8, 8), (72, 4))
DIGEST_HEX_CHARS = 16

# Allowlist axis -> the ParallelismConfig field sizing it; _accepted checks it against the config.
AXIS_FIELDS = MappingProxyType({a.name: a.size_field for a in AXES})


@dataclasses.dataclass(frozen=True)
class LayoutShape:
    """One grid point; the fields are the ``ParallelismConfig`` arguments it sets."""

    world_size: int
    nvlink_domain_size: int
    gpus_per_node: int
    pp_size: int = 1
    ep_size: int = 1
    expert_tp_size: int = 1
    tp_size: int = 1
    cp_size: int = 1
    ep_scope: str = "node"

    @property
    def ep_group_size(self) -> int:
        return self.ep_size * self.expert_tp_size

    @property
    def axes(self) -> frozenset[str]:
        return frozenset(axis for axis, field in AXIS_FIELDS.items() if getattr(self, field) > 1)

    @property
    def key(self) -> str:
        """``w<world> d<domain> n<node>``, each active axis with its size, and the EP scope for a multi-rank group:
        ``w16 d8 n8 pp2 ep4 node``."""
        parts = [f"w{self.world_size}", f"d{self.nvlink_domain_size}", f"n{self.gpus_per_node}"]
        parts += [f"{axis}{getattr(self, AXIS_FIELDS[axis])}" for axis in AXIS_FIELDS if axis in self.axes]
        if self.ep_group_size > 1:
            parts.append(self.ep_scope)
        return " ".join(parts)

    def config_as_rank(self, rank: int) -> ParallelismConfig:
        return make_parallelism_config(rank=rank, **dataclasses.asdict(self))


def _divisors(value: int) -> list[int]:
    return [d for d in range(1, value + 1) if value % d == 0]


def _shapes_on(world_size: int, nvlink_domain_size: int, gpus_per_node: int, pp_size: int) -> Iterator[LayoutShape]:
    """Every allowlisted axis assignment over the stage width's divisors, under both EP scopes where a group spans
    several ranks."""
    sizes = [d for d in _divisors(world_size // pp_size) if d > 1]
    for axes in sorted(sorted(axis_set, key=list(AXIS_FIELDS).index) for axis_set in SUPPORTED_AXIS_SETS):
        if ("pp" in axes) != (pp_size > 1):
            continue
        free_axes = [axis for axis in axes if axis != "pp"]
        for assignment in itertools.product(sizes, repeat=len(free_axes)):
            fields = {AXIS_FIELDS[axis]: size for axis, size in zip(free_axes, assignment, strict=True)}
            shape = LayoutShape(world_size, nvlink_domain_size, gpus_per_node, pp_size=pp_size, **fields)
            yield shape
            if shape.ep_group_size > 1:
                yield dataclasses.replace(shape, ep_scope="global")


def _candidate_shapes() -> Iterator[LayoutShape]:
    for stage_width, pp_size in itertools.product(range(1, STAGE_WIDTH_LIMIT + 1), PIPELINE_SIZES):
        for domain in _divisors(stage_width):
            yield from _shapes_on(pp_size * stage_width, domain, domain, pp_size)
    for world_size, (domain, gpus_per_node), pp_size in itertools.product(
        PRODUCTION_WORLDS, PRODUCTION_DOMAINS, PIPELINE_SIZES
    ):
        if world_size % pp_size == 0:
            yield from _shapes_on(world_size, domain, gpus_per_node, pp_size)


def _accepted(shape: LayoutShape) -> bool:
    try:
        config = shape.config_as_rank(0)
    except ValueError:
        return False
    if config.active_axes != shape.axes:
        raise AssertionError(
            f"AXIS_FIELDS disagrees with ParallelismConfig on {shape.key}: the config turns on "
            f"{sorted(config.active_axes)}, the shape {sorted(shape.axes)}."
        )
    return True


@functools.cache
def layout_shapes() -> tuple[LayoutShape, ...]:
    """Every distinct candidate ``ParallelismConfig`` accepts, in generation order."""
    return tuple(shape for shape in dict.fromkeys(_candidate_shapes()) if _accepted(shape))


def _canonical_json(record: dict) -> str:
    return json.dumps(record, sort_keys=True)


def record_digest(record: dict) -> str:
    return hashlib.sha256(_canonical_json(record).encode()).hexdigest()[:DIGEST_HEX_CHARS]


def _json_object_lines(name: str, rows: dict[str, str]) -> str:
    """``"name": {...}`` with one already-encoded row a line."""
    body = ",\n".join(f"    {json.dumps(key)}: {value}" for key, value in rows.items())
    return f'  "{name}": {{\n{body}\n  }}'


@dataclasses.dataclass(frozen=True)
class GridGolden:
    """A golden over the grid at ``path``: ``record`` maps a shape to JSON-native values, and ``literal_shapes`` are
    stored in full."""

    path: Path
    record: Callable[[LayoutShape], dict]
    literal_shapes: tuple[LayoutShape, ...]

    @property
    def name(self) -> str:
        return self.path.stem

    def render(self) -> str:
        """The literal shapes' records, then every shape's digest in grid order."""
        records = {shape: self.record(shape) for shape in layout_shapes()}
        literal = {shape.key: _canonical_json(records[shape]) for shape in self.literal_shapes}
        digests = {shape.key: json.dumps(record_digest(shape_record)) for shape, shape_record in records.items()}
        return "{\n" + _json_object_lines("literal", literal) + ",\n" + _json_object_lines("digest", digests) + "\n}\n"
