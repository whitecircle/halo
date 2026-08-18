"""The closed hub-conversion op vocabulary and the walker that applies it to checkpoint keys.

Some hub checkpoints keep the vendor's tensor namespace and layouts, bridged only by transformers'
declarative conversion entries inside ``from_pretrained``. The lazy loaders read safetensors
directly, so those entries are translated here into ops the materializer applies after its ranged
read, and :func:`convert_disk_keys` walks the ordered steps over the checkpoint's keys.

The vocabulary is closed: each op acts on one declared dim, so the expert planner can assert
orthogonality to the expert shard axis and keep its ranged per-expert reads. An entry using any
other transformers op raises at translation rather than loading wrong tensors. Which family declares
which entries is injected by the EP package's hub-conversion seam.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

import torch
from transformers.core_model_loading import (
    Chunk,
    Concatenate,
    Interleave,
    PermuteForRope,
    WeightConverter,
    WeightRenaming,
)


@dataclass(frozen=True)
class Deinterleave:
    """Undo an element-interleaved pair layout along ``dim`` into contiguous halves.

    Must stay the exact inverse of transformers' ``Interleave`` op (reshape ``[.., D, ..]`` to
    ``[.., D/2, 2, ..]``, transpose the pair axes, reshape back).
    """

    dim: int

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        shape = list(tensor.shape)
        shape[self.dim : self.dim + 1] = [shape[self.dim] // 2, 2]
        return tensor.reshape(shape).transpose(self.dim, self.dim + 1).reshape(tensor.shape).contiguous()


@dataclass(frozen=True)
class TakeChunk:
    """One equal chunk of the tensor along ``dim``: the per-target half of a fan-out entry."""

    dim: int
    index: int
    count: int

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.chunk(tensor, self.count, dim=self.dim)[self.index].contiguous()


@dataclass(frozen=True)
class PermuteRope:
    """Rows of a complex-RoPE (interleaved-pair) head layout, mapped to the split-halves layout.

    Must stay equal to transformers' ``PermuteForRope`` (``n_heads`` read off the config at
    translation), for a ``[rows, ...]`` weight or a ``[rows]`` bias alike. Always acts on dim 0.
    """

    n_heads: int
    inverse: bool = False

    @property
    def dim(self) -> int:
        return 0

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        half = tensor.shape[0] // self.n_heads // 2
        head_shape = (2, half) if self.inverse else (half, 2)
        permuted = tensor.reshape(self.n_heads, *head_shape, *tensor.shape[1:]).transpose(1, 2)
        return permuted.reshape(tensor.shape).contiguous()


@dataclass(frozen=True)
class Concat:
    """Fan-in of a multi-source conversion entry: the lazy mirror of transformers' ``Concatenate``.

    Leads the op pipeline of the entry's first source; ``siblings`` are its further disk keys in
    declared order, each read through the plan's own ranged expert slice and concatenated along
    ``dim``. The materializer resolves those reads; every other op is a per-tensor callable.
    """

    dim: int
    siblings: tuple[str, ...]


TensorOp = Deinterleave | TakeChunk | PermuteRope
# A converted disk key's targets: (canonical key, op pipeline) per target.
ConvertTargets = tuple[tuple[str, tuple[Concat | TensorOp, ...]], ...]


@dataclass(frozen=True)
class Rename:
    pattern: re.Pattern
    replacement: str
    # Module path (relative to ``base_model_prefix``) whose keys alone this step sees; "" means all.
    scope: str = ""


@dataclass(frozen=True)
class Convert:
    # One pattern per declared source; more than one is a fan-in whose first source carries a Concat.
    patterns: tuple[re.Pattern, ...]
    # Per canonical target: (replacement for the matched span, op pipeline).
    targets: tuple[tuple[str, tuple[TensorOp, ...]], ...]
    concat_dim: int | None = None
    scope: str = ""


def _compile_source(pattern: str) -> re.Pattern:
    """A conversion source pattern as transformers matches it: a regex with ``*`` meaning ``.*``."""
    return re.compile(pattern.replace("*", ".*"))


def translate_renaming(entry: WeightRenaming, scope: str = "") -> Rename:
    """One transformers ``WeightRenaming`` → :class:`Rename`."""
    return Rename(_compile_source(entry.source_patterns[0]), entry.target_patterns[0], scope)


def translate_converter(entry: WeightConverter, config=None, scope: str = "") -> Convert:
    """One transformers ``WeightConverter`` → the closed-vocabulary equivalent, or raise.

    Supported pipelines: ``[Interleave]`` (one target), ``[Chunk]``, ``[Interleave, Chunk]``,
    ``[Chunk, PermuteForRope]`` (one target per chunk, the permutation landing on the targets it
    names), and ``[Concatenate]`` over several sources into one. Anything else needs machinery this
    module does not have, and is refused rather than materialized incorrectly.
    """
    ops = tuple(entry.operations)
    targets = tuple(entry.target_patterns)
    sources = tuple(_compile_source(pattern) for pattern in entry.source_patterns)

    def _kind(op) -> str:
        if isinstance(op, Interleave):
            return "interleave" if not op.inverse else "interleave_inverse"
        if isinstance(op, PermuteForRope):
            return "permute_rope"
        if isinstance(op, Concatenate):
            return "concat"
        return "chunk" if isinstance(op, Chunk) else type(op).__name__

    kinds = tuple(_kind(op) for op in ops)
    concat_dim = None
    if len(sources) > 1 and (kinds != ("concat",) or len(targets) != 1):
        raise ValueError(
            f"Unsupported multi-source conversion entry {entry.source_patterns} with ops {list(kinds)} → "
            f"{len(targets)} target(s): the lazy loaders express a multi-source entry only as one "
            f"Concatenate into one target."
        )
    if kinds == ("concat",) and len(sources) > 1:
        per_target = ((targets[0], ()),)
        concat_dim = ops[0].dim
    elif kinds == ("interleave",) and len(targets) == 1:
        per_target = ((targets[0], (Deinterleave(ops[0].dim),)),)
    elif kinds == ("chunk",) and len(targets) > 1:
        per_target = tuple((t, (TakeChunk(ops[0].dim, i, len(targets)),)) for i, t in enumerate(targets))
    elif kinds == ("interleave", "chunk") and len(targets) > 1:
        per_target = tuple(
            (t, (Deinterleave(ops[0].dim), TakeChunk(ops[1].dim, i, len(targets)))) for i, t in enumerate(targets)
        )
    elif kinds == ("chunk", "permute_rope") and len(targets) > 1:
        permute = _permute_rope(ops[1], config, entry)
        per_target = tuple(
            (t, (TakeChunk(ops[0].dim, i, len(targets)), *((permute,) if _permutes_target(ops[1], t) else ())))
            for i, t in enumerate(targets)
        )
    else:
        raise ValueError(
            f"Unsupported conversion ops {list(kinds)} → {len(targets)} target(s) for "
            f"{entry.source_patterns[0]!r}: the lazy loaders support Interleave/Chunk/PermuteForRope "
            f"pipelines and a multi-source Concatenate only."
        )
    return Convert(sources, per_target, concat_dim, scope)


def _permute_rope(op: PermuteForRope, config, entry: WeightConverter) -> PermuteRope:
    """The head count transformers' op reads at conversion: ``config[.subconfig_key]``'s
    ``num_attention_heads``, with its own default of 1 where the config lacks the field."""
    if config is None or op.permute_layer_names is None:
        raise ValueError(
            f"PermuteForRope for {entry.source_patterns[0]!r} needs the model config for its head "
            f"count and the target names it permutes; got config={config is not None}, "
            f"permute_layer_names={op.permute_layer_names!r}."
        )
    source = getattr(config, op.subconfig_key, config) if op.subconfig_key else config
    return PermuteRope(getattr(source, "num_attention_heads", 1), inverse=op.inverse)


def _permutes_target(op: PermuteForRope, target: str) -> bool:
    return any(name in target for name in op.permute_layer_names)


def _scoped(key: str, scope: str, base_prefix: str) -> tuple[str, str] | None:
    """``(consumed prefix, remainder)`` a scoped step matches against, or ``None`` outside its scope.

    transformers' ``_scoped_match``: a sub-model's entries see only the keys under its module path,
    tried with the base-model prefix first and then bare (a backbone checkpoint loaded into a task
    wrapper); the root's entries see every key whole.
    """
    if not scope:
        return "", key
    for prefix in (f"{base_prefix}{scope}.", f"{scope}."):
        if key.startswith(prefix):
            return prefix, key[len(prefix) :]
    return None


def convert_disk_keys(
    disk_keys: list[str], steps: tuple[Rename | Convert, ...], base_prefix: str = ""
) -> dict[str, ConvertTargets]:
    """Map each disk key to its canonical target(s) with ops; keys the steps do not change are absent.

    Walks the steps in declaration order: a rename substitutes at every occurrence in the key as it
    flows, the first matching converter claims the key and fans it out, and renames declared after
    that converter still apply to the produced targets (the table's ordering comments rely on it). A
    scoped step sees only the keys under its sub-model path. transformers reaches the same targets by
    a different route, so equality with ``from_pretrained`` holds per declared entry set rather than
    by construction, and is pinned bitwise by the per-family lazy-load round-trip tests.

    A multi-source converter is a fan-in: members are grouped by the target they produce and the
    first source carries the group as a leading :class:`Concat`, so the siblings are absent from the
    result. A partially present group raises rather than materializing a truncated tensor.
    """
    converted: dict[str, ConvertTargets] = {}
    # (step position, target key) -> {source index: disk key}: the members of one fan-in group.
    fan_in: dict[tuple[int, str], dict[int, str]] = defaultdict(dict)
    for key in disk_keys:
        current = key
        targets: ConvertTargets | None = None
        fan_in_member = False
        for position, step in enumerate(steps):
            scoped = _scoped(current, step.scope, base_prefix)
            if scoped is None:
                continue
            prefix, remainder = scoped
            if isinstance(step, Rename):
                current = prefix + step.pattern.sub(step.replacement, remainder)
                continue
            source = next((i for i, pattern in enumerate(step.patterns) if pattern.search(remainder)), None)
            if source is None:
                continue
            pattern = step.patterns[source]
            targets = tuple(
                (
                    _apply_tail_renames(
                        prefix + pattern.sub(replacement, remainder), steps[position + 1 :], base_prefix
                    ),
                    ops,
                )
                for replacement, ops in step.targets
            )
            if step.concat_dim is not None:
                fan_in_member = True
                members = fan_in[(position, targets[0][0])]
                if source in members:
                    raise ValueError(
                        f"Checkpoint keys {members[source]!r} and {key!r} both match source "
                        f"{pattern.pattern!r} of the multi-source conversion entry producing "
                        f"{targets[0][0]!r}; only one of them can feed it."
                    )
                members[source] = key
            break
        if fan_in_member:
            continue
        if targets is not None:
            converted[key] = targets
        elif current != key:
            converted[key] = ((current, ()),)

    for (position, target), members in fan_in.items():
        step = steps[position]
        expected = list(range(len(step.patterns)))
        if sorted(members) != expected:
            raise ValueError(
                f"Multi-source conversion entry {[p.pattern for p in step.patterns]} → {target!r}: the "
                f"checkpoint carries source(s) {sorted(members)} of {expected}, so the concatenation "
                f"would materialize a truncated tensor."
            )
        ((_, ops),) = step.targets
        converted[members[0]] = ((target, (Concat(step.concat_dim, tuple(members[i] for i in expected[1:])), *ops)),)
    return converted


def _apply_tail_renames(key: str, tail: tuple[Rename | Convert, ...], base_prefix: str) -> str:
    for step in tail:
        if isinstance(step, Rename) and (scoped := _scoped(key, step.scope, base_prefix)) is not None:
            prefix, remainder = scoped
            key = prefix + step.pattern.sub(step.replacement, remainder)
    return key
