"""The closed hub-conversion op vocabulary and a checkpoint key plan translated into it for the lazy loaders. Each op
acts on one declared dim, so the expert planner keeps its ranged reads; an entry using any other op raises at
translation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers.core_model_loading import (
    Chunk,
    Concatenate,
    Interleave,
    PermuteForRope,
    WeightConverter,
    rename_source_key,
)

from halo.model_loading.checkpoint_keys import CheckpointKeyPlan, ConversionGroup, is_expert_bank_merge


@dataclass(frozen=True)
class Deinterleave:
    """Undo an element-interleaved pair layout along ``dim``; the exact inverse of transformers' ``Interleave``."""

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
    """Complex-RoPE (interleaved-pair) rows to split-halves on dim 0, equal to transformers' ``PermuteForRope``."""

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
    """Fan-in of a multi-source entry, mirroring transformers' ``Concatenate``: leads the first source's pipeline, with
    ``siblings`` its further disk keys in declared order. The materializer resolves their reads; other ops are
    callables."""

    dim: int
    siblings: tuple[str, ...]


_TensorOp = Deinterleave | TakeChunk | PermuteRope
# A converted disk key's targets: (canonical key, op pipeline) per target.
ConvertTargets = tuple[tuple[str, tuple[Concat | _TensorOp, ...]], ...]


def _op_kind(op) -> str:
    if isinstance(op, Interleave):
        return "interleave" if not op.inverse else "interleave_inverse"
    if isinstance(op, PermuteForRope):
        return "permute_rope"
    if isinstance(op, Concatenate):
        return "concat"
    return "chunk" if isinstance(op, Chunk) else type(op).__name__


def _permute_rope(op: PermuteForRope, config, entry: WeightConverter) -> PermuteRope:
    """The head count transformers' op reads: ``config[.subconfig_key]``'s ``num_attention_heads``, default 1."""
    if config is None or op.permute_layer_names is None:
        raise ValueError(
            f"PermuteForRope for {entry.source_patterns[0]!r} needs the model config for its head "
            f"count and the target names it permutes; got config={config is not None}, "
            f"permute_layer_names={op.permute_layer_names!r}."
        )
    source = getattr(config, op.subconfig_key, config) if op.subconfig_key else config
    return PermuteRope(getattr(source, "num_attention_heads", 1), inverse=op.inverse)


def translate_converter(entry: WeightConverter, config=None) -> tuple[tuple[tuple[_TensorOp, ...], ...], int | None]:
    """A ``WeightConverter`` → its op pipeline per target and a fan-in's concat dim (``None`` for one source).
    Supported: ``[Interleave]``, ``[Chunk]``, ``[Interleave, Chunk]``, ``[Chunk, PermuteForRope]``, and
    ``[Concatenate]`` over several sources; anything else is refused rather than materialized incorrectly."""
    ops = tuple(entry.operations)
    targets = tuple(entry.target_patterns)
    multi_source = len(entry.source_patterns) > 1
    kinds = tuple(_op_kind(op) for op in ops)
    concat_dim = None
    if multi_source and (kinds != ("concat",) or len(targets) != 1):
        raise ValueError(
            f"Unsupported multi-source conversion entry {entry.source_patterns} with ops {list(kinds)} → "
            f"{len(targets)} target(s): the lazy loaders express a multi-source entry only as one "
            f"Concatenate into one target."
        )
    if kinds == ("concat",) and multi_source:
        pipelines = ((),)
        concat_dim = ops[0].dim
    elif kinds == ("interleave",) and len(targets) == 1:
        pipelines = ((Deinterleave(ops[0].dim),),)
    elif kinds == ("chunk",) and len(targets) > 1:
        pipelines = tuple((TakeChunk(ops[0].dim, i, len(targets)),) for i in range(len(targets)))
    elif kinds == ("interleave", "chunk") and len(targets) > 1:
        pipelines = tuple(
            (Deinterleave(ops[0].dim), TakeChunk(ops[1].dim, i, len(targets))) for i in range(len(targets))
        )
    elif kinds == ("chunk", "permute_rope") and len(targets) > 1:
        permute = _permute_rope(ops[1], config, entry)
        pipelines = tuple(
            (
                TakeChunk(ops[0].dim, i, len(targets)),
                *((permute,) if any(name in t for name in ops[1].permute_layer_names) else ()),
            )
            for i, t in enumerate(targets)
        )
    else:
        raise ValueError(
            f"Unsupported conversion ops {list(kinds)} → {len(targets)} target(s) for "
            f"{entry.source_patterns[0]!r}: the lazy loaders support Interleave/Chunk/PermuteForRope "
            f"pipelines and a multi-source Concatenate only."
        )
    return pipelines, concat_dim


def _group_targets(group: ConversionGroup, config) -> dict[str, ConvertTargets]:
    """The disk keys of one live converter group → their targets with ops; a fan-in's on its first source."""
    pipelines, concat_dim = translate_converter(group.converter, config)
    if concat_dim is None:
        targets = tuple(zip(group.targets, pipelines, strict=True))
        return {key: targets for key, _ in group.parts}
    sources = group.converter.source_patterns
    by_source: dict[str, str] = {}
    for key, pattern in group.parts:
        if pattern in by_source:
            raise ValueError(
                f"Checkpoint keys {by_source[pattern]!r} and {key!r} both match source {pattern!r} of the "
                f"multi-source conversion entry producing {group.targets[0]!r}; only one of them can feed it."
            )
        by_source[pattern] = key
    if len(by_source) != len(sources):
        raise ValueError(
            f"Multi-source conversion entry {sources} → {group.targets[0]!r}: the checkpoint carries "
            f"source(s) {sorted(map(sources.index, by_source))} of {list(range(len(sources)))}, so the "
            f"concatenation would materialize a truncated tensor."
        )
    first, *siblings = (by_source[pattern] for pattern in sources)
    return {first: ((group.targets[0], (Concat(concat_dim, tuple(siblings)),)),)}


def lazy_key_mapping(plan: CheckpointKeyPlan, config) -> tuple[dict[str, str], dict[str, ConvertTargets]]:
    """Every disk key → the model key it feeds, and every converted key's targets with ops. An expert-bank merge's keys
    keep their per-expert spelling for the ``ExpertFuser``; a partial or doubled fan-in raises."""
    disk_to_model = dict(plan.renamed_by_key)
    converted: dict[str, ConvertTargets] = {}
    for group in plan.groups:
        if is_expert_bank_merge(group.converter):
            unknown = {type(op).__name__ for op in group.converter.operations} - {"MergeModulelist", "Concatenate"}
            if unknown:
                raise ValueError(
                    f"per-expert conversion entry for {group.targets[0]!r}, {group.converter.source_patterns} "
                    f"carries {sorted(unknown)} — the ExpertFuser only reproduces plain "
                    "MergeModulelist/Concatenate merges, so skipping it would silently "
                    "drop a real conversion"
                )
            for key, _ in group.parts:
                disk_to_model[key], _ = rename_source_key(key, plan.renamings, [], plan.base_model_prefix, plan.live)
            continue
        disk_to_model.update((key, group.targets[0]) for key, _ in group.parts)
        if any(target in plan.live for target in group.targets):
            converted.update(_group_targets(group, config))
    return disk_to_model, converted
