"""The closed hub-conversion op vocabulary, the entry kinds left to other owners, and the translation of
checkpoint keys into module keys.

transformers' declarative conversion entries become ops the materializer applies after its ranged read;
each acts on one declared dim, so the expert planner can keep its per-expert ranged reads. An entry using
any other transformers op raises at translation rather than loading wrong tensors.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers.core_model_loading import (
    Chunk,
    Concatenate,
    Interleave,
    PermuteForRope,
    PrefixChange,
    WeightConverter,
    WeightRenaming,
)

# The attribute a multimodal wrapper nests its text model under; a text-only load of its checkpoint drops it.
_LANGUAGE_MODEL_PREFIX = "language_model."
# A conversion source addressing one tensor per module-list index (``…experts.*.w1.weight``).
_PER_INDEX_SOURCE = re.compile(r"\.\*\.")


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
class RenameStep:
    pattern: re.Pattern
    replacement: str
    # Module path (relative to ``base_model_prefix``) whose keys alone this step sees; "" means all.
    scope: str = ""


@dataclass(frozen=True)
class ConvertStep:
    # One pattern per declared source; more than one is a fan-in whose first source carries a Concat.
    patterns: tuple[re.Pattern, ...]
    # Per canonical target: (replacement for the matched span, op pipeline).
    targets: tuple[tuple[str, tuple[TensorOp, ...]], ...]
    concat_dim: int | None = None
    scope: str = ""


def _op_kind(op) -> str:
    if isinstance(op, Interleave):
        return "interleave" if not op.inverse else "interleave_inverse"
    if isinstance(op, PermuteForRope):
        return "permute_rope"
    if isinstance(op, Concatenate):
        return "concat"
    return "chunk" if isinstance(op, Chunk) else type(op).__name__


def _compile_source(pattern: str) -> re.Pattern:
    """A conversion source pattern as transformers matches it: a regex with ``*`` meaning ``.*``."""
    return re.compile(pattern.replace("*", ".*"))


def is_expert_bank_merge(entry) -> bool:
    """Whether ``entry`` fuses one-tensor-per-expert checkpoint keys into an expert bank: a converter whose
    every source carries a per-index wildcard, which in transformers' mappings is only ever an expert list.

    That layout is owned at both ends by the expert code, never by a generic conversion replay: the lazy
    loaders' ``ExpertFuser`` reads it, each family's ``ExpertLayout`` writes it.
    """
    return isinstance(entry, WeightConverter) and all(_PER_INDEX_SOURCE.search(p) for p in entry.source_patterns)


def is_sub_model_prefix_change(conversion) -> bool:
    """Whether ``conversion`` strips the leading level of a sub-model's own keys: a ``PrefixChange`` scoped to
    the sub-model, such as the ``vision_model`` level of a SigLIP tower (Command A+, LFM2-VL), which the hub
    index and both engines' loaders keep and the module tree drops."""
    return (
        isinstance(conversion, PrefixChange)
        and bool(conversion.scope_prefix)
        and bool(conversion.prefix_to_remove)
        and not conversion.model_prefix
    )


def translate_renaming(entry: WeightRenaming, scope: str = "") -> RenameStep:
    """One transformers ``WeightRenaming`` → :class:`RenameStep`."""
    return RenameStep(_compile_source(entry.source_patterns[0]), entry.target_patterns[0], scope)


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


def translate_converter(entry: WeightConverter, config=None, scope: str = "") -> ConvertStep:
    """One transformers ``WeightConverter`` → the closed-vocabulary equivalent, or raise.

    Supported pipelines: ``[Interleave]`` (one target), ``[Chunk]``, ``[Interleave, Chunk]``,
    ``[Chunk, PermuteForRope]`` (one target per chunk, the permutation landing on the targets it
    names), and ``[Concatenate]`` over several sources into one. Anything else needs machinery this
    module does not have, and is refused rather than materialized incorrectly.
    """
    ops = tuple(entry.operations)
    targets = tuple(entry.target_patterns)
    sources = tuple(_compile_source(pattern) for pattern in entry.source_patterns)
    kinds = tuple(_op_kind(op) for op in ops)
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
            (
                t,
                (
                    TakeChunk(ops[0].dim, i, len(targets)),
                    *((permute,) if any(name in t for name in ops[1].permute_layer_names) else ()),
                ),
            )
            for i, t in enumerate(targets)
        )
    else:
        raise ValueError(
            f"Unsupported conversion ops {list(kinds)} → {len(targets)} target(s) for "
            f"{entry.source_patterns[0]!r}: the lazy loaders support Interleave/Chunk/PermuteForRope "
            f"pipelines and a multi-source Concatenate only."
        )
    return ConvertStep(sources, per_target, concat_dim, scope)


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


def _apply_tail_renames(key: str, tail: tuple[RenameStep | ConvertStep, ...], base_prefix: str) -> str:
    for step in tail:
        if isinstance(step, RenameStep) and (scoped := _scoped(key, step.scope, base_prefix)) is not None:
            prefix, remainder = scoped
            key = prefix + step.pattern.sub(step.replacement, remainder)
    return key


def convert_disk_keys(
    disk_keys: list[str], steps: tuple[RenameStep | ConvertStep, ...], base_prefix: str = ""
) -> dict[str, ConvertTargets]:
    """Each disk key's canonical target(s) with ops; keys the steps do not change are absent.

    Steps walk in declaration order: a rename substitutes as the key flows, the first matching converter
    claims and fans it out, and later renames still apply to its targets; a scoped step sees only its
    sub-model's keys. Equality with ``from_pretrained`` (which takes another route) is pinned bitwise by the
    per-family lazy-load round-trip tests. A multi-source converter's first source carries its group as a
    leading :class:`Concat`; a partially present group raises rather than load a truncated tensor.
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
            if isinstance(step, RenameStep):
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


def _align_key(key: str, model_keys: set[str], base_prefix: str, hub_renames, wrapperless: bool) -> str:
    """One checkpoint key in the model's state-dict spelling, each step running only while it is unresolved:
    a nested ``language_model.model.`` text model moved under the wrapper (Mistral3), the base-model prefix
    added then stripped, the hub → module renames, then a ``language_model.`` segment dropped for a VLM
    checkpoint loaded text-only.

    Every step keeps a candidate only when it is a model key, except that the segment drop also keeps one for
    a ``wrapperless`` model: a per-expert key is never a model key, and the fuser locates the fused parameter
    it feeds from the namespace this step gives it.
    """
    new_key = key
    # Mistral3's hub spelling of its text model's own ``model.`` level under the wrapper attribute.
    nested = f"{_LANGUAGE_MODEL_PREFIX}model."
    if new_key not in model_keys and new_key.startswith(nested):
        candidate = f"{base_prefix}{_LANGUAGE_MODEL_PREFIX}{new_key[len(nested) :]}"
        if candidate in model_keys:
            new_key = candidate

    if new_key not in model_keys and base_prefix and not new_key.startswith(base_prefix):
        prefixed = base_prefix + new_key
        if prefixed in model_keys:
            new_key = prefixed

    if new_key not in model_keys and base_prefix and new_key.startswith(base_prefix):
        stripped = new_key[len(base_prefix) :]
        if stripped in model_keys:
            new_key = stripped

    if new_key not in model_keys:
        for hub_spelling, module_spelling in hub_renames:
            candidate = new_key.replace(hub_spelling, module_spelling, 1)
            if candidate in model_keys:
                new_key = candidate
                break

    if new_key not in model_keys and f".{_LANGUAGE_MODEL_PREFIX}" in new_key:
        stripped = new_key.replace(f".{_LANGUAGE_MODEL_PREFIX}", ".", 1)
        if stripped in model_keys or wrapperless:
            new_key = stripped
    if new_key not in model_keys and new_key.startswith(_LANGUAGE_MODEL_PREFIX):
        stripped = new_key[len(_LANGUAGE_MODEL_PREFIX) :]
        if stripped in model_keys or wrapperless:
            new_key = stripped

    return new_key


def build_key_mapping(
    model: nn.Module,
    disk_keys: list[str],
    steps: tuple[RenameStep | ConvertStep, ...] | None = None,
    hub_renames: tuple[tuple[str, str], ...] = (),
) -> tuple[dict[str, str], dict[str, ConvertTargets]]:
    """``disk_key → model_key`` (a converted key's first target), and every converted key's targets with ops.

    The ``steps`` (``None`` for a canonical checkpoint) run first; every target then goes through
    :func:`_align_key`. An unmapped key is skipped by the planner, so an unapplied rename loses a whole module.
    A converted key whose targets all miss the model while the key itself resolves stays unconverted:
    transformers' sources are not vendor-anchored (deepseek_v4's ``\\.norm\\.`` → ``.kv_norm.`` also matches
    the canonical ``model.norm.weight``), and its own loader undoes such a rename too.
    """
    prefix = getattr(model, "base_model_prefix", "")
    base_prefix = f"{prefix}." if prefix else ""
    model_keys = set(model.state_dict().keys())
    wrapperless = not any(f".{_LANGUAGE_MODEL_PREFIX}" in f".{key}" for key in model_keys)
    if steps is None:
        return {key: _align_key(key, model_keys, base_prefix, hub_renames, wrapperless) for key in disk_keys}, {}

    converted = convert_disk_keys(disk_keys, steps, base_prefix)

    disk_to_model: dict[str, str] = {}
    fanout: dict[str, ConvertTargets] = {}
    for key in disk_keys:
        plain = _align_key(key, model_keys, base_prefix, hub_renames, wrapperless)
        aligned = None
        targets = converted.get(key)
        if targets is not None:
            aligned = tuple(
                (_align_key(target, model_keys, base_prefix, hub_renames, wrapperless), ops) for target, ops in targets
            )
            if plain in model_keys and all(target not in model_keys for target, _ in aligned):
                aligned = None
        if aligned is None:
            disk_to_model[key] = plain
        else:
            disk_to_model[key] = aligned[0][0]
            fanout[key] = aligned
    return disk_to_model, fanout
