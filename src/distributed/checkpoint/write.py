"""The collective half of a checkpoint write: the retain-gated gather, the streamed part writer it
feeds, and the index exchange that names every writer's shards.

Resolving a DTensor is a mesh-wide collective every rank must enter while only the writer pays the
host copy — the asymmetry these primitives exist to make impossible to spell. Kept out of
``save.py`` because the EP saver that ladder dispatches to calls them too.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Iterable, Iterator
from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn

from src.checkpoint.config_export import save_model_config
from src.checkpoint.format import (
    HF_STREAM_PART_PREFIX,
    is_tie_reconcile_key,
    normalize_gathered_state_dict,
    reconcile_tie_word_embeddings,
)
from src.checkpoint.shard_writer import StageShardWriter
from src.distributed.runtime import DeferredRankFailure, materialize_dtensor, resolve_param_tensor
from src.models.patches.gpt_oss_sinks import neutralized_gpt_oss_sinks
from src.models.structure import DECODER_LAYER_LIST_ATTRS, persistent_buffers

# One decoder layer is the chunk a streamed save converts as a unit — see :func:`conversion_chunk_key`.
_LAYER_CHUNK_RE = re.compile(rf"^(.*\.(?:{'|'.join(DECODER_LAYER_LIST_ATTRS)})\.\d+)\.")


def resolve_retained(items: Iterable[tuple[str, torch.Tensor]], *, retain: bool) -> dict[str, torch.Tensor]:
    """Gather every item on EVERY rank; return the host copies only where ``retain``.

    ``resolve_param_tensor`` on a DTensor is a mesh-wide collective, so gating the CALL on ``retain``
    is what hangs a gathered save — the writer enters the all-gather and its peers never do.
    ``materialize_dtensor`` is that collective without the device-to-host copy, so the non-writers
    join without holding N× model bytes per node. Public because the EP gathered save walks its own
    key space and still owes its non-expert tensors this symmetry.
    """
    resolved: dict[str, torch.Tensor] = {}
    for name, tensor in items:
        if retain:
            resolved[name] = resolve_param_tensor(tensor)
        else:
            materialize_dtensor(tensor)
    return resolved


def gather_saveable_tensors(model: nn.Module, *, retain: bool) -> dict[str, torch.Tensor]:
    """Everything a gathered save writes for ``model``: parameters, persistent buffers, GptOss sinks.

    Non-persistent buffers (rotary caches, masks) are recomputed on load and stay out. GptOss under
    FA2 drops ``sinks`` from ``named_parameters()``; they are re-emitted neutralized so the reload
    matches, and ``setdefault`` leaves a live sink alone. Purely local, hence retain-only.
    """
    state_dict = resolve_retained(saveable_items(model), retain=retain)
    if retain:
        for name, tensor in neutralized_gpt_oss_sinks(model).items():
            state_dict.setdefault(name, tensor)
    return state_dict


def conversion_chunk_key(name: str) -> str:
    """The group ``name`` must be save-converted with: its decoder-layer prefix, else ``""``.

    A streamed save cannot hand :func:`~src.checkpoint.format.revert_load_conversions` one tensor at
    a time: a reverse fusion (a vision tower's q/k/v into ``in_proj``, a per-expert hub split) needs
    every source key of that entry together. Conversion entries are declared per layer and never
    straddle two, and everything outside the layer list shares one chunk, so both stay closed.
    """
    match = _LAYER_CHUNK_RE.match(name)
    return match.group(1) if match else ""


def chunked_saveable_tensors(
    model: nn.Module,
    *,
    retain: bool,
    items: Iterable[tuple[str, torch.Tensor]] | None = None,
) -> Iterator[dict[str, torch.Tensor]]:
    """:func:`gather_saveable_tensors`, yielded in conversion-closed chunks instead of one dict.

    The gather stays collective and rank-uniform — the chunk plan is built from NAMES only, so it is
    identical on every rank, and each chunk resolves through the same ``retain`` gate — while the
    writer's host RAM peaks at one decoder layer plus one pending shard rather than the whole model.
    """
    pairs = list(saveable_items(model, items))
    # The sinks FA2 dropped from named_parameters() join the walk in their own layer's chunk rather
    # than a trailing pass, so every rank still builds the same chunk sequence.
    present = {name for name, _tensor in pairs}
    pairs += [(name, t) for name, t in neutralized_gpt_oss_sinks(model).items() if name not in present]
    groups: dict[str, list[tuple[str, torch.Tensor]]] = {}
    for name, tensor in pairs:
        groups.setdefault(conversion_chunk_key(name), []).append((name, tensor))
    for group in groups.values():
        yield resolve_retained(group, retain=retain)


def saveable_items(
    model: nn.Module, items: Iterable[tuple[str, torch.Tensor]] | None = None
) -> Iterable[tuple[str, torch.Tensor]]:
    """``items`` when the caller supplied its own pairs, else the parameter + persistent-buffer walk.

    Public for a caller that must SUBTRACT from that walk before handing it back as ``items`` (the TP
    save drops the hand-sliced params its own trailing gather owns), where re-spelling the walk would
    drift from what the chunker iterates."""
    if items is not None:
        return items
    return itertools.chain(
        ((name, param.data) for name, param in model.named_parameters()),
        persistent_buffers(model),
    )


def stream_gathered_checkpoint(
    model: nn.Module,
    chunks: Iterable[dict[str, torch.Tensor]],
    output_dir: str,
    *,
    is_save_rank: bool,
    max_shard_size: str,
) -> None:
    """Write a gathered save straight to safetensors parts, never holding the whole state dict.

    ``chunks`` is the retain-gated gather (:func:`chunked_saveable_tensors`), driven to exhaustion on
    every rank because the gathers inside are collective while only the save rank receives tensors.
    The artifact matches :func:`~src.checkpoint.format.write_gathered_checkpoint` — same normalization
    (per chunk, which is why the chunks are conversion-closed), tie reconcile, config and index —
    with host RAM bounded by one chunk plus one pending shard. The tied pair is the one thing held
    across the stream, because the reconcile compares both halves.

    A failing write raises through :class:`DeferredRankFailure`, reaching every rank at the collective
    below rather than stranding peers in the next chunk's gather.
    """
    writer = StageShardWriter(output_dir, HF_STREAM_PART_PREFIX, max_shard_size, enabled=is_save_rank)
    guard = DeferredRankFailure(f"gathered checkpoint write to {output_dir}")
    # Held across the stream only where the reconcile below can act on them — an untied config makes
    # it a no-op, and these are two of the largest tensors in the checkpoint.
    keep_tied = getattr(getattr(model, "config", None), "tie_word_embeddings", False)
    tied: dict[str, torch.Tensor] = {}

    def stage(chunk: dict[str, torch.Tensor]) -> None:
        for key, tensor in normalize_gathered_state_dict(model, chunk).items():
            if keep_tied and is_tie_reconcile_key(key):
                tied[key] = tensor
            writer.add(key, tensor)

    for chunk in chunks:
        # The generator is driven on every rank (its gathers are collective); only the writer has a
        # chunk to normalize, and only it can fail the deferred guard on one.
        if is_save_rank:
            guard.run(partial(stage, chunk))
    # Collective. A stream that failed anywhere stops the job here rather than finalizing an index
    # over tensors that were never written.
    guard.reject()

    finalize = DeferredRankFailure(f"gathered checkpoint finalize in {output_dir}")
    if is_save_rank:
        if hasattr(model, "config"):
            # Must precede the config write, which persists the flag this may clear.
            reconcile_tie_word_embeddings(model, tied)
            finalize.run(partial(save_model_config, model, output_dir))
        # The index is the last thing written: the parts are only a checkpoint once it names them.
        finalize.run(writer.close_as_hf_checkpoint)
    finalize.reject()


def exchange_shard_index(
    weight_map: dict[str, str], total_bytes: int, *, contribute: bool = True
) -> tuple[dict[str, str], int]:
    """Fold every rank's ``(weight-map fragment, bytes written)`` into the one merged index.

    The byte count rides along because summing what each rank WROTE is the only correct
    ``total_size``: deriving it from ``model.parameters()`` mixes scopes (a DTensor's ``numel()`` is
    global, EP expert params are already per-rank). ``contribute=False`` is for a rank that wrote
    nothing and still must join the collectives.

    Rank 0 — the only index writer — gathers, merges and broadcasts the single result; an all-gather
    would leave every rank holding one fragment per rank of a map only it needs. The merge VERDICT
    travels with the map, so a collision raises everywhere instead of stranding the peers.
    """
    payload = (weight_map, total_bytes) if contribute else None
    if not (dist.is_available() and dist.is_initialized()):
        return merge_shard_index([payload])

    # Non-None on rank 0 alone — ``gather_object`` requires exactly that, and it is what narrows the
    # merge below to the one rank holding every fragment.
    gathered: list[tuple[dict[str, str], int] | None] | None = (
        [None] * dist.get_world_size() if dist.get_rank() == 0 else None
    )
    dist.gather_object(payload, gathered, dst=0)

    # The exception object itself is the verdict, so every rank raises the identical diagnostic
    # naming the colliding key. ANY exception, not just the collision: one escaping rank 0 here
    # would strand every peer in the next collective until the watchdog fires.
    outcome: list[tuple[dict[str, str], int] | RuntimeError | None] = [None]
    if gathered is not None:
        try:
            outcome[0] = merge_shard_index(gathered)
        except Exception as failure:
            # Re-boxed because the verdict travels by pickle and only its message is load-bearing.
            outcome[0] = (
                failure
                if type(failure) is RuntimeError
                else RuntimeError(
                    f"merging the checkpoint index failed on rank 0: {type(failure).__name__}: {failure}"
                )
            )
    dist.broadcast_object_list(outcome, src=0)

    merged = outcome[0]
    if isinstance(merged, RuntimeError):
        raise merged
    if merged is None:
        raise RuntimeError(
            "the merged checkpoint index never reached this rank: rank 0 broadcast no verdict, so "
            "the index it writes cannot be reconciled with the shards this rank wrote."
        )
    return merged


def merge_shard_index(gathered: list[tuple[dict[str, str], int] | None]) -> tuple[dict[str, str], int]:
    """Fold the gathered fragments into one ``(weight_map, total_size)``, refusing a stray collision.

    The rank-local half of :func:`exchange_shard_index`, which makes the raise world-uniform. A key
    claimed by two DIFFERENT files would resolve to whichever fragment merged last and count its
    bytes twice — a checkpoint that loads without error holding the wrong writer's tensor — and both
    callers are key-disjoint by construction, so a second claim is a real defect.

    Bytes are counted once per FILE: on per-node storage each node's writer emits the same filenames.
    """
    weight_map: dict[str, str] = {}
    counted_files: set[str] = set()
    total_size = 0
    for entry in gathered:
        if not entry:
            continue
        fragment, fragment_bytes = entry
        for key, shard in fragment.items():
            if key in weight_map and weight_map[key] != shard:
                raise RuntimeError(
                    f"checkpoint shards disagree on {key!r}: claimed by both {weight_map[key]!r} and "
                    f"{shard!r}. Every key belongs to exactly one writer, so two claims mean the "
                    f"shard split or the global-name map is wrong."
                )
            weight_map[key] = shard
        files = set(fragment.values())
        if not files & counted_files:
            total_size += fragment_bytes
            counted_files |= files
    return weight_map, total_size
