"""The collective half of a gathered checkpoint write: the retain-gated gather and the streamed writer.

Resolving a DTensor is a mesh-wide collective every rank must enter, while only the writer pays the
host copy.
"""

from __future__ import annotations

import itertools
import logging
import re
from collections.abc import Callable, Iterable, Iterator
from functools import partial

import torch
import torch.nn as nn

from halo.checkpoint.config_export import save_model_config
from halo.checkpoint.writing import (
    StreamingShardWriter,
    is_tie_reconcile_key,
    reconcile_tie_word_embeddings,
    save_dtype_caster,
)
from halo.distributed.tensor_parallel.plain_params import iter_hand_sliced_full
from halo.models.attention.gpt_oss_sinks import neutralized_gpt_oss_sinks
from halo.models.config_access import ties_word_embeddings
from halo.models.module_tree import DECODER_LAYER_LIST_ATTRS
from halo.models.param_roles import persistent_buffers
from halo.runtime.consensus import DeferredRankFailure
from halo.runtime.dtensor import gather_full_tensors
from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)

# One decoder layer is the chunk a streamed save converts as a unit; see :func:`conversion_chunk_key`.
_LAYER_CHUNK_RE = re.compile(rf"^((?:.*\.)?(?:{'|'.join(DECODER_LAYER_LIST_ATTRS)})\.\d+)\.")


def saveable_items(
    model: nn.Module, items: Iterable[tuple[str, torch.Tensor]] | None = None
) -> Iterable[tuple[str, torch.Tensor]]:
    """``items`` when given, else the parameter + persistent-buffer walk a gathered save writes."""
    if items is not None:
        return items
    return itertools.chain(
        ((name, param.data) for name, param in model.named_parameters()),
        persistent_buffers(model),
    )


def conversion_chunk_key(name: str) -> str:
    """The group ``name`` must be save-converted with: its decoder-layer prefix (``model.layers.3``,
    ``layers.3`` on a bare backbone, ``encoder.layer.3`` on a BERT-style encoder), else ``""``.

    :func:`~halo.checkpoint.hub_layout.revert_to_load_layout` cannot take one tensor at a time: a reverse
    fusion (a vision tower's q/k/v into ``in_proj``, a per-expert hub split) needs every source key of
    that entry together. Conversion entries are declared per layer and never straddle two, and
    everything outside the layer list shares one chunk, so both chunks stay closed.
    """
    match = _LAYER_CHUNK_RE.match(name)
    return match.group(1) if match else ""


def chunked_saveable_tensors(
    model: nn.Module,
    *,
    retain: bool,
    items: Iterable[tuple[str, torch.Tensor]] | None = None,
    fold: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> Iterator[dict[str, torch.Tensor]]:
    """COLLECTIVE: :func:`saveable_items` and the GptOss sinks, gathered in conversion-closed chunks and kept
    only where ``retain``.

    The chunk plan is built from names only, so every rank builds the same one; the writer's host RAM
    peaks at one decoder layer plus the shard it is filling and one per write thread rather than the whole
    model. ``fold`` derives each tensor's saved value (an out-of-place LoRA fold) inside its chunk's gather,
    never all up front.
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
        yield gather_full_tensors(group if fold is None else ((name, fold(t)) for name, t in group), retain=retain)


def gather_hand_sliced_params(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    retain: bool = True,
    keep_live_dtype: bool = False,
) -> None:
    """COLLECTIVE over the TP group: write ``model``'s hand-sliced TP params into ``state_dict`` as full CPU
    tensors. ``retain=False`` joins every gather but writes nothing (save-rank gating without N× host copies).

    Cast through the model-derived :func:`save_dtype_caster`, as the callers do for the params they resolve
    themselves: a blanket cast would push a norm / balancing / fp32-pinned tensor that happens to be
    hand-sliced down to bf16 while its gathered twin kept the trained dtype.
    """
    cast = save_dtype_caster(model, keep_live_dtype=keep_live_dtype)
    gathered_count = 0
    for name, full_tensor in iter_hand_sliced_full(model):
        if retain:
            state_dict[name] = cast(name, full_tensor.cpu())
        gathered_count += 1

    if gathered_count > 0 and is_global_main_process():
        logger.info(f"All-gathered {gathered_count} TP-sharded non-DTensor params")


def stream_gathered_checkpoint(
    model: nn.Module,
    chunks: Iterable[dict[str, torch.Tensor]],
    output_dir: str,
    *,
    is_save_rank: bool,
    max_shard_size: str,
    normalize: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]],
) -> tuple[int, int]:
    """COLLECTIVE: write a gathered save straight to safetensors parts, then its config and index.

    ``chunks`` is a retain-gated gather driven to exhaustion on every rank; ``normalize`` brings each
    chunk to its on-disk names and dtypes, which is why the chunks are conversion-closed. Host RAM stays
    at one chunk plus the writer's staged parts (:attr:`StreamingShardWriter.max_resident_bytes`: the shard
    it is filling and one per write thread). A failing write is deferred to a collective, so it raises on
    every rank rather than stranding the peers in the next chunk's gather. Returns ``(tensors, bytes)``
    written, zeros off the save rank.
    """
    # Entered, so the parts are written in the background; the exit waits them out before any raise leaves.
    with StreamingShardWriter(output_dir, max_shard_size, enabled=is_save_rank) as writer:
        guard = DeferredRankFailure(f"gathered checkpoint write to {output_dir}")
        # Held across the stream only where the reconcile below can act on them: an untied config makes
        # it a no-op, and these are two of the largest tensors in the checkpoint.
        keep_tied = getattr(model, "config", None) is not None and ties_word_embeddings(model.config)
        tied: dict[str, torch.Tensor] = {}
        written = 0

        def stage(chunk: dict[str, torch.Tensor]) -> None:
            nonlocal written
            for key, tensor in normalize(chunk).items():
                if keep_tied and is_tie_reconcile_key(key):
                    tied[key] = tensor
                writer.add(key, tensor)
                written += 1

        for chunk in chunks:
            # The generator is driven on every rank (its gathers are collective); only the writer has a
            # chunk to normalize, and only it can fail the deferred guard on one.
            if is_save_rank:
                guard.run(partial(stage, chunk))
            del chunk  # released before the next chunk is gathered
        # Collective. A stream that failed anywhere stops the job here rather than finalizing an index
        # over tensors that were never written.
        guard.reject()

        finalize = DeferredRankFailure(f"gathered checkpoint finalize in {output_dir}")
        total_bytes = 0
        if is_save_rank:
            if hasattr(model, "config"):
                # Must precede the config write, which persists the flag this may clear.
                reconcile_tie_word_embeddings(model, tied)
                finalize.run(partial(save_model_config, model, output_dir))
            # The index is the last thing written: the parts are only a checkpoint once it names them.
            total_bytes = finalize.run(writer.close_as_hf_checkpoint) or 0
        # Over the store: the peers wait out the writer's last part, unbounded on slow storage.
        finalize.store_reject("checkpoint/gathered_finalize")
    return written, total_bytes
