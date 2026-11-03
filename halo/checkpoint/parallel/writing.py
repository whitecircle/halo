"""The collective half of a gathered write: resolving a DTensor is a mesh-wide collective every rank must enter, while
only the writer pays the host copy."""

from __future__ import annotations

import itertools
import logging
import re
from collections.abc import Callable, Iterable, Iterator
from functools import partial

import torch
import torch.nn as nn

from halo.checkpoint.config_export import save_model_config
from halo.checkpoint.external_tables import write_external_tables
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
    """The group ``name`` must be save-converted with: its decoder-layer prefix, else ``""``. A reverse fusion needs
    every source key of its entry together; entries are declared per layer and never straddle two, so both chunks stay
    closed."""
    match = _LAYER_CHUNK_RE.match(name)
    return match.group(1) if match else ""


def chunked_saveable_tensors(
    model: nn.Module,
    *,
    retain: bool,
    items: Iterable[tuple[str, torch.Tensor]] | None = None,
    fold: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> Iterator[dict[str, torch.Tensor]]:
    """COLLECTIVE: :func:`saveable_items` and the GptOss sinks, gathered in conversion-closed chunks, kept only where
    ``retain``. The plan is name-only, so every rank builds the same one, and the writer's RAM peaks at one layer plus
    its staged parts. ``fold`` derives each saved value (an out-of-place LoRA fold) inside its chunk's gather."""
    pairs = list(saveable_items(model, items))
    # The sinks FA2 dropped join the walk in their own layer's chunk, so every rank builds the same chunk sequence.
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
    """COLLECTIVE over the TP group: write hand-sliced TP params into ``state_dict`` as full CPU tensors;
    ``retain=False`` joins every gather but writes nothing. Cast through :func:`save_dtype_caster`, since a blanket
    cast would push a pinned hand-sliced tensor down to bf16 while its gathered twin kept the trained dtype."""
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
    """COLLECTIVE: write a gathered save to safetensors parts, then its config, external tables and index. ``chunks``
    is a retain-gated gather driven on every rank; ``normalize`` brings each chunk to its on-disk names and dtypes. A
    failing write is deferred to a collective, so it raises on every rank. Returns ``(tensors, bytes)``, zeros off the
    save rank."""
    # Entered, so the parts are written in the background; the exit waits them out before any raise leaves.
    with StreamingShardWriter(output_dir, max_shard_size, enabled=is_save_rank) as writer:
        guard = DeferredRankFailure(f"gathered checkpoint write to {output_dir}")
        # Held only where the reconcile below can act on them; these are two of the largest tensors in the checkpoint.
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
            # Driven on every rank, as its gathers are collective; only the writer has a chunk to normalize.
            if is_save_rank:
                guard.run(partial(stage, chunk))
            del chunk  # released before the next chunk is gathered
        # Collective, so a stream failed anywhere stops the job before an index over tensors never written.
        guard.reject()

        finalize = DeferredRankFailure(f"gathered checkpoint finalize in {output_dir}")
        total_bytes = 0
        if is_save_rank:
            if hasattr(model, "config"):
                # Must precede the config write, which persists the flag this may clear.
                reconcile_tie_word_embeddings(model, tied)
                finalize.run(partial(save_model_config, model, output_dir))
            finalize.run(partial(write_external_tables, model, writer))
            # The index is the last thing written: the parts are only a checkpoint once it names them.
            total_bytes = finalize.run(writer.close_as_hf_checkpoint) or 0
        # Over the store: the peers wait out the writer's last part, unbounded on slow storage.
        finalize.store_reject("checkpoint/gathered_finalize")
    return written, total_bytes
