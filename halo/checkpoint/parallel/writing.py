"""The collective half of a gathered write: resolving a DTensor is a mesh-wide collective every rank must enter, while
only the writer pays the host copy."""

from __future__ import annotations

import itertools
import logging
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn

from halo.checkpoint.config_export import save_model_config
from halo.checkpoint.external_tables import write_external_tables
from halo.checkpoint.format import STREAM_PART_PREFIX, stripe_writer_part_prefix
from halo.checkpoint.writing import (
    StreamingShardWriter,
    is_tie_reconcile_key,
    merge_streamed_parts,
    publish_hf_parts,
    reconcile_tie_word_embeddings,
    resolve_checkpoint_writers_per_node,
    validate_checkpoint_writers_per_node,
)
from halo.distributed.tensor_parallel.plain_params import iter_hand_sliced_full
from halo.models.attention.gpt_oss_sinks import neutralized_gpt_oss_sinks
from halo.models.config_access import ties_word_embeddings
from halo.models.module_tree import DECODER_LAYER_LIST_ATTRS
from halo.models.mtp.head import mtp_trunk_copy_keys
from halo.models.param_roles import persistent_buffers
from halo.runtime.collectives import combine_across_ranks
from halo.runtime.consensus import DeferredRankFailure, barrier_on_exit
from halo.runtime.dtensor import gather_full_tensors
from halo.runtime.filesystem import is_output_shared_filesystem
from halo.runtime.ranks import (
    get_local_rank,
    get_local_world_size,
    get_node_rank,
    get_num_nodes,
    is_global_main_process,
)

logger = logging.getLogger(__name__)

_LAYER_CHUNK_RE = re.compile(rf"^((?:.*\.)?(?:{'|'.join(DECODER_LAYER_LIST_ATTRS)})\.\d+)\.")


@dataclass(frozen=True)
class WriterStripe:
    """The ranks that write one gathered save and the chunks each owns. ``index`` is this rank's place among the
    ``count`` writers, ``None`` off them. Every rank holds the whole chunk after its gather, so ownership only decides
    who pays the host copy and the write."""

    index: int | None
    count: int

    @property
    def is_writer(self) -> bool:
        return self.index is not None

    def owns(self, chunk: int) -> bool:
        """Chunk 0 (the replicated tensors, tied embeddings and tables) is the first writer's; each later chunk, one
        decoder layer, goes round-robin."""
        return self.index is not None and self.index == (max(chunk - 1, 0) % self.count)


def resolve_writer_stripe(*, is_save_rank: bool) -> WriterStripe:
    """The stripe of a gathered save. A shared output filesystem takes ``HALO_CHECKPOINT_WRITERS_PER_NODE`` writers
    per node, the save rank first; per-node storage keeps the save rank alone, as each node's directory must hold the
    whole checkpoint."""
    if not is_output_shared_filesystem():
        return WriterStripe(0 if is_save_rank else None, 1)
    per_node = min(validate_checkpoint_writers_per_node(resolve_checkpoint_writers_per_node()), get_local_world_size())
    count = get_num_nodes() * per_node
    if is_save_rank:
        return WriterStripe(0, count)
    local_rank = get_local_rank()
    return WriterStripe(get_node_rank() * per_node + local_rank if local_rank < per_node else None, count)


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


def gather_hand_sliced_params(model: nn.Module, *, retain: bool) -> dict[str, torch.Tensor]:
    """COLLECTIVE over the TP group: the hand-sliced TP params as full CPU tensors at their live dtype; ``retain=False``
    joins every gather and keeps nothing."""
    state_dict: dict[str, torch.Tensor] = {}
    gathered_count = 0
    for name, full_tensor in iter_hand_sliced_full(model):
        if retain:
            state_dict[name] = full_tensor.cpu()
        gathered_count += 1

    if gathered_count > 0 and is_global_main_process():
        logger.info(f"All-gathered {gathered_count} TP-sharded non-DTensor params")
    return state_dict


def stream_gathered_checkpoint(
    model: nn.Module,
    chunks: Iterable[dict[str, torch.Tensor]],
    output_dir: str,
    *,
    is_save_rank: bool,
    max_shard_size: str,
    normalize: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]],
    stripe: WriterStripe | None = None,
) -> tuple[int, int]:
    """COLLECTIVE: write a gathered save to safetensors parts, then its config, external tables and index. ``chunks``
    is a retain-gated gather driven on every rank; ``normalize`` brings each chunk to its on-disk names and dtypes. With
    a ``stripe`` of several writers, each retains and writes the chunks it owns (``chunks`` retains per ``stripe.owns``)
    under its own part prefix, and the save rank, the stripe's first writer, renames the parts and writes the one
    index; without one the save rank alone writes. A failing write is deferred to a collective, so it raises on every
    rank. Returns ``(tensors, bytes)``, the whole checkpoint's on the save rank."""
    stripe = stripe or WriterStripe(0 if is_save_rank else None, 1)
    striped = stripe.count > 1
    prefix = stripe_writer_part_prefix(stripe.index) if striped and stripe.is_writer else STREAM_PART_PREFIX
    # Entered, so the parts are written in the background; the exit waits them out before any raise leaves.
    with StreamingShardWriter(output_dir, max_shard_size, enabled=stripe.is_writer, filename_prefix=prefix) as writer:
        guard = DeferredRankFailure(f"gathered checkpoint write to {output_dir}")
        # Held only where the reconcile below can act on them; these are two of the largest tensors in the checkpoint.
        keep_tied = getattr(model, "config", None) is not None and ties_word_embeddings(model.config)
        tied: dict[str, torch.Tensor] = {}
        trunk_copies = mtp_trunk_copy_keys(model)
        written = 0

        def stage(chunk: dict[str, torch.Tensor]) -> None:
            nonlocal written
            for key, tensor in normalize(chunk).items():
                if keep_tied and is_tie_reconcile_key(key):
                    tied[key] = tensor
                writer.add(key, tensor)
                written += 1
                for copy_key in trunk_copies.get(key, ()):
                    # A clone, as a part cannot store two keys over one storage.
                    writer.add(copy_key, tensor.clone())
                    written += 1

        for chunk in chunks:
            # Driven on every rank, as its gathers are collective; only a writer holds a chunk to normalize.
            if stripe.is_writer and chunk:
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
        if striped:
            parts = finalize.run(writer.close_parts) if stripe.is_writer else None
            # Over the store: the peers wait out the slowest writer's last part, unbounded on slow storage.
            finalize.store_reject("checkpoint/gathered_finalize")
            merged = combine_across_ranks(parts, merge_streamed_parts, what="the checkpoint index")
            with barrier_on_exit(f"Writing the checkpoint index to {output_dir}"):
                if is_save_rank:
                    total_bytes = publish_hf_parts(output_dir, merged)
                    written = len(merged.weight_map)
        else:
            if is_save_rank:
                # The index is the last thing written: the parts are only a checkpoint once it names them.
                total_bytes = finalize.run(writer.close_as_hf_checkpoint) or 0
            # Over the store: the peers wait out the writer's last part, unbounded on slow storage.
            finalize.store_reject("checkpoint/gathered_finalize")
    return written, total_bytes
