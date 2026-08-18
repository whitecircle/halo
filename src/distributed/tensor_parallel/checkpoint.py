"""TP model checkpointing: every TP rank reconstructs its full tensors, one save rank writes a
single HF-standard safetensors checkpoint.

One global rank writes on a shared FS, one rank per node on a per-node FS (``fs_aware_save_rank``).
Not ``tp_rank == 0``: several TP groups per node would then race on identical files.
"""

from __future__ import annotations

import logging

import torch.nn as nn

from src.checkpoint.format import DEFAULT_MAX_SHARD_SIZE
from src.distributed.checkpoint.write import (
    chunked_saveable_tensors,
    saveable_items,
    stream_gathered_checkpoint,
)
from src.distributed.runtime import (
    barrier_on_exit,
    fs_aware_makedirs,
    fs_aware_save_rank,
)
from src.distributed.tensor_parallel.state_dict import (
    gather_tp_sharded_non_dtensor_params,
    tp_sharded_non_dtensor_suffixes,
)
from src.models.structure import unwrap_model

logger = logging.getLogger(__name__)


def save_tp_model(
    model: nn.Module,
    output_dir: str,
    tokenizer=None,
    max_shard_size: str = DEFAULT_MAX_SHARD_SIZE,
) -> None:
    """Save a TP model checkpoint: reconstruct full tensors from the DTensors (the collective
    ``full_tensor()``, which every TP-mesh rank must enter) and stream them to disk through
    :func:`~src.distributed.checkpoint.write.stream_gathered_checkpoint`, the writer the FSDP2
    and CP gathered saves share (config reconciliation, save dtype + hub expert layout, HF-standard
    multi-file shards + index).
    """
    fs_aware_makedirs(output_dir)

    is_save_rank = fs_aware_save_rank()

    if is_save_rank:
        logger.info(f"Saving TP model to {output_dir}...")

    model_to_save = unwrap_model(model)

    with barrier_on_exit():
        stream_gathered_checkpoint(
            model_to_save,
            _tp_chunks(model, model_to_save, retain=is_save_rank),
            output_dir,
            is_save_rank=is_save_rank,
            max_shard_size=max_shard_size,
        )
        if is_save_rank:
            logger.info(f"✓ TP model saved to {output_dir}")

            if tokenizer is not None:
                tokenizer.save_pretrained(output_dir)
                logger.info(f"✓ Tokenizer saved to {output_dir}")


def _tp_chunks(model: nn.Module, model_to_save: nn.Module, *, retain: bool):
    """The gathered chunks of a TP save: the DTensor walk, then the hand-sliced params.

    The plain TP-sharded params (GptOss sinks) are not DTensors, so ``full_tensor()`` never
    reconstructs them and they need their own collective pass. No family mapping names them, so a
    chunk of their own needs no hub conversion. Collective on every TP-group rank; only the writer
    retains.

    They are excluded from the parameter walk, which emits the live parameter (this rank's
    ``[heads/tp]`` slice) that the streamed writer would flush before the gathered tensor arrives
    under the same key. The exclusion reads the same ``_tp_sharded_non_dtensor`` registry the gather
    iterates.
    """
    hand_sliced = tp_sharded_non_dtensor_suffixes(model)
    items = ((name, tensor) for name, tensor in saveable_items(model_to_save) if not name.endswith(hand_sliced))
    yield from chunked_saveable_tensors(model_to_save, retain=retain, items=items)
    sharded_plain: dict = {}
    gather_tp_sharded_non_dtensor_params(model, sharded_plain, retain=retain)
    yield sharded_plain
