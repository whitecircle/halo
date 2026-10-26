"""The saver ladder: :func:`save_checkpoint` writes the tokenizer, then runs the saver the ladder picks.

Every rung is rank-uniform, so all ranks pick the same saver, and ``None`` falls through to
``Trainer.save_model``. PP precedes the full-model rungs because a stage is a partial model, and EP
precedes CP so an EP+CP save gathers its experts. The FSDP2, CP and TP rungs share one streamed body.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from functools import partial

import torch
import torch.nn as nn

from halo.checkpoint.parallel.adapters import save_expert_lora_adapters, save_peft_adapters
from halo.checkpoint.parallel.context import CheckpointContext
from halo.checkpoint.parallel.ep import save_ep_gathered_checkpoint, save_ep_sharded_checkpoint
from halo.checkpoint.parallel.pp import save_pp_checkpoint
from halo.checkpoint.parallel.writing import (
    chunked_saveable_tensors,
    gather_hand_sliced_params,
    saveable_items,
    stream_gathered_checkpoint,
)
from halo.checkpoint.writing import normalize_gathered_state_dict
from halo.distributed.tensor_parallel.plain_params import hand_sliced_suffixes
from halo.models.unwrapping import find_peft_model, unwrap_toolkit_wrappers
from halo.runtime.collectives import barrier_on_exit
from halo.runtime.consensus import DeferredRankFailure
from halo.runtime.filesystem import fs_aware_makedirs
from halo.runtime.ranks import get_global_world_size

logger = logging.getLogger(__name__)

# A saver always handles the save it is given; "nobody owns this" is the ladder's ``None``.
CheckpointSaver = Callable[[CheckpointContext, str], None]


def save_checkpoint(ctx: CheckpointContext, output_dir: str) -> bool:
    """COLLECTIVE: save the model through the ladder's saver; ``False`` when none owns the save."""
    saver = select_checkpoint_saver(ctx)
    if saver is None:
        return False
    save_tokenizer(ctx, output_dir)
    saver(ctx, output_dir)
    return True


def save_tokenizer(ctx: CheckpointContext, output_dir: str) -> None:
    """COLLECTIVE: write the run's tokenizer into ``output_dir`` on the save rank; a failure raises on every rank.

    Written before the weights: a save stopped once its index names the weight files is already a
    checkpoint that loads, and it must carry the tokenizer it loads with.
    """
    guard = DeferredRankFailure(f"tokenizer save to {output_dir}")
    if ctx.is_save_rank and ctx.tokenizer is not None:
        guard.run(partial(ctx.tokenizer.save_pretrained, output_dir))
    guard.reject()


def select_checkpoint_saver(ctx: CheckpointContext) -> CheckpointSaver | None:
    """The first rung that owns the save — PEFT adapters, PP, expert-LoRA adapters, EP sharded or
    gathered, CP, TP, FSDP2 — or ``None``."""
    config = ctx.parallelism_config
    # A merged-base save takes the mode's saver even with adapters, and under accelerate FSDP v1 the
    # adapters are flat-param shards whose layout the base Trainer's save owns.
    if find_peft_model(ctx.model) is not None and not ctx.accelerate_manages_fsdp and not ctx.merges_expert_lora:
        return save_peft_adapters
    if config.is_pp_mode:
        return save_pp_checkpoint
    if ctx.has_expert_lora and not ctx.merges_expert_lora:
        return save_expert_lora_adapters
    if (config.has_ep_group and config.is_tp_mode) or ctx.has_ep_layers:
        # One rank has nothing to shard: its sharded save is the gathered one.
        sharded = ctx.save_sharded_ep and get_global_world_size() > 1
        return save_ep_sharded_checkpoint if sharded else save_ep_gathered_checkpoint
    if config.is_cp_mode:
        # The wrapper owns the CP key remap; without it there is no CP save to make, only the base one.
        return save_cp_checkpoint if ctx.cp_wrapper is not None else None
    if config.is_tp_mode:
        return save_tp_checkpoint
    if ctx.toolkit_fsdp2:
        return save_fsdp2_checkpoint
    return None


def _streamed_save(
    ctx: CheckpointContext,
    output_dir: str,
    model: nn.Module,
    chunks: Iterable[dict[str, torch.Tensor]],
    saved: str,
) -> None:
    """COLLECTIVE: the FSDP2/CP/TP body — stream ``chunks`` through the gathered writer.

    Fenced because one rank writes while all must reach the trailing barrier: an ENOSPC would
    otherwise strand the peers.
    """
    with barrier_on_exit():
        stream_gathered_checkpoint(
            model,
            chunks,
            output_dir,
            is_save_rank=ctx.is_save_rank,
            max_shard_size=ctx.max_shard_size,
            normalize=partial(normalize_gathered_state_dict, model, keep_live_dtype=ctx.training_checkpoint),
        )
        if ctx.is_save_rank:
            logger.info(saved)


def save_fsdp2_checkpoint(ctx: CheckpointContext, output_dir: str) -> None:
    """COLLECTIVE: mixin-managed FSDP2, every DTensor param gathered whole."""
    chunks = chunked_saveable_tensors(ctx.model, retain=ctx.is_save_rank)
    _streamed_save(ctx, output_dir, ctx.model, chunks, f"Saved FSDP2 model to {output_dir}")


def save_cp_checkpoint(ctx: CheckpointContext, output_dir: str) -> None:
    """COLLECTIVE: CP without EP, its items the CP wrapper's ``state_dict()``, whose keys it remaps.

    That dict carries the persistent buffers under the remapped keys and holds references only, so the
    streamed gather still resolves them chunk by chunk; the sinks come off the unwrapped model. Each tensor
    is written under its first key only, the key set the FSDP2 and TP savers' ``named_parameters`` walk
    writes: safetensors refuses two keys over one storage (a tied ``lm_head.weight``).
    """
    cp_wrapper = ctx.cp_wrapper
    inner = unwrap_toolkit_wrappers(cp_wrapper)
    # Under keep_vars a tied parameter is one object under both keys.
    first_keys: dict[int, tuple[str, torch.Tensor]] = {}
    for name, value in cp_wrapper.state_dict(keep_vars=True).items():
        first_keys.setdefault(id(value), (name, value.data if isinstance(value, nn.Parameter) else value))
    chunks = chunked_saveable_tensors(inner, retain=ctx.is_save_rank, items=first_keys.values())
    _streamed_save(ctx, output_dir, inner, chunks, f"Saved CP model to {output_dir}")


def _tp_chunks(
    model: nn.Module, model_to_save: nn.Module, *, retain: bool, keep_live_dtype: bool
) -> Iterator[dict[str, torch.Tensor]]:
    """COLLECTIVE: a TP save's gathered chunks — the DTensor walk, then the hand-sliced params whole.

    The plain TP-sharded params (GptOss sinks) are not DTensors, so ``full_tensor()`` never
    reconstructs them; they leave the walk, which would flush this rank's ``[heads/tp]`` slice before
    the gathered tensor arrives under the same key, and get a collective pass and a chunk of their own.
    """
    hand_sliced = hand_sliced_suffixes(model)
    items = ((name, tensor) for name, tensor in saveable_items(model_to_save) if not name.endswith(hand_sliced))
    yield from chunked_saveable_tensors(model_to_save, retain=retain, items=items)
    sharded_plain: dict = {}
    gather_hand_sliced_params(model, sharded_plain, retain=retain, keep_live_dtype=keep_live_dtype)
    yield sharded_plain


def save_tp_checkpoint(ctx: CheckpointContext, output_dir: str) -> None:
    """COLLECTIVE: TP / TP+DP, every TP rank reconstructing the full tensors and the FS-aware save rank
    writing them, not ``tp_rank == 0``, which several TP groups per node would race on."""
    fs_aware_makedirs(output_dir)
    if ctx.is_save_rank:
        logger.info(f"Saving TP model to {output_dir}...")
    model = unwrap_toolkit_wrappers(ctx.model)
    chunks = _tp_chunks(ctx.model, model, retain=ctx.is_save_rank, keep_live_dtype=ctx.training_checkpoint)
    _streamed_save(ctx, output_dir, model, chunks, f"✓ TP model saved to {output_dir}")
