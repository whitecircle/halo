"""Model-weight resume: :func:`load_resume_weights` reloads a checkpoint into the live model for the active mode, or
leaves the reload to the base Trainer's loader."""

from __future__ import annotations

import itertools
import logging
from collections.abc import Iterable
from functools import partial
from typing import TYPE_CHECKING

import torch
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
from torch.distributed.tensor import DTensor
from transformers import PreTrainedModel

from halo.checkpoint.adapters import (
    has_adapter_weight_file,
    missing_resume_adapter_reason,
    resume_adapter_dir,
    resume_adapter_on_own_weights_reason,
    unmarked_merged_checkpoint_reason,
)
from halo.checkpoint.hub_layout import ModuleLayoutView
from halo.checkpoint.parallel.adapters import restore_adapters
from halo.checkpoint.parallel.pp import restore_pp_stage_weights
from halo.checkpoint.parallel.reading import joined_streaming_reader, reject_ep_sharded_resume
from halo.checkpoint.reading import (
    has_whole_model_weight_file,
    is_ep_sharded_checkpoint,
    load_full_state_dict,
    resolve_checkpoint_weights,
)
from halo.checkpoint.weights_source import built_from_checkpoint, describe_weights_source, weights_read_from
from halo.distributed.expert_parallel.moe_layer.expert_lora import model_has_expert_lora
from halo.distributed.fsdp import reshard_fsdp2_modules
from halo.distributed.tensor_parallel.plain_params import hand_sliced_dims
from halo.log import KEY_PREVIEW_COUNT
from halo.models.introspection.param_roles import persistent_buffers
from halo.models.introspection.unwrapping import find_peft_model, unwrap_framework_wrappers, unwrap_toolkit_wrappers
from halo.runtime.collectives import barrier, broadcast_from_rank0
from halo.runtime.consensus import DeferredRankFailure, rank0_verdict, rank_consensus, run_on_rank0_raising_everywhere
from halo.runtime.dtensor import copy_full_tensor
from halo.runtime.ranks import is_global_main_process

if TYPE_CHECKING:
    from halo.checkpoint.parallel.context import CheckpointLoadContext

logger = logging.getLogger(__name__)


def built_from_checkpoint_across_ranks(model, checkpoint: str) -> bool:
    """COLLECTIVE. Rank 0's verdict, as ``realpath`` resolves per node and a per-rank answer splits the world."""
    return broadcast_from_rank0(built_from_checkpoint(weights_read_from(model), checkpoint))


def resume_numel_coverage(model, checkpoint_keys: set[str]) -> tuple[bool, set[str], int, int]:
    """``(coverage_ok, unmatched_names, matched_numel, total_numel)``: whether ``checkpoint_keys`` covers half of the
    live persistent tensors by LOGICAL element count, so another expert namespace (few keys, most bytes) fails and a
    DTensor shard counts in full. Reads named_parameters/named_buffers, as ``state_dict()`` reshards FSDP2 on the
    calling rank."""
    model = unwrap_framework_wrappers(model)
    live = {
        name: tensor.shape.numel()
        for name, tensor in itertools.chain(model.named_parameters(), persistent_buffers(model))
    }
    matched_numel = sum(numel for name, numel in live.items() if name in checkpoint_keys)
    total_numel = sum(live.values())
    unmatched = {name for name in live if name not in checkpoint_keys}
    # Under the floor is never a legitimate resume; smaller gaps (tied lm_head, task heads) are real.
    return matched_numel >= total_numel * 0.5, unmatched, matched_numel, total_numel


def _coverage_refusal(mode: str, checkpoint: str, *, where: str, cause: str) -> str:
    return (
        f"{mode} checkpoint resume from {checkpoint}: fewer than half of the live model's PARAMETERS match "
        f"the checkpoint's keys{where} — resume would silently continue from base weights. The checkpoint "
        f"was written for a different model or wrapper layout{cause}."
    )


def _report_unmatched_coverage(
    mode: str, checkpoint: str, unmatched: set[str], matched_numel: int, total_numel: int
) -> None:
    """Warn about live tensors left at init above the coverage floor, where gaps are legitimate (a tied lm_head)."""
    if not unmatched:
        return
    logger.warning(
        f"{mode} resume: {len(unmatched)} live tensors ({total_numel - matched_numel:,} of {total_numel:,} "
        f"parameters) are absent from {checkpoint} and keep their construction values (first few: "
        f"{sorted(unmatched)[:KEY_PREVIEW_COUNT]})."
    )


def _resume_onto_constructed_model(
    ctx: CheckpointLoadContext, checkpoint: str, model, *, for_best_model: bool
) -> None:
    """EP/CP: base weights only load at construction (fused / wrapped modules cannot ingest HF keys), so only adapters
    reload. A marked merged checkpoint resumes its adapters onto a model built from the BASE; any other checkpoint
    shipping base weights must be what the model was built from. Rank 0 probes all inputs in one broadcast, so refusals
    fire everywhere."""

    def probe() -> tuple[bool, bool, str | None]:
        ships_base_weights = bool(resolve_checkpoint_weights(checkpoint).key_set()) or is_ep_sharded_checkpoint(
            checkpoint
        )
        return (
            built_from_checkpoint(weights_read_from(model), checkpoint),
            ships_base_weights,
            resume_adapter_dir(checkpoint),
        )

    # A failed read degrading to no base weights would take the adapter-only path and silently continue on base weights.
    built_from_ckpt, ships_base_weights, merged_resume_adapter = rank0_verdict(
        probe,
        f"Reading the key set at {checkpoint}, which decides whether it ships base weights (a torn index or safetensors "
        "header: repair or re-save the checkpoint, then resume)",
    )
    if ships_base_weights and for_best_model:
        raise ValueError(
            f"load_best_model_at_end cannot reload {checkpoint} under EP/CP full fine-tune — base weights only load "
            "at construction, so the export would silently carry the LAST weights. Export the best checkpoint "
            "directly instead."
        )
    if merged_resume_adapter is not None:
        if built_from_ckpt:
            raise ValueError(resume_adapter_on_own_weights_reason(checkpoint))
        if restore_adapters(merged_resume_adapter, model, is_cp_mode=ctx.parallelism_config.is_cp_mode) is None:
            raise RuntimeError(missing_resume_adapter_reason(checkpoint))
    else:
        if ships_base_weights and not built_from_ckpt:
            raise ValueError(
                "EP/CP resume requires the model to be constructed FROM the checkpoint: "
                f"{describe_weights_source(weights_read_from(model))}, not '{checkpoint}' — the run would silently "
                "continue on those weights. Launch via the training scripts (they repoint model_name_or_path at "
                "the checkpoint), or pass a model loaded from the checkpoint directory."
            )
        restored = restore_adapters(checkpoint, model, is_cp_mode=ctx.parallelism_config.is_cp_mode)
        unwrapped = unwrap_framework_wrappers(model)
        trains_adapters = find_peft_model(unwrapped) is not None or model_has_expert_lora(unwrapped)
        if ships_base_weights and restored is None and trains_adapters:
            # Only a merged save ships base weights from an adapter run.
            raise ValueError(unmarked_merged_checkpoint_reason(checkpoint))
    if is_global_main_process():
        mode = "+".join(
            m for m in [("EP" if ctx.has_ep_layers else ""), ("CP" if ctx.parallelism_config.is_cp_mode else "")] if m
        )
        logger.info(
            f"Skipping checkpoint base-weight reload for {mode} mode (model already loaded via "
            f"load_distributed_model). Adapters (if any) and trainer state are restored from {checkpoint}."
        )


def _rank0_module_layout(model, state_dict: dict, checkpoint: str) -> dict:
    """COLLECTIVE. Rank 0's checkpoint dict under live names, consumed as it converts; a failure raises everywhere."""
    converted: dict = {}
    run_on_rank0_raising_everywhere(
        lambda: converted.update(
            ModuleLayoutView(unwrap_framework_wrappers(model), list(state_dict)).items(state_dict.pop)
        ),
        lambda e: f"FSDP2 resume from {checkpoint}: mapping the checkpoint onto the live model's names failed: {e}",
    )
    return converted


def _load_fsdp2(ctx: CheckpointLoadContext, checkpoint: str, model, *, for_best_model: bool) -> bool:
    """Load whole weights into DTensor params through ``set_model_state_dict``, which plain ``load_state_dict()``
    cannot do. False when the checkpoint has no readable weights."""
    # Best-model loads run after the final eval left the tree unsharded; every branch must write the sharded DTensors.
    reshard_fsdp2_modules(model)
    if not broadcast_from_rank0(has_whole_model_weight_file(checkpoint)):
        # PEFT runs save adapter-only checkpoints; route through the DTensor-aware restorer.
        if broadcast_from_rank0(has_adapter_weight_file(checkpoint)):
            restore_adapters(checkpoint, model, is_cp_mode=ctx.parallelism_config.is_cp_mode)
            barrier()
            return True
        logger.warning(
            f"No model weights found at {checkpoint} on global rank 0, "
            f"falling back to standard checkpoint loading (all ranks)"
        )
        return False

    # A model constructed from this checkpoint holds these weights, so skip re-reading a 100B+ state dict; a best-model
    # load must still read, since the live weights trained past it.
    if not for_best_model and built_from_checkpoint_across_ranks(model, checkpoint):
        if is_global_main_process():
            logger.info(f"FSDP2 resume: model was constructed from {checkpoint}; skipping re-load.")
        return True

    # Every rank must call in though only rank 0's dict is used, so agree on readability first or a rank-0 raise
    # strands peers in set_model_state_dict.
    reject_ep_sharded_resume(checkpoint)
    read_ok = True
    state_dict: dict = {}
    if is_global_main_process():
        try:
            # None = no weight file the loader recognizes; treated as unreadable, as on the TP path.
            state_dict = load_full_state_dict(checkpoint, device="cpu") or {}
            read_ok = bool(state_dict)
            logger.info(f"Loading FSDP2 checkpoint from {checkpoint} ({len(state_dict)} keys)")
        except Exception as e:
            logger.warning(f"Torn/unreadable model checkpoint at {checkpoint}: {e}")
            read_ok = False
    if not broadcast_from_rank0(read_ok):
        # The base loader re-reads and raises on every rank, failing loud.
        return False
    state_dict = _rank0_module_layout(model, state_dict, checkpoint)

    # strict=False would no-op keys missing the live FQNs and resume BASE weights. Decided on rank 0 off the key sets,
    # since only rank 0 sees the real _IncompatibleKeys, which report a wholesale miss as `unexpected` with
    # `missing_keys` empty.
    coverage_ok = True
    if is_global_main_process():
        coverage_ok, unmatched, matched_numel, total_numel = resume_numel_coverage(model, set(state_dict))
        if coverage_ok:
            _report_unmatched_coverage("FSDP2", checkpoint, unmatched, matched_numel, total_numel)
    if not broadcast_from_rank0(coverage_ok):
        raise RuntimeError(
            _coverage_refusal("FSDP2", checkpoint, where="", cause=" (a different expert namespace, most likely)")
        )

    options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True, strict=False)
    set_model_state_dict(model, state_dict, options=options)
    del state_dict
    if is_global_main_process():
        logger.info(f"✓ FSDP2 checkpoint loaded from {checkpoint}")
    barrier()
    return True


def _copy_into_live(
    tensors: Iterable[tuple[str, torch.Tensor]], live: dict, hand_sliced: dict, tp_size: int, tp_rank: int
) -> None:
    """Write each tensor into this rank's live one: a DTensor takes its placements' shard, a hand-sliced TP param this
    ``tp_rank``'s chunk, anything else the whole."""
    with torch.no_grad():
        for name, value in tensors:
            target = live[name]
            dim = next((d for suffix, d in hand_sliced.items() if name.endswith(suffix)), None)
            if dim is not None and not isinstance(target, DTensor):
                value = value.chunk(tp_size, dim=dim)[tp_rank]
            copy_full_tensor(target, value, src_data_rank=None)
            del value


def _load_streamed(ctx: CheckpointLoadContext, checkpoint: str, model, *, for_best_model: bool) -> bool:
    """COLLECTIVE. Load a gathered checkpoint into a model each rank holds whole or TP-sharded, each copy
    collective-free with ``src_data_rank=None``; False when rank 0 finds no readable weights. TP+DP's strided 2-D
    placement does not invert, so it only resumes a model built FROM the checkpoint. Every verdict is joined before any
    write, a failed copy after."""
    mode = "TP" if ctx.parallelism_config.is_tp_mode else "Unsharded"
    reject_ep_sharded_resume(checkpoint)
    readable = False
    if is_global_main_process():
        try:
            readable = bool(resolve_checkpoint_weights(checkpoint).key_set())
        except Exception as e:
            logger.warning(f"Torn/unreadable model checkpoint at {checkpoint}: {e}")
    if not broadcast_from_rank0(readable):
        logger.warning(
            f"No readable model weights (model.safetensors[.index.json] / pytorch_model.bin) found at {checkpoint} "
            "on global rank 0, falling back to standard checkpoint loading (all ranks)"
        )
        return False

    # The training scripts repoint model_name_or_path at the checkpoint, so a model built from it holds these weights;
    # a best-model load must still read, since the live weights trained past it.
    if not for_best_model and built_from_checkpoint_across_ranks(model, checkpoint):
        if is_global_main_process():
            logger.info(f"{mode} resume: model was constructed from {checkpoint}; skipping the weight reload.")
        return True
    if ctx.fsdp_wrapped:
        if for_best_model:
            remedy = "load_best_model_at_end is unsupported under TP+DP — export the best checkpoint directly"
        else:
            remedy = (
                f"Resume with a model constructed FROM the checkpoint "
                f"({describe_weights_source(weights_read_from(model))}) — launch via the training scripts, "
                f"which repoint the load"
            )
        raise RuntimeError(
            f"TP+DP cannot reload {checkpoint} into the live model: FSDP2 over TP stacks a strided dp shard on the "
            "tp shard, a 2-D placement distribute_tensor does not invert for packed projections (right shape, "
            f"wrong rows), so the reload is refused rather than risked. {remedy}."
        )

    unwrapped = unwrap_framework_wrappers(model)
    live: dict[str, torch.Tensor] = dict(unwrapped.named_parameters())
    live.update(persistent_buffers(unwrapped))
    hand_sliced = hand_sliced_dims(unwrap_toolkit_wrappers(model))

    with joined_streaming_reader(checkpoint, None, what=f"{mode} checkpoint") as reader:
        view = ModuleLayoutView(unwrapped, reader.available)
        # The copy writes only matching keys, so another wrapper layout would resume BASE weights for most of the
        # model. Key counts are agreed with rank 0's, on disk too (one fused tensor stands for every per-expert key),
        # so a short per-node copy cannot pass alone.
        coverage_ok, unmatched, matched_numel, total_numel = resume_numel_coverage(model, view.keys)
        counts = (len(reader.available), len(view.keys))
        same_keys = counts == broadcast_from_rank0(counts)
        if not rank_consensus(coverage_ok and same_keys).all_ok:
            raise RuntimeError(
                _coverage_refusal(
                    mode, checkpoint, where=" on at least one rank", cause=", or a per-node copy is incomplete"
                )
            )
        if is_global_main_process():
            _report_unmatched_coverage(mode, checkpoint, unmatched, matched_numel, total_numel)
            if unexpected := sorted(view.keys - live.keys()):
                logger.warning(
                    f"{mode} resume: {len(unexpected)} checkpoint keys have no live tensor: {unexpected[:KEY_PREVIEW_COUNT]}"
                )

        guard = DeferredRankFailure(f"{mode} checkpoint reload from {checkpoint}")
        guard.run(
            partial(
                _copy_into_live, view.items(reader.get), live, hand_sliced, ctx.parallelism_config.tp_size, ctx.tp_rank
            )
        )
        guard.reject()

    if is_global_main_process():
        tp = f", tp_size={ctx.parallelism_config.tp_size}" if ctx.parallelism_config.is_tp_mode else ""
        logger.info(f"✓ {mode} checkpoint loaded from {checkpoint} ({len(live)} live tensors{tp})")
    barrier()
    return True


def _weight_files_present(checkpoint_dir: str) -> tuple[bool, bool]:
    return has_whole_model_weight_file(checkpoint_dir), has_adapter_weight_file(checkpoint_dir)


def load_resume_weights(
    ctx: CheckpointLoadContext, checkpoint_dir: str, model=None, *, for_best_model: bool = False
) -> bool:
    """COLLECTIVE. Reload ``checkpoint_dir``'s weights into ``model`` for the active mode, in place; False leaves the
    reload to the base Trainer's loader. PP reads this stage's tensors; EP/CP restores adapters only; FSDP2 distributes
    rank 0's tensors; TP and single-process/DDP stream the node's copy. A PEFT adapter-only checkpoint restores at the
    live dtype, where the base's ``load_adapter`` would recast to fp32. ``for_best_model`` never takes the built-from-
    checkpoint skips, since the live weights trained past it; EP/CP full fine-tune and TP+DP raise instead."""
    model = model if model is not None else ctx.model
    if ctx.parallelism_config.is_pp_mode:
        # First: a stage's re-based local names misresolve under every other path.
        restore_pp_stage_weights(ctx, checkpoint_dir, model, for_best_model=for_best_model)
        return True
    if ctx.has_ep_layers or ctx.parallelism_config.is_cp_mode:
        _resume_onto_constructed_model(ctx, checkpoint_dir, model, for_best_model=for_best_model)
        return True
    if ctx.fsdp_wrapped and not ctx.parallelism_config.is_tp_mode:
        return _load_fsdp2(ctx, checkpoint_dir, model, for_best_model=for_best_model)
    if ctx.parallelism_config.is_tp_mode:
        return _load_streamed(ctx, checkpoint_dir, model, for_best_model=for_best_model)
    # The base loader would reload an unsharded transformers model by raw key and an adapter-only checkpoint through an
    # fp32 recast; a reload it owns and a non-transformers model stay with it.
    unwrapped = unwrap_framework_wrappers(model)
    trains_peft = find_peft_model(unwrapped) is not None
    if ctx.base_owns_whole_weight_load or not (trains_peft or isinstance(unwrapped, PreTrainedModel)):
        return False
    whole_weights, adapter_weights = rank0_verdict(
        partial(_weight_files_present, checkpoint_dir), f"Probing {checkpoint_dir} for model weights"
    )
    if trains_peft:
        if whole_weights or not adapter_weights:
            return False
        restore_adapters(checkpoint_dir, model, is_cp_mode=False)
        return True
    if whole_weights:
        return _load_streamed(ctx, checkpoint_dir, model, for_best_model=for_best_model)
    return False
