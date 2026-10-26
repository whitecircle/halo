"""Tensor Parallelism loads: HF-native ``tp_plan`` sharding for a dense model, attention-only TP for a
MoE, and EP+TP with the experts distributed by DeepEP."""

from __future__ import annotations

from accelerate.logging import get_logger
from transformers import DistributedConfig, PreTrainedModel

from src.distributed.expert_parallel.lazy_loader import lazy_loader_supports_checkpoint, load_ep_model_lazy
from src.distributed.expert_parallel.master_weights import restore_fp32_master_parameters
from src.distributed.expert_parallel.patching import patch_moe_model_for_ep
from src.distributed.mesh import create_dp_tp_mesh, get_tp_submesh
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.tensor_parallel.hf_plan import consistent_tied_tp_plan, finish_hf_tp_load
from src.distributed.tensor_parallel.parallelize_attention import apply_tp_to_attention_only
from src.model_loading.buffer_repair import finalize_loaded_model
from src.model_loading.checkpoint_coverage import from_pretrained_verified
from src.model_loading.parallel.model_source import ModelSource, resolve_lazy_checkpoint
from src.model_loading.parallel.whole_model import load_whole_model
from src.runtime.consensus import DeferredRankFailure
from src.runtime.load_coordination import joined_node_load, timed_global_load
from src.runtime.ranks import get_global_rank

logger = get_logger(__name__)


def load_tp_model(
    source: ModelSource,
    pc: ParallelismConfig,
    model_class,
    from_pretrained_kwargs: dict,
    *,
    preserve_checkpoint_precision: bool,
) -> PreTrainedModel:
    """COLLECTIVE: load a dense model with HF-native ``tp_plan`` weight sharding over a (DP, TP) mesh."""
    tp_size, dp_size = pc.tp_size, pc.data_parallel_size
    device_mesh = create_dp_tp_mesh(tp_size=tp_size, dp_size=dp_size)
    tp_mesh = get_tp_submesh(device_mesh)
    logger.info(f"Created device mesh ({'DP x TP' if dp_size > 1 else 'TP only'}): {device_mesh}")

    # Not node-throttled: the load ends in ``tie_weights``, whose equality check on a tied pair of
    # DTensors is an all-reduce on the default group, which a rank-serialized region deadlocks. Each
    # rank's shard streams straight onto its GPU anyway, so host RAM never holds the model.
    with consistent_tied_tp_plan(model_class, from_pretrained_kwargs["config"]):
        model = from_pretrained_verified(
            model_class,
            source.name,
            # The 5.16 spelling of tp_plan="auto"; with an explicit device_mesh the mesh is authoritative.
            # The config's derived tp_size (WORLD_SIZE) is read only by the model's own save_pretrained,
            # which no toolkit path calls on a live TP model.
            distributed_config=DistributedConfig(tp_plan="auto"),
            device_mesh=tp_mesh,
            **from_pretrained_kwargs,
        )
    precision_guard = DeferredRankFailure(f"FP32-master TP checkpoint load from {source.name}")
    # Throttled: collective-free, and every rank stages each full FP32 tensor in host memory before
    # keeping its own shard.
    with joined_node_load(f"FP32-master TP restore from {source.name}", pc.max_concurrent_loading):
        precision_guard.run(
            lambda: restore_fp32_master_parameters(
                model,
                source.name,
                keep_non_ep=pc.fp32_non_ep_params,
                strict=preserve_checkpoint_precision,
                revision=from_pretrained_kwargs.get("revision"),
            )
        )
    precision_guard.reject()
    finish_hf_tp_load(model, device_mesh, tp_size)
    # After finish_hf_tp_load's tie check: the seam re-ties, which would mask a load that left the pair untied.
    finalize_loaded_model(model)

    logger.info(f"Model loaded with TP (tp_size={tp_size})")
    if dp_size > 1:
        logger.info(f"  FSDP2 will be used for DP (dp_size={dp_size})")
    return model


def _apply_attention_only_tp(model: PreTrainedModel, rank: int, tp_size: int, dp_size: int) -> None:
    """Shard attention via DTensor TP, leaving the MoE experts alone, and stamp the mesh onto the model."""
    device_mesh = create_dp_tp_mesh(tp_size=tp_size, dp_size=dp_size)
    num_tp_modules = apply_tp_to_attention_only(model, device_mesh)
    logger.info(f"[Rank {rank}] TP applied to {num_tp_modules} attention modules")
    model._device_mesh = device_mesh


def load_tp_moe_model(
    source: ModelSource,
    pc: ParallelismConfig,
    model_class,
    from_pretrained_kwargs: dict,
    *,
    preserve_checkpoint_precision: bool,
    wraps_experts: bool,
) -> PreTrainedModel:
    """COLLECTIVE: load a MoE with TP on attention only, the experts replicated (HF's ``tp_plan="auto"``
    mis-shards MoE expert biases); ``wraps_experts`` adds the grouped-GEMM expert wrappers."""
    rank = get_global_rank()
    tp_size, dp_size = pc.tp_size, pc.data_parallel_size
    logger.info(f"[Rank {rank}] Loading MoE model for TP-only mode (attention-only TP)...")
    logger.info(f"  TP size: {tp_size}, DP size: {dp_size}")

    ep_config = pc.ep_config() if wraps_experts else None
    model = load_whole_model(
        model_class,
        source.name,
        load_phase="Model load",
        max_concurrent_loading=pc.max_concurrent_loading,
        ep_config=ep_config,
        keep_fp32_non_ep=pc.fp32_non_ep_params,
        ep_wrapped=wraps_experts,
        preserve_checkpoint_precision=preserve_checkpoint_precision,
        **from_pretrained_kwargs,
    )
    _apply_attention_only_tp(model, rank, tp_size, dp_size)

    if wraps_experts:
        model = patch_moe_model_for_ep(model, ep_config)
        logger.info("Applied grouped GEMM wrappers to MoE layers")

    logger.info(f"Model loaded with TP on MoE (tp={tp_size}, dp={dp_size})")
    return model


def load_ep_tp_model(
    source: ModelSource,
    pc: ParallelismConfig,
    model_class,
    from_pretrained_kwargs: dict,
    *,
    preserve_checkpoint_precision: bool,
) -> PreTrainedModel:
    """COLLECTIVE: load a MoE with EP+TP, attention sharded via DTensor and experts distributed via DeepEP.

    ``ep_lazy_loading`` streams the experts through the lazy safetensors loader and applies TP after;
    otherwise every rank loads the whole model through ``from_pretrained``.
    """
    rank = get_global_rank()
    logger.info(f"[Rank {rank}] Loading model for EP+TP mode...")
    logger.info(f"  EP size: {pc.ep_size}, TP size: {pc.tp_size}, DP size: {pc.data_parallel_size}")

    # Before the lazy gate: ep_config() issues new_group on every rank, so a per-rank
    # filesystem probe ahead of it would order the groups per rank (deadlock).
    ep_config = pc.ep_config()
    use_lazy = resolve_lazy_checkpoint(
        source, lazy_requested=pc.ep_lazy_loading, layout_supported=lazy_loader_supports_checkpoint
    )
    if use_lazy:
        logger.info(f"[Rank {rank}] Using lazy safetensors loader for EP+TP")
    elif pc.ep_lazy_loading:
        logger.info(f"[Rank {rank}] Lazy path unavailable, falling back to sequential loading")

    def load() -> PreTrainedModel:
        if use_lazy:
            model = load_ep_model_lazy(
                source.local_dir,
                ep_config,
                model_class=model_class,
                keep_fp32_params=pc.fp32_non_ep_params,
                preserve_checkpoint_precision=preserve_checkpoint_precision,
                **from_pretrained_kwargs,
            )
        else:
            model = load_whole_model(
                model_class,
                source.name,
                load_phase="Model load",
                max_concurrent_loading=pc.max_concurrent_loading,
                ep_config=ep_config,
                keep_fp32_non_ep=pc.fp32_non_ep_params,
                ep_wrapped=True,
                preserve_checkpoint_precision=preserve_checkpoint_precision,
                **from_pretrained_kwargs,
            )
        _apply_attention_only_tp(model, rank, pc.tp_size, pc.data_parallel_size)
        return model if use_lazy else patch_moe_model_for_ep(model, ep_config)

    model = timed_global_load("EP+TP", "ep_tp_lazy" if use_lazy else "ep_tp_huggingface", load)
    logger.info(f"Model loaded with EP+TP (tp={pc.tp_size}, ep={pc.ep_size}, dp={pc.data_parallel_size})")
    return model
