"""Tensor Parallelism loads: HF-native ``tp_plan`` sharding for a dense model, attention-only TP for a
MoE, and EP+TP with the experts distributed by DeepEP."""

from __future__ import annotations

import logging
from contextlib import contextmanager

from transformers import DistributedConfig, PreTrainedModel

from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.distributed.mesh import build_dp_tp_mesh, tp_submesh
from halo.distributed.parallelism_config import ParallelismConfig
from halo.distributed.tensor_parallel.hf_plan import consistent_tied_tp_plan, finish_hf_tp_load
from halo.distributed.tensor_parallel.parallelize_attention import apply_tp_to_attention_only
from halo.hardware import current_device
from halo.model_loading.buffer_repair import finalize_loaded_model
from halo.model_loading.checkpoint_coverage import from_pretrained_verified
from halo.model_loading.parallel.ep import load_ep_model
from halo.model_loading.parallel.fp32_masters import restore_fp32_master_parameters
from halo.model_loading.parallel.model_source import ModelSource
from halo.model_loading.parallel.whole_model import load_whole_model
from halo.models.moe.config_fields import has_routed_experts
from halo.runtime.consensus import DeferredRankFailure
from halo.runtime.load_coordination import joined_node_load
from halo.runtime.ranks import get_global_rank, is_global_main_process

logger = logging.getLogger(__name__)


def uses_attention_only_tp(pc: ParallelismConfig, model_config) -> bool:
    """Whether the TP load shards attention alone through :func:`apply_tp_to_attention_only`: a MoE, with or
    without an EP group. A dense model takes HF's own ``tp_plan``."""
    return pc.is_tp_mode and (pc.has_ep_group or has_routed_experts(model_config))


class _ToolkitMeshViews:
    """The mesh views transformers' TP-only load reads, answered from the toolkit's (dp, tp) mesh."""

    def __init__(self, device_mesh, tp_mesh):
        self._views = {("pp", "fsdp", "tp"): device_mesh, ("tp",): tp_mesh}

    def get_mesh(self, dims):
        key = (dims,) if isinstance(dims, str) else tuple(dims)
        if key not in self._views:
            raise RuntimeError(
                f"transformers asked a TP-only load for the {key} mesh, which the toolkit does not build"
            )
        return self._views[key]


@contextmanager
def _toolkit_tp_mesh(device_mesh, tp_mesh):
    """Shard an HF-native TP load over the toolkit's TP group instead of a mesh transformers builds.

    transformers ignores a passed mesh and builds its own world-wide one, sized to the world unless
    ``fsdp_size > 1``, which also FSDP-wraps the model the toolkit wraps itself. Its single hook for the
    mesh is ``prepare_distribute_model``, answered here for the length of the load.
    """
    if "prepare_distribute_model" in PreTrainedModel.__dict__:
        raise RuntimeError("PreTrainedModel.prepare_distribute_model is already overridden")
    device = current_device()

    def prepare(cls, distributed_config, device_map=None):
        return distributed_config, device, _ToolkitMeshViews(device_mesh, tp_mesh)

    PreTrainedModel.prepare_distribute_model = classmethod(prepare)
    try:
        yield
    finally:
        del PreTrainedModel.prepare_distribute_model


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
    device_mesh = build_dp_tp_mesh(tp_size=tp_size, dp_size=dp_size)
    tp_mesh = tp_submesh(device_mesh)
    if is_global_main_process():
        logger.info(f"Created device mesh ({'DP x TP' if dp_size > 1 else 'TP only'}): {device_mesh}")

    # Not node-throttled: the load ends in ``tie_weights``, whose equality check on a tied pair of
    # DTensors is an all-reduce on the default group, which a rank-serialized region deadlocks. Each
    # rank's shard streams straight onto its GPU anyway, so host RAM never holds the model.
    with (
        consistent_tied_tp_plan(model_class, from_pretrained_kwargs["config"]),
        _toolkit_tp_mesh(device_mesh, tp_mesh),
    ):
        model = from_pretrained_verified(
            model_class,
            source.name,
            distributed_config=DistributedConfig(tp_plan="auto", tp_size=tp_size),
            **from_pretrained_kwargs,
        )
    # The toolkit owns this model's DP wrap, grad reduction, loss scaling and checkpoints. The flag would
    # send HF's Trainer down its own prepare and save paths and refuse checkpoints that carry optimizer
    # state; the size would make it rescale token counts and losses and require its own TP accelerator.
    model._is_distributed_loading_by_transformers = False
    model._tp_size = None
    precision_guard = DeferredRankFailure(f"FP32-master TP checkpoint load from {source.name}")
    # Throttled: collective-free, and every rank stages each full FP32 tensor in host memory before
    # keeping its own shard.
    with joined_node_load(f"FP32-master TP restore from {source.name}", pc.max_concurrent_loading):
        precision_guard.run(
            lambda: restore_fp32_master_parameters(
                model, source.local_dir, keep_non_ep=pc.fp32_non_ep_params, strict=preserve_checkpoint_precision
            )
        )
    precision_guard.reject()
    finish_hf_tp_load(model, device_mesh, tp_size)
    # After finish_hf_tp_load's tie check: the seam re-ties, which would mask a load that left the pair untied.
    finalize_loaded_model(model, weights_dir=source.local_dir)

    if is_global_main_process():
        logger.info(f"Model loaded with TP (tp_size={tp_size})")
        if dp_size > 1:
            logger.info(f"  FSDP2 will be used for DP (dp_size={dp_size})")
    return model


def _apply_attention_only_tp(model: PreTrainedModel, rank: int, tp_size: int, dp_size: int) -> None:
    """Shard the attention projections over the TP mesh (selective TP), leaving the MoE experts alone, and stamp
    the mesh onto the model."""
    device_mesh = build_dp_tp_mesh(tp_size=tp_size, dp_size=dp_size)
    num_tp_modules = apply_tp_to_attention_only(model, device_mesh)
    if is_global_main_process():
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
    mis-shards MoE expert biases); ``wraps_experts`` adds the grouped-GEMM EP layers."""
    rank = get_global_rank()
    tp_size, dp_size = pc.tp_size, pc.data_parallel_size
    if is_global_main_process():
        logger.info(f"[Rank {rank}] Loading MoE model for TP-only mode (attention-only TP)...")
        logger.info(f"  TP size: {tp_size}, DP size: {dp_size}")

    ep_config = pc.ep_config() if wraps_experts else None
    model = load_whole_model(
        model_class,
        source,
        pc,
        load_phase="Model load",
        ep_config=ep_config,
        preserve_checkpoint_precision=preserve_checkpoint_precision,
        **from_pretrained_kwargs,
    )
    _apply_attention_only_tp(model, rank, tp_size, dp_size)

    if wraps_experts:
        model = apply_ep_layers(model, ep_config)
        if is_global_main_process():
            logger.info("Swapped the grouped-GEMM EP layers into the MoE blocks")

    if is_global_main_process():
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
    """COLLECTIVE: load a MoE with EP+TP: the experts distributed by :func:`load_ep_model` (each rank holds
    only its own slice on its device, lazy or eager), then attention sharded via DTensor."""
    rank = get_global_rank()
    if is_global_main_process():
        logger.info(f"[Rank {rank}] Loading model for EP+TP mode...")
        logger.info(f"  EP size: {pc.ep_size}, TP size: {pc.tp_size}, DP size: {pc.data_parallel_size}")

    model = load_ep_model(
        source, pc, model_class, from_pretrained_kwargs, preserve_checkpoint_precision=preserve_checkpoint_precision
    )
    _apply_attention_only_tp(model, rank, pc.tp_size, pc.data_parallel_size)
    if is_global_main_process():
        logger.info(f"Model loaded with EP+TP (tp={pc.tp_size}, ep={pc.ep_size}, dp={pc.data_parallel_size})")
    return model
