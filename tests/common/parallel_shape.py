"""Model-side evidence that each parallel axis a run asked for took effect.

A trainer's ``is_ep_mode`` / ``is_tp_mode`` / ``is_cp_mode`` return the ``ParallelismConfig`` it was
handed, so a check on them passes on a run whose loader silently skipped the axis. These probes read
what the loader and the trainer's wrap left on the model instead: the EP wrappers and the expert
weights they hold, the TP-sharded parameters, the Ulysses attention layers.
"""

from torch.distributed.tensor import DTensor

from src.distributed.context_parallel.base_layer import UlyssesAttentionBase
from src.distributed.mesh import has_tp_dim
from src.models.moe_balancing import config_has_experts
from tests.common.ep_reference import ep_layers
from tests.common.utils import log


def tp_sharded_param_names(model) -> list[str]:
    """Parameters held as DTensors on a mesh with a ``tp`` dim."""
    return [
        name
        for name, param in model.named_parameters()
        if isinstance(param.data, DTensor) and has_tp_dim(param.data.device_mesh)
    ]


def ulysses_attention_layers(model) -> list[UlyssesAttentionBase]:
    """The attention layers CP swapped for their Ulysses wrappers."""
    return [module for module in model.modules() if isinstance(module, UlyssesAttentionBase)]


def _bank_split_ep_way(layer) -> bool:
    """The layer owns an ``ep_size``-th of the expert bank, and its expert weights hold exactly that
    many experts. EP-sharded experts are FSDP-ignored plain tensors, so dim 0 is the local count."""
    weights = layer.expert_named_params()
    return (
        layer.experts_per_rank * layer.ep_size == layer.num_experts
        and layer.experts_per_rank < layer.num_experts
        and layer.expert_end - layer.expert_start == layer.experts_per_rank
        and bool(weights)
        and all(param.shape[0] == layer.experts_per_rank for _, param in weights)
    )


def parallel_shape_checks(model, parallelism_config) -> dict[str, bool]:
    """One named check per parallel axis ``parallelism_config`` enables, each read off ``model``.

    * EP wrappers (``ep_layers_wrapped``) wherever the run needs them and the checkpoint is a MoE: EP
      distribution, or grouped-GEMM expert compute at ``ep_size=1``;
    * ``expert_bank_split_ep_way`` under ``ep_size > 1``, on every EP layer;
    * ``expert_ffn_sharded_etp_way`` under ETP, on every EP layer;
    * ``tp_sharded_params`` under TP;
    * ``cp_attention_wrapped`` under CP, every Ulysses layer spanning ``cp_size`` ranks.

    Call it after the trainer is built, so the FSDP2 wrap has run. No collective.
    """
    pc = parallelism_config
    checks: dict[str, bool] = {}
    layers = ep_layers(model)
    if pc.needs_ep_wrappers and config_has_experts(model.config):
        checks["ep_layers_wrapped"] = bool(layers)
        log(f"  EP layers wrapped: {'PASS' if layers else 'FAIL'} ({len(layers)})")
    if pc.ep_size > 1:
        checks["expert_bank_split_ep_way"] = bool(layers) and all(_bank_split_ep_way(layer) for layer in layers)
        log(f"  Expert bank split {pc.ep_size}-way: {'PASS' if checks['expert_bank_split_ep_way'] else 'FAIL'}")
    if pc.is_expert_tp_mode:
        checks["expert_ffn_sharded_etp_way"] = bool(layers) and all(
            layer.expert_tp_size == pc.expert_tp_size for layer in layers
        )
        log(
            f"  Expert FFN sharded {pc.expert_tp_size}-way: "
            f"{'PASS' if checks['expert_ffn_sharded_etp_way'] else 'FAIL'}"
        )
    if pc.is_tp_mode:
        sharded = tp_sharded_param_names(model)
        checks["tp_sharded_params"] = bool(sharded)
        log(f"  TP-sharded params: {'PASS' if sharded else 'FAIL'} ({len(sharded)})")
    if pc.is_cp_mode:
        attention = ulysses_attention_layers(model)
        checks["cp_attention_wrapped"] = bool(attention) and all(layer.cp_size == pc.cp_size for layer in attention)
        log(f"  Ulysses attention layers: {'PASS' if checks['cp_attention_wrapped'] else 'FAIL'} ({len(attention)})")
    return checks
