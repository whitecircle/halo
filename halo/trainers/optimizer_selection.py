"""Which optimizer a trainer run builds: AdamWBF16, a toolkit-named optimizer, the tensor-type-grouped build, or stock."""

import logging

from halo.optimizers.adamw_bf16 import adamw_bf16_optimizer_from_args
from halo.optimizers.param_groups import build_tensor_type_grouped_optimizer
from halo.optimizers.registry import (
    NAMED_OPTIMIZER_BUILDERS,
    STOCK_ADAMW_OPTIMIZERS,
    optim_name,
    register_custom_optimizers,
)
from halo.runtime.ranks import is_global_main_process
from halo.trainers.parallelism.trainer_rules import reject_conflicting_optimizer_choice

logger = logging.getLogger(__name__)

register_custom_optimizers()


def resolve_bf16_optimizer(explicit: bool | None, config, training_args, *, replicated_ddp: bool) -> bool:
    """Whether the run uses AdamWBF16: the ctor kwarg (tests construct trainers directly), else the config's,
    else auto-enabled under ``bf16`` on the default stock AdamW, so an explicit choice such as ``adamw_bnb_8bit``
    stands."""
    if explicit is None:
        explicit = config.bf16_optimizer
    if explicit is not None:
        reject_conflicting_optimizer_choice(explicit, training_args)
        return explicit
    if training_args is None or not getattr(training_args, "bf16", False):
        return False
    if optim_name(training_args.optim) not in STOCK_ADAMW_OPTIMIZERS:
        return False
    if replicated_ddp:
        # Replicated DDP is outside the validated SR matrix (FSDP/EP/TP/HSDP).
        if is_global_main_process():
            logger.warning(
                "bf16=True with replicated DDP (accelerate MULTI_GPU): "
                "skipping AdamWBF16 stochastic-rounding auto-enable to avoid "
                "cross-replica weight drift. Use FSDP/EP/TP for SR, or set "
                "bf16_optimizer=True explicitly to override."
            )
        return False
    return True


def build_optimizer(
    model, args, decay_parameters, *, bf16_optimizer: bool, config, has_ep_layers: bool, optimizer_cls_and_kwargs
):
    """The toolkit's optimizer for this run, or ``None`` where the base Trainer's stock build applies."""
    optim = optim_name(args.optim)
    if bf16_optimizer:
        return adamw_bf16_optimizer_from_args(model, args, decay_parameters)
    if optim in NAMED_OPTIMIZER_BUILDERS:
        return NAMED_OPTIMIZER_BUILDERS[optim][0](model, args, decay_parameters)
    # Stock SGD's foreach cannot mix FSDP2 DTensors with plain EP tensors; kept narrow to spare other optims.
    if config.fp32_non_ep_params or (optim == "sgd" and has_ep_layers):
        return build_tensor_type_grouped_optimizer(model, args, decay_parameters, optimizer_cls_and_kwargs)
    return None
