"""Configured FP32 parameter masters on the HF tree, before parallel wrappers adopt them.

Selected off the EP layer registry, the same family accessors the wrappers adopt through, so every
loader — eager, lazy EP, PP stage — keeps one checkpoint-backed set at FP32. The trainer's
``fp32_non_ep_params`` upcast (``EpIntrospectionMixin._upcast_non_ep_params_to_fp32``) runs after
wrapping and is a no-op on that set; it covers what no loader read from a checkpoint. Persistent
FP32 buffers are not parameter masters; their family-declared precision is handled by the buffer
loader independently.
"""

from collections.abc import Iterable

import torch
import torch.nn as nn

from src.distributed.expert_parallel.config import EPConfig
from src.distributed.expert_parallel.patching import MOE_LAYER_MAP, ep_claimed_blocks
from src.log import KEY_PREVIEW_COUNT
from src.models.loading.dtype import is_packed_4bit_parameter


def fp32_master_param_keys(
    model: nn.Module, ep_config: EPConfig | None = None, *, keep_non_ep: bool
) -> frozenset[str]:
    """HF parameter keys whose configured training storage is FP32.

    The family's own router/container accessors select the parameters its wrapper adopts. Shared
    experts remain run-dtype, and FSDP-managed ep1 experts ignore ``fp32_experts`` as at wrapper init.
    Object identity includes tied aliases without mistaking a persistent float buffer for a master.
    """
    blocks = ep_claimed_blocks(model) if ep_config is not None else []
    ep_params = {id(param) for _path, block in blocks for param in block.parameters()}
    masters = {id(param) for param in model.parameters() if keep_non_ep and id(param) not in ep_params}
    for _path, block in blocks:
        layer_cls = MOE_LAYER_MAP[type(block).__name__]
        if ep_config.fp32_router:
            router = layer_cls._find_gate_or_router(block)
            if isinstance(router, nn.Parameter):
                masters.add(id(router))
            elif router is not None:
                masters.update(id(param) for param in router.parameters())
        if ep_config.fp32_experts and not ep_config.experts_fsdp_managed:
            masters.update(id(param) for param in layer_cls._find_experts_container(block).parameters())
    return frozenset(
        name
        for name, param in model.state_dict(keep_vars=True).items()
        if isinstance(param, nn.Parameter)
        and param.is_floating_point()
        and not is_packed_4bit_parameter(param)
        and id(param) in masters
    )


def verify_fp32_master_coverage(
    state: dict[str, torch.Tensor], masters: frozenset[str], loaded: Iterable[str], *, source: str
) -> None:
    """Require every selected master's identity, including aliases, without fresh-load exemptions."""
    loaded_ids = {id(state[key]) for key in loaded}
    missing = sorted(key for key in masters if id(state[key]) not in loaded_ids)
    if missing:
        raise RuntimeError(
            f"FP32-master checkpoint load from {source}: {len(missing)} configured master "
            f"tensor(s) were not restored: {missing[:KEY_PREVIEW_COUNT]}. A full-finetune resume "
            "must restore every master; missing-key exemptions cannot retain rounded or random weights."
        )
