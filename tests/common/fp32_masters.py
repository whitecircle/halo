"""An EP model's trainable parameters by the knob that makes each an fp32 master: the routed experts
(``ep_fp32_experts``), the live routers (``ep_fp32_router``) and every other parameter (``fp32_non_ep_params``),
read off the live parameter identities."""

from torch import nn

from halo.distributed.fsdp import reshard_fsdp2_modules
from tests.common.ep_reference import ep_layers, router_parameters

FP32_MASTER_GROUP_NAMES = ("expert", "router", "non_expert")


def fp32_master_groups(model: nn.Module) -> dict[str, dict[str, nn.Parameter]]:
    """``model``'s trainable parameters by name in each of :data:`FP32_MASTER_GROUP_NAMES`, read with the FSDP2
    groups resharded, so a sharded parameter is its registered shard. No collective."""
    reshard_fsdp2_modules(model)
    layers = ep_layers(model)
    experts = {id(parameter) for layer in layers for _, parameter in layer.expert_named_params()}
    routers = {id(parameter) for parameter in router_parameters(layers)}
    groups: dict[str, dict[str, nn.Parameter]] = {name: {} for name in FP32_MASTER_GROUP_NAMES}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            group = "expert" if id(parameter) in experts else "router" if id(parameter) in routers else "non_expert"
            groups[group][name] = parameter
    return groups
