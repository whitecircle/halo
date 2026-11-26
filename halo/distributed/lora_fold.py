"""The out-of-place LoRA fold of a possibly sharded base weight."""

import torch
from torch.distributed.tensor import DTensor, distribute_tensor

from halo.models import peft_adapters
from halo.models.peft_adapters import LoraFolds, adapter_tensors_replaced, is_conv_lora_layer
from halo.runtime.dtensor import materialize_dtensor


def lora_folded_data(param: torch.nn.Parameter, folds: LoraFolds | None) -> torch.Tensor:
    """What PEFT's merge writes into ``param``; COLLECTIVE for a DTensor. A conv layer on a DTensor folds on whole
    tensors and reshards, as a DTensor convolution runs on each rank's shard of the adapter rank dimension."""
    layers = folds.get(id(param)) if folds else None
    if not layers:
        return param.data
    data = param.data
    if not (isinstance(data, DTensor) and any(is_conv_lora_layer(layer) for layer in layers)):
        return peft_adapters.lora_folded(param, layers)
    folded = peft_adapters.lora_folded(
        param,
        layers,
        base=materialize_dtensor(data),
        adapter_scope=lambda layer, adapter: adapter_tensors_replaced(layer, adapter, materialize_dtensor),
    )
    return distribute_tensor(folded, data.device_mesh, data.placements, src_data_rank=None)
