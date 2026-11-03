#!/usr/bin/env python
"""Every registered EP family builds on its tiny model and adopts all the block's parameters, neither dropping nor
copying any."""

import pytest
import torch

from halo.distributed.expert_parallel.families.roster import MOE_LAYER_MAP
from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.expert_parallel.parallelize import apply_ep_layers, ep_claimed_blocks
from tests.common.parallelism import single_process_ep_config
from tests.common.rosters import ep_layer_class_by_model_type
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model

FAMILY_CLASSES = sorted(
    set(MOE_LAYER_MAP.values()) | set(ep_layer_class_by_model_type().values()), key=lambda cls: cls.__name__
)
TINY_FAMILY_OF_CLASS = {ep_layer_class_by_model_type()[name]: name for name in TINY_MOE_FAMILIES}


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


@pytest.mark.parametrize("family_cls", FAMILY_CLASSES, ids=lambda cls: cls.__name__)
def test_the_family_wraps_every_block_it_claims(family_cls):
    name = TINY_FAMILY_OF_CLASS.get(family_cls)
    assert name is not None, f"{family_cls.__name__} has no tiny model in TINY_MOE_FAMILIES"
    model = tiny_family_model(TINY_MOE_FAMILIES[name])
    claimed = {path: MOE_LAYER_MAP[type(block).__name__] for path, block in ep_claimed_blocks(model)}
    parameters = _parameter_count(model)

    apply_ep_layers(model, single_process_ep_config())

    assert set(claimed.values()) == {family_cls}
    wrapped = {path: type(layer) for path, layer in named_ep_layers(model)}
    assert wrapped == claimed
    assert _parameter_count(model) == parameters


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
