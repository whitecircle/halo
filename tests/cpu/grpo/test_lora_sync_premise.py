"""The LoRA sync memory check only means something when the LoRA'd shard on a rank outweighs the fold's allowance
(``memory_bound_discriminates``). A family whose release dims put nearly every LoRA'd byte in one layer needs a deeper
tiny model for that, which ``PREMISE_OVERRIDES`` supplies."""

import pytest
import torch

from tests.common.adapters import mixer_linear_leaves
from tests.common.lora_sync_exactness import FOLD_PEAK_BOUND, PREMISE_OVERRIDES, WORLD_SIZE
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model


def _local_and_largest_bytes(family: str, overrides: dict | None) -> tuple[float, int]:
    with torch.device("meta"):
        model = tiny_family_model(TINY_MOE_FAMILIES[family], overrides=overrides)
    leaves = set(mixer_linear_leaves(model))
    sizes = [
        module.weight.numel() * 2
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] in leaves and "visual" not in name
    ]
    return sum(sizes) / WORLD_SIZE, max(sizes)


@pytest.mark.parametrize("family", ["glm5_next"])
def test_override_restores_the_premise_the_family_lacks_without_it(family: str) -> None:
    local, largest = _local_and_largest_bytes(family, PREMISE_OVERRIDES[family])
    assert local > FOLD_PEAK_BOUND * largest
    local, largest = _local_and_largest_bytes(family, None)
    assert local <= FOLD_PEAK_BOUND * largest


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
