"""The slot layout of a frozen quantized expert container and the banks an EP layer builds from it: per projection,
the packed weight under its own name, block scales under transformers' ``<weight>_scale_inv`` conversion target, and
NVFP4's per-expert ``amax`` and scalar denominator under the release's spelling. The loader fills it
(``halo/model_loading/parallel/frozen_experts.py``)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from halo.kernels.lowp.frozen_bank import FrozenExpertBank


@dataclass(frozen=True)
class FrozenSlot:
    """One projection's live weight name and the release's stem for its sidecars."""

    weight: str
    hub_stem: str

    @property
    def scale(self) -> str:
        return f"{self.weight}_scale_inv"

    @property
    def amax(self) -> str:
        return f"{self.hub_stem}_amax"

    @property
    def denominator(self) -> str:
        return f"{self.hub_stem}_outer_scale_denominator"


FROZEN_SLOTS = (FrozenSlot("gate_up_proj", "w13_weight"), FrozenSlot("down_proj", "w2_weight"))
# The per-expert sidecars, sharded on the expert axis with the weights; the scalar denominator replicates.
FROZEN_EXPERT_SIDECARS = tuple(name for slot in FROZEN_SLOTS for name in (slot.scale, slot.amax))


def is_frozen_bank_container(experts: nn.Module) -> bool:
    """Whether ``experts`` carries frozen slots rather than bf16 expert weights."""
    return FROZEN_SLOTS[0].scale in experts._buffers


def frozen_banks(experts: nn.Module, start: int, end: int, *, already_sharded: bool) -> dict[str, FrozenExpertBank]:
    """This rank's banks off a converted container the load filled; ``already_sharded`` when it holds only this rank's
    experts."""
    banks = {}
    for slot in FROZEN_SLOTS:
        data, scales = experts._buffers[slot.weight], experts._buffers[slot.scale]
        rows = slice(None) if already_sharded else slice(start, end)
        global_scale = None
        if data.dtype == torch.uint8:
            amax = experts._buffers[slot.amax][rows]
            global_scale = (amax.float() / experts._buffers[slot.denominator].float()).contiguous()
        quant_format = "nvfp4" if data.dtype == torch.uint8 else "mxfp8"
        banks[slot.weight] = FrozenExpertBank(
            quant_format, data[rows].contiguous(), scales[rows].contiguous(), global_scale
        )
    return banks
