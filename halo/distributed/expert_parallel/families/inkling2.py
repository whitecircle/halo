"""EP layer for Inkling 2 (Thinking Machines): sigmoid-routed fused experts, softcapped SwiGLU, no shared expert."""

from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from halo.distributed.expert_parallel.moe_layer.expert_lora import ExpertLoraTarget
from halo.distributed.expert_parallel.moe_layer.shared_experts_layer import EPGroupLimitedMoELayerBase
from halo.kernels.lowp.frozen_bank import FrozenExpertBank, build_frozen_banks, frozen_bank_grouped_mm
from halo.models.moe.family_specs import INKLING2

# This layer's buffers per frozen projection: packed data, block scales, NVFP4's per-expert global scale.
_FROZEN_PARTS = ("_frozen_data", "_frozen_scales", "_frozen_global")


class EPInkling2MoELayer(EPGroupLimitedMoELayerBase):
    """``Inkling2TopKRouter`` selects top-k on ``sigmoid(logits)`` plus ``e_score_correction_bias``, renormalizes the
    unbiased scores and scales them by ``route_scale``: the group-limited template at one group. The expert activation
    takes gate and up together, so it is latched as the combine. Experts are bf16 parameters, or, under
    ``frozen_quantized_experts``, frozen banks in the release's NVFP4/MXFP8 layout."""

    HF_MODULE_NAMES = ("Inkling2SparseMoeBlock",)
    FAMILY = INKLING2
    SUPPORTS_FROZEN_BANKS = True
    _SHARED_EXPERT_REQUIRED = False
    # The router has no group limiting and always renormalizes; ``routed_scaling_factor`` stays required.
    _OPTIONAL_ROUTING_KNOBS = ("n_group", "num_group", "topk_group", "norm_topk_prob")

    def _router_logits(self, flat: torch.Tensor) -> torch.Tensor:
        """In fp32, as the upstream router; the weight is upcast too, since a loader may have cast it to the run
        dtype."""
        with torch.amp.autocast("cuda", enabled=False):
            return F.linear(self._router_input(flat).float(), self.gate.weight.float())

    def _adopt_expert_compute(self, original_layer: nn.Module, experts: nn.Module) -> None:
        """``Inkling2SoftcappedSwiGLU(gate, up)`` is the whole combine. The latch takes the bound ``forward``: assigning
        the module would register it as a submodule and leave the class-level ``None`` latch in place."""
        self.act_fn = experts.act_fn
        self._fused_glu_mul = experts.act_fn.forward

    def _adopt_expert_params(self, experts: nn.Module, weights_already_sharded: bool = False) -> None:
        slots = self.FAMILY.frozen_expert_slots
        # The frozen loader registers every slot's scales; a bf16 container holds none.
        if slots[0].scale not in experts._buffers:
            super()._adopt_expert_params(experts, weights_already_sharded=weights_already_sharded)
            return
        banks = build_frozen_banks(
            experts, slots, self.expert_start, self.expert_end, already_sharded=weights_already_sharded
        )
        self._frozen_formats = {name: bank.quant_format for name, bank in banks.items()}
        for name, bank in banks.items():
            # Not persistent: a bank reloads from the release, and no checkpoint of this layer may carry it.
            for part, tensor in zip(_FROZEN_PARTS, (bank.data, bank.scales, bank.global_scale), strict=True):
                if tensor is not None:
                    self.register_buffer(f"{name}{part}", tensor, persistent=False)

    @property
    def has_frozen_banks(self) -> bool:
        return bool(getattr(self, "_frozen_formats", None))

    def build_frozen_bank(self, name: str) -> FrozenExpertBank:
        """The bank over this layer's current buffers, so a device move is seen."""
        data, scales, global_scale = (getattr(self, f"{name}{part}", None) for part in _FROZEN_PARTS)
        return FrozenExpertBank(self._frozen_formats[name], data, scales, global_scale)

    def frozen_expert_banks(self) -> tuple[FrozenExpertBank, ...]:
        return tuple(self.build_frozen_bank(name) for name in getattr(self, "_frozen_formats", ()))

    def frozen_bank_weight_count(self) -> int:
        return sum(bank.num_experts * bank.out_features * bank.in_features for bank in self.frozen_expert_banks())

    def start_gather_expert_state_dict(self, **kwargs):
        if self.has_frozen_banks:
            raise ValueError(
                f"{type(self).__name__} holds frozen quantized expert banks, which no gather, merged save or weight "
                f"sync exports. Save the adapters, or train from a bf16 checkpoint without frozen_quantized_experts."
            )
        return super().start_gather_expert_state_dict(**kwargs)

    @property
    def _stores_fused_glu(self) -> bool:
        return self.has_frozen_banks or super()._stores_fused_glu

    def _expert_intermediate_width(self) -> int:
        if not self.has_frozen_banks:
            return super()._expert_intermediate_width()
        return self.build_frozen_bank("down_proj").in_features

    def _expert_base_proj(
        self, x: torch.Tensor, weight_attr: str, offs: torch.Tensor, out_dtype: torch.dtype
    ) -> torch.Tensor:
        if not self.has_frozen_banks:
            return super()._expert_base_proj(x, weight_attr, offs, out_dtype)
        return frozen_bank_grouped_mm(x.to(out_dtype), self.build_frozen_bank(weight_attr), offs=offs)

    def _expert_base_proj_single(self, idx: int, x: torch.Tensor, weight_attr: str) -> torch.Tensor:
        if not self.has_frozen_banks:
            return super()._expert_base_proj_single(idx, x, weight_attr)
        with torch.no_grad():
            weight = self.build_frozen_bank(weight_attr).dequantize_linear(idx, idx + 1)[0]
        return x @ weight.to(x.dtype).t()

    def _expert_lora_targets(self) -> Iterator[ExpertLoraTarget]:
        """A bank's adapters take its matmul-convention ``[E, K, N]`` shape in bf16, in the family's root order so the
        adapter inits draw the RNG as the bf16 path does."""
        if not self.has_frozen_banks:
            yield from super()._expert_lora_targets()
            return
        for name in (root for root in self.FAMILY.expert_weight_roots if root in self._frozen_formats):
            bank = self.build_frozen_bank(name)
            yield ExpertLoraTarget(
                name, (bank.num_experts, bank.in_features, bank.out_features), torch.bfloat16, bank.data.device
            )
