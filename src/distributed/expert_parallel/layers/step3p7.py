"""EP wrapper for Step-3.7 MoE (fused experts, per-layer post-activation clamp, sigmoid routing)."""

from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn as nn

from src.distributed.expert_parallel.base_layer import EPGroupLimitedMoELayerBase
from src.kernels.fused_glu import fused_silu_then_clamp_mul, is_silu_activation


class EPStep3p7MoELayer(EPGroupLimitedMoELayerBase):
    """EP wrapper for Step-3.7 MoE (``Step3p7SparseMoeBlock``, Step-3.7-Flash) using DeepEP.

    Fused ``gate_up_proj [E, 2M, H]`` (contiguous ``[gate; up]`` halves) / ``down_proj [E, H, M]``
    routed experts with a per-layer clamped SwiGLU (``swiglu_limits[layer_idx]``, applied to the
    activated gate, where DSv4/GLM-5 clamp before the activation; most layers carry no clamp and run
    the base combine), sigmoid routing whose ``e_score_correction_bias`` buffer perturbs selection
    only (plain top-k, no group limiting), always-renormalized unbiased gate weights, the
    block-level routed scaling folded into those weights, and one always-on shared expert
    (``Step3p7MLP``, which carries its own ``swiglu_limits_shared`` clamp). Dense layers
    (``mlp_layer_types``) are plain MLPs and are never wrapped.
    """

    HF_MODULE_NAMES = ("Step3p7SparseMoeBlock",)
    # The composite VLM wrapper spelling Step-3.7-Flash checkpoints write at the top level plus the
    # text tower's spelling; checkpoint-keyed tooling (sharded merge, lazy-load routing) resolves
    # either config to this class.
    HF_MODEL_TYPES = ("step3p7", "step3p5")

    # The shared group-limited routing adds this native slot (an fp32 ``nn.Buffer`` upstream, hub key
    # ``moe.router_bias``) to the selection score (selection-only, DeepSeek-V3 style) and the slot is
    # part of the checkpoint. The modeling has no aux-loss machinery (no ``output_router_logits``, no
    # coefficient), so ``moe_balancing: auto`` resolves to ``bias_update`` here.
    _supports_bias_balancing = True
    _NATIVE_BALANCING_BIAS_ATTR = "gate.e_score_correction_bias"

    # ``Step3p7TopKRouter`` renormalizes the selected weights unconditionally and with no floor;
    # nothing in the block, the router or the config declares ``norm_topk_prob`` or a group knob, so
    # the shared body runs as plain top-k. The block's own ``routed_scaling_factor`` (config
    # ``moe_router_scaling_factor``), which upstream applies to the routed-expert sum, resolves into
    # the per-token weights instead: equivalent, since the combine is linear in the expert outputs.
    # It stays required rather than defaulted: at 1.0 every routed weight misses the model's own
    # ``moe_router_scaling_factor``.
    _TOPK_WEIGHT_NORM_EPS = 0.0
    _OPTIONAL_ROUTING_KNOBS = ("n_group", "topk_group", "norm_topk_prob")

    # ``gate.weight`` is ``[num_experts, hidden]``; the block itself carries no config reference,
    # so the count comes off the router or the experts container.
    _NUM_EXPERTS_ATTR_PATHS = ("gate.num_experts",)

    # Hub layout: per-layer fused-but-split tensors (``moe.gate_proj``/``moe.up_proj`` ``[E, M, H]``,
    # ``moe.down_proj`` ``[E, H, M]``). No per-expert spelling exists, so ``hub_per_expert_keys()``
    # returns None and ``unfuse_moe_experts`` rejects the family. The checkpoint keeps the vendor
    # namespace transformers converts inside ``from_pretrained``: the prefix renames, ``moe.*`` →
    # ``mlp.*``, the two-source ``gate_proj + up_proj → gate_up_proj`` Concatenate (sliced through
    # both sources on the expert axis), and the vision tower's entries. See ``hub_conversion.py``.
    _HUB_CONVERSION_KEYS = ("step3p7", "step3p5_vision")

    # A module-spelled save has no consumer here, so the gathered save applies transformers' own
    # save-side revert per chunk: prefix renames back, ``mlp.gate`` → ``moe.gate``/``moe.router_bias``,
    # ``shared_experts`` → ``share_expert``, and ``gate_up_proj`` split back into the hub's two
    # tensors. Chunk-safe, since the only reverse entry touching an EP layer is that single-source
    # split. The RL weight sync reverts likewise, so vLLM's step3p5 loader gets the names it maps.
    _EXPORTS_HUB_NAMESPACE = True

    # The pinned server has no ``step3p7`` config class: it reads the family only through the
    # release's own ``config.json`` and its ``auto_map`` modules (``moe_num_experts``, ``moe_top_k``,
    # ``moe_layers_enum``, ``attention_other_setting``, per-layer ``rope_theta``). transformers
    # absorbs those spellings at load and re-emits only the native ones, so the server rejects a
    # transformers-written ``config.json``; every toolkit config write carries the source config and
    # modules forward instead.
    _EXPORTS_SOURCE_CONFIG_SCHEMA = True

    # The block always builds its shared expert, so an absent one is an upstream rename rather than a
    # configuration, and adopting nothing would drop it from every output.
    _SHARED_EXPERT_ATTRS = ("shared_experts",)
    _SHARED_EXPERT_REQUIRED = True

    def _init_expert_compute(self, original_layer: nn.Module, experts: nn.Module) -> None:
        """The activation (the base latch serves the unclamped layers through the fused SiLU kernel)
        plus this layer's own clamp bound, where ``inf`` means unclamped (most layers), with
        Step-3.7-Flash clamping layers 43-44 at 7.0. A clamped layer replaces the latch with its own
        combine, whose Triton kernel implements SiLU only, so any other activation falls through to
        the eager form. Read as an attribute: a substituted default un-clamps those layers."""
        super()._init_expert_compute(original_layer, experts)
        self.swiglu_limit = float(experts.limit)
        if not math.isinf(self.swiglu_limit):
            self._fused_glu_mul = (
                partial(fused_silu_then_clamp_mul, limit=self.swiglu_limit)
                if is_silu_activation(self.act_fn)
                else self._act_then_clamp_eager
            )

    def _act_then_clamp_eager(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """Eager form of this layer's clamp for a gate no kernel implements: clamp after the
        activation."""
        return self.act_fn(gate).clamp(max=self.swiglu_limit) * up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)

    def _init_summary_extras(self, original_layer: nn.Module) -> tuple[str, ...]:
        return (f"swiglu_limit={self.swiglu_limit}",)
