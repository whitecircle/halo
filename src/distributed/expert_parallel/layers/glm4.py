"""EP wrapper for GLM-4 MoE Lite's fused sigmoid-routed MoE (Laguna subclasses it)."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.distributed.expert_parallel.base_layer import EPGroupLimitedMoELayerBase


class EPGlm4MoELayer(EPGroupLimitedMoELayerBase):
    """EP wrapper for GLM-4 MoE Lite's fused sigmoid-routed MoE using DeepEP.

    Fused gate_up_proj/down_proj routed experts, group-based routing with sigmoid activation, plus a
    shared expert that processes all tokens (not distributed). Laguna shares the whole shape and
    subclasses this wrapper (:class:`~src.distributed.expert_parallel.layers.laguna.EPLagunaMoELayer`),
    so the two details that differ per family are declarations rather than code: what the block names
    its shared expert, and where it hangs the routing knobs.
    """

    HF_MODULE_NAMES = ("Glm4MoeLiteMoE",)
    HF_MODEL_TYPES = ("glm4_moe_lite",)

    _supports_bias_balancing = True
    # The shared group-limited routing adds this exported native slot to the selection score itself
    # (selection-only, DeepSeek-V3 style), so bias-update adoption rides it and the trained bias
    # ships with every checkpoint — GLM-4's shipped configs all set ``moe_balancing: bias_update``.
    _NATIVE_BALANCING_BIAS_ATTR = "gate.e_score_correction_bias"
    # Hub checkpoint / vLLM loader layout: ``experts.{i}.{gate,up,down}_proj.weight`` (per expert).
    _PER_EXPERT_UNFUSED_KEYS = ("gate_proj", "up_proj", "down_proj")

    # The revisions this wrapper also serves declare no group limiting, so those two fall through to
    # the neutral 1; ``norm_topk_prob`` and ``routed_scaling_factor`` (1.8 in-library) stay required —
    # defaulting either silently rescales every routed weight.
    _OPTIONAL_ROUTING_KNOBS = ("n_group", "topk_group")

    _NUM_EXPERTS_ATTR_PATHS = ("routed_experts.num_experts", "config.num_local_experts")

    # The in-library ``Glm4MoeLiteMoE`` names the container ``experts``; the revisions this wrapper also
    # serves (and their checkpoints) spell it ``routed_experts``, which is why the expert-count path
    # above probes that name first. Laguna inherits both spellings.
    _EXPERTS_CONTAINER_ATTRS = ("experts", "routed_experts")

    # The in-library block names the shared expert in the plural; the hub remote-code revisions this
    # wrapper also serves use the singular. The adopted spelling becomes the export key.
    _SHARED_EXPERT_ATTRS = ("shared_experts", "shared_expert")

    def _native_slot_absence_is_legal(self) -> bool:
        """Only for a remote-code router: the revisions this wrapper also serves ship no correction
        bias at all, while the in-library ``Glm4MoeLiteTopkRouter`` (and Laguna's) registers it
        unconditionally — an absent slot there is a rename, and routing unbiased would be silent.

        Read with a default: this runs on the SELECTION path, which callers reach on layers that
        adopted no router module at all."""
        router = getattr(self, self._ROUTER_ATTR, None) if self._ROUTER_ATTR else None
        return not type(router).__module__.startswith("transformers.models.")

    def _init_routing(self, original_layer: nn.Module) -> None:
        """The shared group-limited knobs, plus the correction bias a lazy load can leave on meta.

        Laguna's pinned remote-code revision omits the correction bias from its checkpoint (the
        in-library format always writes it), so the parameter survives the load unmaterialized and
        every selection it biases would read meta storage."""
        correction_bias = getattr(self.gate, "e_score_correction_bias", None)
        if correction_bias is not None and correction_bias.is_meta:
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.zeros(correction_bias.shape, dtype=correction_bias.dtype),
                requires_grad=correction_bias.requires_grad,
            )
        super()._init_routing(original_layer)

    def _init_summary_extras(self, original_layer: nn.Module) -> tuple[str, ...]:
        return (f"hf_layer={type(original_layer).__name__}",)
