"""EP wrapper for GLM-5 Next MoE (fused experts, clamped SwiGLU, sigmoid noaux-tc routing)."""

from __future__ import annotations

from functools import partial

import torch.nn as nn

from src.distributed.expert_parallel.base_layer import EPGroupLimitedMoELayerBase
from src.kernels.fused_glu import fused_clamped_silu_mul


class EPGlm5NextMoELayer(EPGroupLimitedMoELayerBase):
    """EP wrapper for GLM-5 Next MoE (``Glm5NextTextMoE``, GLM-5.3-Flash) using DeepEP.

    Fused ``gate_up_proj [E, 2M, H]`` (contiguous ``[gate; up]`` halves) / ``down_proj [E, H, M]``
    routed experts with the family's clamped SwiGLU (``swiglu_limit``), DeepSeek-V3-style sigmoid
    routing whose ``e_score_correction_bias`` buffer perturbs selection only
    (``Glm5NextTextTopkRouter`` returns a ``(logits, weights, indices)`` 3-tuple; the wrapper reuses
    the logits and re-derives the rest), and one always-on shared expert (``Glm5NextTextMLP``, which
    carries its own clamp). The first ``first_k_dense_replace`` decoder layers are plain MLPs and are
    never wrapped.
    """

    HF_MODULE_NAMES = ("Glm5NextTextMoE",)
    # The text tower's spelling plus the composite VLM wrapper spelling GLM-5.3-Flash checkpoints
    # write at the top level; checkpoint-keyed tooling (sharded merge, unfuse, lazy-load routing)
    # resolves either config to this class.
    HF_MODEL_TYPES = ("glm5_next", "glm5_next_text")

    # The wrapper re-derives selection from the router's logits, so it can bias top-k itself.
    # ``_ep_severs_aux_loss`` stays False: ``_router_logits`` calls the real ``Glm5NextTextTopkRouter``
    # module, so the HF OutputRecorder still populates ``outputs.router_logits`` and the family's aux
    # loss keeps working under EP, which is where ``moe_balancing: auto`` resolves.
    _supports_bias_balancing = True
    # The shared group-limited routing adds this native slot (an fp32 ``nn.Buffer`` upstream) to the
    # selection score (selection-only, DeepSeek-V3 style) and the slot is part of the checkpoint.
    _NATIVE_BALANCING_BIAS_ATTR = "gate.e_score_correction_bias"

    # Hub layout: ``experts.{i}.{gate,up,down}_proj.weight`` per expert; transformers' glm5_next
    # conversion mapping fuses those into ``experts.gate_up_proj`` on load and reverts on save. The
    # gather writes the fused tensor, so ``unfuse_moe_experts`` repairs a save that bypassed the
    # revert under these names.
    _HUB_PER_EXPERT_KEYS = ("gate_proj", "up_proj", "down_proj")

    # ``gate.weight`` is ``[num_experts, hidden]``; the config spells the count
    # ``num_local_experts`` (attribute-mapped onto its stored ``n_routed_experts``).
    _NUM_EXPERTS_ATTR_PATHS = ("gate.num_experts", "config.num_local_experts")

    # The hub checkpoint keeps the vendor namespace transformers converts inside ``from_pretrained``:
    # hyper-connection ``hc_attn_*``/``hc_ffn_*`` tensors, the KDA ``f_a_proj``/``A_log`` family
    # under ``forget_gate``, and the three-source ``q/k/v_conv1d → conv1d`` Concatenate. The lazy
    # loaders replay that mapping per key (``hub_conversion.py``); the per-expert entries are the
    # ExpertFuser's.
    _HUB_CONVERSION_KEYS = ("glm5_next",)

    # The live module tree spells the KDA/hyper-connection tensors differently from the checkpoint
    # namespace a serving engine reads (the same from_pretrained-only conversion as above), so a
    # sync would land nowhere; no pinned rollout engine loads glm5_next either.
    _supports_weight_sync = False

    # The block always builds its shared expert, so an absent one is an upstream rename rather than a
    # configuration, and adopting nothing would drop it from every output.
    _SHARED_EXPERT_ATTRS = ("shared_experts",)
    _SHARED_EXPERT_REQUIRED = True

    def _init_expert_compute(self, original_layer: nn.Module, experts: nn.Module) -> None:
        """``Glm5NextTextExperts._apply_gate`` implements clamp-then-SiLU with no act_fn on the module
        and no config branch, so the combine latched into the base seam is unconditional; only the
        clamp bound comes off the live experts module, read as an attribute so a rename raises."""
        self.swiglu_limit = float(experts.swiglu_limit)
        self._fused_glu_mul = partial(fused_clamped_silu_mul, limit=self.swiglu_limit)

    def _init_summary_extras(self, original_layer: nn.Module) -> tuple[str, ...]:
        return (f"swiglu_limit={self.swiglu_limit}",)
