"""EP wrapper for DeepSeek-V4 MoE (fused experts, clamped SwiGLU, hash + top-k routing, shared expert)."""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.kernels.fused_glu import fused_clamped_silu_mul, is_silu_activation


class EPDeepseekV4MoELayer(EPMoELayerBase):
    """EP wrapper for DeepSeek-V4 fused experts (``DeepseekV4SparseMoeBlock``) using DeepEP.

    Fused ``gate_up_proj [E, 2M, H]`` (contiguous ``[gate; up]`` halves) / ``down_proj [E, H, M]``
    routed experts with the clamped-SwiGLU combine, a replicated ``DeepseekV4MLP`` shared expert, and
    two router variants riding along as the ``gate`` module (adopted whole, so
    ``e_score_correction_bias``/``tid2eid`` stay checkpointable): ``DeepseekV4TopKRouter``
    (``is_hash=False``) selects on ``scores + e_score_correction_bias`` and gates on the unbiased
    scores; ``DeepseekV4HashRouter`` (``is_hash=True``) selects via the frozen ``tid2eid[input_ids]``
    lookup, which is why this wrapper subclasses :class:`EPMoELayerBase` directly rather than the
    shared-experts forward base (the decoder passes ``input_ids`` into the MoE block).
    """

    HF_MODULE_NAMES = ("DeepseekV4SparseMoeBlock",)
    HF_MODEL_TYPES = ("deepseek_v4",)

    _supports_bias_balancing = True
    # ``_select_experts`` adds this exported native slot to the selection score itself (top-k
    # variant only; hash routing refuses balancing), so bias-update adoption rides it and the
    # trained bias ships with every checkpoint.
    _NATIVE_BALANCING_BIAS_ATTR = "gate.e_score_correction_bias"
    # Routing is re-derived from ``gate.weight``, so ``outputs.router_logits`` stays empty under EP.
    _ep_severs_aux_loss = True
    # vLLM's V4 loader targets the ORIGINAL fp8/fp4-packed release layout — unbridgeable by a gather.
    _supports_weight_sync = False

    # transformers registers ~32 vendor-namespace renames (^embed\.weight$ → embed_tokens.weight, …)
    # plus the expert merges for this family. Not every source is vendor-anchored (5.16's `\.norm\.`
    # → `.kv_norm.` also matches the canonical final norm), so a canonical checkpoint passes through
    # untouched via the mapping's model-key fallback (``build_key_mapping``), not by
    # pattern anchoring.
    _HUB_CONVERSION_KEYS = ("deepseek_v4",)

    # Hub layout: ``experts.{i}.w{1,3,2}.weight`` per expert (LFM-2's spelling), per transformers' own
    # converter for this family. Distinct from ``_supports_weight_sync`` above: vLLM reads the packed
    # release format, but a transformers-side reload of a gathered save does read these names.
    _HUB_PER_EXPERT_KEYS = ("w1", "w3", "w2")

    _NUM_EXPERTS_ATTR_PATHS = ("experts.num_experts", "gate.num_experts")

    _SHARED_EXPERT_ATTRS = ("shared_experts",)

    def _init_routing(self, original_layer: nn.Module) -> None:
        """Top-k plus the router variant and its scoring: hash layers route off the frozen
        ``tid2eid`` table, top-k layers off ``score_fn`` over the gate's logits."""
        super()._init_routing(original_layer)
        self.is_hash = bool(getattr(original_layer, "is_hash", False))
        self.routed_scaling_factor = getattr(self.gate, "routed_scaling_factor", 1.0)
        self.score_fn = self.gate.score_fn

        if self.is_hash and self.gate.tid2eid.numel() > 0 and not self.gate.tid2eid.is_meta:
            # DeepEP dispatch asserts each token's top-k expert ids are DISTINCT (device-side
            # `ptx::deduplicate`), so a duplicate row must fail here, not crash the CUDA context.
            sorted_rows = self.gate.tid2eid.sort(dim=-1).values
            if self.top_k > 1 and bool((sorted_rows[:, 1:] == sorted_rows[:, :-1]).any()):
                raise ValueError(
                    "EPDeepseekV4MoELayer: the hash router's tid2eid table assigns DUPLICATE expert "
                    "ids to at least one token id. DeepEP dispatch requires distinct experts per "
                    "token — fix the tid2eid table (each row must hold top_k distinct expert ids)."
                )

    def _init_expert_compute(self, original_layer: nn.Module, experts: nn.Module) -> None:
        """Read directly (AttributeError = fail loud): DeepseekV4Experts holds config.hidden_act and
        config.swiglu_limit here. A substituted default silently rescales the clamped SwiGLU on
        every expert, and a substituted SiLU also selects the SiLU-only Triton combine.

        The clamp is latched into the base combine seam rather than overriding it, so the construction
        summary names the callable that actually runs. The fused kernel hardcodes SiLU; any other gate
        falls through to the eager form."""
        self.act_fn = experts.act_fn
        self.limit = float(experts.limit)
        self._fused_glu_mul = (
            partial(fused_clamped_silu_mul, limit=self.limit)
            if is_silu_activation(self.act_fn)
            else self._clamped_swiglu_eager
        )

    def _clamped_swiglu_eager(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """The family's combine for a gate no kernel implements — same clamps, this family's act_fn."""
        return self.act_fn(gate.clamp(max=self.limit)) * up.clamp(min=-self.limit, max=self.limit)

    def _init_summary_extras(self, original_layer: nn.Module) -> tuple[str, ...]:
        return (f"is_hash={self.is_hash}",)

    def enable_bias_balancing(self) -> bool:
        """Hash layers keep the frozen ``tid2eid`` selection — a balancing bias could never shift it,
        so only top-k layers create the bias state (the callback then only sees shiftable routers)."""
        if self.is_hash:
            return False
        return super().enable_bias_balancing()

    def _select_experts(self, scores: torch.Tensor, input_ids: torch.Tensor | None) -> torch.Tensor:
        """Top-k indices ``[T, top_k]`` per the router variant (hash lookup or biased top-k)."""
        if self.is_hash:
            if input_ids is None:
                raise RuntimeError(
                    "EPDeepseekV4MoELayer: this is a hash_moe layer (frozen tid2eid routing) but the "
                    "forward received no input_ids — the caller ran the model from inputs_embeds, "
                    "which DeepSeek-V4 hash routing cannot support. Pass input_ids."
                )
            return self.gate.tid2eid[input_ids.reshape(-1)].long()
        return torch.topk(self._selection_scores(scores), self.top_k, dim=-1, sorted=False).indices

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        orig_shape = hidden_states.shape
        _B, _S, H = orig_shape
        input_dtype = hidden_states.dtype

        residuals = hidden_states  # shared expert reads the layer input, not the routed output
        flat = hidden_states.view(-1, H)

        # Disable autocast so the gate matmul runs at the input dtype (upcast when fp32_router).
        with torch.amp.autocast("cuda", enabled=False):
            logits = F.linear(flat.float() if self._fp32_router_input else flat, self.gate.weight)

        scores = self.score_fn(logits.float())
        indices = self._select_experts(scores, input_ids)
        # Replay stays in lockstep across EP layers, so hash layers consume their identity slice too.
        indices = self._maybe_replace_selection(indices)

        # Gate on the UNBIASED scores at the selected indices; V4 always normalizes (+1e-20 floor).
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * self.routed_scaling_factor

        self._record_expert_load(indices)

        shared_fn = (lambda: self.shared_experts(residuals)) if self.shared_experts is not None else None
        return self._dispatch_compute_combine_shared(
            flat, indices.long(), weights.float(), input_dtype, orig_shape, shared_fn
        )
