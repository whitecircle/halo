"""EP wrapper for GptOss MoE (interleaved fused experts with bias)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from src.kernels.fused_glu import GluCombine, fused_glu_enabled
from src.kernels.moe_permute import MoEExpertBiasGather
from src.models.moe.balancing import BALANCING_BIASES_ATTR
from src.models.moe.expert_layout import split_interleaved_gate_up
from src.models.moe.family_specs import GPT_OSS


class EPGptOssMoELayer(EPMoELayerBase):
    """EP wrapper for GptOss fused experts using DeepEP."""

    HF_MODULE_NAMES = ("GptOssMLP",)
    FAMILY = GPT_OSS

    SUPPORTS_BIAS_BALANCING = True

    _NUM_EXPERTS_ATTR_PATHS = ("experts.num_experts", "router.num_experts")

    # The one family that spells its router ``router`` rather than ``gate``.
    _ROUTER_ATTR = "router"

    def _adopt_expert_compute(self, original_layer: nn.Module, experts: nn.Module) -> None:
        """GptOss implements its own interleaved clamped-SwiGLU compute paths, so it latches that combine
        instead of an activation. ``alpha`` and ``limit`` are read as attributes, not with defaults: a
        substituted default corrupts the clamped SwiGLU on every expert."""
        self._gptoss_glu = GluCombine.gptoss(experts.alpha, experts.limit)

    def _adopt_expert_params(self, experts: nn.Module, weights_already_sharded: bool = False):
        """Slice and register expert parameters for this rank.

        GptOss gate_up_proj is interleaved ``[g0,u0,g1,u1,...]``. Under ETP and grouped_mm it is
        de-interleaved into separate gate_proj/up_proj, because stride-2 slicing on grouped_mm
        outputs gives NaN backward gradients. ``weights_already_sharded`` skips dim-0 slicing. Expert
        biases are optional (a bias-less GptOss variant is legal); the weights are not.
        """
        self._require_fused_experts(experts)
        start, end = self._local_expert_range(weights_already_sharded)

        if self.expert_tp_size > 1:
            self._adopt_expert_params_tp(experts, start, end)
            return

        local_gate_up = experts.gate_up_proj.data[start:end]
        if self._use_grouped_mm:
            gate, up = split_interleaved_gate_up(local_gate_up)
            self.gate_proj_gmm = nn.Parameter(gate.contiguous())
            self.up_proj_gmm = nn.Parameter(up.contiguous())
            if getattr(experts, "gate_up_proj_bias", None) is not None:
                gate_bias, up_bias = split_interleaved_gate_up(experts.gate_up_proj_bias.data[start:end])
                self.gate_proj_gmm_bias = nn.Parameter(gate_bias.contiguous())
                self.up_proj_gmm_bias = nn.Parameter(up_bias.contiguous())
        else:
            self.gate_up_proj = nn.Parameter(local_gate_up.clone())
            if getattr(experts, "gate_up_proj_bias", None) is not None:
                self.gate_up_proj_bias = nn.Parameter(experts.gate_up_proj_bias.data[start:end].clone())

        self.down_proj = nn.Parameter(experts.down_proj.data[start:end].clone())
        if getattr(experts, "down_proj_bias", None) is not None:
            self.down_proj_bias = nn.Parameter(experts.down_proj_bias.data[start:end].clone())

    def _adopt_expert_params_tp(self, experts: nn.Module, start: int, end: int):
        """TP-sharded expert params: de-interleave gate_up into gate/up and shard the intermediate
        dim; down_proj shards on dim 1. start/end is the dim-0 slice."""
        tp_rank = self.expert_tp_rank

        # [E, H, 2M] → two [E, H, M]; the biases likewise [E, 2M] → two [E, M].
        gate, up = split_interleaved_gate_up(experts.gate_up_proj.data[start:end])
        self.gate_proj = nn.Parameter(self._etp_narrow(gate.contiguous(), 2).clone())
        self.up_proj = nn.Parameter(self._etp_narrow(up.contiguous(), 2).clone())

        if getattr(experts, "gate_up_proj_bias", None) is not None:
            gate_bias, up_bias = split_interleaved_gate_up(experts.gate_up_proj_bias.data[start:end])
            self.gate_proj_bias = nn.Parameter(self._etp_narrow(gate_bias.contiguous(), 1).clone())
            self.up_proj_bias = nn.Parameter(self._etp_narrow(up_bias.contiguous(), 1).clone())

        full_down = experts.down_proj.data[start:end]
        self.down_proj = nn.Parameter(self._etp_narrow(full_down, 1).clone())

        if getattr(experts, "down_proj_bias", None) is not None and tp_rank == 0:
            self.down_proj_bias = nn.Parameter(experts.down_proj_bias.data[start:end].clone())

    def _glu_combine_name(self) -> str:
        """GptOss runs its own interleaved-bias compute paths and never reaches the base GLU seam; every
        path calls its latched combine, eager under ``HALO_FUSED_GLU=0``."""
        return self._gptoss_glu.name if fused_glu_enabled() else "eager"

    def _gate_weights_at(self, router_logits: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """``GptOssTopKRouter`` weights at ``indices``: the softmax over the selected logits."""
        return F.softmax(router_logits.float().gather(-1, indices), dim=-1)

    def _route_with_bias(self, router_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Balanced routing through the adopted native ``router.bias``.

        The bias enters via ``F.linear`` as it does at serving, so selection is the native top-k over
        bias-inclusive logits and the gate the native softmax over the selected values."""
        # The adopted bias integrates sign-steps in fp32 but F.linear rejects a mixed-dtype bias, and
        # under ep1+fsdp_shard_ep1_experts the router computes in bf16 (see ``_fp32_router_input``):
        # cast at the read, not at the state. ``None``-safe, since a transient-fallback route applies
        # its side-buffer inside ``_deepseek_biased_route`` on a possibly bias-less router.
        bias = self.router.bias
        logits = F.linear(router_input, self.router.weight, bias if bias is None else bias.to(router_input.dtype))
        weights, experts = self._deepseek_biased_route(logits)
        self._record_expert_load(experts)
        return logits, weights, experts.long()

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        B, S, H = hidden_states.shape
        input_dtype = hidden_states.dtype
        flat = hidden_states.view(-1, H)

        bias = getattr(self, BALANCING_BIASES_ATTR, None)
        with torch.amp.autocast("cuda", enabled=False):
            router_input = self._router_input(flat)
            if bias is not None:
                logits, weights, experts = self._route_with_bias(router_input)
            else:
                router_out = self.router(router_input)
                if not (isinstance(router_out, tuple) and len(router_out) == 3):
                    raise RuntimeError(
                        f"GptOss router returned {type(router_out)} (expected a 3-tuple of "
                        f"logits/scores/indices) — transformers changed the router contract; "
                        f"update EPGptOssMoELayer."
                    )
                logits, weights, experts = router_out
                experts, weights = self._replay_router_choice(logits, experts.long(), weights.float())

        result = self._dispatch_compute_combine(flat, experts, weights, input_dtype)
        return result.view(B, S, H), logits.to(input_dtype)

    def _expert_forward(self, idx: int, x: torch.Tensor) -> torch.Tensor:
        """One local expert, read from whichever layout this rank stores (see the base method): ETP
        keeps gate/up de-interleaved under the plain names, every other configuration keeps the pair
        interleaved."""
        if self.expert_tp_size > 1:
            gated = self._gptoss_glu(
                self._proj_with_bias(idx, x, "gate_proj"), self._proj_with_bias(idx, x, "up_proj")
            )
        else:
            gated = self._gptoss_glu(*split_interleaved_gate_up(self._proj_with_bias(idx, x, "gate_up_proj")))
        return self._proj_with_bias(idx, gated, "down_proj")

    def _proj_with_bias(self, idx: int, x: torch.Tensor, weight_attr: str) -> torch.Tensor:
        """Loop-path projection plus the expert's ``<weight_attr>_bias`` where this rank holds one."""
        out = self._expert_proj_single(idx, x, weight_attr)
        bias = getattr(self, f"{weight_attr}_bias", None)
        return out if bias is None else out + bias[idx]

    def _compute_experts_gmm(
        self, tokens: torch.Tensor, experts: torch.Tensor, weights: torch.Tensor, output_dtype: torch.dtype
    ) -> torch.Tensor:
        """Grouped GEMM for GptOss experts: 3 calls (gate + up + down) with de-interleaved weights."""

        def compute(sorted_tokens, offs, eids):
            gate = self._gmm_proj_with_bias(sorted_tokens, "gate_proj_gmm", offs, output_dtype, eids)
            up = self._gmm_proj_with_bias(sorted_tokens, "up_proj_gmm", offs, output_dtype, eids)
            gated = self._gptoss_glu(gate, up)
            return self._gmm_proj_with_bias(gated.to(output_dtype), "down_proj", offs, output_dtype, eids)

        return self._compute_experts_with_grouped_mm(tokens, experts, weights, output_dtype, compute)

    def _gmm_proj_with_bias(
        self, x: torch.Tensor, weight_attr: str, offs: torch.Tensor, out_dtype: torch.dtype, eids: torch.Tensor
    ) -> torch.Tensor:
        """Grouped-GEMM projection plus the per-token gather of ``<weight_attr>_bias`` where this rank holds one."""
        out = self._expert_proj(x, weight_attr, offs, out_dtype)
        bias = getattr(self, f"{weight_attr}_bias", None)
        # MoEExpertBiasGather's backward is an atomic-free GEMM: index_add's value, no bf16 atomic.
        return out if bias is None else out + MoEExpertBiasGather.apply(bias, eids)

    def _grouped_mm_enabled(self) -> bool:
        # ETP stores gate/up under the plain gate_proj/up_proj names the per-expert loop reads, not the
        # gate_proj_gmm/up_proj_gmm pair _compute_experts_gmm needs; _log_init_summary reports the drop.
        return self._use_grouped_mm and self.expert_tp_size <= 1

    def _warm_expert_activation(self, gate_up: torch.Tensor) -> torch.Tensor:
        """Warm whichever activation this layer's compute path calls (see the base method).

        Every path runs the same Triton kernel and differs only in how the halves reach it:
        grouped-GEMM and ETP store them de-interleaved and contiguous, the per-expert loop splits
        them out of one interleaved projection output.
        """
        if self._grouped_mm_enabled() or self.expert_tp_size > 1:
            gate, up = (half.contiguous() for half in gate_up.chunk(2, dim=-1))
        else:
            gate, up = split_interleaved_gate_up(gate_up)
        return self._gptoss_glu(gate, up)
