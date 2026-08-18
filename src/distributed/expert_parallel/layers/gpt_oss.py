"""EP wrapper for GptOss MoE (interleaved fused experts with bias)."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.distributed.expert_parallel.autograd import MoEExpertBiasGather
from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.kernels.fused_glu import fused_gptoss_glu


def interleave_gate_up(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Interleave a gate/up pair along the last axis: ``[g0,u0,g1,u1,...]``.

    Handles weights (``[E,H,M]`` → ``[E,H,2M]``) and biases (``[E,M]`` → ``[E,2M]``) alike, since the
    axis is the last one in both."""
    interleaved = torch.empty(*gate.shape[:-1], 2 * gate.shape[-1], device=gate.device, dtype=gate.dtype)
    interleaved[..., ::2] = gate
    interleaved[..., 1::2] = up
    return interleaved


def split_interleaved_gate_up(interleaved: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(gate, up)`` views of an interleaved trailing axis, the inverse of
    :func:`interleave_gate_up`, for weights and biases alike. Returns views; callers materialize
    with their own layout."""
    return interleaved[..., ::2], interleaved[..., 1::2]


class EPGptOssMoELayer(EPMoELayerBase):
    """EP wrapper for GptOss fused experts using DeepEP."""

    HF_MODULE_NAMES = ("GptOssMLP",)
    HF_MODEL_TYPES = ("gpt_oss",)

    # GptOss exports its fused experts in matmul convention [E, H, 2M] / [E, M, H] (nothing is
    # transposed on the way out), so the contraction dim is axis 1, not the last axis.
    HF_FUSED_EXPERT_CONTRACTION_AXIS = 1

    _supports_bias_balancing = True
    # The hub router's own logit-space bias (``GptOssTopKRouter.bias``). vLLM and SGLang both load it
    # and route with it (top-k on bias-inclusive logits, combine = softmax over the selected values).
    # Adoption re-registers the Parameter as a buffer, so under bias_update it is controller state
    # rather than a trained parameter.
    _NATIVE_BALANCING_BIAS_ATTR = "router.bias"

    _NUM_EXPERTS_ATTR_PATHS = ("experts.num_experts", "router.num_experts")

    # The one family that spells its router ``router`` rather than ``gate``.
    _ROUTER_ATTR = "router"

    # The de-interleaved gate/up pair (plus biases) the grouped-mm path stores instead of the
    # interleaved ``gate_up_proj``. The weight-attr roots below and the shard merge's accepted-key
    # set both derive from this tuple.
    _GMM_EXPERT_KEYS: tuple[str, ...] = (
        "gate_proj_gmm",
        "up_proj_gmm",
        "gate_proj_gmm_bias",
        "up_proj_gmm_bias",
    )

    # Extends the base roots rather than restating them: a root added to the base and missed here
    # would drop an expert weight from grad-sync registration, diverging DP replicas with no error.
    # The additions cover GptOss's biases and the grouped-mm de-interleaved pair; the inherited
    # hasattr-filter picks whichever of the three mutually-exclusive layouts this rank holds.
    _EXPERT_WEIGHT_ATTR_ROOTS: tuple[str, ...] = (
        EPMoELayerBase._EXPERT_WEIGHT_ATTR_ROOTS
        + ("gate_up_proj_bias",)
        + _GMM_EXPERT_KEYS
        + ("gate_proj_bias", "up_proj_bias", "down_proj_bias")
    )

    def _init_expert_compute(self, original_layer: nn.Module, experts: nn.Module) -> None:
        """GptOss implements its own interleaved clamped-SwiGLU compute paths, so it takes these two
        constants instead of an activation. Read as attributes, not with defaults: a substituted
        default corrupts the clamped SwiGLU on every expert."""
        self.alpha = experts.alpha
        self.limit = experts.limit

    def _init_expert_params(self, experts: nn.Module, weights_already_sharded: bool = False):
        """Slice and register expert parameters for this rank.

        GptOss gate_up_proj is interleaved ``[g0,u0,g1,u1,...]``. Under ETP and grouped_mm it is
        de-interleaved into separate gate_proj/up_proj, because stride-2 slicing on grouped_mm
        outputs gives NaN backward gradients. ``weights_already_sharded`` skips dim-0 slicing. Expert
        biases are optional (a bias-less GptOss variant is legal); the weights are not.
        """
        self._require_fused_experts(experts)
        if weights_already_sharded:
            start, end = 0, self.experts_per_rank
        else:
            start, end = self.expert_start, self.expert_end

        if self.expert_tp_size > 1:
            self._init_expert_params_tp(experts, start, end)
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

    def _init_expert_params_tp(self, experts: nn.Module, start: int, end: int):
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
        """GptOss runs its own interleaved-bias compute paths and never reaches the base GLU seam."""
        return "fused_gptoss_glu"

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
        return fused_gptoss_glu(gate, up, self.alpha, self.limit)

    @classmethod
    def merge_shards_to_hf(cls, prefix: str, params: dict) -> dict:
        """Merged GptOss shards → the interleaved HF layout (see the base method).

        GptOss already stores matmul convention, so nothing is transposed; the only work is undoing
        the grouped-GEMM de-interleave the runtime applies on SM90+ (``gate_proj_gmm`` /
        ``up_proj_gmm`` back to one interleaved ``gate_up_proj``), as the gather below does. Raises
        on any expert param outside those two layouts, mirroring the base merge.
        """
        handled = set(cls._HF_FUSED_EXPERT_KEYS) | set(cls._GMM_EXPERT_KEYS)
        unexpected = set(params) - handled
        if unexpected:
            raise ValueError(
                f"{cls.__name__} sharded merge found unexpected expert params {sorted(unexpected)} — "
                f"this merge covers exactly {sorted(handled)}; refusing to drop them."
            )
        result = {}
        consumed: set[str] = set()
        if "gate_proj_gmm" in params:
            result[f"{prefix}.experts.gate_up_proj"] = interleave_gate_up(
                params["gate_proj_gmm"], params["up_proj_gmm"]
            )
            consumed |= {"gate_proj_gmm", "up_proj_gmm"}
            if "gate_proj_gmm_bias" in params:
                result[f"{prefix}.experts.gate_up_proj_bias"] = interleave_gate_up(
                    params["gate_proj_gmm_bias"], params["up_proj_gmm_bias"]
                )
                consumed |= {"gate_proj_gmm_bias", "up_proj_gmm_bias"}
        elif "gate_up_proj" in params:
            result[f"{prefix}.experts.gate_up_proj"] = params["gate_up_proj"]
            consumed.add("gate_up_proj")
            if "gate_up_proj_bias" in params:
                result[f"{prefix}.experts.gate_up_proj_bias"] = params["gate_up_proj_bias"]
                consumed.add("gate_up_proj_bias")

        if "down_proj" in params:
            result[f"{prefix}.experts.down_proj"] = params["down_proj"]
            consumed.add("down_proj")
        if "down_proj_bias" in params:
            result[f"{prefix}.experts.down_proj_bias"] = params["down_proj_bias"]
            consumed.add("down_proj_bias")

        # An accepted key with no branch would vanish from the merged checkpoint, which then loads
        # and is wrong, the same failure the vocabulary check above covers.
        unmapped = set(params) - consumed
        if unmapped:
            raise ValueError(
                f"{cls.__name__} sharded merge accepted declared expert params {sorted(unmapped)} but "
                f"has no merge branch for them — refusing to drop them from the merged checkpoint. "
                f"Add the layout branch alongside the key declaration."
            )
        return result

    def gather_expert_state_dict(self, device: str = "cpu", merge_lora: bool = False, retain: bool = True) -> dict:
        """Gather GptOss experts into the interleaved checkpoint layout."""
        self._reject_merge_lora_under_expert_tp(merge_lora)
        return self._gather_interleaved_experts(
            device, partial(self._materialize_expert_weight, merge_lora=merge_lora), retain=retain
        )

    def gather_fused_expert_state_dict(
        self, device: str = "cpu", merge_lora: bool = False, retain: bool = True
    ) -> dict:
        """GptOss's checkpoint layout is already the fused one (interleaved gate/up + biases), so the
        gather a fused-layout engine loads and the checkpoint gather coincide."""
        return self.gather_expert_state_dict(device, merge_lora=merge_lora, retain=retain)

    def gather_expert_grads(self, device: str = "cpu") -> dict:
        """Expert gradients in the same interleaved layout (see the base method)."""
        return self._gather_interleaved_experts(device, self._expert_grad)

    def _gather_interleaved_experts(
        self, device: str, take: Callable[[str], torch.Tensor], retain: bool = True
    ) -> dict:
        """Reassemble ``experts.gate_up_proj [E,H,2M]`` + ``experts.down_proj [E,M,H]`` and biases from
        whichever internal layout this rank holds, taking each attribute's tensor from ``take``.

        GptOss already stores matmul convention (no transpose). ``down_proj_bias`` lives only on
        ``expert_tp_rank == 0``, so it is EP-gathered only: TP-gathering would issue a collective the
        bias-less TP ranks cannot satisfy. The grouped-GEMM branch re-interleaves from the
        de-interleaved layout the adapter (and the gradient) lives in. ``retain=False`` runs every
        gather and keeps nothing (see :meth:`~EPMoELayerBase.gather_expert_state_dict`).
        """
        gate_up_bias = None
        if self.expert_tp_size > 1:
            gate_up = interleave_gate_up(
                self._tp_all_gather_cat(take("gate_proj"), dim=2), self._tp_all_gather_cat(take("up_proj"), dim=2)
            )
            down = self._tp_all_gather_cat(take("down_proj"), dim=1)
            if hasattr(self, "gate_proj_bias"):
                gate_up_bias = interleave_gate_up(
                    self._tp_all_gather_cat(take("gate_proj_bias"), dim=1),
                    self._tp_all_gather_cat(take("up_proj_bias"), dim=1),
                )
        elif hasattr(self, "gate_proj_gmm"):
            gate_up = interleave_gate_up(take("gate_proj_gmm"), take("up_proj_gmm"))
            down = take("down_proj")
            if hasattr(self, "gate_proj_gmm_bias"):
                gate_up_bias = interleave_gate_up(take("gate_proj_gmm_bias"), take("up_proj_gmm_bias"))
        else:
            gate_up = take("gate_up_proj")
            down = take("down_proj")
            if hasattr(self, "gate_up_proj_bias"):
                gate_up_bias = take("gate_up_proj_bias")

        gathered = {
            "experts.gate_up_proj": self._ep_all_gather_cat(gate_up),
            "experts.down_proj": self._ep_all_gather_cat(down),
        }
        if gate_up_bias is not None:
            gathered["experts.gate_up_proj_bias"] = self._ep_all_gather_cat(gate_up_bias)
        if hasattr(self, "down_proj_bias"):
            gathered["experts.down_proj_bias"] = self._ep_all_gather_cat(take("down_proj_bias"))
        if not retain:
            return {}
        return {key: tensor.to(device) for key, tensor in gathered.items()}

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

        bias = getattr(self, "balancing_biases", None)
        with torch.amp.autocast("cuda", enabled=False):
            router_input = flat.float() if self._fp32_router_input else flat
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
                weights = weights.float()
                experts = experts.long()
                if self._forced_topk_indices is not None:
                    # Replay re-derives weights from live logits, keeping the router grad live.
                    experts = self._maybe_replace_selection(experts)
                    weights = F.softmax(logits.float().gather(-1, experts), dim=-1)

        result = self._dispatch_compute_combine(flat, experts, weights, input_dtype)
        return result.view(B, S, H), logits.to(input_dtype)

    def _expert_forward(self, idx: int, x: torch.Tensor) -> torch.Tensor:
        """One local expert, read from whichever layout this rank stores (see the base method): ETP
        keeps gate/up de-interleaved under the plain names, every other configuration keeps the pair
        interleaved."""
        if self.expert_tp_size > 1:
            return self._expert_forward_tp(idx, x)
        gate_up = self._expert_proj_single(idx, x, "gate_up_proj")
        if hasattr(self, "gate_up_proj_bias"):
            gate_up = gate_up + self.gate_up_proj_bias[idx]
        gated = fused_gptoss_glu(*split_interleaved_gate_up(gate_up), self.alpha, self.limit)
        out = self._expert_proj_single(idx, gated, "down_proj")
        if hasattr(self, "down_proj_bias"):
            out = out + self.down_proj_bias[idx]
        return out

    def _expert_forward_tp(self, idx: int, x: torch.Tensor) -> torch.Tensor:
        """GptOss expert forward with de-interleaved TP-sharded weights."""
        gate = self._expert_proj_single(idx, x, "gate_proj")
        up = self._expert_proj_single(idx, x, "up_proj")
        if hasattr(self, "gate_proj_bias"):
            gate = gate + self.gate_proj_bias[idx]
        if hasattr(self, "up_proj_bias"):
            up = up + self.up_proj_bias[idx]
        gated = fused_gptoss_glu(gate, up, self.alpha, self.limit)
        out = self._expert_proj_single(idx, gated, "down_proj")
        if hasattr(self, "down_proj_bias"):
            out = out + self.down_proj_bias[idx]
        return out

    def _compute_experts_gmm(
        self, tokens: torch.Tensor, experts: torch.Tensor, weights: torch.Tensor, output_dtype: torch.dtype
    ) -> torch.Tensor:
        """Grouped GEMM for GptOss experts: 3 calls (gate + up + down) with de-interleaved weights."""

        def compute(sorted_tokens, offs, eids):
            gate = self._expert_proj(sorted_tokens, "gate_proj_gmm", offs, output_dtype)
            up = self._expert_proj(sorted_tokens, "up_proj_gmm", offs, output_dtype)
            # MoEExpertBiasGather's backward is an atomic-free GEMM: index_add's value, no bf16 atomic.
            if hasattr(self, "gate_proj_gmm_bias"):
                gate = gate + MoEExpertBiasGather.apply(self.gate_proj_gmm_bias, eids)
            if hasattr(self, "up_proj_gmm_bias"):
                up = up + MoEExpertBiasGather.apply(self.up_proj_gmm_bias, eids)
            gated = fused_gptoss_glu(gate, up, self.alpha, self.limit)
            out = self._expert_proj(gated.to(output_dtype), "down_proj", offs, output_dtype)
            if hasattr(self, "down_proj_bias"):
                out = out + MoEExpertBiasGather.apply(self.down_proj_bias, eids)
            return out

        return self._compute_experts_with_grouped_mm(tokens, experts, weights, output_dtype, compute)

    def _grouped_mm_enabled(self) -> bool:
        # ETP stores gate/up under the plain gate_proj/up_proj names the per-expert loop reads, not the
        # gate_proj_gmm/up_proj_gmm pair _compute_experts_gmm needs; _log_init_summary reports the drop.
        return self._use_grouped_mm and self.expert_tp_size <= 1
