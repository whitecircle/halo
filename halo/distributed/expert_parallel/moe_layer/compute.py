"""Expert compute for EP MoE layers: GLU combine, per-expert loop and grouped-GEMM paths, weights, kernel warm-up."""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch
import torch.nn as nn
from transformers.activations import ACT2FN

from halo.distributed.expert_parallel.config import EPConfig
from halo.kernels.expert_gemm import expert_gemm
from halo.kernels.fused_glu import GluCombine, fused_glu_enabled, resolve_fused_glu_mul
from halo.kernels.grouped_mm import GROUPED_MM_SMALL_TILE_DIM
from halo.kernels.lowp.spec import QuantFormat
from halo.kernels.moe_permute import MoEGatherPermute, MoEWeightedUnpermute, warm_permute_kernels
from halo.kernels.moe_route_sort import RECEIVE_COUNTS_DISAGREE, sort_routing_by_expert, warm_route_sort_kernels
from halo.models.moe.expert_layout import ExpertLayoutKind
from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)

# Tokens the expert-kernel warmup runs. The kernels do not specialize on their row count
# (``do_not_specialize``), so one count compiles the binary every dispatch size runs.
_WARMUP_TOKENS = 16


class EPExpertComputeMixin:
    """Expert compute, mixed into :class:`EPMoELayerBase`, whose construction builds the state annotated here."""

    ep_config: EPConfig
    ep_size: int
    experts_per_rank: int
    expert_start: int
    expert_end: int
    expert_tp_size: int
    expert_tp_rank: int
    fp32_experts: bool
    act_fn: Callable[[torch.Tensor], torch.Tensor]
    _use_grouped_mm: bool
    _expert_kernels_warmed: bool
    _lowp_quant_format: QuantFormat | None
    _lowp_weight_cacheable: bool
    _expert_lora_attrs: frozenset[str]
    _expert_adapters_enabled: bool
    expert_lora_scaling: float
    expert_lora_dropout: nn.Module

    # Fused GLU-combine kernel, derived from the resolved activation by ``_resolve_activation``; None = eager.
    _fused_glu_mul: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None

    def set_lowp_compute(self, quant_format: QuantFormat, *, weight_cacheable: bool) -> bool:
        """Set this layer's expert-compute quantization format; return whether it applies, False on the
        per-expert loop, where the grouped kernels and their fake-quant never run."""
        self._lowp_quant_format = quant_format
        self._lowp_weight_cacheable = weight_cacheable
        return self._use_grouped_mm

    def _resolve_activation(self, source: nn.Module, *, original_layer: nn.Module) -> None:
        """Set ``act_fn`` from ``source``, else by name off ``original_layer`` or its config, and latch the fused combine
        that activation has. Naming none raises, since a default would train another non-linearity."""
        act_fn = getattr(source, "act_fn", None)
        if act_fn is None:
            hidden_act = getattr(original_layer, "hidden_act", None)
            if hidden_act is None:
                hidden_act = getattr(getattr(original_layer, "config", None), "hidden_act", None)
            if hidden_act is None:
                raise AttributeError(
                    f"{type(self).__name__}: neither {type(source).__name__}.act_fn nor a hidden_act on "
                    f"{type(original_layer).__name__} or its config names the expert activation."
                )
            act_fn = ACT2FN[hidden_act]
        self.act_fn = act_fn
        self._fused_glu_mul = resolve_fused_glu_mul(act_fn)

    def _init_fused_glu_params(self, experts: nn.Module, weights_already_sharded: bool = False):
        """Store this rank's fused-GLU experts in matmul convention ``[E, K, N]``: ``gate_up_proj`` + ``down_proj``, or
        gate and up apart under ETP, where sharding fused halves misaligns them, or on a ``SEPARATE_GLU`` family."""
        self._require_fused_experts(experts)
        start, end = self._local_expert_range(weights_already_sharded)
        gate_up = experts.gate_up_proj.data[start:end]  # [E, 2M, H], dim 1 = [gate(M) | up(M)]
        down = experts.down_proj.data[start:end]  # [E, H, M]
        if self.expert_tp_size > 1 or self.FAMILY.expert_layout.kind is ExpertLayoutKind.SEPARATE_GLU:
            self._store_separate_glu_params(*gate_up.chunk(2, dim=1), down)
        else:
            self.gate_up_proj = nn.Parameter(gate_up.transpose(1, 2).contiguous())
            self.down_proj = nn.Parameter(down.transpose(1, 2).contiguous())

    def _local_expert_range(self, weights_already_sharded: bool) -> tuple[int, int]:
        """This rank's ``[start, end)`` in the HF expert container: all of it once the lazy loader sliced it."""
        return (0, self.experts_per_rank) if weights_already_sharded else (self.expert_start, self.expert_end)

    @staticmethod
    def _require_fused_experts(experts: nn.Module) -> nn.Module:
        """Raise rather than register no expert weight and train an empty expert set."""
        missing = [name for name in ("gate_up_proj", "down_proj") if not hasattr(experts, name)]
        if missing:
            raise AttributeError(
                f"Expected fused-GLU experts carrying gate_up_proj and down_proj, but "
                f"{type(experts).__name__} carries no {missing} — the family renamed its expert "
                f"tensors upstream."
            )
        return experts

    def _store_separate_glu_params(self, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> None:
        """Store this rank's ``F.linear``-convention ``gate``/``up`` ``[E, M, H]`` and ``down`` ``[E, H, M]`` as
        matmul-convention ``gate_proj``/``up_proj``/``down_proj``, ETP-sharded."""
        self.gate_proj = nn.Parameter(self._etp_narrow(gate, 1).transpose(1, 2).contiguous())
        self.up_proj = nn.Parameter(self._etp_narrow(up, 1).transpose(1, 2).contiguous())
        self.down_proj = nn.Parameter(self._etp_narrow(down, 2).transpose(1, 2).contiguous())

    def _etp_narrow(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        """A view of this ETP rank's contiguous slice of ``tensor`` along the expert-intermediate axis ``dim``."""
        shard = self._etp_shard_size(tensor.shape[dim])
        return tensor.narrow(dim, self.expert_tp_rank * shard, shard)

    def _etp_shard_size(self, intermediate: int) -> int:
        """Per-rank size of an ETP-sharded expert intermediate dim; raises when ``expert_tp_size`` does not
        divide it, which would drop the top units of every expert."""
        if intermediate % self.expert_tp_size != 0:
            raise ValueError(
                f"Expert TP requires the expert intermediate size ({intermediate}) to be divisible by "
                f"expert_tensor_parallel_size ({self.expert_tp_size}); got remainder "
                f"{intermediate % self.expert_tp_size}. Reduce expert_tensor_parallel_size."
            )
        return intermediate // self.expert_tp_size

    def _upcast_experts_to_fp32(self) -> None:
        """Upcast the expert weights and adapters under ``fp32_experts``, keeping each ``requires_grad`` (the
        LoRA-frozen base stays frozen); skipped where FSDP2 manages the experts and casts them to bf16."""
        if not self.fp32_experts:
            return
        if self.ep_config.experts_fsdp_managed:
            logger.warning(
                f"{type(self).__name__}: fp32_experts has NO effect with fsdp_shard_ep1_experts at "
                f"ep_group_size==1 — FSDP2 manages the replicated experts and its bf16 param_dtype "
                f"policy casts them for compute, so the upcast would only mix fp32 experts into a bf16 "
                f"fully_shard group. Skipping; for fp32 master weights set fp32_non_ep_params: true "
                f"AND fsdp_shard_ep1_experts: false (the managed-experts combination is refused at "
                f"config time)."
            )
            return
        for name, param in self.expert_named_params():
            setattr(self, name, nn.Parameter(param.data.float(), requires_grad=param.requires_grad))
        if is_global_main_process():
            logger.info(f"{type(self).__name__}: FP32 experts (compute stays BF16)")

    def _warm_expert_kernels(self, device: torch.device, dtype: torch.dtype, hidden: int, top_k: int) -> None:
        """Launch this layer's expert kernels, forward and backward, before its first dispatch: a kernel's first launch
        compiles (Triton) and loads (CUDA lazy loading) it, and inside the span either stalls every peer in DeepEP's
        barrier. The grouped path warms at every ``ep_size``: its GEMM tile and ``index_select`` kernel follow the
        receive, and a rank first needing one later still holds up the expert-TP reduce or FSDP's next collective."""
        grouped, dispatched = self._use_grouped_mm, self.ep_size > 1
        if self._expert_kernels_warmed or not (grouped or dispatched):
            return
        self._expert_kernels_warmed = True
        width = 2 * self.down_proj.shape[1]  # down_proj is [E, M, H] on every layout, ETP included
        # Zeros, not a draw: the recompute restores the RNG to region entry and this latched warmup does not
        # rerun, so a draw would shift expert-LoRA dropout's sample. Grad on and inference mode off, since a
        # first forward under either still has to compile the backward kernels.
        with torch.inference_mode(False):
            # Identity saved-tensor hooks: under a non-reentrant checkpoint the backward below would unpack a
            # packed tensor, run the recompute mid-forward and replay a dispatch the original pass has not made.
            with torch.enable_grad(), torch.autograd.graph.saved_tensors_hooks(lambda t: t, lambda t: t):
                if grouped:
                    self._warm_expert_gemm(device, dtype, hidden)
                    warm_route_sort_kernels(top_k, self.experts_per_rank, device)
                    warm_permute_kernels(_WARMUP_TOKENS, hidden, top_k, device, dtype)
                if dispatched:
                    gate_up = torch.zeros((_WARMUP_TOKENS, width), device=device, dtype=dtype, requires_grad=True)
                    torch.autograd.grad(self._warm_expert_activation(gate_up).sum(), gate_up)
            if dispatched:
                with torch.no_grad():
                    self._warm_expert_activation(torch.zeros((_WARMUP_TOKENS, width), device=device, dtype=dtype))

    def _warm_expert_gemm(self, device: torch.device, dtype: torch.dtype, hidden: int) -> None:
        """Run :meth:`_grouped_expert_ffn`, forward and backward, over zero tokens at one row per local expert and at
        one past :data:`GROUPED_MM_SMALL_TILE_DIM`: torch's bf16 grouped GEMM picks its tile by mean rows per group,
        which change with every receive. The backward also covers the expert-bias gather, whose one-hot GEMM cuBLAS
        picks by receive length. The RNG is restored, since expert-LoRA dropout draws; adapters run even if the first
        forward has them off, as training does."""
        experts, adapters_enabled = self.experts_per_rank, self._expert_adapters_enabled
        self._expert_adapters_enabled = True
        try:
            trainable = [param for _, param in self.expert_named_params() if param.requires_grad]
            with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
                for rows_per_expert in (1, GROUPED_MM_SMALL_TILE_DIM + 1):
                    rows = experts * rows_per_expert
                    tokens = torch.zeros((rows, hidden), device=device, dtype=dtype, requires_grad=True)
                    offs = torch.arange(1, experts + 1, device=device, dtype=torch.int32) * rows_per_expert
                    expert_ids = torch.arange(rows, device=device) // rows_per_expert
                    with torch.amp.autocast("cuda", dtype=dtype):
                        out = self._grouped_expert_ffn(tokens, offs, expert_ids, dtype)
                    torch.autograd.grad(out.sum(), [tokens, *trainable], allow_unused=True)
        finally:
            self._expert_adapters_enabled = adapters_enabled

    def _warm_expert_activation(self, gate_up: torch.Tensor) -> torch.Tensor:
        """The expert activation over a synthetic ``[T, 2M]``, in the operand layout the compute path hands
        it: Triton specializes a launch on its row strides."""
        if self._stores_fused_glu:
            return self._glu_combine_packed(gate_up)
        gate, up = gate_up.chunk(2, dim=-1)
        return self._glu_combine(gate.contiguous(), up.contiguous())

    def _compute_experts(
        self,
        tokens: torch.Tensor,
        experts: torch.Tensor,
        weights: torch.Tensor,
        output_dtype: torch.dtype,
        *,
        recv_counts: tuple[int, ...] | None,
    ) -> torch.Tensor:
        """Grouped GEMM when enabled and non-empty, else the per-expert loop. ``experts``/``weights`` are
        ``[T, top_k]`` local ids and gate weights; ``recv_counts`` counts per local expert, None at ``ep_size`` 1."""
        if self._use_grouped_mm and tokens.shape[0] > 0:
            return self._compute_experts_grouped(tokens, experts, weights, output_dtype, recv_counts=recv_counts)
        return self._compute_experts_loop(tokens, experts, weights, output_dtype, recv_counts=recv_counts)

    def _compute_experts_grouped(
        self,
        tokens: torch.Tensor,
        experts: torch.Tensor,
        weights: torch.Tensor,
        output_dtype: torch.dtype,
        *,
        recv_counts: tuple[int, ...] | None,
    ) -> torch.Tensor:
        """Sort -> :meth:`_grouped_expert_ffn` under autocast -> weighted unpermute. ``F.grouped_mm`` does not autocast,
        so the FFN casts its weights to ``output_dtype``."""
        sorted_tokens, offs, sorted_token_idx, sorted_weights, sorted_expert_ids, inv_map = (
            self._sort_tokens_for_grouped_mm(tokens, experts, weights, recv_counts=recv_counts)
        )

        with torch.amp.autocast("cuda", dtype=output_dtype):
            expert_out = self._grouped_expert_ffn(sorted_tokens, offs, sorted_expert_ids, output_dtype)

        return MoEWeightedUnpermute.apply(
            expert_out.to(output_dtype).contiguous(), sorted_weights.to(output_dtype), sorted_token_idx, inv_map
        )

    def _grouped_expert_ffn(
        self,
        sorted_tokens: torch.Tensor,
        offs: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """The expert FFN over tokens sorted by expert; ``offs`` are their int32 group ends."""
        return self._glu_ffn(sorted_tokens, lambda x, attr: self._expert_proj(x, attr, offs, output_dtype))

    def _require_recv_counts(self, recv_counts: tuple[int, ...] | None) -> tuple[int, ...]:
        """``recv_counts`` when it holds one count per local expert; at ``ep_size > 1`` it sizes the index reads."""
        if recv_counts is None or len(recv_counts) != self.experts_per_rank:
            raise RuntimeError(
                f"{type(self).__name__}: the expert compute at ep_size={self.ep_size} needs the dispatch's "
                f"per-local-expert receive counts ({self.experts_per_rank} of them), got {recv_counts!r}."
            )
        return recv_counts

    def _sort_tokens_for_grouped_mm(
        self,
        tokens: torch.Tensor,
        experts: torch.Tensor,
        weights: torch.Tensor,
        *,
        recv_counts: tuple[int, ...] | None,
    ):
        """``(sorted_tokens, offs, sorted_token_idx, sorted_weights, sorted_expert_ids, inv_map)`` over the valid local
        assignments; DeepEP's -1 padding is dropped (it would misalign the groups). At ``ep_size > 1`` the host-side
        counts size the sort, so it reads nothing back from the device."""
        counts = self._require_recv_counts(recv_counts) if self.ep_size > 1 else None
        sorted_token_idx, sorted_weights, sorted_expert_ids, offs, inv_map = sort_routing_by_expert(
            experts, weights, self.experts_per_rank, counts
        )
        # Atomic-free in both directions: the bf16 ``index_add_`` of the default gather backward and the
        # scatter-back is a CAS-loop atomic, ~1.5 ms per call on the gpt-oss-20b ep8 shape.
        sorted_tokens = MoEGatherPermute.apply(tokens, sorted_token_idx, inv_map)
        return sorted_tokens, offs, sorted_token_idx, sorted_weights, sorted_expert_ids, inv_map

    def _expert_proj(
        self, x: torch.Tensor, weight_attr: str, offs: torch.Tensor, out_dtype: torch.dtype
    ) -> torch.Tensor:
        """Grouped-GEMM expert projection ``x @ W[weight_attr]``, plus the LoRA delta where it is adapted."""
        out = self._grouped_mm(x, getattr(self, weight_attr).to(out_dtype), offs=offs)
        if self._expert_adapters_enabled and weight_attr in self._expert_lora_attrs:
            # Adapters stay bf16 under lowp: block-scaling along rank r is unrepresentable (r < block size).
            lora_a = getattr(self, f"{weight_attr}_lora_A").to(out_dtype)
            lora_b = getattr(self, f"{weight_attr}_lora_B").to(out_dtype)
            hidden = self._grouped_mm(self.expert_lora_dropout(x), lora_a, offs=offs, lowp=False)
            out = out + self.expert_lora_scaling * self._grouped_mm(hidden, lora_b, offs=offs, lowp=False)
        return out

    def _grouped_mm(
        self,
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        *,
        offs: torch.Tensor | None = None,
        lowp: bool = True,
    ) -> torch.Tensor:
        """Grouped GEMM at this layer's quantization format; ``lowp=False`` keeps bf16 (the LoRA adapters).
        Expert biases go through ``MoEExpertBiasGather``, never a fused bias."""
        return expert_gemm(
            mat_a,
            mat_b,
            offs=offs,
            quant_format=self._lowp_quant_format if lowp else None,
            weight_cacheable=self._lowp_weight_cacheable,
        )

    def _compute_experts_loop(
        self,
        tokens: torch.Tensor,
        experts: torch.Tensor,
        weights: torch.Tensor,
        output_dtype: torch.dtype,
        *,
        recv_counts: tuple[int, ...] | None,
    ) -> torch.Tensor:
        """The per-expert loop: :meth:`_expert_forward` per expert, outputs weighted and accumulated."""
        if tokens.shape[0] == 0:
            return torch.zeros((0, tokens.shape[-1]), device=tokens.device, dtype=output_dtype)

        counts = self._require_recv_counts(recv_counts) if self.ep_size > 1 else None
        output = torch.zeros(tokens.shape, device=tokens.device, dtype=output_dtype)

        with torch.amp.autocast("cuda", dtype=output_dtype, enabled=self.fp32_experts):
            for local_idx in range(self.experts_per_rank):
                slots = (experts == local_idx).t()
                if counts is None:
                    top_k_pos, token_idx = torch.where(slots)
                else:
                    # Sized off the host count, where torch.where would read the size back per expert. A
                    # wrong count would drop slots or pad with -1, which indexes the last row.
                    torch._assert_async(slots.sum() == counts[local_idx], RECEIVE_COUNTS_DISAGREE)
                    top_k_pos, token_idx = torch.nonzero_static(slots, size=counts[local_idx]).unbind(1)
                if token_idx.numel() == 0:
                    continue
                expert_weights = weights[token_idx, top_k_pos]
                out = self._expert_forward(local_idx, tokens.index_select(0, token_idx))
                expert_output = out.to(output_dtype) * expert_weights.unsqueeze(-1).to(output_dtype)
                output.index_add_(0, token_idx, expert_output)

        return output

    def _expert_forward(self, idx: int, x: torch.Tensor) -> torch.Tensor:
        return self._glu_ffn(x, lambda t, attr: self._expert_proj_single(idx, t, attr))

    @property
    def _stores_fused_glu(self) -> bool:
        """Whether the GLU halves are stored fused as ``gate_up_proj``, not apart (Qwen3, Bailing, any ETP run)."""
        return hasattr(self, "gate_up_proj")

    def _glu_ffn(self, x: torch.Tensor, proj: Callable[[torch.Tensor, str], torch.Tensor]) -> torch.Tensor:
        """The GLU FFN through ``proj(x, weight_attr)`` on either stored layout, for the loop and grouped paths."""
        if self._stores_fused_glu:
            return proj(self._glu_combine_packed(proj(x, "gate_up_proj")), "down_proj")
        return proj(self._glu_combine(proj(x, "gate_proj"), proj(x, "up_proj")), "down_proj")

    def _expert_proj_single(self, idx: int, x: torch.Tensor, weight_attr: str) -> torch.Tensor:
        out = x @ getattr(self, weight_attr)[idx]
        if self._expert_adapters_enabled and weight_attr in self._expert_lora_attrs:
            lora_a = getattr(self, f"{weight_attr}_lora_A")[idx]
            lora_b = getattr(self, f"{weight_attr}_lora_B")[idx]
            out = out + self.expert_lora_scaling * (self.expert_lora_dropout(x) @ lora_a @ lora_b)
        return out

    def _glu_combine(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """``act_fn(gate) * up`` through the latched fused kernel where one exists; a family with its own combine
        latches it in ``_fused_glu_mul``."""
        if self._fused_glu_mul is not None:
            return self._fused_glu_mul(gate, up)
        return self.act_fn(gate) * up

    def _glu_combine_packed(self, gate_up: torch.Tensor) -> torch.Tensor:
        """:meth:`_glu_combine` over a fused ``[gate | up]`` output, in place for a :class:`GluCombine` latch."""
        if isinstance(self._fused_glu_mul, GluCombine):
            return self._fused_glu_mul.packed(gate_up)
        gate, up = gate_up.chunk(2, dim=-1)
        return self._glu_combine(gate, up)

    def _glu_combine_name(self) -> str:
        """The combine this layer runs, read off the latch; ``eager`` under ``HALO_FUSED_GLU=0``."""
        combine = self._fused_glu_mul
        if combine is None or not fused_glu_enabled():
            return "eager"
        if isinstance(combine, GluCombine):
            return combine.name
        combine = getattr(combine, "func", combine)
        return getattr(combine, "__name__", type(combine).__name__)
