"""Source edits to vLLM's Inkling 2 image (``vllm/vllm-openai:inkling-2-small``), applied through ``engine_source_patch``.

The image's fused all-reduce subclasses FlashInfer's multicast memory with the one-argument ``_get_allocation_prop`` of
an older FlashInfer; its FlashInfer 0.7.3 calls it with the handle type too, so every tensor-parallel launch fails
before the weights load. The override passes the handle type through. The release's ``config.json`` also
carries a ModelOpt ``quantization_config``, which vLLM's ModelOpt detection takes over ``--quantization inkling2``
and the Inkling 2 model then refuses; vLLM reads Inkling 2's precision from ``hf_quant_config.json``, so the release
serves as downloaded.

Expert LoRA over the NVFP4 / MXFP8 experts:

The quantized experts' tile-padded layout already holds each local expert's rows contiguous and in expert order, which
is the row contract ``lora.Inkling2MoEWithLoRA`` adds its BF16 grouped-GEMM deltas over. The gate/up delta lands in the
w13 GEMM's bf16 output before ``swiglu_quantize`` reads it; the down delta takes the activation recomputed in bf16
(the quantizing kernel keeps none) and lands in the w2 output before the combine. Padding rows compute garbage that no
combine reads.

Three Triton kernels keep the path to what the deltas cost:

- ``layout_rows`` maps each layout row to its flat (token, slot) index and finds each expert's first row in one launch;
- ``situ`` recomputes the bf16 activation of the routed rows only, with ``swiglu_quantize``'s own arithmetic;
- the grouped GEMM masks each row's columns to its adapter slot, so one shrink over every slot's A, side by side, and
  one expand over every slot's B replace a pair of GEMMs per slot.

python3 patch_vllm_inkling2.py            # patch the installed vllm in place
python3 patch_vllm_inkling2.py --verify   # assert the post-images only
"""

import pathlib

from engine_source_patch import Edit, main

_ALLREDUCE = "models/inkling2/ops/allreduce.py"
_MOE = "models/inkling2/moe.py"
_LORA = "models/inkling2/lora.py"
_GEMM = "models/inkling2/ops/grouped_gemm.py"
_ACT = "models/inkling2/ops/activation.py"

_HF_CONFIG = "transformers_utils/config.py"
_QUANT_CONFIG_BEFORE = """    quantization_config = config_dict.get("quantization_config", None)
"""
_QUANT_CONFIG_AFTER = """    quantization_config = config_dict.get("quantization_config", None)
    # Inkling 2's config.json also carries a ModelOpt entry; its precision
    # metadata is hf_quant_config.json's (Inkling2QuantConfig).
    if config.model_type == "inkling2":
        quantization_config = None
"""

_ALLOCATION_BEFORE = """        def _get_allocation_prop(self, buf_size: int):
            allocation_prop, mc_prop = super()._get_allocation_prop(buf_size)
"""
_ALLOCATION_AFTER = """        def _get_allocation_prop(self, buf_size: int, *handle_type):
            allocation_prop, mc_prop = super()._get_allocation_prop(buf_size, *handle_type)
"""

# --- ops/grouped_gemm.py: a per-row adapter-slot mask, and the layout rows of the quantized experts ---

_GEMM_ARGS_BEFORE = """    ACCUMULATE: tl.constexpr,
    BM: tl.constexpr,
"""
_GEMM_ARGS_AFTER = """    ACCUMULATE: tl.constexpr,
    TokenLora,
    RANK: tl.constexpr,
    BM: tl.constexpr,
"""
_GEMM_STORE_BEFORE = """    y_ptrs = Y + rows[:, None] * N + cols[None, :]
"""
_GEMM_STORE_AFTER = """    if RANK > 0:
        # Columns [s * RANK, (s + 1) * RANK) are adapter slot s's: each row
        # keeps its token's slot (-1: none).
        tokens = tl.load(Perm + rows, row_mask, other=0) // TOPK
        slot = tl.load(TokenLora + tokens, row_mask, other=-1)
        acc = tl.where(cols[None, :] // RANK == slot[:, None], acc, 0.0)
    y_ptrs = Y + rows[:, None] * N + cols[None, :]
"""
_GEMM_SIG_BEFORE = """    gather: bool,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
"""
_GEMM_SIG_AFTER = """    gather: bool,
    out: torch.Tensor | None = None,
    token_lora: torch.Tensor | None = None,
    rank: int = 0,
) -> torch.Tensor:
"""
_GEMM_DOC_BEFORE = """    uninitialized; with ``out``, adds into it (FP32 accumulation either way).
    \"\"\"
"""
_GEMM_DOC_AFTER = """    uninitialized; with ``out``, adds into it (FP32 accumulation either way).
    With ``rank``, ``weight`` stacks adapter slots of ``rank`` rows each and
    row ``r`` keeps only the columns of slot ``token_lora[perm[r] // topk]``.
    \"\"\"
"""
_GEMM_LAUNCH_BEFORE = """            gather,
            accumulate,
            block_m,
"""
_GEMM_LAUNCH_AFTER = """            gather,
            accumulate,
            token_lora,
            rank,
            block_m,
"""
_GEMM_END_BEFORE = """            num_warps=4,
            num_stages=3,
        )
    return out
"""
_GEMM_END_AFTER = '''            num_warps=4,
            num_stages=3,
        )
    return out


@triton.jit
def _layout_rows_kernel(
    Src2Dst,
    Tiles,
    Perm,
    Offsets,
    n,
    MAX_CTAS: tl.constexpr,
    CTAS_POW2: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    TILE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid == tl.num_programs(0) - 1:
        # Tiles hold their experts in order: an expert's first row is TILE
        # times the live tiles of the experts before it.
        ctas = tl.arange(0, CTAS_POW2)
        live = ctas < tl.load(Tiles + 2 * MAX_CTAS)
        tile_experts = tl.load(Tiles + ctas, live, other=NUM_EXPERTS)
        for start in range(0, NUM_EXPERTS + 1, 16):
            experts = start + tl.arange(0, 16)
            before = tile_experts[None, :] < experts[:, None]
            offsets = tl.sum(before.to(tl.int32), 1) * TILE
            tl.store(Offsets + experts, offsets, experts <= NUM_EXPERTS)
        return
    flat = pid * BLOCK + tl.arange(0, BLOCK)
    dst = tl.load(Src2Dst + flat, flat < n, other=-1)
    tl.store(Perm + dst, flat, dst >= 0)


def layout_rows(
    src2dst: torch.Tensor, tiles: torch.Tensor, rows: int, max_ctas: int, tile: int, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``grouped_gemm``'s ``(perm, offsets)`` over the tile-padded layout of
    ``ops.moe.route``: each layout row's flat (token * topk + slot) index, 0
    for a padding row, and each local expert's first layout row, then the
    padded end. Nothing is read back to the host (CUDA graph capture)."""
    perm = torch.zeros(rows, dtype=torch.int32, device=src2dst.device)
    offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=src2dst.device)
    block = 1024
    _layout_rows_kernel[(triton.cdiv(src2dst.numel(), block) + 1,)](
        src2dst,
        tiles,
        perm,
        offsets,
        src2dst.numel(),
        max_ctas,
        triton.next_power_of_2(max_ctas),
        num_experts,
        tile,
        block,
        num_warps=4,
    )
    return perm, offsets
'''

# --- ops/activation.py: the bf16 activation of the routed rows ---

_ACT_ANCHOR = """def swiglu_quantize(
"""
_ACT_SITU = '''@triton.jit
def _situ_rows_kernel(
    X,
    Gammas,
    Out,
    Rows,
    NumRows,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    GATE_CAP: tl.constexpr,
    UP_CAP: tl.constexpr,
):
    num_rows = tl.load(NumRows)
    start = tl.program_id(1) * BLOCK
    cols = start + tl.arange(0, BLOCK)
    r = tl.program_id(0)
    while r < num_rows:
        row = tl.load(Rows + r).to(tl.int64)
        y = _situ(X, row, start, tl.load(Gammas + row), N, BLOCK, GATE_CAP, UP_CAP)
        tl.store(Out + row * N + cols, y.to(tl.bfloat16))
        r += tl.num_programs(0)


def situ(
    x: torch.Tensor,
    gammas: torch.Tensor,
    gate_cap: float,
    up_cap: float,
    rows: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """bf16 ``softcap(gate) * sigmoid(gate) * softcap(up) * gamma`` of the
    routed rows of ``x`` (``swiglu_quantize``'s layout, its arithmetic), before
    any quantization: the down projection's LoRA input. Other rows are left
    uninitialized."""
    routed, num_routed = rows
    n = x.shape[1] // 2
    block = 256
    assert n % block == 0
    out = x.new_empty(x.shape[0], n)
    _situ_rows_kernel[(min(routed.numel(), 16384), n // block)](
        x,
        gammas,
        out,
        routed,
        num_routed,
        N=n,
        BLOCK=block,
        GATE_CAP=gate_cap,
        UP_CAP=up_cap,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


'''

# --- moe.py: the experts call the LoRA between their GEMMs ---

_PROTOCOL_BEFORE = """    ``[offsets[0], offsets[-1])``. The NVFP4 / MXFP8 experts here (tile-padded
    rows, quantized activations) do not support it; the BF16 experts a model
    may plug in instead (SkyRL's ``Inkling2BF16Experts``) do.
    \"\"\"
"""
_PROTOCOL_AFTER = """    ``[offsets[0], offsets[-1])``. The NVFP4 / MXFP8 experts here pass their
    tile-padded layout (``ops.grouped_gemm.layout_rows``); its padding rows map
    to token 0.
    \"\"\"
"""
_IMPORT_BEFORE = """from .ops.activation import swiglu_quantize
"""
_IMPORT_AFTER = """from .ops.activation import situ, swiglu_quantize
"""
_IMPORT2_BEFORE = """from .ops.block_scales import shuffle_rows, swizzle_block_scales
"""
_IMPORT2_AFTER = """from .ops.block_scales import shuffle_rows, swizzle_block_scales
from .ops.grouped_gemm import layout_rows
"""
_FORWARD_BEFORE = """        reduce: bool = True,
        push: SeqParallel | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
        \"\"\"``routing`` and ``quantized`` (``x`` in the expert layout) are from
        ``ops.moe.route``. The routing weight is applied inside the SwiGLU,
        before the down projection. Without ``reduce``, returns the local
        expert rows and ``src2dst`` for the consumer to combine and reduce,
        or pushes their combine to the token shard owners.\"\"\"
        src2dst, gammas, rows, tiles = routing
        gate_up = self._gemm(quantized, "w13", tiles, plan, 0)
        m = plan.max_ctas
"""
_FORWARD_AFTER = """        reduce: bool = True,
        push: SeqParallel | None = None,
        lora: ExpertsLoRA | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
        \"\"\"``routing`` and ``quantized`` (``x`` in the expert layout) are from
        ``ops.moe.route``. The routing weight is applied inside the SwiGLU,
        before the down projection. Without ``reduce``, returns the local
        expert rows and ``src2dst`` for the consumer to combine and reduce,
        or pushes their combine to the token shard owners. ``lora`` adds its
        deltas to the bf16 GEMM outputs, the gate/up one before the SwiGLU.\"\"\"
        src2dst, gammas, rows, tiles = routing
        gate_up = self._gemm(quantized, "w13", tiles, plan, 0)
        m = plan.max_ctas
        if lora is not None:
            perm, offsets = layout_rows(
                src2dst, tiles, plan.rows, m, plan.tile, self.num_experts
            )
            lora.add_w13(gate_up, x, perm, offsets, self.topk)
            activation = situ(
                gate_up, gammas, self.gate_cap, self.up_cap, (rows, tiles[2 * m + 2 :])
            )
"""
_DOWN_BEFORE = """        down = self._gemm(activated, "w2", tiles, plan, 1)
        if push is not None:
"""
_DOWN_AFTER = """        down = self._gemm(activated, "w2", tiles, plan, 1)
        if lora is not None:
            lora.add_w2(down, activation, perm, offsets, self.topk)
            del activation
        if push is not None:
"""
_RAISE_BEFORE = """        to the token shard owners.
        Padding tokens (``is_padding`` of the forward context) are routed to
        no expert and come out as 0. ``lora`` (``Inkling2MoEWithLoRA``) adds
        per-expert LoRA deltas inside the expert GEMMs, before any reduce or
        push; the NVFP4 / MXFP8 experts here do not support it.\"\"\"
        if lora is not None:
            raise NotImplementedError(
                "Inkling2 expert LoRA needs BF16 experts; the NVFP4 / MXFP8 "
                "experts do not support it"
            )
"""
_RAISE_AFTER = """        to the token shard owners.
        Padding tokens (``is_padding`` of the forward context) are routed to
        no expert and come out as 0. ``lora`` (``Inkling2MoEWithLoRA``) adds
        per-expert LoRA deltas inside the expert GEMMs, before any reduce or
        push.\"\"\"
"""
_CALL_BEFORE = """        return experts(x, routing, quantized, plan, reduce, push)
"""
_CALL_AFTER = """        return experts(x, routing, quantized, plan, reduce, push, lora)
"""

# --- lora.py: every slot's adapters side by side, one masked shrink and one expand per projection ---

_LORA_DOC_BEFORE = """and adds the deltas between the expert GEMMs through ``moe.ExpertsLoRA``,
which needs BF16 experts (the NVFP4 / MXFP8 ones raise).
"""
_LORA_DOC_AFTER = """and adds the deltas between the expert GEMMs through ``moe.ExpertsLoRA``.
"""
_LORA_WEIGHTS_BEFORE = """        def zeros(*shape: int) -> torch.Tensor:
            return torch.zeros(
                max_loras,
                self.num_local_experts,
                *shape,
                dtype=lora_config.lora_dtype,
                device=device,
            )

        # (w13, w2); dims [max_loras, local experts, r, in] / [..., out, r].
        self.lora_a_stacked = (zeros(rank, hidden), zeros(rank, inter))
        self.lora_b_stacked = (zeros(2 * inter, rank), zeros(hidden, rank))

    def reset_lora(self, index: int) -> None:
        for stacked in (*self.lora_a_stacked, *self.lora_b_stacked):
            stacked[index].zero_()
"""
_LORA_WEIGHTS_AFTER = """        self.rank = rank
        width = max_loras * rank

        def zeros(*shape: int) -> torch.Tensor:
            return torch.zeros(
                self.num_local_experts,
                *shape,
                dtype=lora_config.lora_dtype,
                device=device,
            )

        # (w13, w2); every slot's rank rows side by side, slot s at
        # [s * r, (s + 1) * r): [local experts, slots * r, in] / [..., out, slots * r].
        self.lora_a_stacked = (zeros(width, hidden), zeros(width, inter))
        self.lora_b_stacked = (zeros(2 * inter, width), zeros(hidden, width))

    def reset_lora(self, index: int) -> None:
        span = slice(index * self.rank, (index + 1) * self.rank)
        for a in self.lora_a_stacked:
            a[:, span].zero_()
        for b in self.lora_b_stacked:
            b[:, :, span].zero_()
"""
_LORA_SET_BEFORE = """            a, b = self._local(a), self._local(b)
            rank = a.shape[-2]
            a_stacked[index, :, :rank].copy_(a, non_blocking=True)
            b_stacked[index, :, :, :rank].copy_(b, non_blocking=True)
"""
_LORA_SET_AFTER = """            a, b = self._local(a), self._local(b)
            span = slice(index * self.rank, index * self.rank + a.shape[-2])
            a_stacked[:, span].copy_(a, non_blocking=True)
            b_stacked[:, :, span].copy_(b, non_blocking=True)
"""
_LORA_ADD_BEFORE = """        assert self.punica_wrapper is not None
        # Each row's adapter slot (-1: none), from its token's.
        token_lora = self.punica_wrapper._token_lora_indices[: x.shape[0]]
        row_lora = token_lora[perm // topk]
        slots = torch.arange(self.max_loras, device=x.device)
        self._row_masks = (row_lora[None, :] == slots[:, None]).to(x.dtype)
        self._add(gate_up, x, 0, perm, offsets, topk, gather=True)
"""
_LORA_ADD_AFTER = """        assert self.punica_wrapper is not None
        # Each token's adapter slot (-1: none).
        self._token_lora = self.punica_wrapper._token_lora_indices[: x.shape[0]]
        self._add(gate_up, x, 0, perm, offsets, topk, gather=True)
"""
_LORA_ADD2_BEFORE = """        self._add(down, activated, 1, perm, offsets, topk, gather=False)
        self._row_masks = None

    def _add(self, out, x, proj, perm, offsets, topk, gather) -> None:
        assert self._row_masks is not None
        num_rows = out.shape[0]
        for slot in range(self.max_loras):
            a = self.lora_a_stacked[proj][slot]
            b = self.lora_b_stacked[proj][slot]
            shrunk = grouped_gemm(x, a, perm, offsets, num_rows, topk, gather)
            shrunk.mul_(self._row_masks[slot, :, None])
            grouped_gemm(shrunk, b, perm, offsets, num_rows, topk, False, out=out)
"""
_LORA_ADD2_AFTER = """        self._add(down, activated, 1, perm, offsets, topk, gather=False)
        self._token_lora = None

    def _add(self, out, x, proj, perm, offsets, topk, gather) -> None:
        \"\"\"One shrink over every slot's A, each row keeping its own slot's
        columns, then one expand over every slot's B.\"\"\"
        assert self._token_lora is not None
        num_rows = out.shape[0]
        a, b = self.lora_a_stacked[proj], self.lora_b_stacked[proj]
        shrunk = grouped_gemm(
            x, a, perm, offsets, num_rows, topk, gather, None, self._token_lora, self.rank
        )
        grouped_gemm(shrunk, b, perm, offsets, num_rows, topk, False, out=out)
"""
_LORA_INIT_BEFORE = """        self._row_masks: torch.Tensor | None = None
"""
_LORA_INIT_AFTER = """        self._token_lora: torch.Tensor | None = None
"""

EDITS = (
    Edit(_ALLREDUCE, _ALLOCATION_BEFORE, _ALLOCATION_AFTER, "multicast allocation handle type"),
    Edit(_HF_CONFIG, _QUANT_CONFIG_BEFORE, _QUANT_CONFIG_AFTER, "Inkling 2 reads hf_quant_config.json"),
    Edit(_GEMM, _GEMM_ARGS_BEFORE, _GEMM_ARGS_AFTER, "grouped GEMM slot-mask arguments"),
    Edit(_GEMM, _GEMM_STORE_BEFORE, _GEMM_STORE_AFTER, "grouped GEMM slot mask"),
    Edit(_GEMM, _GEMM_SIG_BEFORE, _GEMM_SIG_AFTER, "grouped_gemm token_lora / rank"),
    Edit(_GEMM, _GEMM_DOC_BEFORE, _GEMM_DOC_AFTER, "grouped_gemm docstring"),
    Edit(_GEMM, _GEMM_LAUNCH_BEFORE, _GEMM_LAUNCH_AFTER, "grouped_gemm launch"),
    Edit(_GEMM, _GEMM_END_BEFORE, _GEMM_END_AFTER, "layout_rows kernel"),
    Edit(_ACT, _ACT_ANCHOR, _ACT_SITU + _ACT_ANCHOR, "bf16 SiTU of the routed rows"),
    Edit(_MOE, _PROTOCOL_BEFORE, _PROTOCOL_AFTER, "ExpertsLoRA docstring"),
    Edit(_MOE, _IMPORT_BEFORE, _IMPORT_AFTER, "moe imports situ"),
    Edit(_MOE, _IMPORT2_BEFORE, _IMPORT2_AFTER, "moe imports layout_rows"),
    Edit(_MOE, _FORWARD_BEFORE, _FORWARD_AFTER, "experts forward: gate/up LoRA delta"),
    Edit(_MOE, _DOWN_BEFORE, _DOWN_AFTER, "experts forward: down LoRA delta"),
    Edit(_MOE, _RAISE_BEFORE, _RAISE_AFTER, "Inkling2MoE expert LoRA refusal"),
    Edit(_MOE, _CALL_BEFORE, _CALL_AFTER, "Inkling2MoE passes the LoRA to the experts"),
    Edit(_LORA, _LORA_DOC_BEFORE, _LORA_DOC_AFTER, "lora docstring"),
    Edit(_LORA, _LORA_INIT_BEFORE, _LORA_INIT_AFTER, "lora token slots"),
    Edit(_LORA, _LORA_WEIGHTS_BEFORE, _LORA_WEIGHTS_AFTER, "lora slot-stacked weights"),
    Edit(_LORA, _LORA_SET_BEFORE, _LORA_SET_AFTER, "lora set_lora into its slot"),
    Edit(_LORA, _LORA_ADD_BEFORE, _LORA_ADD_AFTER, "lora add_w13"),
    Edit(_LORA, _LORA_ADD2_BEFORE, _LORA_ADD2_AFTER, "lora masked shrink and one expand"),
)


def _no_check(_root: pathlib.Path) -> None:
    return None


if __name__ == "__main__":
    main("vllm", EDITS, _no_check)
