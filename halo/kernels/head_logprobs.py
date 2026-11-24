"""Log-probs (and entropy) of target tokens straight from hidden states, never materializing the logits: a dual-chunked
(sequence × vocab) matmul with online softmax and a recompute backward. A row may carry several targets that share its
one ``log Z`` (a sampled token and its candidates). A LoRA adapter's delta joins each tile as its low-rank pair."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from halo.kernels.fp32_matmul import gemm_dtype, mm_fp32

# The working set is a few [seq, vocab] fp32 tiles (4096 × 16384 × 4 B = 256 MiB each); tiles sized well below this
# leave the sweep launch-bound on large-vocabulary models.
SEQ_TILE = 4096
VOCAB_TILE = 16384

# A CPU compile costs seconds per graph on a path only tests take, so there the same functions run eagerly.
_COMPILED_DEVICE_TYPES = frozenset({"cuda"})

# ``deterministic`` skips on-device benchmarking, so ranks seeing the same tiles (TP, ETP replicas) and reruns compile
# the same kernels. One thread builds a graph's few small kernels in-process, sparing a pool of up to 32 workers per
# rank.
_TILE_COMPILE_OPTIONS = {"deterministic": True, "compile_threads": 1}


@dataclass(frozen=True, eq=False)
class LowRankDelta:
    """``scaling · inputs @ down.T @ up.T`` added to the head's logits: a LoRA adapter's ``lora_A`` (``down``,
    ``[r, H]``) and ``lora_B`` (``up``, ``[V, r]``) weights, ``inputs`` its own ``(..., H)`` input (the head's input
    after the adapter's dropout)."""

    inputs: torch.Tensor
    down: torch.Tensor
    up: torch.Tensor
    scaling: float


def _low_rank_deltas(scalings: tuple[float, ...], tensors: Sequence[torch.Tensor]) -> list[LowRankDelta]:
    return [
        LowRankDelta(inputs, down, up, scaling)
        for scaling, (inputs, down, up) in zip(
            scalings, (tensors[i : i + 3] for i in range(0, len(tensors), 3)), strict=True
        )
    ]


def _low_rank_factors(low_rank: Sequence[LowRankDelta], start: int, end: int) -> list[torch.Tensor]:
    """Each delta's ``scaling · inputs @ down.T`` over rows ``[start, end)``: ``[rows, r]`` fp32, once per sequence
    tile, so a vocab tile pays one ``[rows, r] @ [r, width]`` GEMM per adapter."""
    return [mm_fp32(delta.inputs[start:end], delta.down.t()) * delta.scaling for delta in low_rank]


def _tile_raw(hidden_chunk, weight_chunk, low_rank, factors, vocab_start, vocab_end) -> torch.Tensor:
    """The tile's fp32 logits before the bias: the base GEMM plus each low-rank delta, never ``W + B·A``. The delta's
    product runs in fp32: its inner dimension is the rank, so it costs what a bf16 one does (the tile's read and write
    bound both), while rounding the factor and ``B`` to bf16 moved log-probs by up to 4e-2 nats."""
    raw = mm_fp32(hidden_chunk, weight_chunk.t())
    for delta, factor in zip(low_rank, factors, strict=True):
        raw.addmm_(factor, delta.up[vocab_start:vocab_end].t().float())
    return raw


def _head_logits(raw, bias, vocab_start, logit_scale, softcap) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Before the temperature: the fp32 bias, the logit scale, then the softcap, as the model's forward orders them.
    Also returns the cap's ``tanh``, which its derivative reuses."""
    logits = raw
    if bias is not None:
        logits = logits + bias.narrow(0, vocab_start, raw.shape[1])
    if logit_scale is not None:
        logits = logits * logit_scale
    if softcap is None:
        return logits, None
    tanh = torch.tanh(logits / softcap)
    return tanh * softcap, tanh


def _is_target(targets, vocab_start, width) -> torch.Tensor:
    """An iota compare rather than an index write, so a compiled tile keeps the one-hot inside its one kernel."""
    return torch.arange(width, device=targets.device) == (targets - vocab_start).unsqueeze(-1)


def _target_slots(targets, vocab_start, width) -> tuple[torch.Tensor, torch.Tensor]:
    """``(column, in_tile)`` of each ``[rows, m]`` target in this vocab tile; an id below 0 is in no tile."""
    column = targets - vocab_start
    in_tile = (targets >= 0) & (column >= 0) & (column < width)
    return column.clamp(0, width - 1), in_tile


def _forward_tile(
    raw, bias, targets, vocab_start, max_old, sum_exp, sum_exp_z, target_logit, inv_t, logit_scale, softcap
):
    """One vocab tile folded into ``(max, Σ exp(z − max), Σ exp(z − max)·z or None, target logit)`` over
    temperature-scaled logits ``z``. The entropy rides the same sweep (``log Z − Σ exp(z − max)·z / Σ exp(z − max)``),
    sparing a second full-vocab matmul."""
    logits = _head_logits(raw, bias, vocab_start, logit_scale, softcap)[0] * inv_t
    max_new = torch.maximum(max_old, logits.amax(dim=-1))
    rescale = torch.exp(max_old - max_new)
    exps = torch.exp(logits - max_new.unsqueeze(-1))
    sum_exp = sum_exp * rescale + exps.sum(dim=-1)
    if sum_exp_z is not None:
        sum_exp_z = sum_exp_z * rescale + (exps * logits).sum(dim=-1)
    if targets.dim() == 2:
        column, in_tile = _target_slots(targets, vocab_start, raw.shape[1])
        return max_new, sum_exp, sum_exp_z, target_logit + torch.where(in_tile, logits.gather(1, column), 0.0)
    is_target = _is_target(targets, vocab_start, raw.shape[1])
    target_logit = target_logit + torch.where(is_target, logits, 0.0).sum(dim=-1)
    return max_new, sum_exp, sum_exp_z, target_logit


def _backward_tile(raw, bias, targets, vocab_start, log_z, grad, inv_t, logit_scale, softcap, grad_dtype):
    """``grad · (onehot − softmax)`` carried through the temperature, the cap's ``1 − tanh²`` and the logit scale, in
    the dtype the gradient GEMMs take. The column sum, the bias's gradient, is taken before that cast. With ``m``
    targets per row it is ``Σ_j grad_j · onehot_j − (Σ_j grad_j) · softmax`` over the unmasked ones, the one-hots
    scatter-added so no ``[rows, m, width]`` plane exists."""
    logits, tanh = _head_logits(raw, bias, vocab_start, logit_scale, softcap)
    probs = torch.exp(logits * inv_t - log_z.unsqueeze(-1))
    if targets.dim() == 2:
        column, in_tile = _target_slots(targets, vocab_start, raw.shape[1])
        grad = torch.where(targets >= 0, grad, 0.0)
        picked = torch.zeros_like(probs).scatter_add(1, column, torch.where(in_tile, grad, 0.0))
        grad_logits = (picked - grad.sum(dim=-1, keepdim=True) * probs) * inv_t
    else:
        grad = grad.unsqueeze(-1)
        is_target = _is_target(targets, vocab_start, raw.shape[1])
        grad_logits = (torch.where(is_target, grad, 0.0) - grad * probs) * inv_t
    if tanh is not None:
        grad_logits = grad_logits * (1.0 - tanh.square())
    if logit_scale is not None:
        grad_logits = grad_logits * logit_scale
    return grad_logits.to(grad_dtype), grad_logits.sum(dim=0) if bias is not None else None


def _compile_tile(fn):
    """Compiled whole (a graph break raises): rows, tile width and vocab offset are symbolic, head scalars inputs."""
    return torch.compile(fn, dynamic=True, fullgraph=True, options=_TILE_COMPILE_OPTIONS)


_COMPILED_TILES = (_compile_tile(_forward_tile), _compile_tile(_backward_tile))


def _tile_functions(raw: torch.Tensor):
    """Eager on a tile one row or column wide. Dynamo specializes a size of 1, so such a tile, which a ragged batch can
    end on at any step, would compile a graph of its own mid-run, on that rank alone."""
    if raw.device.type in _COMPILED_DEVICE_TYPES and min(raw.shape) > 1:
        return _COMPILED_TILES
    return _forward_tile, _backward_tile


def _rows(t: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """A contiguous copy, as a compiled tile's guards would key on a view's base sizes, which change from step to step,
    or the stride 0 of an expanded upstream gradient (``logps.sum()``)."""
    return t[start:end].clone(memory_format=torch.contiguous_format)


def _tile_bias(bias: torch.Tensor | None) -> torch.Tensor | None:
    """fp32 and detached, so neither its dtype nor whether it trains gives a compiled tile a graph of its own."""
    return None if bias is None else bias.detach().float()


def _autocast_off(device: torch.device):
    """A caller's bf16 autocast would round an fp32 GEMM's logits to bf16 and give compiled tiles a graph per state."""
    return torch.autocast(device.type, enabled=False)


def _online_softmax_tile(
    hidden_chunk, weight, targets_chunk, bias, inv_t, logit_scale, softcap, compute_entropy, low_rank, factors
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """One sequence tile swept over every vocab tile: ``(target logit, log Z, entropy or None)``."""
    device = hidden_chunk.device
    n_chunk = hidden_chunk.shape[0]
    max_old = torch.full((n_chunk,), float("-inf"), device=device, dtype=torch.float32)
    sum_exp = torch.zeros((n_chunk,), device=device, dtype=torch.float32)
    sum_exp_z = torch.zeros((n_chunk,), device=device, dtype=torch.float32) if compute_entropy else None
    target_logit = torch.zeros(targets_chunk.shape, device=device, dtype=torch.float32)

    for vocab_start in range(0, weight.shape[0], VOCAB_TILE):
        vocab_end = min(vocab_start + VOCAB_TILE, weight.shape[0])
        weight_chunk = weight[vocab_start:vocab_end].to(hidden_chunk.dtype)
        raw = _tile_raw(hidden_chunk, weight_chunk, low_rank, factors, vocab_start, vocab_end)
        forward_tile, _ = _tile_functions(raw)
        max_old, sum_exp, sum_exp_z, target_logit = forward_tile(
            raw,
            bias,
            targets_chunk,
            vocab_start,
            max_old,
            sum_exp,
            sum_exp_z,
            target_logit,
            inv_t,
            logit_scale,
            softcap,
        )

    log_z = max_old + torch.log(sum_exp)
    return target_logit, log_z, log_z - sum_exp_z / sum_exp if compute_entropy else None


@torch.no_grad()
def _logprobs_forward(
    hidden, weight, targets, bias, temperature, logit_scale, softcap, compute_entropy, low_rank
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_rows = hidden.shape[0]
    inv_t = 1.0 / temperature
    bias = _tile_bias(bias)
    logprobs = torch.empty(targets.shape, device=hidden.device, dtype=torch.float32)
    log_z = torch.empty((n_rows,), device=hidden.device, dtype=torch.float32)
    entropy = torch.zeros((n_rows,), device=hidden.device, dtype=torch.float32)

    for seq_start in range(0, n_rows, SEQ_TILE):
        seq_end = min(seq_start + SEQ_TILE, n_rows)
        target_logit, log_z_chunk, entropy_chunk = _online_softmax_tile(
            hidden[seq_start:seq_end],
            weight,
            _rows(targets, seq_start, seq_end),
            bias,
            inv_t,
            logit_scale,
            softcap,
            compute_entropy,
            low_rank,
            _low_rank_factors(low_rank, seq_start, seq_end),
        )
        log_z[seq_start:seq_end] = log_z_chunk
        logprobs[seq_start:seq_end] = target_logit - (log_z_chunk if targets.dim() == 1 else log_z_chunk.unsqueeze(-1))
        if compute_entropy:
            entropy[seq_start:seq_end] = entropy_chunk
    if targets.dim() == 2:
        logprobs = logprobs.masked_fill(targets < 0, float("-inf"))
    return logprobs, log_z, entropy


def _logprobs_backward(
    hidden, weight, targets, bias, log_z, grad_logprobs, temperature, logit_scale, softcap, needs_grad, low_rank
):
    """``(grad_hidden, grad_weight, grad_bias, per delta (grad_inputs, grad_down, grad_up))``, ``None`` where
    ``needs_grad`` is false, so a frozen head builds no ``[V, H]`` buffer and runs no weight-gradient GEMM. Each logits
    tile is recomputed from the saved ``log_z`` instead of a stored ``[T, V]`` plane."""
    inv_t = 1.0 / temperature
    bias = _tile_bias(bias)
    n_rows, _ = hidden.shape
    vocab_size = weight.shape[0]
    grad_dtype = gemm_dtype(hidden)
    needs_hidden, needs_weight, needs_bias, needs_low_rank = needs_grad

    grad_hidden = torch.zeros(hidden.shape, device=hidden.device, dtype=torch.float32) if needs_hidden else None
    grad_weight = torch.zeros(weight.shape, device=weight.device, dtype=torch.float32) if needs_weight else None
    grad_bias = (
        torch.zeros((vocab_size,), device=weight.device, dtype=torch.float32)
        if bias is not None and needs_bias
        else None
    )
    grad_low_rank = [
        tuple(
            torch.zeros(t.shape, device=t.device, dtype=torch.float32) if needs else None
            for t, needs in zip((delta.inputs, delta.down, delta.up), flags, strict=True)
        )
        for delta, flags in zip(low_rank, needs_low_rank, strict=True)
    ]
    grad_logprobs = grad_logprobs.to(torch.float32)

    for seq_start in range(0, n_rows, SEQ_TILE):
        seq_end = min(seq_start + SEQ_TILE, n_rows)
        hidden_chunk = hidden[seq_start:seq_end]
        targets_chunk, log_z_chunk, grad_chunk = (
            _rows(t, seq_start, seq_end) for t in (targets, log_z, grad_logprobs)
        )
        factors = _low_rank_factors(low_rank, seq_start, seq_end)
        # A delta whose input and A are both frozen needs no gradient through its factor.
        grad_factors = [
            torch.zeros_like(factor) if grad_inputs is not None or grad_down is not None else None
            for factor, (grad_inputs, grad_down, _) in zip(factors, grad_low_rank, strict=True)
        ]
        for vocab_start in range(0, vocab_size, VOCAB_TILE):
            vocab_end = min(vocab_start + VOCAB_TILE, vocab_size)
            weight_chunk = weight[vocab_start:vocab_end].to(hidden_chunk.dtype)
            raw = _tile_raw(hidden_chunk, weight_chunk, low_rank, factors, vocab_start, vocab_end)
            _, backward_tile = _tile_functions(raw)
            grad_logits, grad_bias_tile = backward_tile(
                raw,
                bias,
                targets_chunk,
                vocab_start,
                log_z_chunk,
                grad_chunk,
                inv_t,
                logit_scale,
                softcap,
                grad_dtype,
            )
            if grad_hidden is not None:
                mm_fp32(grad_logits, weight_chunk, out=grad_hidden[seq_start:seq_end])
            if grad_weight is not None:
                mm_fp32(grad_logits.t(), hidden_chunk, out=grad_weight[vocab_start:vocab_end])
            if grad_bias is not None:
                grad_bias[vocab_start:vocab_end].add_(grad_bias_tile)
            for delta, factor, grad_factor, (_, _, grad_up) in zip(
                low_rank, factors, grad_factors, grad_low_rank, strict=True
            ):
                if grad_factor is not None:
                    mm_fp32(grad_logits, delta.up[vocab_start:vocab_end].to(grad_logits.dtype), out=grad_factor)
                if grad_up is not None:
                    mm_fp32(grad_logits.t(), factor.to(grad_logits.dtype), out=grad_up[vocab_start:vocab_end])
        for delta, grad_factor, (grad_inputs, grad_down, _) in zip(low_rank, grad_factors, grad_low_rank, strict=True):
            if grad_factor is None:
                continue
            # The factor is ``scaling · x @ A.T``, so the gradient reaching ``x`` and ``A`` takes the scaling.
            grad_factor = grad_factor * delta.scaling
            inputs_chunk = delta.inputs[seq_start:seq_end]
            if grad_inputs is not None:
                mm_fp32(grad_factor, delta.down, out=grad_inputs[seq_start:seq_end])
            if grad_down is not None:
                mm_fp32(grad_factor.t(), inputs_chunk, out=grad_down)

    return grad_hidden, grad_weight, grad_bias, grad_low_rank


class _HeadLogprobFunction(torch.autograd.Function):
    """Only the log-probs carry gradient; the entropy is a detached diagnostic. Each low-rank delta rides as its
    ``scaling`` in ``scalings`` and its three tensors, in order, in ``low_rank_tensors``."""

    @staticmethod
    def forward(
        ctx,
        hidden,
        weight,
        targets,
        bias,
        temperature,
        logit_scale,
        softcap,
        compute_entropy,
        scalings,
        *low_rank_tensors,
    ):
        low_rank = _low_rank_deltas(scalings, low_rank_tensors)
        with _autocast_off(hidden.device):
            logprobs, log_z, entropy = _logprobs_forward(
                hidden, weight, targets, bias, temperature, logit_scale, softcap, compute_entropy, low_rank
            )
        if bias is None:
            bias = hidden.new_empty((0,))
        ctx.save_for_backward(hidden, weight, targets, bias, log_z, *low_rank_tensors)
        ctx.has_bias = bias.numel() > 0
        ctx.temperature = temperature
        ctx.logit_scale = logit_scale
        ctx.softcap = softcap
        ctx.scalings = scalings
        ctx.mark_non_differentiable(entropy)
        return logprobs, entropy

    @staticmethod
    def backward(ctx, grad_logprobs, _grad_entropy):
        hidden, weight, targets, bias, log_z, *low_rank_tensors = ctx.saved_tensors
        bias = bias if ctx.has_bias else None
        needs_hidden, needs_weight, _, needs_bias = ctx.needs_input_grad[:4]
        needs_low_rank = ctx.needs_input_grad[9:]
        with _autocast_off(hidden.device):
            grad_hidden, grad_weight, grad_bias, grad_low_rank = _logprobs_backward(
                hidden,
                weight,
                targets,
                bias,
                log_z,
                grad_logprobs,
                ctx.temperature,
                ctx.logit_scale,
                ctx.softcap,
                (
                    needs_hidden,
                    needs_weight,
                    needs_bias,
                    [needs_low_rank[i : i + 3] for i in range(0, len(needs_low_rank), 3)],
                ),
                _low_rank_deltas(ctx.scalings, low_rank_tensors),
            )
        grad_low_rank_tensors = [
            None if grad is None else grad.to(tensor.dtype)
            for grad, tensor in zip((grad for grads in grad_low_rank for grad in grads), low_rank_tensors, strict=True)
        ]
        return (
            None if grad_hidden is None else grad_hidden.to(hidden.dtype),
            None if grad_weight is None else grad_weight.to(weight.dtype),
            None,
            None if grad_bias is None else grad_bias.to(bias.dtype),
            None,
            None,
            None,
            None,
            None,
            *grad_low_rank_tensors,
        )


def chunked_head_logprobs(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    bias: torch.Tensor | None,
    temperature: float,
    *,
    logit_scale: float | None = None,
    softcap: float | None = None,
    compute_entropy: bool = False,
    low_rank: Sequence[LowRankDelta] = (),
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``(logps, entropy)`` of ``targets`` under ``softcap(logit_scale · (hidden @ weight.T + Σ delta + bias)) /
    temperature``, each ``low_rank`` delta's ``inputs`` shaped like ``hidden``. ``hidden`` ``(..., H)`` and ``weight``
    ``(V, H)``. ``targets`` is ``(...)``, one per position, or ``(..., m)`` (``hidden``'s leading shape plus ``m``),
    ``m`` per position sharing its one ``log Z``: column 0 the sampled token, the rest candidate ids; an id below 0 is
    masked, its log-prob ``-inf`` and its gradient zero. ``logps`` is shaped like ``targets``, ``entropy`` like one
    target per position. The ``(..., m)`` form compiles a pair of tile graphs of its own, so a run's first call in each
    form is its warm-up."""
    multi = targets.dim() > 1 and targets.shape[:-1] == hidden.shape[:-1]
    flat_targets = targets.reshape(-1, targets.shape[-1]) if multi else targets.reshape(-1)
    low_rank_tensors = [
        tensor
        for delta in low_rank
        for tensor in (delta.inputs.reshape(-1, delta.inputs.shape[-1]), delta.down, delta.up)
    ]
    logps, entropy = _HeadLogprobFunction.apply(
        hidden.reshape(-1, hidden.shape[-1]),
        weight,
        flat_targets,
        bias,
        temperature,
        logit_scale,
        softcap,
        compute_entropy,
        tuple(delta.scaling for delta in low_rank),
        *low_rank_tensors,
    )
    entropy_shape = targets.shape[:-1] if multi else targets.shape
    return logps.reshape(targets.shape), entropy.reshape(entropy_shape) if compute_entropy else None
