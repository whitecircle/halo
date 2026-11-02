"""Log-probs (and entropy) of target tokens straight from hidden states, never materializing the logits.

A dual-chunked (sequence × vocab) matmul with online softmax and a recompute backward; the head's logit
scale, softcap and temperature apply inside every tile. Everything a tile does after its GEMM is one function
per direction (:func:`_forward_tile`, :func:`_backward_tile`), which runs on CUDA as one compiled graph. The
logits-plane form is :mod:`halo.kernels.logprobs`.
"""

import torch

# The working set is a few [seq, vocab] fp32 tiles (4096 × 16384 × 4 B = 256 MiB each); the loop runs
# (T / seq) × (V / vocab) iterations, so tiles sized well below this leave the sweep launch-bound on
# large-vocabulary models.
SEQ_TILE = 4096
VOCAB_TILE = 16384

# Device types whose tiles run compiled. A CPU compile costs seconds per graph on a path only tests take,
# so there the same functions run eagerly.
_COMPILED_DEVICE_TYPES = frozenset({"cuda"})

# ``deterministic`` picks kernel configs without benchmarking on the device, so ranks that see the same tiles
# (TP and ETP replicas) and reruns compile the same kernels. Each graph is a few small kernels, so one compile
# thread builds them in-process instead of starting a pool of up to 32 compile workers per rank.
_TILE_COMPILE_OPTIONS = {"deterministic": True, "compile_threads": 1}


def _gemm_dtype(t: torch.Tensor) -> torch.dtype:
    """The dtype both operands of a GEMM against ``t`` take: ``t``'s half dtype on CUDA, where the tensor
    cores accumulate in fp32, and fp32 otherwise."""
    return t.dtype if t.is_cuda and t.dtype in (torch.bfloat16, torch.float16) else torch.float32


def _matmul_fp32(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """``a @ b`` as fp32, or ``out += a @ b`` into an fp32 ``out``.

    On CUDA with a bf16 or fp16 ``b``, both operands are in ``b``'s dtype (an fp32 ``a`` is cast to it
    first) and the tensor-core GEMM accumulates in fp32 and returns fp32, so neither the product nor the
    running sum ``out`` is rounded to half precision. Any other ``b`` (CPU, fp32) takes an fp32 matmul. Half
    operands are not upcast on CUDA: under the pinned ``highest`` fp32 matmul precision that product would
    run on the CUDA cores.
    """
    dtype = _gemm_dtype(b)
    if dtype != torch.float32:
        a = a.to(dtype)
        if out is None:
            return torch.mm(a, b, out_dtype=torch.float32)
        return torch.addmm(out, a, b, out_dtype=torch.float32, out=out)
    product = a.float() @ b.float()
    return product if out is None else out.add_(product)


def _head_logits(raw, bias, vocab_start, logit_scale, softcap) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The head's fp32 logits on one GEMM tile before the temperature (the fp32 bias, the logit scale, then the
    softcap ``cap · tanh(logits / cap)``, as the model's forward orders them), and the cap's ``tanh``, which
    its derivative reuses (``None`` uncapped)."""
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
    """``[rows, width]`` one-hot of each row's target inside the tile, all false for a target outside it. An
    iota compare rather than an index write, so a compiled tile keeps it inside its one kernel."""
    return torch.arange(width, device=targets.device) == (targets - vocab_start).unsqueeze(-1)


def _forward_tile(
    raw, bias, targets, vocab_start, max_old, sum_exp, sum_exp_z, target_logit, inv_t, logit_scale, softcap
):
    """One vocab tile folded into a sequence tile's online-softmax accumulators over the temperature-scaled
    logits ``z``: ``(max, Σ exp(z − max), Σ exp(z − max)·z or None, target logit)``, each ``(rows,)`` fp32.

    The entropy needs only ``Σ exp(z − max)·z`` on top of the softmax accumulators, so it rides the same
    sweep (``entropy = log Z − Σ exp(z − max)·z / Σ exp(z − max)``) instead of a second full-vocab matmul.
    """
    logits = _head_logits(raw, bias, vocab_start, logit_scale, softcap)[0] * inv_t
    max_new = torch.maximum(max_old, logits.amax(dim=-1))
    rescale = torch.exp(max_old - max_new)
    exps = torch.exp(logits - max_new.unsqueeze(-1))
    sum_exp = sum_exp * rescale + exps.sum(dim=-1)
    if sum_exp_z is not None:
        sum_exp_z = sum_exp_z * rescale + (exps * logits).sum(dim=-1)
    is_target = _is_target(targets, vocab_start, raw.shape[1])
    target_logit = target_logit + torch.where(is_target, logits, 0.0).sum(dim=-1)
    return max_new, sum_exp, sum_exp_z, target_logit


def _backward_tile(raw, bias, targets, vocab_start, log_z, grad, inv_t, logit_scale, softcap, grad_dtype):
    """The loss gradient on one GEMM tile, ``grad · (onehot − softmax)`` carried through the temperature, the
    cap's ``1 − tanh²`` and the logit scale: ``(gradient in grad_dtype, its fp32 column sum or None)``. The
    gradient comes out in the dtype the two gradient GEMMs take; the column sum, the bias's gradient, is
    taken before that cast."""
    logits, tanh = _head_logits(raw, bias, vocab_start, logit_scale, softcap)
    probs = torch.exp(logits * inv_t - log_z.unsqueeze(-1))
    grad = grad.unsqueeze(-1)
    is_target = _is_target(targets, vocab_start, raw.shape[1])
    grad_logits = (torch.where(is_target, grad, 0.0) - grad * probs) * inv_t
    if tanh is not None:
        grad_logits = grad_logits * (1.0 - tanh.square())
    if logit_scale is not None:
        grad_logits = grad_logits * logit_scale
    return grad_logits.to(grad_dtype), grad_logits.sum(dim=0) if bias is not None else None


def _compile_tile(fn):
    """``fn`` compiled whole (a graph break raises) for every tile shape at once: rows, tile width and vocab
    offset are symbolic, and the head's scalars are graph inputs."""
    return torch.compile(fn, dynamic=True, fullgraph=True, options=_TILE_COMPILE_OPTIONS)


_COMPILED_TILES = (_compile_tile(_forward_tile), _compile_tile(_backward_tile))


def _tile_functions(raw: torch.Tensor):
    """``(forward_tile, backward_tile)`` for the GEMM tile ``raw``: compiled on a compiled device type, eager on
    a tile one row or one column wide. Dynamo specializes a size of 1, so such a tile, which a ragged batch can
    end on at any step, would otherwise compile a graph of its own mid-run, on that rank alone."""
    if raw.device.type in _COMPILED_DEVICE_TYPES and min(raw.shape) > 1:
        return _COMPILED_TILES
    return _forward_tile, _backward_tile


def _rows(t: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """``t[start:end]`` as a contiguous tensor of its own, for a compiled tile, whose guards would otherwise key on
    its layout: a view's base sizes (the sweep's row count, a caller's batch), which change from step to step, or
    the stride 0 of an expanded upstream gradient (``logps.sum()``)."""
    return t[start:end].clone(memory_format=torch.contiguous_format)


def _tile_bias(bias: torch.Tensor | None) -> torch.Tensor | None:
    """The bias as the tiles take it, fp32 and detached, so neither its dtype nor whether it trains (a policy's
    head, then a frozen reference's) gives a compiled tile a graph of its own."""
    return None if bias is None else bias.detach().float()


def _autocast_off(device: torch.device):
    """Autocast off around a sweep, which picks every GEMM dtype itself (:func:`_matmul_fp32`): a caller's bf16
    autocast would round an fp32 GEMM's logits to bf16, and give the compiled tiles a graph per autocast state."""
    return torch.autocast(device.type, enabled=False)


def _online_softmax_tile(
    hidden_chunk, weight, targets_chunk, bias, inv_t, logit_scale, softcap, compute_entropy
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """One sequence tile swept over every vocab tile: ``(target logit, log Z, entropy or None)``, each
    ``(rows,)`` fp32."""
    device = hidden_chunk.device
    n_chunk = hidden_chunk.shape[0]
    max_old = torch.full((n_chunk,), float("-inf"), device=device, dtype=torch.float32)
    sum_exp = torch.zeros((n_chunk,), device=device, dtype=torch.float32)
    sum_exp_z = torch.zeros((n_chunk,), device=device, dtype=torch.float32) if compute_entropy else None
    target_logit = torch.zeros((n_chunk,), device=device, dtype=torch.float32)

    for vocab_start in range(0, weight.shape[0], VOCAB_TILE):
        weight_chunk = weight[vocab_start : vocab_start + VOCAB_TILE].to(hidden_chunk.dtype)
        raw = _matmul_fp32(hidden_chunk, weight_chunk.t())
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
    hidden, weight, targets, bias, temperature, logit_scale, softcap, compute_entropy
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(logprobs, log_z, entropy)``, each ``(n_rows,)`` fp32; the entropy is zeros unless ``compute_entropy``."""
    n_rows = hidden.shape[0]
    inv_t = 1.0 / temperature
    bias = _tile_bias(bias)
    logprobs = torch.empty((n_rows,), device=hidden.device, dtype=torch.float32)
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
        )
        log_z[seq_start:seq_end] = log_z_chunk
        logprobs[seq_start:seq_end] = target_logit - log_z_chunk
        if compute_entropy:
            entropy[seq_start:seq_end] = entropy_chunk
    return logprobs, log_z, entropy


def _logprobs_backward(hidden, weight, targets, bias, log_z, grad_logprobs, temperature, logit_scale, softcap):
    """Each logits tile is recomputed from the saved ``log_z`` instead of a stored ``[T, V]`` plane, and its
    gradient (:func:`_backward_tile`) feeds both gradient GEMMs in the activations' GEMM dtype, accumulated in
    fp32 (:func:`_matmul_fp32`)."""
    inv_t = 1.0 / temperature
    bias = _tile_bias(bias)
    n_rows, _ = hidden.shape
    vocab_size = weight.shape[0]
    has_bias = bias is not None
    grad_dtype = _gemm_dtype(hidden)

    grad_hidden = torch.zeros(hidden.shape, device=hidden.device, dtype=torch.float32)
    grad_weight = torch.zeros(weight.shape, device=weight.device, dtype=torch.float32)
    grad_bias = torch.zeros((vocab_size,), device=weight.device, dtype=torch.float32) if has_bias else None
    grad_logprobs = grad_logprobs.to(torch.float32)

    for seq_start in range(0, n_rows, SEQ_TILE):
        seq_end = min(seq_start + SEQ_TILE, n_rows)
        hidden_chunk = hidden[seq_start:seq_end]
        targets_chunk, log_z_chunk, grad_chunk = (
            _rows(t, seq_start, seq_end) for t in (targets, log_z, grad_logprobs)
        )
        for vocab_start in range(0, vocab_size, VOCAB_TILE):
            vocab_end = min(vocab_start + VOCAB_TILE, vocab_size)
            weight_chunk = weight[vocab_start:vocab_end].to(hidden_chunk.dtype)
            raw = _matmul_fp32(hidden_chunk, weight_chunk.t())
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
            _matmul_fp32(grad_logits, weight_chunk, out=grad_hidden[seq_start:seq_end])
            _matmul_fp32(grad_logits.t(), hidden_chunk, out=grad_weight[vocab_start:vocab_end])
            if has_bias:
                grad_bias[vocab_start:vocab_end].add_(grad_bias_tile)

    return grad_hidden, grad_weight, grad_bias


class _HeadLogprobFunction(torch.autograd.Function):
    """Selective log-probs and entropy in one vocab sweep, with the recompute backward. Only the
    log-probs carry gradient; the entropy is a detached diagnostic (``mark_non_differentiable``)."""

    @staticmethod
    def forward(ctx, hidden, weight, targets, bias, temperature, logit_scale, softcap, compute_entropy):
        with _autocast_off(hidden.device):
            logprobs, log_z, entropy = _logprobs_forward(
                hidden, weight, targets, bias, temperature, logit_scale, softcap, compute_entropy
            )
        if bias is None:
            bias = hidden.new_empty((0,))
        ctx.save_for_backward(hidden, weight, targets, bias, log_z)
        ctx.has_bias = bias.numel() > 0
        ctx.temperature = temperature
        ctx.logit_scale = logit_scale
        ctx.softcap = softcap
        ctx.mark_non_differentiable(entropy)
        return logprobs, entropy

    @staticmethod
    def backward(ctx, grad_logprobs, _grad_entropy):
        hidden, weight, targets, bias, log_z = ctx.saved_tensors
        bias = bias if ctx.has_bias else None
        with _autocast_off(hidden.device):
            grad_hidden, grad_weight, grad_bias = _logprobs_backward(
                hidden, weight, targets, bias, log_z, grad_logprobs, ctx.temperature, ctx.logit_scale, ctx.softcap
            )
        grad_bias = None if bias is None else grad_bias.to(bias.dtype)
        return grad_hidden.to(hidden.dtype), grad_weight.to(weight.dtype), None, grad_bias, None, None, None, None


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
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``(logps, entropy)`` of ``targets`` under ``softcap(logit_scale · (hidden @ weight.T + bias)) / temperature``.

    ``hidden`` ``(..., H)``, ``weight`` ``(V, H)``, ``targets`` ``(...)``; both outputs are shaped like
    ``targets``. Differentiable in ``hidden``, ``weight`` and ``bias``; ``temperature`` matches TRL's
    pre-softmax division. ``entropy`` (``-Σ p·log p`` over the vocabulary) is ``None`` unless ``compute_entropy``.
    """
    logps, entropy = _HeadLogprobFunction.apply(
        hidden.reshape(-1, hidden.shape[-1]),
        weight,
        targets.reshape(-1),
        bias,
        temperature,
        logit_scale,
        softcap,
        compute_entropy,
    )
    return logps.reshape(targets.shape), entropy.reshape(targets.shape) if compute_entropy else None
