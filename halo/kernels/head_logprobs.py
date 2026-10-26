"""Log-probs (and entropy) of target tokens straight from hidden states, never materializing the logits.

A dual-chunked (sequence × vocab) matmul with online softmax and a recompute backward; the head's logit
scale, softcap and temperature apply inside every tile. The logits-plane form is :mod:`halo.kernels.logprobs`.
"""

import torch

# The working set is a few [seq, vocab] fp32 tiles (4096 × 16384 × 4 B = 256 MiB each); the loop runs
# (T / seq) × (V / vocab) iterations, so tiles sized well below this leave the sweep launch-bound on
# large-vocabulary models.
SEQ_TILE = 4096
VOCAB_TILE = 16384


def _matmul_fp32(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """``a @ b`` as fp32, or ``out += a @ b`` into an fp32 ``out``.

    On CUDA with a bf16 or fp16 ``b``, both operands are in ``b``'s dtype (an fp32 ``a``, such as the
    backward's gradient tile, is cast to it first) and the tensor-core GEMM accumulates in fp32 and
    returns fp32, so neither the product nor the running sum ``out`` is rounded to half precision. Any
    other ``b`` (CPU, fp32) takes an fp32 matmul. Half operands are not upcast on CUDA: under the pinned
    ``highest`` fp32 matmul precision that product would run on the CUDA cores.
    """
    if b.is_cuda and b.dtype in (torch.bfloat16, torch.float16):
        a = a.to(b.dtype)
        if out is None:
            return torch.mm(a, b, out_dtype=torch.float32)
        return torch.addmm(out, a, b, out_dtype=torch.float32, out=out)
    product = a.float() @ b.float()
    return product if out is None else out.add_(product)


def _logits_tile(
    hidden_chunk, weight_chunk, bias, vocab_start, vocab_end, logit_scale, softcap, inv_t
) -> torch.Tensor:
    """One ``[seq, vocab-tile]`` plane of temperature-scaled logits: the logit scale, then the softcap
    (``cap · tanh(logits / cap)``), then the temperature, as the model's forward orders them."""
    logits_chunk = _matmul_fp32(hidden_chunk, weight_chunk.to(hidden_chunk.dtype).t())
    if bias is not None:
        logits_chunk.add_(bias[vocab_start:vocab_end].to(torch.float32))
    if logit_scale is not None:
        logits_chunk.mul_(logit_scale)
    if softcap is not None:
        logits_chunk.div_(softcap).tanh_().mul_(softcap)
    return logits_chunk.mul_(inv_t)


def _online_softmax_tile(
    hidden_chunk, weight, targets_chunk, bias, inv_t, logit_scale, softcap, compute_entropy
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """One sequence tile swept over every vocab tile: ``(target logit, log Z, entropy or None)``, each
    ``(rows,)`` fp32.

    The entropy needs only ``u = Σ exp(z−m)·z`` on top of the online-softmax accumulators, so it rides
    the same sweep (``entropy = log Z − u/s``) instead of a second full-vocab matmul.
    """
    device = hidden_chunk.device
    n_chunk = hidden_chunk.shape[0]
    vocab_size = weight.shape[0]
    max_old = torch.full((n_chunk,), float("-inf"), device=device, dtype=torch.float32)
    sum_exp = torch.zeros((n_chunk,), device=device, dtype=torch.float32)
    sum_exp_z = torch.zeros((n_chunk,), device=device, dtype=torch.float32) if compute_entropy else None
    target_logit = torch.zeros((n_chunk,), device=device, dtype=torch.float32)
    row_idx = torch.arange(n_chunk, device=device)

    for vocab_start in range(0, vocab_size, VOCAB_TILE):
        vocab_end = min(vocab_start + VOCAB_TILE, vocab_size)
        logits_chunk = _logits_tile(
            hidden_chunk, weight[vocab_start:vocab_end], bias, vocab_start, vocab_end, logit_scale, softcap, inv_t
        )

        chunk_max = logits_chunk.amax(dim=-1)
        max_new = torch.maximum(max_old, chunk_max)
        rescale = torch.exp(max_old - max_new)
        chunk_exp = torch.exp(logits_chunk - max_new.unsqueeze(-1))

        sum_exp = sum_exp * rescale + chunk_exp.sum(dim=-1)
        if compute_entropy:
            sum_exp_z = sum_exp_z * rescale + (chunk_exp * logits_chunk).sum(dim=-1)
        max_old = max_new

        in_chunk = (targets_chunk >= vocab_start) & (targets_chunk < vocab_end)
        local_idx = torch.clamp(targets_chunk - vocab_start, 0, vocab_end - vocab_start - 1)
        target_logit += logits_chunk[row_idx, local_idx] * in_chunk

    log_z = max_old + torch.log(sum_exp)
    return target_logit, log_z, log_z - sum_exp_z / sum_exp if compute_entropy else None


@torch.no_grad()
def _logprobs_forward(
    hidden, weight, targets, bias, temperature, logit_scale, softcap, compute_entropy
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(logprobs, log_z, entropy)``, each ``(n_rows,)`` fp32; the entropy is zeros unless ``compute_entropy``."""
    n_rows = hidden.shape[0]
    inv_t = 1.0 / temperature
    logprobs = torch.empty((n_rows,), device=hidden.device, dtype=torch.float32)
    log_z = torch.empty((n_rows,), device=hidden.device, dtype=torch.float32)
    entropy = torch.zeros((n_rows,), device=hidden.device, dtype=torch.float32)

    for seq_start in range(0, n_rows, SEQ_TILE):
        seq_end = min(seq_start + SEQ_TILE, n_rows)
        target_logit, log_z_chunk, entropy_chunk = _online_softmax_tile(
            hidden[seq_start:seq_end],
            weight,
            targets[seq_start:seq_end],
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
    """Each logits tile is recomputed from the saved ``log_z`` instead of a stored ``[T, V]`` plane. Both
    gradient GEMMs take the fp32 gradient tile cast to the activations' dtype and accumulate in fp32
    (:func:`_matmul_fp32`). Under a softcap the tile's gradient carries the ``tanh`` derivative,
    ``1 − (logits / cap)²`` on the capped logits, and under a logit scale that scale."""
    inv_t = 1.0 / temperature
    n_rows, _ = hidden.shape
    vocab_size = weight.shape[0]
    has_bias = bias is not None

    grad_hidden = torch.zeros(hidden.shape, device=hidden.device, dtype=torch.float32)
    grad_weight = torch.zeros(weight.shape, device=weight.device, dtype=torch.float32)
    grad_bias = torch.zeros((vocab_size,), device=weight.device, dtype=torch.float32) if has_bias else None
    grad_logprobs = grad_logprobs.to(torch.float32)

    for seq_start in range(0, n_rows, SEQ_TILE):
        seq_end = min(seq_start + SEQ_TILE, n_rows)
        hidden_chunk = hidden[seq_start:seq_end]
        targets_chunk = targets[seq_start:seq_end]
        grad_chunk = grad_logprobs[seq_start:seq_end]
        logz_chunk = log_z[seq_start:seq_end]
        row_idx = torch.arange(seq_end - seq_start, device=hidden.device)

        for vocab_start in range(0, vocab_size, VOCAB_TILE):
            vocab_end = min(vocab_start + VOCAB_TILE, vocab_size)
            weight_chunk = weight[vocab_start:vocab_end]
            logits_chunk = _logits_tile(
                hidden_chunk, weight_chunk, bias, vocab_start, vocab_end, logit_scale, softcap, inv_t
            )

            probs = torch.exp(logits_chunk - logz_chunk.unsqueeze(-1))
            grad_logits = (-grad_chunk).unsqueeze(-1) * probs

            in_chunk = (targets_chunk >= vocab_start) & (targets_chunk < vocab_end)
            local_idx = torch.clamp(targets_chunk - vocab_start, 0, vocab_end - vocab_start - 1)
            grad_logits[row_idx, local_idx] += grad_chunk * in_chunk
            grad_logits.mul_(inv_t)
            if softcap is not None:
                grad_logits.mul_(1.0 - (logits_chunk * (temperature / softcap)).square())
            if logit_scale is not None:
                grad_logits.mul_(logit_scale)

            _matmul_fp32(grad_logits, weight_chunk.to(hidden_chunk.dtype), out=grad_hidden[seq_start:seq_end])
            _matmul_fp32(grad_logits.t(), hidden_chunk, out=grad_weight[vocab_start:vocab_end])
            if has_bias:
                grad_bias[vocab_start:vocab_end].add_(grad_logits.sum(dim=0))

    return grad_hidden, grad_weight, grad_bias


class _HeadLogprobFunction(torch.autograd.Function):
    """Selective log-probs and entropy in one vocab sweep, with the recompute backward. Only the
    log-probs carry gradient; the entropy is a detached diagnostic (``mark_non_differentiable``)."""

    @staticmethod
    def forward(ctx, hidden, weight, targets, bias, temperature, logit_scale, softcap, compute_entropy):
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
