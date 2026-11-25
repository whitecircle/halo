#!/usr/bin/env python
"""One causal attention call per sequence length on one GPU, no model: a naive chain that materializes the score
matrix, SDPA as PyTorch dispatches it, and the FlashAttention builds installed (FA4 and FA2 on Blackwell). Each records
device time and peak memory above its inputs, for the forward and for forward plus backward.

    python tests/gpu/profiling/benchmark_attention_scaling.py --output attention.json
"""

import argparse
from collections.abc import Callable

import torch
from torch.nn.attention import SDPBackend

from halo.models.attention.selection import flash_kernel, preferred_flash_impl
from tests.gpu.profiling.measurement import gpu_state, peak_and_seconds, profile_kernels, run_record, write_record

FLASH_IMPLEMENTATIONS = ("flash_attention_2", "flash_attention_3", "flash_attention_4")
IMPLEMENTATIONS = ("naive", "sdpa", *FLASH_IMPLEMENTATIONS)
PASSES = ("forward", "forward_backward")
# A causal backward runs five matmuls of the forward's two.
BACKWARD_FLOP_RATIO = 2.5
REPEAT_SECONDS = 0.1
PROFILED_CALLS = 3


def _naive(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """``softmax(q k^T / sqrt(d)) v`` on ``[batch, heads, seq, head_dim]``: bf16 scores, fp32 softmax, causal mask."""
    seq = q.shape[-2]
    scores = (q @ k.transpose(-2, -1)).float() * q.shape[-1] ** -0.5
    future = torch.ones(seq, seq, dtype=torch.bool, device=q.device).triu_(1)
    scores = scores.masked_fill(future, float("-inf"))
    return scores.softmax(dim=-1).to(q.dtype) @ v


def _attention(implementation: str) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """The call on ``[batch, seq, heads, head_dim]`` operands, the layout a decoder layer hands attention."""
    if implementation in FLASH_IMPLEMENTATIONS:
        flash = flash_kernel(implementation, varlen=False)

        def flash_call(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            out = flash(q, k, v, causal=True)
            return out[0] if isinstance(out, tuple) else out  # FA4 also returns the log-sum-exp.

        return flash_call
    if implementation == "sdpa":
        return lambda q, k, v: torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        ).transpose(1, 2)
    return lambda q, k, v: _naive(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)


def _sdpa_backend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> str:
    choice = torch._fused_sdp_choice(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
    return SDPBackend(choice).name


def _step(call: Callable, q, k, v, grad_out, attention_pass: str) -> Callable[[], None]:
    if attention_pass == "forward":

        def forward() -> None:
            call(q, k, v)

        return forward

    def forward_backward() -> None:
        call(q, k, v).backward(grad_out)
        q.grad = k.grad = v.grad = None

    return forward_backward


def _point(implementation: str, attention_pass: str, seq: int, args) -> dict:
    shape = (args.batch, seq, args.heads, args.head_dim)
    q, k, v = (
        torch.randn(shape, device="cuda", dtype=torch.bfloat16).requires_grad_(attention_pass == "forward_backward")
        for _ in range(3)
    )
    grad_out = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    step = _step(_attention(implementation), q, k, v, grad_out, attention_pass)
    forward_flops = 2 * args.batch * args.heads * seq * seq * args.head_dim  # Two matmuls, half the square: causal.
    flops = forward_flops * (1 if attention_pass == "forward" else 1 + BACKWARD_FLOP_RATIO)
    row = {
        "implementation": implementation,
        "pass": attention_pass,
        "seq": seq,
        "batch": args.batch,
        "heads": args.heads,
        "head_dim": args.head_dim,
        "causal_flops": flops,
        "score_matrix_fp32_bytes": 4 * args.batch * args.heads * seq * seq,
        "input_bytes": 3 * q.numel() * q.element_size(),
    }
    if implementation == "sdpa":
        row["sdpa_backend"] = _sdpa_backend(q, k, v)
    row.update(peak_and_seconds(step, repeats=args.repeats, repeat_seconds=REPEAT_SECONDS))
    if "oom" not in row:
        row["tflops_median"] = flops / row["seconds_summary"]["median"] / 1e12
        row["state"] = gpu_state()
    return row


def _kernel_time(row: dict) -> None:
    """Adds the kernels' own seconds per call, below ``seconds`` when the host's launch path, not the GPU, sets the
    pace. Run after every timed point: a profiling session slows the process's later eager launches."""
    shape = (row["batch"], row["seq"], row["heads"], row["head_dim"])
    training = row["pass"] == "forward_backward"
    q, k, v = (torch.randn(shape, device="cuda", dtype=torch.bfloat16).requires_grad_(training) for _ in range(3))
    grad_out = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    kernels = profile_kernels(_step(_attention(row["implementation"]), q, k, v, grad_out, row["pass"]), PROFILED_CALLS)
    row["kernel_seconds_per_call"] = sum(kernels["kernel_us"]) / 1e6 / PROFILED_CALLS
    row["kernels_per_call"] = kernels["kernels_per_call"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, help="JSON file, rewritten after each point")
    parser.add_argument("--seqs", type=int, nargs="+", default=(1024, 2048, 4096, 8192, 16384, 32768, 65536))
    parser.add_argument(
        "--implementations",
        nargs="+",
        choices=IMPLEMENTATIONS,
        help="Default: naive, sdpa, this GPU's flash build and FlashAttention-2",
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=7, help="Timed repeats per point; each is one sample")
    args = parser.parse_args()

    torch.cuda.set_device(0)
    implementations = args.implementations or list(
        dict.fromkeys(("naive", "sdpa", preferred_flash_impl(), "flash_attention_2"))
    )
    record = {**run_record(), "rows": []}
    for implementation in implementations:
        for attention_pass in PASSES:
            for seq in args.seqs:
                row = _point(implementation, attention_pass, seq, args)
                record["rows"].append(row)
                write_record(args.output, record)
                torch.cuda.empty_cache()
                if "oom" in row:
                    print(f"  {implementation:>6} {attention_pass:>16} seq={seq:>6}  OOM; longer rows skipped")
                    break
                print(
                    f"  {implementation:>6} {attention_pass:>16} seq={seq:>6}  "
                    f"{row['seconds_summary']['median'] * 1e3:9.3f} ms  {row['tflops_median']:7.1f} TFLOP/s  "
                    f"peak +{row['peak_bytes_above_inputs'] / 2**30:7.2f} GiB  {row.get('sdpa_backend', '')}"
                )
    for row in record["rows"]:
        if "oom" not in row:
            _kernel_time(row)
            torch.cuda.empty_cache()
    write_record(args.output, record)


if __name__ == "__main__":
    main()
