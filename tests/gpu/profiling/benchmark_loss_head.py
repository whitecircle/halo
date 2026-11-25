#!/usr/bin/env python
"""The language-model loss on one GPU at a real head shape, no model: plain logits plus cross-entropy (the logits
upcast to fp32, as transformers' loss does) against Liger's fused linear cross-entropy, which never materializes the
logits. Device time and peak memory above the inputs, for the forward and for forward plus backward.

    python tests/gpu/profiling/benchmark_loss_head.py --output loss_head.json
"""

import argparse
from collections.abc import Callable

import torch
import torch.nn.functional as F
from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss

from tests.gpu.profiling.measurement import gpu_state, peak_and_seconds, run_record, write_record

ARMS = ("logits_ce", "liger_flce")
PASSES = ("forward", "forward_backward")
REPEAT_SECONDS = 0.1
SEED = 0


def _loss(arm: str) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    if arm == "liger_flce":
        fused = LigerFusedLinearCrossEntropyLoss()
        return lambda hidden, weight, targets: fused(weight, hidden, targets)
    return lambda hidden, weight, targets: F.cross_entropy((hidden @ weight.t()).float(), targets)


def _point(arm: str, loss_pass: str, tokens: int, hidden_size: int, vocab: int, repeats: int) -> dict:
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    training = loss_pass == "forward_backward"
    hidden = torch.randn(tokens, hidden_size, device="cuda", generator=generator, dtype=torch.bfloat16)
    weight = torch.randn(vocab, hidden_size, device="cuda", generator=generator, dtype=torch.bfloat16)
    weight = (weight * hidden_size**-0.5).requires_grad_(training)
    hidden.requires_grad_(training)
    targets = torch.randint(0, vocab, (tokens,), device="cuda", generator=generator)
    loss = _loss(arm)

    def step() -> None:
        if training:
            loss(hidden, weight, targets).backward()
            hidden.grad = weight.grad = None
        else:
            with torch.no_grad():
                loss(hidden, weight, targets)

    row = {
        "arm": arm,
        "pass": loss_pass,
        "tokens": tokens,
        "hidden_size": hidden_size,
        "vocab": vocab,
        "logits_bf16_bytes": 2 * tokens * vocab,
        "input_bytes": 2 * (hidden.numel() + weight.numel()),
    }
    row.update(peak_and_seconds(step, repeats=repeats, repeat_seconds=REPEAT_SECONDS))
    if "oom" not in row:
        row["state"] = gpu_state()
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, help="JSON file, rewritten after each point")
    parser.add_argument("--tokens", type=int, nargs="+", default=(4096, 16384, 32768, 65536))
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=(2560, 4096), help="Qwen3-4B and Qwen3-8B")
    parser.add_argument("--vocab", type=int, default=151936, help="Qwen3's vocabulary")
    parser.add_argument("--repeats", type=int, default=7, help="Timed repeats per point; each is one sample")
    args = parser.parse_args()

    torch.cuda.set_device(0)
    record = {**run_record(), "rows": []}
    for hidden_size in args.hidden_sizes:
        for arm in ARMS:
            for loss_pass in PASSES:
                for tokens in args.tokens:
                    row = _point(arm, loss_pass, tokens, hidden_size, args.vocab, args.repeats)
                    record["rows"].append(row)
                    write_record(args.output, record)
                    torch.cuda.empty_cache()
                    if "oom" in row:
                        print(f"  H={hidden_size} {arm:>10} {loss_pass:>16} T={tokens:>6}  OOM", flush=True)
                        break
                    print(
                        f"  H={hidden_size} {arm:>10} {loss_pass:>16} T={tokens:>6}  "
                        f"{row['seconds_summary']['median'] * 1e3:8.2f} ms  "
                        f"peak +{row['peak_bytes_above_inputs'] / 2**30:6.2f} GiB",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
