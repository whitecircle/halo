"""Shared argparse setup for the profiling benchmarks.

Individual benchmarks extend the parser with add_argument() before calling
parse_args().
"""

import argparse

import torch
from transformers import AutoConfig

from src.models.patches.attention import resolve_attn_implementation
from tests.common.models import DEFAULT_MODEL, MODEL_CONFIGS


def create_benchmark_parser(
    description: str,
    require_ep: bool = True,
) -> argparse.ArgumentParser:
    """Create ArgumentParser with common benchmark arguments.

    Common args: --model, --model_path, --attn_implementation, --no_liger,
                 --ep, --seq, --steps, --warmup, --batch_size, --num_samples.

    Returns parser that benchmarks can extend with custom args.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        choices=list(MODEL_CONFIGS.keys()),
        help="Model config key",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Override model path (instead of using MODEL_CONFIGS)",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        help="Attention backend. Unset (default) auto-detects per architecture: "
        "flash_attention_4 on Blackwell (B200/B300), flash_attention_3 on Hopper "
        "(H100/H200), flash_attention_2 otherwise — matching the production path.",
    )
    parser.add_argument(
        "--no_liger",
        action="store_true",
        help="Disable Liger kernels",
    )
    if require_ep:
        parser.add_argument(
            "--ep",
            type=int,
            required=True,
            help="Expert parallel size",
        )
    parser.add_argument(
        "--seq",
        type=int,
        default=8192,
        help="Sequence length (default 8192 — the standard profiling length)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Training steps",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Warmup steps for MFU measurement (>=3: FA4's first-use CuTe JIT compile must land in "
        "warmup, not a measured step, on a cold kernel cache — else the first measured step is "
        "inflated ~10x and throughput reads artificially low)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Per-device train batch size",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=1,
        help="Gradient accumulation steps. Under PP leave at 1 and set --pp_microbatches instead "
        "(the pipeline's microbatches ARE the accumulation); >1 is for matching a baseline's "
        "global batch to a PP run's.",
    )
    parser.add_argument(
        "--pp",
        type=int,
        default=1,
        help="Pipeline parallel size (stages). Splits decoder layers into pp contiguous stages; "
        "data_parallel_size divides by pp (a whole pipeline chain consumes one batch).",
    )
    parser.add_argument(
        "--pp_microbatches",
        type=int,
        default=0,
        help="Microbatches per optimizer step under PP (0 = resolved from gradient_accumulation_"
        "steps). per_device_train_batch_size must be divisible by it.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=64,
        help="Number of dummy samples",
    )
    return parser


def pp_topology_kwargs(pp_size: int, world_size: int) -> dict:
    """ParallelismConfig topology kwargs for benchmarking PP on a single physical NVLink domain.

    When the PP engine lands, this simulates one NVLink domain per stage (PP requires stage
    boundaries on NVLink-domain boundaries, which one 8-GPU domain cannot satisfy for pp > 1), so
    each stage owns a whole simulated domain and every intra-stage EP/FSDP group stays NVLink-local.
    The benchmarks build ``ParallelismConfig`` directly, bypassing the config-time rejection in
    ``parallelism_config_from_args``, so this helper rejects ``--pp > 1`` up front rather than
    failing after model load.
    """
    if pp_size <= 1:
        return {}
    raise NotImplementedError(
        "Pipeline parallelism is not yet available in this release — run the benchmark with "
        "--pp 1 (see agent-docs/parallelism/pipeline-parallelism.md)."
    )


def resolve_benchmark_attn(
    model_name: str,
    attn_implementation: str | None,
    dtype: torch.dtype = torch.bfloat16,
) -> str:
    """Concrete attention backend for a benchmark that calls ``from_pretrained`` directly.

    ``load_distributed_model`` resolves ``attn_implementation=None`` through
    :func:`resolve_attn_implementation` (GPU auto-detect, per-family limits and the sinks
    validator), but a direct ``from_pretrained`` falls back to transformers' own default,
    ``sdpa``. Benchmarks that bypass the loader resolve the backend here, or they measure a
    different kernel than production runs.
    """
    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    return resolve_attn_implementation(model_config, attn_implementation, dtype)
