"""Shared argparse setup and baseline-key spelling for the profiling benchmarks."""

import argparse

import torch
from transformers import AutoConfig

from halo.models.attention.gpt_oss_sinks import SinksPolicy
from halo.models.attention.selection import resolve_attn_implementation
from tests.common.models import DEFAULT_MODEL, MODEL_CONFIGS


def create_benchmark_parser(
    description: str,
    require_ep: bool = True,
) -> argparse.ArgumentParser:
    """ArgumentParser with the common benchmark arguments, for benchmarks to extend."""
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
        "(the pipeline's microbatches are the accumulation); >1 matches a baseline's global batch "
        "to a PP run's.",
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


def axis_tags(**sizes: int) -> list[str]:
    """``<axis><size>`` for every parallel axis wider than one rank: how a baseline key spells its layout."""
    return [f"{axis}{size}" for axis, size in sizes.items() if size > 1]


def pp_topology_kwargs(pp_size: int, world_size: int) -> dict:
    """ParallelismConfig kwargs for benchmarking PP on one physical NVLink domain, simulating one domain per stage. PP
    needs stage boundaries on domain boundaries. P2P boundary activations ride NVLink instead of RDMA: an upper bound
    on cross-node throughput."""
    if pp_size <= 1:
        return {}
    stage_world = world_size // pp_size
    return {"gpus_per_node": stage_world, "nvlink_domain_size": stage_world}


def resolve_benchmark_attn(
    model_name: str,
    attn_implementation: str | None,
    dtype: torch.dtype = torch.bfloat16,
) -> str:
    """Concrete attention backend for a benchmark calling ``from_pretrained`` directly, whose own default ``sdpa`` is a
    different kernel than production."""
    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    return resolve_attn_implementation(model_config, attn_implementation, dtype, sinks_policy=SinksPolicy.NEUTRALIZED)
