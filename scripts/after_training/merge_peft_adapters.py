#!/usr/bin/env python
"""Merge a PEFT (LoRA/QLoRA) adapter into its base model, producing a standalone checkpoint.

A LoRA run's output directory carries only ``adapter_model.safetensors`` + ``adapter_config.json``;
this loads the base model, applies the adapter and saves a standard HuggingFace checkpoint. Covers
causal-LM and sequence-classification (``--task classification``) adapters; VLM bases and
``trust_remote_code`` families are auto-detected from the base model path. ``--output_dir`` is
written fresh — every ``model*.safetensors`` or index the completed save did not produce is swept
away afterwards.

Usage:
    python scripts/after_training/merge_peft_adapters.py \\
        --adapter_dir checkpoints/sft-qwen3-4b-lora --output_dir checkpoints/sft-qwen3-4b-merged

    # Reward/classification adapter; --device_map auto offloads a model too large for one GPU
    python scripts/after_training/merge_peft_adapters.py \\
        --adapter_dir checkpoints/reward-lora --output_dir checkpoints/reward-merged \\
        --task classification --num_labels 1
"""

import argparse
import logging

import torch
from accelerate import PartialState
from transformers import AutoModelForSequenceClassification

import src.distributed.expert_parallel.layers.roster  # noqa: F401 — registers the EP export roster the config finalizer requires
from scripts._common import add_max_shard_size_arg, add_trust_remote_code_arg
from src.checkpoint.adapters import merge_adapter_into_base
from src.checkpoint.format import DEFAULT_MAX_SHARD_SIZE
from src.log import configure_cli_logging
from src.models.loading.checkpoint_coverage import from_pretrained_verified
from src.models.loading.dtype import DTYPE_BY_NAME
from src.models.loading.model_preparation import auto_load_model

configure_cli_logging()
logger = logging.getLogger(__name__)


def _load_base_model(
    base_model_path: str,
    task: str,
    dtype: torch.dtype,
    device_map: str | None,
    num_labels: int,
    attn_implementation: str | None,
    *,
    excuse_task_head: bool = False,
    trust_remote_code: bool = True,
):
    """Load the base model with appropriate class and settings.

    Both branches go through the checkpoint-coverage gate: a base directory that is truncated, or
    whose keys do not match the architecture, otherwise random-initializes the absent tensors with
    only a log line, and the merge would ship those weights as a finished model. Which absence may
    be excused is decided by :func:`merge_adapter_into_base` off the adapter's own declaration.
    """
    model_kwargs = {
        "dtype": dtype,
        "trust_remote_code": trust_remote_code,
    }
    if device_map:
        model_kwargs["device_map"] = device_map
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    if task == "classification":
        model_kwargs["num_labels"] = num_labels
        return from_pretrained_verified(
            AutoModelForSequenceClassification,
            base_model_path,
            excuse_task_head=excuse_task_head,
            **model_kwargs,
        )

    # causal_lm: resolve the widest Auto* class from the config so VLM base models load as the full
    # *ForConditionalGeneration wrapper instead of the text-only subclass.
    return auto_load_model(base_model_path, **model_kwargs)


def merge_peft_adapter(
    adapter_dir: str,
    output_dir: str,
    task: str = "causal_lm",
    dtype: torch.dtype = torch.bfloat16,
    device_map: str | None = None,
    num_labels: int = 1,
    max_shard_size: str = DEFAULT_MAX_SHARD_SIZE,
    attn_implementation: str | None = None,
    trust_remote_code: bool = True,
    verbose: bool = True,
):
    """Merge a PEFT adapter into its base model and save the result.

    Args:
        adapter_dir: Path to the PEFT adapter directory (contains adapter_model.safetensors)
        output_dir: Path to save the merged model
        task: Model task type ("causal_lm" or "classification")
        dtype: Model dtype for loading/saving
        device_map: Device map for loading large models (e.g., "auto", "cpu")
        num_labels: Number of labels for classification models
        max_shard_size: Maximum shard size for output safetensors
        attn_implementation: Override attention implementation (e.g., "flash_attention_2", "sdpa")
        trust_remote_code: Passed to the base-model and processor loads
        verbose: Log progress messages
    """
    if verbose:
        logger.info(f"Task: {task}, dtype: {dtype}")

    def load_base_model(base_model_path: str, *, excuse_task_head: bool):
        return _load_base_model(
            base_model_path,
            task,
            dtype,
            device_map,
            num_labels,
            attn_implementation,
            excuse_task_head=excuse_task_head,
            trust_remote_code=trust_remote_code,
        )

    merged_model = merge_adapter_into_base(
        adapter_dir,
        output_dir,
        load_base_model=load_base_model,
        tool="merge_peft_adapters",
        device_map=device_map,
        max_shard_size=max_shard_size,
        trust_remote_code=trust_remote_code,
        verbose=verbose,
    )

    if verbose:
        total_params = sum(p.numel() for p in merged_model.parameters())
        logger.info(f"Merged model saved to: {output_dir}")
        logger.info(f"Total parameters: {total_params:,}")

    del merged_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Merge PEFT (LoRA) adapter into base model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Standard merge
    python scripts/after_training/merge_peft_adapters.py \\
        --adapter_dir checkpoints/sft-lora \\
        --output_dir checkpoints/sft-merged

    # Classification model
    python scripts/after_training/merge_peft_adapters.py \\
        --adapter_dir checkpoints/reward-lora \\
        --output_dir checkpoints/reward-merged \\
        --task classification --num_labels 1

    # Large model with device offloading
    python scripts/after_training/merge_peft_adapters.py \\
        --adapter_dir checkpoints/large-lora \\
        --output_dir checkpoints/large-merged \\
        --device_map auto
        """,
    )
    parser.add_argument(
        "--adapter_dir",
        required=True,
        help="Path to PEFT adapter directory (contains adapter_model.safetensors)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Path to save the merged model (any model*.safetensors/index already there is removed first)",
    )
    parser.add_argument(
        "--task",
        choices=["causal_lm", "classification"],
        default="causal_lm",
        help="Model task type (default: causal_lm)",
    )
    parser.add_argument(
        "--dtype",
        choices=list(DTYPE_BY_NAME),
        default="bf16",
        help="Model dtype (default: bf16)",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default=None,
        help="Device map for loading large models (e.g., 'auto', 'cpu')",
    )
    parser.add_argument(
        "--num_labels",
        type=int,
        default=1,
        help="Number of labels for classification models (default: 1)",
    )
    add_max_shard_size_arg(parser)
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        help="Override attention implementation (e.g., 'flash_attention_2', 'sdpa')",
    )
    add_trust_remote_code_arg(parser)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    args = parser.parse_args()

    # The model-loading helpers log through the accelerate logger, which raises unless the state is initialized;
    # this script is a plain single-process tool, so initialize it explicitly.
    PartialState()

    merge_peft_adapter(
        adapter_dir=args.adapter_dir,
        output_dir=args.output_dir,
        task=args.task,
        dtype=DTYPE_BY_NAME[args.dtype],
        device_map=args.device_map,
        num_labels=args.num_labels,
        max_shard_size=args.max_shard_size,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
