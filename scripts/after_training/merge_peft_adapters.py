#!/usr/bin/env python
"""Merge a PEFT (LoRA/QLoRA) adapter into its base model, producing a standalone checkpoint.

Covers causal-LM and sequence-classification (``--task classification``) adapters; a VLM base is detected from its
config. ``--fp32_norms`` writes normalization leaves in fp32, the layout ``convert_to_bf16.py`` writes.
``--output_dir`` is written fresh: any ``model*.safetensors`` or index the save did not produce is removed."""

import argparse

from halo.checkpoint.tools.adapter_merge import DEFAULT_NUM_LABELS, merge_peft_adapter
from halo.checkpoint.tools.offline_export import ADAPTER_TASKS
from halo.log import configure_cli_logging
from halo.model_loading.dtype import DTYPE_BY_NAME
from scripts.cli_arguments import (
    add_device_map_arg,
    add_dtype_arg,
    add_max_shard_size_arg,
    add_task_arg,
    add_trust_remote_code_arg,
)


def main():
    configure_cli_logging()
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
        help="Path to save the merged model (every model*.safetensors/index the completed save did not "
        "produce is removed afterwards)",
    )
    add_task_arg(parser, tasks=ADAPTER_TASKS)
    add_dtype_arg(parser)
    add_device_map_arg(parser)
    parser.add_argument(
        "--num_labels",
        type=int,
        default=None,
        help=f"Number of labels of a --task classification head (default: {DEFAULT_NUM_LABELS}); refused under any "
        "other task",
    )
    add_max_shard_size_arg(parser)
    add_trust_remote_code_arg(parser, default=None)
    parser.add_argument(
        "--fp32_norms",
        action="store_true",
        help="Write every normalization leaf in fp32 instead of --dtype (convert_to_bf16.py's layout)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    args = parser.parse_args()
    if args.num_labels is not None and args.task != "classification":
        parser.error(f"--num_labels sizes a classification head, and --task {args.task} loads none")

    merge_peft_adapter(
        adapter_dir=args.adapter_dir,
        output_dir=args.output_dir,
        task=args.task,
        dtype=DTYPE_BY_NAME[args.dtype],
        device_map=args.device_map,
        num_labels=DEFAULT_NUM_LABELS if args.num_labels is None else args.num_labels,
        max_shard_size=args.max_shard_size,
        trust_remote_code=args.trust_remote_code,
        fp32_norms=args.fp32_norms,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
