#!/usr/bin/env python
"""Merge an EP-sharded checkpoint into one ``from_pretrained``-loadable HuggingFace checkpoint.

Required for sharded EP saves (``save_sharded_ep: true``): each rank writes its own expert slices
under ``.shard_N`` suffixes for fast parallel I/O, a layout no loader reads (the EP loader refuses it
with these merge instructions). The merge writes each family's ``ExpertLayout`` through the same
``to_hub_expert_tensors`` the gathered save applies, so merged-from-sharded equals gathered.
A gathered save (``save_sharded_ep: false``) needs no merge; it is slower to write but directly
loadable. The merge carries the save's resume state, optimizer shards included, so the merged directory
resumes exactly on the topology that wrote it; the per-rank files are hard-linked when the output is on
the input's filesystem, and a later training save into either directory replaces them there rather than
writing through the link.

The merge is streamed: input shards are opened once and read group by group (one MoE layer's
experts, or one replicated tensor, at a time), each merged group is transformed and handed straight
to an incremental shard writer. Peak host RAM is one merged layer plus one pending output shard
rather than the whole model, so a 397B artifact merges on an ordinary host.

Output is HuggingFace-style sharded safetensors (``model-0000N-of-000XX.safetensors`` +
``model.safetensors.index.json``); ``--max_shard_size`` sets the per-file cap.

Usage:
    python scripts/after_training/merge_ep_shards.py \\
        --input_dir /path/to/ep_checkpoint --output_dir /path/to/merged_checkpoint
"""

import argparse

from halo.checkpoint.ep_shard_merge import merge_ep_shards
from halo.log import configure_cli_logging
from scripts.cli_arguments import add_max_shard_size_arg


def main():
    configure_cli_logging()
    parser = argparse.ArgumentParser(
        description="Merge EP-sharded checkpoint into standard HuggingFace format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Required after a run with save_sharded_ep: true, whose per-rank shards load only once merged.
A run with save_sharded_ep: false writes a gathered checkpoint that needs no merge.

Examples:
    # Merge Qwen3 MoE EP checkpoint
    python scripts/after_training/merge_ep_shards.py \\
        --input_dir checkpoints/qwen3-moe-ep \\
        --output_dir checkpoints/qwen3-moe-merged

    # Merge GptOss EP checkpoint
    python scripts/after_training/merge_ep_shards.py \\
        --input_dir checkpoints/gpt-oss-ep \\
        --output_dir checkpoints/gpt-oss-merged
        """,
    )
    parser.add_argument("--input_dir", required=True, help="Path to the per-rank sharded checkpoint")
    parser.add_argument("--output_dir", required=True, help="Path to save the merged checkpoint")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    add_max_shard_size_arg(parser)
    parser.add_argument(
        "--delete_input_shards",
        action="store_true",
        help="Delete the input shard files after the merged checkpoint is fully saved (not before: "
        "a failed merge must not destroy the only copy of the weights). Peak disk usage therefore "
        "briefly holds input shards + merged output simultaneously.",
    )
    args = parser.parse_args()

    merge_ep_shards(
        args.input_dir,
        args.output_dir,
        verbose=not args.quiet,
        max_shard_size=args.max_shard_size,
        delete_input_shards=args.delete_input_shards,
    )
    print(f"\n✓ Merged EP checkpoint saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
