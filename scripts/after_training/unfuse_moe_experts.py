"""Un-fuse stacked MoE expert weights into per-expert tensors for older serving stacks.

transformers stores some MoE experts **fused** as contiguous GLU halves: one
``…experts.gate_up_proj`` ``[E, 2I, H]`` (``cat([gate, up], dim=1)``, gate first) and one
``…experts.down_proj`` ``[E, H, I]`` per layer. Stock ``from_pretrained`` and SGLang read this
natively, but some engine model-loaders (vLLM's ``glm4_moe_lite``) expect the legacy per-expert
layout ``…experts.{i}.{gate_proj,up_proj,down_proj}.weight``. This script rewrites a gathered
checkpoint to the per-expert layout so those engines can load it. Weights are unchanged: only the key
layout and the gate/up split, with no transpose and no re-quantization.

The projection names are the checkpoint family's own, read from its spec
(``MoEFamilySpec.expert_layout.per_expert_keys``: ``gate_proj``/``up_proj``/``down_proj`` or
``w1``/``w3``/``w2`` depending on the family). A ``model_type`` whose spec declares no per-expert
hub layout is refused, since every name emitted for it would be one nothing reads: a family serving
from the fused halves, GptOss's interleaved ``[E, H, 2M]`` ``gate_up_proj`` (already the layout
serving engines expect), or Inkling's interleaved TM ``w13_weight``. The refusal message names the
supported ``model_type`` set.

Usage:
    python scripts/after_training/unfuse_moe_experts.py --input_dir <ckpt> --output_dir <dst> [--max_shard_size 5GB]

A checkpoint with no fused expert keys is reported and copied through unchanged.
"""

import argparse

from halo.checkpoint.expert_unfuse import unfuse_checkpoint
from halo.log import configure_cli_logging
from scripts.cli_arguments import add_max_shard_size_arg


def main() -> None:
    configure_cli_logging()
    parser = argparse.ArgumentParser(description="Un-fuse contiguous fused-GLU MoE experts to per-expert layout.")
    parser.add_argument("--input_dir", required=True, help="Input checkpoint directory (fused expert layout).")
    parser.add_argument("--output_dir", required=True, help="Output directory (per-expert layout).")
    add_max_shard_size_arg(parser)
    args = parser.parse_args()
    unfuse_checkpoint(args.input_dir, args.output_dir, max_shard_size=args.max_shard_size)


if __name__ == "__main__":
    main()
