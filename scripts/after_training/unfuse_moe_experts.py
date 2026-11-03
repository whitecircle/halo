"""Un-fuse stacked MoE expert weights into per-expert tensors for older serving stacks.

Some engine loaders (vLLM's ``glm4_moe_lite``) expect ``…experts.{i}.{gate_proj,up_proj,down_proj}.weight`` over the
fused ``gate_up_proj`` ``[E, 2I, H]`` + ``down_proj`` ``[E, H, I]``. Weights are unchanged, only keys and the gate/up
split. Projection names come from ``MoEFamilySpec.expert_layout.per_expert_keys``; a ``model_type`` declaring no
per-expert hub layout is refused. A checkpoint with no fused expert keys is copied through unchanged."""

import argparse

from halo.checkpoint.tools.expert_unfuse import unfuse_checkpoint
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
