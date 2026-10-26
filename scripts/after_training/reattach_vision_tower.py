"""Re-attach a multimodal base's vision tower to a text-only export, restoring the wrapper layout.

A ``text_only_model`` run trains and exports the CausalLM sibling of a VLM checkpoint (``model.*``
keys, ``*_text`` config, no vision tower, no MTP tail). Both pinned engines register the text CausalLM
classes for Qwen3.5/3.6 and load that export as-is, text only. This script rebuilds the wrapper-layout
artifact: the trained text weights are re-prefixed to ``model.language_model.*``, the untrained vision
tower (``model.visual.*``) and any MTP tail (``mtp.*``) stream back in from the base checkpoint, and
the composite config is regrafted with the trained text config, so the served model runs the trained
text weights (router balancing included) under the multimodal class.

The input must be a text-only export (its ``model_type`` names the ``*_text`` sub-config) whose keys
are in that layout; a wrapper-layout checkpoint, or text keys already under the wrapper prefix, is
refused. The base must be the multimodal checkpoint the run trained from (same architecture); a base
with no vision keys is refused.

Usage:
    python scripts/after_training/reattach_vision_tower.py \\
        --input_dir <text-only export> --model_id <multimodal base repo or dir> --output_dir <dst>
"""

import argparse

from halo.checkpoint.vision_reattach import reattach_vision_tower
from halo.log import configure_cli_logging
from scripts.cli_arguments import add_hub_source_args, add_max_shard_size_arg, add_trust_remote_code_arg


def main() -> None:
    configure_cli_logging()
    parser = argparse.ArgumentParser(description="Re-attach a multimodal base's vision tower to a text-only export.")
    parser.add_argument("--input_dir", required=True, help="Text-only export directory.")
    add_hub_source_args(parser, source="The multimodal base checkpoint the run trained from")
    parser.add_argument("--output_dir", required=True, help="Output directory (wrapper layout).")
    add_max_shard_size_arg(parser)
    # Both configs load through it, and --model_id may be a Hub repo, so it is opt-in as on every
    # sibling whose source can be one.
    add_trust_remote_code_arg(parser, default=False)
    args = parser.parse_args()
    reattach_vision_tower(
        args.input_dir,
        args.model_id,
        args.output_dir,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
        max_shard_size=args.max_shard_size,
    )


if __name__ == "__main__":
    main()
