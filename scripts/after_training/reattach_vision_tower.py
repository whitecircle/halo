"""Re-attach a multimodal base's vision tower to a text-only export, restoring the wrapper layout.

Trained text weights are re-prefixed to ``model.language_model.*``, or the family's declared ``text_only_layout``, the
vision tower and any MTP tail stream back in from the base (its head untrained against this trunk, which warns; a head
the export trained supersedes it), and the composite config is regrafted. The input must be a text-only export and the
base the multimodal checkpoint the run trained from; anything else is refused. Both pinned engines also load the export
as-is."""

import argparse

from halo.checkpoint.tools.vision_reattach import reattach_vision_tower
from halo.log import configure_cli_logging
from scripts.cli_arguments import add_hub_source_args, add_max_shard_size_arg, add_trust_remote_code_arg


def main() -> None:
    configure_cli_logging()
    parser = argparse.ArgumentParser(description="Re-attach a multimodal base's vision tower to a text-only export.")
    parser.add_argument("--input_dir", required=True, help="Text-only export directory.")
    add_hub_source_args(parser, source="The multimodal base checkpoint the run trained from")
    parser.add_argument("--output_dir", required=True, help="Output directory (wrapper layout).")
    add_max_shard_size_arg(parser)
    # --model_id may be a Hub repo, so opt-in as on every sibling whose source can be one.
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
