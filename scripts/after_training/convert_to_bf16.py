"""Convert a checkpoint to BF16: a full model with its normalization layers in fp32, or (``--peft``) a PEFT adapter's
weights file, without loading its base.

``--verify`` asserts the saved safetensors headers are primarily bfloat16; ``--check_inference`` runs a short pass. A
full-model save writes ``--output_dir`` fresh: every ``model*.safetensors`` or index the completed save did not produce
is removed afterwards."""

import argparse

from halo.checkpoint.tools.bf16_conversion import convert_to_bf16
from halo.log import configure_cli_logging
from scripts.cli_arguments import add_device_map_arg, add_max_shard_size_arg, add_task_arg, add_trust_remote_code_arg


def parse_args():
    parser = argparse.ArgumentParser(description="Convert model to BF16 format")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Local source directory; a Hub repo id is refused, so download it first",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help=(
            "Output directory for the BF16 model. A full-model save removes every model*.safetensors/index "
            "the completed save did not produce; a --peft save writes the adapter and its directory's other "
            "files and removes nothing"
        ),
    )
    add_task_arg(parser)
    parser.add_argument(
        "--peft",
        action="store_true",
        help="The source is a PEFT adapter: its weights are rewritten at bf16 without loading the base, and it "
        "stays an adapter (merge one with merge_peft_adapters.py)",
    )
    add_device_map_arg(parser)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Assert the saved checkpoint, or the --peft adapter, is stored in BF16 (reads the safetensors "
        "headers; raises on failure).",
    )
    parser.add_argument(
        "--check_inference",
        action="store_true",
        help="Run a test inference on the converted model to verify functionality",
    )
    add_max_shard_size_arg(parser, note="A --peft save writes no model weights and ignores it.")
    add_trust_remote_code_arg(parser, default=None)
    return parser.parse_args()


def main():
    configure_cli_logging()
    args = parse_args()
    convert_to_bf16(
        args.input_dir,
        args.output_dir,
        args.task,
        is_peft=args.peft,
        device_map=args.device_map,
        verify=args.verify,
        check_inference=args.check_inference,
        trust_remote_code=args.trust_remote_code,
        max_shard_size=args.max_shard_size,
    )


if __name__ == "__main__":
    main()
