#!/usr/bin/env python
"""Write a saved LoRA adapter as the adapter directory a rollout engine loads by path, as adapter sync publishes it.

A training save keys each module by the live tree and holds Halo's grouped expert adapters, where the engine reads the
base checkpoint's names and its own expert layout. The base is built from its config alone, so no weight is read. What
the engine cannot serve is refused before any write, with the reason."""

import argparse

from halo.configs.grpo_environmental import AsyncRolloutConfig
from halo.entrypoints.engine_adapter_export import export_engine_adapter
from halo.log import configure_cli_logging
from halo.weight_sync.registry import rollout_backends
from scripts.cli_arguments import add_base_revision_arg, add_trust_remote_code_arg


def main():
    configure_cli_logging()
    parser = argparse.ArgumentParser(
        description="Write a saved LoRA adapter as the directory a rollout engine loads",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example (Inkling 2 attention or attention+expert adapters on the vLLM Inkling 2 tag):
    python scripts/after_training/export_engine_adapter.py \\
        --adapter_dir checkpoints/inkling2-sft-lora/checkpoint-500 \\
        --output_dir adapters/inkling2-sft-500

    curl -X POST http://localhost:8000/v1/load_lora_adapter -H 'Content-Type: application/json' \\
        -d '{"lora_name": "sft-500", "lora_path": "/abs/path/adapters/inkling2-sft-500"}'
        """,
    )
    parser.add_argument(
        "--adapter_dir",
        required=True,
        help="A saved adapter directory (adapter_config.json and adapter_model.safetensors), attention or "
        "attention+expert",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="The engine adapter directory to write; it must not exist. The server reads it by this path",
    )
    parser.add_argument(
        "--rollout_backend",
        choices=rollout_backends(),
        default=AsyncRolloutConfig.rollout_backend,
        help="The engine that will serve the adapter (default: %(default)s)",
    )
    add_base_revision_arg(parser)
    add_trust_remote_code_arg(parser, default=None)
    args = parser.parse_args()
    export_engine_adapter(
        args.adapter_dir,
        args.output_dir,
        rollout_backend=args.rollout_backend,
        base_revision=args.base_revision,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
