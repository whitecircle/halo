"""Argparse flags shared by the checkpoint-handling entry scripts.

The shard cap, the Hub-capable source block and the remote-code switch are defined once so the
tools chained over a single artifact (``after_training/``, ``before_training/``,
``inference/reward_model/``) accept the same spelling and defaults. Flags only; the drivers they
feed live in ``src/``.
"""

import argparse

from src.checkpoint.format import DEFAULT_MAX_SHARD_SIZE

# What ``--model_id`` accepts; kept in step with ``resolve_checkpoint_source``.
HUB_SOURCE_HELP = "Hub repo id or a local checkpoint directory"


def add_max_shard_size_arg(parser: argparse.ArgumentParser, *, note: str = "") -> argparse.ArgumentParser:
    """Add the ``--max_shard_size`` flag for a tool that writes a safetensors checkpoint.

    Defaults to :data:`~src.checkpoint.format.DEFAULT_MAX_SHARD_SIZE`. ``note`` appends a caveat for
    outputs that ignore the cap (adapter files, single-file checkpoints rewritten in place).
    """
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default=DEFAULT_MAX_SHARD_SIZE,
        help=f"Maximum size of one output safetensors shard, e.g. '2GB' (default: {DEFAULT_MAX_SHARD_SIZE})."
        + (f" {note}" if note else ""),
    )
    return parser


def add_hub_source_args(
    parser: argparse.ArgumentParser, *, source: str, default: str | None = None, revision: bool = True
) -> argparse.ArgumentParser:
    """Add the ``--model_id`` flag of a tool whose source may be a Hub repo, and its revision pin.

    ``source`` names what the tool reads there; ``default`` names the release a converter targets
    and makes the flag optional; ``revision=False`` suits a tool that threads no revision, which
    would otherwise advertise a pin it ignores.
    """
    parser.add_argument(
        "--model_id",
        type=str,
        default=default,
        required=default is None,
        help=f"{source} — {HUB_SOURCE_HELP}." + (f" Default: {default}." if default else ""),
    )
    if revision:
        parser.add_argument(
            "--revision",
            type=str,
            default=None,
            help="Hub revision to pin (ignored for a local source).",
        )
    return parser


def add_trust_remote_code_arg(parser: argparse.ArgumentParser, *, default: bool = True) -> argparse.ArgumentParser:
    """Add the ``--trust_remote_code`` flag for a tool that loads model code.

    The default follows the input source, not the tool:

    * a local checkpoint or adapter (``--input_dir`` / ``--adapter_dir``, or the tokenizer of the run
      being prepared) defaults on, since the remote-code families in the roster (Bailing/Ling,
      Laguna, sink-carrying gpt-oss derivatives) ship their modeling files inside it;
    * a source that may be a Hub repo (``--model_id``) defaults off, so a freshly downloaded
      third-party repo does not execute its own code.

    ``--trust_remote_code`` / ``--no-trust_remote_code`` overrides either way.
    """
    parser.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=default,
        help=f"Execute the checkpoint's own modeling/config code when loading it (default: {default} "
        f"— {'the remote-code families, e.g. Bailing/Ling, do not load without it' if default else 'this source may be a Hub repo'}). "
        f"Pass {'--no-trust_remote_code for a source you do not trust' if default else '--trust_remote_code for a remote-code family (Bailing/Ling, Laguna) you trust'}.",
    )
    return parser
