"""Flag surface the checkpoint-handling entry scripts share across every ``scripts/`` subtree.

The shard cap, the Hub-capable source block and the remote-code switch are spelled once here —
one spelling, one default, one description — because these tools are chained on a single artifact,
and a per-script copy is how two links end up capping their shards differently, or how the same
remote-code checkpoint loads through one tool and is refused by the next. They sit above the
subtrees that take them (``after_training/``, ``before_training/``, ``inference/reward_model/``),
as :mod:`scripts.inference._common` does over its own.

Flags only — every driver they feed stays in ``src/``.
"""

import argparse

from src.checkpoint.format import DEFAULT_MAX_SHARD_SIZE

# What a ``--model_id`` accepts, spelled once (:func:`add_hub_source_args`) so no tool's help
# promises less than ``resolve_checkpoint_source`` resolves.
HUB_SOURCE_HELP = "Hub repo id or a local checkpoint directory"


def add_max_shard_size_arg(parser: argparse.ArgumentParser, *, note: str = "") -> argparse.ArgumentParser:
    """Add the ``--max_shard_size`` flag every tool writing a safetensors checkpoint exposes.

    Default :data:`~src.checkpoint.format.DEFAULT_MAX_SHARD_SIZE`. ``note`` appends the caveat a particular
    output carries — a save that writes adapter files and ignores the cap, a single-file checkpoint
    rewritten as the one file it came in as.
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

    One spelling for a source ``resolve_checkpoint_source`` accepts in either form, so a tool cannot
    promise a narrower input than the resolver takes. ``source`` names what the tool reads there;
    ``default`` names the single release a converter targets and makes the flag optional;
    ``revision=False`` is for a tool that threads no revision — an unread flag would advertise a pin
    the tool ignores.
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
    """Add the ``--trust_remote_code`` flag every tool that loads model code exposes.

    One spelling, and a default that follows the tool's **input source** rather than the tool:

    * a local checkpoint or adapter (``--input_dir`` / ``--adapter_dir``, or the tokenizer of the run
      being prepared) defaults **on** — the operator already chose to train or export that artifact,
      and the remote-code families in the roster (Bailing/Ling, Laguna, sink-carrying gpt-oss
      derivatives) ship their modeling files inside it, so defaulting off makes the documented
      invocation fail on them;
    * a source that may be a Hub repo (``--model_id``) defaults **off** — a freshly downloaded
      third-party repo must not execute its own code because a tool was pointed at it.

    Either way the opposite is one flag away (``--trust_remote_code`` /
    ``--no-trust_remote_code``); no tool hardcodes the decision.
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
