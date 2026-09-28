"""Argparse flags shared across the ``scripts/`` subtrees.

The shard cap, the Hub-capable source block, the dtype and device-map pair and the remote-code
switch are defined once so the checkpoint tools chained over a single artifact (``after_training/``,
``before_training/``) accept the same spelling and defaults. The reward-model scorers
(``inference/reward_model/``) take the remote-code switch and the dtype flag, the latter as
``--rm_dtype`` (their device is their own ``--rm_device``), and ``dataset_deduplication.py`` the
remote-code switch. The OpenAI-compatible endpoint block (``--base_url``, ``--api_key``) is defined
once for the same reason across the generation, eval and playground CLIs (``inference/``,
``environments/``), which all drive one served model; it adds a required ``--model`` for every CLI
but the two playgrounds. Flags only; the drivers they feed live in ``src/``.
"""

import argparse

from src.checkpoint.format import DEFAULT_MAX_SHARD_SIZE
from src.inference.openai_client import DEFAULT_LOCAL_BASE_URL, resolve_local_api_key
from src.models.loading.dtype import DTYPE_BY_NAME

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


def add_dtype_arg(
    parser: argparse.ArgumentParser,
    *,
    flag: str = "--dtype",
    help: str = "Dtype of the output checkpoint (default: %(default)s).",
) -> argparse.ArgumentParser:
    """Add a dtype flag, as :data:`DTYPE_BY_NAME` names it: ``--dtype`` for a tool that writes a
    checkpoint, or ``flag``/``help`` for one whose dtype is another model's compute dtype."""
    parser.add_argument(flag, type=str, default="bfloat16", choices=list(DTYPE_BY_NAME), help=help)
    return parser


def add_device_map_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the ``--device_map`` flag for a tool that loads a whole model through ``from_pretrained``."""
    parser.add_argument(
        "--device_map",
        type=str,
        default=None,
        help="Device map for the model load, e.g. 'auto' or 'cpu' (default: none, the model loads on the CPU).",
    )
    return parser


def add_openai_endpoint_args(
    parser: argparse.ArgumentParser, *, model_help: str | None = None
) -> argparse.ArgumentParser:
    """Add the OpenAI-compatible endpoint a generation, eval or playground CLI drives: ``--base_url``,
    ``--api_key`` and, given ``model_help``, a required ``--model``.

    One spelling and one key-resolution policy, since these CLIs point at the same served model and a
    command line has to carry from one to the next. Without ``model_help`` no model flag is added,
    for a CLI where the name is optional (its own flag) or typed into its UI.
    """
    parser.add_argument(
        "--base_url",
        type=str,
        default=DEFAULT_LOCAL_BASE_URL,
        help="OpenAI-compatible base URL (default: %(default)s).",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=resolve_local_api_key(),
        help="API key for the endpoint (default: $VLLM_API_KEY, else $OPENAI_API_KEY, else the placeholder a "
        "keyless local server accepts; a hosted endpoint such as OpenRouter needs a real key).",
    )
    if model_help is not None:
        parser.add_argument("--model", type=str, required=True, help=model_help)
    return parser
