#!/usr/bin/env python
"""Offline SFT dataset preparation: tokenize, optionally pack and shard, then publish, so training skips tokenization
at startup. The flags are :class:`PreprocessingConfig`'s plus :class:`PrepareDatasetArguments`'s; a YAML file of them
works as first argument."""

from halo.configs.parser import HaloArgumentParser
from halo.data.preprocessed.metadata import PreprocessingConfig
from halo.data.preprocessed.prepare_dataset_tool import PrepareDatasetArguments, bake_num_proc, run_dataset_preparation
from halo.log import configure_cli_logging
from halo.model_loading.tokenizer_setup import load_chat_template
from scripts.cli_arguments import add_trust_remote_code_arg


def parse_args() -> tuple[PrepareDatasetArguments, PreprocessingConfig]:
    """Completion masking follows the mode as training does (on for chat, refused for text), and the worker count and
    chat template are recorded resolved. ``--model_name_or_path`` may be a Hub repo, so its code stays opt-in."""
    parser = add_trust_remote_code_arg(
        HaloArgumentParser((PrepareDatasetArguments, PreprocessingConfig)),
        default=PreprocessingConfig.trust_remote_code,
    )
    parser.set_defaults(train_on_completions_only=None)
    values = parser.load_values()
    values.setdefault("trust_remote_code", parser.get_default("trust_remote_code"))
    values.setdefault("train_on_completions_only", values.get("mode", PreprocessingConfig.mode) == "chat")
    values["num_proc"] = values.get("num_proc") or bake_num_proc()
    if values.get("chat_template"):
        values["chat_template"] = load_chat_template(values["chat_template"])
    return parser.parse_dict(values)


def main():
    args, config = parse_args()
    configure_cli_logging(verbose=args.verbose)
    run_dataset_preparation(args, config)


if __name__ == "__main__":
    main()
