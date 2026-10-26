"""The flag surface of the reward-model scorers (``score`` / ``rejection_sample``): the endpoint,
prompt-row, generation and reward-model blocks on one parser; the runs live in ``halo/entrypoints/``."""

import argparse

from scripts.cli_arguments import HUB_SOURCE_HELP, add_dtype_arg, add_openai_endpoint_args, add_trust_remote_code_arg
from scripts.inference.inference_arguments import add_generation_args, add_prompt_field_args


def build_generation_parser(description: str, *, temperature_default: float) -> argparse.ArgumentParser:
    """The CLI parser shared by the RM inference scripts: API, data, generation, and RM arguments.

    Scripts add their own extras (``--n_hypos``, ``--output_format``, ``--rm_max_batch_size``, …)
    on the returned parser.
    """
    parser = argparse.ArgumentParser(description=description)
    add_openai_endpoint_args(parser, model_help="Model name for API requests")

    # Data configuration
    parser.add_argument("--dataset", type=str, required=True, help="Local JSONL file with the prompts.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Directory the run's JSONL output (named from the prompts file, the model and the run) is appended "
        "to and resumed from (default: %(default)s).",
    )
    add_prompt_field_args(parser)
    parser.add_argument(
        "--follow_up_prompt_field",
        type=str,
        default="follow_up_prompt",
        help="Row field with follow-up turns sent after the first answer (default: %(default)s).",
    )
    parser.add_argument(
        "--correct_answer_field",
        type=str,
        default="correct_answer",
        help="Row field with the ground-truth answer shown to the reward model (default: %(default)s).",
    )

    # Generation configuration
    add_generation_args(parser, temperature_default=temperature_default)

    # Reward model configuration
    parser.add_argument(
        "--rm_model_id", type=str, required=True, help=f"The sequence-classification reward model — {HUB_SOURCE_HELP}."
    )
    # --rm_model_id may be a Hub repo, so remote code stays opt-in.
    add_trust_remote_code_arg(parser, default=False)
    parser.add_argument(
        "--rm_attn_implementation",
        type=str,
        default="sdpa",
        help="Attention implementation the reward model loads with (default: %(default)s).",
    )
    parser.add_argument(
        "--rm_max_seq_len",
        type=int,
        default=16000,
        help="Longest conversation (tokens) the reward model scores; a longer one is dropped and counted, "
        "never truncated (default: %(default)s)",
    )
    parser.add_argument(
        "--rm_device", type=str, default="cuda:0", help="Device the reward model scores on (default: %(default)s)."
    )
    add_dtype_arg(
        parser,
        flag="--rm_dtype",
        help="Reward-model compute dtype (default: %(default)s, the toolkit-wide default — float16's "
        "narrow range overflows on out-of-distribution reward logits)",
    )

    return parser
