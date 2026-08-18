"""Add tokens to a model's vocabulary and resize its embeddings, saving a patched checkpoint.

New token embeddings are initialized from the mean of the existing ones. ``--patterns`` takes a
JSON list (inline or a file path) of the strings to add. ``--reset_sinks``
additionally fills every attention sink with its dtype minimum, disabling the sink mechanism (same
transform as ``scripts/after_training/reset_sinks.py``, applied before the save).

Only patterns that currently tokenize to more than one token are added: a pattern already covered by
a single token would gain nothing and would shift every id above it.

Growth only: the rows above ``len(tokenizer)`` are unassigned padding and every special token
(harmony EOS included) sits below it, so a shrink recovers only padding, while any target under
``len(tokenizer)`` cuts into the specials and the model can no longer stop generating.

``--output_dir`` is written fresh: after a completed save, every ``model*.safetensors`` and index
the save did not produce is removed.

Usage:
    python scripts/before_training/patch_vocab.py \\
        --model_id "unsloth/gpt-oss-120b-BF16" \\
        --output_dir "./models/gpt-oss-120b-BF16-patched" \\
        --patterns '["<tool_call>", "<tool_result>"]' --reset_sinks
"""

import argparse
import json
import logging

import torch
from accelerate import PartialState
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

import src.distributed.expert_parallel.layers.roster  # noqa: F401 — registers the EP export roster the config finalizer requires
from scripts._common import add_hub_source_args, add_max_shard_size_arg, add_trust_remote_code_arg
from src.checkpoint.tool_io import (
    preflight_model_load_resources,
    reject_in_place_conversion,
    reject_sharded_checkpoint,
    save_full_checkpoint,
)
from src.log import configure_cli_logging
from src.models.loading.dtype import DTYPE_BY_NAME
from src.models.loading.model_preparation import auto_load_model
from src.models.loading.tokenizer_setup import load_processing_class
from src.models.patches.gpt_oss_sinks import SinksPolicy, apply_sinks_policy, stamped_sinks_policy
from src.models.structure import resolve_tokenizer

configure_cli_logging()
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Patch vocabulary of a model by adding new tokens")
    # No --revision: the vocabulary patch threads none, and an unread flag would advertise a pin it
    # ignores. Pin by downloading the source first, then pointing --model_id at it.
    add_hub_source_args(
        parser, source="The model whose vocabulary is patched (e.g. 'unsloth/gpt-oss-120b-BF16')", revision=False
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help=(
            "Directory to save the patched model and tokenizer "
            "(any model*.safetensors/index already there is removed first)"
        ),
    )
    parser.add_argument(
        "--patterns",
        type=str,
        default=None,
        help="JSON string or path to JSON file containing list of patterns to add as tokens",
    )
    parser.add_argument(
        "--chat_template",
        type=str,
        default=None,
        help="Optional: Path to a file containing a custom chat template to set on the tokenizer",
    )
    parser.add_argument(
        "--reset_sinks",
        action="store_true",
        help="Reset all attention sinks to minimum value before saving",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=list(DTYPE_BY_NAME),
        help="Torch dtype to load the model with (default: bfloat16)",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default=None,
        help="Device map for model loading (e.g., 'auto', 'cpu'). If not set, loads on CPU.",
    )
    add_max_shard_size_arg(parser)
    # --model_id is usually a Hub repo here (this is the tool that patches a freshly downloaded
    # third-party checkpoint), so remote code stays opt-in.
    add_trust_remote_code_arg(parser, default=False)
    return parser.parse_args()


def load_patterns(patterns_arg: str | None) -> list[str]:
    """Load patterns from a JSON string on the command line or a JSON file path."""
    if patterns_arg:
        try:
            patterns = json.loads(patterns_arg)
            logger.info(f"Loaded {len(patterns)} patterns from command line")
            return patterns
        except json.JSONDecodeError:
            with open(patterns_arg) as f:
                patterns = json.load(f)
            logger.info(f"Loaded {len(patterns)} patterns from file: {patterns_arg}")
            return patterns

    return []


def filter_patterns_needing_multiple_tokens(
    tokenizer: PreTrainedTokenizer, patterns: list[str]
) -> list[tuple[str, int]]:
    """Filter patterns that currently use more than 1 token."""
    patterns_to_add = []
    for pattern in patterns:
        tokens = tokenizer.encode(pattern, add_special_tokens=False)
        if len(tokens) > 1:
            patterns_to_add.append((pattern, len(tokens)))
    return patterns_to_add


def add_tokens_to_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    patterns: list[str],
) -> int:
    """
    Add new tokens to tokenizer and resize model embeddings.

    Returns the number of tokens added.
    """
    original_vocab_size = len(tokenizer)
    logger.info(f"Original vocabulary size: {original_vocab_size}")

    patterns_to_add = filter_patterns_needing_multiple_tokens(tokenizer, patterns)

    if not patterns_to_add:
        logger.info("No patterns need to be added (all already single tokens)")
        return 0

    logger.info(f"Found {len(patterns_to_add)} patterns that use >1 token:")
    for i, (text, num_tokens) in enumerate(patterns_to_add, 1):
        logger.info(f"{i:2d}. '{text}' - {num_tokens} tokens -> 1 token")

    new_tokens = [text for text, _ in patterns_to_add]
    num_added = tokenizer.add_tokens(new_tokens)
    logger.info(f"Added {num_added} new tokens to tokenizer")

    new_vocab_size = len(tokenizer)
    logger.info(f"New vocabulary size: {new_vocab_size}")
    logger.info(f"Difference: +{new_vocab_size - original_vocab_size}")

    logger.info("Calculating average embeddings for initialization...")
    with torch.no_grad():
        input_embeddings = model.get_input_embeddings().weight
        avg_input_embedding = input_embeddings[:original_vocab_size].mean(dim=0)

        output_embeddings = model.get_output_embeddings().weight
        avg_output_embedding = output_embeddings[:original_vocab_size].mean(dim=0)

    logger.info(f"Model embeddings shape before: {model.get_input_embeddings().weight.shape}")

    # Grow only. Models like gpt-oss ship an embedding padded larger than len(tokenizer): the rows
    # above len(tokenizer) are unassigned and every special (harmony EOS, <|call|>) sits below it, so a
    # shrink recovers only padding and any target under len(tokenizer) cuts the specials.
    current_embedding_size = model.get_input_embeddings().weight.shape[0]
    if new_vocab_size > current_embedding_size:
        model.resize_token_embeddings(new_vocab_size)
    else:
        logger.info(
            f"Keeping existing embedding size {current_embedding_size} (>= new vocab {new_vocab_size}); "
            "added tokens reuse existing padding rows — not shrinking (would drop high special tokens)."
        )

    logger.info(f"Model embeddings shape after resize: {model.get_input_embeddings().weight.shape}")

    with torch.no_grad():
        for i in range(original_vocab_size, new_vocab_size):
            model.get_input_embeddings().weight[i] = avg_input_embedding
            model.get_output_embeddings().weight[i] = avg_output_embedding

    logger.info(f"Successfully initialized {new_vocab_size - original_vocab_size} new token embeddings!")

    logger.info("Verification: Testing new tokens in tokenizer")
    for i, (text, _old_num_tokens) in enumerate(patterns_to_add[:5], 1):
        new_tokens_enc = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(new_tokens_enc)
        match = "✓" if decoded == text else "✗"
        logger.info(f"{i}. '{text}' -> {len(new_tokens_enc)} token(s), ID: {new_tokens_enc}, Match: {match}")

    return num_added


def main():
    args = parse_args()
    # Refused up front: an in-place patch has no undo, and a per-rank EP/TP-sharded source would load
    # with randomly initialized experts (missing keys only warn) while the patched save looked
    # complete.
    reject_in_place_conversion(args.model_id, args.output_dir)
    reject_sharded_checkpoint(args.model_id)
    # Without --device_map the whole model lands in host RAM, and the patched output is about the
    # source's size on disk (not reported for a Hub id, where there is nothing local to measure).
    preflight_model_load_resources(args.model_id, args.output_dir, tool="patch_vocab", device_map=args.device_map)

    # The shared model_preparation utilities use accelerate's rank-aware logger.
    PartialState()

    torch_dtype = DTYPE_BY_NAME[args.torch_dtype]

    logger.info("PATCH VOCABULARY SCRIPT")
    logger.info(f"Model: {args.model_id}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Dtype: {args.torch_dtype}")
    logger.info(f"Reset sinks: {args.reset_sinks}")

    patterns = load_patterns(args.patterns)
    if patterns:
        logger.info(f"Patterns to process: {len(patterns)}")

    # Full processor for VLMs, whose processor config and chat template have to survive the save, else
    # a plain tokenizer. New tokens go into this object's tokenizer, so the patched VLM reloads as
    # multimodal.
    logger.info("Loading tokenizer/processor...")
    processing_class = load_processing_class(args.model_id, trust_remote_code=args.trust_remote_code)
    if processing_class is None:
        processing_class = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    tokenizer = resolve_tokenizer(processing_class)

    if args.chat_template:
        logger.info(f"Loading custom chat template from: {args.chat_template}")
        with open(args.chat_template) as f:
            chat_template = f.read()
        tokenizer.chat_template = chat_template

    # auto_load_model resolves the widest Auto* class (AutoModelForCausalLM would drop a vision
    # tower) and routes through the checkpoint-coverage gate: a truncated or key-mismatched source
    # would otherwise load with randomly initialized tensors and be saved as a complete-looking
    # patch.
    logger.info("Loading model...")
    model_kwargs = {"dtype": torch_dtype}
    if args.device_map:
        model_kwargs["device_map"] = args.device_map

    model = auto_load_model(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
        **model_kwargs,
    )
    logger.info(f"Loaded model class: {type(model).__name__} (model_type={model.config.model_type})")

    if patterns:
        logger.info("ADDING NEW TOKENS")
        add_tokens_to_model(model, tokenizer, patterns)

    if args.reset_sinks:
        logger.info("RESETTING ATTENTION SINKS")
        # The trainers' own seam rather than a second walk: it neutralizes what every trainer
        # neutralizes and stamps the sink policy the provenance record and the merge tools read, so a
        # patched checkpoint and a trained one agree on what their sinks mean.
        apply_sinks_policy(model, model.config, policy=SinksPolicy.NEUTRALIZED)
        # It returns without error on a family that carries no sinks, and the stamp is the only
        # signal it did anything: unchecked, the flag would save an untouched checkpoint that
        # downstream tools (provenance, the merge tools, the RL sink gate) read as sink-free.
        if stamped_sinks_policy(model) is None:
            raise ValueError(
                f"--reset_sinks was requested, but {args.model_id} (model_type="
                f"{model.config.model_type!r}) carries no attention sinks — nothing was reset and the "
                f"saved checkpoint would be identical. Drop --reset_sinks; it applies to sink-carrying "
                f"families (gpt-oss)."
            )

    logger.info("SAVING MODEL AND TOKENIZER")
    logger.info(f"Saving to: {args.output_dir}")

    # _tied_weights_keys belongs to remote_code_compat (apply_remote_code_compat_shims, run by the
    # auto_load_model above): a per-instance {k: k} here would declare a self-tie on these untied
    # checkpoints, and save_pretrained would drop the head.

    # Save, restore model_type, then sweep; the processing class carries the patched tokenizer.
    save_full_checkpoint(
        model,
        args.output_dir,
        processing_class=processing_class,
        source_dir=args.model_id,
        max_shard_size=args.max_shard_size,
    )

    logger.info("DONE!")


if __name__ == "__main__":
    main()
