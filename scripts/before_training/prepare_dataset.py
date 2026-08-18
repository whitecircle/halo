#!/usr/bin/env python
"""Offline SFT dataset preparation: tokenize, optionally pack, optionally shard, then publish.

Reads from S3 / HuggingFace Hub / a local path and writes back to S3 or the Hub, so training skips
tokenization at startup (``sft.py`` loads the result directly). ``--vlm`` runs the processor path
(pixel_values + image_grid_thw); packing is text-only. Other methods (DPO, SMPO, GRPO) use different
data formats and are not handled here.

Usage:
    python scripts/before_training/prepare_dataset.py \\
        --input "s3://bucket/raw/my_dataset" --output "s3://bucket/preprocessed/my_dataset" \\
        --model-name "Qwen/Qwen3-8B" --max-length 8192 [--pack-sequences] [--num-shards 64] [--vlm]
"""

import argparse
import contextlib
import os
import shutil
import sys
import tempfile

from accelerate import PartialState
from accelerate.logging import get_logger
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, upload_folder
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizer

from scripts._common import add_trust_remote_code_arg
from src.checkpoint.tool_io import DISPLACED_SUFFIX, clear_staging_path
from src.data.pipeline.preprocessed_metadata import PreprocessingConfig
from src.data.pipeline.preprocessing import preprocess_dataset
from src.data.pipeline.processing import resolve_map_num_proc
from src.data.pipeline.tokenizer_backend import TOKENIZER_BACKENDS
from src.data.sources.loading import load_dataset_from_source
from src.data.sources.paths import METADATA_FILE, parse_dataset_destination, parse_s3_uri
from src.data.sources.s3_client import S3Client
from src.log import configure_cli_logging
from src.models.loading.tokenizer_setup import load_chat_template

# The rank-aware accelerate logger, at INFO: the shared data helpers this tool drives log through
# it, and ``configure_cli_logging`` in main() is what lets those records reach a handler.
logger = get_logger(__name__, log_level="INFO")

# Fixed so --test-size cuts the same split on every re-run: the labels are baked into the published
# artifact, and a re-preparation that reshuffled would leak train rows into a previous run's eval.
_SPLIT_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess dataset for efficient training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input dataset path (S3 URI, HF Hub ID, or local path)",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output path (S3 URI 's3://...', HF Hub 'hf://org/name', or local path)",
    )

    parser.add_argument(
        "--model-name",
        "-m",
        required=True,
        help="Model name or path for tokenizer",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=8192,
        help="Maximum tokenized sequence length; over-length conversations are dropped, not "
        "truncated (default: 8192). Stamped into the dataset metadata and re-checked against the "
        "training config's max_length at load. No model is loaded here, so it cannot resolve to a "
        "context window — pass it explicitly.",
    )
    add_trust_remote_code_arg(parser)

    parser.add_argument(
        "--mode",
        default="chat",
        choices=["chat", "text"],
        help="Tokenization mode: 'chat' (apply chat template to the conversation "
        "field, for SFT) or 'text' (raw-text causal-LM for (continued) "
        "pretraining — tokenize --text-field directly, append EOS per document). "
        "Default: chat.",
    )
    parser.add_argument(
        "--text-field",
        default="text",
        help="(mode=text) dataset column holding the raw text (default: text).",
    )
    parser.add_argument(
        "--no-append-eos",
        action="store_true",
        help="(mode=text) do NOT append EOS to each document before packing "
        "(by default an EOS is appended to preserve document boundaries).",
    )
    parser.add_argument(
        "--conversation-field",
        default="prompt",
        help="Name of conversation field in dataset (default: prompt — the documented SFT shape and "
        "the training-side default; a value disagreeing with the training config is rejected at "
        "training startup)",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="System prompt to prepend to conversations",
    )
    parser.add_argument(
        "--tools-field",
        default=None,
        help="Dataset field holding a tool list (list of dicts or JSON string) forwarded into "
        "apply_chat_template — must match the tools_field used at training time",
    )
    parser.add_argument(
        "--interleaved-thinking",
        action="store_true",
        help="Keep thinking blocks in non-final assistant turns (clear_thinking=False), matching the "
        "live training path's interleaved_thinking flag. Only meaningful for templates with a "
        "clear_thinking switch (GLM family); rejected with --vlm",
    )
    parser.add_argument(
        "--no-system-role",
        action="store_true",
        help="Model doesn't support system role (merge with first user message)",
    )

    parser.add_argument(
        "--train-on-completions-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Mask user turns, training only on assistant responses. Unset resolves per --mode: on "
        "for chat (matching the training-side default — the labels are baked here and training "
        "refuses a dataset whose masking disagrees with its config), off for text, which renders no "
        "chat template and rejects the setting outright. On requires --assistant-message-template; "
        "pass --no-train-on-completions-only to bake full-sequence labels instead",
    )
    parser.add_argument(
        "--assistant-message-template",
        default=None,
        help="Template marking start of assistant response (e.g., '<|im_start|>assistant\\n')",
    )

    parser.add_argument(
        "--pack-sequences",
        action="store_true",
        help="Pack sequences together for efficient training (text SFT only, not VLM)",
    )
    parser.add_argument(
        "--packing-strategy",
        default="bfd",
        choices=["bfd", "bfd_split", "wrapped"],
        help="TRL packing strategy: 'bfd' (best-fit-decreasing, keeps document boundaries but "
        "DISCARDS the overflow past --max-length), 'bfd_split' (same packing, overflow split into "
        "later examples — the lossless choice for pre-training), or 'wrapped' (concatenate-and-chunk "
        "across boundaries). Default: bfd.",
    )

    parser.add_argument(
        "--vlm",
        action="store_true",
        help="Enable VLM (Vision-Language Model) preprocessing mode",
    )
    parser.add_argument(
        "--images-field",
        default=None,
        help="(--vlm) dataset column holding the row's image(s), for datasets that keep them OUTSIDE "
        "the conversation (the hub VLM shape: FineVision / the_cauldron / Docmatix). Merged into the "
        "message content exactly as the runtime VLM path does — filling the conversation's image "
        "placeholders in order, else prepended to the first user turn. Omit for datasets that embed "
        "their images in the conversation; an image column named by nothing is refused",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=None,
        help="Minimum pixels for VLM image processing",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help="Maximum pixels for VLM image processing",
    )

    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Number of shards to create (default: 1 = no sharding)",
    )

    parser.add_argument(
        "--num-proc",
        type=int,
        default=None,
        help="Processes for dataset processing (default: HALO_DATASET_NUM_PROC).",
    )
    parser.add_argument(
        "--tokenizer-backend",
        default="hf",
        choices=list(TOKENIZER_BACKENDS),
        help="Text→ids backend: 'hf' (default) or 'gigatoken' (optional extra; "
        "token IDs verified identical at startup).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=None,
        help="Create test split with this fraction (e.g., 0.01)",
    )

    # Recorded into metadata.json: each of these changes the ids that get baked, so the training-side
    # compatibility check has to see them.
    parser.add_argument(
        "--pad-token",
        default=None,
        help="Override pad token (recorded in metadata.json and re-checked at training time)",
    )
    parser.add_argument(
        "--eos-token",
        default=None,
        help="Override EOS token — moves the completion-mask boundaries, so it is recorded in "
        "metadata.json and re-checked at training time",
    )
    parser.add_argument(
        "--bos-token",
        default=None,
        help="Override BOS token (recorded in metadata.json and re-checked at training time)",
    )
    parser.add_argument(
        "--chat-template",
        default=None,
        help="Override chat template: a Jinja2 string, or a path to a .jinja/.jinja2/.j2 file. The "
        "RESOLVED template text is recorded in metadata.json and re-checked at training time",
    )

    parser.add_argument(
        "--private",
        action="store_true",
        help="Make HuggingFace Hub dataset private",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token for private repos",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without processing",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()
    args.train_on_completions_only = resolve_train_on_completions_only(args)
    return args


def resolve_train_on_completions_only(args) -> bool:
    """Whether to bake completion-only labels, resolving an untouched flag against ``--mode``.

    Chat mode masks by default, matching the training side (which refuses a dataset whose baked
    masking disagrees with its config). Text mode does not: it renders no chat template, so
    ``PreprocessingConfig`` rejects the setting as inapplicable — a flat default-on would make every
    documented raw-text pre-training invocation exit on a flag its user never passed. An explicit
    flag is honored as given, so asking for masking in text mode still fails loud.
    """
    if args.train_on_completions_only is not None:
        return args.train_on_completions_only
    return args.mode == "chat"


def reject_unbuildable_masking(args) -> None:
    """Refuse completion-only masking with no response marker, before anything is loaded.

    ``PreprocessingConfig`` enforces the same pair, but only once the input dataset is already in
    hand — minutes of S3 download to report a two-flag mistake. The labels are baked into the
    artifact here, so there is no later stage that could supply the marker instead.
    """
    if args.train_on_completions_only and not args.assistant_message_template:
        raise ValueError(
            "--train-on-completions-only (on by default in --mode chat, matching the training-side "
            "default) requires --assistant-message-template: the response marker your model's chat "
            "template renders, e.g. '<|im_start|>assistant\\n'. The completion-only labels are baked "
            "into the dataset here, so the marker cannot be supplied later. Pass it, or pass "
            "--no-train-on-completions-only to bake full-sequence labels instead."
        )


def apply_tokenizer_overrides(tokenizer, args) -> None:
    """Apply the CLI's pad/eos/bos/chat-template overrides, then guarantee a pad token.

    Shared by the plain-tokenizer and VLM-processor paths so a tokenized dataset carries the same
    special tokens either way — a pad/eos mismatch silently changes what the collator masks. The
    same overrides are recorded in the config (:func:`build_preprocessing_config`), which is what
    lets training verify the run's tokenizer matches the one that baked the rows.
    """
    if args.pad_token:
        tokenizer.pad_token = args.pad_token
    if args.eos_token:
        tokenizer.eos_token = args.eos_token
    if args.bos_token:
        tokenizer.bos_token = args.bos_token
    if args.chat_template:
        tokenizer.chat_template = load_chat_template(args.chat_template)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def setup_tokenizer(args) -> PreTrainedTokenizer:
    """Load and configure tokenizer."""
    logger.info(f"Loading tokenizer from {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )

    apply_tokenizer_overrides(tokenizer, args)

    logger.info(f"Tokenizer vocab size: {len(tokenizer)}")
    logger.info(f"Pad token: {tokenizer.pad_token}")
    logger.info(f"EOS token: {tokenizer.eos_token}")

    return tokenizer


def setup_vlm_processor(args):
    """Load and configure VLM processor."""
    logger.info(f"Loading VLM processor from {args.model_name}")

    processor_kwargs = {"trust_remote_code": args.trust_remote_code}

    if args.min_pixels is not None:
        processor_kwargs["min_pixels"] = args.min_pixels
    if args.max_pixels is not None:
        processor_kwargs["max_pixels"] = args.max_pixels

    processor = AutoProcessor.from_pretrained(args.model_name, **processor_kwargs)

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    apply_tokenizer_overrides(tokenizer, args)

    logger.info(f"VLM processor type: {type(processor).__name__}")
    logger.info(f"Tokenizer vocab size: {len(tokenizer)}")

    return processor


def load_input_dataset(args) -> DatasetDict:
    """Load input dataset from any source."""
    logger.info(f"Loading dataset from {args.input}")

    dataset = load_dataset_from_source(args.input)

    if isinstance(dataset, Dataset):
        if args.test_size:
            logger.info(f"Creating train/test split with test_size={args.test_size}")
            dataset = dataset.train_test_split(args.test_size, seed=_SPLIT_SEED)
        else:
            logger.info("No test split, using entire dataset as train")
            dataset = DatasetDict({"train": dataset})
    elif args.test_size:
        # A local file, a save_to_disk dir, or a split-less Hub id loads as a DatasetDict — the most
        # common input form. Splitting only the bare-Dataset case would drop --test-size silently.
        if set(dataset) & {"test", "validation"}:
            raise ValueError(
                f"--test-size was given but {args.input} already carries an eval split "
                f"({sorted(set(dataset) & {'test', 'validation'})}). Drop the flag, or point --input "
                "at the train split alone."
            )
        if "train" not in dataset:
            raise ValueError(f"--test-size needs a 'train' split to cut; {args.input} has {sorted(dataset)}.")
        logger.info(f"Creating train/test split with test_size={args.test_size}")
        dataset = DatasetDict({**dataset, **dataset["train"].train_test_split(args.test_size, seed=_SPLIT_SEED)})

    for split, ds in dataset.items():
        logger.info(f"  {split}: {len(ds)} examples")
        if ds.column_names:
            logger.info(f"    columns: {ds.column_names}")

    return dataset


def save_to_s3(output_dir: str, s3_uri: str, overwrite: bool = False):
    """Upload preprocessed dataset to S3."""
    logger.info(f"Uploading to S3: {s3_uri}")

    bucket, key = parse_s3_uri(s3_uri)
    client = S3Client(bucket=bucket)

    # push_folder already performs the exists → delete/raise dance; just translate its error to
    # the CLI's --overwrite hint.
    try:
        client.push_folder(output_dir, key, subfolder=None, overwrite=overwrite, show_progress=True)
    except FileExistsError as e:
        raise FileExistsError(f"Dataset already exists at {s3_uri}. Use --overwrite to replace.") from e
    logger.info(f"Uploaded to {s3_uri}")


def save_to_local(output_dir: str, dest: str, overwrite: bool = False):
    """Publish the staged dataset to a local path, replacing an existing one only on a complete copy.

    Copy into a sibling staging directory first, then swap (:func:`clear_staging_path` — the shared
    staged-publish contract): an ``--overwrite`` run that removed the old directory up front and
    died mid-copy would destroy the only copy, leaving a partial tree whose ``metadata.json`` makes
    ``is_preprocessed_dataset`` accept it — training would then run on a silently truncated dataset.
    The rotated previous dataset is what this path recovers from, so the staging copy is the one
    discarded on failure.
    """
    dest = os.path.abspath(dest)
    if os.path.exists(dest) and not overwrite:
        raise FileExistsError(f"Output path already exists: {dest}. Use --overwrite to replace.")

    staged = clear_staging_path(dest)
    previous = clear_staging_path(dest, DISPLACED_SUFFIX)
    try:
        shutil.copytree(output_dir, staged)
        rotated = os.path.exists(dest)
        if rotated:
            os.rename(dest, previous)
        try:
            os.rename(staged, dest)
        except OSError:
            # Both renames are within one directory, so this is near-impossible — but leaving the
            # destination missing while the old dataset sits under a scratch name is the one outcome
            # this function exists to prevent.
            if rotated:
                os.rename(previous, dest)
            raise
        shutil.rmtree(previous, ignore_errors=True)
    finally:
        shutil.rmtree(staged, ignore_errors=True)
    logger.info(f"Saved to {dest}")


def save_to_hf_hub(
    output_dir: str,
    repo_id: str,
    private: bool = False,
    token: str | None = None,
    overwrite: bool = False,
):
    """Upload preprocessed dataset to HuggingFace Hub."""
    logger.info(f"Uploading to HuggingFace Hub: {repo_id}")

    api = HfApi(token=token)

    # Same refusal the s3 and local writers make: upload_folder overwrites unconditionally, so
    # without this an existing Hub dataset is replaced by a run that never asked to replace it.
    if not overwrite and api.repo_exists(repo_id, repo_type="dataset"):
        raise FileExistsError(f"Dataset already exists at hf://{repo_id}. Use --overwrite to replace.")

    # No blanket catch: ``exist_ok=True`` already absorbs the one benign failure (the repo is
    # already there), so anything else — a bad token, no write scope, an invalid repo id — is a
    # failure the upload below cannot recover from, and swallowing it here only moves the error to
    # a place that no longer names the cause.
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)

    # metadata.json is the completion marker `is_preprocessed_dataset` keys on, so it is cleared
    # first and re-uploaded last: at no point does the repo carry a metadata.json beside a payload
    # that is a mix of the old dataset and a partly-uploaded new one. A run that dies mid-upload
    # leaves a repo that reads as RAW (loud) instead of as a silently truncated preprocessed one.
    with contextlib.suppress(EntryNotFoundError, RepositoryNotFoundError):
        api.delete_file(METADATA_FILE, repo_id=repo_id, repo_type="dataset")

    upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        ignore_patterns=[METADATA_FILE],
    )
    api.upload_file(
        path_or_fileobj=os.path.join(output_dir, METADATA_FILE),
        path_in_repo=METADATA_FILE,
        repo_id=repo_id,
        repo_type="dataset",
    )
    logger.info(f"Uploaded to https://huggingface.co/datasets/{repo_id}")


def main():
    args = parse_args()
    configure_cli_logging(verbose=args.verbose)

    # The shared data-loading helpers log through the accelerate logger, which raises unless the accelerate
    # state is initialized; this is a plain single-process tool, so initialize a minimal state explicitly.
    state = PartialState()
    if state.num_processes > 1:
        # The tokenization passes coordinate across ranks, but the publish does not: every rank would
        # push the same artifact to one S3 key, local directory or Hub repo.
        sys.exit(
            f"prepare_dataset.py is a single-process tool but was launched with {state.num_processes} "
            "processes. Launch it with plain python, not torchrun/accelerate."
        )

    # Output-side classification: a bare "org/name" stays LOCAL — only an explicit hf:// uploads.
    output_type, _, output_key = parse_dataset_destination(args.output)

    if args.vlm and args.pack_sequences:
        logger.error("Packing is not supported for VLM datasets. Remove --pack-sequences.")
        sys.exit(1)

    # PreprocessingConfig enforces this pair too, but only once the input dataset is downloaded.
    if args.images_field and not args.vlm:
        logger.error("--images-field only applies to VLM preprocessing. Add --vlm, or drop the flag.")
        sys.exit(1)

    # Before the input dataset is fetched: the config that owns this pair is only built afterwards.
    reject_unbuildable_masking(args)

    mode_str = "VLM" if args.vlm else "Text SFT"
    logger.info("=" * 60)
    logger.info(f"Dataset Preprocessing ({mode_str})")
    logger.info("=" * 60)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output} (type: {output_type})")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Max length: {args.max_length}")
    if not args.vlm:
        logger.info(f"Pack sequences: {args.pack_sequences}")
    logger.info(f"Num shards: {args.num_shards}")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("DRY RUN - no changes will be made")
        return

    tokenizer_or_processor = setup_vlm_processor(args) if args.vlm else setup_tokenizer(args)

    dataset = load_input_dataset(args)

    config = PreprocessingConfig(
        model_name_or_path=args.model_name,
        max_length=args.max_length,
        mode=args.mode,
        text_field=args.text_field,
        append_eos=not args.no_append_eos,
        conversation_field=args.conversation_field,
        system_prompt=args.system_prompt,
        model_supports_system_role=not args.no_system_role,
        tools_field=args.tools_field,
        interleaved_thinking=args.interleaved_thinking,
        images_field=args.images_field,
        train_on_completions_only=args.train_on_completions_only,
        assistant_message_template=args.assistant_message_template,
        pad_token=args.pad_token,
        eos_token=args.eos_token,
        bos_token=args.bos_token,
        # Resolved, not the path: the training side compares against its own resolved template.
        chat_template=load_chat_template(args.chat_template) if args.chat_template else None,
        pack_sequences=args.pack_sequences,
        packing_strategy=args.packing_strategy,
        num_shards=args.num_shards,
        num_proc=resolve_map_num_proc(args.num_proc),
        tokenizer_backend=args.tokenizer_backend,
        is_vlm=args.vlm,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Processing dataset (temp dir: {temp_dir})")

        result = preprocess_dataset(
            dataset=dataset,
            tokenizer_or_processor=tokenizer_or_processor,
            config=config,
            output_dir=temp_dir,
        )

        metadata = result["metadata"]
        logger.info("=" * 60)
        logger.info("Preprocessing Complete")
        logger.info("=" * 60)
        logger.info(f"Train examples: {metadata.total_train_examples}")
        logger.info(f"Test examples: {metadata.total_test_examples}")
        logger.info(f"Num shards: {metadata.num_shards}")
        if args.vlm:
            logger.info(f"VLM mode: {metadata.is_vlm}")
            logger.info(f"Has pixel_values: {metadata.has_pixel_values}")
        else:
            logger.info(f"Packed: {metadata.packed}")
        logger.info("=" * 60)

        if output_type == "s3":
            save_to_s3(temp_dir, args.output, args.overwrite)
        elif output_type == "hf_hub":
            save_to_hf_hub(temp_dir, output_key, args.private, args.hf_token, args.overwrite)
        elif output_type == "local":
            # output_key, not args.output: the destination parser strips a file:// prefix.
            save_to_local(temp_dir, output_key, args.overwrite)

    logger.info("Done!")


if __name__ == "__main__":
    main()
