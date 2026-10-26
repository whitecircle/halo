#!/usr/bin/env python
"""Offline SFT dataset preparation: tokenize, optionally pack, optionally shard, then publish to S3, the Hub or a
local path, so training skips tokenization at startup. The flags are :class:`PreprocessingConfig`'s fields plus
:class:`PrepareDatasetArguments`'s, and a YAML file of them works as the first argument."""

import contextlib
import logging
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field

from accelerate import PartialState
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, upload_folder
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, PreTrainedTokenizer

from scripts.cli_arguments import add_trust_remote_code_arg
from src.checkpoint.atomic import DISPLACED_SUFFIX, clear_staging_path
from src.configs.parser import HaloArgumentParser
from src.data.pipeline.coordinated_map import resolve_map_num_proc
from src.data.preprocessed.metadata import PreprocessingConfig
from src.data.preprocessed.preparation import preprocess_dataset
from src.data.render_columns import reject_missing_render_column
from src.data.sources.loading import load_dataset_from_source
from src.data.sources.paths import EVAL_SPLIT_NAMES, METADATA_FILE, parse_dataset_destination, parse_s3_uri
from src.data.sources.s3_client import S3Client
from src.log import configure_cli_logging
from src.model_loading.tokenizer_setup import (
    adopt_tokenizer_chat_template,
    load_chat_template,
    resolve_tokenizer,
    set_run_chat_template,
)
from src.models.segment_markers import reject_compressed_kv_rows

logger = logging.getLogger(__name__)

# Fixed so --test_size cuts the same split on every re-run: the assignment is baked into the
# published artifact, and a re-preparation that reshuffled it would no longer match.
_SPLIT_SEED = 42


@dataclass
class PrepareDatasetArguments:
    """The tool's own flags; the bake's are :class:`PreprocessingConfig`'s."""

    input: str = field(metadata={"help": "Input dataset: an S3 URI, a Hub ID or a local path."})
    output: str = field(metadata={"help": "Destination: an S3 URI, hf://org/name or a local path."})
    test_size: float | None = field(
        default=None, metadata={"help": "Fraction of a single-split input held out as the test split."}
    )
    private: bool = field(default=False, metadata={"help": "Make a Hub dataset private."})
    hf_token: str | None = field(
        default=None, metadata={"help": "Token for the hf:// upload; the loads use the ambient login."}
    )
    overwrite: bool = field(default=False, metadata={"help": "Replace an existing output."})
    dry_run: bool = field(default=False, metadata={"help": "Show what would be done without processing."})
    verbose: bool = field(default=False, metadata={"help": "Log at DEBUG."})


def parse_args() -> tuple[PrepareDatasetArguments, PreprocessingConfig]:
    """The tool's arguments and the bake's config, built before anything is fetched. Where the tool's defaults
    differ from the config's: completion masking follows the mode as training does (on for chat, while text
    refuses it), the model's own code is trusted, and the worker count and chat template are recorded resolved."""
    parser = add_trust_remote_code_arg(HaloArgumentParser((PrepareDatasetArguments, PreprocessingConfig)))
    # The field's dashed alias outlives the canonical flag's takeover of its name, so it takes the default too.
    parser.set_defaults(trust_remote_code=True, train_on_completions_only=None)
    values = parser.load_values()
    values.setdefault("trust_remote_code", parser.get_default("trust_remote_code"))
    values.setdefault("train_on_completions_only", values.get("mode", PreprocessingConfig.mode) == "chat")
    values["num_proc"] = resolve_map_num_proc(values.get("num_proc"))
    if values.get("chat_template"):
        values["chat_template"] = load_chat_template(values["chat_template"])
    return parser.parse_dict(values)


def apply_tokenizer_overrides(tokenizer, config: PreprocessingConfig) -> None:
    """Apply the config's recorded pad/eos/bos/chat-template overrides, then guarantee a pad token; the tokenizer
    and processor paths share it, so an artifact carries the same special tokens either way."""
    if config.pad_token:
        tokenizer.pad_token = config.pad_token
    if config.eos_token:
        tokenizer.eos_token = config.eos_token
    if config.bos_token:
        tokenizer.bos_token = config.bos_token
    if config.chat_template:
        set_run_chat_template(tokenizer, config.chat_template)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def setup_tokenizer(config: PreprocessingConfig) -> PreTrainedTokenizer:
    logger.info(f"Loading tokenizer from {config.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=config.trust_remote_code)
    apply_tokenizer_overrides(tokenizer, config)

    logger.info(f"Tokenizer vocab size: {len(tokenizer)}")
    logger.info(f"Pad token: {tokenizer.pad_token}")
    logger.info(f"EOS token: {tokenizer.eos_token}")

    return tokenizer


def setup_vlm_processor(config: PreprocessingConfig):
    logger.info(f"Loading VLM processor from {config.model_name_or_path}")
    processor_kwargs = {"trust_remote_code": config.trust_remote_code}
    if config.min_pixels is not None:
        processor_kwargs["min_pixels"] = config.min_pixels
    if config.max_pixels is not None:
        processor_kwargs["max_pixels"] = config.max_pixels
    processor = AutoProcessor.from_pretrained(config.model_name_or_path, **processor_kwargs)
    tokenizer = resolve_tokenizer(processor)
    apply_tokenizer_overrides(tokenizer, config)
    adopt_tokenizer_chat_template(processor)

    logger.info(f"VLM processor type: {type(processor).__name__}")
    logger.info(f"Tokenizer vocab size: {len(tokenizer)}")

    return processor


def load_input_dataset(args: PrepareDatasetArguments) -> DatasetDict:
    """The input as a DatasetDict with a ``train`` split, its test split cut by ``test_size`` when set."""
    logger.info(f"Loading dataset from {args.input}")

    dataset = load_dataset_from_source(args.input)

    if isinstance(dataset, Dataset):
        if args.test_size:
            logger.info(f"Creating train/test split with test_size={args.test_size}")
            dataset = dataset.train_test_split(args.test_size, seed=_SPLIT_SEED)
        else:
            logger.info("No test split, using entire dataset as train")
            dataset = DatasetDict({"train": dataset})
    elif "train" not in dataset:
        # Only the train and test splits are baked, so a split set without train would publish an
        # artifact holding no rows.
        raise ValueError(
            f"{args.input} has no 'train' split ({sorted(dataset)}). Point --input at the split to "
            "prepare: an '@<split>' suffix on a Hub ID, or the split's own directory of a saved DatasetDict."
        )
    elif args.test_size:
        # A local file, a save_to_disk dir or a split-less Hub id loads as a DatasetDict, the most
        # common input form. Splitting only the bare-Dataset case would drop --test_size.
        held_out = [name for name in EVAL_SPLIT_NAMES if name in dataset]
        if held_out:
            raise ValueError(
                f"--test_size was given but {args.input} already carries an eval split ({held_out}). "
                "Drop the flag, or point --input at the train split alone."
            )
        logger.info(f"Creating train/test split with test_size={args.test_size}")
        dataset = DatasetDict({**dataset, **dataset["train"].train_test_split(args.test_size, seed=_SPLIT_SEED)})

    for split, ds in dataset.items():
        logger.info(f"  {split}: {len(ds)} examples")
        if ds.column_names:
            logger.info(f"    columns: {ds.column_names}")

    return dataset


def require_declared_columns(dataset: DatasetDict, config: PreprocessingConfig, source: str) -> None:
    """Refuse a column knob of this mode naming no column of the splits the artifact bakes, before a map worker
    dies on it with a bare ``KeyError`` or a tools or images column is silently left out."""
    rendered = "text_field" if config.mode == "text" else "conversation_field"
    for knob in (rendered, "tools_field", "images_field"):
        column = getattr(config, knob)
        if column:
            reject_missing_render_column(dataset, source, knob, column)


def save_to_s3(output_dir: str, s3_uri: str, overwrite: bool = False):
    """Upload preprocessed dataset to S3."""
    logger.info(f"Uploading to S3: {s3_uri}")

    bucket, key = parse_s3_uri(s3_uri)
    client = S3Client(bucket=bucket)

    # push_folder already handles the exists / delete / raise sequence; translate its error into the
    # CLI's --overwrite hint.
    try:
        client.push_folder(output_dir, key, subfolder=None, overwrite=overwrite, show_progress=True)
    except FileExistsError as e:
        raise FileExistsError(f"Dataset already exists at {s3_uri}. Use --overwrite to replace.") from e
    logger.info(f"Uploaded to {s3_uri}")


def save_to_local(output_dir: str, dest: str, overwrite: bool = False):
    """Publish the staged dataset to a local path through a sibling copy swapped in whole: removing the old one
    first, a run dying mid-copy would leave a partial tree whose ``metadata.json`` still reads as preprocessed."""
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
            # Never leave the destination missing while the old dataset sits under a scratch name.
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
    """Upload the staged dataset to the Hub, ``metadata.json`` (the completion marker) cleared first and uploaded
    last, so an interrupted upload reads as raw rather than as a truncated preprocessed dataset."""
    logger.info(f"Uploading to HuggingFace Hub: {repo_id}")

    api = HfApi(token=token)

    # upload_folder overwrites unconditionally.
    if not overwrite and api.repo_exists(repo_id, repo_type="dataset"):
        raise FileExistsError(f"Dataset already exists at hf://{repo_id}. Use --overwrite to replace.")

    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)

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
    args, config = parse_args()
    configure_cli_logging(verbose=args.verbose)

    state = PartialState()
    if state.num_processes > 1:
        # The tokenization passes coordinate across ranks, but the publish does not: every rank would
        # push the same artifact to one S3 key, local directory or Hub repo.
        sys.exit(
            f"prepare_dataset.py is a single-process tool but was launched with {state.num_processes} "
            "processes. Launch it with plain python, not torchrun/accelerate."
        )

    # Output-side classification: a bare "org/name" stays local; only an explicit hf:// uploads.
    output_type, _, output_key = parse_dataset_destination(args.output)

    if config.pack_sequences:
        hf_config = AutoConfig.from_pretrained(config.model_name_or_path, trust_remote_code=config.trust_remote_code)
        reject_compressed_kv_rows(hf_config, "--pack_sequences")

    logger.info("=" * 60)
    logger.info(f"Dataset Preprocessing ({'VLM' if config.is_vlm else 'Text SFT'})")
    logger.info("=" * 60)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output} (type: {output_type})")
    logger.info(f"Model: {config.model_name_or_path}")
    logger.info(f"Max length: {config.max_length}")
    if not config.is_vlm:
        logger.info(f"Pack sequences: {config.pack_sequences}")
    logger.info(f"Num shards: {config.num_shards}")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("DRY RUN - no changes will be made")
        return

    tokenizer_or_processor = setup_vlm_processor(config) if config.is_vlm else setup_tokenizer(config)
    dataset = load_input_dataset(args)
    require_declared_columns(dataset, config, args.input)

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
        if config.is_vlm:
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
