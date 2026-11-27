"""The offline preparation tool's run around the bake, behind ``scripts/before_training/prepare_dataset.py``:
arguments, input, staging and the publish to a local path, S3 or the Hub."""

import contextlib
import hashlib
import logging
import os
import shutil
import sys
from dataclasses import dataclass, field

from accelerate import PartialState
from datasets import Dataset, DatasetDict
from filelock import FileLock, Timeout
from huggingface_hub import HfApi, upload_folder
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, PreTrainedTokenizer

from halo.checkpoint.atomic import DISPLACED_SUFFIX, clear_staging_path
from halo.data.collators.collator_rules import reject_unpackable_family
from halo.data.conversation import TEMPLATE_COLUMN_KNOBS
from halo.data.pipeline.coordinated_map import dataset_num_proc_override, ensure_cache_dir, scratch_dataset_maps
from halo.data.preprocessed.metadata import PreprocessingConfig
from halo.data.preprocessed.preparation import preprocess_dataset
from halo.data.render_columns import reject_missing_render_column
from halo.data.sources.loading import load_dataset_from_source, split_train_test
from halo.data.sources.paths import EVAL_SPLIT_NAMES, METADATA_FILE, parse_dataset_destination, parse_s3_uri
from halo.data.sources.s3_client import S3Client
from halo.data.vlm_render import apply_image_pixel_bounds
from halo.hardware import available_cpu_count
from halo.model_loading.tokenizer_setup import adopt_tokenizer_chat_template, resolve_tokenizer, set_run_chat_template

logger = logging.getLogger(__name__)

# CPUs the bake's default worker count leaves to the tool's own process and the host.
_BAKE_RESERVED_CPUS = 2


@dataclass
class PrepareDatasetArguments:
    """The tool's own flags; the bake's are :class:`PreprocessingConfig`'s."""

    input: str = field(metadata={"help": "Input dataset: an S3 URI, a Hub ID or a local path."})
    output: str = field(metadata={"help": "Destination: an S3 URI, hf://org/name or a local path."})
    test_size: float | None = field(
        default=None, metadata={"help": "Fraction of a single-split input held out as the test split."}
    )
    private: bool = field(default=False, metadata={"help": "Make a Hub dataset private."})
    overwrite: bool = field(default=False, metadata={"help": "Replace an existing output."})
    dry_run: bool = field(default=False, metadata={"help": "Show what would be done without processing."})
    verbose: bool = field(default=False, metadata={"help": "Log at DEBUG."})
    allow_cross_document_linear_attention: bool = field(
        default=False,
        metadata={
            "help": "Bake --pack_sequences rows for a model whose linear-attention layers or other sequence mixers "
            "receive no document boundary. Training them takes the same opt-in; off, the bake is refused as training "
            "would refuse its rows."
        },
    )


def bake_num_proc() -> int:
    """``HALO_DATASET_NUM_PROC``, else every CPU but :data:`_BAKE_RESERVED_CPUS`: the tool runs alone on its host, so
    the training default's share and cap would idle most of a large one."""
    override = dataset_num_proc_override()
    return override if override is not None else max(1, available_cpu_count() - _BAKE_RESERVED_CPUS)


def apply_tokenizer_overrides(tokenizer, config: PreprocessingConfig) -> None:
    """Apply recorded pad/eos/bos/chat-template overrides, then guarantee a pad token; both load paths share it."""
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


def load_tokenizer(config: PreprocessingConfig) -> PreTrainedTokenizer:
    logger.info(f"Loading tokenizer from {config.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=config.trust_remote_code)
    apply_tokenizer_overrides(tokenizer, config)

    logger.info(f"Tokenizer vocab size: {len(tokenizer)}")
    logger.info(f"Pad token: {tokenizer.pad_token}")
    logger.info(f"EOS token: {tokenizer.eos_token}")

    return tokenizer


def load_vlm_processor(config: PreprocessingConfig):
    logger.info(f"Loading VLM processor from {config.model_name_or_path}")
    processor = AutoProcessor.from_pretrained(config.model_name_or_path, trust_remote_code=config.trust_remote_code)
    apply_image_pixel_bounds(processor, config.min_pixels, config.max_pixels)
    tokenizer = resolve_tokenizer(processor)
    apply_tokenizer_overrides(tokenizer, config)
    adopt_tokenizer_chat_template(processor)

    logger.info(f"VLM processor type: {type(processor).__name__}")
    logger.info(f"Tokenizer vocab size: {len(tokenizer)}")

    return processor


def load_input_dataset(args: PrepareDatasetArguments) -> DatasetDict:
    """A DatasetDict with a ``train`` split, its test split cut by ``test_size`` when set."""
    logger.info(f"Loading dataset from {args.input}")

    dataset = load_dataset_from_source(args.input)

    if isinstance(dataset, Dataset):
        if args.test_size:
            logger.info(f"Creating train/test split with test_size={args.test_size}")
            dataset = split_train_test(dataset, args.test_size)
        else:
            logger.info("No test split, using entire dataset as train")
            dataset = DatasetDict({"train": dataset})
    elif "train" not in dataset:
        # Only train and test are baked, so a split set without train would publish an artifact holding no rows.
        raise ValueError(
            f"{args.input} has no 'train' split ({sorted(dataset)}). Point --input at the split to "
            "prepare: an '@<split>' suffix on a Hub ID, or the split's own directory of a saved DatasetDict."
        )
    elif args.test_size:
        # The most common input form loads as a DatasetDict; splitting only a bare Dataset would drop --test_size.
        held_out = [name for name in EVAL_SPLIT_NAMES if name in dataset]
        if held_out:
            raise ValueError(
                f"--test_size was given but {args.input} already carries an eval split ({held_out}). "
                "Drop the flag, or point --input at the train split alone."
            )
        logger.info(f"Creating train/test split with test_size={args.test_size}")
        dataset = DatasetDict({**dataset, **split_train_test(dataset["train"], args.test_size)})

    for split, ds in dataset.items():
        logger.info(f"  {split}: {len(ds)} examples")
        if ds.column_names:
            logger.info(f"    columns: {ds.column_names}")

    return dataset


def require_declared_columns(dataset: DatasetDict, config: PreprocessingConfig, source: str) -> None:
    """Refuse a column knob naming no column of the baked splits, before a map worker dies on a bare ``KeyError`` or an
    optional column is silently left out."""
    rendered = "text_field" if config.mode == "text" else "conversation_field"
    for knob in (rendered, *TEMPLATE_COLUMN_KNOBS, "images_field"):
        column = getattr(config, knob)
        if column:
            reject_missing_render_column(dataset, source, knob, column)


def save_to_s3(output_dir: str, s3_uri: str, overwrite: bool = False):
    logger.info(f"Uploading to S3: {s3_uri}")
    bucket, key = parse_s3_uri(s3_uri)
    S3Client(bucket=bucket).push_folder(output_dir, key, subfolder=None, overwrite=overwrite, show_progress=True)
    logger.info(f"Uploaded to {s3_uri}")


def stage_local_output(dest: str, overwrite: bool = False) -> str:
    """The sibling of ``dest`` a local output bakes into, on its filesystem so the publish is a rename. An existing
    ``dest`` without ``overwrite`` raises here, before the bake."""
    if os.path.exists(dest) and not overwrite:
        raise FileExistsError(f"Output path already exists: {dest}. Use --overwrite to replace.")
    return clear_staging_path(dest)


def stage_remote_output(output_type: str, output_key: str, scratch: str, args: PrepareDatasetArguments) -> str:
    """The directory an S3 or Hub output bakes into, beside ``scratch``. The destination is probed first, so an existing
    one without ``--overwrite``, and on S3 a missing bucket or credentials that may not read it, raise before the bake;
    credentials that may read but not write fail at the upload."""
    if output_type == "s3":
        bucket, key = parse_s3_uri(args.output)
        exists = S3Client(bucket=bucket).exists(key)
    else:
        exists = HfApi().repo_exists(output_key, repo_type="dataset")
    if exists and not args.overwrite:
        raise FileExistsError(f"Dataset already exists at {args.output}. Use --overwrite to replace.")
    return clear_staging_path(scratch)


def publish_local_output(staged: str, dest: str) -> None:
    """Swap the baked ``staged`` tree in as ``dest`` whole: removing the old one first, a run dying mid-write would
    leave a partial tree whose ``metadata.json`` still reads as preprocessed."""
    previous = clear_staging_path(dest, DISPLACED_SUFFIX)
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
    logger.info(f"Saved to {dest}")


def save_to_hf_hub(output_dir: str, repo_id: str, private: bool = False):
    """Upload the staged dataset to the Hub, ``metadata.json`` (the completion marker) cleared first and uploaded
    last, so an interrupted upload reads as raw rather than as a truncated preprocessed dataset. The old bake's files
    the new one lacks go with the upload."""
    logger.info(f"Uploading to HuggingFace Hub: {repo_id}")

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)

    with contextlib.suppress(EntryNotFoundError, RepositoryNotFoundError):
        api.delete_file(METADATA_FILE, repo_id=repo_id, repo_type="dataset")

    upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        repo_type="dataset",
        ignore_patterns=[METADATA_FILE],
        # A sharded bake replaced by an unsharded one would otherwise keep its shard index, which the loader reads
        # first. The dataset card stays.
        delete_patterns=["dataset_dict.json", "train/*", "test/*"],
    )
    api.upload_file(
        path_or_fileobj=os.path.join(output_dir, METADATA_FILE),
        path_in_repo=METADATA_FILE,
        repo_id=repo_id,
        repo_type="dataset",
    )
    logger.info(f"Uploaded to https://huggingface.co/datasets/{repo_id}")


def publish_remote_output(staged: str, output_type: str, output_key: str, args: PrepareDatasetArguments) -> None:
    """Upload ``staged`` and remove it. A failed upload keeps it and names it: only the upload is left to redo."""
    try:
        if output_type == "s3":
            save_to_s3(staged, args.output, args.overwrite)
        else:
            save_to_hf_hub(staged, output_key, args.private)
    except Exception as e:
        finish = (
            f" `python scripts/before_training/s3.py push {staged} {args.output} --overwrite` finishes the publish."
            if output_type == "s3"
            else ""
        )
        raise RuntimeError(
            f"Publishing the bake to {args.output} failed ({type(e).__name__}: {e}). The baked dataset is kept at "
            f"{staged} until the next run to this output.{finish}"
        ) from e
    shutil.rmtree(staged)


def bake_scratch_path(destination: str) -> str:
    """The directory a bake holds intermediate maps in, named after the destination so a rerun reaps a killed one's."""
    return os.path.join(ensure_cache_dir(), f"prepare-dataset-{hashlib.md5(destination.encode()).hexdigest()}")


def bake(
    dataset: DatasetDict, tokenizer_or_processor, config: PreprocessingConfig, output_dir: str, scratch: str
) -> None:
    """Every intermediate map writes ``scratch``, cleared of a killed bake's first and removed afterwards, since no
    later run re-reads them."""
    logger.info(f"Processing dataset into {output_dir}")
    shutil.rmtree(scratch, ignore_errors=True)
    with scratch_dataset_maps(scratch):
        metadata = preprocess_dataset(
            dataset=dataset, tokenizer_or_processor=tokenizer_or_processor, config=config, output_dir=output_dir
        )["metadata"]

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


def run_dataset_preparation(args: PrepareDatasetArguments, config: PreprocessingConfig) -> None:
    """Refused under a multi-process launch and while another run bakes the same output; a dry run stops after the
    summary. The destination is checked before the input loads; a local output is baked beside it and swapped in."""
    state = PartialState()
    if state.num_processes > 1:
        # The publish does not coordinate: every rank would push one artifact to one S3 key, directory or Hub repo.
        sys.exit(
            f"prepare_dataset.py is a single-process tool but was launched with {state.num_processes} "
            "processes. Launch it with plain python, not torchrun/accelerate."
        )

    # Output-side classification: a bare "org/name" stays local; only an explicit hf:// uploads.
    output_type, output_bucket, output_key = parse_dataset_destination(args.output)

    if args.allow_cross_document_linear_attention and not config.pack_sequences:
        raise ValueError("--allow_cross_document_linear_attention needs --pack_sequences: only packed rows cross.")
    if config.pack_sequences:
        hf_config = AutoConfig.from_pretrained(config.model_name_or_path, trust_remote_code=config.trust_remote_code)
        reject_unpackable_family(
            hf_config,
            allow_cross_document_linear_attention=args.allow_cross_document_linear_attention,
            flag="--pack_sequences",
        )

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

    # The parsed key, not args.output: the parser strips a file:// prefix and an S3 key's trailing slashes, so every
    # spelling of one output takes one lock.
    if output_type == "local":
        destination = os.path.abspath(output_key)
    elif output_type == "s3":
        destination = f"s3://{output_bucket}/{output_key}"
    else:
        destination = args.output
    scratch = bake_scratch_path(destination)
    try:
        held = FileLock(f"{scratch}.lock").acquire(timeout=0)
    except Timeout:
        sys.exit(
            f"Another prepare_dataset.py is baking {args.output}: two bakes of one output remove each other's scratch "
            "and staged output. Wait for it to finish, or bake to another --output."
        )
    with held:
        if output_type == "local":
            staged = stage_local_output(destination, args.overwrite)
        else:
            staged = stage_remote_output(output_type, output_key, scratch, args)
        tokenizer_or_processor = load_vlm_processor(config) if config.is_vlm else load_tokenizer(config)
        dataset = load_input_dataset(args)
        require_declared_columns(dataset, config, args.input)
        try:
            bake(dataset, tokenizer_or_processor, config, staged, scratch)
            if output_type == "local":
                publish_local_output(staged, destination)
        except BaseException:
            shutil.rmtree(staged, ignore_errors=True)
            raise
        if output_type != "local":
            publish_remote_output(staged, output_type, output_key, args)

    logger.info("Done!")
