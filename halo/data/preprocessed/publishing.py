"""The offline preparation tool's run around the bake: its own arguments and worker count, the tokenizer or processor
and the input it bakes, the bake into a staging directory, and the publish to a local path, S3 or the Hub. Behind
``scripts/before_training/prepare_dataset.py``."""

import contextlib
import hashlib
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

from halo.checkpoint.atomic import DISPLACED_SUFFIX, clear_staging_path
from halo.data.pipeline.coordinated_map import dataset_num_proc_override, ensure_cache_dir, scratch_dataset_maps
from halo.data.preprocessed.metadata import PreprocessingConfig
from halo.data.preprocessed.preparation import preprocess_dataset
from halo.data.render_columns import reject_missing_render_column
from halo.data.sources.loading import load_dataset_from_source
from halo.data.sources.paths import EVAL_SPLIT_NAMES, METADATA_FILE, parse_dataset_destination, parse_s3_uri
from halo.data.sources.s3_client import S3Client
from halo.hardware import available_cpu_count
from halo.model_loading.tokenizer_setup import adopt_tokenizer_chat_template, resolve_tokenizer, set_run_chat_template
from halo.models.segment_markers import reject_compressed_kv_rows

logger = logging.getLogger(__name__)

# Fixed so --test_size cuts the same split on every re-run: the assignment is baked into the
# published artifact, and a re-preparation that reshuffled it would no longer match.
_SPLIT_SEED = 42

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
    hf_token: str | None = field(
        default=None, metadata={"help": "Token for the hf:// upload; the loads use the ambient login."}
    )
    overwrite: bool = field(default=False, metadata={"help": "Replace an existing output."})
    dry_run: bool = field(default=False, metadata={"help": "Show what would be done without processing."})
    verbose: bool = field(default=False, metadata={"help": "Log at DEBUG."})


def bake_num_proc() -> int:
    """The bake's default worker count: ``HALO_DATASET_NUM_PROC``, else every CPU the process may run on but
    :data:`_BAKE_RESERVED_CPUS`. The tool runs alone on its host, so the training default's share and cap would idle
    most of a large one."""
    override = dataset_num_proc_override()
    return override if override is not None else max(1, available_cpu_count() - _BAKE_RESERVED_CPUS)


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


def stage_local_output(dest: str, overwrite: bool = False) -> str:
    """The sibling of ``dest`` a local output bakes into, cleared of an earlier run's: on the destination's
    filesystem, so the publish is a rename rather than a second write. An existing ``dest`` without ``overwrite``
    raises here, before the bake."""
    if os.path.exists(dest) and not overwrite:
        raise FileExistsError(f"Output path already exists: {dest}. Use --overwrite to replace.")
    return clear_staging_path(dest)


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


def bake_scratch_path(destination: str) -> str:
    """The datasets-cache directory a bake to ``destination`` holds its intermediate maps in, cleared of what a killed
    bake to it left: named after the destination, so a rerun reaps those full corpus copies before it bakes."""
    path = os.path.join(ensure_cache_dir(), f"prepare-dataset-{hashlib.md5(destination.encode()).hexdigest()}")
    shutil.rmtree(path, ignore_errors=True)
    return path


def bake(
    dataset: DatasetDict, tokenizer_or_processor, config: PreprocessingConfig, output_dir: str, destination: str
) -> None:
    """Preprocess ``dataset`` into ``output_dir`` for ``destination``. Every intermediate map writes a scratch
    directory of the datasets cache (:func:`bake_scratch_path`), removed afterwards, since no later run re-reads
    them."""
    logger.info(f"Processing dataset into {output_dir}")
    with scratch_dataset_maps(bake_scratch_path(destination)):
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
    """Bake ``args.input`` under ``config`` and publish it to ``args.output``: refused under a multi-process launch,
    a dry run stops after the summary, and a local output is baked beside its destination and swapped in."""
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

    if output_type == "local":
        # output_key, not args.output: the destination parser strips a file:// prefix.
        dest = os.path.abspath(output_key)
        staged = stage_local_output(dest, args.overwrite)
        try:
            bake(dataset, tokenizer_or_processor, config, staged, dest)
            publish_local_output(staged, dest)
        finally:
            shutil.rmtree(staged, ignore_errors=True)
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            bake(dataset, tokenizer_or_processor, config, temp_dir, args.output)
            if output_type == "s3":
                save_to_s3(temp_dir, args.output, args.overwrite)
            else:
                save_to_hf_hub(temp_dir, output_key, args.private, args.hf_token, args.overwrite)

    logger.info("Done!")
