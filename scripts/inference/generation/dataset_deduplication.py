#!/usr/bin/env python
"""Semantic deduplication of a large dataset via sentence embeddings + FAISS similarity search.

Embeds ``--text_field``, then drops every row whose cosine similarity to an earlier row exceeds
``--similarity_threshold``.

Usage:
    python scripts/inference/generation/dataset_deduplication.py \
        --input_path dataset.jsonl \
        --output_path deduplicated_dataset.jsonl \
        --text_field "text" \
        --model_name "sentence-transformers/all-MiniLM-L6-v2" \
        --similarity_threshold 0.95 \
        --batch_size 32
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModel, AutoTokenizer

from scripts._common import add_trust_remote_code_arg
from src.data.deduplication import faiss_deduplicate_mr, faiss_deduplicate_mr_multistep, process_texts
from src.data.sources.paths import DATA_FILE_BUILDERS, parse_dataset_source
from src.models.loading.checkpoint_coverage import from_pretrained_verified
from src.models.patches.buffer_fixes import finalize_loaded_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    # `src` configures the root logger at import, which makes an unforced basicConfig a no-op — root
    # would stay at WARNING and every progress/summary line below (including the dedup counts and the
    # file log) would be silently dropped.
    force=True,
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Deduplicate large datasets using semantic embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input/Output arguments
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help=f"Local data file ({', '.join(DATA_FILE_BUILDERS)}) or HuggingFace dataset id 'org/name'",
    )
    parser.add_argument("--output_path", type=str, required=True, help="Path to output deduplicated dataset")
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["jsonl", "json", "parquet", "csv"],
        default="jsonl",
        help="Output format for the deduplicated dataset",
    )

    # Dataset configuration
    parser.add_argument(
        "--text_field", type=str, required=True, help="Name of the field containing text to deduplicate on"
    )
    parser.add_argument(
        "--additional_fields",
        type=str,
        nargs="+",
        default=None,
        help="Keep only these fields in the output (whitelist; if not specified, all fields are kept). "
        "nargs=+ so a bare --additional_fields is rejected rather than whitelisting nothing and "
        "stripping every column.",
    )
    parser.add_argument(
        "--dataset_config", type=str, default=None, help="Dataset configuration name (for HuggingFace datasets)"
    )
    parser.add_argument(
        "--dataset_split", type=str, default="train", help="Dataset split to use (for HuggingFace datasets)"
    )

    # Model configuration
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Name or path of the embedding model",
    )
    add_trust_remote_code_arg(parser, default=False)
    parser.add_argument("--model_max_length", type=int, default=512, help="Maximum sequence length for tokenization")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for embedding computation")
    parser.add_argument("--device", type=str, default="auto", help="Device to use for computation (auto, cpu, cuda)")

    # Deduplication parameters
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.95,
        help="Cosine similarity threshold for considering items as duplicates",
    )
    parser.add_argument("--dedup_batch_size", type=int, default=100000, help="Batch size for deduplication processing")
    parser.add_argument(
        "--dedup_steps",
        type=int,
        default=3,
        help="Number of deduplication steps (multi-step approach for better results)",
    )
    parser.add_argument(
        "--max_workers", type=int, default=None, help="Maximum number of worker threads for parallel processing"
    )

    # Performance and debugging
    parser.add_argument("--save_embeddings", action="store_true", help="Save computed embeddings to disk for reuse")
    parser.add_argument("--load_embeddings", type=str, default=None, help="Path to precomputed embeddings file (.npy)")
    parser.add_argument(
        "--embeddings_output_path",
        type=str,
        default=None,
        help="Where to write the computed embeddings; only used together with --save_embeddings "
        "(default: <output_path>.embeddings.npy)",
    )
    parser.add_argument("--dry_run", action="store_true", help="Perform a dry run without saving the output")
    parser.add_argument(
        "--sample_size", type=int, default=None, help="Process only a sample of the dataset (for testing)"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser.parse_args()


def setup_device(device_arg: str) -> torch.device:
    """Setup and return the appropriate device."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")
    else:
        device = torch.device(device_arg)
        logger.info(f"Using specified device: {device}")

    return device


def load_dataset_from_path(
    input_path: str,
    text_field: str,
    dataset_config: str | None = None,
    dataset_split: str = "train",
    sample_size: int | None = None,
) -> Dataset:
    """Load the input dataset from a local data file or the HuggingFace Hub.

    Classified by :func:`parse_dataset_source` — the toolkit's own convention — rather than by
    whether the path happens to exist on this host: an existence probe silently reads a stale local
    file that shadows a Hub id, and reads a mistyped local path as a Hub id whose failure names the
    network instead of the typo. The builder comes from ``DATA_FILE_BUILDERS``, the same table the
    classifier and ``src.data.sources.loading`` read, so an extension can never be accepted as
    local and then have no builder here.
    """
    logger.info(f"Loading dataset from: {input_path}")

    source_type, _bucket, path = parse_dataset_source(input_path)
    if source_type == "s3":
        raise ValueError(
            f"'{input_path}' is an S3 URI; this tool reads a local data file or a Hub dataset id. "
            f"Download it first, or point --input_path at 'org/name'."
        )

    if source_type == "hf_hub":
        logger.info(f"Loading dataset from the HuggingFace Hub: {path}")
        dataset = load_dataset(path, name=dataset_config, split=dataset_split)
    else:
        builder = next((b for ext, b in DATA_FILE_BUILDERS.items() if path.endswith(ext)), None)
        if builder is None:
            raise ValueError(
                f"'{input_path}' is neither a data file ({', '.join(DATA_FILE_BUILDERS)}) nor a Hub "
                f"dataset id 'org/name'."
            )
        dataset = load_dataset(builder, data_files=path, split="train")

    if text_field not in dataset.column_names:
        raise ValueError(f"Text field '{text_field}' not found in dataset. Available fields: {dataset.column_names}")

    if sample_size is not None and len(dataset) > sample_size:
        logger.info(f"Sampling {sample_size} examples from {len(dataset)} total")
        dataset = dataset.shuffle(seed=42).select(range(sample_size))

    logger.info(f"Loaded dataset with {len(dataset)} examples")
    logger.info(f"Dataset columns: {dataset.column_names}")

    return dataset


def compute_embeddings(
    texts: list[str],
    model_name: str,
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 512,
    save_path: str | None = None,
    trust_remote_code: bool = False,
) -> np.ndarray:
    """Compute embeddings for a list of texts."""
    logger.info(f"Computing embeddings using model: {model_name}")
    logger.info(f"Processing {len(texts)} texts with batch size {batch_size}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    # Gated like every other model load in scripts/: an embedding checkpoint whose tensors are
    # randomly initialized still returns vectors, and the dedup decisions they drive look ordinary.
    model = from_pretrained_verified(AutoModel, model_name, trust_remote_code=trust_remote_code)
    model.to(device)
    # Non-persistent buffers come back uninitialized on transformers 5; unrepaired, the embeddings the
    # dedup decisions are drawn from are computed with a dead RoPE. Run once the tensors are placed.
    finalize_loaded_model(model)
    model.eval()
    tokenizer.model_max_length = max_length

    start_time = time.time()
    embeddings = process_texts(
        texts=texts, batch_size=batch_size, model=model, tokenizer=tokenizer, device=device, normalize=True
    )

    computation_time = time.time() - start_time
    logger.info(f"Computed {len(embeddings)} embeddings in {computation_time:.2f}s")
    logger.info(f"Embedding shape: {embeddings.shape}")

    if save_path:
        logger.info(f"Saving embeddings to: {save_path}")
        np.save(save_path, embeddings)

    return embeddings


def deduplicate_embeddings(
    embeddings: np.ndarray,
    similarity_threshold: float = 0.95,
    batch_size: int = 100000,
    steps: int = 3,
    max_workers: int | None = None,
) -> np.ndarray:
    """Perform deduplication on embeddings."""
    logger.info(f"Starting deduplication with threshold {similarity_threshold}")
    logger.info(f"Input embeddings shape: {embeddings.shape}")

    start_time = time.time()

    if steps > 1:
        logger.info(f"Using multi-step deduplication with {steps} steps")
        _, unique_indices, sizes_history = faiss_deduplicate_mr_multistep(
            embeddings=embeddings.astype(np.float32),
            steps_count=steps,
            max_workers=max_workers or os.cpu_count(),
            batch_size=batch_size,
            similarity_threshold=similarity_threshold,
        )

        logger.info("Deduplication steps history:")
        for i, size in enumerate(sizes_history):
            if i == 0:
                logger.info(f"  Step {i} (initial): {size} examples")
            else:
                reduction = sizes_history[i - 1] - size
                reduction_pct = (reduction / sizes_history[i - 1]) * 100
                logger.info(f"  Step {i}: {size} examples (removed {reduction}, -{reduction_pct:.1f}%)")
    else:
        logger.info("Using single-step deduplication")
        _, unique_indices = faiss_deduplicate_mr(
            embeddings=embeddings.astype(np.float32),
            max_workers=max_workers or os.cpu_count(),
            batch_size=batch_size,
            similarity_threshold=similarity_threshold,
        )

    dedup_time = time.time() - start_time
    original_size = len(embeddings)
    final_size = len(unique_indices)
    removed_count = original_size - final_size
    removal_percentage = (removed_count / original_size) * 100

    logger.info(f"Deduplication completed in {dedup_time:.2f}s")
    logger.info(f"Original size: {original_size}")
    logger.info(f"Final size: {final_size}")
    logger.info(f"Removed: {removed_count} ({removal_percentage:.2f}%)")

    return unique_indices


def save_deduplicated_dataset(
    dataset: Dataset,
    unique_indices: np.ndarray,
    output_path: str,
    output_format: str = "jsonl",
    additional_fields: list[str] | None = None,
) -> None:
    """Save the deduplicated dataset."""
    logger.info(f"Saving deduplicated dataset to: {output_path}")

    deduplicated_dataset = dataset.select(unique_indices.tolist())

    if additional_fields is not None:
        columns_to_keep = set(additional_fields)
        # A name that matches no column is a typo, and dropping it silently writes an output missing
        # exactly the column the caller asked to keep — invisible until something downstream reads it.
        unknown = sorted(columns_to_keep - set(deduplicated_dataset.column_names))
        if unknown:
            raise ValueError(
                f"--additional_fields names {unknown}, which the dataset does not carry (available "
                f"columns: {sorted(deduplicated_dataset.column_names)}). Fix the names, or drop them "
                f"— keeping a column that does not exist silently writes an output without it."
            )
        columns_to_remove = set(deduplicated_dataset.column_names) - columns_to_keep
        if columns_to_remove:
            deduplicated_dataset = deduplicated_dataset.remove_columns(list(columns_to_remove))
            logger.info(f"Keeping only specified fields: {additional_fields}")

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_format == "jsonl":
        deduplicated_dataset.to_json(output_path, lines=True)
    elif output_format == "json":
        deduplicated_dataset.to_json(output_path, lines=False)
    elif output_format == "parquet":
        deduplicated_dataset.to_parquet(output_path)
    elif output_format == "csv":
        deduplicated_dataset.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    logger.info(f"Saved {len(deduplicated_dataset)} deduplicated examples")


def main() -> int:
    """Run the deduplication pipeline, returning the process exit code."""
    args = parse_arguments()

    # Attached once the output path is known, so the log lands next to the dataset it describes
    # rather than in whatever directory the script was launched from.
    log_path = Path(args.output_path).with_suffix(".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.getLogger().addHandler(logging.FileHandler(log_path))

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Starting dataset deduplication")
    logger.info(f"Arguments: {vars(args)}")

    try:
        device = setup_device(args.device)

        dataset = load_dataset_from_path(
            input_path=args.input_path,
            text_field=args.text_field,
            dataset_config=args.dataset_config,
            dataset_split=args.dataset_split,
            sample_size=args.sample_size,
        )

        texts = dataset[args.text_field]
        logger.info(f"Extracted {len(texts)} texts from field '{args.text_field}'")

        if args.load_embeddings:
            logger.info(f"Loading precomputed embeddings from: {args.load_embeddings}")
            embeddings = np.load(args.load_embeddings)
            if len(embeddings) != len(texts):
                logger.error(f"Embeddings count ({len(embeddings)}) doesn't match texts count ({len(texts)})")
                return 1
        else:
            embeddings_path = None
            if args.save_embeddings:
                embeddings_path = args.embeddings_output_path or f"{args.output_path}.embeddings.npy"

            embeddings = compute_embeddings(
                texts=texts,
                model_name=args.model_name,
                device=device,
                batch_size=args.batch_size,
                max_length=args.model_max_length,
                save_path=embeddings_path,
                trust_remote_code=args.trust_remote_code,
            )

        unique_indices = deduplicate_embeddings(
            embeddings=embeddings,
            similarity_threshold=args.similarity_threshold,
            batch_size=args.dedup_batch_size,
            steps=args.dedup_steps,
            max_workers=args.max_workers,
        )

        if args.dry_run:
            logger.info("Dry run completed - no output saved")
        else:
            save_deduplicated_dataset(
                dataset=dataset,
                unique_indices=unique_indices,
                output_path=args.output_path,
                output_format=args.output_format,
                additional_fields=args.additional_fields,
            )
    except Exception:
        # One handler for the whole pipeline: every stage announces itself before it can fail, so
        # the preceding log line names the stage and this puts the traceback in the file log above.
        logger.exception("Dataset deduplication failed")
        return 1

    logger.info("Dataset deduplication completed successfully!")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
