"""Semantic deduplication of a corpus (mean-pooled sentence embeddings + FAISS similarity search), behind
``scripts/inference/deduplicate_dataset.py``."""

import argparse
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import faiss
import numpy as np
import torch
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from halo.data.sources.loading import load_dataset_split
from halo.hardware import available_cpu_count
from halo.model_loading.checkpoint_coverage import from_pretrained_verified
from halo.model_loading.finalization import finalize_loaded_model

logger = logging.getLogger(__name__)

# These decide which rows are read and which member of a near-duplicate group survives, so an unseeded RNG would make
# two runs over identical data keep different rows.
DEDUP_SHUFFLE_SEED = 42

OUTPUT_WRITERS: dict[str, Callable[[Dataset, str], object]] = {
    "jsonl": lambda dataset, path: dataset.to_json(path, lines=True),
    "json": lambda dataset, path: dataset.to_json(path, lines=False),
    "parquet": lambda dataset, path: dataset.to_parquet(path),
    "csv": lambda dataset, path: dataset.to_csv(path, index=False),
}


def _shuffle_matrix_with_mapping(matrix: np.ndarray, rng: np.random.Generator | None = None):
    """Row-shuffle ``matrix``, returning the permutation too. :func:`deduplicate_embeddings` threads one generator
    across its steps, so each shuffles differently while the run stays reproducible."""
    generator = rng if rng is not None else np.random.default_rng(DEDUP_SHUFFLE_SEED)
    permuted_indices = generator.permutation(matrix.shape[0])
    shuffled_matrix = matrix[permuted_indices]
    return shuffled_matrix, permuted_indices


def _normalize_embeddings(embeddings: np.array):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Guard zero-norm rows so they don't become inf/nan and corrupt the dedup similarity matrix.
    return embeddings / np.maximum(norms, 1e-12)


def embed_texts(texts, batch_size, model, tokenizer, device, normalize=True):
    """Mean-pooled over the attention mask, row-normalized by default."""
    embeddings = []
    texts = list(texts)

    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i : i + batch_size]
        batch = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
        batch = {k: v.to(device) for k, v in batch.items()}
        mask = batch["attention_mask"]
        with torch.inference_mode():
            hidden = model(**batch).last_hidden_state.masked_fill(~mask[..., None].bool(), 0.0)
            pooled_output = hidden.sum(dim=1) / mask.sum(dim=1)[..., None]
        embeddings.append(pooled_output.detach().cpu())

    embeddings = torch.cat(embeddings, dim=0).numpy()
    if normalize:
        embeddings = _normalize_embeddings(embeddings)
    return embeddings


def _deduplicate_block(embeddings: np.ndarray, similarity_threshold: float) -> np.ndarray:
    """Surviving row indices: each kept row drops every neighbor above ``similarity_threshold``."""
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    result = index.range_search(embeddings, similarity_threshold)
    # lims: result ranges per query; indices: neighbor ids (distances unused).
    lims, _, indices = result

    keep = np.ones(len(embeddings), dtype=bool)
    visited = np.zeros(len(embeddings), dtype=bool)

    for i in range(len(embeddings)):
        if visited[i]:
            continue

        start_idx, end_idx = lims[i], lims[i + 1]
        neighbors = indices[start_idx:end_idx]

        neighbors = neighbors[neighbors != i]

        visited[neighbors] = True
        keep[neighbors] = False

    return np.where(keep)[0]


def deduplicate_blocks(
    embeddings: np.ndarray,
    *,
    batch_size: int,
    similarity_threshold: float,
    max_workers: int | None = None,
) -> np.ndarray:
    """Survivors within each ``batch_size`` block, in batch order: the next pass shuffles them with the seeded
    generator, and an order set by which thread finished first would make a seeded run keep different rows."""
    starts = range(0, embeddings.shape[0], batch_size)
    with ThreadPoolExecutor(max_workers=max_workers or available_cpu_count()) as executor:
        futures = [
            executor.submit(_deduplicate_block, embeddings[start : start + batch_size], similarity_threshold)
            for start in starts
        ]
        return np.concatenate(
            [
                future.result() + start
                for start, future in zip(starts, tqdm(futures, desc="Processing batches", unit="batch"), strict=True)
            ]
        )


def deduplicate_embeddings(
    embeddings: np.ndarray,
    *,
    steps_count: int,
    batch_size: int,
    similarity_threshold: float,
    max_workers: int | None = None,
    seed: int = DEDUP_SHUFFLE_SEED,
) -> tuple[np.ndarray, list[int]]:
    """Surviving rows after ``steps_count`` reshuffled passes, and the row count before and after each pass."""
    # One generator: the steps must shuffle differently to re-batch near-duplicates, while the run stays reproducible.
    rng = np.random.default_rng(seed)
    surviving_indices = np.arange(len(embeddings))
    progress_embeddings = embeddings
    sizes_history = [len(embeddings)]

    for _ in tqdm(range(steps_count), desc="Running global dedup step", total=steps_count):
        shuffled_embeddings, shuffled_indices = _shuffle_matrix_with_mapping(progress_embeddings, rng)
        surviving_indices = surviving_indices[shuffled_indices]
        shuffled_embeddings = shuffled_embeddings.astype(np.float32)
        unique_indices = deduplicate_blocks(
            shuffled_embeddings,
            max_workers=max_workers,
            batch_size=batch_size,
            similarity_threshold=similarity_threshold,
        )
        progress_embeddings = shuffled_embeddings[unique_indices]
        sizes_history.append(len(progress_embeddings))
        surviving_indices = surviving_indices[unique_indices]

    return surviving_indices, sizes_history


def resolve_device(device_arg: str) -> torch.device:
    """``auto`` is CUDA when a GPU is visible, else CPU; any other value goes to :class:`torch.device`."""
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
    dataset_config: str | None,
    dataset_split: str,
    sample_size: int | None,
) -> Dataset:
    """Load the input split with ``text_field`` checked and an optional seeded sample of ``sample_size`` rows."""
    logger.info(f"Loading dataset from: {input_path}")
    dataset = load_dataset_split(input_path, split=dataset_split, config=dataset_config)

    if text_field not in dataset.column_names:
        raise ValueError(f"Text field '{text_field}' not found in dataset. Available fields: {dataset.column_names}")

    if sample_size is not None and len(dataset) > sample_size:
        logger.info(f"Sampling {sample_size} examples from {len(dataset)} total")
        dataset = dataset.shuffle(seed=DEDUP_SHUFFLE_SEED).select(range(sample_size))

    logger.info(f"Loaded dataset with {len(dataset)} examples")
    logger.info(f"Dataset columns: {dataset.column_names}")

    return dataset


def compute_embeddings(
    texts: list[str],
    model_name: str,
    device: torch.device,
    batch_size: int,
    max_length: int,
    save_path: str | None,
    trust_remote_code: bool,
) -> np.ndarray:
    logger.info(f"Computing embeddings using model: {model_name}")
    logger.info(f"Processing {len(texts)} texts with batch size {batch_size}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    # Gated: an embedding checkpoint with random tensors still returns vectors, and the dedup decisions look ordinary.
    model = from_pretrained_verified(AutoModel, model_name, trust_remote_code=trust_remote_code)
    model.to(device)
    # Non-persistent buffers come back uninitialized on transformers 5, leaving a dead RoPE. Run after placement.
    finalize_loaded_model(model)
    model.eval()
    tokenizer.model_max_length = max_length

    start_time = time.time()
    embeddings = embed_texts(
        texts=texts, batch_size=batch_size, model=model, tokenizer=tokenizer, device=device, normalize=True
    )

    computation_time = time.time() - start_time
    logger.info(f"Computed {len(embeddings)} embeddings in {computation_time:.2f}s")
    logger.info(f"Embedding shape: {embeddings.shape}")

    if save_path:
        logger.info(f"Saving embeddings to: {save_path}")
        np.save(save_path, embeddings)

    return embeddings


def select_unique_rows(
    embeddings: np.ndarray,
    similarity_threshold: float,
    batch_size: int,
    steps: int,
    max_workers: int | None,
) -> np.ndarray:
    """One pass of :func:`deduplicate_blocks`, or ``steps`` reshuffled passes of :func:`deduplicate_embeddings`."""
    logger.info(f"Starting deduplication with threshold {similarity_threshold}")
    logger.info(f"Input embeddings shape: {embeddings.shape}")

    start_time = time.time()

    if steps > 1:
        logger.info(f"Using multi-step deduplication with {steps} steps")
        unique_indices, sizes_history = deduplicate_embeddings(
            embeddings=embeddings.astype(np.float32),
            steps_count=steps,
            max_workers=max_workers,
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
        unique_indices = deduplicate_blocks(
            embeddings=embeddings.astype(np.float32),
            max_workers=max_workers,
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
    logger.info(f"Saving deduplicated dataset to: {output_path}")

    deduplicated_dataset = dataset.select(unique_indices.tolist())

    if additional_fields is not None:
        columns_to_keep = set(additional_fields)
        # A name matching no column is a typo; dropping it would silently lose a column the caller asked to keep.
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

    OUTPUT_WRITERS[output_format](deduplicated_dataset, output_path)

    logger.info(f"Saved {len(deduplicated_dataset)} deduplicated examples")


def run_deduplication(args: argparse.Namespace) -> int:
    """Returns the process exit code."""
    logger.info("Starting dataset deduplication")
    logger.info(f"Arguments: {vars(args)}")

    try:
        device = resolve_device(args.device)

        dataset = load_dataset_from_path(
            input_path=args.dataset,
            text_field=args.text_field,
            dataset_config=args.config,
            dataset_split=args.split,
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
                model_name=args.model_id,
                device=device,
                batch_size=args.batch_size,
                max_length=args.model_max_length,
                save_path=embeddings_path,
                trust_remote_code=args.trust_remote_code,
            )

        unique_indices = select_unique_rows(
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
        # One handler for the pipeline: every stage logs before it can fail, so the preceding line names the stage.
        logger.exception("Dataset deduplication failed")
        return 1

    logger.info("Dataset deduplication completed successfully!")
    return 0
