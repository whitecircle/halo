"""Semantic deduplication of a corpus (mean-pooled sentence embeddings + FAISS similarity search), behind
``scripts/before_training/deduplicate_dataset.py``."""

import argparse
import logging
import tempfile
import time
from collections.abc import Callable
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

# Rows one similarity search queries at once: its result holds every pair over the threshold, so a block of K identical
# rows searched whole would return K squared pairs.
_SEARCH_QUERY_ROWS = 4096

OUTPUT_WRITERS: dict[str, Callable[[Dataset, str], object]] = {
    "jsonl": lambda dataset, path: dataset.to_json(path, lines=True),
    "json": lambda dataset, path: dataset.to_json(path, lines=False),
    "parquet": lambda dataset, path: dataset.to_parquet(path),
    "csv": lambda dataset, path: dataset.to_csv(path, index=False),
}


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
    """Surviving row indices: each kept row, in order, drops every neighbor above ``similarity_threshold``. Rows are
    queried :data:`_SEARCH_QUERY_ROWS` at a time, those already dropped left out."""
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    keep = np.ones(len(embeddings), dtype=bool)
    for start in range(0, len(embeddings), _SEARCH_QUERY_ROWS):
        queries = np.flatnonzero(keep[start : start + _SEARCH_QUERY_ROWS]) + start
        if not len(queries):
            continue
        # lims: result ranges per query; indices: neighbor ids (distances unused).
        lims, _, indices = index.range_search(embeddings[queries], similarity_threshold)
        for position, row in enumerate(queries):
            if keep[row]:
                neighbors = indices[lims[position] : lims[position + 1]]
                keep[neighbors[neighbors != row]] = False

    return np.flatnonzero(keep)


def deduplicate_blocks(
    embeddings: np.ndarray, rows: np.ndarray, *, batch_size: int, similarity_threshold: float
) -> np.ndarray:
    """The ``rows`` of ``embeddings`` that survive within their ``batch_size`` block of ``rows``, in ``rows`` order: a
    pair in different blocks is never compared. One block at a time, each search on every OpenMP thread."""
    blocks = [rows[start : start + batch_size] for start in range(0, len(rows), batch_size)]
    return np.concatenate(
        [
            block[_deduplicate_block(embeddings[block], similarity_threshold)]
            for block in tqdm(blocks, desc="Processing batches", unit="batch")
        ]
    )


def deduplicate_embeddings(
    embeddings: np.ndarray,
    *,
    steps_count: int,
    batch_size: int,
    similarity_threshold: float,
    seed: int = DEDUP_SHUFFLE_SEED,
) -> tuple[np.ndarray, list[int]]:
    """Surviving rows in input order after ``steps_count`` passes, each over the survivors reshuffled into new blocks,
    and the row count before and after each pass."""
    # One generator: the steps must shuffle differently to re-batch near-duplicates, while the run stays reproducible.
    rng = np.random.default_rng(seed)
    surviving = np.arange(len(embeddings))
    sizes_history = [len(embeddings)]

    for _ in tqdm(range(steps_count), desc="Running global dedup step", total=steps_count):
        surviving = deduplicate_blocks(
            embeddings,
            surviving[rng.permutation(len(surviving))],
            batch_size=batch_size,
            similarity_threshold=similarity_threshold,
        )
        sizes_history.append(len(surviving))

    return np.sort(surviving), sizes_history


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


def load_dedup_input(
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
) -> np.ndarray:
    """One pass of :func:`deduplicate_blocks` over the rows in order, or ``steps`` reshuffled passes of
    :func:`deduplicate_embeddings`."""
    logger.info(f"Starting deduplication with threshold {similarity_threshold}")
    logger.info(f"Input embeddings shape: {embeddings.shape}")
    embeddings = np.asarray(embeddings, dtype=np.float32)  # no copy when already float32
    if len(embeddings) > batch_size:
        met = 1 - (1 - batch_size / len(embeddings)) ** steps
        logger.warning(
            f"Rows are compared only within blocks of {batch_size}: a duplicate pair at random positions shares one "
            f"in {steps} pass(es) with probability about {met:.0%}, and otherwise both rows survive. Raise "
            f"--dedup_batch_size or --dedup_steps to compare more pairs."
        )

    start_time = time.time()

    if steps > 1:
        logger.info(f"Using multi-step deduplication with {steps} steps")
        unique_indices, sizes_history = deduplicate_embeddings(
            embeddings=embeddings,
            steps_count=steps,
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
            embeddings,
            np.arange(len(embeddings)),
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
    keep_fields: list[str] | None = None,
) -> None:
    logger.info(f"Saving deduplicated dataset to: {output_path}")

    deduplicated_dataset = dataset.select(unique_indices.tolist())

    if keep_fields is not None:
        columns_to_keep = set(keep_fields)
        # A name matching no column is a typo; dropping it would silently lose a column the caller asked to keep.
        unknown = sorted(columns_to_keep - set(deduplicated_dataset.column_names))
        if unknown:
            raise ValueError(
                f"--keep_fields names {unknown}, which the dataset does not carry (available "
                f"columns: {sorted(deduplicated_dataset.column_names)}). Fix the names, or drop them "
                f"— keeping a column that does not exist silently writes an output without it."
            )
        columns_to_remove = set(deduplicated_dataset.column_names) - columns_to_keep
        if columns_to_remove:
            deduplicated_dataset = deduplicated_dataset.remove_columns(list(columns_to_remove))
            logger.info(f"Keeping only specified fields: {keep_fields}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Written under its own name, so the writer infers the same compression, then renamed over the output: a run killed
    # mid-write would leave whole lines that load as a smaller dataset.
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=f".{output.name}.") as staging:
        staged = Path(staging) / output.name
        OUTPUT_WRITERS[output_format](deduplicated_dataset, str(staged))
        staged.replace(output)

    logger.info(f"Saved {len(deduplicated_dataset)} deduplicated examples")


def run_deduplication(args: argparse.Namespace) -> int:
    """Returns the process exit code."""
    logger.info("Starting dataset deduplication")
    logger.info(f"Arguments: {vars(args)}")

    # faiss sizes its pool by the host's cores, not a container's CPU quota.
    faiss.omp_set_num_threads(args.max_workers or available_cpu_count())
    try:
        device = resolve_device(args.device)

        dataset = load_dedup_input(
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
            embeddings = compute_embeddings(
                texts=texts,
                model_name=args.model_id,
                device=device,
                batch_size=args.batch_size,
                max_length=args.model_max_length,
                save_path=args.save_embeddings,
                trust_remote_code=args.trust_remote_code,
            )

        unique_indices = select_unique_rows(
            embeddings=embeddings,
            similarity_threshold=args.similarity_threshold,
            batch_size=args.dedup_batch_size,
            steps=args.dedup_steps,
        )

        if args.dry_run:
            logger.info("Dry run completed - no output saved")
        else:
            save_deduplicated_dataset(
                dataset=dataset,
                unique_indices=unique_indices,
                output_path=args.output_path,
                output_format=args.output_format,
                keep_fields=args.keep_fields,
            )
    except Exception:
        # One handler for the pipeline: every stage logs before it can fail, so the preceding line names the stage.
        logger.exception("Dataset deduplication failed")
        return 1

    logger.info("Dataset deduplication completed successfully!")
    return 0
