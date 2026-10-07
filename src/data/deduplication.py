"""Semantic deduplication of a corpus: mean-pooled sentence embeddings + FAISS similarity search.

Driver behind ``scripts/inference/generation/dataset_deduplication.py``.
"""

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

import faiss
import numpy as np
import torch
from tqdm.auto import tqdm

__all__ = [
    "process_texts",
    "faiss_deduplicate_mr",
    "faiss_deduplicate_mr_multistep",
]


# Fixed default seed: the shuffle decides WHICH member of a near-duplicate group survives dedup, so
# an unseeded global RNG would make two runs over identical data keep different rows.
DEDUP_SHUFFLE_SEED = 42


def _shuffle_matrix_with_mapping(matrix: np.ndarray, rng: np.random.Generator | None = None):
    """Row-shuffle ``matrix``, returning it alongside the permutation that produced it.

    ``rng`` threads a caller's generator through — :func:`faiss_deduplicate_mr_multistep` advances
    ONE generator across its steps, so each step shuffles differently while the whole run stays
    reproducible. ``None`` builds a generator seeded with :data:`DEDUP_SHUFFLE_SEED`.
    """
    generator = rng if rng is not None else np.random.default_rng(DEDUP_SHUFFLE_SEED)
    permuted_indices = generator.permutation(matrix.shape[0])
    shuffled_matrix = matrix[permuted_indices]
    return shuffled_matrix, permuted_indices


def _normalize_embeddings(embeddings: np.array):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Guard zero-norm rows so they don't become inf/nan and corrupt the dedup similarity matrix.
    return embeddings / np.maximum(norms, 1e-12)


def _average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def process_texts(texts, batch_size, model, tokenizer, device, normalize=True):
    embeddings = []
    texts = list(texts)

    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i : i + batch_size]
        batch = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.inference_mode():
            output = model(**batch).last_hidden_state
            pooled_output = _average_pool(output, batch["attention_mask"])
        embeddings.append(pooled_output.detach().cpu())

    embeddings = torch.cat(embeddings, dim=0).numpy()
    if normalize:
        embeddings = _normalize_embeddings(embeddings)
    return embeddings


def _faiss_deduplicate_single(embeddings: np.ndarray, similarity_threshold=0.9) -> np.ndarray:
    """Indices of the rows of ``embeddings`` that survive: each kept row drops every neighbor above
    ``similarity_threshold``."""
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


def faiss_deduplicate_mr(
    embeddings: np.ndarray,
    max_workers=cpu_count(),  # noqa: B008  intentional import-time default
    batch_size=100_000,
    similarity_threshold=0.9,
) -> np.ndarray:
    """Indices of the rows that survive deduplication within each ``batch_size`` block of ``embeddings``,
    in batch order.

    Batch order, not completion order: the next multistep pass shuffles the survivors with the seeded
    generator, and the script selects the output rows in this order, so an order set by which thread
    finished first would make a seeded run keep different rows.
    """
    starts = range(0, embeddings.shape[0], batch_size)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_faiss_deduplicate_single, embeddings[start : start + batch_size], similarity_threshold)
            for start in starts
        ]
        return np.concatenate(
            [
                future.result() + start
                for start, future in zip(starts, tqdm(futures, desc="Processing batches", unit="batch"), strict=True)
            ]
        )


def faiss_deduplicate_mr_multistep(
    embeddings: np.ndarray,
    steps_count=3,
    max_workers=cpu_count(),  # noqa: B008  intentional import-time default
    batch_size=100_000,
    similarity_threshold=0.9,
    seed: int = DEDUP_SHUFFLE_SEED,
) -> tuple[np.ndarray, list[int]]:
    """Indices of the surviving rows of ``embeddings`` after ``steps_count`` reshuffled passes of
    :func:`faiss_deduplicate_mr`, and the row count before and after each pass."""
    # One generator for the whole run: the steps must shuffle differently (that is what re-batches
    # near-duplicates across batch boundaries) while the run as a whole stays reproducible.
    rng = np.random.default_rng(seed)
    progress_indicies_mapping = np.arange(len(embeddings))
    progress_embeddings = embeddings
    sizes_history = [len(embeddings)]

    for _ in tqdm(range(steps_count), desc="Running global dedup step", total=steps_count):
        shuffled_embeddings, shuffled_indices = _shuffle_matrix_with_mapping(progress_embeddings, rng)
        progress_indicies_mapping = progress_indicies_mapping[shuffled_indices]
        shuffled_embeddings = shuffled_embeddings.astype(np.float32)
        unique_indices = faiss_deduplicate_mr(
            shuffled_embeddings,
            max_workers=max_workers,
            batch_size=batch_size,
            similarity_threshold=similarity_threshold,
        )
        progress_embeddings = shuffled_embeddings[unique_indices]
        sizes_history.append(len(progress_embeddings))
        progress_indicies_mapping = progress_indicies_mapping[unique_indices]

    return progress_indicies_mapping, sizes_history
