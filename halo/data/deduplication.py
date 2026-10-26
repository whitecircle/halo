"""Semantic deduplication of a corpus: mean-pooled sentence embeddings + FAISS similarity search, the driver
behind ``scripts/inference/deduplicate_dataset.py``."""

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

import faiss
import numpy as np
import torch
from tqdm.auto import tqdm

# Fixed default seed: the shuffle decides WHICH member of a near-duplicate group survives dedup, so
# an unseeded global RNG would make two runs over identical data keep different rows.
DEDUP_SHUFFLE_SEED = 42


def _shuffle_matrix_with_mapping(matrix: np.ndarray, rng: np.random.Generator | None = None):
    """Row-shuffle ``matrix``, returning it alongside the permutation that produced it.

    ``rng`` threads a caller's generator through — :func:`deduplicate_embeddings` advances
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


def embed_texts(texts, batch_size, model, tokenizer, device, normalize=True):
    """Mean-pooled (over the attention mask) last hidden states of ``texts``, row-normalized by default."""
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


def _deduplicate_block(embeddings: np.ndarray, similarity_threshold=0.9) -> np.ndarray:
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


def deduplicate_blocks(
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
    steps_count=3,
    max_workers=cpu_count(),  # noqa: B008  intentional import-time default
    batch_size=100_000,
    similarity_threshold=0.9,
    seed: int = DEDUP_SHUFFLE_SEED,
) -> tuple[np.ndarray, list[int]]:
    """Indices of the surviving rows of ``embeddings`` after ``steps_count`` reshuffled passes of
    :func:`deduplicate_blocks`, and the row count before and after each pass."""
    # One generator for the whole run: the steps must shuffle differently (that is what re-batches
    # near-duplicates across batch boundaries) while the run as a whole stays reproducible.
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
