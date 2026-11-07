"""Stand-ins for the rank-balance suites: a skewed text round per rank, and the online trainer's real loss on it."""

import torch

from tests.common.length_deal import ragged_round, with_world_normalizer

VOCAB = 64
ROWS_PER_RANK = 4


def skewed_round(rank: int, nprocs: int, *, seed: int = 3) -> dict:
    """Rank 0 holds the longest rows (7x the last rank's); every row tells itself apart by its first prompt
    token and its advantage. Its widths differ per rank, as the rollout's padding to the local longest row makes them."""
    scale = 1 + 6 * (nprocs - 1 - rank) // (nprocs - 1)
    gen = torch.Generator().manual_seed(seed + rank)
    prompts = (torch.randint(2, 4, (ROWS_PER_RANK,), generator=gen) * scale).tolist()
    completions = (torch.randint(2, 6, (ROWS_PER_RANK,), generator=gen) * scale).tolist()
    batch = ragged_round(prompts, completions, vocab=VOCAB, seed=seed + rank)
    batch.pop("row_id")
    batch["advantages"] = torch.arange(ROWS_PER_RANK, dtype=torch.float32) + 10 * rank
    batch["old_per_token_logps"] = -torch.rand_like(batch["importance_sampling_ratio"]) * batch["completion_mask"]
    return batch


def reversed_order_round(ctx, row_tokens: list[int], vocab: int) -> dict:
    """Each rank's rows in the opposite order on odd ranks, so the ranks' bins differ in content and, with an uneven
    total, in size."""
    lengths = row_tokens if ctx.rank % 2 == 0 else row_tokens[::-1]
    halves = [n // 2 for n in lengths]
    completions = [n - h for n, h in zip(lengths, halves, strict=True)]
    return with_world_normalizer(ragged_round(halves, completions, vocab=vocab, seed=ctx.rank, device=ctx.device))


def row_tokens(batch: dict) -> torch.Tensor:
    return batch["prompt_mask"].sum(dim=1) + batch["completion_mask"].sum(dim=1)


def row_contents(batch: dict) -> dict[float, tuple]:
    """Each row's real tokens and per-token values, keyed by its advantage, with the padding stripped."""
    prompt, completion = batch["prompt_mask"].bool(), batch["completion_mask"].bool()
    return {
        float(batch["advantages"][i]): (
            batch["prompt_ids"][i][prompt[i]].tolist(),
            batch["completion_ids"][i][completion[i]].tolist(),
            batch["old_per_token_logps"][i][completion[i]].tolist(),
            batch["importance_sampling_ratio"][i][completion[i]].tolist(),
        )
        for i in range(batch["advantages"].numel())
    }
