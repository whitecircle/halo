"""Sizes one rollout rank's Ray actor pool and in-flight cap from the run's actor budget and round shape."""

from dataclasses import dataclass

# Below this a rank's episodes queue behind a handful of actors, and the pool starts to bound a round.
MIN_ACTORS_PER_RANK = 4


@dataclass(frozen=True)
class RolloutPoolPlan:
    actors_per_rank: int
    max_concurrent: int
    warnings: tuple[str, ...]


def resolve_rollout_pool(
    *,
    num_workers: int,
    shared_cluster: bool,
    rollout_ranks: int,
    rollout_ranks_per_node: int,
    rows_per_round: int,
    eval_rows: int | None,
    max_concurrent_rollouts: int | None,
) -> RolloutPoolPlan:
    """The budget ``num_workers`` is per node on a local Ray (split over that node's rollout ranks) and cluster-wide on a
    shared one (split over every rollout rank); a rank gets at least one actor. An unset ``max_concurrent_rollouts``
    is the larger of a train and an eval round, so a round runs in one wave. The returned ``warnings`` name a split
    that leaves a rank thin, an explicit cap below the actor count (raised to it) and one below a round (waves)."""
    warnings: list[str] = []
    if shared_cluster:
        scope, divisor = "the shared Ray cluster", rollout_ranks
    else:
        scope, divisor = "each node's local Ray", rollout_ranks_per_node
    actors = max(1, num_workers // divisor)
    if actors < min(MIN_ACTORS_PER_RANK, rows_per_round):
        warnings.append(
            f"num_rollout_workers ({num_workers}) over {divisor} rollout ranks of {scope} leaves {actors} actor(s) "
            f"per rank, below {MIN_ACTORS_PER_RANK}; raise it to at least {MIN_ACTORS_PER_RANK * divisor}."
        )

    if max_concurrent_rollouts is None:
        cap = max(rows_per_round, eval_rows or 0, actors)
    else:
        cap = max_concurrent_rollouts
        if cap < actors:
            warnings.append(f"max_concurrent_rollouts ({cap}) < actors per rank ({actors}), raised to {actors}.")
            cap = actors
        if cap < rows_per_round:
            warnings.append(
                f"max_concurrent_rollouts ({cap}) is below the {rows_per_round} rows of a round: it runs in "
                f"{-(-rows_per_round // cap)} waves, each able to take an episode_timeout."
            )
    return RolloutPoolPlan(actors_per_rank=actors, max_concurrent=cap, warnings=tuple(warnings))
