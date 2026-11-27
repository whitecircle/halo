"""Offline-GRPO rewards precomputed by a config's reward terms through the TRL reward functions the online arm trains
with, so a term prices a completion offline as it does online."""

import inspect
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from halo.rewards.graders.verifiable import RLVR_GRADERS, RLVR_REWARD_SOURCES, AccuracyTerm
from halo.rewards.terms import parse_reward_terms, read_context_files
from halo.rewards.trl_reward_functions import ScorerRewardFunction, build_trl_reward_functions

logger = logging.getLogger(__name__)

# Rows scored together, then written: the scorers' concurrency spans a batch, and a failure loses one batch at most.
ROWS_PER_BATCH = 64
# The keyword the RLVR graders and a scorer's reference read the ground truth from.
_ANSWER = "answer"


def build_reward_functions(config_path: str | Path, *, with_reference: bool) -> tuple[list[Callable], list[float]]:
    """The ``rewards`` list of a training YAML (an online-GRPO recipe's included) as TRL reward functions. An
    ``accuracy`` term without the reference raises before any scorer is built."""
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    if "rewards" not in config:
        raise ValueError(f"{config_path} has no 'rewards' list of reward terms")
    terms = parse_reward_terms(read_context_files(config["rewards"]), RLVR_REWARD_SOURCES)
    if not with_reference and any(isinstance(term, AccuracyTerm) for term in terms):
        raise ValueError(
            f"{config_path} has an accuracy reward term, which compares each completion with its row's answer, and "
            "no --answer_field names the row field holding it; pass --answer_field <field>"
        )
    return build_trl_reward_functions(terms, RLVR_GRADERS, reference_column=_ANSWER if with_reference else None)


async def score_rows(
    rows: Sequence[Mapping[str, Any]],
    functions: Sequence[Callable],
    weights: Sequence[float],
    *,
    answer_field: str | None,
) -> tuple[list[dict[str, Any]], int]:
    """Each row with its ``rewards``, per completion the weighted sum of its terms TRL trains on, and the count of
    completions dropped: one a term left without a score is not written, so no outage becomes a reward. A row left with
    no completion is dropped."""
    pairs = [(row, completion) for row in rows for completion in row["completions"]]
    kwargs: dict[str, Any] = {"prompts": [row["prompt"] for row, _ in pairs], "completions": [c for _, c in pairs]}
    if answer_field is not None:
        kwargs[_ANSWER] = [row[answer_field] for row, _ in pairs]
    columns = []
    for function in functions:
        values = function(**kwargs)
        columns.append(await values if inspect.isawaitable(values) else values)
    per_completion = iter(zip(*columns, strict=True))
    scored, dropped = [], 0
    for row in rows:
        kept, rewards = [], []
        for completion in row["completions"]:
            values = next(per_completion)
            if any(value is None for value in values):
                dropped += 1
                continue
            kept.append(completion)
            rewards.append(sum(weight * value for weight, value in zip(weights, values, strict=True)))
        if kept:
            scored.append({**row, "completions": kept, "rewards": rewards})
    return scored, dropped


async def run_offline_reward_scoring(
    config_path: str | Path, input_path: str | Path, output_path: str | Path, *, answer_field: str | None
) -> None:
    """Probes every external term, then writes each row of ``input_path`` (``prompt`` and ``completions``) to
    ``output_path`` with its ``rewards``. Raises when no row is left."""
    functions, weights = build_reward_functions(config_path, with_reference=answer_field is not None)
    scorers = [function.scorer for function in functions if isinstance(function, ScorerRewardFunction)]
    for scorer in scorers:
        await scorer.verify()
    lines = Path(input_path).read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    written = dropped = 0
    try:
        with open(output_path, "w", encoding="utf-8") as out:
            for start in range(0, len(rows), ROWS_PER_BATCH):
                scored, batch_dropped = await score_rows(
                    rows[start : start + ROWS_PER_BATCH], functions, weights, answer_field=answer_field
                )
                out.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in scored)
                written += len(scored)
                dropped += batch_dropped
    finally:
        for scorer in scorers:
            await scorer.aclose()
    logger.info(f"Wrote {written} of {len(rows)} rows to {output_path}; {dropped} unscored completion(s) dropped")
    if not written:
        raise RuntimeError(f"no row of {input_path} kept a scored completion; check the reward terms' backends")
