#!/usr/bin/env python
"""Offline GRPO refuses a prompt that tokenizes to nothing, at dataset map time, on every rank.

Its completions would otherwise train against a context the data never held. The refusal lives in
the tokenize map rather than the collator: the map runs inside ``coordinated_map``, whose failure
join raises on every rank, where a collator raise on the one rank that drew the row would leave its
peers in the next collective.

    python tests/cpu/grpo/test_offline_grpo_empty_prompt.py
"""

import datetime
import os

import pytest
from datasets import Dataset

from src.data.pipeline.processing import coordinated_map
from src.trainers.grpo.offline import tokenize_offline_grpo_rows
from tests.common.gloo import run_gloo_ranks

WORLD_SIZE = 2
# Far below a real barrier wait: a rank left hanging fails the test instead of stalling the suite.
PG_TIMEOUT_SEC = 15
EOS_ID = 900
EMPTY_ROW = 1


class _WordTokenizer:
    """One id per whitespace word; no BOS of its own, so an empty prompt stays empty."""

    bos_token = None
    bos_token_id = None
    eos_token_id = EOS_ID

    def __call__(self, text, add_special_tokens=True, truncation=False, max_length=None, padding=False):
        ids = [100 + i for i, _ in enumerate(text.split())]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def decode(self, ids):
        return " ".join(f"w{i}" for i in ids)


def _rows() -> dict:
    return {
        "prompt": ["a real prompt", "", "another prompt"],
        "completions": [["yes", "no"], ["yes", "no"], ["yes", "no"]],
        "rewards": [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    }


def _map_kwargs() -> dict:
    return {
        "processing_class": _WordTokenizer(),
        "max_prompt_length": None,
        "max_completion_length": None,
        "advantage_method": "z_norm",
        "best_completion_emphasis": 1.0,
    }


def test_an_empty_prompt_is_refused_with_its_row_index():
    rows = _rows()
    with pytest.raises(ValueError, match=f"row {EMPTY_ROW}.*no tokens"):
        tokenize_offline_grpo_rows(rows, list(range(len(rows["prompt"]))), **_map_kwargs())


def test_non_empty_prompts_still_tokenize():
    rows = {key: [value[0]] for key, value in _rows().items()}
    out = tokenize_offline_grpo_rows(rows, [0], **_map_kwargs())
    assert out["prompt_input_ids"] == [[100, 101, 102]] * 2


def _worker(rank: int, tmp_dir: str) -> None:
    try:
        coordinated_map(
            Dataset.from_dict(_rows()),
            tokenize_offline_grpo_rows,
            num_proc=1,
            remove_columns=["prompt", "completions", "rewards"],
            desc="empty-prompt refusal",
            batched=True,
            with_indices=True,
            fn_kwargs=_map_kwargs(),
        )
        outcome = "NO RAISE"
    except Exception as e:
        outcome = f"{type(e).__name__}: {e}"
    with open(os.path.join(tmp_dir, f"result_{rank}.txt"), "w") as fh:
        fh.write(outcome)


def test_every_rank_raises_the_refusal_during_the_coordinated_map(tmp_path):
    run_gloo_ranks(
        _worker,
        WORLD_SIZE,
        str(tmp_path),
        pg_timeout=datetime.timedelta(seconds=PG_TIMEOUT_SEC),
        env={"HF_DATASETS_CACHE": str(tmp_path / "hf_datasets")},
    )
    for rank in range(WORLD_SIZE):
        result = (tmp_path / f"result_{rank}.txt").read_text()
        assert "timed out" not in result and "Timeout" not in result, f"rank {rank} only saw a timeout: {result}"
        assert f"row {EMPTY_ROW}" in result and "no tokens" in result, f"rank {rank}: {result}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
