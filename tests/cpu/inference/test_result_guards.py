#!/usr/bin/env python
"""An all-failed inference run must exit non-zero, on every script that writes a batch of results.

The script here reached the end of a totally failed run and reported success:

* ``reward_model/rm_scoring.py`` cannot guard on ``if not rewards``: ``rewards`` is pre-seeded from
  the rows a previous run already wrote, so one resumed row makes the list truthy and the guard can
  only ever fire on a cold start — the run least likely to be silently misconfigured. The count
  that decides whether THIS run produced anything is the growth, not the length.

Run: pytest tests/cpu/inference/test_result_guards.py
"""

import asyncio
import json
import sys
import types

import pandas as pd
import pytest

from scripts.inference.reward_model import _common as rm_common
from scripts.inference.reward_model import rm_scoring

# --- rm_scoring: reward-model scoring ------------------------------------------------------------


class _DeadClient:
    """API client whose every generation call fails, as a wrong --model_name or dead endpoint does."""

    def __init__(self):
        async def _create(**_kwargs):
            raise RuntimeError("simulated endpoint failure")

        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=_create))


def _prepare_rm_scoring(monkeypatch, tmp_path, *, prompt_ids, resumed_rows, client):
    """Stub rm_scoring's model + endpoint edges; returns (argv, output_path)."""
    prompts = tmp_path / "prompts.jsonl"
    prompts.touch()
    df = pd.DataFrame({"id": prompt_ids, "prompt": [[{"role": "user", "content": f"q{i}"}] for i in prompt_ids]})

    monkeypatch.setattr(rm_scoring, "load_prompts_dataframe", lambda args: df)
    # Both endpoints are opened, and the batch is scored, through the shared module — that is where
    # the stubs go, or the script would boot a real client and load a real reward model.
    monkeypatch.setattr(rm_common, "create_openai_client", lambda *a, **k: client)
    monkeypatch.setattr(rm_common, "load_reward_model", lambda *a, **k: (None, None, None))
    # The reward head is never reached on the failure paths; on the success path it stands in for it.
    monkeypatch.setattr(rm_common, "score_conversations", lambda *a, **k: [1.0])

    output_path = rm_common.build_output_path(str(tmp_path), str(prompts), "gen-model", "rm_scoring")
    if resumed_rows:
        output_path.write_text(
            "\n".join(json.dumps(row) for row in resumed_rows) + "\n",
            encoding="utf-8",
        )

    argv = [
        "rm_scoring.py",
        "--model_name",
        "gen-model",
        "--prompts_source",
        str(prompts),
        "--rm_model_path",
        "org/rm",
        "--output_folder",
        str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    return output_path


def _resumed(row_id: int, reward: float = 0.5):
    return {"id": row_id, "response": "old", "reward": reward, "full_conversation": []}


def test_rm_scoring_raises_when_every_pending_row_failed(monkeypatch, tmp_path):
    """Cold run against a dead endpoint: nothing scored, so the run is a failure."""
    _prepare_rm_scoring(monkeypatch, tmp_path, prompt_ids=[1, 2], resumed_rows=[], client=_DeadClient())

    with pytest.raises(RuntimeError, match="No usable result"):
        asyncio.run(rm_scoring.main())


def test_rm_scoring_resumed_rows_do_not_satisfy_the_guard(monkeypatch, tmp_path):
    """THE regression: the resumed rows are already in `rewards`, so a length check passes and the
    run reports the PREVIOUS run's rows as its own output. Every pending row failed here — a resume
    is not exempt from the guard just because an earlier run left something behind."""
    _prepare_rm_scoring(monkeypatch, tmp_path, prompt_ids=[1, 2], resumed_rows=[_resumed(1)], client=_DeadClient())

    with pytest.raises(RuntimeError, match="No usable result"):
        asyncio.run(rm_scoring.main())


def test_rm_scoring_all_rows_already_processed_is_not_a_failure(monkeypatch, tmp_path):
    """Nothing pending means nothing to produce — a completed run re-run must stay a success."""
    _prepare_rm_scoring(monkeypatch, tmp_path, prompt_ids=[1], resumed_rows=[_resumed(1)], client=_DeadClient())

    asyncio.run(rm_scoring.main())


class _LiveClient:
    """API client that answers every generation call."""

    def __init__(self):
        async def _create(**_kwargs):
            message = types.SimpleNamespace(model_dump=lambda exclude=None: {"role": "assistant", "content": "hi"})
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=_create))


def test_rm_scoring_one_scored_row_is_the_other_side_of_the_boundary(monkeypatch, tmp_path):
    """The guard keys on producing NOTHING, so a resume that scores its pending rows still passes."""
    output_path = _prepare_rm_scoring(
        monkeypatch, tmp_path, prompt_ids=[1, 2], resumed_rows=[_resumed(1)], client=_LiveClient()
    )

    asyncio.run(rm_scoring.main())

    assert len(pd.read_json(output_path, lines=True)) == 2, "the resumed row plus the one scored now"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
