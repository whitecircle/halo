#!/usr/bin/env python
"""An all-failed inference run must exit non-zero. ``reward_model/score.py`` pre-seeds ``rewards`` from rows a
previous run wrote, so the guard counts the growth and not the length."""

import asyncio
import json
import sys
import types

import pandas as pd
import pytest
from openai.types.chat import ChatCompletionMessage

from halo.entrypoints import reward_scoring
from scripts.inference.reward_model import score as score_script


class _DeadClient:
    def __init__(self):
        async def _create(**_kwargs):
            raise RuntimeError("simulated endpoint failure")

        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=_create))


def _prepare_score(monkeypatch, tmp_path, *, prompt_ids, resumed_rows, client):
    """Stub the scorer's model + endpoint edges; returns (argv, output_path)."""
    prompts = tmp_path / "prompts.jsonl"
    prompts.touch()
    df = pd.DataFrame({"id": prompt_ids, "prompt": [[{"role": "user", "content": f"q{i}"}] for i in prompt_ids]})

    monkeypatch.setattr(reward_scoring, "load_prompts_dataframe", lambda args: df)
    # Both endpoints and the batch go through the shared module, so the stubs go there.
    monkeypatch.setattr(reward_scoring, "build_openai_client", lambda *a, **k: client)
    monkeypatch.setattr(reward_scoring, "load_reward_model", lambda *a, **k: (None, None, None))
    # The reward head is never reached on the failure paths; on the success path it stands in for it.
    monkeypatch.setattr(reward_scoring, "score_conversations", lambda *a, **k: [1.0])

    output_path = reward_scoring.build_output_path(str(tmp_path), str(prompts), "gen-model", "reward_scoring")
    if resumed_rows:
        output_path.write_text(
            "\n".join(json.dumps(row) for row in resumed_rows) + "\n",
            encoding="utf-8",
        )

    argv = [
        "score.py",
        "--model",
        "gen-model",
        "--dataset",
        str(prompts),
        "--rm_model_id",
        "org/rm",
        "--output_dir",
        str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    return output_path


def _resumed(row_id: int, reward: float = 0.5):
    return {"id": row_id, "response": "old", "reward": reward, "full_conversation": []}


def test_score_raises_when_every_pending_row_failed(monkeypatch, tmp_path):
    _prepare_score(monkeypatch, tmp_path, prompt_ids=[1, 2], resumed_rows=[], client=_DeadClient())

    with pytest.raises(RuntimeError, match="No usable result"):
        asyncio.run(score_script.main())


def test_score_resumed_rows_do_not_satisfy_the_guard(monkeypatch, tmp_path):
    """The resumed rows are already in ``rewards``, so a length check would report the previous run's rows as this
    run's output."""
    _prepare_score(monkeypatch, tmp_path, prompt_ids=[1, 2], resumed_rows=[_resumed(1)], client=_DeadClient())

    with pytest.raises(RuntimeError, match="No usable result"):
        asyncio.run(score_script.main())


def test_score_all_rows_already_processed_is_not_a_failure(monkeypatch, tmp_path):
    _prepare_score(monkeypatch, tmp_path, prompt_ids=[1], resumed_rows=[_resumed(1)], client=_DeadClient())

    asyncio.run(score_script.main())


class _LiveClient:
    def __init__(self):
        async def _create(**_kwargs):
            message = ChatCompletionMessage(role="assistant", content="hi")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=_create))


def test_score_one_scored_row_is_the_other_side_of_the_boundary(monkeypatch, tmp_path):
    output_path = _prepare_score(
        monkeypatch, tmp_path, prompt_ids=[1, 2], resumed_rows=[_resumed(1)], client=_LiveClient()
    )

    asyncio.run(score_script.main())

    assert len(pd.read_json(output_path, lines=True)) == 2, "the resumed row plus the one scored now"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
