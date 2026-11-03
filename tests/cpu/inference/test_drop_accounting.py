#!/usr/bin/env python
"""Every dropped row must be counted, and an interrupted run must not exit 0."""

import argparse
import asyncio
import json
import logging
import signal
import sys
import types

import pandas as pd
import pytest
from openai.types.chat import ChatCompletionMessage

from halo.entrypoints import rejection_sampling, resumable_generation, reward_scoring
from halo.inference.openai_client import EmptyChoicesError
from scripts.inference import inference_arguments
from scripts.inference.reward_model import score as score_script


def _prompt_rows(count: int) -> list[dict]:
    return [{"id": i, "prompt": [{"role": "user", "content": f"q{i}"}]} for i in range(count)]


class _DeadClient:
    def __init__(self):
        async def _create(**_kwargs):
            raise RuntimeError("simulated endpoint failure")

        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=_create))


def test_score_names_how_many_rows_the_endpoint_lost(monkeypatch, tmp_path):
    """A guard firing on an empty result cannot say why; the failure count points at the endpoint and clears the
    input columns."""
    prompts = tmp_path / "prompts.jsonl"
    prompts.touch()
    df = pd.DataFrame({"id": [1, 2, 3], "prompt": [[{"role": "user", "content": "q"}] for _ in range(3)]})
    monkeypatch.setattr(reward_scoring, "load_prompts_dataframe", lambda args: df)
    monkeypatch.setattr(reward_scoring, "build_openai_client", lambda *a, **k: _DeadClient())
    monkeypatch.setattr(reward_scoring, "load_reward_model", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score.py",
            "--model",
            "gen-model",
            "--dataset",
            str(prompts),
            "--rm_model_id",
            "org/rm",
            "--output_dir",
            str(tmp_path),
        ],
    )

    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(score_script.main())

    assert "failed=3" in str(excinfo.value), str(excinfo.value)


class _TruncatingClient:
    """A client whose every generation stops at the token cap (``finish_reason="length"``)."""

    def __init__(self):
        async def _create(**_kwargs):
            message = ChatCompletionMessage(role="assistant", content="frag")
            choice = types.SimpleNamespace(message=message, finish_reason="length")
            return types.SimpleNamespace(choices=[choice])

        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=_create))


def _rm_args(**overrides):
    args = types.SimpleNamespace(
        model="gen-model",
        temperature=0.0,
        max_tokens=8,
        id_field="id",
        prompt_field="prompt",
        follow_up_prompt_field="follow_up_prompt",
        correct_answer_field="correct_answer",
        local_system_prompt_field="system_prompt",
        global_system_prompt=None,
        n_hypos=2,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_generate_chat_message_reports_the_finish_reason():
    """Without the reason the scoring scripts cannot tell a finished answer from a fragment cut at --max_tokens."""
    message, finish_reason = asyncio.run(
        reward_scoring.generate_chat_message(
            _TruncatingClient(), [{"role": "user", "content": "q"}], _rm_args(), {"type": "text"}
        )
    )
    assert finish_reason == "length"
    assert message["content"] == "frag"


def test_generate_chat_message_names_a_reply_without_choices():
    """An aggregator can answer 200 with no choices, which is a named per-row failure and never an ``IndexError``."""

    class _EmptyClient:
        def __init__(self):
            async def _create(**_kwargs):
                return types.SimpleNamespace(choices=[], error={"code": 400, "message": "bad request"})

            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=_create))

    with pytest.raises(EmptyChoicesError, match="'gen-model'.*bad request"):
        asyncio.run(
            reward_scoring.generate_chat_message(
                _EmptyClient(), [{"role": "user", "content": "q"}], _rm_args(), {"type": "text"}
            )
        )


def test_score_drops_and_counts_a_truncated_hypothesis(monkeypatch, tmp_path):
    """Scoring a fragment writes a reward for text the policy never finished."""
    prompts = tmp_path / "prompts.jsonl"
    prompts.touch()
    df = pd.DataFrame({"id": [1, 2, 3], "prompt": [[{"role": "user", "content": "q"}] for _ in range(3)]})
    monkeypatch.setattr(reward_scoring, "load_prompts_dataframe", lambda args: df)
    monkeypatch.setattr(reward_scoring, "build_openai_client", lambda *a, **k: _TruncatingClient())
    monkeypatch.setattr(reward_scoring, "load_reward_model", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(reward_scoring, "score_conversations", lambda *a, **k: [1.0])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score.py",
            "--model",
            "gen-model",
            "--dataset",
            str(prompts),
            "--rm_model_id",
            "org/rm",
            "--output_dir",
            str(tmp_path),
        ],
    )

    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(score_script.main())

    assert "truncated=3" in str(excinfo.value), str(excinfo.value)


def test_rejection_sampling_drops_a_row_whose_hypotheses_were_truncated():
    """Fewer than two usable hypotheses cannot form a preference pair, so the row never reaches the scorer."""
    stats = {"failed": 0, "truncated": 0, "skipped_truncated": 0}
    queue = asyncio.Queue()
    row = pd.Series({"id": 7, "prompt": [{"role": "user", "content": "q"}]})

    asyncio.run(
        rejection_sampling.generate_hypotheses(
            _TruncatingClient(), row, _rm_args(), asyncio.Semaphore(1), queue, stats
        )
    )

    assert stats["skipped_truncated"] == 1
    assert stats["truncated"] == 2
    assert stats["failed"] == 0
    assert queue.empty()


def test_score_output_path_helper_is_the_one_under_test(tmp_path):
    """Pins the driver above to the real output-path contract rather than a path it invented."""
    assert reward_scoring.build_output_path(str(tmp_path), "p.jsonl", "m", "reward_scoring").parent == tmp_path


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_an_interrupt_exits_with_the_shell_convention_for_a_signalled_process(signum):
    """A wrapper, a CI step or a ``&&`` chain reads 0 as a complete dataset and consumes a partial output."""
    with pytest.raises(SystemExit) as excinfo:
        resumable_generation._signal_handler(signum, None)

    assert excinfo.value.code == 128 + signum


def test_the_saved_progress_hint_survives(caplog):
    """The non-zero exit is only actionable with the resume instruction beside it."""
    with caplog.at_level(logging.INFO, logger=resumable_generation.__name__), pytest.raises(SystemExit):
        resumable_generation._signal_handler(signal.SIGINT, None)

    assert any("resume by re-running" in message for message in caplog.messages), caplog.messages


def test_a_worker_that_dies_on_the_last_item_still_surfaces_its_exception():
    """``task_done`` runs in the worker's ``finally``, so a worker raising on the final item drains the queue on its
    way out and both tasks land in ``done`` together. Reading the drain first would discard the exception and
    report success over a truncated output."""

    async def scenario():
        queue = asyncio.Queue()
        for item in range(3):
            queue.put_nowait(item)

        async def worker():
            while True:
                item = await queue.get()
                try:
                    if item == 2:
                        raise RuntimeError("no space left on device")
                finally:
                    queue.task_done()

        task = asyncio.create_task(worker())
        await rejection_sampling.join_worker_queue(queue, task)

    with pytest.raises(RuntimeError, match="no space left on device"):
        asyncio.run(scenario())


def test_a_drained_queue_with_a_live_worker_returns():
    """The ordinary end of a run must return and not hang on a worker that loops forever waiting for items."""

    async def scenario():
        queue = asyncio.Queue()
        for item in range(3):
            queue.put_nowait(item)

        async def worker():
            while True:
                await queue.get()
                queue.task_done()

        task = asyncio.create_task(worker())
        await asyncio.wait_for(rejection_sampling.join_worker_queue(queue, task), timeout=5)
        task.cancel()

    asyncio.run(scenario())


def test_the_fixture_rows_carry_the_fields_the_cli_actually_reads():
    """A renamed id or prompt field would leave every driver feeding rows the real code path skips, so the
    accounting would read counts off a run that scored nothing."""
    parser = argparse.ArgumentParser()
    inference_arguments.add_s3_dataset_args(parser)
    defaults = parser.parse_args(["--dataset", "in", "--output_path", "out"])

    row = _prompt_rows(1)[0]
    assert defaults.id_field in row, f"the drivers' rows carry no {defaults.id_field!r}: {row}"
    assert defaults.prompt_field in row, f"the drivers' rows carry no {defaults.prompt_field!r}: {row}"
    assert json.loads(json.dumps(row)), "the CLIs write their rows as JSONL"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
