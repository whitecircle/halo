#!/usr/bin/env python
"""Every dropped row must be counted, and an interrupted run must not exit 0.

Each path below reports partial or failed work as clean work unless the accounting holds:

* ``reward_model/rm_scoring.py`` must count each per-row failure, or neither its own summary nor the
  guard's tally can say how many rows the endpoint lost.
* ``_common.run_async_cli``'s signal handler must not exit 0: a SIGTERM'd (or Ctrl-C'd) job then
  reports success to whatever ran it and the next pipeline stage consumes a partial output dataset.
* ``rm_rejection_sampling.join_worker_queue`` must not race a queue drain against its worker —
  reading the drain first drops the worker's exception whenever the two complete together.

Run: pytest tests/cpu/inference/test_drop_accounting.py
"""

import argparse
import asyncio
import json
import signal
import sys
import types

import pandas as pd
import pytest
from loguru import logger as loguru_logger

from scripts.inference import _common
from scripts.inference.reward_model import _common as rm_common
from scripts.inference.reward_model import rm_rejection_sampling, rm_scoring


def _prompt_rows(count: int) -> list[dict]:
    return [{"id": i, "prompt": [{"role": "user", "content": f"q{i}"}]} for i in range(count)]


# --- rm_scoring: per-row failures reach the guard's tally -----------------------------------------


class _DeadClient:
    """A client whose every generation call fails, as a dead endpoint or wrong model name does."""

    def __init__(self):
        async def _create(**_kwargs):
            raise RuntimeError("simulated endpoint failure")

        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=_create))


def test_rm_scoring_names_how_many_rows_the_endpoint_lost(monkeypatch, tmp_path):
    """The guard fired on "nothing produced" but could not say why; the failure count is what points
    at the endpoint rather than at the input columns."""
    prompts = tmp_path / "prompts.jsonl"
    prompts.touch()
    df = pd.DataFrame({"id": [1, 2, 3], "prompt": [[{"role": "user", "content": "q"}] for _ in range(3)]})
    monkeypatch.setattr(rm_scoring, "load_prompts_dataframe", lambda args: df)
    monkeypatch.setattr(rm_common, "create_openai_client", lambda *a, **k: _DeadClient())
    monkeypatch.setattr(rm_common, "load_reward_model", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rm_scoring.py",
            "--model_name",
            "gen-model",
            "--prompts_source",
            str(prompts),
            "--rm_model_path",
            "org/rm",
            "--output_folder",
            str(tmp_path),
        ],
    )

    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(rm_scoring.main())

    assert "failed=3" in str(excinfo.value), str(excinfo.value)


class _TruncatingClient:
    """A client whose every generation stops at the token cap (``finish_reason="length"``)."""

    def __init__(self):
        async def _create(**_kwargs):
            message = types.SimpleNamespace(model_dump=lambda exclude=None: {"role": "assistant", "content": "frag"})
            choice = types.SimpleNamespace(message=message, finish_reason="length")
            return types.SimpleNamespace(choices=[choice])

        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=_create))


def _rm_args(**overrides):
    args = types.SimpleNamespace(
        model_name="gen-model",
        temperature=0.0,
        max_gen_tokens=8,
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
    """The reason is part of the contract: without it the scoring scripts cannot tell a finished
    answer from a fragment cut at --max_gen_tokens."""
    message, finish_reason = asyncio.run(
        rm_common.generate_chat_message(
            _TruncatingClient(), [{"role": "user", "content": "q"}], _rm_args(), {"type": "text"}
        )
    )
    assert finish_reason == "length"
    assert message["content"] == "frag"


def test_rm_scoring_drops_and_counts_a_truncated_hypothesis(monkeypatch, tmp_path):
    """A hypothesis cut at the token cap is a fragment; scoring it writes a reward for text the
    policy never finished, so the row is dropped and named in the tally."""
    prompts = tmp_path / "prompts.jsonl"
    prompts.touch()
    df = pd.DataFrame({"id": [1, 2, 3], "prompt": [[{"role": "user", "content": "q"}] for _ in range(3)]})
    monkeypatch.setattr(rm_scoring, "load_prompts_dataframe", lambda args: df)
    monkeypatch.setattr(rm_common, "create_openai_client", lambda *a, **k: _TruncatingClient())
    monkeypatch.setattr(rm_common, "load_reward_model", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(rm_common, "score_conversations", lambda *a, **k: [1.0])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rm_scoring.py",
            "--model_name",
            "gen-model",
            "--prompts_source",
            str(prompts),
            "--rm_model_path",
            "org/rm",
            "--output_folder",
            str(tmp_path),
        ],
    )

    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(rm_scoring.main())

    assert "truncated=3" in str(excinfo.value), str(excinfo.value)


def test_rm_rejection_sampling_drops_a_row_whose_hypotheses_were_truncated():
    """Fewer than two usable hypotheses cannot form a preference pair, so the row never reaches the
    scorer — counted, not silently queued with a fragment as the chosen or rejected side."""
    stats = {"failed": 0, "truncated": 0, "skipped_truncated": 0}
    queue = asyncio.Queue()
    row = pd.Series({"id": 7, "prompt": [{"role": "user", "content": "q"}]})

    asyncio.run(
        rm_rejection_sampling.generate_hypotheses(
            _TruncatingClient(), row, _rm_args(), asyncio.Semaphore(1), queue, stats
        )
    )

    assert stats["skipped_truncated"] == 1
    assert stats["truncated"] == 2
    assert stats["failed"] == 0
    assert queue.empty()


def test_rm_scoring_output_path_helper_is_the_one_under_test(tmp_path):
    """Pins the driver above to the real output-path contract rather than a path it invented."""
    assert rm_common.build_output_path(str(tmp_path), "p.jsonl", "m", "rm_scoring").parent == tmp_path


# --- run_async_cli: an interrupted run is not a successful one ------------------------------------


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_an_interrupt_exits_with_the_shell_convention_for_a_signalled_process(signum):
    """128 + signo, not 0: a wrapper script, a CI step or a ``&&`` chain reads 0 as "the dataset is
    complete" and proceeds to consume a partially written output."""
    with pytest.raises(SystemExit) as excinfo:
        _common._signal_handler(signum, None)

    assert excinfo.value.code == 128 + signum


def test_the_saved_progress_hint_survives():
    """The non-zero exit is only actionable with the resume instruction beside it.

    Read off a loguru sink rather than ``capsys``: the CLI logs through loguru, whose handler binds
    to the real stderr at import and never reaches pytest's capture.
    """
    lines: list[str] = []
    sink_id = loguru_logger.add(lines.append, format="{message}")
    try:
        with pytest.raises(SystemExit):
            _common._signal_handler(signal.SIGINT, None)
    finally:
        loguru_logger.remove(sink_id)

    assert any("resume by re-running" in line for line in lines), lines


# --- rm_rejection_sampling: a dying worker is never mistaken for a drained queue ------------------


def test_a_worker_that_dies_on_the_last_item_still_surfaces_its_exception():
    """``task_done`` runs in the worker's ``finally``, so a worker raising on the FINAL item drains
    the queue on its way out and BOTH tasks land in ``done`` together.

    Reading the drain first there returns cleanly and discards the worker's exception — the CLI then
    reports a successful run over an output file that stops at the row the write failed on, which is
    the exact silent-truncation this join exists to prevent.
    """

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
        await rm_rejection_sampling.join_worker_queue(queue, task)

    with pytest.raises(RuntimeError, match="no space left on device"):
        asyncio.run(scenario())


def test_a_drained_queue_with_a_live_worker_returns():
    """Anti-over-rejection: the ordinary end of a run must still return rather than hang on the
    worker, which loops forever waiting for items that will never come."""

    async def scenario():
        queue = asyncio.Queue()
        for item in range(3):
            queue.put_nowait(item)

        async def worker():
            while True:
                await queue.get()
                queue.task_done()

        task = asyncio.create_task(worker())
        await asyncio.wait_for(rm_rejection_sampling.join_worker_queue(queue, task), timeout=5)
        task.cancel()

    asyncio.run(scenario())


def test_the_fixture_rows_carry_the_fields_the_cli_actually_reads():
    """Guards the drivers above, which key their rows by hand.

    A rename of the CLI's id or prompt field would leave every driver feeding rows the real code
    path skips, so the accounting assertions would read their expected counts off a run that scored
    nothing. Taken from the shared parser's own defaults, not from a literal repeated here.
    """
    parser = argparse.ArgumentParser()
    _common.add_s3_dataset_args(parser)
    defaults = parser.parse_args(["--input_path", "in", "--output_path", "out"])

    row = _prompt_rows(1)[0]
    assert defaults.id_field in row, f"the drivers' rows carry no {defaults.id_field!r}: {row}"
    assert defaults.prompt_field in row, f"the drivers' rows carry no {defaults.prompt_field!r}: {row}"
    assert json.loads(json.dumps(row)), "the CLIs write their rows as JSONL"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
