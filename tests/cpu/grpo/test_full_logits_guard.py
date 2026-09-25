#!/usr/bin/env python
"""The GRPO trainers weigh the full-logits loss plane against free device memory at construction.

Without ``use_chunked_grpo_logprobs`` the loss forward holds ``rows × width × vocab`` logits at once; on
a large-vocabulary model with long completions that alone outgrows the device, and the run OOMs at the
first step after rollouts were already paid for. The guard raises when the plane cannot fit, warns when
it takes a large share, and names the remedy either way.

    python tests/cpu/grpo/test_full_logits_guard.py
"""

import ast
import inspect
import logging
import textwrap
import types

import pytest
import torch

import src.trainers.grpo.mixins.chunked_logprobs as chunked
from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer
from src.trainers.grpo.mixins.chunked_logprobs import ChunkedLogprobsCore, LogitsWidth, full_logits_verdict
from src.trainers.grpo.offline import OfflineGRPOTrainer
from src.trainers.grpo.online import DistributedGRPOTrainer

_GIB = 1024**3
_WIDTH = LogitsWidth(1024, "max_completion_length")
_TRAIN_ROWS = "per_device_train_batch_size"


def test_the_verdict_prices_rows_width_vocab_at_four_bytes():
    # 2 × 1024 × 131072 × 4 B = 1 GiB exactly.
    assert full_logits_verdict(2, _TRAIN_ROWS, _WIDTH, 131072, free_bytes=4 * _GIB) is None
    fatal, message = full_logits_verdict(2, _TRAIN_ROWS, _WIDTH, 131072, free_bytes=int(1.5 * _GIB))
    assert not fatal and "1.0 GiB" in message and "67%" in message
    fatal, message = full_logits_verdict(2, _TRAIN_ROWS, _WIDTH, 131072, free_bytes=_GIB - 1)
    assert fatal and "more than" in message


@pytest.mark.parametrize("free", [int(1.5 * _GIB), _GIB // 2])
def test_every_verdict_names_the_remedy_and_the_settings_behind_the_plane(free):
    width = LogitsWidth(1024, "max_train_row_tokens (unset: the 1023-token context window)")
    _, message = full_logits_verdict(2, "per_device_eval_batch_size", width, 131072, free_bytes=free)
    assert "use_chunked_grpo_logprobs: true" in message
    assert "per_device_eval_batch_size (2)" in message
    assert "max_train_row_tokens (unset: the 1023-token context window)" in message


def _host(
    *,
    chunked_on: bool = False,
    rows: int = 2,
    vocab: int | None = 131072,
    eval_rows: int = 1,
    eval_strategy: str = "no",
    eval_on_start: bool = False,
):
    head = None if vocab is None else types.SimpleNamespace(weight=torch.empty(vocab, 0))
    model = types.SimpleNamespace(get_output_embeddings=lambda: head)
    return types.SimpleNamespace(
        _use_chunked_grpo_logprobs=chunked_on,
        args=types.SimpleNamespace(
            per_device_train_batch_size=rows,
            per_device_eval_batch_size=eval_rows,
            eval_strategy=eval_strategy,
            eval_on_start=eval_on_start,
        ),
        model=model,
        accelerator=types.SimpleNamespace(unwrap_model=lambda m: m, is_main_process=True, device="cuda:3"),
    )


@pytest.fixture
def free_memory(monkeypatch):
    """Pretend a CUDA device with ``free`` bytes available; records the device each read asked about."""
    asked: list = []

    def _set(free: int):
        monkeypatch.setattr(chunked.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            chunked.torch.cuda, "mem_get_info", lambda device=None: asked.append(device) or (free, 8 * free)
        )
        return asked

    return _set


def test_an_oversized_plane_refuses_the_run(free_memory):
    asked = free_memory(_GIB // 2)
    with pytest.raises(RuntimeError, match="use_chunked_grpo_logprobs: true"):
        ChunkedLogprobsCore._check_full_logits_fit(_host(), _WIDTH)
    assert asked == ["cuda:3"], "free memory must be read on the trainer's own device"


def test_a_plane_another_rank_cannot_hold_refuses_the_run_here_too(free_memory, monkeypatch):
    """A refusal on one rank alone would leave its peers in the next collective."""
    free_memory(64 * _GIB)
    monkeypatch.setattr(chunked, "rank_consensus", lambda ok: (False, True))
    with pytest.raises(RuntimeError, match="Another rank"):
        ChunkedLogprobsCore._check_full_logits_fit(_host(), _WIDTH)


def test_a_plane_taking_most_of_the_free_memory_warns(free_memory, caplog):
    free_memory(int(1.5 * _GIB))
    with caplog.at_level(logging.WARNING, logger=chunked.__name__):
        ChunkedLogprobsCore._check_full_logits_fit(_host(), _WIDTH)
    assert any("use_chunked_grpo_logprobs" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(
    ("eval_strategy", "eval_on_start"), [("steps", False), ("no", True)], ids=["eval-strategy", "eval-on-start"]
)
def test_an_evaluating_run_weighs_the_wider_eval_chunk(free_memory, eval_strategy, eval_on_start):
    """The eval loss forward chunks per_device_eval_batch_size rows: 1 train row fits in 1.5 GiB, the
    4 eval rows (2 GiB) do not, and the refusal names the eval batch as the setting to lower."""
    free_memory(int(1.5 * _GIB))
    host = _host(rows=1, eval_rows=4, eval_strategy=eval_strategy, eval_on_start=eval_on_start)
    with pytest.raises(RuntimeError, match=r"lower per_device_eval_batch_size \(4\)"):
        ChunkedLogprobsCore._check_full_logits_fit(host, _WIDTH)


def test_a_run_that_never_evaluates_weighs_only_the_train_chunk(free_memory, caplog):
    free_memory(4 * _GIB)
    with caplog.at_level(logging.WARNING, logger=chunked.__name__):
        ChunkedLogprobsCore._check_full_logits_fit(_host(rows=1, eval_rows=16), _WIDTH)
    assert not caplog.records


def test_a_narrower_eval_chunk_leaves_the_train_chunk_the_one_weighed(free_memory):
    free_memory(_GIB // 2)
    host = _host(rows=2, eval_rows=1, eval_strategy="steps")
    with pytest.raises(RuntimeError, match=r"lower per_device_train_batch_size \(2\)"):
        ChunkedLogprobsCore._check_full_logits_fit(host, _WIDTH)


def test_a_small_plane_stays_quiet(free_memory, caplog):
    free_memory(64 * _GIB)
    with caplog.at_level(logging.WARNING, logger=chunked.__name__):
        ChunkedLogprobsCore._check_full_logits_fit(_host(), _WIDTH)
    assert not caplog.records


@pytest.mark.parametrize(("chunked_on", "width"), [(True, _WIDTH), (False, None)], ids=["chunked", "unbounded"])
def test_chunked_log_probs_or_an_unbounded_width_skip_the_check(free_memory, chunked_on, width):
    free_memory(1)
    ChunkedLogprobsCore._check_full_logits_fit(_host(chunked_on=chunked_on), width)


def test_a_model_without_an_output_embedding_is_refused_by_name(free_memory):
    free_memory(64 * _GIB)
    with pytest.raises(ValueError, match="get_output_embeddings"):
        ChunkedLogprobsCore._check_full_logits_fit(_host(vocab=None), _WIDTH)


def _env_host(*, per_turn: bool, row_cap: int | None):
    return types.SimpleNamespace(
        _train_on_sampled_tokens=per_turn,
        async_config=types.SimpleNamespace(rollout_max_tokens=4096, max_train_row_tokens=row_cap),
        _context_limit=lambda: 32768,
    )


@pytest.mark.parametrize(
    ("per_turn", "row_cap", "expected"),
    [
        (True, 16384, LogitsWidth(4097, "rollout_max_tokens")),
        (False, 16384, LogitsWidth(16385, "max_train_row_tokens")),
        (False, None, LogitsWidth(32769, "max_train_row_tokens (unset: the 32768-token context window)")),
    ],
    ids=["per-turn", "row-cap", "context-window"],
)
def test_the_environmental_width_names_the_setting_that_bounds_it(per_turn, row_cap, expected):
    host = _env_host(per_turn=per_turn, row_cap=row_cap)
    assert DistributedAsyncEnvironmentalGRPOTrainer._loss_logits_width(host) == expected


def test_the_online_width_is_the_completion_budget_or_unchecked_without_one():
    assert DistributedGRPOTrainer._loss_logits_width(types.SimpleNamespace(max_completion_length=256)) == (
        LogitsWidth(257, "max_completion_length")
    )
    assert DistributedGRPOTrainer._loss_logits_width(types.SimpleNamespace(max_completion_length=None)) is None


def _call_lines(fn, name: str) -> list[int]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == name
    ]


def _offline_host(*, pp: bool = False, max_completion_length: int | None = 256):
    return types.SimpleNamespace(
        _pp_runtime=object() if pp else None,
        max_completion_length=max_completion_length,
    )


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        (_offline_host(), LogitsWidth(257, "max_completion_length")),
        (_offline_host(max_completion_length=None), None),
        (_offline_host(max_completion_length=0), None),
        (_offline_host(pp=True), None),
    ],
    ids=["bounded", "unset", "non-positive", "pipeline"],
)
def test_the_offline_width_is_the_completion_budget_off_pp(host, expected):
    """Under PP the last stage's plane is the pipeline's, which chunked log-probs cannot remove; an
    unset budget leaves the width to the stored completions. Neither is weighed."""
    assert OfflineGRPOTrainer._loss_logits_width(host) == expected


def test_offline_construction_weighs_the_plane_after_the_model_is_placed():
    """The offline trainer has no shared spine to call the check; its own ctor must, once the parallel
    wrap has put the model where the free memory is read."""
    checks = _call_lines(OfflineGRPOTrainer.__init__, "_check_full_logits_fit")
    setups = _call_lines(OfflineGRPOTrainer.__init__, "_setup_distributed_modes")
    assert checks and setups and min(checks) > max(setups)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
