#!/usr/bin/env python
"""What the fenced weight sync does besides pushing: credit the pause to in-flight episodes, and
release the score-only clients' HTTP sessions at cleanup.

    python tests/cpu/grpo/test_weight_sync_fenced_sync_bookkeeping.py
"""

import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.configs.rollout_config import RolloutConfig
from src.environments.ray_actors import RolloutManager
from src.trainers.grpo.rollout.async_rollouts import AsyncRolloutMixin

_SYNC_S = 0.05


class _Trainer(AsyncRolloutMixin):
    """The attributes the fenced sync reads, around a push that takes ``_SYNC_S`` (or declines, or fails)."""

    def __init__(self, manager, outcome):
        self._rollout_manager = manager
        self.accelerator = SimpleNamespace(is_main_process=True)
        self.state = SimpleNamespace(global_step=3)
        self.async_config = SimpleNamespace(rollout_backend="vllm")
        self._outcome = outcome

    def _sync_weights_to_engine(self, force=False):
        if self._outcome == "fail":
            raise ConnectionError("server gone")
        time.sleep(_SYNC_S)
        return self._outcome == "pushed"


def _manager() -> RolloutManager:
    return RolloutManager(
        num_workers=1, env_type="native_math", env_config={}, server_urls=["http://x"], rollout_config=RolloutConfig()
    )


def _window_closed(manager: RolloutManager) -> bool:
    """A closed window stops accruing: the credit reads the same a moment later."""
    before = manager.paused_seconds
    time.sleep(0.01)
    return manager.paused_seconds == before


def test_a_push_credits_its_duration_and_closes_the_window():
    manager = _manager()
    assert _Trainer(manager, "pushed")._sync_weights_to_engine_fenced() is True
    assert manager.paused_seconds >= _SYNC_S
    assert _window_closed(manager), "the window must close, or the credit grows forever"


def test_a_declined_step_credits_nothing():
    manager = _manager()
    assert _Trainer(manager, "declined")._sync_weights_to_engine_fenced() is False
    assert manager.paused_seconds == 0.0
    assert _window_closed(manager)


def test_a_failed_push_closes_the_window_before_raising():
    manager = _manager()
    with pytest.raises(RuntimeError, match="server gone"):
        _Trainer(manager, "fail")._sync_weights_to_engine_fenced()
    assert _window_closed(manager)
    assert manager.paused_seconds == 0.0, "a push that never landed paused no engine"


def test_cleanup_closes_the_score_only_clients_sessions():
    trainer = _Trainer(None, "pushed")
    trainer._weight_sync_client = None
    trainer._loop = None
    trainer._prefetch_thread = None
    trainer._multi_server_mode = False
    sessions = [Mock(), Mock()]
    trainer._engine_rescore_clients_list = [SimpleNamespace(session=s) for s in sessions]

    trainer._cleanup_async_components()

    assert all(s.close.call_count == 1 for s in sessions)
    assert trainer._engine_rescore_clients_list is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
