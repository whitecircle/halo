"""The parity probe after a trainer's first push: the engine's log-probs on one fixed row must sit within the
trainer-versus-engine floor of the trainer's own, once per trainer."""

import contextlib
from types import SimpleNamespace

import pytest
import torch

from halo.trainers.grpo import online
from halo.trainers.grpo.environmental import async_rollouts
from halo.trainers.grpo.environmental.trainer import DistributedAsyncEnvironmentalGRPOTrainer
from halo.trainers.grpo.online import DistributedOnlineGRPOTrainer
from halo.trainers.grpo.rollout.push_parity import PARITY_MAX_MEAN_GAP_NATS, verify_first_push_parity
from halo.weight_sync.clients.vllm import VLLMWeightSyncClient
from halo.weight_sync.pool import WeightSyncPool

_ROW = 1100


class _Engine:
    def __init__(self, offset: float = 0.0, window: int | None = None):
        self.offset, self.window, self.scored = offset, window, []

    def served_max_model_len(self):
        return self.window

    def score_completion_logprobs(self, prompt_ids, completion_ids):
        self.scored.append((prompt_ids, completion_ids))
        return [-2.0 + self.offset] * len(completion_ids)


def _trainer(*, cp: bool = False):
    def logps(_model, ids, _mask, logits_to_keep):
        assert ids.shape == (1, logits_to_keep + 1)
        return torch.full((1, logits_to_keep), -2.0), None, None

    return SimpleNamespace(
        first_push_verified=False,
        parallelism_config=SimpleNamespace(is_cp_mode=cp),
        processing_class=lambda _text: {"input_ids": list(range(_ROW + 40))},
        accelerator=SimpleNamespace(device=torch.device("cpu")),
        model=object(),
        _get_per_token_logps_and_entropies=logps,
    )


def test_an_engine_at_the_trainers_log_probs_passes_once():
    trainer, engine = _trainer(), _Engine(offset=PARITY_MAX_MEAN_GAP_NATS / 2)
    verify_first_push_parity(trainer, engine)
    verify_first_push_parity(trainer, engine)
    assert len(engine.scored) == 1
    assert len(engine.scored[0][1]) == _ROW - 1
    assert trainer.first_push_verified


@pytest.mark.parametrize("offset", [3.0, float("nan"), float("inf")])
def test_an_engine_that_serves_other_weights_is_refused_when_the_probe_raises(offset, monkeypatch):
    monkeypatch.setenv("HALO_WEIGHT_SYNC_PARITY_PROBE", "raise")
    with pytest.raises(ValueError, match="may not have taken the pushed weights"):
        verify_first_push_parity(_trainer(), _Engine(offset=offset))


def test_the_default_probe_warns_and_lets_the_run_go_on(caplog):
    with caplog.at_level("WARNING"):
        verify_first_push_parity(_trainer(), _Engine(offset=3.0))
    assert "may not have taken the pushed weights" in caplog.text


def test_an_unknown_probe_mode_is_refused(monkeypatch):
    monkeypatch.setenv("HALO_WEIGHT_SYNC_PARITY_PROBE", "1")
    with pytest.raises(ValueError, match="HALO_WEIGHT_SYNC_PARITY_PROBE"):
        verify_first_push_parity(_trainer(), _Engine())


def test_the_served_window_bounds_the_row():
    engine = _Engine(window=100)
    verify_first_push_parity(_trainer(), engine)
    assert len(engine.scored[0][1]) == 98


def test_the_probe_turns_off(monkeypatch):
    monkeypatch.setenv("HALO_WEIGHT_SYNC_PARITY_PROBE", "off")
    engine = _Engine(offset=9.0)
    verify_first_push_parity(_trainer(), engine)
    assert not engine.scored


def test_context_parallelism_skips_the_probe():
    engine = _Engine(offset=9.0)
    verify_first_push_parity(_trainer(cp=True), engine)
    assert not engine.scored


def test_a_pool_scores_on_its_first_server():
    pool = WeightSyncPool(
        [{"url": "http://a"}, {"url": "http://b"}],
        connection_timeout=1,
        client_cls=VLLMWeightSyncClient,
        base_group_port=0,
    )
    with pytest.raises(RuntimeError, match="no connected client"):
        _ = pool.scoring_client
    first, second = _Engine(), _Engine()
    pool._clients = [first, second]
    verify_first_push_parity(_trainer(), pool)
    assert first.scored and not second.scored


def test_the_environmental_trainer_probes_after_each_push_it_makes(monkeypatch):
    probed = []
    monkeypatch.setattr(async_rollouts, "sync_trainer_weights", lambda trainer, client: None)
    monkeypatch.setattr(async_rollouts, "profiling_context", lambda trainer, name: contextlib.nullcontext())
    monkeypatch.setattr(async_rollouts, "verify_first_push_parity", lambda trainer, client: probed.append(client))
    trainer = object.__new__(DistributedAsyncEnvironmentalGRPOTrainer)
    trainer.state = SimpleNamespace(global_step=1)
    trainer.async_config = SimpleNamespace(sync_weights_every_n_steps=4)
    trainer._weight_sync_group_formed = True
    trainer._weight_sync_client = client = object()
    assert trainer._sync_weights_to_engine() is False
    assert probed == []
    assert trainer._sync_weights_to_engine(force=True) is True
    assert probed == [client]


def test_the_online_trainer_probes_after_each_push_it_makes(monkeypatch):
    probed = []
    monkeypatch.setattr(online, "sync_trainer_weights", lambda trainer, client: None)
    monkeypatch.setattr(online, "verify_first_push_parity", lambda trainer, client: probed.append(client))
    trainer = object.__new__(DistributedOnlineGRPOTrainer)
    trainer.accelerator = SimpleNamespace(is_main_process=True)
    trainer.vllm_generation = SimpleNamespace(vllm_client=(client := object()))
    trainer._distributed_sync_weights()
    assert probed == [client]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
