"""``isr_engine_reference``: the mask stages read the engine's current-vs-sampling log-ratio on
re-scored rows (pure staleness), the trainer diff elsewhere; the re-score fans a trajectory's rows out
to one server, never raises, and is refused at construction where it could not mean what it says."""

import inspect
import types
from collections import defaultdict

import pytest
import torch
from accelerate import PartialState

from src.distributed.nccl.clients.sglang import SGLangWeightSyncClient
from src.distributed.nccl.clients.vllm import VLLMWeightSyncClient as VLLMClient
from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer
from src.trainers.grpo.objective.logratio import select_mask_logratio
from src.trainers.grpo.rollout import async_rollouts

PartialState()  # the re-score reports through accelerate's logger, which refuses to log without it


def test_mask_logratio_reads_the_engine_diff_only_on_rescored_rows():
    sampling = torch.tensor([[-1.0, -2.0, 0.0], [-1.0, -1.0, -1.0]])
    recompute = sampling + torch.tensor([[-0.05, -0.05, 0.0], [-0.05, -0.05, -0.05]])  # a −0.05 "numerics" floor
    engine_now = sampling + torch.tensor([[0.01, -0.01, 0.0], [0.0, 0.0, 0.0]])  # ±0.01 of real staleness
    corrected = torch.tensor([[True, True, False], [True, True, True]])  # the third token of row 0 is sampler-certain
    trainer_diff = (recompute - sampling) * corrected
    mask_diff, stats = select_mask_logratio(
        trainer_diff, recompute, sampling, engine_now, corrected, torch.tensor([True, False])
    )
    assert torch.allclose(mask_diff[0], torch.tensor([0.01, -0.01, 0.0]))
    assert torch.allclose(mask_diff[1], trainer_diff[1]), "a row without a re-score keeps the trainer diff"
    assert stats["sampling/engine_logratio_mean"] == pytest.approx(0.0)
    # recompute − engine over the two re-scored tokens: (−0.06 + −0.04) / 2
    assert stats["sampling/numerics_logratio_mean"] == pytest.approx(-0.05)
    assert stats["sampling/engine_rescore_coverage"] == pytest.approx(2 / 5)


def test_numerics_mean_is_recompute_minus_engine_over_rescored_tokens():
    sampling = torch.zeros(1, 4)
    recompute = torch.full((1, 4), -0.03)
    engine_now = torch.full((1, 4), 0.01)
    corrected = torch.ones(1, 4, dtype=torch.bool)
    _, stats = select_mask_logratio(
        recompute - sampling, recompute, sampling, engine_now, corrected, torch.tensor([True])
    )
    assert stats["sampling/numerics_logratio_mean"] == pytest.approx(-0.04)
    assert stats["sampling/engine_logratio_mean"] == pytest.approx(0.01)


class _FakeClient:
    def __init__(self, fail: bool = False):
        self.calls = []
        self.fail = fail

    def score_completion_logprobs(self, prompt_ids, completion_ids):
        self.calls.append((tuple(prompt_ids), tuple(completion_ids)))
        if self.fail:
            raise ConnectionError("server gone")
        return [-0.5] * len(completion_ids)


def _rescore_host(clients):
    host = types.SimpleNamespace(
        _engine_rescore_clients_list=clients,
        _weight_sync_client=None,  # every rank but the main one holds no sync client
        _metrics={"train": defaultdict(list)},
    )
    host._engine_rescore_clients = DistributedAsyncEnvironmentalGRPOTrainer._engine_rescore_clients.__get__(host)
    return DistributedAsyncEnvironmentalGRPOTrainer._rescore_rows_on_engine.__get__(host), host


def test_rescore_clients_are_built_per_rank_from_the_server_urls(monkeypatch):
    """The weight-sync client exists on the main process only; the re-score runs on every rank, so
    its clients come from the server URLs, score-only (no communicator), one per server."""
    built = []

    class _Scorer:
        def __init__(self, base_url, connection_timeout):
            built.append((base_url, connection_timeout))

    monkeypatch.setattr(async_rollouts, "resolve_weight_sync_client", lambda backend: _Scorer)
    host = types.SimpleNamespace(
        _engine_rescore_clients_list=None,
        _weight_sync_client=None,
        _multi_server_mode=True,
        async_config=types.SimpleNamespace(
            rollout_backend="vllm",
            rollout_connection_timeout=7.0,
            rollout_server_configs=[{"url": "http://a:8000"}, {"url": "http://b:8001"}],
        ),
    )
    clients = DistributedAsyncEnvironmentalGRPOTrainer._engine_rescore_clients.__get__(host)()
    assert built == [("http://a:8000", 7.0), ("http://b:8001", 7.0)]
    assert DistributedAsyncEnvironmentalGRPOTrainer._engine_rescore_clients.__get__(host)() is clients, "built once"
    host.async_config.rollout_server_configs = []
    host._multi_server_mode = False
    host.async_config.rollout_server_url = "http://single:8000"
    host._engine_rescore_clients_list = None
    DistributedAsyncEnvironmentalGRPOTrainer._engine_rescore_clients.__get__(host)()
    assert built[-1] == ("http://single:8000", 7.0)


def test_rescore_path_never_reads_the_main_process_sync_client():
    for method in (
        DistributedAsyncEnvironmentalGRPOTrainer._engine_rescore_clients,
        DistributedAsyncEnvironmentalGRPOTrainer._rescore_rows_on_engine,
    ):
        assert "self._weight_sync_client" not in inspect.getsource(method), method.__name__


def _rows():
    prompts = [torch.tensor([1, 2]), torch.tensor([1, 2, 3]), torch.tensor([4]), torch.tensor([5, 6])]
    completions = [torch.tensor([7, 8]), torch.tensor([9]), torch.tensor([10, 11, 12]), torch.tensor([13])]
    return prompts, completions


def test_rescore_routes_a_trajectory_to_one_server_and_skips_rows_without_sampling():
    clients = [_FakeClient(), _FakeClient()]
    rescore, host = _rescore_host(clients)
    prompts, completions = _rows()
    # trajectory 0 has two turns (rows 0, 1), trajectories 1 and 2 one each; row 3 has no sampling logps
    scored = rescore(prompts, completions, [True, True, True, False], [2, 1, 1], "train")
    assert [s is not None for s in scored] == [True, True, True, False]
    assert torch.equal(scored[2], torch.full((3,), -0.5))
    assert {c[0] for c in clients[0].calls} == {(1, 2), (1, 2, 3)}, "both rows of trajectory 0 hit server 0"
    assert {c[0] for c in clients[1].calls} == {(4,)}, "trajectory 1 hits server 1; row 3 never scored"
    assert host._metrics["train"]["sampling/engine_rescore_miss_frac"] == [0.0]


def test_rescore_reports_partial_failures_and_never_raises():
    clients = [_FakeClient(fail=True), _FakeClient()]
    rescore, host = _rescore_host(clients)
    prompts, completions = _rows()
    scored = rescore(prompts, completions, [True, True, True, True], [2, 1, 1], "train")
    # trajectories 0 and 2 land on the failing server 0, trajectory 1 on server 1
    assert scored[0] is None and scored[1] is None and scored[3] is None, "a failed request is that row's miss"
    assert torch.equal(scored[2], torch.full((3,), -0.5))
    assert host._metrics["train"]["sampling/engine_rescore_miss_frac"] == [pytest.approx(0.75)]
    rescore, host = _rescore_host([_FakeClient(fail=True)])
    scored = rescore(prompts, completions, [True, True, True, True], [2, 1, 1], "train")
    assert scored == [None] * 4
    assert host._metrics["train"]["sampling/engine_rescore_miss_frac"] == [1.0]


def test_rescore_rejects_a_length_mismatch_as_that_rows_miss():
    class _Short(_FakeClient):
        def score_completion_logprobs(self, prompt_ids, completion_ids):
            return [-0.5]

    rescore, host = _rescore_host([_Short()])
    prompts, completions = _rows()
    scored = rescore(prompts, completions, [True, True, True, True], [2, 1, 1], "train")
    assert scored[1] is not None and scored[2] is None, "one log-prob for a three-token completion is a miss"


def _gate_probe(*, is_correction=True, temperature=1.0, top_p=1.0):
    trainer = DistributedAsyncEnvironmentalGRPOTrainer.__new__(DistributedAsyncEnvironmentalGRPOTrainer)
    trainer._is_correction = is_correction
    trainer.async_config = types.SimpleNamespace(rollout_temperature=temperature, rollout_top_p=top_p)
    return trainer


def test_engine_reference_gate_accepts_the_identity_sampler_on_vllm():
    _gate_probe()._validate_engine_reference(VLLMClient)


@pytest.mark.parametrize(
    ("kwargs", "client_cls", "match"),
    [
        ({"is_correction": False}, VLLMClient, "importance-sampling correction"),
        (
            {},
            type("NoRescore", (VLLMClient,), {"SUPPORTS_ENGINE_RESCORE": False, "BACKEND_NAME": "Other"}),
            "not available on Other",
        ),
        ({"temperature": 1.1}, VLLMClient, "rollout_temperature 1.0"),
        ({"top_p": 0.95}, VLLMClient, "rollout_top_p 1.0"),
    ],
)
def test_engine_reference_gate_refuses(kwargs, client_cls, match):
    with pytest.raises(ValueError, match=match):
        _gate_probe(**kwargs)._validate_engine_reference(client_cls)


def test_both_engines_declare_the_rescore():
    assert VLLMClient.SUPPORTS_ENGINE_RESCORE is True
    assert SGLangWeightSyncClient.SUPPORTS_ENGINE_RESCORE is True
    _gate_probe()._validate_engine_reference(SGLangWeightSyncClient)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
