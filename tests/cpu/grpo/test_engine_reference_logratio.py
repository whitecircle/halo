"""``isr_engine_reference``: the mask stages read the engine's current-vs-sampling log-ratio on
re-scored rows (pure staleness), the trainer diff elsewhere; the re-score fans a trajectory's rows out
to one server, never raises, and is refused at construction where it could not mean what it says, and
at startup against an engine whose prompt log-probs depend on how the prefill was batched."""

import ast
import inspect
import math
import re
import threading
import types
from collections import defaultdict
from unittest import mock

import pytest
import torch

from halo.configs.grpo_environmental import AsyncRolloutConfig
from halo.trainers.grpo.environmental import async_rollouts, importance_sampling
from halo.trainers.grpo.environmental.environmental_rules import reject_unsupported_engine_reference
from halo.trainers.grpo.environmental.trainer import DistributedAsyncEnvironmentalGRPOTrainer
from halo.trainers.grpo.environmental.trajectory_tokenize import TurnRow
from halo.trainers.grpo.objective.logratio import select_mask_logratio
from halo.trainers.grpo.rollout import engine_preflight
from halo.trainers.grpo.rollout.engine_preflight import _probe_prompt_logprobs, verify_engine_prompt_logprobs
from halo.weight_sync.clients.sglang import SGLangWeightSyncClient
from halo.weight_sync.clients.vllm import VLLMWeightSyncClient as VLLMClient
from tests.common.fake_engines import GRAPH_CAPTURE_TOKENS, FakeEngineServer, fake_server_class
from tests.common.repo import REPO_ROOT
from tests.common.world_metrics import attach_world_metrics, flushed_metrics

ENV_SCRIPT = REPO_ROOT / "scripts/training/grpo/environmental.py"


def _pair(stats: dict, key: str) -> tuple[float, float]:
    """A diagnostic's ``(numerator, denominator)`` read off the device, where the stage leaves both."""
    assert all(isinstance(v, torch.Tensor) for v in stats[key]), "the pair must stay on device for one read"
    return tuple(float(v) for v in stats[key])


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
    assert _pair(stats, "sampling/engine_logratio_mean") == (pytest.approx(0.0), 2)
    # recompute − engine summed over the two re-scored tokens: −0.06 + −0.04
    assert _pair(stats, "sampling/numerics_logratio_mean") == (pytest.approx(-0.10), 2)
    assert _pair(stats, "sampling/engine_rescore_coverage") == (2, 5)


def test_numerics_mean_is_recompute_minus_engine_over_rescored_tokens():
    sampling = torch.zeros(1, 4)
    recompute = torch.full((1, 4), -0.03)
    engine_now = torch.full((1, 4), 0.01)
    corrected = torch.ones(1, 4, dtype=torch.bool)
    _, stats = select_mask_logratio(
        recompute - sampling, recompute, sampling, engine_now, corrected, torch.tensor([True])
    )
    assert _pair(stats, "sampling/numerics_logratio_mean") == (pytest.approx(-0.16), 4)
    assert _pair(stats, "sampling/engine_logratio_mean") == (pytest.approx(0.04), 4)


class _FakeClient:
    def __init__(self, fail: bool = False):
        self.calls = []
        self.fail = fail

    def score_completion_logprobs(self, prompt_ids, completion_ids):
        self.calls.append((tuple(prompt_ids), tuple(completion_ids)))
        if self.fail:
            raise ConnectionError("server gone")
        return [-0.5] * len(completion_ids)


class _ThreadRecordingClient(_FakeClient):
    """Records the worker thread of every request: a ``ThreadPoolExecutor`` names its threads after the pool."""

    def __init__(self):
        super().__init__()
        self.threads = []

    def score_completion_logprobs(self, prompt_ids, completion_ids):
        self.threads.append(threading.current_thread().name)
        return super().score_completion_logprobs(prompt_ids, completion_ids)


def _rescore_host(clients):
    host = types.SimpleNamespace(
        _engine_rescore_clients=clients,
        _weight_sync_client=None,  # every rank but the main one holds no sync client
        _metrics={"train": defaultdict(list)},
        async_config=types.SimpleNamespace(isr_engine_rescore_per_server=128),
    )
    attach_world_metrics(host)
    rescore = DistributedAsyncEnvironmentalGRPOTrainer._rescore_rows_on_engine.__get__(host)

    def rescore_rows(prompts, completions, has_sampling, turns_per_traj):
        rows = [
            TurnRow(
                prompt, completion, torch.ones_like(completion), torch.zeros(len(completion)) if has else None, None
            )
            for prompt, completion, has in zip(prompts, completions, has_sampling, strict=True)
        ]
        return rescore(rows, turns_per_traj)

    return rescore_rows, host


def test_rescore_clients_are_built_per_rank_from_the_server_urls(monkeypatch):
    """The weight-sync client exists on the main process only; the re-score runs on every rank, so
    its clients come from the server URLs, score-only (no communicator), one per server."""
    built = []

    class _Scorer:
        def __init__(self, base_url, connection_timeout):
            built.append((base_url, connection_timeout))

    monkeypatch.setattr(async_rollouts, "resolve_weight_sync_client", lambda backend: _Scorer)
    host = types.SimpleNamespace(
        _weight_sync_client=None,
        _train_loader_batch_size=lambda: 8,
        async_config=AsyncRolloutConfig(
            rollout_connection_timeout=7.0,
            rollout_server_url="http://single:8000",
            rollout_server_configs=[{"url": "http://a:8000"}, {"url": "http://b:8001"}],
        ),
    )
    build = DistributedAsyncEnvironmentalGRPOTrainer._build_engine_rescore_clients.__get__(host)
    assert len(build()) == 2
    assert built == [("http://a:8000", 7.0), ("http://b:8001", 7.0)], "the configs list overrides the single URL"
    # Trajectory t re-scores on server t mod N: a one-trajectory round reaches server 0 alone.
    host._train_loader_batch_size = lambda: 1
    before = len(built)
    assert len(build()) == 1 and built[before:] == [("http://a:8000", 7.0)]
    host._train_loader_batch_size = lambda: 8
    for single_server in (None, []):
        host.async_config.rollout_server_configs = single_server
        before = len(built)
        build()
        assert built[before:] == [("http://single:8000", 7.0)], f"configs={single_server!r} builds the single URL"


def test_rescore_path_never_reads_the_main_process_sync_client():
    for method in (
        DistributedAsyncEnvironmentalGRPOTrainer._build_engine_rescore_clients,
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
    scored = rescore(prompts, completions, [True, True, True, False], [2, 1, 1])
    assert [s is not None for s in scored] == [True, True, True, False]
    assert torch.equal(scored[2], torch.full((3,), -0.5))
    assert {c[0] for c in clients[0].calls} == {(1, 2), (1, 2, 3)}, "both rows of trajectory 0 hit server 0"
    assert {c[0] for c in clients[1].calls} == {(4,)}, "trajectory 1 hits server 1; row 3 never scored"
    assert flushed_metrics(host)["sampling/engine_rescore_miss_frac"] == [0.0]


def test_rescore_reports_partial_failures_and_never_raises():
    clients = [_FakeClient(fail=True), _FakeClient()]
    rescore, host = _rescore_host(clients)
    prompts, completions = _rows()
    scored = rescore(prompts, completions, [True, True, True, True], [2, 1, 1])
    # trajectories 0 and 2 land on the failing server 0, trajectory 1 on server 1
    assert scored[0] is None and scored[1] is None and scored[3] is None, "a failed request is that row's miss"
    assert torch.equal(scored[2], torch.full((3,), -0.5))
    assert flushed_metrics(host)["sampling/engine_rescore_miss_frac"] == [pytest.approx(0.75)]
    rescore, host = _rescore_host([_FakeClient(fail=True)])
    scored = rescore(prompts, completions, [True, True, True, True], [2, 1, 1])
    assert scored == [None] * 4
    assert flushed_metrics(host)["sampling/engine_rescore_miss_frac"] == [1.0]


@pytest.mark.parametrize(
    ("failing", "level", "silent"), [((True, False), "warning", "error"), ((True,), "error", "warning")]
)
def test_a_rescore_failure_is_reported_by_the_rank_that_hit_it(monkeypatch, failing, level, silent):
    """Each rank re-scores its own rows, so a failure is rank-local: guarded to the main process, a dead route
    on rank 3 would never print."""
    recorder = mock.MagicMock()
    monkeypatch.setattr(importance_sampling, "logger", recorder)
    monkeypatch.setattr(importance_sampling, "get_global_rank", lambda: 3)
    monkeypatch.setattr(importance_sampling, "is_global_main_process", lambda: False)
    rescore, _host = _rescore_host([_FakeClient(fail=fail) for fail in failing])
    prompts, completions = _rows()
    rescore(prompts, completions, [True, True, True, True], [2, 1, 1])
    getattr(recorder, silent).assert_not_called()
    (msg,), _ = getattr(recorder, level).call_args
    assert msg.startswith("[rank 3] isr_engine_reference:"), msg


@pytest.mark.parametrize(("world_size", "per_rank"), [(1, 128), (8, 16), (64, 2), (1024, 1)])
def test_the_rescore_load_on_a_server_is_bounded_whatever_the_world_size(world_size, per_rank):
    """Every rank re-scores its own rows on the same servers, so a fixed per-rank width puts world × width requests
    in flight on a server; each rank keeps its share of the per-server budget there, and one request at the least."""
    assert importance_sampling.resolve_rescore_concurrency(128, world_size) == per_rank
    assert per_rank * world_size <= max(128, world_size)


def test_each_server_takes_its_requests_through_a_pool_of_this_ranks_share(monkeypatch):
    """Every rank maps its first trajectories to server 0, so one pool sized for every server would let a rank put
    all of its width on that one; a pool per server caps each server at this rank's share."""
    widths = []
    real_pool = importance_sampling.ThreadPoolExecutor

    def recording_pool(max_workers):
        widths.append(max_workers)
        return real_pool(max_workers)

    monkeypatch.setattr(importance_sampling, "ThreadPoolExecutor", recording_pool)
    monkeypatch.setattr(importance_sampling, "get_global_world_size", lambda: 32)
    clients = [_ThreadRecordingClient(), _ThreadRecordingClient()]
    rescore, host = _rescore_host(clients)
    host.async_config.isr_engine_rescore_per_server = 64
    prompts = [torch.tensor([i]) for i in range(6)]
    completions = [torch.tensor([i, i]) for i in range(6)]

    rescore(prompts, completions, [True] * 6, [3, 1, 2])

    assert widths == [2, 2], f"pool widths {widths}, not 64 per server ÷ 32 ranks for each of 2 servers"
    pools = [{name.rsplit("_", 1)[0] for name in client.threads} for client in clients]
    assert all(len(pool) == 1 for pool in pools) and pools[0] != pools[1], f"servers shared a pool: {pools}"
    assert [len(client.threads) for client in clients] == [5, 1], "trajectories 0 and 2 on server 0, 1 on server 1"


def test_a_rescore_before_the_components_started_raises():
    """Every request would otherwise fail as a miss and the step train on the trainer's reference, quietly."""
    rescore, _host = _rescore_host(None)
    prompts, completions = _rows()
    with pytest.raises(RuntimeError, match="before the rollout components started"):
        rescore(prompts, completions, [True, True, True, True], [2, 1, 1])


def test_rescore_rejects_a_length_mismatch_as_that_rows_miss():
    class _Short(_FakeClient):
        def score_completion_logprobs(self, prompt_ids, completion_ids):
            return [-0.5]

    rescore, host = _rescore_host([_Short()])
    prompts, completions = _rows()
    scored = rescore(prompts, completions, [True, True, True, True], [2, 1, 1])
    assert scored[1] is not None and scored[2] is None, "one log-prob for a three-token completion is a miss"


def _gate(client_cls, *, is_correction=True, **sampler) -> None:
    """The gate over a rollout config at the identity sampler, with ``sampler`` overriding its knobs."""
    config = AsyncRolloutConfig(**{"rollout_temperature": 1.0, "rollout_top_p": 1.0, **sampler})
    reject_unsupported_engine_reference(client_cls, config, is_correction)


def test_engine_reference_gate_accepts_the_identity_sampler_on_vllm():
    _gate(VLLMClient)


@pytest.mark.parametrize(
    ("kwargs", "client_cls", "match"),
    [
        ({"is_correction": False}, VLLMClient, "importance-sampling correction"),
        (
            {},
            type("NoRescore", (VLLMClient,), {"SUPPORTS_ENGINE_RESCORE": False, "BACKEND_NAME": "Other"}),
            "not available on Other",
        ),
        ({"rollout_temperature": 1.1}, VLLMClient, re.escape("got {'rollout_temperature': 1.1}")),
        ({"rollout_top_p": 0.95}, VLLMClient, re.escape("got {'rollout_top_p': 0.95}")),
        # Each filter cuts or reshapes the sampling distribution the prefill echo never sees.
        ({"rollout_top_k": 20}, VLLMClient, re.escape("got {'rollout_top_k': 20}")),
        ({"rollout_min_p": 0.05}, VLLMClient, re.escape("got {'rollout_min_p': 0.05}")),
        ({"rollout_repetition_penalty": 1.1}, VLLMClient, re.escape("got {'rollout_repetition_penalty': 1.1}")),
    ],
)
def test_engine_reference_gate_refuses(kwargs, client_cls, match):
    with pytest.raises(ValueError, match=match):
        _gate(client_cls, **kwargs)


def test_both_engines_declare_the_rescore():
    assert VLLMClient.SUPPORTS_ENGINE_RESCORE is True
    assert SGLangWeightSyncClient.SUPPORTS_ENGINE_RESCORE is True
    _gate(SGLangWeightSyncClient)


def _word_tokenizer(text):
    """One id per whitespace-separated word, clear of the low special-token range."""
    return {"input_ids": [100 + int(word) for word in text.split()]}


@pytest.fixture
def prefill_server(client_cls):
    server = fake_server_class(client_cls)()
    try:
        yield server
    finally:
        server.close()


def _scored_lengths(server: FakeEngineServer) -> list[int]:
    """Token count of every prefill-log-prob request the server answered, in order, on either engine's route."""
    return [
        len(body["prompt"] if path == "/v1/completions" else body["input_ids"])
        for path, body in server.posted
        if path in ("/v1/completions", "/generate")
    ]


_ENGINES = pytest.mark.parametrize("client_cls", [VLLMClient, SGLangWeightSyncClient], ids=["vllm", "sglang"])


@_ENGINES
@pytest.mark.parametrize(
    ("shift", "capture_tokens"),
    [(-15.0, GRAPH_CAPTURE_TOKENS), (float("nan"), GRAPH_CAPTURE_TOKENS), (-15.0, 4096)],
    ids=["garbage-prefixes", "non-finite-prefixes", "garbage-whole"],
)
def test_prompt_logprob_probe_refuses_an_engine_whose_graph_prefills_are_garbage(
    prefill_server, client_cls, shift, capture_tokens
):
    """vLLM's V1 runner under MTP (measured on 0.26.0) returns garbage prompt log-probs whenever the prefill
    replays a captured graph. The probe's prefixes do while its whole sequence does not, so their shared
    positions disagree; a capture range raised past the whole sequence corrupts it too, and its mean NLL gives
    it away. Either way the run is refused with the cause and the remedies."""
    prefill_server.graph_prefill_shift = shift
    prefill_server.graph_capture_tokens = capture_tokens
    with pytest.raises(ValueError, match="speculative decoding") as raised:
        verify_engine_prompt_logprobs([prefill_server.url], _word_tokenizer, backend=client_cls.BACKEND_KEY)
    assert "Dockerfile.vllm" in str(raised.value) and "isr_engine_reference: false" in str(raised.value)


@_ENGINES
def test_prompt_logprob_probe_accepts_batch_shape_noise(prefill_server, client_cls):
    """A shift within bf16 batch-shape noise passes, once the probe has scored a whole sequence that prefills
    eagerly and prefixes of it that each replay a graph; the gap it measures is that shift alone, so each
    prefix position is compared with the same position of the whole."""
    prefill_server.graph_prefill_shift = -0.5
    verify_engine_prompt_logprobs([prefill_server.url], _word_tokenizer, backend=client_cls.BACKEND_KEY)
    whole, *prefixes = _scored_lengths(prefill_server)
    assert whole > GRAPH_CAPTURE_TOKENS, "the reference must prefill outside the graph capture range"
    assert prefixes and all(length <= GRAPH_CAPTURE_TOKENS for length in prefixes), prefixes

    _, gap, _ = _probe_prompt_logprobs(client_cls(base_url=prefill_server.url), list(range(100, 100 + whole)))
    assert gap == pytest.approx(0.5)


class _OneNaNPrefixPosition:
    """Every prefix scores one NaN log-prob behind finite ones; the whole sequence scores clean."""

    def score_completion_logprobs(self, prompt_ids, completion_ids, timeout):
        values = [-0.1] * len(completion_ids)
        if len(completion_ids) <= GRAPH_CAPTURE_TOKENS:
            values[3] = float("nan")
        return values


def test_prompt_logprob_probe_reads_a_non_finite_position_as_an_infinite_gap():
    """``max`` keeps whichever operand it met first when the other is NaN, so a NaN behind a finite gap
    would otherwise vanish from the verdict."""
    _, gap, _ = _probe_prompt_logprobs(_OneNaNPrefixPosition(), list(range(1100)))
    assert gap == math.inf


@_ENGINES
def test_a_preflight_request_never_retries_a_read_past_its_timeout(prefill_server, client_cls):
    """Every rank waits on rank 0's probes: a stalled server must cost one timeout per request, where the
    client's own session would retry the read five times."""
    url = prefill_server.url
    retry = engine_preflight._probe_server(
        client_cls, url, lambda client: client._session.get_adapter(url).max_retries, None, "probe"
    )
    assert retry.read == 0 and retry.connect > 0


def test_the_env_script_runs_the_prompt_logprob_probe_under_isr_engine_reference():
    """The probe guards the re-score only, so the script gates it on the knob that turns the re-score on."""
    guards = [
        ast.unparse(node.test)
        for node in ast.walk(ast.parse(ENV_SCRIPT.read_text()))
        if isinstance(node, ast.If)
        and any(
            isinstance(call, ast.Call) and ast.unparse(call.func) == "verify_engine_prompt_logprobs"
            for call in ast.walk(node)
        )
    ]
    assert guards == ["async_config.isr_engine_reference"], guards


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
