"""``isr_engine_reference``: the mask stages read the engine's current-vs-sampling log-ratio on
re-scored rows (pure staleness), the trainer diff elsewhere; the re-score fans a trajectory's rows out
to one server, never raises, and is refused at construction where it could not mean what it says, and
at startup against an engine whose prompt log-probs depend on how the prefill was batched."""

import ast
import inspect
import math
import re
import types
from collections import defaultdict
from unittest import mock

import pytest
import torch
from accelerate import PartialState

from src.configs.async_training_config import AsyncTrainingConfig
from src.distributed.nccl.clients.sglang import SGLangWeightSyncClient
from src.distributed.nccl.clients.vllm import VLLMWeightSyncClient as VLLMClient
from src.trainers.grpo import environmental
from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer
from src.trainers.grpo.objective.logratio import select_mask_logratio
from src.trainers.grpo.rollout import async_rollouts, weight_sync_clients
from src.trainers.grpo.rollout.weight_sync_clients import (
    _probe_prompt_logprobs,
    verify_engine_prompt_logprobs_synced,
)
from tests.common.grpo_metrics import attach_world_metrics, flushed_metrics
from tests.common.utils import REPO_ROOT
from tests.cpu.grpo.test_weight_sync_protocol import GRAPH_CAPTURE_TOKENS, FakeVLLMServer

PartialState()  # the re-score reports through accelerate's logger, which refuses to log without it

ENV_SCRIPT = REPO_ROOT / "scripts/training/environmental_grpo.py"


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


def _rescore_host(clients):
    host = types.SimpleNamespace(
        _engine_rescore_clients=clients,
        _weight_sync_client=None,  # every rank but the main one holds no sync client
        _metrics={"train": defaultdict(list)},
    )
    attach_world_metrics(host)
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
        _weight_sync_client=None,
        _train_loader_batch_size=lambda: 8,
        async_config=AsyncTrainingConfig(
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
    """Each rank re-scores its own rows, so a failure is rank-local, and accelerate's adapter drops every
    record off the main process by default: unless the report opts out, a dead route on rank 3 would never
    print."""
    recorder = mock.MagicMock()
    monkeypatch.setattr(environmental, "logger", recorder)
    monkeypatch.setattr(environmental, "get_global_rank", lambda: 3)
    rescore, _host = _rescore_host([_FakeClient(fail=fail) for fail in failing])
    prompts, completions = _rows()
    rescore(prompts, completions, [True, True, True, True], [2, 1, 1])
    getattr(recorder, silent).assert_not_called()
    (msg,), kwargs = getattr(recorder, level).call_args
    assert kwargs.get("main_process_only") is False, "the adapter's default drops a non-main rank's report"
    assert msg.startswith("[rank 3] isr_engine_reference:"), msg


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


def _gate_probe(*, is_correction=True, **sampler):
    """A trainer whose rollout config is the identity sampler, with ``sampler`` overriding its knobs."""
    trainer = DistributedAsyncEnvironmentalGRPOTrainer.__new__(DistributedAsyncEnvironmentalGRPOTrainer)
    trainer._is_correction = is_correction
    trainer.async_config = AsyncTrainingConfig(**{"rollout_temperature": 1.0, "rollout_top_p": 1.0, **sampler})
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
        _gate_probe(**kwargs)._validate_engine_reference(client_cls)


def test_both_engines_declare_the_rescore():
    assert VLLMClient.SUPPORTS_ENGINE_RESCORE is True
    assert SGLangWeightSyncClient.SUPPORTS_ENGINE_RESCORE is True
    _gate_probe()._validate_engine_reference(SGLangWeightSyncClient)


def _word_tokenizer(text):
    """One id per whitespace-separated word, clear of the low special-token range."""
    return {"input_ids": [100 + int(word) for word in text.split()]}


@pytest.fixture
def prefill_server():
    server = FakeVLLMServer()
    try:
        yield server
    finally:
        server.close()


def _scored_lengths(server: FakeVLLMServer) -> list[int]:
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
    """vLLM 0.26.0 under MTP returns garbage prompt log-probs whenever the prefill replays a captured graph.
    The probe's prefixes do while its whole sequence does not, so their shared positions disagree; a capture
    range raised past the whole sequence corrupts it too, and its mean NLL gives it away. Either way the run
    is refused with the cause and the remedies."""
    prefill_server.graph_prefill_shift = shift
    prefill_server.graph_capture_tokens = capture_tokens
    with pytest.raises(ValueError, match="speculative decoding") as raised:
        verify_engine_prompt_logprobs_synced([prefill_server.url], _word_tokenizer, backend=client_cls.BACKEND_KEY)
    assert "Dockerfile.vllm" in str(raised.value) and "isr_engine_reference: false" in str(raised.value)


@_ENGINES
def test_prompt_logprob_probe_accepts_batch_shape_noise(prefill_server, client_cls):
    """A shift within bf16 batch-shape noise passes, once the probe has scored a whole sequence that prefills
    eagerly and prefixes of it that each replay a graph; the gap it measures is that shift alone, so each
    prefix position is compared with the same position of the whole."""
    prefill_server.graph_prefill_shift = -0.5
    verify_engine_prompt_logprobs_synced([prefill_server.url], _word_tokenizer, backend=client_cls.BACKEND_KEY)
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
    retry = weight_sync_clients._probe_server(
        client_cls, url, lambda client: client.session.get_adapter(url).max_retries, None, "probe"
    )
    assert retry.read == 0 and retry.connect > 0


def test_the_env_script_runs_the_prompt_logprob_probe_under_isr_engine_reference():
    """The probe guards the re-score only, so the script gates it on the knob that turns the re-score on."""
    guards = [
        ast.unparse(node.test)
        for node in ast.walk(ast.parse(ENV_SCRIPT.read_text()))
        if isinstance(node, ast.If)
        and any(
            isinstance(call, ast.Call) and ast.unparse(call.func) == "verify_engine_prompt_logprobs_synced"
            for call in ast.walk(node)
        )
    ]
    assert guards == ["async_config.isr_engine_reference"], guards


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
