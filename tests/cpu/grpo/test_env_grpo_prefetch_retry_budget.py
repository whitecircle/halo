#!/usr/bin/env python
"""Env-GRPO refuses a prefetched round that spans more weight syncs than a turn's retries on an engine whose
pause aborts the turns in flight.

The refusal reads ``PAUSE_ABORTS_IN_FLIGHT`` off the configured backend's client: SGLang's pause aborts the turns
in flight, vLLM's keeps them. The rollout actor re-issues an aborted turn up to ``max_retries`` times before its
episode fails. A prefetched round is in flight from the step that submits it to the one that trains it
``prefetch_depth`` rounds later, ``steps_per_generation × num_iterations ÷ gradient_accumulation_steps`` optimizer
steps per round, so with a sync every ``sync_weights_every_n_steps`` it can span the ceiling of their quotient.
Above ``max_retries`` the run is refused at trainer construction; a run that does not prefetch rolls no round out
across a sync, and one server that opts in (``enable_prefetch: true``) spans the syncs as several do. Each client
with that one fact flipped takes the other verdict.

Every shipped env-GRPO recipe passes as shipped; each shipped SGLang recipe is refused at ``prefetch_depth: 2``
and passes the same depth on vLLM, with syncs half as often, or with a retry per sync. The script runs the same
refusal ahead of the model load.

    python tests/cpu/grpo/test_env_grpo_prefetch_retry_budget.py
"""

import dataclasses
from types import SimpleNamespace

import pytest
from trl import GRPOConfig

import scripts.training.grpo.environmental as env_script
from halo.configs.grpo_environmental import AsyncRolloutConfig
from halo.environments.rollout_config import SGLANG_BACKEND
from halo.trainers.grpo.environmental.environmental_rules import reject_prefetch_past_retry_budget
from halo.weight_sync.registry import resolve_weight_sync_client, rollout_backends
from tests.cpu.config.test_examples_parse import (
    ENV_GRPO_EXAMPLES,
    ENV_GRPO_SCRIPT,
    EXAMPLES_ROOT,
    _bf16_capable,  # noqa: F401 — autouse fixture: validate the configs as a training host does
    parser_for,
)

_SGLANG_EXAMPLES = sorted((EXAMPLES_ROOT / "grpo" / "environmental").glob("*/sglang/*.yaml"))
# The shipped SGLang recipe the knob-by-knob cases vary.
_GPTOSS_SGLANG = EXAMPLES_ROOT / "grpo/environmental/gptoss/sglang/gptoss-20b-code-contests-full-ep1.yaml"


def _recipe_id(config) -> str:
    return config.relative_to(EXAMPLES_ROOT / "grpo" / "environmental").as_posix()


class _ModelLoadReached(Exception):
    """Raised by the stand-in model load: the script got past every pre-load gate."""


def _script_parse(config, *overrides: str) -> tuple:
    """Every dataclass the script parses, as a launch of the recipe with ``overrides`` builds them."""
    parser = parser_for(ENV_GRPO_SCRIPT)
    return parser.parse_dict(parser.load_values([str(config), *overrides]))


def _parsed(config, *overrides: str) -> tuple[AsyncRolloutConfig, GRPOConfig]:
    """The recipe's rollout and GRPO configs as a launch with ``overrides`` builds them."""
    parsed = _script_parse(config, *overrides)
    async_config = next(obj for obj in parsed if isinstance(obj, AsyncRolloutConfig))
    return async_config, next(obj for obj in parsed if isinstance(obj, GRPOConfig))


def _gate(async_config: AsyncRolloutConfig, args: GRPOConfig) -> None:
    """The refusal as the trainer and the script run it: against the configured backend's client."""
    reject_prefetch_past_retry_budget(resolve_weight_sync_client(async_config.rollout_backend), async_config, args)


def test_the_sglang_recipes_exist():
    assert _SGLANG_EXAMPLES, "no shipped SGLang recipe found; the sweeps below would check nothing"
    assert _GPTOSS_SGLANG in _SGLANG_EXAMPLES


@pytest.mark.parametrize("config", ENV_GRPO_EXAMPLES, ids=_recipe_id)
def test_every_shipped_env_grpo_recipe_passes_as_shipped(config):
    _gate(*_parsed(config))


@pytest.mark.parametrize(
    ("config", "overrides", "syncs"),
    [
        *((config, ["--prefetch_depth=2"], 2) for config in _SGLANG_EXAMPLES),
        (_GPTOSS_SGLANG, ["--num_iterations=2"], 2),
        (_GPTOSS_SGLANG, ["--steps_per_generation=48"], 2),
        (_GPTOSS_SGLANG, ["--max_retries=0"], 1),
        # A span that is no whole number of sync periods rounds up.
        (_GPTOSS_SGLANG, ["--prefetch_depth=2", "--sync_weights_every_n_steps=5", "--max_retries=0"], 1),
        (_GPTOSS_SGLANG, ["--steps_per_generation=40"], 2),
    ],
    ids=[
        *(f"{_recipe_id(config)}-depth2" for config in _SGLANG_EXAMPLES),
        "two-iterations-per-round",
        "two-optimizer-steps-per-round",
        "no-retry",
        "two-fifths-of-a-sync-period-in-flight",
        "five-thirds-of-a-sync-period-per-round",
    ],
)
def test_an_sglang_round_spanning_more_syncs_than_its_retry_budget_is_refused(config, overrides, syncs):
    async_config, args = _parsed(config, *overrides)
    assert async_config.rollout_backend == SGLANG_BACKEND and async_config.prefetch_active()
    with pytest.raises(ValueError, match=rf"a prefetched round stays in flight across up to {syncs} syncs"):
        _gate(async_config, args)


@pytest.mark.parametrize(
    ("config", "overrides"),
    [
        *((config, ["--prefetch_depth=2", "--rollout_backend=vllm"]) for config in _SGLANG_EXAMPLES),
        (_GPTOSS_SGLANG, ["--prefetch_depth=2", "--sync_weights_every_n_steps=2"]),
        (_GPTOSS_SGLANG, ["--prefetch_depth=2", "--max_retries=2"]),
        (_GPTOSS_SGLANG, ["--prefetch_depth=2", "--steps_per_generation=8"]),
    ],
    ids=[
        *(f"{_recipe_id(config)}-depth2-on-vllm" for config in _SGLANG_EXAMPLES),
        "sync-every-two-steps",
        "a-retry-per-sync",
        "three-rounds-per-optimizer-step",
    ],
)
def test_a_round_within_its_retry_budget_or_on_vllm_passes(config, overrides):
    async_config, args = _parsed(config, *overrides)
    assert async_config.prefetch_depth == 2 and async_config.prefetch_active()
    _gate(async_config, args)


def test_a_single_sglang_server_at_the_default_prefetch_passes_any_round_length():
    """One server at the default ``enable_prefetch`` does not prefetch: every round collects on the trainer thread
    between syncs, so no sync lands on it however many steps it trains."""
    async_config, args = _parsed(_GPTOSS_SGLANG, "--num_iterations=2")
    async_config = dataclasses.replace(
        async_config, rollout_server_configs=None, enable_prefetch=AsyncRolloutConfig.enable_prefetch
    )
    assert not async_config.prefetch_active()
    _gate(async_config, args)


@pytest.mark.parametrize(
    ("overrides", "syncs"),
    [(["--prefetch_depth=2"], 2), (["--num_iterations=2"], 2), ([], None)],
    ids=["depth2", "two-iterations-per-round", "depth1"],
)
def test_a_single_sglang_server_that_opts_in_is_held_to_the_same_budget(overrides, syncs):
    """``enable_prefetch: true`` prefetches on one server, whose every sync aborts the round's turns as two servers'
    do: the same span is refused, and the shipped one-sync span passes."""
    async_config, args = _parsed(_GPTOSS_SGLANG, "--enable_prefetch=true", *overrides)
    async_config = dataclasses.replace(async_config, rollout_server_configs=None)
    assert async_config.prefetch_active() and len(async_config.get_server_urls()) == 1
    if syncs is None:
        _gate(async_config, args)
        return
    with pytest.raises(ValueError, match=rf"a prefetched round stays in flight across up to {syncs} syncs"):
        _gate(async_config, args)


@pytest.mark.parametrize("backend", rollout_backends())
def test_the_refusal_follows_the_clients_declared_pause(backend):
    """The same depth-2 round, two syncs on one retry, against the backend's client and a copy of it with only
    ``PAUSE_ABORTS_IN_FLIGHT`` flipped: refused exactly under the client that declares an aborting pause."""
    async_config, args = _parsed(_GPTOSS_SGLANG, "--prefetch_depth=2", f"--rollout_backend={backend}")
    client = resolve_weight_sync_client(backend)
    flipped = type(
        f"Flipped{client.__name__}", (client,), {"PAUSE_ABORTS_IN_FLIGHT": not client.PAUSE_ABORTS_IN_FLIGHT}
    )
    aborting, keeping = (client, flipped) if client.PAUSE_ABORTS_IN_FLIGHT else (flipped, client)
    with pytest.raises(ValueError, match=f"Every {aborting.BACKEND_NAME} weight sync aborts the turns in flight"):
        reject_prefetch_past_retry_budget(aborting, async_config, args)
    reject_prefetch_past_retry_budget(keeping, async_config, args)


@pytest.mark.parametrize(
    ("overrides", "refused"), [(["--prefetch_depth=2"], True), ([], False)], ids=["depth2", "shipped"]
)
def test_the_script_refuses_before_the_model_load(monkeypatch, overrides, refused):
    """The trainer's refusal follows the policy load, minutes on a large model on every rank; the script runs it
    first, and lets the shipped recipe through to the load."""
    parsed = _script_parse(_GPTOSS_SGLANG, *overrides)

    def model_load(*_args, **_kwargs):
        raise _ModelLoadReached

    monkeypatch.setattr(env_script, "parse_script_args", lambda *_: parsed)
    monkeypatch.setattr(env_script, "init_training_script", lambda *a, **k: SimpleNamespace(parallelism_config=None))
    monkeypatch.setattr(env_script, "load_model_and_tokenizer", model_load)
    expected = (
        pytest.raises(ValueError, match="a prefetched round stays in flight across up to 2 syncs")
        if refused
        else pytest.raises(_ModelLoadReached)
    )
    with expected:
        env_script.main()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
