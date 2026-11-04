"""The live half of ``rollout_max_thinking_tokens``: is the budget enforced, and did a turn respect it. The engine
enforces this vLLM-only field, so both halves need a live server with a reasoning parser; without one it 400s every
request carrying the field."""

from collections.abc import Callable

import requests

from halo.inference.response import get_reasoning_text
from halo.trainers.grpo.environmental.trainer import DistributedAsyncEnvironmentalGRPOTrainer
from tests.common.on_policy_e2e import PROBE_TIMEOUT_S
from tests.common.rank_log import log

# Long enough that an unbudgeted answer reasons past the budget; the A/B measures enforcement only while the model
# would overrun.
_PROBE_PROMPT = "Multiply 3847 by 2913 by hand, showing every intermediate product before the final answer."
_PROBE_MAX_TOKENS = 768
# Sampled and seeded, since greedy on an easy prompt opens no reasoning channel. Each attempt pairs a free and a
# budgeted request on one seed; the probe needs one attempt whose free run overruns the budget.
_PROBE_ATTEMPTS = 4
_PROBE_TEMPERATURE = 1.0
# Enough of a refused server's body to carry its reason; the rest is a JSON envelope.
_DETAIL_CHARS = 400
# The engine cuts on generated ids while these counts re-tokenize decoded text, and the two disagree by at most a
# boundary token.
RETOKENIZE_SLACK = 1


def _reasoning_tokens(tokenizer, text: str | None) -> int:
    """Length of one turn's CoT, counted the way the trainer's own metrics count it."""
    return len(tokenizer(text, add_special_tokens=False)["input_ids"]) if text else 0


def _chat(server_url: str, model_name: str, budget: int | None, seed: int) -> requests.Response:
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": _PROBE_PROMPT}],
        "max_tokens": _PROBE_MAX_TOKENS,
        "temperature": _PROBE_TEMPERATURE,
        "seed": seed,
        # vLLM sets the template's enable_thinking from this, without which Gemma 4's opt-in template opens no
        # reasoning channel. gpt-oss reads it as the harmony level; Qwen3/3.5/3.6 think either way unless
        # enable_thinking is false.
        "reasoning_effort": "high",
    }
    if budget is not None:
        payload["thinking_token_budget"] = budget
    return requests.post(f"{server_url}/v1/chat/completions", json=payload, timeout=PROBE_TIMEOUT_S)


def _cot(server_url: str, model_name: str, budget: int | None, seed: int) -> str | None:
    """One request's CoT; vLLM answers ``reasoning`` and SGLang ``reasoning_content``, and reading one spelling scores
    zero on the other."""
    return get_reasoning_text(_chat(server_url, model_name, budget, seed).json()["choices"][0]["message"])


def record_budget_acceptance(server_url: str, model_name: str, budget: int, checks: dict) -> bool:
    """RANK 0: whether the server takes the field; a refusal also fails enforcement. Before the load, since a server
    with no parser 400s and the run would train on all-masked batches. Acceptance alone leaves
    ``thinking_budget_probe_ran`` failed."""
    probe = _chat(server_url, model_name, budget, seed=0)
    accepted = probe.status_code == requests.codes.ok
    checks["server_accepts_the_thinking_budget"] = accepted
    if accepted:
        checks["thinking_budget_probe_ran"] = False
    else:
        checks["server_enforced_the_thinking_budget"] = False
        log(f"  thinking_token_budget={budget} refused: HTTP {probe.status_code} {probe.text[:_DETAIL_CHARS]}")
    return accepted


def record_budget_enforcement(server_url: str, model_name: str, budget: int, tokenizer, checks: dict) -> None:
    """RANK 0: the same prompt reasons less with the field than without it, on the policy the server serves. Acceptance
    alone would pass an engine that ignores the field. It must read the checkpoint's own policy: a trained gpt-oss
    writes turns with no analysis channel, which the budget has nothing to cap."""
    # Separate verdicts: a free arm that never overran measured nothing, and should not blame the engine.
    for attempt in range(_PROBE_ATTEMPTS):
        free = _reasoning_tokens(tokenizer, _cot(server_url, model_name, None, attempt))
        if free > budget + RETOKENIZE_SLACK:
            capped = _reasoning_tokens(tokenizer, _cot(server_url, model_name, budget, attempt))
            checks["thinking_budget_probe_elicited_reasoning"] = True
            checks["server_enforced_the_thinking_budget"] = capped <= budget + RETOKENIZE_SLACK
            log(f"  seed {attempt}: {free} reasoning tokens free, {capped} with budget {budget}")
            return
        log(f"  seed {attempt}: free run reasoned {free} tokens (needs > {budget + RETOKENIZE_SLACK}) — retrying")
    checks["thinking_budget_probe_elicited_reasoning"] = False
    log(f"  no free run overran budget {budget} in {_PROBE_ATTEMPTS} attempts — the probe measured nothing")


class ReasoningTurnRecorder(DistributedAsyncEnvironmentalGRPOTrainer):
    """The environmental trainer, keeping each turn's CoT length sampled by the checkpoint's own policy, before the
    first optimizer step. The budget bounds each turn, and a mean can sit under it while a turn ran over. Later steps
    leave the policy writing malformed turns that never open the channel the engine counts in, while the parser still
    reports them as reasoning."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampled_reasoning_tokens: list[int] = []
        # Rank 0's enforcement A/B, set by the e2e; it runs once, right after the first push.
        self.budget_probe: Callable[[], None] | None = None

    def _start_rollout_generation(self):
        """Probe the budget once the engine serves this checkpoint: a tier's server keeps the previous row's policy
        until the first push."""
        first_push = not self._rollout_generation_started
        super()._start_rollout_generation()
        if first_push and self.budget_probe is not None:
            self.budget_probe()

    def _log_rollout_metrics(self, results, mode: str, reasoning_tokens):
        if self.state.global_step == 0:
            for tokens in reasoning_tokens:
                self.sampled_reasoning_tokens.extend(tokens)
        return super()._log_rollout_metrics(results, mode, reasoning_tokens)


def record_thinking_budget_checks(trainer: ReasoningTurnRecorder, checks: dict[str, bool], budget: int) -> None:
    """Every turn the policy sampled reasoned within ``budget``, and one reasoned at all. Rank-local, so a rank fails
    on its own."""
    counts = trainer.sampled_reasoning_tokens
    checks["thinking_budget_sampled_a_turn"] = bool(counts)
    # A run whose turns emitted no CoT trivially satisfies the budget.
    checks["thinking_budget_some_turn_reasoned"] = any(count > 0 for count in counts)
    checks["thinking_budget_capped_every_turn"] = all(count <= budget + RETOKENIZE_SLACK for count in counts)
    log(f"  reasoning tokens over {len(counts)} sampled turn(s), budget {budget}: {counts}")
