"""The live half of ``rollout_max_thinking_tokens``: is the budget enforced, and did a turn respect it.

The budget is a vLLM-only request field — ``generation_control_fields``
(:mod:`src.environments.engine_wire`) puts ``thinking_token_budget`` at the top level of the chat
request — and the ENGINE enforces it: nothing in the toolkit truncates or masks a turn's CoT, and
reasoning ids are not captured separately from the turn's sampled ids. So both halves need a live
server: whether this one honours the field at all, and what the rollouts then reasoned.

The server needs a reasoning parser (gpt-oss: ``--reasoning-parser-plugin
/opt/gpt_oss_reasoning_parser.py --reasoning-parser openai_gptoss``) and ``VLLM_USE_V2_MODEL_RUNNER=0``;
missing either it answers every request carrying the field with a 400.
"""

import requests

from src.inference.response import get_reasoning_text
from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer
from tests.common.on_policy_e2e import PROBE_TIMEOUT_S
from tests.common.utils import log

# Long enough that an unbudgeted answer reasons past the budget under test — the A/B is only a
# measurement of enforcement while the model would otherwise have overrun it.
_PROBE_PROMPT = "Multiply 3847 by 2913 by hand, showing every intermediate product before the final answer."
_PROBE_MAX_TOKENS = 768
# Sampled, seeded attempts: greedy decoding on an easy prompt answers with no reasoning channel at
# all, which measures nothing. Each attempt pairs a free and a budgeted request on one seed; the
# probe needs one attempt whose FREE run overruns the budget, and says so loudly when none does.
_PROBE_ATTEMPTS = 4
_PROBE_TEMPERATURE = 1.0
# Enough of a refused server's body to carry its reason into the log; the rest is a JSON envelope.
_DETAIL_CHARS = 400
# The engine cuts on generated ids while every count here re-tokenizes the decoded reasoning text,
# and the two disagree by at most a boundary token.
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
        # Top-level, exactly as generation_control_fields sends it: this spelling is what the engines
        # derive their thinking toggles from — a request without it renders with thinking off and no
        # reasoning channel ever opens, whatever the sampling.
        "reasoning_effort": "high",
    }
    if budget is not None:
        payload["thinking_token_budget"] = budget
    return requests.post(f"{server_url}/v1/chat/completions", json=payload, timeout=PROBE_TIMEOUT_S)


def _cot(server_url: str, model_name: str, budget: int | None, seed: int) -> str | None:
    """One request's CoT, read the way the rollout path reads it (:func:`get_reasoning_text`).

    vLLM answers ``reasoning`` and SGLang ``reasoning_content``; a probe reading one spelling scores
    an enforced budget and an ignored one identically, as zero reasoning tokens.
    """
    return get_reasoning_text(_chat(server_url, model_name, budget, seed).json()["choices"][0]["message"])


def record_budget_enforcement(server_url: str, model_name: str, budget: int, tokenizer, checks: dict) -> None:
    """RANK 0: the server takes the field, and the SAME prompt reasons less with it than without.

    An acceptance check alone would pass on an engine that parses the field and ignores it, since the
    arithmetic prompts a rollout runs answer well under any sane budget. The A/B measures enforcement
    directly and needs no rollout; its unbudgeted arm is also what proves this checkpoint opens a
    reasoning channel at all.
    """
    probe = _chat(server_url, model_name, budget, seed=0)
    checks["server_accepts_the_thinking_budget"] = probe.status_code == requests.codes.ok
    if probe.status_code != requests.codes.ok:
        log(f"  thinking_token_budget={budget} refused: HTTP {probe.status_code} {probe.text[:_DETAIL_CHARS]}")
        checks["server_enforced_the_thinking_budget"] = False
        return
    # Elicitation and enforcement are separate verdicts: an attempt set whose free arm never overran
    # measured nothing, and folding that into "not enforced" would blame the engine for the probe.
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
    """The environmental trainer, keeping each sampled assistant turn's CoT length.

    ``_log_rollout_metrics`` is the one place the trainer sees a step's episodes on every rank before
    they are reduced to means, and it already owns the per-turn count. A mean would not do: the budget
    bounds each TURN, and a mean of per-episode sums can sit under it while a turn ran over.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampled_reasoning_tokens: list[int] = []

    def _log_rollout_metrics(self, results, mode: str):
        for result in results:
            if result.trajectory is not None:
                self.sampled_reasoning_tokens.extend(self._assistant_turn_reasoning_tokens(result.trajectory))
        return super()._log_rollout_metrics(results, mode)


def record_thinking_budget_checks(trainer: ReasoningTurnRecorder, checks: dict[str, bool], budget: int) -> None:
    """Every sampled turn reasoned within ``budget``, and at least one of them reasoned at all.

    Rank-local: each rank grades the turns it collected, so a rank whose engine ignored the budget
    fails on its own rather than behind rank 0's verdict.
    """
    counts = trainer.sampled_reasoning_tokens
    checks["thinking_budget_sampled_a_turn"] = bool(counts)
    # ANTI-VACUITY: a run whose turns emitted no CoT satisfies "nothing exceeded the budget" trivially.
    checks["thinking_budget_some_turn_reasoned"] = any(count > 0 for count in counts)
    checks["thinking_budget_capped_every_turn"] = all(count <= budget + RETOKENIZE_SLACK for count in counts)
    log(f"  reasoning tokens over {len(counts)} sampled turn(s), budget {budget}: {counts}")
