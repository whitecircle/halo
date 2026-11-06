"""COLLECTIVE preflights of each rollout server: its context window, its launch flags, and whether its log-probs are the
reference the IS ratio divides by. Rank 0 waits for every server, as group formation would, so a loading one cannot pass unread; its
refusal raises everywhere, while a server that is up and then cannot answer a probe only warns. Rank 0 probes the servers
concurrently and the verdict reaches the peers over the c10d store, so a slow startup is never bounded by the NCCL
watchdog."""

import logging
import math
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
from typing import Any, NamedTuple, TypeVar

import requests
from urllib3.util.retry import Retry

from halo.env import env_positive_int
from halo.environments.engine_wire import build_payload, capture_generation_tokens
from halo.environments.rollout_config import SAMPLER_FILTERS_OFF, RolloutConfig
from halo.model_loading.tokenizer_setup import resolve_tokenizer
from halo.runtime.consensus import refusal_text, store_run_on_rank0_raising_everywhere
from halo.weight_sync.clients.base import ServerFlags, WeightSyncClientBase
from halo.weight_sync.group_eligibility import find_fleet_disagreement, find_unread_server
from halo.weight_sync.registry import resolve_weight_sync_client

logger = logging.getLogger(__name__)

# Retries a refused connection or a 5xx, never a read past its timeout: every rank waits on rank 0's probes.
_PROBE_RETRY = Retry(
    total=3,
    connect=3,
    read=0,
    status=3,
    status_forcelist=[500, 502, 503],
    backoff_factor=1,
    allowed_methods=["POST", "GET"],
)
_PROBE_TIMEOUT_S = 60.0
# Servers probed at once: a thread per server is HTTP only, but 64 simultaneous /server_info reads are a herd.
_PREFLIGHT_PARALLELISM = env_positive_int("HALO_PREFLIGHT_PARALLELISM", 16)
_T = TypeVar("_T")

# A timed generation long enough that prefill and the HTTP round trip stay a small share of it, and the fewest tokens
# a reply may carry for its rate to count.
_RATE_PROBE_TOKENS = 64
_RATE_PROBE_MIN_TOKENS = 32
# One idle sequence decodes faster than the same sequence amid a rollout round, so a turn needing over this share of
# its deadline at the idle rate is warned about, and one needing over this multiple of it is refused outright.
_DEADLINE_WARN_SHARE = 0.5
_DEADLINE_REFUSE_MULTIPLE = 2.0

# The drafter overwrite moves every position by 8-22 nats. Batch-shape bf16 noise stays under ~0.5 nats where the
# model is sure of the token but reaches several nats at low-probability ones (Gemma 4 on vLLM, SGLang and the
# trainer alike), so the gap is judged over the positions the whole sequence scores above the confidence floor.
_PROMPT_LOGPROB_TOLERANCE_NATS = 2.0
_PROMPT_LOGPROB_CONFIDENT = -1.0
# Any servable model all but predicts a counting sequence, while garbage prompt log-probs average 12-22
# nats: the bound catches a whole sequence that itself prefilled as a graph (a raised capture range).
_PROMPT_LOGPROB_PROBE_MAX_MEAN_NLL = 4.0

_PER_TOKEN_RATIO_REMEDY = (
    "take the ratio per token: on online GRPO a token_* vllm_importance_sampling_mode with a loss_type other than "
    "vespo; on the environmental arm, whose correction is per token already, drop isr_geo_band_min/max and "
    "isr_opsm_delta"
)


class AdapterRequirements(NamedTuple):
    """What serving the policy as an adapter asks of a server: the largest rank and the module names its adapters hold,
    and how many versions it keeps loaded at once."""

    rank: int
    modules: tuple[str, ...]
    live_versions: int


class ServerRequirements(NamedTuple):
    """What a run needs of its servers' launch flags. ``reasoning_consumers`` names the knobs reading a reasoning
    parser."""

    is_moe: bool
    sends_tools: bool = False
    captures_routing: bool = False
    reasoning_consumers: tuple[str, ...] = ()
    sampled_token_ids: bool = False


def _no_tool_parser(client_cls: type[WeightSyncClientBase]) -> str:
    return (
        f"it serves no tool parser and the environment sends tools, so calls stay text, each turn reads as a final "
        f"answer and every episode scores alike with no error. Serve with {client_cls.TOOL_PARSER_FLAG}."
    )


def _server_flag_refusals(
    client_cls: type[WeightSyncClientBase], flags: ServerFlags, needs: ServerRequirements
) -> list[str]:
    """Each flag a run needs and a server reports wrong; a field the reply does not carry reads as unread."""
    refusals = []
    if not flags.toolkit_image:
        refusals.append(
            f"it does not report the toolkit's image, so updates would land wrong with no error. "
            f"{client_cls.IMAGE_REMEDY}"
        )
    if needs.is_moe and flags.moe_backend not in (None, "triton"):
        refusals.append(
            f"its MoE backend is {flags.moe_backend!r}, which repacks expert weights, so every push would corrupt "
            f"them behind a 200. Serve with {client_cls.MOE_BACKEND_FLAG}."
        )
    if needs.sends_tools and flags.tool_call_parser == "":
        refusals.append(_no_tool_parser(client_cls))
    if needs.sends_tools and flags.strict_tool_calling:
        refusals.append(
            "VLLM_ENFORCE_STRICT_TOOL_CALLING is on, so a parser with a default strict level samples tool-bearing "
            "requests under a grammar the trainer's log-probs never see. Set it to 0, as the image does."
        )
    if needs.captures_routing and flags.captures_routing is False:
        refusals.append(
            "routing_replay: rollout needs the routed experts, which it does not capture. Serve with "
            "--enable-return-routed-experts."
        )
    if needs.captures_routing and flags.torch_compile:
        refusals.append("routing_replay: rollout under --enable-torch-compile returns corrupt rows. Turn it off.")
    if needs.reasoning_consumers and flags.reasoning_parser == "":
        refusals.append(
            f"it has no reasoning parser and the run reads reasoning ({', '.join(needs.reasoning_consumers)}): a "
            f"reasoning cap would answer 400 on every rollout, the price charge nothing and the floor score every "
            f"episode as maximal under-use. Serve with {client_cls.REASONING_PARSER_FLAG}, or turn them off."
        )
    return refusals


def _adapter_refusals(flags: ServerFlags, needs: AdapterRequirements) -> list[str]:
    """Each launch fact that would make a server serve the adapter wrong or not at all; an unreported one reads as
    unread."""
    facts = flags.extras
    refusals = []
    if not facts.get("enable_lora"):
        refusals.append("it was launched without --enable-lora, so it can load no adapter.")
    if (rank := facts.get("max_lora_rank")) is not None and int(rank) < needs.rank:
        refusals.append(f"its --max-lora-rank is {rank}, below the run's lora_r {needs.rank}.")
    served = facts.get("lora_target_modules")
    if served is not None and "all" not in served and (missing := sorted(set(needs.modules) - set(served))):
        refusals.append(
            f"its --lora-target-modules {sorted(served)} omit {missing}, which the engine would drop from the adapter "
            f"without a trace."
        )
    loaded = facts.get("max_loaded_loras")
    if loaded is not None and int(loaded) < needs.live_versions:
        refusals.append(
            f"its --max-loaded-loras is {loaded}, below the {needs.live_versions} versions the run keeps live, so a "
            f"load would evict one a round still runs under."
        )
    per_batch = facts.get("max_loras_per_batch")
    if per_batch is not None and int(per_batch) < 2:
        refusals.append(
            f"its --max-loras-per-batch is {per_batch}, so rounds on two versions could not share a batch."
        )
    if facts.get("speculative_algorithm"):
        refusals.append("it serves with speculative decoding, whose draft runs without the adapter.")
    if flags.quantization:
        refusals.append(f"its base is quantized ({flags.quantization}), not the trainer's.")
    if int(facts.get("dp_size") or 1) != 1:
        refusals.append("it runs data-parallel replicas, which cannot load an adapter at runtime.")
    remedy = (
        " Serve with the adapter flags (the SGLANG_LORA_MAX_RANK launch variable of the SGLang compose service), or "
        "set rollout_adapter_sync: false."
    )
    return [refusal + remedy for refusal in refusals]


def _verify_adapter_servers_on_rank0(
    client_cls: type[WeightSyncClientBase], urls: list[str], needs: AdapterRequirements
) -> None:
    readings = _probe_servers(client_cls, urls, client_cls.read_server_flags, None, "read the launch flags of")
    for url, flags in zip(urls, readings, strict=True):
        if flags is None:
            continue
        if refusals := _adapter_refusals(flags, needs):
            raise ValueError(f"Rollout server {url} cannot serve the policy as an adapter: " + " ".join(refusals))
        logger.info(f"Rollout server {url} serves adapters: {dict(flags.extras)}")


def _engine_prompt_ids(
    client: WeightSyncClientBase, rollout: RolloutConfig, messages: list[dict], efforts: list
) -> list[list[int] | None]:
    """The prompt ids the server renders for ``messages`` under each effort's level, one 1-token turn each."""
    prompts = []
    for effort in efforts:
        payload = build_payload(messages, rollout, effort.level if effort else None)
        reply = client.request(
            "POST", "/v1/chat/completions", retry=True, timeout=_PROBE_TIMEOUT_S, json=payload
        ).json()
        prompts.append(capture_generation_tokens(reply["choices"][0], reply, rollout.backend)[2])
    return prompts


def _verify_server_chat_template_on_rank0(
    client_cls: type[WeightSyncClientBase],
    urls: list[str],
    trainer_prompt_ids: Callable[[Any], list[int]],
    messages: list[dict],
    efforts: list,
    rollout: RolloutConfig,
) -> None:
    try:
        trainer = [trainer_prompt_ids(effort) for effort in efforts]
    except Exception as e:  # a template's own raise, here on a contract the run may never draw
        logger.warning(f"Could not render the trainer's prompt for the chat-template probe, so it is skipped: {e}")
        return
    probe = partial(_engine_prompt_ids, rollout=rollout, messages=messages, efforts=efforts)
    engines = _probe_servers(client_cls, urls, probe, None, "read the prompt ids of")
    for url, engine in zip(urls, engines, strict=True):
        if engine is None or None in engine:
            continue
        if trainer[0] != trainer[-1] and engine[0] == engine[-1]:
            raise ValueError(
                f"Rollout server {url} renders the same prompt whatever the episode's reasoning effort and budget, and "
                f"the trainer's chat template does not: the server's template ignores the level, so the reasoning "
                f"terms would charge for a contract the model never read. Serve the file chat_template: pins "
                f"({client_cls.CHAT_TEMPLATE_FLAG})."
            )
        if engine[0] != trainer[0]:
            shared = min(len(engine[0]), len(trainer[0]))
            at = next((i for i in range(shared) if engine[0][i] != trainer[0][i]), shared)
            logger.warning(
                f"Rollout server {url} renders this prompt differently from the trainer, first at token {at} of "
                f"{len(engine[0])}/{len(trainer[0])}. Rows train on the engine's ids, so this is harmless where the "
                f"engine only derives a thinking toggle, and a re-rendered path scores a prompt never generated "
                f"where the templates differ: check {client_cls.CHAT_TEMPLATE_FLAG} against chat_template:."
            )


def _verify_server_flags_on_rank0(
    client_cls: type[WeightSyncClientBase], urls: list[str], needs: ServerRequirements
) -> None:
    readings = _across_servers(partial(_read_server_facts, client_cls, needs=needs), urls)
    for url, (flags, accepted, sampled) in zip(urls, readings, strict=True):
        refusals = [] if flags is None else _server_flag_refusals(client_cls, flags, needs)
        if accepted is False:
            refusals.append(_no_tool_parser(client_cls))
        if sampled is False:
            refusals.append(
                "train_on_sampled_tokens needs the ids a turn sampled, and its logprobs carry no token_id: "
                "entries, so every turn would train on a re-tokenized render. Serve with "
                "--return-tokens-as-token-ids, or set train_on_sampled_tokens: false."
            )
        if refusals:
            raise ValueError(f"Rollout server {url} cannot serve this run: " + " ".join(refusals))
        logger.info(
            f"Rollout server {url}: launch flags {'unread, the server did not answer' if flags is None else flags}"
        )


def _verify_group_servers_on_rank0(client_cls: type[WeightSyncClientBase], urls: list[str]) -> None:
    """One group broadcasts one set of tensors to every server, so servers that disagree on the model or on a launch
    flag (dtype, quantization, MoE backend, parsers) would each load or serve it differently behind a 200. A flag a
    server does not report is not compared."""
    models = _probe_servers(client_cls, urls, client_cls.served_model_id, None, "read the served model of")
    launch_flags = _probe_servers(client_cls, urls, client_cls.read_server_flags, None, "read the launch flags of")
    if (unread := find_unread_server(urls, models, launch_flags)) is not None:
        raise ValueError(
            f"Rollout server {unread} did not answer for its served model and launch flags, and "
            f"rollout_sync_topology: single_group cannot tell whether it loads what the group broadcasts. "
            f"Start every server before the trainer, or set rollout_sync_topology: per_server."
        )
    if (disagreement := find_fleet_disagreement(urls, models, launch_flags)) is not None:
        what, found = disagreement
        raise ValueError(
            f"Rollout servers disagree on their {what} ({found}), but rollout_sync_topology: single_group "
            f"broadcasts one set of tensors to all of them. Serve one {what}, or set "
            f"rollout_sync_topology: per_server."
        )


def _probe_top_logprobs(
    client: WeightSyncClientBase, model_id: str, temperature: float, top_p: float, filters: Mapping[str, int | float]
) -> list[float] | None:
    """The descending top logprobs of one 1-token completion, ``None`` when unreadable."""
    body = {
        "model": model_id,
        "prompt": "The capital of France is",
        "max_tokens": 1,
        "temperature": temperature,
        "top_p": top_p,
        **filters,
        "logprobs": 2,
        "seed": 0,
    }
    try:
        resp = client.request("POST", "/v1/completions", retry=True, timeout=_PROBE_TIMEOUT_S, json=body)
        top = resp.json()["choices"][0]["logprobs"]["top_logprobs"][0]
        values = sorted((float(v) for v in top.values() if v is not None), reverse=True)
    except (requests.exceptions.RequestException, KeyError, IndexError, TypeError, ValueError) as e:
        logger.warning(f"Sampler-logprob probe against {client.base_url} failed (T={temperature}, top_p={top_p}): {e}")
        return None
    return values


def probe_sampler_logprob_semantics(
    client: WeightSyncClientBase, *, top_k: int, min_p: float, repetition_penalty: float
) -> tuple[bool | None, bool | None]:
    """``None`` where the probe could not tell. Prefill logits are the same under every sampler, so temperature 2
    halves the top-1/top-2 gap only if applied before the logprobs, and top-p 0.5 lifts the top logprob only if
    renormalized."""
    try:
        model_id = client.served_model_id()
    except Exception as e:
        logger.warning(f"Sampler-logprob probe: could not resolve the model served at {client.base_url}: {e}")
        return None, None
    filters = {"top_k": top_k, "min_p": min_p, "repetition_penalty": repetition_penalty}
    base = _probe_top_logprobs(client, model_id, 1.0, 1.0, filters)
    hot = _probe_top_logprobs(client, model_id, 2.0, 1.0, filters)
    nucleus = _probe_top_logprobs(client, model_id, 1.0, 0.5, filters)
    if base is None or len(base) < 2:
        return None, None
    temperature_applied = None
    base_gap = base[0] - base[1]
    if hot is not None and len(hot) >= 2 and base_gap > 0:
        # Processed logprobs halve the gap at temperature 2; raw ones leave it whole.
        temperature_applied = (hot[0] - hot[1]) / base_gap < 0.75
    # A renormalization lifts the top logprob by -log(nucleus mass).
    nucleus_renormalized = None if nucleus is None else nucleus[0] - base[0] > 1e-3
    return temperature_applied, nucleus_renormalized


def _probe_server(
    client_cls: type[WeightSyncClientBase],
    url: str,
    probe: Callable[[WeightSyncClientBase], Any],
    default: Any,
    failure: str,
) -> Any:
    """``probe`` over a fresh client for ``url``, else ``default`` with a warning."""
    client = None
    try:
        client = client_cls(base_url=url, retry_policy=_PROBE_RETRY)
        return probe(client)
    except Exception as e:
        logger.warning(f"Could not {failure} {url}: {e}")
        return default
    finally:
        if client is not None:
            client.close_http_session()


def _across_servers(per_server: Callable[[str], _T], urls: list[str]) -> list[_T]:
    """``per_server(url)`` for every url concurrently, in url order; the first url's failure raises once all finish."""
    if not urls:
        return []
    with ThreadPoolExecutor(max_workers=min(len(urls), _PREFLIGHT_PARALLELISM)) as pool:
        return list(pool.map(per_server, urls))


def _probe_servers(
    client_cls: type[WeightSyncClientBase],
    urls: list[str],
    probe: Callable[[WeightSyncClientBase], Any],
    default: Any,
    failure: str,
) -> list[Any]:
    """:func:`_probe_server` over every url concurrently, in url order."""
    return _across_servers(lambda url: _probe_server(client_cls, url, probe, default, failure), urls)


def _read_server_facts(
    client_cls: type[WeightSyncClientBase], url: str, needs: ServerRequirements
) -> tuple[ServerFlags | None, bool | None, bool | None]:
    """The launch flags of ``url`` and, where the run needs them, whether it accepts a tool-bearing request and
    reports the ids it sampled; ``None`` for a probe that did not answer."""
    flags = _probe_server(client_cls, url, client_cls.read_server_flags, None, "read the launch flags of")
    accepted = sampled = None
    if needs.sends_tools:
        accepted = _probe_server(
            client_cls, url, client_cls.accepts_tool_requests, None, "probe a tool-bearing request on"
        )
    if needs.sampled_token_ids:
        sampled = _probe_server(
            client_cls, url, client_cls.reports_sampled_token_ids, None, "probe the sampled-token ids of"
        )
    return flags, accepted, sampled


def _verify_context_window_on_rank0(
    client_cls: type[WeightSyncClientBase],
    urls: list[str],
    single_turn_tokens: int,
    full_trajectory_tokens: int | None,
) -> None:
    windows = _probe_servers(client_cls, urls, client_cls.served_max_model_len, None, "read max_model_len from")
    for url, mml in zip(urls, windows, strict=True):
        if mml is None:
            continue
        logger.info(f"Rollout server {url}: max_model_len={mml}")
        if single_turn_tokens > mml:
            raise ValueError(
                f"Rollout server {url} context window ({mml}) is smaller than one rollout turn "
                f"({single_turn_tokens} = max_prompt_length + per-turn generation). Lower "
                f"rollout_max_tokens/max_completion_length or serve a longer-context model."
            )
        if full_trajectory_tokens and full_trajectory_tokens > mml:
            logger.warning(
                f"Rollout server {url} context window ({mml}) < worst-case trajectory budget "
                f"({full_trajectory_tokens} = max_prompt_length + what an episode may generate: the smaller of "
                f"max_turns × rollout_max_tokens and rollout_max_episode_tokens). A long multi-turn rollout that "
                f"grows past {mml} tokens can OOM the training forward before the fail-on-overflow check — set "
                f"rollout_max_episode_tokens, or lower max_turns or rollout_max_tokens, so the worst case fits."
            )


def probe_generation_rate(client: WeightSyncClientBase) -> float | None:
    """Tokens per second of one idle sequence, the better of two timed completions (the first may warm a kernel);
    ``None`` when neither returned enough tokens to time."""
    body = {
        "model": client.served_model_id(),
        "prompt": "1 2 3 4 5 6 7 8 9 10 11 12",
        "max_tokens": _RATE_PROBE_TOKENS,
        "temperature": 0.0,
        "ignore_eos": True,
    }
    rates = []
    for _ in range(2):
        start = time.perf_counter()
        reply = client.request("POST", "/v1/completions", retry=False, timeout=_PROBE_TIMEOUT_S, json=body).json()
        elapsed = time.perf_counter() - start
        tokens = reply["usage"]["completion_tokens"]
        if tokens >= _RATE_PROBE_MIN_TOKENS:
            rates.append(tokens / elapsed)
    return max(rates, default=None)


def _verify_turn_deadline_on_rank0(
    client_cls: type[WeightSyncClientBase], urls: list[str], max_tokens: int, request_timeout: float
) -> None:
    for url in urls:
        rate = _probe_server(client_cls, url, probe_generation_rate, None, "time a generation on")
        if rate is None:
            continue
        needed = max_tokens / rate
        logger.info(
            f"Rollout server {url}: {rate:.0f} tokens/s on one idle sequence, so a {max_tokens}-token turn takes "
            f"~{needed:.0f}s of the {request_timeout:g}s request_timeout"
        )
        advice = (
            f"A turn that runs to rollout_max_tokens={max_tokens} needs ~{needed:.0f}s at the {rate:.0f} tokens/s "
            f"one idle sequence decodes on {url}, against request_timeout={request_timeout:g}s, and a sequence amid a "
            f"rollout round decodes slower. A timed-out episode trains as a masked row, so the batch keeps only the "
            f"episodes that finished fast. Set request_timeout to at least {math.ceil(2 * needed)} (and episode_timeout "
            f"to hold (1 + max_retries) of them), or lower rollout_max_tokens."
        )
        if needed > _DEADLINE_REFUSE_MULTIPLE * request_timeout:
            raise ValueError(f"Even an idle rollout server cannot finish a full turn inside its deadline. {advice}")
        if needed > _DEADLINE_WARN_SHARE * request_timeout:
            logger.warning(advice)


def _verify_probed_sampler(
    url: str, temperature: float, cuts: list[str], temperature_applied: bool | None, nucleus_renormalized: bool | None
) -> None:
    """``cuts`` are the sampler cuts a sequence-summing consumer reads, empty where none does."""
    if temperature != 1.0:
        if temperature_applied is None:
            logger.warning(
                f"Rollout server {url}: could not verify that its logprobs carry the sampling temperature "
                f"({temperature}); the IS ratio is biased if they are raw pre-temperature values (vLLM's default)."
            )
        elif not temperature_applied:
            raise ValueError(
                f"Rollout server {url} reports RAW (pre-temperature) logprobs while the sampling temperature is "
                f"{temperature}: the IS ratio would divide the trainer's temperature-{temperature} log-probs "
                f"by temperature-1 ones, biasing every token weight. Serve vLLM with "
                f"`--logprobs-mode processed_logprobs` (the compose recipe sets it), or leave "
                f"SGLANG_RETURN_ORIGINAL_LOGPROB unset on SGLang, or sample at temperature 1.0."
            )
    if not cuts:
        return
    if nucleus_renormalized is None:
        logger.warning(
            f"Rollout server {url}: could not verify whether its logprobs are renormalized over the "
            f"sampler's cut; with {', '.join(cuts)} a renormalized reference biases every "
            f"sequence-summed log-ratio (geometric band, OPSM, sequence-level IS)."
        )
    elif nucleus_renormalized:
        raise ValueError(
            f"Rollout server {url} reports logprobs renormalized over the sampler's cut while "
            f"{', '.join(cuts)} and the per-token log-ratios are summed over each sequence: every "
            f"uncertain position is lifted by the mass the cut kept, so the trajectory geometric band and "
            f"OPSM's per-trajectory mean log-ratio read the sum as drift and a sequence-level vLLM IS ratio "
            f"collapses toward 0 (sequence_mask only zeroes ratios ABOVE the cap, so the run stalls "
            f"silently). Sample with every cut off — top_p: 1.0, top_k: 0, min_p: 0.0 (rollout_top_p: 1.0, "
            f"rollout_top_k: -1, rollout_min_p: 0.0 on the environmental arm) — or {_PER_TOKEN_RATIO_REMEDY}."
        )


def _verify_sampler_logprob_reference_on_rank0(
    client_cls: type[WeightSyncClientBase],
    urls: list[str],
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    repetition_penalty: float,
    sequence_ratio_active: bool,
) -> None:
    if sequence_ratio_active and repetition_penalty != 1.0:
        raise ValueError(
            f"repetition_penalty={repetition_penalty} while the per-token log-ratios are summed over each "
            f"sequence: both engines' sampling logprobs carry the penalty and the trainer's recomputed ones do "
            f"not, so every penalized position reads as drift in the trajectory geometric band and OPSM, and a "
            f"sequence-level vLLM IS ratio collapses. Sample with repetition_penalty: 1.0 "
            f"(rollout_repetition_penalty: 1.0 on the environmental arm), or {_PER_TOKEN_RATIO_REMEDY}."
        )
    # A renormalizing cut biases only a consumer that sums the per-token log-ratios over a sequence.
    cuts = [
        f"{name}={value}"
        for name, value, on in (
            ("top_p", top_p, top_p < 1.0),
            ("top_k", top_k, top_k > 0),
            ("min_p", min_p, min_p > 0),
        )
        if on and sequence_ratio_active
    ]
    if temperature == 1.0 and not cuts:
        return
    # Each probe request moves one knob from the identity sampler, so every other filter goes out off.
    probe = partial(probe_sampler_logprob_semantics, **SAMPLER_FILTERS_OFF)
    readings = _probe_servers(client_cls, urls, probe, (None, None), "probe the sampler-logprob semantics of")
    for url, semantics in zip(urls, readings, strict=True):
        _verify_probed_sampler(url, temperature, cuts, *semantics)


def _probe_prompt_logprobs(client: WeightSyncClientBase, token_ids: list[int]) -> tuple[float, float, int]:
    """``(mean NLL, gap, length)``: the largest per-position gap to a prefix scored alone, over the positions the whole
    scores above :data:`_PROMPT_LOGPROB_CONFIDENT` (infinite for a non-finite log-prob)."""
    anchor, rest = token_ids[:1], token_ids[1:]
    score = partial(client.score_completion_logprobs, anchor, timeout=_PROBE_TIMEOUT_S)
    whole = score(rest)
    divergences = []
    # Each prefix prefills as a captured graph. One without a confident position adds no gap, so a whole that scores
    # none (itself garbage under a raised capture range) is left to the mean-NLL bound.
    for length in (8, 64, 128, 256, 448):
        pairs = list(zip(score(rest[: length - 1]), whole[: length - 1], strict=True))
        gaps = [abs(a - b) for a, b in pairs if b > _PROMPT_LOGPROB_CONFIDENT]
        finite = all(math.isfinite(a) and math.isfinite(b) for a, b in pairs)
        divergences.append((max(gaps, default=0.0) if finite else math.inf, length))
    gap, length = max(divergences)
    return -sum(whole) / len(whole), gap, length


def counting_probe_token_ids(tokenizer) -> list[int]:
    """A counting sequence cut past vLLM's CUDA-graph capture range (512 tokens), so a server prefills it eagerly."""
    counting = " ".join(str(n) for n in range(2048))
    return resolve_tokenizer(tokenizer)(counting)["input_ids"][:1100]


def _verify_engine_prompt_logprobs_on_rank0(
    client_cls: type[WeightSyncClientBase], urls: list[str], tokenizer
) -> None:
    token_ids = counting_probe_token_ids(tokenizer)
    probe = partial(_probe_prompt_logprobs, token_ids=token_ids)
    results = _probe_servers(client_cls, urls, probe, None, "probe the prompt log-probs of")
    for url, result in zip(urls, results, strict=True):
        if result is None:
            continue
        mean_nll, gap, length = result
        # Negated so a NaN mean, for which every comparison is False, is refused too.
        if not (mean_nll <= _PROMPT_LOGPROB_PROBE_MAX_MEAN_NLL and gap <= _PROMPT_LOGPROB_TOLERANCE_NATS):
            raise ValueError(
                f"Rollout server {url} returns prompt log-probs that do not follow from the tokens: a "
                f"{len(token_ids)}-token counting probe scored whole averages {mean_nll:.2f} nats per token (bound "
                f"{_PROMPT_LOGPROB_PROBE_MAX_MEAN_NLL}), and its first {length} tokens scored alone differ from the "
                f"same positions, where the whole scores above {_PROMPT_LOGPROB_CONFIDENT} log-prob, by up to "
                f"{gap:.2f} nats (tolerance {_PROMPT_LOGPROB_TOLERANCE_NATS}), so "
                f"isr_engine_reference would compare the sampling log-probs against garbage. The known cause is "
                f"vLLM's V1 model runner with speculative decoding (MTP): the drafter's CUDA graph overwrites the "
                f"hidden states the prompt log-probs are read from whenever the prefill runs as a graph "
                f"(vllm-project/vllm#53488). Serve vLLM from the image Dockerfile.vllm builds, which pins Model "
                f"Runner V2, turn speculative decoding off, or set isr_engine_reference: false."
            )
        logger.info(
            f"Rollout server {url}: prompt log-probs follow from the tokens (probe mean NLL {mean_nll:.3f}, "
            f"largest prefix gap {gap:.3f} nats at confident positions)"
        )


def _wait_then_check(
    check: Callable[..., None],
    client_cls: type[WeightSyncClientBase],
    urls: list[str],
    connection_timeout: float,
    *args,
) -> None:
    def wait_for(url: str) -> None:
        # Building the client polls its liveness route, raising ConnectionError past the deadline. A server held by a
        # dead trainer's pause still answers it, and each probe would wait out its timeout and pass unread.
        client = client_cls(base_url=url, connection_timeout=connection_timeout, retry_policy=_PROBE_RETRY)
        try:
            client.probe_generation()
        finally:
            client.close_http_session()

    _across_servers(wait_for, urls)
    check(client_cls, urls, *args)


def _on_rank0(check: Callable[..., None], backend: str, urls: list[str], connection_timeout: float, *args) -> None:
    """COLLECTIVE over the c10d store, as rank 0 may wait on loading servers for longer than the NCCL watchdog allows.
    The backend is resolved on every rank, since a rank-0-only raise there would strand the peers in the verdict."""
    client_cls = resolve_weight_sync_client(backend)
    step = partial(_wait_then_check, check, client_cls, urls, connection_timeout, *args)
    store_run_on_rank0_raising_everywhere(f"preflight/{check.__name__}", step, refusal_text, ValueError)


def verify_context_window(
    urls: list[str],
    single_turn_tokens: int,
    full_trajectory_tokens: int | None = None,
    *,
    backend: str,
    connection_timeout: float,
) -> None:
    """COLLECTIVE. Refuses a window that cannot hold ``single_turn_tokens``; a ``full_trajectory_tokens`` worst case
    past it only warns."""
    _on_rank0(
        _verify_context_window_on_rank0, backend, urls, connection_timeout, single_turn_tokens, full_trajectory_tokens
    )


def verify_turn_deadline(
    urls: list[str], max_tokens: int, request_timeout: float, *, backend: str, connection_timeout: float
) -> None:
    """COLLECTIVE. Times a short generation on each server and refuses a ``max_tokens`` turn that could not finish
    inside ``request_timeout`` even at that idle rate; warns where a loaded server would miss it."""
    _on_rank0(_verify_turn_deadline_on_rank0, backend, urls, connection_timeout, max_tokens, request_timeout)


def verify_adapter_servers(
    urls: list[str], needs: AdapterRequirements, *, backend: str, connection_timeout: float
) -> None:
    """COLLECTIVE. Refuses a server whose launch flags cannot hold the adapters the run loads."""
    _on_rank0(_verify_adapter_servers_on_rank0, backend, urls, connection_timeout, needs)


def verify_server_flags(
    urls: list[str], needs: ServerRequirements, *, backend: str, connection_timeout: float
) -> None:
    """COLLECTIVE. Refuses a server off the toolkit's image or launched with a flag the run's sync or log-probs need,
    each silently wrong otherwise."""
    _on_rank0(_verify_server_flags_on_rank0, backend, urls, connection_timeout, needs)


def verify_group_servers(urls: list[str], *, backend: str, connection_timeout: float) -> None:
    """COLLECTIVE. Refuses servers of one weight-sync group that serve different models or MoE backends."""
    _on_rank0(_verify_group_servers_on_rank0, backend, urls, connection_timeout)


def verify_server_chat_template(
    urls: list[str],
    trainer_prompt_ids: Callable[[Any], list[int]],
    messages: list[dict],
    efforts: list,
    rollout: RolloutConfig,
    *,
    connection_timeout: float,
) -> None:
    """COLLECTIVE. Refuses a server whose chat template ignores the level the run draws, which rows on the engine's
    prompt ids would never show, and warns where a render differs at all. ``messages`` carry no tools, as a client-side
    schema render drifts on correct setups, and only the refusal is certain: an engine derives toggles the trainer's
    lack. ``efforts`` are the two contracts compared, ``None`` for no level."""
    probe_config = replace(
        rollout, capture_token_ids=True, capture_routed_experts=False, max_tokens=1, max_thinking_tokens=None
    )
    _on_rank0(
        _verify_server_chat_template_on_rank0,
        rollout.backend,
        urls,
        connection_timeout,
        trainer_prompt_ids,
        messages,
        efforts,
        probe_config,
    )


def verify_sampler_logprob_reference(
    urls: list[str],
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    repetition_penalty: float,
    sequence_ratio_active: bool,
    backend: str,
    connection_timeout: float,
) -> None:
    """COLLECTIVE. Raw logprobs tilt every weight to π^T / π^1 while the ratio reads ≈ 1. A reference renormalized over
    a sampler cut lifts every uncertain position, which a sequence-summing consumer reads as drift; so does a
    ``repetition_penalty`` other than 1."""
    _on_rank0(
        _verify_sampler_logprob_reference_on_rank0,
        backend,
        urls,
        connection_timeout,
        temperature,
        top_p,
        top_k,
        min_p,
        repetition_penalty,
        sequence_ratio_active,
    )


def verify_engine_prompt_logprobs(urls: list[str], tokenizer, *, backend: str, connection_timeout: float) -> None:
    """COLLECTIVE. Scores a counting sequence that prefills eagerly, then prefixes at captured-graph lengths, and
    compares the shared positions the whole scores confidently."""
    _on_rank0(_verify_engine_prompt_logprobs_on_rank0, backend, urls, connection_timeout, tokenizer)
