"""COLLECTIVE preflights of each rollout server before the trainer is built: its context window, and whether its
sampler and prompt log-probs are the reference the IS ratio divides by. Each probes on rank 0 and raises its
refusal on every rank; a server that cannot answer only warns, since a probe never fails the run by itself."""

import logging
import math
from collections.abc import Callable, Mapping
from functools import partial
from typing import Any

import requests
from urllib3.util.retry import Retry

from halo.environments.rollout_config import SAMPLER_FILTERS_OFF
from halo.model_loading.tokenizer_setup import resolve_tokenizer
from halo.runtime.consensus import raise_rank0_failure, refusal_text
from halo.weight_sync.clients.base import WeightSyncClientBase
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

# Batch-shape bf16 noise stays under ~0.5 nats per token; the drafter overwrite moves every one by 8-22.
_PROMPT_LOGPROB_TOLERANCE_NATS = 2.0
# Any servable model all but predicts a counting sequence, while garbage prompt log-probs average 12-22
# nats: the bound catches a whole sequence that itself prefilled as a graph (a raised capture range).
_PROMPT_LOGPROB_PROBE_MAX_MEAN_NLL = 4.0

# How each arm stops summing the IS log-ratios over a sequence.
_PER_TOKEN_RATIO_REMEDY = (
    "take the ratio per token: on online GRPO a token_* vllm_importance_sampling_mode with a loss_type other than "
    "vespo; on the environmental arm, whose correction is per token already, drop isr_geo_band_min/max and "
    "isr_opsm_delta"
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
    """``(temperature_applied, nucleus_renormalized)``: whether the engine's per-token logprobs are the SAMPLING
    distribution's (after temperature, after a top-p renormalization) or the raw values, ``None`` where the probe
    could not tell, read off three 1-token completions of one prompt.

    The prefill logits are the same under every sampler, so temperature 2 halves the top-1/top-2 gap only if
    applied before the logprobs, and top-p 0.5 lifts the top logprob only if renormalized into them. The
    caller passes the other filters' off values, each request moving one knob from the identity sampler.
    """
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
    """``probe`` over a fresh client for ``url``, else ``default`` with the warning ``Could not <failure> <url>``."""
    client = None
    try:
        client = client_cls(base_url=url, retry_policy=_PROBE_RETRY)
        return probe(client)
    except Exception as e:
        logger.warning(f"Could not {failure} {url}: {e}")
        return default
    finally:
        if client is not None:
            client.close_session()


def _verify_context_window_on_rank0(
    client_cls: type[WeightSyncClientBase],
    urls: list[str],
    single_turn_tokens: int,
    full_trajectory_tokens: int | None,
) -> None:
    """Rank 0's body of :func:`verify_context_window`."""
    for url in urls:
        mml = _probe_server(client_cls, url, client_cls.served_max_model_len, None, "read max_model_len from")
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


def _verify_probed_sampler(
    url: str, temperature: float, cuts: list[str], temperature_applied: bool | None, nucleus_renormalized: bool | None
) -> None:
    """Refuse the server at ``url`` whose probed logprobs are not the IS reference, and warn where the probe could
    not tell. ``cuts`` are the sampler cuts a sequence-summing consumer reads, empty where none does."""
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
    """Rank 0's body of :func:`verify_sampler_logprob_reference`."""
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
    for url in urls:
        semantics = _probe_server(client_cls, url, probe, (None, None), "probe the sampler-logprob semantics of")
        _verify_probed_sampler(url, temperature, cuts, *semantics)


def _probe_prompt_logprobs(client: WeightSyncClientBase, token_ids: list[int]) -> tuple[float, float, int]:
    """``(mean NLL, gap, length)``: the whole sequence's mean NLL, and the largest per-position gap to a prefix
    scored alone (infinite for a non-finite log-prob), with that prefix's length."""
    anchor, rest = token_ids[:1], token_ids[1:]
    score = partial(client.score_completion_logprobs, anchor, timeout=_PROBE_TIMEOUT_S)
    whole = score(rest)
    divergences = []
    # Each prefix prefills as a captured graph.
    for length in (8, 64, 128, 256, 448):
        gaps = [abs(a - b) for a, b in zip(score(rest[: length - 1]), whole[: length - 1], strict=True)]
        divergences.append((max(gaps) if all(map(math.isfinite, gaps)) else math.inf, length))
    gap, length = max(divergences)
    return -sum(whole) / len(whole), gap, length


def _verify_engine_prompt_logprobs_on_rank0(
    client_cls: type[WeightSyncClientBase], urls: list[str], tokenizer
) -> None:
    """Rank 0's body of :func:`verify_engine_prompt_logprobs`."""
    # A counting sequence cut past vLLM's CUDA-graph capture range (512 tokens), so the whole prefills eagerly.
    counting = " ".join(str(n) for n in range(2048))
    token_ids = resolve_tokenizer(tokenizer)(counting)["input_ids"][:1100]
    probe = partial(_probe_prompt_logprobs, token_ids=token_ids)
    for url in urls:
        result = _probe_server(client_cls, url, probe, None, "probe the prompt log-probs of")
        if result is None:
            continue
        mean_nll, gap, length = result
        # Negated so a NaN mean, for which every comparison is False, is refused too.
        if not (mean_nll <= _PROMPT_LOGPROB_PROBE_MAX_MEAN_NLL and gap <= _PROMPT_LOGPROB_TOLERANCE_NATS):
            raise ValueError(
                f"Rollout server {url} returns prompt log-probs that do not follow from the tokens: a "
                f"{len(token_ids)}-token counting probe scored whole averages {mean_nll:.2f} nats per token (bound "
                f"{_PROMPT_LOGPROB_PROBE_MAX_MEAN_NLL}), and its first {length} tokens scored alone differ from the "
                f"same positions by up to {gap:.2f} nats (tolerance {_PROMPT_LOGPROB_TOLERANCE_NATS}), so "
                f"isr_engine_reference would compare the sampling log-probs against garbage. The known cause is "
                f"vLLM's V1 model runner with speculative decoding (MTP): the drafter's CUDA graph overwrites the "
                f"hidden states the prompt log-probs are read from whenever the prefill runs as a graph "
                f"(vllm-project/vllm#53488). Serve vLLM from the image Dockerfile.vllm builds, which pins Model "
                f"Runner V2, turn speculative decoding off, or set isr_engine_reference: false."
            )
        logger.info(
            f"Rollout server {url}: prompt log-probs follow from the tokens (probe mean NLL {mean_nll:.3f}, "
            f"largest prefix gap {gap:.3f} nats)"
        )


def _on_rank0(check: Callable[..., None], backend: str, *args) -> None:
    """COLLECTIVE: run ``check(client_cls, *args)`` on rank 0 and raise its refusal on every rank. The backend is
    resolved on every rank, since a rank-0-only raise there would strand the peers in the verdict broadcast."""
    client_cls = resolve_weight_sync_client(backend)
    raise_rank0_failure(partial(check, client_cls, *args), refusal_text, ValueError)


def verify_context_window(
    urls: list[str], single_turn_tokens: int, full_trajectory_tokens: int | None = None, *, backend: str
) -> None:
    """COLLECTIVE: refuse a rollout server whose context window cannot hold ``single_turn_tokens``; a
    ``full_trajectory_tokens`` worst case past it only warns."""
    _on_rank0(_verify_context_window_on_rank0, backend, urls, single_turn_tokens, full_trajectory_tokens)


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
) -> None:
    """COLLECTIVE: refuse a rollout server whose per-token logprobs are not the reference the IS ratio divides by.

    The trainer scores at the sampling temperature, so the engine's logprobs must carry it too (raw values
    tilt every weight to π^T / π^1 while the ratio still reads ≈ 1). A reference renormalized over a sampler
    cut (top-p < 1, top-k > 0, min-p > 0) lifts every uncertain position, which a consumer summing the
    per-token log-ratios over a sequence (``sequence_ratio_active``) reads as drift; so does a
    ``repetition_penalty`` other than 1, refused there without a probe.
    """
    _on_rank0(
        _verify_sampler_logprob_reference_on_rank0,
        backend,
        urls,
        temperature,
        top_p,
        top_k,
        min_p,
        repetition_penalty,
        sequence_ratio_active,
    )


def verify_engine_prompt_logprobs(urls: list[str], tokenizer, *, backend: str) -> None:
    """COLLECTIVE: refuse a rollout server whose prompt log-probs, which ``isr_engine_reference`` reads, do not
    follow from the tokens alone. The probe scores a counting sequence that prefills eagerly, then prefixes of
    it at captured-graph lengths, and compares the shared positions.
    """
    _on_rank0(_verify_engine_prompt_logprobs_on_rank0, backend, urls, tokenizer)
