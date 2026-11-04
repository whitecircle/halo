#!/usr/bin/env python
"""A checkpoint's head as a server drafts with it: the acceptance over first-turn prompts at temperature 1, by the
engine's own counters. The row launches no server: it reads the one at ``VLLM_SERVER_URL`` or ``SGLANG_SERVER_URL``, which
must run the checkpoint and speculation the row is for.

    VLLM_MODEL=$EXPORT VLLM_SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":1}' VLLM_CUDA_DEVICES=0,1,2,3 \\
        VLLM_TP=4 docker compose -f docker-compose.vllm.yml up -d vllm-server
    torchrun --nproc_per_node=1 tests/gpu/trainers/sft/test_sft_mtp_head_serving.py --engine vllm --at_least 0.7

``--control_url`` serves a second checkpoint with its own head, the same prompts in the same order, and holds this one
to within ``--within_control`` of it. ``--plain_url`` serves this checkpoint without speculation: a draft changes a
forward's batch shape and so its bf16 rounding, so greedy output may part from the plain server's only at a near-tie
(vLLM alone). A server that refuses to start on a checkpoint with no head never reaches a row: its exit is the
result, as ``agent-docs/optimization/mtp-heads.md`` says.
"""

import argparse
import math
import sys
from concurrent.futures import ThreadPoolExecutor

import requests
from transformers import AutoTokenizer

from halo.env import env_str
from tests.common.harness import gpu_test_main
from tests.common.mtp_real_rows import prometheus_total, ultrachat_conversations
from tests.common.rank_log import log

_MAX_TOKENS = 256
_GREEDY_TOKENS = 128
_GREEDY_PROMPTS = 32
_GREEDY_TOP_LOGPROBS = 5
_GREEDY_SPLIT_BAR = 1.0
_WORKERS = 16
_REQUEST_TIMEOUT_S = 1800
_DEFAULT_URLS = {"vllm": "http://localhost:8000", "sglang": "http://localhost:30000"}
_URL_VARS = {"vllm": "VLLM_SERVER_URL", "sglang": "SGLANG_SERVER_URL"}


def _served_model(url: str) -> str:
    return requests.get(f"{url}/v1/models", timeout=60).json()["data"][0]["id"]


def _first_turn_prompts(tokenizer, count: int) -> list[str]:
    conversations = ultrachat_conversations("test_gen").select(range(count))
    return [
        tokenizer.apply_chat_template(row["messages"][:1], tokenize=False, add_generation_prompt=True)
        for row in conversations
    ]


def _vllm_completion(url: str, model: str, prompt: str, *, temperature: float, max_tokens: int) -> str:
    response = requests.post(
        f"{url}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 1.0,
            "top_k": 0,
        },
        timeout=_REQUEST_TIMEOUT_S,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["text"]


def _sglang_generate(url: str, prompt: str, *, temperature: float, max_tokens: int) -> dict:
    response = requests.post(
        f"{url}/generate",
        json={
            "text": prompt,
            "sampling_params": {"temperature": temperature, "max_new_tokens": max_tokens, "top_p": 1.0, "top_k": -1},
        },
        timeout=_REQUEST_TIMEOUT_S,
    )
    response.raise_for_status()
    return response.json()


def _draft_counts(engine: str, url: str, prompts: list[str]) -> tuple[float, float]:
    """Over this call's requests: drafts the target accepted and drafts proposed, at the first draft position."""
    with ThreadPoolExecutor(_WORKERS) as pool:
        if engine == "vllm":
            model = _served_model(url)
            before = requests.get(f"{url}/metrics", timeout=60).text
            list(
                pool.map(
                    lambda prompt: _vllm_completion(url, model, prompt, temperature=1.0, max_tokens=_MAX_TOKENS),
                    prompts,
                )
            )
            after = requests.get(f"{url}/metrics", timeout=60).text
            accepted = "vllm:spec_decode_num_accepted_tokens_per_pos"
            drafts = "vllm:spec_decode_num_drafts"
            proposed = prometheus_total(after, drafts) - prometheus_total(before, drafts)
            won = prometheus_total(after, accepted, position="0") - prometheus_total(before, accepted, position="0")
        else:
            replies = list(
                pool.map(
                    lambda prompt: _sglang_generate(url, prompt, temperature=1.0, max_tokens=_MAX_TOKENS)["meta_info"],
                    prompts,
                )
            )
            proposed = sum(reply.get("spec_num_proposed_drafts", 0) for reply in replies)
            won = sum(reply.get("spec_num_correct_drafts", 0) for reply in replies)
    return won, proposed


def _greedy_logprobs(url: str, prompts: list[str]) -> list[dict]:
    model = _served_model(url)
    return [
        requests.post(
            f"{url}/v1/completions",
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": _GREEDY_TOKENS,
                "temperature": 0.0,
                "logprobs": _GREEDY_TOP_LOGPROBS,
            },
            timeout=_REQUEST_TIMEOUT_S,
        ).json()["choices"][0]["logprobs"]
        for prompt in prompts
    ]


def _first_split(drafted: dict, plain: dict) -> tuple[int, float] | None:
    """Where two greedy outputs first part and the wider of the servers' margins for their own pick over the other's;
    ``None`` when they never part. A pick out of the other's top logprobs, or one server stopping early, is no near-tie."""
    for index, (ours, theirs) in enumerate(zip(drafted["tokens"], plain["tokens"], strict=False)):
        if ours != theirs:
            top_ours, top_theirs = drafted["top_logprobs"][index], plain["top_logprobs"][index]
            margins = (
                top_ours[ours] - top_ours.get(theirs, -math.inf),
                top_theirs[theirs] - top_theirs.get(ours, -math.inf),
            )
            return index, max(margins)
    if len(drafted["tokens"]) != len(plain["tokens"]):
        return min(len(drafted["tokens"]), len(plain["tokens"])), math.inf
    return None


@gpu_test_main(exact_world_size=1)
def run(ctx) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=sorted(_DEFAULT_URLS), required=True)
    parser.add_argument("--at_least", type=float, default=None, help="The least acceptance this server must read.")
    parser.add_argument(
        "--at_most", type=float, default=None, help="The most it may read, for a head that cannot work."
    )
    parser.add_argument("--prompts", type=int, default=64)
    parser.add_argument("--control_url", default=None)
    parser.add_argument("--within_control", type=float, default=0.03)
    parser.add_argument("--plain_url", default=None)
    cli = parser.parse_args()
    if cli.plain_url and cli.engine != "vllm":
        parser.error("--plain_url reads vLLM's completion logprobs.")

    url = env_str(_URL_VARS[cli.engine]) or _DEFAULT_URLS[cli.engine]
    tokenizer = AutoTokenizer.from_pretrained(_served_model(url))
    prompts = _first_turn_prompts(tokenizer, cli.prompts)
    won, proposed = _draft_counts(cli.engine, url, prompts)
    if not proposed:
        log(f"SKIP: the {cli.engine} server at {url} proposed no draft: it runs no speculative config")
        sys.exit(0)
    acceptance = won / proposed
    log(f"  {cli.engine} at {url}: acceptance {acceptance:.4f} over {proposed:.0f} drafts of {len(prompts)} prompts")
    metrics = {"acceptance": acceptance, "drafts": proposed}
    checks = {"the_server_drafted_on_every_prompt": proposed >= len(prompts)}
    if cli.at_least is not None:
        checks["acceptance_reached_its_bar"] = acceptance >= cli.at_least
    if cli.at_most is not None:
        checks["acceptance_stayed_under_its_bar"] = acceptance <= cli.at_most
    if cli.control_url:
        control_won, control_proposed = _draft_counts(cli.engine, cli.control_url, prompts)
        control = control_won / control_proposed
        metrics["control_acceptance"] = control
        log(f"  control at {cli.control_url}: acceptance {control:.4f}")
        checks["acceptance_within_its_control"] = acceptance >= control - cli.within_control
    if cli.plain_url:
        drafted = _greedy_logprobs(url, prompts[:_GREEDY_PROMPTS])
        plain = _greedy_logprobs(cli.plain_url, prompts[:_GREEDY_PROMPTS])
        splits = [_first_split(ours, theirs) for ours, theirs in zip(drafted, plain, strict=True)]
        for index, split in enumerate(splits):
            log(
                f"  prompt {index}: "
                + ("same greedy output" if split is None else f"parts at {split[0]}, margin {split[1]:.3f}")
            )
        parted = [split[1] for split in splits if split is not None]
        metrics["greedy_prompts_parted"] = len(parted)
        checks["greedy_output_parts_only_at_a_near_tie"] = all(margin <= _GREEDY_SPLIT_BAR for margin in parted)
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    raise SystemExit(run())
