#!/usr/bin/env python
"""End-to-end SGLang rollout + NCCL weight sync against a live server.

Covers the two halves of the SGLang backend that only a running engine can prove:

1. **Sampled-token capture.** SGLang's OpenAI ``logprobs`` reports each token as text and drops the
   id; the ids survive only in the ``meta_info`` it echoes per choice under ``return_meta_info``.
   The test drives the production payload builder and capture reader, so a regression in either
   spelling shows up as missing ids rather than a silent fall back to re-tokenization.
2. **Weight sync.** The trainer joins the engine's c10d group as rank 0 and broadcasts. The check is
   behavioural — greedy output must change after a sync — because a sync can return ``200 OK`` on
   every call and still not land the weights.

Prerequisites (``make test-gpu-sglang`` sets both up):
    SGLANG_CUDA_DEVICES=7 SGLANG_MODEL=Qwen/Qwen3-0.6B docker compose -f docker-compose.sglang.yml up -d
    # the trainer must own a GPU the server does not, and both need the socket NCCL transport

Usage (the trainer must set the SAME five NCCL settings the compose server does — a trainer on P2P/SHM
against a socket-only server blocks both ranks in the first broadcast):
    CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_NET=Socket NCCL_IB_DISABLE=1 \
        NCCL_NET_PLUGIN=none torchrun --nproc_per_node=1 \
        tests/gpu/trainers/grpo/test_sglang_weight_sync_e2e.py
"""

import json
import urllib.request

import torch

from src.env import env_int, env_str
from tests.common.harness import gpu_test_main, record_check
from tests.common.models import QWEN3_0_6B
from tests.common.utils import log

# ``or`` (not an env_str default): an exported-but-empty SGLANG_SERVER_URL passes the conftest gate,
# which reads it the same way, so the client must fall back to the same URL rather than to "".
SGLANG_SERVER_URL = env_str("SGLANG_SERVER_URL") or "http://localhost:30000"
GROUP_PORT = env_int("HALO_TEST_SGLANG_GROUP_PORT", 51216)
PROBE_PROMPT = "Count: 1 2 3"


def _post(path: str, payload: dict, timeout: float = 180.0) -> dict:
    req = urllib.request.Request(
        f"{SGLANG_SERVER_URL}{path}", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def _greedy_probe() -> list[tuple[int, float]]:
    """Greedy sampled (token_id, logprob) pairs — the served policy's fingerprint."""
    data = _post(
        "/v1/chat/completions",
        {
            "model": QWEN3_0_6B,
            "messages": [{"role": "user", "content": PROBE_PROMPT}],
            "max_tokens": 8,
            "temperature": 0.0,
            "logprobs": True,
            "return_meta_info": True,
        },
    )
    triples = data["choices"][0]["meta_info"]["output_token_logprobs"]
    return [(int(t[1]), round(float(t[0]), 4)) for t in triples]


def test_capture_via_production_path():
    """The real payload builder + capture reader must recover ids, logprobs and prompt ids."""
    from src.environments.engine_wire import capture_generation_tokens
    from src.environments.ray_actors import EnvironmentActor, RolloutConfig

    cls = EnvironmentActor.__ray_metadata__.modified_class
    actor = cls.__new__(cls)
    actor.__init__(actor_id=0, env_type="native_math", env_config={"max_turns": 3})
    config = RolloutConfig(
        backend="sglang", capture_token_ids=True, max_tokens=16, temperature=0.0, model_name=QWEN3_0_6B
    )
    payload = actor._build_payload([{"role": "user", "content": "What is 2+2?"}], config)

    assert payload["return_meta_info"] is True, "SGLang needs return_meta_info or the ids are dropped"
    assert payload["return_prompt_token_ids"] is True
    assert "return_token_ids" not in payload, "vLLM's spelling would be silently ignored by SGLang"

    data = _post("/v1/chat/completions", payload)
    choice = data["choices"][0]
    ids, logprobs, prompt_ids = capture_generation_tokens(choice, data, "sglang")

    assert ids, "no sampled token ids captured — train_on_sampled_tokens would silently re-tokenize"
    assert logprobs and len(logprobs) == len(ids)
    assert prompt_ids, "no engine-rendered prompt ids captured"
    # The id count is the engine's own accounting, so a truncated capture cannot pass unnoticed.
    assert len(ids) == choice["meta_info"]["completion_tokens"], (
        f"captured {len(ids)} ids but the engine generated {choice['meta_info']['completion_tokens']}"
    )


def test_weight_sync_changes_the_served_policy():
    """A sync must actually land: greedy output before != after a perturbation."""
    from transformers import AutoModelForCausalLM

    from src.distributed.nccl.clients.sglang import SGLangWeightSyncClient

    baseline, repeat = _greedy_probe(), _greedy_probe()
    # Anti-vacuity: greedy decoding is deterministic, so a later difference is the sync, not sampling.
    assert baseline == repeat, "greedy probe is not deterministic; a post-sync difference would prove nothing"

    model = AutoModelForCausalLM.from_pretrained(QWEN3_0_6B, dtype=torch.bfloat16).to("cuda:0")
    params = dict(model.named_parameters())
    targets = [n for n in params if "layers.0.self_attn" in n and n.endswith(".weight")]
    assert targets, "no layer-0 attention weights found — the perturbation below would be a no-op"

    client = SGLangWeightSyncClient(base_url=SGLANG_SERVER_URL, group_port=GROUP_PORT, connection_timeout=120)
    client.init_communicator(device=torch.device("cuda", 0))
    try:
        with torch.no_grad():
            for name in targets:
                params[name].mul_(1.02)
        for name in targets:
            client.update_named_param(name, params[name].detach())
        client.reset_prefix_cache()

        after = _greedy_probe()
        assert after != baseline, "served policy unchanged after weight sync — the broadcast did not land"
    finally:
        client.close_communicator()

    # The engine must still be usable: a failed update leaves SGLang partially written, and its own
    # docs say to discard such a server rather than keep serving from it.
    assert _greedy_probe(), "server unusable after weight sync"


def run(ctx) -> dict:
    log(f"SGLang weight-sync e2e against {SGLANG_SERVER_URL}")
    checks: dict[str, bool] = {}
    record_check(checks, "capture_via_production_path", test_capture_via_production_path)
    record_check(checks, "weight_sync_changes_the_served_policy", test_weight_sync_changes_the_served_policy)
    return {"checks": checks}


main = gpu_test_main(exact_world_size=1, prefix="sglang_weight_sync_e2e", partial_state=False)(run)

if __name__ == "__main__":
    main()
