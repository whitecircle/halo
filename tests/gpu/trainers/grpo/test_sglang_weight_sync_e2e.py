#!/usr/bin/env python
"""SGLang rollout and NCCL weight sync against a live server (``make test-gpu-sglang`` sets it up).

SGLang's ``logprobs`` drop token ids, which survive only in the per-choice ``meta_info``. A sync can return ``200 OK``
and not land the weights, so greedy output must change. The trainer needs its own GPU, and both ends must agree on the
NCCL net or both block in the first broadcast.
"""

import json
import urllib.request

import torch
from transformers import AutoModelForCausalLM

from halo.environments.engine_wire import capture_generation_tokens
from halo.environments.ray_actors import EnvironmentActor
from halo.environments.rollout_config import RolloutConfig
from halo.weight_sync.clients.sglang import SGLangWeightSyncClient
from tests.common.harness import gpu_test_main, record_check
from tests.common.models import QWEN3_0_6B
from tests.common.rank_log import log
from tests.common.weight_sync import SGLANG_SERVER_URL, push, weight_transfer_port

PROBE_PROMPT = "Count: 1 2 3"


def _post(path: str, payload: dict, timeout: float = 180.0) -> dict:
    req = urllib.request.Request(
        f"{SGLANG_SERVER_URL}{path}", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def _greedy_probe() -> list[tuple[int, float]]:
    """Greedy sampled (token_id, logprob) pairs, the served policy's fingerprint."""
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
    # The engine's own accounting, so a truncated capture cannot pass.
    assert len(ids) == choice["meta_info"]["completion_tokens"], (
        f"captured {len(ids)} ids but the engine generated {choice['meta_info']['completion_tokens']}"
    )


def test_weight_sync_changes_the_served_policy():
    baseline, repeat = _greedy_probe(), _greedy_probe()
    # Greedy decoding is deterministic, so a later difference is the sync, not sampling.
    assert baseline == repeat, "greedy probe is not deterministic; a post-sync difference would prove nothing"

    model = AutoModelForCausalLM.from_pretrained(QWEN3_0_6B, dtype=torch.bfloat16).to("cuda:0")
    params = dict(model.named_parameters())
    targets = [n for n in params if "layers.0.self_attn" in n and n.endswith(".weight")]
    assert targets, "no layer-0 attention weights found — the perturbation below would be a no-op"

    group_port = weight_transfer_port("HALO_TEST_SGLANG_GROUP_PORT")
    client = SGLangWeightSyncClient(base_url=SGLANG_SERVER_URL, group_port=group_port, connection_timeout=120)
    client.init_communicator(device=torch.device("cuda", 0))
    try:
        with torch.no_grad():
            for name in targets:
                params[name].mul_(1.02)
        push(client, [(name, params[name].detach()) for name in targets])

        after = _greedy_probe()
        assert after != baseline, "served policy unchanged after weight sync — the broadcast did not land"
    finally:
        client.close_communicator()

    # A failed update leaves SGLang partially written, and its docs say to discard such a server.
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
