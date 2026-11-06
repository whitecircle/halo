#!/usr/bin/env python
"""``rollout_adapter_sync`` against a live SGLang server launched with the adapter flags (``SGLANG_LORA_MAX_RANK=16``,
the default module list, Qwen3-0.6B): the base goes out once, a few training steps later the adapter alone is loaded,
and the engine's log-probs for the served adapter match the trainer's and a merged push's, while a request that names no
adapter does not. The trainer needs its own GPU.
"""

from types import SimpleNamespace

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from halo.trainers.grpo.rollout import adapter_sync
from halo.trainers.grpo.rollout.adapter_sync import (
    ADAPTER_NAME_PREFIX,
    AdapterSync,
    build_adapter_requirements,
    reject_unadaptable_model,
    resolve_adapter_serving,
)
from halo.trainers.grpo.rollout.engine_preflight import verify_adapter_servers
from halo.trainers.grpo.rollout.weight_push import sync_trainer_weights
from halo.weight_sync.adapter_servers import AdapterServers
from halo.weight_sync.adapter_versions import AdapterVersions
from halo.weight_sync.clients.sglang import SGLangWeightSyncClient
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.rank_log import log
from tests.common.weight_sync import SGLANG_SERVER_URL, weight_transfer_port
from tests.common.weight_sync_group import form_group

# The mean |engine - trainer| log-prob gap of a dense bf16 model on text it was not trained on; the first run records
# the measured value, and a gap past this is a wrong adapter, not noise.
ENGINE_TRAINER_MEAN_ABS_NATS = 0.1
# The adapter must move the policy by this many times the floor, or "differs from the base" proves nothing.
MIN_SHIFT_OVER_FLOOR = 4
TEXT = (
    "The quick brown fox jumps over the lazy dog. Counting upward from one gives 1 2 3 4 5 6 7 8 9 10, and "
    "counting back down gives 10 9 8 7 6 5 4 3 2 1. A model that has learned this text assigns it high probability."
)
PROMPT_TOKENS = 12
TRAIN_STEPS = 25
LEARNING_RATE = 3e-3


def _train(model, input_ids: torch.Tensor, steps: int) -> None:
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)
    for _ in range(steps):
        loss = model(input_ids=input_ids, labels=input_ids).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


@torch.no_grad()
def _trainer_logprobs(model, ids: list[int]) -> torch.Tensor:
    tokens = torch.tensor([ids], device=model.device)
    logprobs = torch.log_softmax(model(input_ids=tokens).logits[0].float(), dim=-1)
    return logprobs[PROMPT_TOKENS - 1 : -1].gather(1, tokens[0, PROMPT_TOKENS:, None])[:, 0].cpu()


def _engine_logprobs(client: SGLangWeightSyncClient, ids: list[int], adapter: str | None) -> torch.Tensor:
    client.adapter_source = lambda: adapter
    return torch.tensor(client.score_completion_logprobs(ids[:PROMPT_TOKENS], ids[PROMPT_TOKENS:]))


def _gap(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().mean())


def run(ctx) -> dict:
    ids = AutoTokenizer.from_pretrained(QWEN3_0_6B)(TEXT)["input_ids"]
    input_ids = torch.tensor([ids], device=ctx.device)
    model = AutoModelForCausalLM.from_pretrained(QWEN3_0_6B, dtype=torch.bfloat16).to(ctx.device)
    peft_model = get_peft_model(
        model, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    ).to(torch.bfloat16)
    parallelism = SimpleNamespace(is_tp_mode=False, is_pp_mode=False, is_cp_mode=False)
    reject_unadaptable_model(peft_model, parallelism, SGLangWeightSyncClient)
    serving = resolve_adapter_serving(peft_model)
    trainer = SimpleNamespace(
        model=peft_model,
        parallelism_config=SimpleNamespace(is_tp_mode=False, has_ep_group=False, is_expert_tp_mode=False),
        accelerator=SimpleNamespace(is_main_process=True),
        state=SimpleNamespace(global_step=0),
        _get_tp_rank=lambda: 0,
    )
    checks: dict[str, bool] = {}

    verify_adapter_servers(
        [SGLANG_SERVER_URL],
        build_adapter_requirements(serving, SimpleNamespace(prefetch_active=lambda: True, prefetch_depth=1)),
        backend="sglang",
        connection_timeout=120,
    )
    pool = form_group(
        SGLangWeightSyncClient,
        [SGLANG_SERVER_URL],
        weight_transfer_port("HALO_TEST_SGLANG_GROUP_PORT"),
        ctx.device,
        "per_server",
    )
    servers = AdapterServers([SGLANG_SERVER_URL], client_cls=SGLangWeightSyncClient, connection_timeout=120)
    client = SGLangWeightSyncClient(base_url=SGLANG_SERVER_URL, connection_timeout=120)
    versions = AdapterVersions()
    sync = AdapterSync(serving, versions)
    sync.servers = servers
    base_pushes = []
    real_push = adapter_sync.sync_trainer_weights
    adapter_sync.sync_trainer_weights = lambda *a, **kw: (base_pushes.append(kw), real_push(*a, **kw))[1]
    try:
        _train(peft_model, input_ids, TRAIN_STEPS)
        sync.sync(trainer, pool)
        first = versions.current
        with peft_model.disable_adapter():
            trainer_base = _trainer_logprobs(peft_model, ids)
        trainer_adapted = _trainer_logprobs(peft_model, ids)
        engine_adapted = _engine_logprobs(client, ids, first)
        engine_base = _engine_logprobs(client, ids, None)
        log(
            f"gaps: adapted engine vs trainer {_gap(engine_adapted, trainer_adapted):.4f}, base engine vs trainer "
            f"{_gap(engine_base, trainer_base):.4f}, adapter shift {_gap(trainer_adapted, trainer_base):.4f}"
        )
        checks["premise_the_adapter_moves_the_policy"] = (
            _gap(trainer_adapted, trainer_base) > MIN_SHIFT_OVER_FLOOR * ENGINE_TRAINER_MEAN_ABS_NATS
        )
        checks["the_base_went_out_once_without_the_adapter"] = base_pushes == [{"fold_adapters": False}]
        checks["engine_adapter_matches_the_trainer"] = (
            _gap(engine_adapted, trainer_adapted) <= ENGINE_TRAINER_MEAN_ABS_NATS
        )
        checks["engine_base_is_the_frozen_base"] = _gap(engine_base, trainer_base) <= ENGINE_TRAINER_MEAN_ABS_NATS
        # Negative control: a request naming no adapter reads the base, far from the trainer's adapted policy.
        checks["a_request_naming_no_adapter_misses_the_policy"] = (
            _gap(engine_base, trainer_adapted) > MIN_SHIFT_OVER_FLOOR * ENGINE_TRAINER_MEAN_ABS_NATS
        )

        _train(peft_model, input_ids, TRAIN_STEPS)
        sync.sync(trainer, pool)
        trainer_adapted = _trainer_logprobs(peft_model, ids)
        engine_adapted = _engine_logprobs(client, ids, versions.current)
        checks["a_later_sync_sent_the_adapter_alone"] = len(base_pushes) == 1 and versions.current != first
        checks["the_second_adapter_matches_the_trainer"] = (
            _gap(engine_adapted, trainer_adapted) <= ENGINE_TRAINER_MEAN_ABS_NATS
        )
        checks["the_old_version_was_retired"] = client.loaded_adapters() == [versions.current]

        with versions.lease() as running:
            sync.sync(trainer, pool)
            checks["a_version_in_flight_stays_loaded"] = sorted(client.loaded_adapters()) == sorted(
                [running, versions.current]
            )
        sync.sync(trainer, pool)
        checks["a_released_version_is_retired"] = client.loaded_adapters() == [versions.current]

        trainer_adapted = _trainer_logprobs(peft_model, ids)
        engine_adapted = _engine_logprobs(client, ids, versions.current)
        real_push(trainer, pool)
        engine_merged = _engine_logprobs(client, ids, None)
        log(
            f"merged: engine merged vs adapted {_gap(engine_merged, engine_adapted):.4f}, merged vs trainer "
            f"{_gap(engine_merged, trainer_adapted):.4f}"
        )
        checks["adapter_served_matches_a_merged_push"] = (
            _gap(engine_merged, engine_adapted) <= ENGINE_TRAINER_MEAN_ABS_NATS
        )
    finally:
        adapter_sync.sync_trainer_weights = real_push
        servers.unload([name for names in servers.loaded() for name in names if name.startswith(ADAPTER_NAME_PREFIX)])
        sync_trainer_weights(trainer, pool, fold_adapters=False)
        servers.close()
        client.close_http_session()
        pool.close_communicator()
    return {"checks": checks}


main = gpu_test_main(exact_world_size=1, prefix="adapter_sync_e2e", partial_state=False)(run)

if __name__ == "__main__":
    main()
