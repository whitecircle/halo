---
name: rl-setup
description: >-
  Wire up online GRPO (RLVR) or environmental (multi-turn tool-use) GRPO for
  Halo: bring up the separate rollout container — vLLM 0.26.0 (cu13), or
  SGLang 0.5.17 for environmental GRPO (rollout_backend: sglang, gpt-oss + ep1
  only) — point the trainer at it via AsyncTrainingConfig rollout URLs, establish
  NCCL weight sync through the vendored client in src/distributed/nccl/
  (parallelism-aware gather — EP/TP/ETP and multi-rank FSDP2 all participate),
  select the RL environment by registry name, and set the key GRPO
  hyperparameters. USER-INVOKED ONLY — invoke when the user explicitly asks to set
  up / configure / wire online or environmental GRPO, the vLLM or SGLang rollout
  server, Ray rollout actors, or NCCL weight sync.
disable-model-invocation: true
allowed-tools:
  - Read
  - Grep
  - Glob
  - Bash
---

# rl-setup

Wire up online GRPO (RLVR) or environmental (multi-turn tool-use) GRPO. The
generation engine (vLLM) runs in a **separate container** — the training env
cannot import vLLM (the server image ships its own torch and
`transformers 5.14.1`, ABI-incompatible with the training image's PyTorch
2.11+cu130 + `transformers 5.16`). `Dockerfile.vllm` pins that 5.14 line and
asserts it at build: vLLM's Gemma 4 code reads the 5.14 config schema that 5.16
folds into `per_layer_config`, so a 5.16 server makes Gemma 4 unservable
(`agent-docs/infrastructure/rollout-servers.md`, config-schema parity). Three further
build gates keep the sync honest — layerwise-reload skip-list coverage, the
weight-transfer re-init patch, and the gpt-oss parser-plugin verifier — so a green
build is what proves the server can be synced (`wiring.md` §1).
Trainer and vLLM communicate over **HTTP** (generation) and **NCCL** (weight
sync via the vendored client). For concrete commands, field-by-field references,
and full launch examples see [`wiring.md`](wiring.md).

## End-to-end wiring

1. **Bring up the vLLM container.** Build `Dockerfile.vllm` (`vllm-server:0.26.0`)
   and start it via `docker-compose.vllm.yml`. It serves the policy model with
   `--weight-transfer-config '{"backend": "nccl"}'` so the NCCL weight-transfer
   endpoints are live. One flow, no version branch: `/pause` →
   `/start_weight_update` → N × `/update_weights` → `/finish_weight_update` →
   `/resume`. **MoE models also need `--moe-backend triton`** (compose default) — the
   auto-selected FlashInfer/CUTLASS backends repack expert weights at load and
   corrupt synced updates. vLLM and the trainer must be on **different GPUs** (set
   `VLLM_CUDA_DEVICES` vs `TRAINER_CUDA_DEVICES`); both compose services run
   `network_mode: host` (the weight-transfer NCCL rendezvous uses an ephemeral
   trainer-advertised port a bridge network can't reach). Both the server and
   training images install the one `nvidia-nccl-cu13` version `uv.lock` pins,
   resolved by `docker/nccl_pin.py`, so the weight-transfer NCCL group links the
   same ABI on both ends.

2. **Point the trainer at it.** Env GRPO reads `AsyncTrainingConfig` from YAML:
   `rollout_backend` (`vllm` default, `sglang` where supported), then
   `rollout_server_url` (single server) or `rollout_server_configs`
   (multi-server) and `rollout_connection_timeout`. The NCCL TCPStore port is
   TRL's own `vllm_group_port` on `GRPOConfig`, not an `AsyncTrainingConfig`
   field. The parser migrates no spelling — an unknown key
   raises rather than being renamed. Online
   GRPO (RLVR) uses TRL's native vLLM fields instead (`use_vllm: true`,
   `vllm_mode: server`, `vllm_server_host`, `vllm_server_port`).

3. **NCCL weight sync** is the vendored `VLLMWeightSyncClient`
   (`src/distributed/nccl/`) — trainer is NCCL rank 0, vLLM workers rank 1+,
   weights pushed as packed broadcasts (`pause → packed NCCL broadcast →
   resume`). Online and env GRPO share **one** gather routine,
   `gather_and_send_weights` (`src/trainers/grpo/rollout/weight_sync.py`), which is
   **parallelism- and PEFT-aware**: in EP / TP / ETP modes **and** under
   multi-rank FSDP2 DP, *every* rank joins the collective gather
   (each EP layer's `gather_expert_state_dict` for experts, `materialize_dtensor`
   plus `iter_tp_sharded_non_dtensor_full` for the hand-sliced TP shards) and only the
   global-main TP-rank-0 process sends; LoRA adapters are merged into the base
   and forwarded under base-model names. Routing multi-rank DP through the
   single-process path would deadlock right after vLLM pauses, so this is not
   optional.

4. **Pick the environment** (env GRPO only) by registry name in
   `EnvironmentConfig.environment_type` (e.g. `react_math`, `native_coding`, `swe`,
   `mcp`, `qa_search`, `code_contests`, `codeforces`, `exam_qa`).
   Per-env knobs go in `environment_kwargs` (e.g. `search_backend`, `open_book`,
   `mcp_server`, `timeout_per_test`);
   reward shaping via `success_reward` / `failure_reward`
   / `max_turns` (default `None` = the environment class's own). Custom envs: pass `environment_cls`
   (a `BaseEnvironment` subclass) instead. See the table in `wiring.md`.

5. **Key GRPO hyperparameters** (both flavors): `num_generations` (group size),
   `beta` (KL to ref; `0.0` disables the ref model), `epsilon` (clip),
   `scale_rewards`, `temperature`, `loss_type`. Env GRPO adds rollout sampling
   (`rollout_temperature` / `rollout_top_p` / `rollout_max_tokens`), Ray pool
   sizing (`num_rollout_workers`, `max_concurrent_rollouts`), prefetch
   (`enable_prefetch`), and `sync_weights_every_n_steps`.
   `rollout_max_thinking_tokens` caps CoT per turn and is enforced **engine-side**
   (vLLM `thinking_token_budget`): it needs a server reasoning parser and
   `VLLM_USE_V2_MODEL_RUNNER=0`, and is refused under `rollout_backend: sglang`.

## Parallelism note

GRPO trainers support **EP** (experts distributed) and **TP** (dense weights
DTensor-sharded); generation is external, so parallelism only affects the
training forward/backward and the weight-sync gather. **CP is unsupported** for
GRPO (`logits_to_keep` + global log-prob sums are incompatible with sequence
splitting) — the GRPO trainers inherit the mixin's `_supports_cp = False` default. Use `torchrun` (not
`accelerate`) for EP/TP.

## Single-server vs multi-server

Single server blocks during weight sync, so prefetch is auto-disabled with a
warning (no overlap possible). For rollout/sync overlap, run multiple servers
via `rollout_server_configs`. Multi-rank runs flush all servers concurrently
during the sync; the rolling sync that keeps (N-1) servers generating exists
only on the single-process path (no EP wrappers, no PEFT).

## Sources of truth
`wiring.md` + `agent-docs/training-methods/grpo/` document the setup. The code is the **ultimate** authority:
`src/trainers/grpo/environmental.py`, `src/distributed/nccl/` (the vendored weight-sync client), and
`Dockerfile.vllm` decide the real handshake — when a doc, this skill, or memory disagrees, or you are
unsure, read those files before wiring it up. (`CLAUDE.md`: docs-first, the code wins.) Related skills:
`data` (the `{prompt, answer}` env-GRPO format), `checkpoints` (merge the trained policy for serving/eval).
