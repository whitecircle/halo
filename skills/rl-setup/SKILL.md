---
name: rl-setup
description: >-
  Wire up online GRPO (RLVR) or async GRPO with environments (multi-turn tool-use) for
  Halo: bring up the separate rollout container — vLLM 0.32.0 (cu13), or
  SGLang 0.5.22 for async GRPO with environments (rollout_backend: sglang, the families
  its loaders can take an update for)
  — point the trainer at it via AsyncRolloutConfig rollout URLs, establish
  NCCL weight sync through the engine client in halo/weight_sync/
  (parallelism-aware gather — EP/TP/ETP and multi-rank FSDP2 all participate),
  select the RL environment by registry name, and set the key GRPO
  hyperparameters. USER-INVOKED ONLY — invoke when the user explicitly asks to set
  up / configure / wire online or async GRPO, the vLLM or SGLang rollout
  server, Ray rollout actors, or NCCL weight sync.
disable-model-invocation: true
allowed-tools:
  - Read
  - Grep
  - Glob
  - Bash
---

# rl-setup

Wire up online GRPO (RLVR) or async GRPO with environments (multi-turn tool-use). The generation engine
runs in a **separate container** (vLLM 0.32.0, or SGLang 0.5.22 for async GRPO): the training image
cannot import either engine. Trainer and engine talk over **HTTP** (generation) and **NCCL** (weight sync
through the engine client in `halo/weight_sync/`). The owner page for everything server-side is
`agent-docs/infrastructure/rollout-servers.md`; a green image build is what proves the server can be
synced (its build gates). Concrete commands: [`wiring.md`](wiring.md).

## End-to-end wiring

1. **Bring up the server.** `make build-vllm` (or `build-sglang`), or pull the public image and point
   `VLLM_IMAGE` / `SGLANG_IMAGE` at it; then `docker-compose.vllm.yml` / `docker-compose.sglang.yml`, which
   refuse to start without `HALO_SCRATCH` (vLLM) / `SGLANG_MODEL_DIR` (SGLang): export them or set them in
   the repo-root `.env`. Server and trainer GPUs must be **disjoint** (`VLLM_CUDA_DEVICES` vs
   `TRAINER_CUDA_DEVICES`: a rank cannot broadcast to itself), both containers run `network_mode: host`,
   and **MoE needs the triton expert backend** (the compose default; the auto backends repack expert
   weights and corrupt synced updates). Use the toolkit images, not upstream: both pin the training
   image's NCCL. Server flags, parsers per family, R3: `rollout-servers.md#vllm` / `#sglang`.

2. **Point the trainer at it.** Env GRPO reads `AsyncRolloutConfig` from YAML: `rollout_backend`
   (`vllm` default; `sglang` for the families its loaders take an update for — the trainer refuses the
   rest at construction, naming the loader reason), then `rollout_server_url` (single server) or
   `rollout_server_configs` (multi-server), `rollout_connection_timeout` and the NCCL group port
   `rollout_group_port`. Online GRPO (RLVR) uses TRL's native vLLM fields instead
   (`use_vllm: true`, `vllm_mode: server`, `vllm_server_host`, `vllm_server_port`, `vllm_group_port`). Field defaults:
   `agent-docs/reference/configuration-reference.md`.

3. **NCCL weight sync** is `VLLMWeightSyncClient` / `SGLangWeightSyncClient`
   (`halo/weight_sync/clients/`): the trainer is NCCL rank 0, the server workers rank 1+. Online and env
   GRPO share **one** parallelism- and PEFT-aware gather, `gather_and_send_weights`
   (`halo/trainers/grpo/rollout/weight_push.py`): under EP / TP / ETP **and** multi-rank FSDP2 every rank
   joins the gather and only the forwarding rank sends; LoRA folds out of place into each base weight.
   Routing multi-rank DP through a single-process path deadlocks right after the server pauses. Which
   families each engine can update, and what refuses at construction:
   `rollout-servers.md#which-families-each-engine-serves`, `#construction-gates`.

4. **Pick the environment** (env GRPO only) by registry name in `environment_type`; per-env knobs go in
   `environment_kwargs`; reward shaping is the `rewards:` term list; `max_turns: null` keeps the class
   default. Registry and knobs: `agent-docs/training-methods/async-grpo/environments/README.md#registry`.
   Custom envs: pass `environment_cls` (an `EnvironmentBase` subclass) instead.

5. **Key GRPO hyperparameters** (both flavors): `num_generations` (group size), `beta` (KL to ref;
   `0.0` disables the ref model), `epsilon` (clip), `scale_rewards`, `temperature`, `loss_type`. Env GRPO
   adds rollout sampling (`rollout_temperature` / `rollout_top_p` / `rollout_max_tokens` /
   `rollout_chat_template_kwargs`), Ray pool sizing (`num_rollout_workers`, `max_concurrent_rollouts`),
   prefetch and `sync_weights_every_n_steps`, and the reasoning budgets: `rollout_max_thinking_tokens` is
   enforced engine-side on vLLM only (it needs a server reasoning parser) and refused under
   `rollout_backend: sglang`, as is `carry_reasoning`
   (`agent-docs/training-methods/async-grpo/reasoning.md`).

## Parallelism note

Online and environment GRPO trainers support **EP** (experts distributed), **TP** (dense weights
DTensor-sharded) and **ETP**; generation is external, so parallelism only affects the
training forward/backward and the weight-sync gather. **CP is unsupported** for
these trainers: environment GRPO uses `logits_to_keep`, and online GRPO runs a separate-length
rollout sequence. Both inherit `SUPPORTS_CP = False` and declare their own `PP_REFUSAL`. Offline GRPO
supports CP for full fine-tuning, without a rollout server. Use `torchrun` (not
`accelerate`) for EP/TP.

## Single-server vs multi-server

Prefetch overlaps rollouts with training, not with the sync: every push pauses
all servers together. It is on by default with 2+ `rollout_server_configs`; a
single server runs serial, training each round on rollouts of the current weights, unless
`enable_prefetch: true`, which trains each round on samples one round old (IS-corrected). `prefetch_depth: N` (default 1)
collects N rounds ahead at once, so the servers stay full at a round's tail; above
1 it is refused where prefetch is off. On SGLang each sync aborts in-flight turns,
so a prefetched round spanning more syncs than `max_retries` is refused.

## Sources of truth

`agent-docs/infrastructure/rollout-servers.md` + `agent-docs/training-methods/` document the setup
(`grpo/rewards/README.md` owns the term list, `async-grpo/setup.md` the method's launch). The code is the
**ultimate** authority: `halo/trainers/grpo/environmental/trainer.py`, `halo/weight_sync/` (the weight-sync
clients), and `Dockerfile.vllm` decide the real handshake — when a doc, this skill, or memory disagrees,
or you are unsure, read those files before wiring it up. (`CLAUDE.md`: docs-first, the code wins.) Related
skills: `data` (the async-GRPO format: `prompt`, plus `answer` where the environment's `requires_answer`
is set), `checkpoints` (merge the trained policy for serving/eval).
