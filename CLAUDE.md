# CLAUDE.md

Guidance for Claude Code and other AI agents working in this repository. These rules override default behavior — follow them exactly. `agent-docs/` is the detailed reference for how the code behaves; this file is the rulebook and a map into it. `human-docs/` is the concise human guide.

## Rules

### Environment & tooling
- **Run everything in the Docker image.** The host has **no usable Python** (`import torch` fails). All deps (PyTorch 2.11+cu130, DeepEP, Flash Attention) live only in the images: `halo:blackwell` (B200/B300, SM100/103, FA4+FA2, no FA3) or `halo:hopper` (H100/H200, SM90, FA2+FA3), built credential-free from source (`make build-blackwell` / `make build-hopper` — no token or secret needed) or pulled prebuilt from **Amazon ECR Public** (anonymous, no AWS account): `docker pull public.ecr.aws/whitecircle/halo:{blackwell,hopper}` — moving tags plus immutable `-$(VERSION)` pins, deliberately no `latest` (it would let a Hopper host pull a Blackwell image); the RL servers are `:vllm-0.26.0` and `:sglang-0.5.17` in the same repository (`make push-public-all` publishes, gated by `docker/scan_image.sh`). On a host whose docker default runtime rejects `--gpus` (e.g. sysbox-runc), set `DOCKER_RUNTIME=nvidia` on any `make` target. Images are built with **uv** into the system interpreter — **no Poetry, no venv** — so `python`/`torchrun`/`accelerate`/`pytest`/`ruff`/`mkdocs` are on `PATH`; call them directly, no prefix. Dependency changes go through uv (`uv lock`, `uv pip install`; see `pyproject.toml` `[tool.uv]`). Standard detached launch:
  ```bash
  # Resolve the scratch volume dynamically (see Preflight) — never hardcode a path.
  D=$(findmnt -rbno TARGET,AVAIL,FSTYPE | awk '$3!~/tmpfs|overlay|squashfs|nfs|fuse|autofs/ && $2+0>20e9{print $2,$1}' | sort -rn | head -1 | awk '{print $2}')
  docker run -d --rm --name <job> --gpus all \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=128g \
    --cap-add=SYS_PTRACE --env-file .env \
    -e HF_HOME="$D/hf" -e HF_DATASETS_CACHE="$D/hf/datasets" \
    -e TMPDIR="$D/tmp" -e HALO_DATA_ROOT="$D" \
    -v $(pwd):/workspace -v "$D:$D" -v /root/.aws:/root/.aws -w /workspace \
    halo:blackwell \
    bash -lc "torchrun --nproc_per_node=8 <script> <config> > \"$D/<job>.log\" 2>&1"
  ```
  Provide your own repo-root `.env` (`WANDB_API_KEY`/`HF_TOKEN` as needed, plus any `AWS_*` for S3; start from `.env.example`); it is **not auto-loaded** — pass it with `--env-file`. `HF_HOME`/`HF_DATASETS_CACHE` (HF caches), `TMPDIR` (temp), and `HALO_DATA_ROOT` (toolkit scratch — S3 dataset cache → `$HALO_DATA_ROOT/s3_datasets`, profiler artifacts → `$HALO_DATA_ROOT/profiling`) redirect writes off the small root FS. They are a **convention pointing at a large mounted volume**, not code defaults (each falls back to `~/.cache/...`), and `/mnt` is **not guaranteed large** (on some hosts `/mnt` shares the small root device) — confirm with `findmnt`/`df -h` and point them at the real large volume. Pre-cached S3 datasets (`md5("<bucket>/<key>")`) then load without live AWS. See `agent-docs/infrastructure/docker.md`.
- **Launcher:** `torchrun` for all multi-GPU — EP/CP/TP/ETP **and** plain FSDP2 data-parallel (`torchrun --nproc_per_node=N scripts/training/<script>.py <config>` lands on FSDP2 ZeRO-2); `python` for single-GPU/LoRA. `accelerate launch` + the `launcher-configs/accelerate/*.yaml` configs remain a supported option for plain FSDP data-parallel, but torchrun is the default. The `halo` CLI (`src/cli.py`) picks the launcher — `halo launch <method> <config>`, `halo run <tool>`.
- **Preflight every run — disk, RAM, GPU.** Send large outputs, caches (`HF_DATASETS_CACHE`), and logs to a **verified** large mounted volume; the root FS is small and on some hosts shares a device with `/mnt`. **Check the target before any multi-GB write** (`df -h` / `findmnt` / `readlink -f` — a path's name does not prove its capacity). Before launching also confirm free host RAM and GPU health/free memory (`nvidia-smi`; `scripts/profiling/nvlink_health.py` for NVLink) — never launch onto a full disk, an OOM-prone host, or a degraded or already-occupied GPU. Scratch/temp markdown → `/tmp`.
- **Env vars.** `src/env.py` is the single home for reading toolkit knobs (`HALO_`/`DIST_`/`VLLM_`) — one convention via `env_flag`/`env_int`/`env_float`/`env_str`, plus `HALO_DATA_ROOT` + `data_path()` for scratch. Route a new toolkit var through it, never a raw `os.environ.get`. Launcher (`LOCAL_RANK`/`SLURM_*`/`RANK`), `ACCELERATE_*`, and HF/OS vars are read raw at their owner by design. Full catalogue with defaults (incl. `NVLINK_DOMAIN_SIZE`, the NVL72 locality unit): `agent-docs/reference/configuration-reference.md` (Environment variables).

### Code style
- **Clean and professional — no slop.** No inline / function-local imports — the *only* exception is a genuinely optional or arch-specific dependency that may be absent at runtime (`flash_attn`, `deep_ep`, `kernels`), marked `# noqa: PLC0415`. A **circular import is never that exception** — fix it structurally (move the shared symbol to a leaf module, or stop a package `__init__` from eagerly importing a heavy subpackage), never with a function-local import or a lazy-import wrapper. No duplicated logic, no hardcoded paths or magic values (read them from config/env — `src/env.py` is the home for env flags), no sprawling or misplaced files, no papering over a root cause with a local patch. Reuse existing code, and put each thing where it belongs.
- **Good file structure.** Lay a module out top-down: docstring → imports → module-level constants → public API/functions. Keep module-level constants near the top (a constant that depends on a helper goes just after it), group related code, and place a file in the directory that owns its concern. Don't bury a definition mid-file.
- **Comments are sparse and load-bearing.** Explain a non-obvious *why*; do not narrate what the code plainly does, do not leave long block comments, and never tell stories in comments (past runs, "previously X", "was broken then fixed", dated notes) — that history belongs in git, not in source.
- **Derive from the class hierarchy or a registry** — a per-class attribute the base unions over registered subclasses, polymorphism or a registry — never a hand-maintained list or a `model_type ==` string-ladder that duplicates what a class already knows.
- **Generalize to the whole roster** (dense + every MoE family) and to large scale, while **avoiding over-engineering** (no gold-plating, no speculative abstraction, no registry where branches are genuinely distinct).
- **Fail loud** over a silent fallback that hides a real error.
- **Refactors must be behavior-preserving and proven so** — ruff + in-image tests, ideally an equivalence check — before they land.
- **Hold the whole matrix in mind** before landing anything load-bearing: all parallelism modes (EP/CP/TP/ETP + combinations, incl. `ep_size==1`+`fsdp_shard_ep1_experts` DTensor experts vs `ep_size>1` FSDP-ignored plain tensors), multi-node topologies (node-local vs cross-node EP, NVLink domains, shared vs non-shared FS, deferred cross-node DP), and every supported model (each MoE family has its own expert layout/gather). A change correct for one model/single-node/dense-FSDP can silently break another (a plain-vs-DTensor `copy_`, a collective that must run on every rank, a per-family interleave, a non-shared-FS writer race). Prefer model-agnostic, sharding-agnostic seams.

### Docs & markdown
- **Docs first.** Check `agent-docs/` (the detailed AI reference) before searching the code; only read `src/` directly if the docs don't answer it. `human-docs/` is the concise human guide.
- **Markdown only in `agent-docs/`** (detailed reference), **`human-docs/`** (human guide) **and `skills/`** (agent skills; `.claude/skills` and `.agents/skills` are symlinks to it for Claude Code and Codex discovery). Root GitHub-convention files (`README.md`, `CONTRIBUTING.md`, `AGENTS.md`, `CLAUDE.md`, `.github/**`) are the exception. Never create `.md` elsewhere; scratch/design notes go to `/tmp`, not the repo.
- **Keep docs current.** A significant `src/` change updates its owning doc page (ownership map: `skills/docs/docs-ownership.md`). Docs follow the lean **present-state** style charter (`skills/docs/style.md`, via the `docs` skill): load-bearing facts only, no changelog/story-telling, cite sparingly. Run `mkdocs build --strict` after `agent-docs/` edits; `human-docs/` is plain GitHub-rendered markdown (no build), so just keep its relative links valid.

### Process
- **Read code thoroughly before changing it.** Don't judge from an isolated grep hit — read whole files when they aren't too large, and follow definitions and references through the call graph to understand the actual logic and its callers.
- **Use multiple agents, often — it's the default, not the exception.** For any substantive investigation, audit, review, or refactor, fan out several **disjoint-scoped** subagents (Explore / general-purpose) that work in parallel, then **adversarially self-review** — verify every finding against the actual code and runtime data (agents can be wrong), and read their diffs yourself before accepting.
- **Verify before calling it done.** After a code change, run ruff and the in-image tests it touches, confirm it complies with these rules (no slop, holds across the parallelism/model matrix), and prefer an independent review pass — a fresh agent checking correctness and rule-compliance — on anything load-bearing.
- **Fix root causes,** not symptoms — state the mechanism, then fix it (not "lower the LR" around a drift).
- **Finish the user-facing path, not just the guard.** Every supported configuration owes the user a complete path: train → resume → **export/serve**. Turning a silently-wrong artifact into a loudly-unusable one is half a fix — if the change leaves a checkpoint no tool accepts, close that gap in the same change instead of documenting the dead end. A trainable artifact nobody can serve is not a deliverable, and "only the guard was asked for" is not a reason to ship one.
- **Contribution gate + bar** (`CONTRIBUTING.md`): Halo is a **public, gated** project — every change starts as an accepted **issue** (use a `.github/ISSUE_TEMPLATE`), a maintainer must `/approve` you (which adds you to `.github/APPROVED_CONTRIBUTORS` on the `allowlist` branch) **before** any PR, and PRs from anyone not on that list (without write access) are **auto-commented and closed** by `.github/workflows/pr-gate.yml`. A reaction, comment, branch, or draft reserves nothing. **Any coding agent working in this repo must respect that gate** — do not open a PR off an unapproved path. The bar: own every line, disclose the AI scaffold + models in the PR (template checkbox), keep diffs under ~2,000 lines, and write tests that **fail when the behavior breaks** (no tautological / smoke-only / vacuous assertions). Full guide: `agent-docs/contributing/index.md`.

## Project Overview

Halo — LLM alignment toolkit extending HuggingFace (Transformers, TRL, Accelerate) with Expert / Context / Tensor / Expert-Tensor Parallelism and alignment methods (SMPO, Offline/Online/Environmental GRPO, distillation, and more). 10+ training methods across dense and 15 MoE families (GptOss, Qwen3, Qwen3.5/3.6, GLM4, GLM5-Next, Laguna, Inkling, Cohere2MoE/Command A+, Gemma4, Bailing/Ling, LFM2, Mistral4, DeepSeek-V4, Zaya, Step-3.7) via DeepEP, multi-turn RL with Ray actors, and multi-node training. Datasets default to HuggingFace Hub or local paths, with optional S3 (`s3://my-bucket/...`, bring your own credentials).

**Stack:** Python 3.12 · uv (`pyproject.toml` + `uv.lock`) · PyTorch ~2.11 (+cu130) · Transformers ~5.16 · TRL ~1.6 · Accelerate ~1.11 · PEFT ~0.18 · Liger ^0.8. Base image NGC `nvcr.io/nvidia/pytorch:26.03-py3` (Ubuntu 24.04, CUDA 13.2).

**RL serving:** vLLM 0.26.0 runs as a **separate** container (`Dockerfile.vllm` + `docker-compose.vllm.yml`, both `network_mode: host`) with native NCCL weight transfer; the training env uses a vendored NCCL client (`src/distributed/nccl/`), never imports vLLM (ABI-incompatible stacks). Two gates keep expert sync honest: the server's layerwise-reload patch must skip the layers that `copy_` their weights directly (`FusedMoE`/`RoutedExperts` experts, `OAIAttention` sinks, `Gemma4Router` — `RoutedExperts`, `OAIAttention` and `Gemma4Router` asserted at build), and the server must run `--moe-backend triton` (auto FLASHINFER/CUTLASS repack expert weights and corrupt updates). A third gate is trainer-side: weight sync refuses a GptOss whose sinks the FA2 `reset_sinks` reset removed, since nothing would be pushed for those slots. The image also asserts its weight-transfer re-init patch at build, without which the engine strands an NCCL communicator per trainer connection until `ncclCommInitRank` fails; the trainer's own client aborts its communicator on close for the same reason. Environmental GRPO can target **SGLang** instead (`rollout_backend: sglang`, `Dockerfile.sglang` + `docker-compose.sglang.yml`, NCCL matched to the training image); it is refused at construction under any expert distribution and for every MoE family but GptOss (the only layer implementing the fused expert gather SGLang loads — 0.5.17's `qwen3_moe` loader maps per-expert names only and silently drops fused keys, so Qwen3 MoE is refused too). SGLang R3 capture needs `--enable-return-routed-experts --moe-runner-backend triton`; `rollout_max_thinking_tokens` stays vLLM-only; the trainer wants `fsdp_reshard_after_backward: false` (SGLang's sync forces socket NCCL process-global, making FSDP2's per-microstep reshard the dominant step cost otherwise) — a plain-DP/CP/EP lever, rejected under TP or PP. Online GRPO is vLLM-only by construction. See `agent-docs/infrastructure/rollout-servers.md` and `agent-docs/training-methods/grpo/environmental-grpo.md`.

**Debug/profiling** (`agent-docs/reference/debugging.md`): distributed-debug helpers in `src/diagnostics/debugging.py` (opt-in via env vars, zero cost when off); `enable_torch_profiler: true` captures per-rank traces with EP phases labeled; `scripts/profiling/trace_report.py` (TraceLens workbooks), `py_spy_diag.py` (attach py-spy; needs `--cap-add=SYS_PTRACE`), `nvlink_health.py` (torch-free NVLink preflight — separates hard faults / near-exhausted FEC from the benign `Xid 145` churn).

## Commands

```bash
make install                     # uv pip install from uv.lock (in-image)

# Single GPU / LoRA
python scripts/training/sft.py examples/sft/qwen3/qwen3-4b-ultrachat.yaml
# Plain FSDP data-parallel (accelerate)
accelerate launch --config_file launcher-configs/accelerate/fsdp2_gradop_config.yaml \
    scripts/training/sft.py examples/sft/qwen3/qwen3-4b-ultrachat.yaml
# EP/CP/TP (torchrun)
torchrun --nproc_per_node=8 scripts/training/sft.py \
    examples/sft/gptoss/gptoss-20b-multinode-ep.yaml --expert_parallel_size=8
```
Training is YAML-driven; override any field on the CLI (`--learning_rate=1e-5 --max_length=32000`). Redirect `HF_DATASETS_CACHE`/`TMPDIR` to `/mnt` and detach with `nohup ... > /mnt/<job>.log 2>&1 &` for production runs.

### Tests

pytest-driven. CPU tests are pytest-native (`pytest -m cpu`); GPU tests are `torchrun` scripts registered in `tests/gpu/manifest.py`, launched by marker via `tests/gpu/conftest.py`. Both run standalone too. `make test-cpu` / `test-gpu-core` / `test-gpu-full` wrap the in-image invocation. New tests must **fail when the behavior breaks** (`agent-docs/contributing/index.md`).

```bash
python tests/cpu/callbacks/test_variable_scheduler.py                 # CPU (no GPU)
torchrun --nproc_per_node=2 \                                         # GPU, standalone
    tests/gpu/trainers/preference/test_smpo_fsdp.py
```

Never hardcode `--master_port` in a test — the launcher allocates a free one per run
(`tests/common/ports.py::free_port`); fixed ports raced between back-to-back tests and CI shards.
A CPU test ends in `raise SystemExit(pytest.main([__file__, "-v"]))`, a GPU test in the `gpu_test_main`
entry (`tests/common/harness.py`) that owns its lifecycle; neither declares `pytestmark` (the `cpu`
marker is applied by path) nor bootstraps `sys.path` (the root `tests/conftest.py` owns the import
path) — `tests/cpu/conventions/test_test_conventions.py` fails the suite over any of the three. Per-suite
`HALO_TEST_*` knobs override the model, parallel axes, attention impl and step counts of a GPU suite
without editing it (catalogue: `agent-docs/contributing/index.md`).

`tests/cpu/` is grouped by subsystem (`callbacks/ checkpoint/ config/ conventions/ data/ diagnostics/ environments/ grpo/ inference/ kernels/ models/ optimizers/ parallelism/ peft/ trainers/`). `tests/gpu/` splits into `trainers/{sft,preference,grpo,lora,other}/`, `parallelism/{ep,cp,tp,combined}/`, plus `data/ kernels/ optimizers/ profiling/`. Shared helpers in `tests/common/`.

## Architecture

```
src/
├── trainers/            # all extend DistributedTrainerMixin (mixins/base.py — lifecycle/accelerator/optimizer
│   │                    #   spine; mixins/ also holds checkpointing, dataloader, ep_introspection, grad_sync,
│   │                    #   token_metrics, validation, pipeline (composed by the base; per-trainer mixins like
│   │                    #   StoredMetrics are mixed in individually) + pp_gates, loss_masks, grad_clip functions)
│   ├── sft.py preference/ (DPO,SMPO,KTO)  reward/ (bradley_terry, classification)
│   ├── embedding/       # SBERT trainer + sentence_transformers_compat.py (ST patches, preloaded-model shim)
│   ├── grpo/            # online, offline, environmental + objective/ mixins/ subpackages and rollout/
│   │                    #   (weight_sync, weight_sync_clients, async_rollouts, routing_replay,
│   │                    #   trajectory_tokenize, trajectory_spans, rollout_metrics, completions_logging)
│   └── distillation/    # teacher_distillation, self_distillation, sdpg (online); losses.py = the shared
│                        #   objectives + SDPG schedule, teacher_losses.py = the eight off-policy losses
├── optimizers/          # AdamWBF16 (SR), Muon, FlashAdamW (each module owns its build_*, named in registry.py)
├── kernels/             # grouped_gemm (precision dispatch) over the grouped_mm_autograd primitive,
│                        #   fused_glu, histogram, liger/, lowp/(quantization, linear, deepgemm,
│                        #   mixed_precision — the opt-in fp8/fp4 stack)
├── models/              # sharding-agnostic model side: structure.py (module-tree introspection),
│                        #   moe_balancing.py (mode resolve + router/EP-family registries the layer classes
│                        #     push into, and the offline balancing-slot apply), modality.py (multimodal
│                        #   checkpoint detection), attention_geometry.py (head-dim / KV-head resolve),
│                        #   seq_cls_heads.py (Gemma4 + Qwen3.5-MoE seq-cls, registered by an explicit
│                        #   import in loading/model_preparation.py),
│                        #   patches/(attention — backend select off hardware.py's arch predicates —
│                        #     gpt_oss_sinks, zaya, kernel_dispatch, remote_code_compat +
│                        #     remote_code_hooks = the one wrap of transformers' remote-class funnel, buffer_fixes),
│                        #   loading/(model_preparation = Auto* class + family patches + the shared post-load
│                        #     finalize_run_model, config_levels = composite-config field access, tokenizer_setup =
│                        #     processing class + length budget, dtype = dtype + fp32-matmul precision,
│                        #     checkpoint_coverage = random-init gate; lazy_safetensors/ = generic safetensors
│                        #     lazy-load core — index, key alignment, plans, conversion ops, meta shell —
│                        #     shared by the EP + PP loaders)
├── distributed/
│   ├── parallelism_config.py  # ParallelismConfig — central validation gate (every config-time raise)
│   ├── group_layout.py mesh.py  # EP/CP rank math; DeviceMesh construction + the typed ParallelDims view
│   ├── runtime.py filesystem.py fsdp.py nvlink.py  # rank/world + barriers + consensus, FS coordination,
│   │                    #   FSDP2 wraps/reshard, fabric probes (runtime.py is the package leaf; the rest import it)
│   ├── module_registry.py  # HF-class→claimant registries (EP+CP+PP)
│   ├── grad_reduce.py    # bucketed gradient all-reduce (EP cross-replica, TP replicated, QLoRA sweeps)
│   ├── loading/         # parallelism-aware model construction: model_loading (the load_distributed_model
│   │                    #   dispatcher), vlm_setup (modality-aware entry points), frozen_models (unsharded
│   │                    #   teacher/reference loads + freeze), peft_setup, warmup (the fenced FA4 warm-up)
│   ├── checkpoint/      # save.py saver ladder (select_checkpoint_saver → FSDP2/CP/TP/EP savers), weight loader, OptimizerShardStore,
│   │                    #   write.py (the collective half of a write: retain-gated gather, streamed parts, index exchange),
│   │                    #   coordination.py (rank consensus shared by both halves of a resume), PeftAdapterSaver
│   ├── expert_parallel/ context_parallel/ tensor_parallel/ pipeline_parallel/  # EP (DeepEP), CP (Ulysses), TP (DTensor, incl. tie_plan.py), PP
│   │                    #   expert_parallel/balancing_strategy.py owns the router-balancing export contract
│   └── nccl/            # vendored vLLM weight-sync client (registry.py resolves the rollout backend)
├── checkpoint/          # sharding-agnostic checkpoint layer (no src.distributed / torch.distributed):
│                        #   format.py (on-disk spellings, save-dtype casts, the layout cascade),
│                        #   config_export.py (what an exported config.json must contain), adapters.py (saved-PEFT
│                        #   layout, shape gates, merge-into-base), tool_io.py (tool-side directory I/O),
│                        #   shard_writer.py (StageShardWriter — incremental safetensors parts),
│                        #   fp8_dequant.py (streaming fp8 → bf16)
├── data/                # collators/ = batch-time collate_fn (completions_only, packing incl. the padded CP
│                        #   collator, offline_grpo, vlm, vlm_preference, smpo, self_distill, classification
│                        #   + fixed_shape wrapper; factory.py builds only completions_only/packing —
│                        #   every other collator is imported from its leaf)
│                        #   pipeline/ = dataset-time coordinated_map (conversation/preference/prompt rendering,
│                        #     row_processors, preprocessing bake, preprocessed_metadata = the metadata.json
│                        #     contract, vlm_dataset)
│                        #   sources/(paths, s3_client, dataset_cache, loading, sharded_dataset)
│                        #   vlm.py (the cross-package VLM leaf both halves render through)
│                        #   spans.py (turn terminators, completion spans and the one completion-mask implementation)
│                        #   shard_index.py (torch-free shard_index.json contract) probe_consensus.py deduplication.py
├── environments/        # RL envs: base.py (the env protocol) + episode.py (the rollout-driver contract),
│                        #   registry, ray_actors, eval_runner, rewards, engine_wire (rollout wire format),
│                        #   tools/ (incl. web_search), sandbox/, envs/(protocols/, tasks/)
├── callbacks/           # ParameterStats, EfficiencyCallback, MoEMetricsCallback, RouterBiasBalancingCallback,
│                        #   VariableSchedulerCallback, TorchProfilerCallback; wiring.build_perf_callbacks
│                        #   resolves moe_balancing
├── diagnostics/         # opt-in, off by default: debugging.py (hang/skew consistency checks, py-spy capture),
│                        #   profiling.py (torch-profiler traces, CUDA memory snapshots),
│                        #   performance_monitor.py (opt-in EP span timing)
├── training/            # entry-script plumbing: parser.py (H4ArgumentParser), environment.py (output dir,
│                        #   HF caches, seed, resume detection), script_runner.py (the scripts/training/**
│                        #   backbone), parallelism_args.py (DistributedArguments → ParallelismConfig),
│                        #   run_logging.py (run.log tee)
├── inference/           # OpenAI-compatible client for served rollout/judge endpoints: openai_client.py,
│                        #   response.py (OpenAIResponse), resume_store.py (resumable request logs)
├── env.py log.py hardware.py  # torch-free env-var helpers (single home for os.environ reads); root logging +
│                        #   warn_once; GPU model detection + peak-FLOPs table + host-RAM probe
└── cli.py args/ configs/  # halo CLI; per-method arg dataclasses; config classes
```
`scripts/` uses pipeline-stage folders: `training/ inference/ before_training/ after_training/ profiling/ environments/` (plus `diagrams/`, the docs-figure generators). Entry scripts share their flag surface per subtree — `scripts/inference/_common.py`, `scripts/inference/reward_model/_common.py`, `scripts/environments/_common.py` — or above them where the tools chain across subtrees (`scripts/_common.py`, the checkpoint tools' flags) — while the driver stays in `src/`. Configs in `examples/` split per method then per model family; environmental GRPO further splits per rollout backend (`vllm/`, plus `sglang/` where supported — gpt-oss only, ep1 only) with one file per adapter (`-lora-`/`-full-`) × expert distribution (`-ep1`/`-ep4`). Layer map and leaf-module contracts: `agent-docs/reference/architecture.md`; per-topic depth via the doc index below.

**Distributed trainers** (all extend `DistributedTrainerMixin`; EP/TP/ETP on all, CP where noted):

| Trainer | Purpose | CP |
|---|---|:--:|
| `DistributedSFTTrainer` | Supervised fine-tuning | ✅ |
| `SmoothMarginPOTrainer` | Reference-free preference (SMPO) | ✅ |
| `OfflineGRPOTrainer` / `DistributedGRPOTrainer` | Offline / online GRPO | ❌ |
| `DistributedAsyncEnvironmentalGRPOTrainer` | Multi-turn environment RL (Ray + vLLM) | ❌ |
| `DistributedDPOTrainer` / `DistributedKTOTrainer` | DPO / KTO | ❌ |
| `DistributedRewardTrainer` / `ClassificationTrainer` | BT reward / sequence classification | ❌ |
| `DistributedDistillationTrainer` / `DistributedSelfDistillationTrainer` / `DistributedSDPGTrainer` | Knowledge / self- / online (SDPG) distillation | ❌ |
| `EmbeddingTrainer` | Embedding fine-tuning (SBERT losses) | ❌ |

CP is declare-to-enable per trainer (`_supports_cp`, default off) — the CP column above **is** the gate; nothing inspects a trainer's loss for CP-safety, so a new trainer must verify its own objective before flipping the flag. Config classes: `ParallelismConfig`, `OfflineGRPOConfig`, `SmoothMarginPOConfig`, `DistillationConfig`, `ClassificationConfig`, `EmbeddingConfig`, `EnvironmentConfig`, `AsyncTrainingConfig` (+ `RolloutConfig`, the pickle-safe leaf mirror of its rollout fields that `get_rollout_config()` hands the Ray actors). Full field reference: `agent-docs/reference/configuration-reference.md`.

## Parallelism

`data_parallel_size = world_size / max(cp_size, tp_size, expert_tp_size)` (EP is orthogonal to DP; the divisor is a **max**, not a product — no two of those axes may exceed 1 at once, so they never compound). `ParallelismConfig` validates and **rejects** invalid shapes at config time — which axis combinations may run at all is an **allowlist** (`SUPPORTED_AXIS_SETS` in `parallelism_config.py`) checked before any rank math, so an unlisted combination is rejected by default rather than running unvalidated.

| Mode | Notes |
|---|---|
| EP | DeepEP expert routing; orthogonal to DP |
| TP | DTensor mesh; `tp_size` must divide the NVLink domain. Refuses every LoRA shape at trainer construction — native EP expert adapters included, since both TP gates skip them by param identity |
| CP | Ulysses attention; reduces DP |
| ETP pure (`ep_size=1`) | expert FFN sharded `expert_tp_size`-way, experts replicated (MoE-only, experimental) |
| EP+CP | node-local EP needs `ep_group_size == nvlink_domain_size` |
| EP+TP | TP node-local. On **>1 NVLink domain** the EP group must be a **single** global one (`ep_size==world`, `ep_scope=global`); within one domain multiple EP groups are fine (`ep2+tp2` on 8 = 4 EP groups) |
| EP+ETP (`ep_size>1` & `expert_tp_size>1`) | experimental; expert-TP reduce runs in **token space** (outside DeepEP dispatch→combine). Cross-node: exactly one ETP group per NVLink domain |
| TP+ETP (EP+TP+ETP included) · TP+CP · ETP+CP | **NOT SUPPORTED** |
| PP | **not yet available in this release** — the config surface (`pipeline_parallel_size`/`pipeline_schedule`/`pipeline_microbatches`/`pipeline_split`), rank math, trainer gates (`src/trainers/mixins/pp_gates.py`) and stage/loss seams (`src/distributed/pipeline_parallel/`) ship, but the schedule engine behind `PipelineRuntime` does not: `pipeline_parallel_size > 1` is rejected at config time. See `agent-docs/parallelism/pipeline-parallelism.md` |

**Single-domain pure EP topology (prevents wasted 8×B300 runs):** one dispatch group per NVLink domain — `ep_group_size == nvlink_domain_size` (shrink the job, or size `ep_size` to the domain), or `ep_size == 2`. **`ep_size > 2` with `ep_group_size < nvlink_domain_size`** (e.g. ep4 on 8 → two 4-rank groups) is **rejected at config time** — the DeepEP combine barriers race FSDP2's DP-wide NCCL: the `elastic` default faults with `Invalid access of peer GPU memory over nvlink`, `legacy` deadlocks (`is_racy_single_domain_multigroup_ep`; trainer re-checks hand-built configs). The unit is the NVLink domain, not the OS node — on NVL72 with `NVLINK_DOMAIN_SIZE=72` that rejects `ep8` too. For 4-way+ sharding across all 8 GPUs, combine EP with **ETP** (EP+TP does not help — attention TP leaves `ep_group_size` at 4, so `ep4+tp2` trips the same rejection). **Multi-node multi-group EP is supported** (EP groups are DP replicas; cross-replica average deferred to a post-backward sweep). `CUDA_DEVICE_MAX_CONNECTIONS=1` is baked into the images (latched at `deep_ep` import's `cuInit` — a Python write is too late); free default, +9.7% on ep8. See `agent-docs/infrastructure/deepep.md`, `agent-docs/parallelism/multi-node.md`; multi-node launch recipes in `agent-docs/parallelism/launch-recipes.md` (+ `agent-docs/infrastructure/runpod.md`, `agent-docs/infrastructure/skypilot.md`).

**When a run fails** — config rejection, NCCL hang/timeout, OOM, DeepEP build/runtime fault — start at `agent-docs/reference/troubleshooting.md` (symptom → cause → fix); live-hang, py-spy, and collective-skew diagnosis in `agent-docs/reference/debugging.md`.

## Data & config

**Dataset formats** (`agent-docs/data/dataset-formats.md`):
- SFT: `{"prompt": [{"role","content"}, ...]}` (field via `conversation_field`)
- Preference (DPO/SMPO): `{"prompt", "chosen", "rejected"}` (all `list[dict]`)
- Offline GRPO: `{"prompt", "completions": [[...]], "rewards": [...]}`
- Environmental GRPO: `{"prompt", "answer"}`

Sources: `s3://`, HF Hub, or local (`src/data/sources/`). Offline tokenize/pack/shard via `scripts/before_training/prepare_dataset.py`; `ShardedDatasetLoader` for per-rank shards; `fs_aware_main_first()` adapts to shared vs local FS (`DIST_SHARED_FILESYSTEM`, default `1` = shared NFS/Lustre; set `0` for per-node local storage). Filesystem coordination: `agent-docs/data/filesystem-handling.md`.

**YAML parser** (`H4ArgumentParser`, `src/training/parser.py`): auto-applies toolkit defaults `use_liger_kernel: true`, `bf16: true` and `logging_nan_inf_filter: false`; migrates no spelling — every retired key, TRL's own `max_seq_length` included, hits the unknown-key raise; rejects YAML 1.1 bool spellings (`yes`/`no`/`on`/`off` — 1.2 parses them as truthy strings) on bool fields; expands strftime **directives only** in `output_dir` (other `%` prose survives).

**Add a method:** arg/config class in `src/args/` or `src/configs/` → subclass `DistributedTrainerMixin` in `src/trainers/<family>/` → script in `scripts/training/` → YAML in `examples/` → tests in `tests/`. Trainer `__init__` does `kwargs = self._init_distributed_config(kwargs)` then `super().__init__(...)` then `self._setup_distributed_modes()`.

## Key facts & gotchas

Terse rules; depth lives in the linked docs.

- **FSDP2** (`fully_shard`, `reshard_after_forward=False`) for all torchrun DP. EP modules are FSDP-ignored **except** when experts are truly replicated (`ep_group_size==1`): `fsdp_shard_ep1_experts` (default `True`) shards them so their reduce-scatter is the sole grad sync (frees DP-growing memory, grad-equivalent, RL-safe). Anything that reads parameters outside a forward — the optimizer build, every checkpoint writer, `save_model`, RL weight sync — calls `reshard_fsdp2_modules` first: a forward leaves transient **unsharded** params registered while the optimizer steps the shards, so reading them ships state one optimizer step stale (and mis-routes the PEFT save, whose DTensor probe then sees plain tensors). Accelerate FSDP v1 has a known corruption bug with SHARD_GRAD_OP/FULL_SHARD — prefer FSDP v2 or DDP.
- **EP expert export** goes through each family's `gather_expert_state_dict` (gathered saves, vLLM sync, `--merge_expert_lora_on_save` all route here). Per-expert un-fused hub layouts come from `_PER_EXPERT_UNFUSED_KEYS` auto-split (declared by GLM4/LFM2; Laguna inherits GLM4's) or the shared `_gather_individual_glu_state_dict` helper (Bailing/Qwen3); only genuinely distinct layouts override the gather with their own logic (GptOss re-interleave, Gemma4 prefix-strip; Zaya, Cohere2, glm5_next and step3p7 use the base fused gather — Zaya's hub layout is fused-native, while Cohere2's/glm5_next's per-expert and step3p7's per-layer-stacked hub spellings are restored by transformers' save-side converter). `agent-docs/reference/checkpoints.md`.
- **Attention:** production auto-selects **FA4** on Blackwell (`flash_attn.cute`; 2.1–3.7× the FA2 attention kernel, smaller end-to-end), FA2+FA3 on Hopper (FA3 needs the from-source build + `docker/training/flash_attn_split_stubs_hopper.cpp`). **GptOss sinks** (`src/models/patches/gpt_oss_sinks.py`, one `SinksPolicy` per run): SFT neutralizes them (frozen `dtype.min`) so FA2/SDPA match eager; on-policy RL (`reset_sinks: false`) keeps them live and frozen, so `validate_attn_implementation` rejects FA2/SDPA and only sink-carrying impls run; `train_sinks: true` (SFT, full fine-tuning, FA4 or eager) trains them — FA4's fused backward emits no sink gradient, so the loader routes grad-requiring sinks through an exact sink-less + `sigmoid(lse - sink)` rescale. `agent-docs/optimization/flash-attention.md`.
- **EP requires DeepEP V2** (`deep_ep.ElasticBuffer` over NCCL Gin; no NCCL fallback; hidden padded to ×256). Do not install `nvidia-nvshmem-cu12`. One arena per forward: the first MoE layer's all-reduced MAX capacity (aligned to 256) is cached and reused by every later layer (`HALO_EP_CAPACITY_DEDUP`, default on), and a layer that dispatches past it raises instead of under-sizing the wire buffer. A pre-hook on the outermost module the loop calls opens each scope — re-register it whenever a wrapper lands above that module (a `PeftModel` reaches the model it wraps through `.forward()`, running no pre-hook), and call `bump_forward_generation()` from any path that enters the backbone directly (TRL's chunked log-probs). `agent-docs/infrastructure/deepep.md`.
- **Grouped GEMM** (`torch.nn.functional.grouped_mm`) auto-enabled for MoE expert compute on SM90+ (`use_grouped_gemm: false` to disable). `agent-docs/optimization/grouped-gemm.md`.
- **AdamWBF16** auto-enabled when `bf16: true` on FSDP/EP/TP (6 B/param): stochastic rounding on the weight write **and** `exp_avg_sq` (nearest biased the 2nd moment ~+50%). Not auto-enabled under DDP (outside the validated SR matrix; `bf16_optimizer=True` opts in). `fp32_grad_reduce: true` reduces grads in fp32 at bf16 storage (~2.2× tighter at world=8). `agent-docs/optimization/bf16-optimizer.md`.
- **fp32 matmul precision** pinned to `highest` at model load (`configure_float32_matmul_precision`, knob `HALO_FP32_MATMUL_PRECISION`; Dockerfile also sets `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0`). The NGC image defaults fp32 matmuls to TF32, whose 10-bit mantissa collapses adjacent long-context RoPE token positions past 2048 (the large `inv_freq @ position_ids`) — corrupting every RoPE model beyond 2048 tokens. bf16 matmuls (the bulk of a bf16 run) are unaffected. `agent-docs/reference/configuration-reference.md`.
- **Mixed precision** (`lowp_precision: bf16|fp8|fp4|mxfp4`): bf16/fp32 masters + low-precision matmul; **bf16 is the production default** (no fp4 MMA win at our MoE shapes). Simulated backend = correct numerics + QAT oracle, not faster; native DeepGEMM is opt-in (`HALO_DEEPGEMM_NATIVE=1`) and net-slower here. `quantize_to_lowp.py` exports mxfp8/nvfp4. `agent-docs/optimization/low-precision-moe-kernels.md`.
- **Router balancing** (`moe_balancing`, resolved by `resolve_balancing_mode` in `src/models/moe_balancing.py`): `bias_update` (DeepSeek-V3 ALF) holds an **export contract** — the sign-updates must land in the family's own checkpoint slot so the served model routes as trained: native buffer (Zaya `balancing_biases`), adopted slot (GLM-4/GLM-5/Laguna/DSv4/Inkling/Step-3.7 `e_score_correction_bias`, Bailing `expert_bias`, LFM-2 `expert_bias`, GPT-OSS `router.bias`), or — where that slot is config-gated and off (LFM-2 `use_expert_bias: false`) — materialized: zero buffer plus the config flag flipped for the export. An adopted `nn.Parameter` slot is re-registered as a persistent buffer under the same key, freezing it out of gradient training. The verdict is read off the *enabled* tree, not the class: a renamed upstream slot or one already FSDP-sharded as a DTensor falls back to the side-buffer, and `bias_update` then raises. Families with **no slot** (Qwen3, Qwen3.5/3.6, Mistral4, Cohere2 MoE) refuse `bias_update`; the explicit `bias_update_transient` opts into a trainer-only side-buffer there (exports serve without it — near-tie top-k flips). `auto` → `bias_update` on the three signals (native buffer; wrapper severs the aux path; forward never consults `output_router_logits` while a wrapper accepts the bias) **only where the bias exports**; Mistral4, Cohere2 MoE and multimodal Qwen3.5/3.6 resolve to `none` + warning instead; every other MoE resolves to `aux_loss` only where the enabled tree can serve it (the forward declares `output_router_logits` — Laguna, GLM-5, Qwen3, GPT-OSS …) and otherwise to `none` + warning (a forward that never takes the flag with no bias-accepting wrapper on the tree — the `ep_size=1` + `use_grouped_gemm: false` window for GLM-4, LFM-2, Step-3.7); dense → `none`. `aux_loss` is left off — warning, not crash — when the model has no usable `router_aux_loss_coef`; Gemma 4 lands there and its EP wrapper accepts no bias either, so it has **no balancing route at all** and an explicit bias mode raises. `aux_loss` is inert under a policy-gradient (GRPO) loss, and both bias modes are downgraded to `none` under weight-sync RL (the sync ships parameters only). Biases are checkpointed to `router_balancing_biases.pt` and restored on resume; the mode's config writes (zeroed coef, forced `output_router_logits`, the forced-off stamp) are run-scoped and restored in every exported `config.json` (`config_export_ready`), the save keeps balancing tensors at trained fp32 (`balancing_param_keys`), and the PEFT merge tools apply the sidecar into a merged model's native slots (`apply_router_balancing_sidecar` in `src/models/moe_balancing.py`, off the slot registry the EP layer classes register into). `agent-docs/training-methods/callbacks.md`.
- **Zaya** (`Zyphra/ZAYA1-*`) is native in transformers 5.14+ (hub `main` = the native fused format; no revision pin, no `trust_remote_code`). Toolkit patches at load (`src/models/patches/zaya.py`): expert-load recording for the gate's native `balancing_biases` (auto → `bias_update` in every mode), a GC refusal (upstream allows GC; backward recompute through the CCA `nn.Conv1d` faults in cuDNN and per-layer GC re-wraps the EDA state), and flash `position_ids` plumbing for packing. Supported: EP (lazy loading included), EP+ETP, plain FSDP2 — all **without GC**. Unsupported: TP, CP, PP. Packing: attention isolates, the CCA conv crosses document boundaries (accepted mixer-class sharing). Pre-5.14 per-expert checkpoints are unreadable — re-derive via `patch_vocab.py` from hub `main`. `agent-docs/models/zaya.md`.
- **Model loading** batches ranks per node. `max_concurrent_loading` is unset (`null`) by default and resolves node-width-aware to `min(4, max(1, local_world_size // 2))` — 4 on an 8-GPU node, 2 on a 4-GPU tray; every explicit value is used verbatim (`1` for CPU-RAM-constrained, `0` for all-parallel). Every path that materializes weights — the distributed loaders and any `scripts/` tool whose model reaches a forward — ends in `finalize_loaded_model` (`src/models/patches/buffer_fixes.py`): transformers 5 re-materializes non-persistent buffers as uninitialized memory, and a remote-code family that overrides `_init_weights` without `super()` never repairs them (a garbage or zeroed `inv_freq` degenerates RoPE to NoPE at a plausible loss). `tests/cpu/models/test_load_finalization.py` sweeps the loaders and classifies each tool inference-vs-conversion.
- **Checkpoint-tool CLIs** share one source/destination spelling (`--input_dir`/`--adapter_dir` → `--output_dir`, `--model_id` where the source may be a Hub repo) and one `--trust_remote_code`, whose default follows that source: a local checkpoint or adapter defaults **on**, a Hub-capable `--model_id` defaults **off**. Retired flag spellings are removed, not aliased. `agent-docs/reference/scripts-reference.md`.

## Documentation

- **Get started** (`agent-docs/index.md`): [Installation](agent-docs/getting-started/installation.md) · [Quickstart](agent-docs/getting-started/quickstart.md) · [Configuration](agent-docs/getting-started/configuration.md) · [Choosing a Method](agent-docs/getting-started/choosing-a-method.md)
- **Models**: [Overview](agent-docs/models/index.md) · [Qwen3](agent-docs/models/qwen3.md) · [Qwen3.5/3.6](agent-docs/models/qwen3_5.md) · [GPT-OSS](agent-docs/models/gpt-oss.md) · [GLM-4](agent-docs/models/glm4.md) · [GLM-5 Next](agent-docs/models/glm5-next.md) · [Laguna](agent-docs/models/laguna.md) · [Inkling](agent-docs/models/inkling.md) · [Gemma 4](agent-docs/models/gemma4.md) · [Bailing/Ling](agent-docs/models/bailing.md) · [LFM-2](agent-docs/models/lfm2.md) · [Mistral4](agent-docs/models/mistral4.md) · [DeepSeek-V4](agent-docs/models/deepseek-v4.md) · [Zaya](agent-docs/models/zaya.md) · [Cohere2 MoE](agent-docs/models/cohere2-moe.md) · [Step-3.7 Flash](agent-docs/models/step3p7.md) · [Adding a Model](agent-docs/models/adding-a-model.md)
- **Training methods**: [Overview](agent-docs/training-methods/index.md) · [SFT](agent-docs/training-methods/sft.md) · [Pretraining](agent-docs/training-methods/pretraining.md) · [DPO](agent-docs/training-methods/preference/dpo.md) · [SMPO](agent-docs/training-methods/preference/smpo.md) · [KTO](agent-docs/training-methods/preference/kto.md) · [GRPO](agent-docs/training-methods/grpo/index.md) ([Offline](agent-docs/training-methods/grpo/offline-grpo.md) · [Online](agent-docs/training-methods/grpo/online-grpo.md) · [Environmental](agent-docs/training-methods/grpo/environmental-grpo.md) · [Comparison](agent-docs/training-methods/grpo/grpo-comparison.md)) · [Preference](agent-docs/training-methods/preference/index.md) · [Reward](agent-docs/training-methods/preference/reward-modeling.md) · [Classification](agent-docs/training-methods/classification.md) · [Distillation](agent-docs/training-methods/distillation/index.md) ([Teacher](agent-docs/training-methods/distillation/teacher-distillation.md) · [Self](agent-docs/training-methods/distillation/self-distillation.md) · [SDPG](agent-docs/training-methods/distillation/online-sdpg.md)) · [Embedding](agent-docs/training-methods/embedding.md) · [Callbacks](agent-docs/training-methods/callbacks.md)
- **Environments**: [Overview](agent-docs/training-methods/grpo/environments/index.md) · [ReAct](agent-docs/training-methods/grpo/environments/react.md) · [Native Tool-Use](agent-docs/training-methods/grpo/environments/native-tool-use.md) · [SWE](agent-docs/training-methods/grpo/environments/swe-environment.md) · [Code Contests](agent-docs/training-methods/grpo/environments/code-contests.md) · [Sandboxes](agent-docs/training-methods/grpo/environments/sandbox.md) · [MCP](agent-docs/training-methods/grpo/environments/mcp.md) · [Benchmarks](agent-docs/training-methods/grpo/environments/benchmarks.md) · [Custom](agent-docs/training-methods/grpo/environments/custom-environments.md)
- **Parallelism**: [Overview](agent-docs/parallelism/index.md) · [Data](agent-docs/parallelism/data-parallelism.md) · [Expert](agent-docs/parallelism/expert-parallelism.md) · [Expert-Tensor](agent-docs/parallelism/expert-tensor-parallelism.md) · [Tensor](agent-docs/parallelism/tensor-parallelism.md) · [Context](agent-docs/parallelism/context-parallelism.md) · [Pipeline](agent-docs/parallelism/pipeline-parallelism.md) · [Multi-Node](agent-docs/parallelism/multi-node.md) · [Large-Scale Scenarios](agent-docs/parallelism/large-scale-scenarios.md) · [Launch Recipes](agent-docs/parallelism/launch-recipes.md) · [Data Loading](agent-docs/parallelism/data-loading.md)
- **Data**: [Overview](agent-docs/data/index.md) · [Formats](agent-docs/data/dataset-formats.md) · [Collators](agent-docs/data/collators.md) · [Pre-Processing](agent-docs/data/dataset-preparation.md) · [Filesystem](agent-docs/data/filesystem-handling.md) · [S3](agent-docs/data/s3-utilities.md)
- **Optimization**: [Overview](agent-docs/optimization/index.md) · [PEFT](agent-docs/optimization/peft.md) · [BF16 Optimizer](agent-docs/optimization/bf16-optimizer.md) · [Padding-Free](agent-docs/optimization/padding-free-collator.md) · [Flash Attention](agent-docs/optimization/flash-attention.md) · [Grouped GEMM](agent-docs/optimization/grouped-gemm.md) · [Liger](agent-docs/optimization/liger-kernels.md) · [torch.compile](agent-docs/optimization/torch-compile.md) · [Throughput](agent-docs/optimization/throughput-benchmarks.md) · [Low-Precision MoE](agent-docs/optimization/low-precision-moe-kernels.md) · [Muon](agent-docs/optimization/muon-optimizer.md) · [FlashAdamW](agent-docs/optimization/flash-adamw.md) · [vs Stock TRL](agent-docs/optimization/halo-vs-stock-trl.md)
- **Infrastructure**: [Overview](agent-docs/infrastructure/index.md) · [Docker](agent-docs/infrastructure/docker.md) · [AWS Auth](agent-docs/infrastructure/aws-auth.md) · [DeepEP](agent-docs/infrastructure/deepep.md) · [RunPod](agent-docs/infrastructure/runpod.md) · [SkyPilot](agent-docs/infrastructure/skypilot.md) · [Nomad](agent-docs/infrastructure/nomad.md) · [Rollout Servers](agent-docs/infrastructure/rollout-servers.md) · [Ray Cluster](agent-docs/infrastructure/ray.md) · [CI](agent-docs/infrastructure/ci.md)
- **Reference**: [Overview](agent-docs/reference/index.md) · [Why This Framework](agent-docs/reference/why-this-framework.md) · [GPU Training Theory](agent-docs/reference/gpu-training-theory.md) · [Architecture](agent-docs/reference/architecture.md) · [Trainer Architecture](agent-docs/reference/trainer-architecture.md) · [Configuration](agent-docs/reference/configuration-reference.md) · [Checkpoints](agent-docs/reference/checkpoints.md) · [Model Merging](agent-docs/reference/model-merging.md) · [Scripts](agent-docs/reference/scripts-reference.md) · [Scale & Limits](agent-docs/reference/scale-and-limitations.md) · [Debugging](agent-docs/reference/debugging.md) · [Troubleshooting](agent-docs/reference/troubleshooting.md) · [Glossary](agent-docs/reference/glossary.md)
- **Development**: [Contributing](agent-docs/contributing/index.md) · [Dev Environment](agent-docs/contributing/development-environment.md)
