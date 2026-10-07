# Environment Variables

Two things to know before the tables:

**`.env` is never auto-loaded** — not by `docker run`, not by the code. Put
secrets there and pass them in with `docker run --env-file .env` (or a plain
`export`). Compose is the exception: it reads the repo-root `.env` for `${VAR}`
substitution, and the vLLM file's training service loads it into the container.

**The image already sets the tricky ones** — NCCL tuning, CUDA connection
limits, the TF32 fix. Don't paste `-e NCCL_*=...` flags in from other clusters;
the baked defaults are deliberate. The exceptions are the EFA recipe
(`make ... EFA=1`), the no-fabric recipe a GRPO trainer shares with its rollout
server ([cookbook container](cookbooks/README.md#start-the-training-container)),
and `NCCL_SOCKET_IFNAME` on a multi-homed host — see [Clusters](clusters.md).

## Paths — pass these, pointed at a big disk

| Variable | Default | What it holds |
| --- | --- | --- |
| `HF_HOME` | `~/.cache/huggingface` | model downloads |
| `HF_DATASETS_CACHE` | `$HF_HOME/datasets` | dataset / Arrow cache |
| `TMPDIR` | `/tmp` | temp files |
| `HALO_DATA_ROOT` | `~/.cache/halo` | Halo scratch: S3 dataset cache, profiler output |

The defaults land on the root filesystem, which is usually too small for real
runs — see [Installation](installation.md). On the host side, the `make`
targets and the vLLM compose file's `training` service share one variable for
the large volume: `HALO_SCRATCH` (default `/mnt`). Export it once on a host
whose big disk lives elsewhere and every `make` target mounts and caches there.

## Secrets — put these in `.env`

| Variable | Needed for |
| --- | --- |
| `HF_TOKEN` | gated HuggingFace models and datasets |
| `WANDB_API_KEY` | Weights & Biases logging |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_DEFAULT_REGION` | `s3://` datasets and checkpoints (mounting `~/.aws` works too) |
| `OPENAI_API_KEY` / `OPENROUTER_API_KEY` | the external-LLM judge and generation scripts |
| `SERPER_API_KEY` / `TAVILY_API_KEY` / `BRAVE_API_KEY` | the matching web-search backend in the search RL environments (`duckduckgo` needs none) |
| `VLLM_API_KEY` | the `scripts/inference/` and `scripts/environments/` CLIs dialing an authenticated OpenAI-compatible endpoint (a vLLM or SGLang server, or a hosted one); falls back to `OPENAI_API_KEY`, then to the `EMPTY` placeholder a keyless local server accepts |

## Logging and run identity

| Variable | Default | Notes |
| --- | --- | --- |
| `WANDB_PROJECT` | the config's `project_name` | overwritten unconditionally by the trainer setup — an exported value does not survive |
| `WANDB_RUN_ID` | derived from `output_dir` + launch time | export a fixed value on the whole job to continue the same wandb run across restarts |
| `WANDB_RESUME` | unset | wandb SDK knob: set `allow` alongside a fixed `WANDB_RUN_ID` to append instead of starting a new run |
| `CLEARML_PROJECT` / `CLEARML_TASK` | `project_name` / `run_name` basename | set by the trainer, read when `report_to` includes `clearml` |
| `TOKENIZERS_PARALLELISM` | `false` | set at package import (the Rust thread pool deadlocks against dataset-map workers); an exported value wins |

## Multi-node — situational

| Variable | Default | When to set |
| --- | --- | --- |
| `DIST_SHARED_FILESYSTEM` | `1` | umbrella for the two below; set `0` when nodes have per-node local disk instead of shared NFS/Lustre |
| `DIST_INPUT_SHARED_FILESYSTEM` | the umbrella | read side — model/dataset downloads, dataset map/pack, HF caches |
| `DIST_OUTPUT_SHARED_FILESYSTEM` | the umbrella | write side — checkpoints, `run.log`, dumped artifacts |
| `DIST_STORE_TIMEOUT_HOURS` | `4` | raise when one rank's download, dataset map or pack, or queued model load runs longer than four hours while the others wait; this is not the NCCL watchdog |
| `DIST_NCCL_TIMEOUT_MINUTES` | `30` | raise when 100B-scale gathered checkpoint saves or large cross-node all-to-alls outlast the NCCL watchdog |
| `NVLINK_DOMAIN_SIZE` | GPUs per node | `72` on an NVL72 rack, whose NVLink domain spans the rack |
| `NCCL_SOCKET_IFNAME` | `^docker,veth` in the compose bases, `make test-gpu-vllm`/`-sglang` and the no-fabric GRPO recipe; `^lo,docker,veth,tailscale` under `EFA=1` and the EFA overlays; otherwise unset | pin NCCL to the fast NIC on multi-homed nodes |
| `NCCL_NET_PLUGIN=ofi NCCL_NET=Libfabric` | unset | AWS EFA only: the trainer via `make ... EFA=1`, a rollout server via its compose EFA overlay — see [Clusters](clusters.md) |

A side variable inherits the umbrella while unset and overrides it once set.
The case for splitting them: on a multi-node run over NFS/EFS, rank 0 writing
the HF cache while remote ranks read those same inodes is a cross-node
read-after-write that NFS surfaces as `Stale file handle`. Set
`DIST_INPUT_SHARED_FILESYSTEM=0` and leave the umbrella shared, so checkpoints
still land as one authoritative copy. All three must be identical on every
rank; rank 0's values are broadcast and any disagreeing rank warns.

## Tuning knobs worth knowing

Halo has some thirty-five more `HALO_*` knobs, all optional and all defaulted to
production-sane values. They're read through `src/env.py`, so booleans accept
`1/true/yes/on`, and a non-numeric value warns and falls back instead of
crashing mid-run. These are the ones that come up:

| Variable | Default | Purpose |
| --- | --- | --- |
| `HALO_S3_DEFAULT_BUCKET` | unset | bucket for the key-only S3 helpers when a path names none — they raise until it is set |
| `HALO_DATASET_NUM_PROC` | `max(1, min(cpus/4, 4))` | dataset map/filter workers; pin it fleet-wide on heterogeneous nodes |
| `HALO_FP32_MATMUL_PRECISION` | `highest` | fp32 matmul mode; `high` opts back into TF32, which corrupts long-context RoPE — leave it alone |
| `HALO_DEEPEP_GPU_TIMEOUT_SECONDS` | `100` | device-side spin budget of the dispatch/combine barrier — bounds rank skew |
| `HALO_DEEPEP_NUM_QPS` | auto | RDMA queue pairs (elastic backend); more can speed the cross-node all-to-all on EFA — A/B it |
| `HALO_DEEPGEMM_NATIVE` | `0` | native DeepGEMM low-precision kernels — net-slower at the MoE shapes benchmarked here |
| `HALO_FUSED_GLU` | `1` | `0` runs every GLU combine (experts and dense MLPs) eager instead of the fused Triton kernels — the switch when a GLU kernel fails to compile or launch on your GPU |
| `HALO_FLEX_SLIDING` | `1` | `0` builds Gemma 4 on plain SDPA instead of FlexAttention on its sliding layers — the switch when that kernel fails on your GPU, or to run `full_determinism` on more than one GPU |
| `HALO_SANDBOX_BACKEND` / `HALO_SANDBOX_URL` | `local` / unset | code-execution sandbox for RL environments: `local`, `bubblewrap`, or `remote`. `local` does not confine the program; `bubblewrap` needs root and extra container rights ([Async GRPO](training-methods/async-grpo-environments.md#the-environments)) |
| `HALO_ALLOW_MISSING_CHECKPOINT_KEYS` | `0` | demote the missing-checkpoint-key error to a warning; only for deliberately partial checkpoints |
| `CUDA_DEVICE_MAX_CONNECTIONS` | `1`, baked into both images | driver-owned, latched at `deep_ep`'s `cuInit` — a Python write is too late; `1` is the setting EP is validated with, at no measurable throughput cost |

The compose files' rollout-server switches (`VLLM_ENABLE_R3` / `SGLANG_ENABLE_R3`,
`VLLM_SPECULATIVE_CONFIG`, `VLLM_ATTENTION_BACKEND` / `SGLANG_ATTENTION_BACKEND`, …)
are `docker compose` interpolation variables: export them or put them in `.env`
where you run compose. Neither engine reads them, so a hand-run server takes
`--enable-return-routed-experts` / `--speculative-config` / `--attention-backend`
directly. `VLLM_GROUP_HOST` / `SGLANG_GROUP_HOST`, the weight-sync dial-back
address, are read by the trainer. See [Rollout Servers](rollout-servers.md).

The rest — DeepEP buffer sizing, gradient-bucket geometry, low-precision cache
switches, weight-sync timeouts, the EP profiling switches — are cataloged with
their defaults in the
[Configuration Reference](../agent-docs/reference/configuration-reference.md) ↗;
the `HALO_TEST_*` and `*_SERVER_URL` variables belong to the test launcher and
live in [Contributing](../agent-docs/contributing/README.md) ↗. Reach for the
debug switches when [Troubleshooting](troubleshooting.md) sends you there.
