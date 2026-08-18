# Environment Variables

Two things to know before the tables:

**`.env` is never auto-loaded** — not by `docker run`, not by the code. Put
secrets there and pass them in with `docker run --env-file .env` (or a plain
`export`). The one exception is the vLLM compose file, whose training service
reads the repo-root `.env` itself.

**The image already sets the tricky ones** — NCCL tuning, CUDA connection
limits, the TF32 fix. Don't paste `-e NCCL_*=...` flags in from elsewhere; the
baked defaults are deliberate.

## Paths — pass these, pointed at a big disk

| Variable | Default | What it holds |
| --- | --- | --- |
| `HF_HOME` | `~/.cache/huggingface` | model downloads |
| `HF_DATASETS_CACHE` | `~/.cache/huggingface/datasets` | dataset / Arrow cache |
| `TMPDIR` | `/tmp` | temp files |
| `HALO_DATA_ROOT` | `~/.cache/halo` | Halo scratch: S3 dataset cache, profiler output |

The defaults land on the root filesystem, which is usually too small for real
runs — see [Installation](installation.md). On the host side, the `make`
targets and the compose files share one variable for the large volume:
`HALO_SCRATCH` (default `/mnt`). Export it once on a host whose big disk lives
elsewhere and every `make` target mounts and caches there.

## Secrets — put these in `.env`

| Variable | Needed for |
| --- | --- |
| `HF_TOKEN` | gated HuggingFace models and datasets |
| `WANDB_API_KEY` | Weights & Biases logging |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_DEFAULT_REGION` | `s3://` datasets and checkpoints (mounting `~/.aws` works too) |
| `OPENAI_API_KEY` / `OPENROUTER_API_KEY` | the external-LLM judge and generation scripts |
| `SERPER_API_KEY` / `TAVILY_API_KEY` / `BRAVE_API_KEY` | the matching web-search backend in the search RL environments (`duckduckgo` needs none) |
| `VLLM_API_KEY` | the playground and generation scripts dialing an authenticated vLLM endpoint; falls back to `OPENAI_API_KEY`, then to the `EMPTY` placeholder a keyless local server accepts |

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
| `DIST_STORE_TIMEOUT_HOURS` | `4` | raise when one rank's model download or corpus pack runs longer than four hours while the others wait; this is not the NCCL watchdog |
| `DIST_NCCL_TIMEOUT_MINUTES` | `30` | raise when slow dataset prep or 100B-scale checkpoint saves outlast the NCCL watchdog |
| `NVLINK_DOMAIN_SIZE` | GPUs per node | `72` on GB200/GB300 NVL72 racks |
| `NCCL_SOCKET_IFNAME` | auto | pin NCCL to the fast NIC on multi-homed nodes |
| `FI_PROVIDER=efa` + friends | unset | AWS EFA fabrics only — see [Clusters](clusters.md) |

A side variable inherits the umbrella while unset and overrides it once set.
The case for splitting them: on a multi-node run over NFS/EFS, rank 0 writing
the HF cache while remote ranks read those same inodes is a cross-node
read-after-write that NFS surfaces as `Stale file handle`. Set
`DIST_INPUT_SHARED_FILESYSTEM=0` and leave the umbrella shared, so checkpoints
still land as one authoritative copy. All three must be identical on every
rank; rank 0's values are broadcast and any disagreeing rank warns.

## Tuning knobs worth knowing

Halo has around twenty more `HALO_*` knobs, all optional and all defaulted to
production-sane values. They're read through `src/env.py`, so booleans accept
`1/true/yes/on`, and a non-numeric value warns and falls back instead of
crashing mid-run. These are the ones that come up:

| Variable | Default | Purpose |
| --- | --- | --- |
| `HALO_S3_DEFAULT_BUCKET` | `my-bucket` (placeholder) | bucket for the key-only S3 helpers when a path names none — set it before using them |
| `HALO_DATASET_NUM_PROC` | `max(1, min(cpus/4, 4))` | dataset map/filter workers; pin it fleet-wide on heterogeneous nodes |
| `HALO_FP32_MATMUL_PRECISION` | `highest` | fp32 matmul mode; `high` opts back into TF32, which corrupts long-context RoPE — leave it alone |
| `HALO_DEEPEP_GPU_TIMEOUT_SECONDS` | `100` | device-side spin budget of the dispatch/combine barrier — bounds rank skew |
| `HALO_DEEPEP_NUM_QPS` | auto | RDMA queue pairs; helps on EFA |
| `HALO_DEEPGEMM_NATIVE` | `0` | native DeepGEMM low-precision kernels — net-slower at the MoE shapes benchmarked here |
| `HALO_SANDBOX_BACKEND` / `HALO_SANDBOX_URL` | `local` / unset | code-execution sandbox for RL environments: `local`, `bubblewrap`, or `remote` |
| `VLLM_GROUP_HOST` / `SGLANG_GROUP_HOST` | auto | trainer IP the rollout server dials back for the weight-sync group; set it when the server runs on another host |
| `VLLM_USE_V2_MODEL_RUNNER` | unset | server-side: must be `0` for any run setting `rollout_max_thinking_tokens` (V2 rejects thinking budgets with a 400) |
| `HALO_ALLOW_MISSING_CHECKPOINT_KEYS` | `0` | demote the missing-checkpoint-key error to a warning; only for deliberately partial checkpoints |
| `CUDA_DEVICE_MAX_CONNECTIONS` | `1`, baked into both images | driver-owned, latched at `deep_ep`'s `cuInit` — a Python write is too late; `1` is worth +9.7% on ep8 |

The rest — DeepEP buffer sizing, gradient-bucket geometry, low-precision cache
switches, weight-sync timeouts, the EP profiling switches — are catalogued with
their defaults in the
[Configuration Reference](../agent-docs/reference/configuration-reference.md) ↗;
the `HALO_TEST_*` and `*_SERVER_URL` variables belong to the test launcher and
live in [Contributing](../agent-docs/contributing/README.md) ↗. Reach for the
debug switches when [Troubleshooting](troubleshooting.md) sends you there.
