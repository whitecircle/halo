# Environment variables

Most runs need only the paths and secrets below. The image already sets the
tricky ones.

**`.env` is not loaded automatically.** Halo's code never reads it, and
`docker run` needs `--env-file .env`. Two launch paths pass it for you:

- `make train` and the GPU test targets (set `ENV_FILE=` to turn it off).
- The `training` service in `docker-compose.vllm.yml`. Compose also reads the
  repo-root `.env` for `${VAR}` substitution.

**Leave NCCL settings to the image.** It bakes in the NCCL tuning, the CUDA
connection limit and the TF32 fix, so don't copy `-e NCCL_*=...` flags from
other clusters. The exceptions:

- the EFA recipe (`make ... EFA=1`)
- the no-fabric recipe a GRPO trainer shares with its rollout server
  ([cookbook container](cookbooks/README.md#start-the-training-container))
- `NCCL_SOCKET_IFNAME` on a multi-homed host ([Clusters](clusters.md))

## Paths

Point all four at a large disk:

| Variable | Default | What it holds |
| --- | --- | --- |
| `HF_HOME` | `~/.cache/huggingface` | model downloads |
| `HF_DATASETS_CACHE` | `$HF_HOME/datasets` | dataset and Arrow cache |
| `TMPDIR` | `/tmp` | temp files |
| `HALO_DATA_ROOT` | `~/.cache/halo` | Halo scratch: S3 dataset cache, profiler output |

The defaults land on the root filesystem, which is usually too small for real
runs ([Installation](installation.md)).

On the host, the `make` targets take the large volume from `HALO_SCRATCH`
(default `/mnt`). They mount it and put all four paths on it. The compose
`training` service uses it for `TMPDIR` and `HALO_DATA_ROOT`; its HF cache
follows the host's `HF_HOME`. Export `HALO_SCRATCH` once on a host whose large
disk is elsewhere.

## Secrets

Put these in `.env`:

| Variable | Needed for |
| --- | --- |
| `HF_TOKEN` | gated HuggingFace models and datasets |
| `WANDB_API_KEY` | Weights & Biases logging |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_DEFAULT_REGION` | `s3://` datasets and checkpoints (mounting `~/.aws` also works) |
| `OPENAI_API_KEY` / `OPENROUTER_API_KEY` | the external-LLM judge and generation scripts |
| `SERPER_API_KEY` / `TAVILY_API_KEY` / `BRAVE_API_KEY` | the matching web-search backend in the search RL environments (`duckduckgo` needs no key) |
| `VLLM_API_KEY` | the `scripts/inference/` and `scripts/environments/` CLIs calling an OpenAI-compatible endpoint that requires a key. Falls back to `OPENAI_API_KEY`, then to `EMPTY`, which a keyless local server accepts |

## Logging and run identity

| Variable | Default | Notes |
| --- | --- | --- |
| `WANDB_PROJECT` | the config's `project_name` | the trainer always sets it from `project_name`, so an exported value is overwritten |
| `WANDB_RUN_ID` | derived from `output_dir` and launch time | export a fixed value on the whole job to continue the same W&B run across restarts |
| `WANDB_RESUME` | unset | a W&B SDK setting: `allow`, with a fixed `WANDB_RUN_ID`, appends to that run instead of starting a new one |
| `CLEARML_PROJECT` / `CLEARML_TASK` | `project_name` / derived from `run_name` | set by the trainer; read when `report_to` includes `clearml` |
| `TOKENIZERS_PARALLELISM` | `false` | set at import to avoid a tokenizer deadlock in dataset-map workers; an exported value wins |

## Multi-node

Set these only when the situation calls for it:

| Variable | Default | When to set |
| --- | --- | --- |
| `DIST_SHARED_FILESYSTEM` | `1` | `0` when nodes have their own local disks instead of shared NFS or Lustre; covers both variables below |
| `DIST_INPUT_SHARED_FILESYSTEM` | follows `DIST_SHARED_FILESYSTEM` | read side: model and dataset downloads, dataset map and pack, HF caches |
| `DIST_OUTPUT_SHARED_FILESYSTEM` | follows `DIST_SHARED_FILESYSTEM` | write side: checkpoints, `run.log`, dumped artifacts |
| `DIST_STORE_TIMEOUT_HOURS` | `4` | raise when one rank's download, dataset map or pack, or queued model load takes longer while the others wait. This is not the NCCL watchdog |
| `DIST_NCCL_TIMEOUT_MINUTES` | `30` | raise when very large gathered checkpoint saves or cross-node all-to-alls outlast the NCCL watchdog |
| `NVLINK_DOMAIN_SIZE` | GPUs per node | `72` on an NVL72 rack, whose NVLink domain spans the rack |
| `NCCL_SOCKET_IFNAME` | set by the compose files, `make ... EFA=1` and the rollout-server test targets; otherwise unset | pin NCCL to the fast NIC on a multi-homed node |
| `NCCL_NET_PLUGIN=ofi NCCL_NET=Libfabric` | unset | AWS EFA only. `make ... EFA=1` sets them for the trainer, the compose EFA overlay for a rollout server ([Clusters](clusters.md)) |

Split the read and write sides when NFS or EFS reports `Stale file handle` on a
multi-node run. That happens when rank 0 writes the HF cache while other nodes
read it. Set `DIST_INPUT_SHARED_FILESYSTEM=0` and keep `DIST_SHARED_FILESYSTEM=1`:
each node downloads its own copy, and checkpoints still land once.

Set all three to the same values on every rank. Rank 0's values win, and a rank
that disagrees logs a warning.

The `NCCL_SOCKET_IFNAME` default each recipe sets is in
[Rollout servers](../agent-docs/infrastructure/rollout-servers.md) ↗.

## Tuning knobs

Halo reads about thirty-five more `HALO_*` knobs, all optional. Booleans accept
`1`, `true`, `yes` and `on`. A non-numeric value where a number belongs logs a
warning and falls back to the default. These are the ones that come up:

| Variable | Default | Purpose |
| --- | --- | --- |
| `HALO_S3_DEFAULT_BUCKET` | unset | bucket for the S3 helpers when a path names only a key; they raise until it is set |
| `HALO_DATASET_NUM_PROC` | `max(1, min(cpus/4, 4))` | dataset map and filter workers; pin it cluster-wide when nodes differ in CPU count |
| `HALO_FP32_MATMUL_PRECISION` | `highest` | fp32 matmul mode. `high` turns TF32 back on, which corrupts long-context RoPE, so leave it alone |
| `HALO_DEEPEP_GPU_TIMEOUT_SECONDS` | `100` | seconds a rank waits at the DeepEP dispatch/combine barrier for its peers, which bounds rank skew |
| `HALO_DEEPEP_NUM_QPS` | auto | RDMA queue pairs for DeepEP. More can speed up the cross-node all-to-all on EFA; A/B test it |
| `HALO_DEEPGEMM_NATIVE` | `0` | native DeepGEMM low-precision kernels; net-slower at the MoE shapes benchmarked here |
| `HALO_FUSED_GLU` | `1` | `0` runs every GLU combine (experts and dense MLPs) in eager PyTorch instead of the fused Triton kernels. Use it when a GLU kernel fails to compile or launch on your GPU |
| `HALO_FLEX_SLIDING` | `1` | `0` runs Gemma 4 on plain SDPA instead of FlexAttention on its sliding layers. Use it when that kernel fails on your GPU, or to run `full_determinism` on more than one GPU |
| `HALO_SANDBOX_BACKEND` / `HALO_SANDBOX_URL` | `local` / unset | code-execution sandbox for RL environments: `local`, `bubblewrap` or `remote`. `local` does not isolate the program; `bubblewrap` needs root and extra container privileges ([Async GRPO](training-methods/async-grpo-environments.md#the-environments)) |
| `HALO_ALLOW_MISSING_CHECKPOINT_KEYS` | `0` | turn the missing-checkpoint-key error into a warning; only for deliberately partial checkpoints |
| `CUDA_DEVICE_MAX_CONNECTIONS` | `1`, set in both images | read once when CUDA initializes, so setting it from Python is too late. EP is validated at `1`, which costs no measurable throughput |

The rest (DeepEP buffer sizing, gradient-bucket geometry, low-precision cache
switches, weight-sync timeouts, EP profiling) are listed with their defaults in
the [Configuration Reference](../agent-docs/reference/configuration-reference.md) ↗.
The `HALO_TEST_*` and `*_SERVER_URL` variables belong to the test launcher and
are in [Contributing](../agent-docs/contributing/README.md) ↗. Turn on the debug
switches when [Troubleshooting](troubleshooting.md) points you to them.

## Rollout-server switches

The compose files' rollout-server switches (`VLLM_ENABLE_R3` /
`SGLANG_ENABLE_R3`, `VLLM_SPECULATIVE_CONFIG`, `VLLM_ATTENTION_BACKEND` /
`SGLANG_ATTENTION_BACKEND` and others) are `docker compose` interpolation
variables. Export them, or put them in `.env`, where you run compose. A server
you start by hand takes the matching flags directly: `--enable-return-routed-experts`,
`--speculative-config`, `--attention-backend`.

`VLLM_GROUP_HOST` and `SGLANG_GROUP_HOST` are read by the trainer: the address
the rollout server connects back to for weight sync. See
[Rollout servers](rollout-servers.md).
