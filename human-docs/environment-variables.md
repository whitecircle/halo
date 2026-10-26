# Environment variables

Most runs set only the paths and secrets below; the image already sets the
tricky ones. Every toolkit variable, with its default, is in the
[Configuration Reference](../agent-docs/reference/configuration-reference.md#environment-variables) ↗.

Halo never loads `.env` itself
([Installation](installation.md#3-create-a-env-file)). The GPU test targets pass
it like `make train` does (`ENV_FILE=` turns that off), and compose also reads
it for `${VAR}` substitution.

**Leave NCCL settings to the image.** It bakes in the NCCL tuning, the CUDA
connection limit and the TF32 fix, so don't copy `-e NCCL_*=...` flags from
other clusters. The exceptions are the EFA recipe (`make ... EFA=1`), the
no-fabric recipe a GRPO trainer shares with its rollout server
([cookbook container](cookbooks/README.md#start-the-training-container)), and
`NCCL_SOCKET_IFNAME` on a multi-homed host ([Clusters](clusters.md)).

## Paths

Point all four at a large disk; their defaults land on the root filesystem,
which is usually too small for real runs:

- `HF_HOME`: model downloads.
- `HF_DATASETS_CACHE`: the dataset and Arrow cache.
- `TMPDIR`: temp files.
- `HALO_DATA_ROOT`: Halo's scratch (S3 dataset cache, profiler output).

On the host, the `make` targets mount the volume named by `HALO_SCRATCH` and
put all four on it; unset, they pick the local mount other than `/` with the
most free space.
The compose `training` service uses it for `TMPDIR` and `HALO_DATA_ROOT`
(falling back to `/mnt`); its HF cache follows the host's `HF_HOME`. Export
`HALO_SCRATCH` once to choose the volume yourself.

## Secrets

`.env` holds the ones you need: `HF_TOKEN` (gated models and datasets),
`WANDB_API_KEY`, the `AWS_*` trio for `s3://` (or a mounted `~/.aws`),
`OPENAI_API_KEY` / `OPENROUTER_API_KEY` for the external-LLM judge and the
generation scripts, `SERPER_API_KEY` / `TAVILY_API_KEY` / `BRAVE_API_KEY` for the
matching web-search backend (`duckduckgo` needs none), and `VLLM_API_KEY` for a
keyed OpenAI-compatible endpoint.

## When to set the others

Boolean variables accept `1`, `true`, `yes` and `on`; a malformed number logs a
warning and falls back to the default.

- **Per-node disks:** `DIST_SHARED_FILESYSTEM=0`. On NFS or EFS that reports
  `Stale file handle`, set only `DIST_INPUT_SHARED_FILESYSTEM=0` and keep the
  output side shared ([Clusters](clusters.md#storage)). Set them identically on
  every rank.
- **Long waits:** raise `DIST_STORE_TIMEOUT_HOURS` when one rank's download,
  dataset pack or queued model load outlasts it, and `DIST_NCCL_TIMEOUT_MINUTES`
  for very large gathered saves or cross-node all-to-alls.
- **NVL72:** `NVLINK_DOMAIN_SIZE=72`, since the NVLink domain spans the rack.
- **One W&B run across restarts:** a fixed `WANDB_RUN_ID` on the whole job, with
  `WANDB_RESUME=allow` ([Monitoring](monitoring.md)).
- **S3 helpers given a bare key:** `HALO_S3_DEFAULT_BUCKET`.
- **Nodes with different CPU counts:** pin `HALO_DATASET_NUM_PROC` cluster-wide.
- **A fused kernel fails on your GPU:** `HALO_FUSED_GLU=0` (the GLU kernels) or
  `HALO_FLEX_SLIDING=0` (Gemma 4's FlexAttention, also needed for
  `full_determinism` on more than one GPU).
- **RL code sandboxes:** `HALO_SANDBOX_BACKEND`; the default `local` does not
  isolate the program ([Async GRPO](training-methods/async-grpo-environments.md#the-environments)).
- **A rollout server dialing back to the trainer:** `VLLM_GROUP_HOST` /
  `SGLANG_GROUP_HOST` when the trainer's routable address is not on its
  default-route NIC ([Rollout servers](rollout-servers.md)).

The rollout-server switches in the compose files (`VLLM_ENABLE_R3`,
`VLLM_SPECULATIVE_CONFIG`, the `*_ATTENTION_BACKEND` pair and others) are
`docker compose` interpolation variables: export them, or put them in `.env`,
where you run compose. The `HALO_TEST_*` variables belong to the test launcher
([Contributing](../agent-docs/contributing/README.md) ↗).
