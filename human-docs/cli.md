# The `halo` CLI

The image ships a `halo` command with two verbs:

- `halo launch` starts a training method.
- `halo run` starts any other tool: checkpoint post-processing, data prep,
  inference, diagnostics.

Each one maps a name to a script under `scripts/` and runs it with the right
launcher. You can also run the scripts directly with `python` or `torchrun`.

## `halo launch`

```bash
halo launch <method> <config.yaml> [launcher flags] [field overrides]
```

| Flag | Short | Meaning |
| --- | --- | --- |
| `--nproc N` | `-n` | number of processes (GPUs); `N > 1` launches with `torchrun` |
| `--accelerate <yaml>` | `-a` | launch with `accelerate launch` and this config (plain FSDP) |
| `--port P` | `-p` | rendezvous port for a multi-process launch; set it when two jobs share a host |
| `--numa-binding <mode>` | | `torchrun`'s NUMA binding (`node` runs each rank on its GPU's NUMA node); `torchrun` launches only, refused with `--accelerate` or one process ([NUMA binding](../agent-docs/optimization/throughput-benchmarks.md#numa-binding) ↗) |
| `--list` | | print every method and its script |
| `--dry_run` | | print the command instead of running it |
| `--root <dir>` | | repository root (defaults to the checkout the CLI lives in); the script imports `halo` and `scripts` from it |

The CLI picks the launcher:

- one process: plain `python`
- `--nproc` above 1: `torchrun` (required for EP, CP, TP, ETP and PP)
- `--accelerate`: `accelerate launch`

Other flags after the config go to the trainer as is, so config overrides ride
along:

```bash
halo launch sft examples/sft/qwen3/qwen3-4b-ultrachat-lora.yaml
halo launch sft examples/sft/qwen3/qwen3-4b-ultrachat.yaml -n 8
halo launch sft examples/sft/gptoss/gptoss-20b-multinode-ep.yaml -n 8 \
    --expert_parallel_size=8 --use_grouped_gemm=true --learning_rate=1e-5
```

A method name is the script's path under `scripts/training/`, with hyphens for
underscores, grouped as `examples/` is: `sft`, `classification`, `preference/smpo`,
`grpo/offline`, `grpo/online`, `grpo/environmental` (async GRPO with environments),
`reward/reward-model`, `distillation/self-distill`. A file stem unique across the
methods works alone (`smpo`, `dpo`, `reward-model`).

The config path can be absolute, or relative to the repo root or your current
directory. The run executes from the repo root, so relative paths inside the
config, such as `output_dir`, resolve there.

The CLI is single-node and rejects torchrun's multi-node flags. For a multi-node
job, run one `torchrun` per node ([Clusters and multi-node](clusters.md)).

## `halo run`

```bash
halo run <tool> [launcher flags] [tool flags]
```

`halo run` takes `--list`, `--dry_run` and `--root`, and runs the tool as one
process from your current directory: every tool is single-process. Everything
else goes to the tool, a Gradio app's own `--port` included.

When a tool flag has the same name as a launcher flag (`--help`, `--dry_run`,
`--root`, `--list`), put `--` before it: `halo run <tool> -- --help` prints the
tool's help, not the launcher's.

`halo run --list` prints the catalog:

| Tool | Purpose |
| --- | --- |
| `merge-ep-shards` / `merge-pp-stages` | merge a per-rank EP save or the per-node directories of a pipeline-parallel save into one loadable checkpoint; see [Checkpoints](checkpoints.md) |
| `merge-peft-adapters` / `merge-models` | fold an adapter into its base, or merge several checkpoints in weight space |
| `convert-to-bf16` / `quantize-to-lowp` / `unfuse-moe-experts` / `reset-sinks` / `reattach-vision-tower` | checkpoint post-processing: bf16 export, low-precision export, per-expert layout, GPT-OSS sink reset, the vision tower back on a text-only export |
| `patch-vocab` | add tokens to a model's vocabulary before training |
| `convert-glm5-bf16` / `convert-mistral4-bf16` / `convert-deepseek-v4-bf16` | convert that family's low-precision Hub release to bf16, required before training it |
| `prepare-dataset` / `s3` | tokenize SFT or pretraining data offline, optionally packed and sharded; move datasets to and from S3 |
| `compact-code-tests` / `prepare-code-dataset` | build a code-contests pool: cap the test corpus, then build prompts, pack tests and checker, and publish rating bands |
| `deduplicate-dataset` | deduplicate generated or collected data |
| `openai-batched-generation` | batched generation against a vLLM or OpenAI-compatible endpoint |
| `inference/reward-model/score` | generate one response per prompt against an endpoint and score it with a reward model |
| `inference/reward-model/rejection-sample` | generate N responses per prompt, score them, and keep the best and worst as preference pairs or offline GRPO data |
| `eval-env` / `eval-code-contests` / `regrade-trajectories` | evaluate a model on an RL environment through an OpenAI-compatible endpoint; re-grade recorded code-contest trajectories |
| `gradio-openai-chatbot` / `gradio-environment-playground` | chat with a served model, or play an environment against it, in the browser |
| `nvlink-health` / `py-spy-diag` / `trace-report` | preflight and debugging; see [Troubleshooting](troubleshooting.md) |
| `weight-sync-transport` | check which transport the trainer-to-rollout-server weight sync uses |
| `check-links` | check every link, anchor and code reference in the docs |

```bash
halo run merge-ep-shards --input_dir <ep-checkpoint-dir> --output_dir <merged-dir>
halo run quantize-to-lowp --input_dir <bf16-model-dir> --output_dir <nvfp4-out-dir> --format nvfp4
```

[Writing a config](configuration.md) covers the config file. Every script and
flag is in the [Scripts reference](../agent-docs/reference/scripts-reference.md) ↗.
