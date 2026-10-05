# The `halo` CLI

The image ships a `halo` command with two verbs: `halo launch` starts a training
method, `halo run` starts every other tool (checkpoint surgery, data prep,
inference, diagnostics). Both are thin wrappers that resolve a name to a script
under `scripts/` and exec the right launcher. The scripts also run directly
with `python`/`torchrun`; the CLI saves the typing.

## `halo launch`

```bash
halo launch <method> <config.yaml> [launcher flags] [field overrides]
```

| Flag | Short | Meaning |
| --- | --- | --- |
| `--nproc N` | `-n` | processes (GPUs); `N > 1` switches to `torchrun` |
| `--accelerate <yaml>` | `-a` | use `accelerate launch` with this config (standard FSDP) |
| `--port P` | `-p` | rendezvous port for a multi-process launch — set it when two jobs share a host |
| `--list` | | print every indexed method and its script |
| `--dry-run` | | print the exact command instead of running it |

Launcher selection: plain `python` for a single process, `torchrun` when
`--nproc` is above 1 (required for EP/CP/TP), `accelerate launch` when
`--accelerate` is given. Any other flag after the config goes to the trainer
untouched, so config overrides ride along:

```bash
halo launch sft examples/sft/qwen3/qwen3-4b-ultrachat-lora.yaml
halo launch sft examples/sft/qwen3/qwen3-4b-ultrachat.yaml -n 8
halo launch sft examples/sft/gptoss/gptoss-20b-multinode-ep.yaml -n 8 \
    --expert_parallel_size=8 --learning_rate=1e-5
```

A method name is the script's file stem with underscores as hyphens (`sft`,
`smpo`, `dpo`, `rlvr` for online GRPO, `offline-grpo`, `environmental-grpo` for
async GRPO with environments, …),
or its path under `scripts/training/` (`preference/smpo`, `online-grpo/rlvr`,
`distillation/self-distill`) if a stem is ever ambiguous. Config paths may be
absolute or relative to the repo root or your current directory; the CLI
absolutizes before it changes directory, and rejects a path found in neither
place up front.

The CLI is single-node. Multi-node jobs call `torchrun` directly, one per node —
see [Clusters](clusters.md).

## `halo run`

```bash
halo run <tool> [launcher flags] [tool flags]
```

Same flags minus `--accelerate`; everything else goes to the tool. A standalone `--` is only
needed before a flag the launcher owns itself (`--help`, `--dry-run`, `--port`, `--root`, `--list`,
`-n`, `-p`) when you mean the tool's: `halo run <tool> -- --help` prints the tool's help, not the
launcher's. `halo run --list` prints the full catalog. The ones you'll actually reach for:

| Tool | Purpose |
| --- | --- |
| `merge-ep-shards` / `merge-peft-adapters` / `merge-models` / `convert-to-bf16` / `quantize-to-lowp` / `unfuse-moe-experts` / `reset-sinks` | checkpoint post-processing — see [Checkpoints](checkpoints.md#post-processing-tools) |
| `convert-glm5-bf16` / `convert-mistral4-bf16` / `convert-deepseek-v4-bf16` | dequantize that family's low-precision hub release to bf16 — required before training it |
| `prepare-dataset` | tokenize, pack, and shard a corpus offline |
| `compact-code-tests` / `prepare-code-dataset` | build a code-contests pool: cap the test corpus, then compose prompts, pack tests and checker, and publish rating bands |
| `dataset-deduplication` | deduplicate generated or collected data |
| `openai-batched-generation` | batched generation against a vLLM / OpenAI endpoint |
| `rm-scoring` / `rm-rejection-sampling` | score completions / best-of-N with a reward model |
| `run-env` | evaluate an RL environment offline |
| `nvlink-health` / `py-spy-diag` / `trace-report` | preflight and debugging (see [Troubleshooting](troubleshooting.md)) |
| `weight-sync-transport` | check which transport a trainer↔rollout-server weight sync formed on |

```bash
halo run merge-ep-shards --input_dir <ep-checkpoint-dir> --output_dir <merged-dir>
halo run quantize-to-lowp --input_dir <bf16-model-dir> --output_dir <nvfp4-out-dir> --format nvfp4
```

What goes in the config file the launch names is
[Writing a Config](configuration.md). The complete script catalog with every
flag: [Scripts Reference](../agent-docs/reference/scripts-reference.md) ↗.
