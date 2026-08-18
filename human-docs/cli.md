# The `halo` CLI

The image ships a `halo` command with two verbs: `halo launch` starts a training
method, `halo run` starts every other tool (checkpoint surgery, data prep,
inference, diagnostics). Both are thin wrappers that resolve a name to a script
under `scripts/` and exec the right launcher. The scripts still run directly
with `python`/`torchrun`; the CLI just saves the typing.

## `halo launch`

```bash
halo launch <method> <config.yaml> [flags] [-- trainer flags]
```

| Flag | Short | Meaning |
| --- | --- | --- |
| `--nproc N` | `-n` | processes (GPUs); `N > 1` switches to `torchrun` |
| `--accelerate <yaml>` | `-a` | use `accelerate launch` with this config (standard FSDP) |
| `--port P` | `-p` | rendezvous port — set it when running two jobs on one host |
| `--list` | | print every indexed method and its script |
| `--dry-run` | | print the exact command instead of running it |

Launcher selection: plain `python` for a single process, `torchrun` when
`--nproc` is above 1 (required for EP/CP/TP), `accelerate launch` when
`--accelerate` is given. Anything after `--` goes to the trainer untouched, so
config overrides ride along:

```bash
halo launch sft examples/sft/qwen3/qwen3-4b-ultrachat-lora.yaml
halo launch sft examples/sft/qwen3/qwen3-4b-ultrachat.yaml -n 8
halo launch sft examples/sft/gptoss/gptoss-20b-multinode-ep.yaml -n 8 -- \
    --expert_parallel_size=8 --learning_rate=1e-5
```

A method name is the script's file stem with underscores as hyphens (`sft`,
`smpo`, `dpo`, `rlvr` for online GRPO, `offline-grpo`, `environmental-grpo`, …),
or its path under `scripts/training/` (`preference/smpo`, `online-grpo/rlvr`,
`distillation/self-distill`) if a stem is ever ambiguous. Config paths may be
absolute or relative to the repo root or your current directory; the CLI
absolutizes before it changes directory, and rejects a path found in neither
place up front.

The CLI is single-node. Multi-node jobs call `torchrun` directly, one per node —
see [Clusters](clusters.md).

## `halo run`

```bash
halo run <tool> [flags] [-- tool flags]
```

Same flags minus `--accelerate`. Every tool answers `-- --help`, and
`halo run --list` prints the full catalog. The ones you'll actually reach for:

| Tool | Purpose |
| --- | --- |
| `merge-ep-shards` | merge a sharded EP save into one checkpoint |
| `merge-peft-adapters` | fold a LoRA adapter into its base model |
| `merge-models` | weight-space merge (linear, SLERP, task-arithmetic, TIES) |
| `convert-to-bf16` / `quantize-to-lowp` | cast to bf16 / quantize to mxfp8 or nvfp4 |
| `unfuse-moe-experts` | rewrite fused MoE experts to the per-expert hub layout (refuses a family with no per-expert hub form: GptOss, Inkling, Gemma4, Mistral4, Zaya, Step-3.7 Flash) |
| `reset-sinks` | disable the attention-sink mechanism in a GPT-OSS checkpoint |
| `convert-glm5-bf16` | dequantize the fp8 GLM-5.3-Flash release to bf16 — required before training that family |
| `prepare-dataset` | tokenize, pack, and shard a corpus offline |
| `dataset-deduplication` | deduplicate generated or collected data |
| `openai-batched-generation` | batched generation against a vLLM / OpenAI endpoint |
| `rm-scoring` / `rm-rejection-sampling` | score completions / best-of-N with a reward model |
| `run-env` | evaluate an RL environment offline |
| `nvlink-health` / `py-spy-diag` / `trace-report` | preflight and debugging (see [Troubleshooting](troubleshooting.md)) |

```bash
halo run merge-ep-shards -- --input_dir /mnt/ckpt --output_dir /mnt/merged
halo run quantize-to-lowp -- --input_dir /mnt/bf16 --output_dir /mnt/fp4 --format nvfp4
```

The complete script catalog with every flag is in the reference:
[Scripts Reference](../agent-docs/reference/scripts-reference.md) ↗.
