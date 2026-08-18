# Halo User Guide

Halo is White Circle's distributed-training toolkit for LLMs. It extends the
HuggingFace stack (Transformers, TRL, Accelerate) with Expert, Context,
Tensor, and Expert-Tensor Parallelism, and covers pre-training through fully
asynchronous multi-turn RL on dense models and fifteen MoE families — rollouts
run on a separate vLLM or SGLang server with NCCL weight sync, no Megatron or
veRL backend. A HuggingFace model goes in; a HuggingFace model comes out,
loadable with plain `from_pretrained`.

These pages are short and task-first. Links marked ↗ leave this guide for
[`agent-docs/`](../agent-docs/README.md), the exhaustive reference — written
for the AI agents that maintain the repo, and readable by anyone.

## Reading order for a first run

1. [Installation](installation.md) — pull or build the Docker image, start a
   container, set up `.env` and caches.
2. [Quickstart](quickstart.md) — launch a training run from an example
   config and watch it.
3. [Choosing a Method](choosing-a-method.md) — pick the right trainer for
   your data and goal.

## Running a training job

Training is YAML-driven: one config file, plus any field overridden on the
command line.

- [Model Cookbooks](cookbooks/README.md) — end-to-end recipes per model family,
  from `docker pull` to a trained checkpoint.
- [The halo CLI](cli.md) — `halo launch` for the training methods, `halo run`
  for every other tool.
- [Configuration](configuration.md) — the blocks a config has, how the model
  is loaded, and the defaults the parser applies.
- [Environment Variables](environment-variables.md) — the handful you set;
  the image already sets the tricky ones.

## Data in, checkpoints out

- [Datasets](data.md) — the columns each method reads, mixing sources, and
  offline tokenize/pack/shard for large corpora.
- [Checkpoints & Export](checkpoints.md) — resume, merge shards and adapters,
  quantize, serve with vLLM, upload to the Hub.

## Scaling across GPUs and nodes

- [Parallelism](parallelism.md) — EP/CP/TP/ETP, and the layout rules that
  reject a bad shape before it costs a run.
- [Clusters & Multi-Node](clusters.md) — one torchrun per node, network
  fabric, shared vs local storage, SkyPilot and RunPod.
- [Supported Matrix](supported-matrix.md) — model family × mode, plus
  hardware, runtime versions, and the standing limits.

## Operating a run

- [Monitoring](monitoring.md) — the log file, W&B / ClearML, and the metrics
  and traces worth turning on.
- [Troubleshooting](troubleshooting.md) — OOM, rejected configs, hangs, and
  the RL-specific failures.
- [AI Tooling](ai-tooling.md) — repo-aware agent skills (optional Claude Code install)
  the training images ship with.

## Extending and contributing

- [Model Integration Cost](model-integration-cost.md) — what supporting a new
  model family actually takes.
- [Contributing](contributing.md) — the issue-first, approval-gated process
  and the bar a PR has to clear.

Background reading, if you want the reasoning under the machinery:
[GPU Training Theory](../agent-docs/reference/gpu-training-theory.md) ↗.
