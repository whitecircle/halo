# Halo User Guide

Halo is a distributed training toolkit for large language models, built on the
HuggingFace stack (Transformers, TRL, Accelerate). It adds expert, context,
tensor and expert-tensor parallelism to native HuggingFace models, and covers
supervised fine-tuning through fully asynchronous multi-turn RL, on dense models
and fifteen MoE families. A HuggingFace model goes in, a HuggingFace model comes
out, loadable with plain `from_pretrained`.

Read this guide top to bottom for a first run, or jump to the section you need.
Links marked ↗ leave the guide for [`agent-docs/`](../agent-docs/README.md), the
exhaustive reference — written for the AI agents that maintain the repo, and
readable by anyone who wants the mechanism.

## Get started

- [Installation](installation.md) — pull or build the image, start a container,
  set up `.env` and the caches.
- [Quickstart](quickstart.md) — launch an example config and watch it train.
- [Choosing a Method](choosing-a-method.md) — map the data you have to the
  trainer you want.

## Train

- [Training Methods](training-methods/README.md) — the index, with what each
  method costs to run.
- [SFT](training-methods/sft.md) — supervised fine-tuning, VLM and continued
  pretraining included.
- [Preference](training-methods/preference.md) — DPO, SMPO and KTO.
- [Reward and Classification](training-methods/reward-and-classification.md) —
  Bradley-Terry scorers and sequence-level labels.
- [Distillation](training-methods/distillation.md) — teacher, self and online
  SDPG.
- [Embedding](training-methods/embedding.md) — retrieval and similarity tuning
  on SentenceTransformers losses.
- [Offline GRPO](training-methods/offline-grpo.md) — RL from completions you
  already scored.
- [Online GRPO](training-methods/online-grpo.md) — RLVR: the model generates,
  verifiable rewards score it.
- [Async GRPO with Environments](training-methods/async-grpo-environments.md) —
  multi-turn tool-using RL with Ray actors and overlapped rollouts.

## Configure

- [Writing a Config](configuration.md) — the blocks, the keys that matter, CLI
  overrides, and what the parser refuses.
- [Datasets](data.md) — the columns each method reads, mixing sources, and
  offline tokenize/pack/shard.
- [Environment Variables](environment-variables.md) — the handful you set; the
  image already sets the tricky ones.
- [The halo CLI](cli.md) — `halo launch` for training, `halo run` for every
  other tool.

## Serve rollouts

- [Rollout Servers](rollout-servers.md) — the vLLM or SGLang container the RL
  methods generate against, and the NCCL weight sync.

## Scale

- [Parallelism](parallelism.md) — EP/CP/TP/ETP and the layout rules that reject
  a bad shape before it costs a run.
- [Clusters and Multi-Node](clusters.md) — one torchrun per node, fabric,
  storage, SkyPilot, RunPod, Nomad.
- [Performance](performance.md) — the throughput to expect, the levers that
  move it, and the ones that don't.
- [Supported Matrix](supported-matrix.md) — model family × mode, plus runtime
  versions and the standing limits.

## Operate

- [Checkpoints](checkpoints.md) — save, resume, merge, quantize, serve, upload.
- [Monitoring](monitoring.md) — the log file, W&B / ClearML, and the metrics
  worth turning on.
- [Troubleshooting](troubleshooting.md) — OOM, rejected configs, hangs, and the
  RL-specific failures.

## Models

- [Supported Models](models.md) — every family at a glance, and what to know
  about each.
- [Model Cookbooks](cookbooks/README.md) — end-to-end recipes per family, from
  `docker pull` to a trained checkpoint.
- [Model Integration Cost](model-integration-cost.md) — what supporting a new
  family actually takes.

## Contribute

- [Contributing](contributing.md) — the issue-first, approval-gated process and
  the bar a PR clears.
- [AI Tooling](ai-tooling.md) — the repo-aware agent skills the images ship
  with.

For the reasoning under the machinery — arithmetic intensity, where a step's
time goes, why the defaults are the defaults — read
[GPU Training Theory](../agent-docs/reference/gpu-training-theory.md) ↗.
