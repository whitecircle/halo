# Halo user guide

Halo is a distributed training toolkit for large language models, built on the
HuggingFace stack (Transformers, TRL, Accelerate). It adds expert, context,
tensor and expert-tensor parallelism to native HuggingFace models.

It covers supervised fine-tuning through asynchronous multi-turn RL, on dense
models and fifteen MoE families. A HuggingFace model goes in, and a HuggingFace
model you can load with `from_pretrained` comes out.

Read the guide in order for a first run, or jump to the section you need. Links
marked ↗ go to [`agent-docs/`](../agent-docs/README.md), the detailed reference
written for the AI agents that maintain the repo. Read it when you want the
mechanism.

## Get started

- [Installation](installation.md): pull or build the image, start a container,
  set up `.env` and the caches.
- [Quickstart](quickstart.md): launch an example config and watch it train.
- [Choosing a method](choosing-a-method.md): match the data you have to a
  trainer.

## Train

- [Training methods](training-methods/README.md): the index, with what each
  method costs to run.
- [SFT](training-methods/sft.md): supervised fine-tuning, including VLMs and
  continued pretraining.
- [Preference](training-methods/preference.md): DPO, SMPO and KTO.
- [Reward and classification](training-methods/reward-and-classification.md):
  Bradley-Terry scorers and sequence-level labels.
- [Distillation](training-methods/distillation.md): teacher, self and online
  SDPG.
- [Embedding](training-methods/embedding.md): retrieval and similarity tuning
  with SentenceTransformers losses.
- [Offline GRPO](training-methods/offline-grpo.md): RL from completions you
  already scored.
- [Online GRPO](training-methods/online-grpo.md): RLVR, where the model
  generates and verifiable rewards score it.
- [Async GRPO with Environments](training-methods/async-grpo-environments.md):
  multi-turn, tool-using RL with Ray actors and overlapped rollouts.

## Configure

- [Writing a config](configuration.md): the blocks, the keys that matter, CLI
  overrides, and what the parser refuses.
- [Datasets](data.md): the columns each method reads, mixing sources, and
  offline tokenize, pack and shard.
- [Environment variables](environment-variables.md): the few you set. The image
  already sets the tricky ones.
- [The `halo` CLI](cli.md): `halo launch` for training, `halo run` for every
  other tool.

## Serve rollouts

- [Rollout servers](rollout-servers.md): the vLLM or SGLang container the RL
  methods generate with, and the NCCL weight sync.

## Scale

- [Parallelism](parallelism.md): EP, CP, TP and ETP, and the layout rules that
  reject a bad shape before it costs you a run.
- [Clusters and multi-node](clusters.md): one torchrun per node, fabric,
  storage, SkyPilot, RunPod, Nomad.
- [Performance](performance.md): the throughput to expect, the levers that move
  it, and the ones that don't.
- [Supported matrix](supported-matrix.md): model family × mode, runtime
  versions and the standing limits.

## Operate

- [Checkpoints](checkpoints.md): save, resume, merge, quantize, serve, upload.
- [Monitoring](monitoring.md): the log file, W&B or ClearML, and the metrics
  worth turning on.
- [Troubleshooting](troubleshooting.md): OOM, rejected configs, hangs, and
  RL-specific failures.

## Models

- [Supported models](models.md): every family at a glance, and what to know
  about each.
- [Model cookbooks](cookbooks/README.md): end-to-end recipes per family, from
  `docker pull` to a trained checkpoint.
- [Model integration cost](model-integration-cost.md): what it takes to support
  a new family.

## Contribute

- [Contributing](contributing.md): the issue-first, approval-gated process and
  the bar a PR has to clear.
- [AI tooling](ai-tooling.md): the repo-aware agent skills that ship with the
  images.

[GPU Training Theory](../agent-docs/reference/gpu-training-theory.md) ↗ explains
why the defaults are what they are: arithmetic intensity, and where a training
step's time goes.
