# Quickstart

This page takes you from a running container to a finished run. If you don't
have a container yet, start with [Installation](installation.md).

## 1. Pick a recipe

`examples/` holds a config per method and model family. These make good first
runs. Check the hardware column before you launch.

| Recipe | Config | Hardware |
| --- | --- | --- |
| Qwen3 LoRA SFT | `examples/sft/qwen3/qwen3-4b-ultrachat-lora.yaml` | 1 GPU |
| Qwen3 QLoRA SFT | `examples/sft/qwen3/qwen3-4b-ultrachat-qlora.yaml` | 1 GPU (7.9 GiB peak; fits 24 GB) |
| Qwen3 full SFT | `examples/sft/qwen3/qwen3-4b-ultrachat.yaml` | 1–8 GPUs |
| GPT-OSS EP SFT | `examples/sft/gptoss/gptoss-20b-multinode-ep.yaml` | 2 × 8 GPUs as written (MoE, expert parallel) |
| SMPO | `examples/preference/qwen3_5/smpo-qwen3.5-9b-tulu3-prefmix.yaml` | 8 GPUs |
| DPO | `examples/preference/qwen3_5/dpo-qwen3.5-9b-tulu3-prefmix.yaml` | 8 GPUs |
| Offline GRPO | `examples/grpo/offline/qwen3_5/offline-grpo-qwen3.6-35b-a3b-gsm8k.yaml` | 8 GPUs |
| Online GRPO (RLVR) | `examples/grpo/online/qwen3/online-grpo-qwen3-4b-smoke.yaml` | trainer + vLLM server |
| Async GRPO with environments | `examples/grpo/environmental/environmental-grpo-template.yaml` | trainer + vLLM + Ray |

The offline GRPO and async GRPO configs ship with a placeholder `dataset`. Point
it at your data first.

[Choosing a method](choosing-a-method.md) matches your data to a trainer. Each
[training method](training-methods/README.md) page lists its data format and
keys.

The online RL recipes also need a separate [rollout server](rollout-servers.md).
Start with SFT and set that up later.

## 2. Launch it

Inside the container:

```bash
# 1 GPU (LoRA). Without -n the run uses one GPU, however many the container sees.
halo launch sft examples/sft/qwen3/qwen3-4b-ultrachat-lora.yaml

# 8 GPUs, full fine-tune
halo launch sft examples/sft/qwen3/qwen3-4b-ultrachat.yaml -n 8

# 8 GPUs, MoE with expert parallelism
halo launch sft examples/sft/qwen3_5/qwen3.5-35b-a3b-ultrachat-ep.yaml -n 8
```

To change a field without editing the file, add it after the config path:
`--learning_rate=1e-5 --max_length=32000`. A few container fields, such as
`dataset` and `rewards`, can only be set in the YAML. `halo launch --list` prints
every method. See [The `halo` CLI](cli.md) and
[Writing a config](configuration.md).

For a long run, detach the container. In the `docker run` command, replace `-it`
with `-d --name myrun` and pass `bash -lc "halo launch ..."` as the command.

## 3. Watch it

The run copies rank 0's console output to `<output_dir>/log/run.log`. For the
full fine-tune above:

```bash
tail -f checkpoints/sft-qwen3-4b-ultrachat/log/run.log
```

Most examples set `report_to: wandb`. With `WANDB_API_KEY` in your `.env`, the
run shows up in Weights & Biases with loss and learning rate; throughput and MoE
metrics are opt-in. See [Monitoring](monitoring.md).

## 4. Use the result

Checkpoints land in `output_dir` as standard HuggingFace models. Load them with
`from_pretrained` or upload them to the Hub as they are. Most families also serve
on vLLM directly; a few need a conversion first or have no serving path.

Two kinds of run need one more step:

- A LoRA run saves an adapter. Serve it as an adapter or merge it into the base.
- The opt-in sharded EP save (`save_sharded_ep: true`) needs
  `halo run merge-ep-shards` first.

[Checkpoints](checkpoints.md) covers saving, resuming, export and serving.
