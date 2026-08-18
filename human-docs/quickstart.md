# Quickstart

From a running container to a finished run. If you haven't pulled the image and
started a container yet, do [Installation](installation.md) first.

## 1. Pick a recipe

`examples/` holds a runnable config per method and model family. Good starting
points — check the hardware column before launching:

| Recipe | Config | Hardware |
| --- | --- | --- |
| Qwen3 LoRA SFT | `examples/sft/qwen3/qwen3-4b-ultrachat-lora.yaml` | 1 GPU |
| Qwen3 QLoRA SFT | `examples/sft/qwen3/qwen3-4b-ultrachat-qlora.yaml` | 1 GPU (24 GB is enough) |
| Qwen3 full SFT | `examples/sft/qwen3/qwen3-4b-ultrachat.yaml` | 1–8 GPUs |
| GPT-OSS EP SFT | `examples/sft/gptoss/gptoss-20b-multinode-ep.yaml` | 2 × 8 GPUs as written (MoE, expert parallel) |
| SMPO | `examples/preference/qwen3_5/smpo-qwen3.5-9b-tulu3-prefmix.yaml` | 1–8 GPUs |
| DPO | `examples/preference/qwen3_5/dpo-qwen3.5-9b-tulu3-prefmix.yaml` | 1–8 GPUs |
| Offline GRPO | `examples/grpo/offline/qwen3_5/offline-grpo-qwen3.6-35b-a3b-gsm8k.yaml` | 8 GPUs |
| Online GRPO (RLVR) | `examples/grpo/online/qwen3/online-grpo-qwen3-4b-smoke.yaml` | trainer + vLLM server |
| Environmental GRPO | `examples/grpo/environmental/environmental-grpo-template.yaml` | trainer + vLLM + Ray |

Not sure which method you need? [Choosing a Method](choosing-a-method.md) maps
the data you have to the trainer you want. The online RL recipes need a separate
vLLM container; start with SFT and set that up later.

## 2. Launch it

Inside the container:

```bash
# Single GPU (LoRA)
halo launch sft examples/sft/qwen3/qwen3-4b-ultrachat-lora.yaml

# 8 GPUs, full fine-tune
halo launch sft examples/sft/qwen3/qwen3-4b-ultrachat.yaml -n 8

# 8 GPUs, MoE with expert parallelism
halo launch sft examples/sft/qwen3_5/qwen3.5-35b-a3b-ultrachat-ep.yaml -n 8
```

Any config field can be overridden on the command line after `--`
(`--learning_rate=1e-5 --max_length=32000`). `halo launch --list` shows all
methods; the full CLI story is in [The halo CLI](cli.md).

For a long run, detach the container instead of holding a terminal open:
replace `-it` with `-d --name myrun` in the `docker run` command and give it
a `bash -lc "halo launch ..."` to execute.

## 3. Watch it

Everything the run prints is mirrored to `<output_dir>/log/run.log`:

```bash
tail -f checkpoints/sft-qwen3-4b-ultrachat/log/run.log
```

With `report_to: wandb` in the config (most examples have it) and a
`WANDB_API_KEY` in your `.env`, the run appears in Weights & Biases —
loss, learning rate, and optionally throughput and MoE metrics. See
[Monitoring](monitoring.md).

## 4. Use the result

Checkpoints land in `output_dir` in standard HuggingFace form: by default you
can `from_pretrained` them, serve them with vLLM, or upload them to the Hub
as-is. LoRA runs save an adapter instead, and the opt-in sharded save modes need
a one-command merge first. [Checkpoints & Export](checkpoints.md) covers all of
it.
