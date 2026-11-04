# Nemotron 3.5 Super VL Cookbook

NVIDIA's [Nemotron 3.5 Super VL](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16)
interleaves 40 Mamba-2, 40 MoE and 8 attention layers, 120B parameters in all, with a C-RADIO vision tower.
Each MoE layer routes 512 experts at top-22 on a 1024-wide latent.

## Support

Modes: the Nemotron-H row of the [Supported Matrix](../supported-matrix.md#model-families); online RL:
[Rollout engines](../supported-matrix.md#rollout-engines).

- **Checkpoint:** `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`, loaded through transformers' own
  classes with no conversion step.
- **GPUs:** EP8 on eight B300s, 64 experts per GPU.
- **Lightning:** `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` is the same text class with no vision tower. Its
  recipe, `examples/sft/nemotron_h/nemotron-3.5-lightning-ep8-full.yaml`, ran 5 steps at EP8 and 8,192 tokens at 9,986
  tok/s per GPU and 55–60 GB reserved. A full-depth export serves as saved
  ([Lightning](../../agent-docs/models/nemotron-h.md#lightning) ↗). The rest of this page is Super VL.
- **What ran.** All four recipes and the image recipe ran on the release on eight B300s, 5 to 20 steps each, plus an
  exact resume at a 9-layer cut and exports byte for byte at full depth. No run has trained to a quality metric. Both
  engines have served an untrained reattached export, on one image prompt and one text prompt each.

| Recipe | Length | tok/s per GPU | Reserved by GPU |
| --- | ---: | ---: | ---: |
| Full fine-tune | 8,192 | 3,057 | 186–202 GB |
| Full fine-tune | 32,000 | 3,221 | 268–283 GB |
| Full, experts frozen | 50,000 | 3,246 | 241–283 GB |
| LoRA | 16,384 | 3,000 | 118–165 GB |
| SMPO, a full pair | 2 × 4,096 | — | 178–184 GB |
| SMPO, the recipe's Tulu rows | — | — | 165–170 GB |
| Offline GRPO, the recipe's rows | — | — | 169–179 GB |
| Images | about 6,790 | 3,198 | 184–188 GB |
| Half image rows, half text | — | 3,166 | 183–186 GB |

A B300 reports 288 GB, and the top GPU reached 99.4% at 32,000 and 99% at 50,000, so treat those as the limit. Each
`max_length` is a starting point: watch the first steps' peak.

## Recipes

Run these in the [training container](README.md#start-the-training-container). All four load the text tower
with `text_only_model: true`.

### Full fine-tune

```bash
halo launch sft examples/sft/nemotron_h/nemotron-3.5-super-vl-ep8-full.yaml -n 8 \
  --output_dir=/data/checkpoints/nemotron-sft
```

`--max_length=32000` also ran. Freezing the experts with `freeze_layers_patterns` on `*.mixer.up_proj` and
`*.mixer.down_proj` reached `--max_length=50000`. Each save writes about 241 GB, and a full resume checkpoint comes to
about 725 GB by arithmetic.

### LoRA

```bash
halo launch sft examples/sft/nemotron_h/nemotron-3.5-super-vl-ep8-lora.yaml -n 8 \
  --output_dir=/data/checkpoints/nemotron-lora
```

These targets adapt attention, the Mamba projections and the experts. Every save folds the adapters into a full
checkpoint, 251 GB with its adapter state, so raise `save_steps` when disk is tight.

### SMPO

```bash
halo launch smpo examples/preference/nemotron_h/smpo-nemotron-3.5-super-vl-tulu3-prefmix-ep8.yaml -n 8 \
  --output_dir=/data/checkpoints/nemotron-smpo
```

### Offline GRPO

Copy the recipe and point `dataset` at your scored completions
([Offline GRPO](../training-methods/offline-grpo.md)):

```bash
cp examples/grpo/offline/nemotron_h/offline-grpo-nemotron-3.5-super-vl-gsm8k-ep8.yaml nemotron-grpo.yaml
halo launch grpo/offline nemotron-grpo.yaml -n 8 --output_dir=/data/checkpoints/nemotron-grpo
```

### Images

Copy the full fine-tune recipe, delete `text_only_model` and `packing`, and point `dataset` at rows with images
([SFT](../training-methods/sft.md#data)). The composite computes its loss on the plain head, which holds more
per token.

Image and text rows can mix under `torchrun`: half-image rows ran at EP8 on the release with no hang. Training the
vision tower on mixed rows rests on tiny-model tests only.

With Liger on, a training step logs `mean_token_accuracy` 0 and `entropy` ln(vocab), because Liger writes its gradient
into the fp32 logits the forward returns. The loss, the gradients and eval metrics that mask label -100 are right; a
`compute_metrics` that reads logits at ignored positions sees zeros in eval too.

## Settings that matter

- **Gradient checkpointing stays on.** By arithmetic a token holds about 66 MB without it, so a GPU fits about
  1,650 tokens per rank. It costs 37% on a Mamba layer.
- **Speed.** The Mamba scan dominates, at 64.8% of the tower's forward and backward at EP8, and throughput stays flat
  with length. Rows of many short documents scan slower: 64 × 128 costs 1.9× one 8,192-token document on a Mamba layer.
- **Attention.** Leave `attn_implementation` unset: FA4 runs both towers on B300, with FA2 as the fallback.
- **Pad token.** `pad_token: <unk>` keeps the export's config equal to the release's. The tokenizer's own pad is
  its eos.
- **Chat template.** The hub template renders an assistant turn with no reasoning as `<think></think>` plus its
  text, so the empty block trains as part of the completion.
- **Router balancing.** `auto` picks `bias_update`, which trains the router's own bias and ships it in every
  checkpoint. The bias stays fp32 and routes with balancing off as well.
- **LoRA.** `lora_r` must be a multiple of 8. An adapter on `conv1d` is refused.

## Limits

- No CP, TP or HSDP: the Mamba scan runs along the sequence axis, and attention is 0.24% of the parameters.
- No tree packing.
- No online RL. Both engines serve the release, but no update into either is validated, so Halo refuses weight
  sync at startup. Offline GRPO needs no server: score completions from a served export and train them with the
  recipe above.
- DPO and KTO take text rows, a full fine-tune with `precompute_ref_log_probs: true`, and refuse images.
- No video or audio rows. They raise.
- `trust_remote_code: true` on the composite is refused at the load: the hub's modeling file hands its text tower
  no document markers, so leave it unset. `text_only_model: true` builds transformers' class either way.
- Pipeline parallelism takes the text tower only and passes tiny-model GPU tests only.
- `save_sharded_ep` and `merge-ep-shards` are refused. Gathered saves stream one layer at a time.
- `merge-peft-adapters` refuses expert adapters, which fold through `merge_expert_lora_on_save`.

## Export and serve

Smoke-test a text-only export with `AutoModelForCausalLM` ([snippet](README.md#smoke-test-a-checkpoint)). Stock
transformers clamps Mamba's time steps at a floor neither the run nor the engines use, so it scores 0.05–0.09 nats
worse on real text than the run did. To serve the export, rebuild the composite, which also streams back the
vision tower and the `mtp.*` head at the release's weights:

```bash
halo run reattach-vision-tower --input_dir /data/checkpoints/nemotron-sft \
  --model_id nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16 \
  --output_dir /data/checkpoints/nemotron-sft-vl
```

Serve from the [host](README.md#serve-from-the-host). vLLM started this way needs `--trust-remote-code` and
`--mamba-ssm-cache-dtype float32`, which `docker-compose.vllm.yml` takes as `VLLM_TRUST_REMOTE_CODE=1` and
`VLLM_EXTRA_ARGS`:

```bash
docker run -d --rm --name nemotron-vllm --gpus '"device=0,1"' --ipc=host --network host \
  -v "$HF_HOME":/root/.cache/huggingface -v "$HALO_SCRATCH":/data:ro "$VLLM_IMAGE" \
  /data/checkpoints/nemotron-sft-vl --port 8000 --trust-remote-code --tensor-parallel-size 2 \
  --mamba-ssm-cache-dtype float32 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser nemotron_v3
```

SGLang's compose file on port 30000 takes:

```bash
SGLANG_MODEL="$HALO_SCRATCH/checkpoints/nemotron-sft-vl" SGLANG_CUDA_DEVICES=0,1 SGLANG_TP=2 \
SGLANG_TRUST_REMOTE_CODE=1 SGLANG_TOOL_PARSER=qwen3_coder SGLANG_REASONING_PARSER=nemotron_3 \
  docker compose -f docker-compose.sglang.yml up -d sglang-server
```

Don't copy the model card's `--mamba-ssm-dtype` flags onto this image: they are for the card's dedicated image.

## Reference

- [Nemotron-H model notes](../../agent-docs/models/nemotron-h.md) ↗: the Mamba scan, the memory model, the
  export layouts
- [Offline GRPO](../training-methods/offline-grpo.md), [Preference](../training-methods/preference.md)
- [Model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16)
