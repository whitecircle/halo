# Command A+ Cookbook

[Command A+](https://huggingface.co/CohereLabs/command-a-plus-05-2026-bf16) is a 200B+ Cohere2 MoE inside a
vision-language wrapper. It has 128 routed experts and picks eight per token with a sigmoid router. Four
shared experts run on every token, and their output is averaged with the routed output. The recipe
trains on text.

## Support

| FSDP | EP | CP | TP | ETP | EP+CP | EP+TP | LoRA | Online RL |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | No |

Only EP8 has run on the full checkpoint. CP, TP, ETP, EP+CP, EP+TP and LoRA pass the GPU tests on a
tiny model, so try a short run before a long one.

- **Checkpoint:** `CohereLabs/command-a-plus-05-2026-bf16`, pinned to the revision in the recipe.
- **GPUs:** eight B300s at EP8, 16 experts per GPU, at 1,024 tokens. Inference in BF16 needs at least
  four B200s.

## Recipes

Run these in the [training container](README.md#start-the-training-container).

### Full fine-tune

```bash
halo launch sft examples/sft/cohere2_moe/command-a-plus-ultrachat-ep.yaml -n 8 \
  --output_dir=/data/checkpoints/command-a-plus-sft
```

### Other layouts

These have been tested only on a tiny model. Add one of them to the full fine-tune command:

- EP8 + CP2, for long sequences: `--context_parallel_size=2` (the recipe already turns packing off)
- EP8 + TP2, when attention memory is the limit: `--tensor_parallel_size=2`
- Pure ETP8, to shard every expert instead of placing whole experts: `--expert_parallel_size=1 --expert_tensor_parallel_size=8`
- EP2 + ETP4, experimental: `--expert_parallel_size=2 --expert_tensor_parallel_size=4`

### LoRA

```bash
halo launch sft examples/sft/cohere2_moe/command-a-plus-ultrachat-ep.yaml -n 8 \
  --use_peft=true --learning_rate=1e-4 \
  --lora_target_modules=q_proj,k_proj,v_proj,o_proj \
  --output_dir=/data/checkpoints/command-a-plus-lora
```

The GPU tests cover LoRA, expert adapters included, on a tiny model: a merged save and an exact resume.

### GRPO

Online GRPO is not available for this family (see [Limits](#limits)). Offline GRPO trains on
pre-generated, scored completions and needs no server: see [Offline GRPO](../training-methods/offline-grpo.md).

## Settings that matter

- **Sequence length.** `max_length: 1024` peaks at about 255 GiB of the B300's ~268 GiB. That leaves
  DeepEP's buffer room to grow. At 4,096 the allocation fails once a longer batch forces it to grow.
- **Router balancing.** The recipe sets `moe_balancing: bias_update_transient` with
  `router_balancing_rate: 1.0e-3`. The model has no usable auxiliary loss and no exportable bias slot,
  so plain `bias_update` raises. Exported checkpoints serve without the transient bias, so near-tied
  expert picks can differ between training and serving. Drop the line to train unbalanced but serve
  exactly what you trained.
- **Precision.** The experts train in BF16 (`fp32_experts: false`). FP32 masters for 200B of expert
  parameters do not fit.
- **Pinned settings.** The recipe pins `attn_implementation: flash_attention_2` and the model revision
  its memory numbers were measured on. It keeps `packing: false` to stay on that measured shape, though
  text data supports packing.

## Limits

- No online RL. Weight sync for this family has not been validated on either engine, so Halo refuses
  online and async GRPO when it builds the trainer.
- Serving an export is not tested end to end. The steps below follow from the engines' loaders.

## Export and serve

The gathered save uses the hub's tensor names, vision tower included, and keeps the experts as
transformers' fused pair.

- **transformers:** load the save with `AutoModelForImageTextToText` and `AutoTokenizer`, or
  `AutoProcessor` for images ([snippet](README.md#smoke-test-a-checkpoint)).
- **vLLM 0.26.0:** point it at the save. It reads the fused experts. It also requires tied embeddings,
  which training keeps unless it trained the output head on its own.
- **SGLang 0.5.17:** run `halo run unfuse-moe-experts` first. SGLang's Cohere2 loader reads one tensor
  per expert and skips the fused pair without an error. The rewrite gives the hub's exact layout.

## Reference

- [Cohere2 MoE model notes](../../agent-docs/models/cohere2-moe.md) ↗: routing, the CP wrapper, test
  coverage
- [Offline GRPO](../training-methods/offline-grpo.md)
- [Model card](https://huggingface.co/CohereLabs/command-a-plus-05-2026-bf16)
