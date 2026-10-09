# Mistral 4 MoE Cookbook

[Mistral Small 4 119B](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603) has 128 routed experts
plus a shared expert and picks four per token. It uses MLA attention and ships with a Pixtral vision
encoder; the recipe trains on text and keeps the multimodal wrapper intact. The public checkpoint stores
its language-model weights in FP8, so convert it to BF16 once before training.

## Support

| FSDP | EP | CP | TP | ETP | EP+CP | EP+TP | LoRA | Online RL |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Yes | Yes | Yes | Yes | Yes | partial | Yes | Yes | No |

`partial`: EP+CP is a valid shape, but its only GPU test on this family is a tiny-model LoRA run. Try a
short run before a long one.

- **Checkpoint:** `mistralai/Mistral-Small-4-119B-2603`, converted to BF16.
- **GPUs:** eight at EP8, 16 experts per GPU, at 32,000 tokens. Plan for about 500 GB of disk for the FP8
  download and the BF16 copy together.

## Recipes

Run these in the [training container](README.md#start-the-training-container).

### Convert the checkpoint

```bash
halo run convert-mistral4-bf16 --model_id mistralai/Mistral-Small-4-119B-2603 \
  --output_dir /data/models/mistral-small-4-119b-bf16
```

The converter streams one shard at a time and writes a standard BF16 Hugging Face checkpoint.

### Full fine-tune

```bash
halo launch sft examples/sft/mistral4/mistral-small-4-119b-ultrachat-ep.yaml -n 8 \
  --model_name_or_path=/data/models/mistral-small-4-119b-bf16 \
  --output_dir=/data/checkpoints/mistral-small-4-sft
```

### Other layouts

Add one of these to the full fine-tune command:

- EP8 + CP2, for long sequences (`partial`, see above): `--context_parallel_size=2 --packing=false`
- EP8 + TP2, when attention memory is the limit: `--tensor_parallel_size=2`. TP shards the MLA
  expansion and output projections and keeps the compression projections whole.
- Pure ETP8, to shard every expert instead of placing whole experts: `--expert_parallel_size=1 --expert_tensor_parallel_size=8`
- EP2 + ETP4, experimental: `--expert_parallel_size=2 --expert_tensor_parallel_size=4`

### LoRA

```bash
halo launch sft examples/sft/mistral4/mistral-small-4-119b-ultrachat-ep.yaml -n 8 \
  --model_name_or_path=/data/models/mistral-small-4-119b-bf16 \
  --use_peft=true --learning_rate=1e-4 \
  --lora_target_modules=q_a_proj,q_b_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj \
  --output_dir=/data/checkpoints/mistral-small-4-lora
```

### GRPO

Online GRPO is not available for this family (see [Limits](#limits)). Offline GRPO trains on
pre-generated, scored completions and needs no server: see [Offline GRPO](../training-methods/offline-grpo.md).

## Settings that matter

- **Router balancing.** The recipe sets `moe_balancing: bias_update_transient`. The router has no bias
  slot, so plain `bias_update` raises, and it has no auxiliary loss either. The transient bias balances
  training only: exported checkpoints serve without it, so near-tied expert picks can differ between
  training and serving.
- **Chat template.** The recipe forces `jinja-templates/mistral4/mistral4-multiturn.jinja`, because
  `mistral4-instruct.jinja` accepts only single-turn rows and UltraChat is multi-turn. It marks
  completions with `assistant_message_template: "[/INST]"`.
- **Attention.** The recipe pins `flash_attention_2`.

## Limits

- No online RL. Neither vLLM 0.26.0 nor SGLang 0.5.17 has a `mistral4` model class, so Halo refuses
  online and async GRPO when it builds the trainer.
- Neither pinned engine serves a Halo export. vLLM serves the public repo only through its
  Mistral-native `params.json` layout, which a Halo export does not have. vLLM's generic transformers
  backend (`--model-impl transformers`) is untested here.
- LoRA is refused under TP, and attention TP can't combine with ETP.

## Export and serve

The gathered save is a standard Hugging Face checkpoint with the vision tower intact. Run inference
with transformers: load it with `AutoModelForImageTextToText` and `AutoProcessor`, which take images
too ([snippet](README.md#smoke-test-a-checkpoint)). To serve the base model, point vLLM at the public
hub repo.

## Reference

- [Mistral 4 model notes](../../agent-docs/models/mistral4.md) ↗: routing, the CP and TP plans for MLA,
  the serving gap
- [Offline GRPO](../training-methods/offline-grpo.md)
- [Model card](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603)
