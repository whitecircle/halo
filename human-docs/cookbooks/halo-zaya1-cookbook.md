# ZAYA1 Cookbook

[ZAYA1-8B](https://huggingface.co/Zyphra/ZAYA1-8B) has 16 routed experts and picks one per token. Its
attention runs a convolution over the sequence (CCA). ZAYA1 is native in transformers, so hub `main`
loads with no revision pin and no remote code. It is the most constrained family in Halo.

## Support

| FSDP | EP | CP | TP | ETP | EP+CP | EP+TP | LoRA | Online RL |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Yes | Yes | No | No | Yes | No | No | Yes | No |

Every mode runs without gradient checkpointing.

- **Checkpoint:** `Zyphra/ZAYA1-8B` at hub `main`.
- **GPUs:** one for plain training, eight at EP8 (two experts per GPU).

## Recipes

Run these in the [training container](README.md#start-the-training-container).

### Full fine-tune

On one GPU:

```bash
halo launch sft examples/sft/zaya/zaya-1-8b-ultrachat.yaml \
  --output_dir=/data/checkpoints/zaya1-8b-sft
```

At EP8 on eight GPUs:

```bash
halo launch sft examples/sft/zaya/zaya-1-8b-ultrachat-ep.yaml -n 8 \
  --output_dir=/data/checkpoints/zaya1-8b-sft-ep8
```

Both recipes train at 4,096 tokens. For long context at EP8, raise the limit with
`--max_length=32768`, which fits at `per_device_train_batch_size: 1` because the fused loss never builds
the full logits tensor.

### Other layouts

- Pure ETP2, to shard every expert across two GPUs: add `-n 2 --expert_tensor_parallel_size=2` to the
  one-GPU command.

### LoRA

```bash
halo launch sft examples/sft/zaya/zaya-1-8b-ultrachat.yaml \
  --use_peft=true --learning_rate=1e-4 \
  --lora_target_modules=q_proj,k_proj,v_proj_current,v_proj_delayed,o_proj \
  --output_dir=/data/checkpoints/zaya1-8b-lora
```

### GRPO

Online GRPO is not available for this family (see [Limits](#limits)). Offline GRPO trains on
pre-generated, scored completions and needs no server: see [Offline GRPO](../training-methods/offline-grpo.md).

## Settings that matter

- **Gradient checkpointing stays off.** Halo refuses it in every mode, so keep
  `gradient_checkpointing: false`. Recomputing CCA's convolutions faults in cuDNN in Halo's CUDA 13.2
  images.
- **Router balancing.** `auto` picks `bias_update`, which updates the router's own `balancing_biases`
  after each step, so the final bias ships in every checkpoint. `router_balancing_rate` sets the step
  size. ZAYA1 has no auxiliary loss, which is why the recipes set `output_router_logits: false`.
- **Packing.** The recipes pack. Attention stays within each packed document, but the CCA convolution
  and the delayed value projection still mix neighboring documents. Add `--packing=false` where that
  mixing is not acceptable.

## Limits

- No gradient checkpointing, in any mode.
- No CP or TP. CCA's sequence convolution breaks a sequence split, and its attention has no TP plan.
- No online RL. vLLM 0.26.0 has no native ZAYA1 class, and SGLang 0.5.17's loader reads only the legacy
  per-expert layout, so Halo refuses weight sync to either engine when it builds the trainer.
- No routing replay.
- The legacy checkpoint `Zyphra/ZAYA1-8B-legacy` doesn't load. Start from hub `main`.

## Export and serve

The gathered save writes the two native fused expert tensors per layer, which `from_pretrained` reads
back. `save_sharded_ep: true` works as well, and `halo run merge-ep-shards` rebuilds the same layout.

Neither pinned engine serves a Halo export. vLLM has only its generic transformers backend, which is
untested on this family, and SGLang's loader reads only the legacy layout. Run inference with
transformers and `AutoModelForCausalLM` ([snippet](README.md#smoke-test-a-checkpoint)).

## Reference

- [Zaya model notes](../../agent-docs/models/zaya.md) ↗: the router state, the GC refusal, packing
- [Offline GRPO](../training-methods/offline-grpo.md)
- [Model card](https://huggingface.co/Zyphra/ZAYA1-8B)
