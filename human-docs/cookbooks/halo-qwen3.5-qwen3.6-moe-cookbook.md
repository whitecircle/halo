# Qwen3.5 / Qwen3.6 MoE Cookbook

[Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) and
[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) share one model family. Each has 256 routed
experts plus a shared expert and picks eight per token. Of its 40 layers, 10 use full attention and 30
use GatedDeltaNet linear attention. The checkpoints are multimodal, and the recipes train their language
model on text.

## Support

Modes: the Qwen3.5 / Qwen3.6 MoE row of the [Supported Matrix](../supported-matrix.md#model-families); online RL:
[Rollout engines](../supported-matrix.md#rollout-engines).

- **Checkpoints:** `Qwen/Qwen3.6-35B-A3B` at revision `995ad96eacd98c81ed38be0c5b274b04031597b0` and
  `Qwen/Qwen3.5-35B-A3B` at `59d61f3ce65a6d9863b86d2e96597125219dc754`. The 122B-A10B trains from
  `examples/sft/qwen3_5/qwen3.5-122b-a10b-ep.yaml` at EP8 on one node. Across two nodes, add
  `--expert_parallel_size=16 --ep_scope=global --max_length=8192`.
- **GPUs:** eight at EP8, 32 experts per GPU, at 33,000 tokens: about 17,200 tok/s per GPU at 155.5 GiB peak on a warm
  Triton cache.

## Recipes

Run these in the [training container](README.md#start-the-training-container).

### Full fine-tune

The shipped recipe trains Qwen3.5. To train Qwen3.6, override the model and its revision:

```bash
halo launch sft examples/sft/qwen3_5/qwen3.5-35b-a3b-ultrachat-ep.yaml -n 8 \
  --model_name_or_path=Qwen/Qwen3.6-35B-A3B \
  --model_revision=995ad96eacd98c81ed38be0c5b274b04031597b0 \
  --output_dir=/data/checkpoints/qwen3.6-35b-a3b-sft
```

Drop the first two overrides to train Qwen3.5.

### Other layouts

Add one of these to the full fine-tune command:

- EP8 + TP2, when attention memory is the limit: `--tensor_parallel_size=2`
- Pure ETP8, to shard every expert instead of placing whole experts: `--expert_parallel_size=1 --expert_tensor_parallel_size=8`
- A 4-way expert split: `--expert_parallel_size=4 --expert_tensor_parallel_size=2`

### LoRA

```bash
halo launch sft examples/sft/qwen3_5/qwen3.5-35b-a3b-ultrachat-ep.yaml -n 8 \
  --model_name_or_path=Qwen/Qwen3.6-35B-A3B \
  --model_revision=995ad96eacd98c81ed38be0c5b274b04031597b0 \
  --use_peft=true --learning_rate=1e-4 \
  --lora_target_modules=q_proj,k_proj,v_proj,o_proj \
  --output_dir=/data/checkpoints/qwen3.6-35b-a3b-lora
```

### MTP head

Train the `mtp.*` head the release ships, so a server can draft with it: over a frozen trunk, or inside a fine-tune
with `qwen3.6-35b-a3b-mtp-joint.yaml`. A fit repairs a stale head and holds a healthy one. The recipe fits chat rows
under the think render, not a server's own samples. Its 150 steps served at 0.834 acceptance on vLLM against the
release head's 0.831, on 64 first-turn prompts, and that head alone reads 0.83 to 0.84 across launches, so no served
gain is measured. What the `joint` recipe does over a fine-tune is not measured.

```bash
halo launch sft examples/sft/qwen3_5/qwen3.6-35b-a3b-mtp-head-only.yaml -n 8 \
  --output_dir=/data/checkpoints/qwen3.6-35b-a3b-mtp-head
```

The fit scores held-out rows before its first step and at its last, and exits non-zero if acceptance fell. Its last
checkpoint then holds the rejected head, so serve the one the fit started from. `mtp_train_scope: detached` keeps a
head current during a fine-tune without changing the trunk's objective. Serve with
`VLLM_SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":1}'`, or on SGLang with
[its one-draft flags](../../agent-docs/optimization/mtp-heads.md#serving) ↗.

### GRPO

The shipped recipes train Qwen3.6, most of them on code contests. Read
[Before a GRPO run](README.md#before-a-grpo-run) first. This one is a full fine-tune at EP4 with two vLLM
servers on GPUs 4–7.

On the host, start the servers on the SFT checkpoint:

```bash
export VLLM_MODEL=/data/checkpoints/qwen3.6-35b-a3b-sft
export VLLM_CHAT_TEMPLATE=/jinja-templates/qwen3/qwen3.6-reasoning-effort.jinja
export VLLM_TOOL_PARSER=qwen3_xml VLLM_REASONING_PARSER=qwen3
VLLM_CUDA_DEVICES=4,5 VLLM_TP=2 VLLM_PORT=8000 \
  docker compose -p qwen36-rollout-0 -f docker-compose.vllm.yml up -d vllm-server
VLLM_CUDA_DEVICES=6,7 VLLM_TP=2 VLLM_PORT=8001 \
  docker compose -p qwen36-rollout-1 -f docker-compose.vllm.yml up -d vllm-server
```

In the training container, copy the recipe, set its `dataset`, and launch on GPUs 0–3:

```bash
cp examples/grpo/environmental/qwen3_5/vllm/qwen3.6-35b-a3b-code-contests-full-ep4.yaml qwen3.6-grpo.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 DIST_NCCL_TIMEOUT_MINUTES=60 \
  halo launch grpo/environmental qwen3.6-grpo.yaml -n 4 \
  --model_name_or_path=/data/checkpoints/qwen3.6-35b-a3b-sft \
  --output_dir=/data/checkpoints/qwen3.6-35b-a3b-grpo
```

The other configs in `examples/grpo/environmental/qwen3_5/vllm/` change the environment, the adapter or
the EP size. The full fine-tune ep1 code-contests recipe is a curriculum: run `-stage1-codeforces`,
`-stage2-hard` and `-stage3-extra-hard` in order, each from the previous stage's checkpoint.

The `sglang/` configs (ep1) run on SGLang. The full fine-tune expects two TP=1 servers on GPUs 6 and 7
and a six-GPU trainer:

```bash
export SGLANG_MODEL="$HALO_SCRATCH/checkpoints/qwen3.6-35b-a3b-sft"
export SGLANG_CHAT_TEMPLATE=/jinja-templates/qwen3/qwen3.6-reasoning-effort.jinja SGLANG_REASONING_PARSER=qwen3
SGLANG_CUDA_DEVICES=6 SGLANG_PORT=30000 \
  docker compose -p qwen36-sglang-0 -f docker-compose.sglang.yml up -d sglang-server
SGLANG_CUDA_DEVICES=7 SGLANG_PORT=30001 \
  docker compose -p qwen36-sglang-1 -f docker-compose.sglang.yml up -d sglang-server
```

```bash
cp examples/grpo/environmental/qwen3_5/sglang/qwen3.6-35b-a3b-code-contests-full-ep1.yaml qwen3.6-grpo-sglang.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 DIST_NCCL_TIMEOUT_MINUTES=60 \
  halo launch grpo/environmental qwen3.6-grpo-sglang.yaml -n 6 \
  --model_name_or_path=/data/checkpoints/qwen3.6-35b-a3b-sft \
  --output_dir=/data/checkpoints/qwen3.6-35b-a3b-grpo-sglang
```

## Settings that matter

- **Attention.** The SFT recipe pins `flash_attention_2`, and the GRPO recipes pin `sdpa`. Unpinned,
  Blackwell runs FA4, and FA2 on packed SFT rows
  ([Attention backends](../supported-matrix.md#attention-backends)).
- **Packing, not padding-free.** Halo refuses `padding_free: true` on this family: its variable-length path
  crashes FA2 on the multimodal RoPE. Rows that share a long prefix can pack as trees on FA2
  (`tree_packing: true`;
  [Tree Packing](../../agent-docs/optimization/tree-packing.md) ↗).
- **Router balancing.** The SFT recipes set `moe_balancing: bias_update_transient`, and the GRPO recipes
  set `none`. The router has no bias slot, so plain `bias_update` raises. In SFT, `aux_loss` (what `auto`
  picks) balances through the loss and exports by construction. The transient bias balances training
  only: exported checkpoints serve without it, so near-tied expert picks can differ between trainer and
  server.
- **Chat template.** The SFT recipes render `qwen3-multiturn.jinja`, which writes assistant turns with no think block.
  `eval_loss` reads about 3.3 at step 0 on Qwen3.6-35B-A3B and about 0.83 after two updates. Train and serve under the
  same render.
- **Long context.** The SFT recipe sets `fp32_non_ep_params: true` for stable training at 33,000 tokens.
- **Tool parser and reasoning cap.** vLLM needs `VLLM_TOOL_PARSER=qwen3_xml`. The default `hermes`
  leaves the XML tool calls as text, and the run stops by its second round. The code-contests recipes cap each
  turn's reasoning at the answer room's edge, which needs `VLLM_REASONING_PARSER=qwen3`, or every request fails with a 400
  ([Reasoning caps and budgets](../training-methods/async-grpo-environments.md#reasoning-caps-and-budgets)).
- **SGLang rollouts.** `rollout_max_thinking_tokens` and `carry_reasoning` are refused at startup, and
  only `rollout_max_tokens` bounds a turn.

## Limits

- No CP: the linear-attention layers can't split the sequence across ranks.
- Pipeline parallelism takes the text tower only (`text_only_model: true`), with `moe_balancing: none`;
  packed rows keep their documents apart, one packed row per microbatch
  ([Qwen3.5 / Qwen3.6 model notes](../../agent-docs/models/qwen3_5.md) ↗).
- TP stops at 2 because the checkpoints have two KV heads. TP shards only the 10 full-attention layers,
  so per-GPU memory drops much less than `1/tp_size`.
- Online RL refuses `text_only_model: true`.
- Saves drop the hub checkpoint's `mtp.*` multi-token-prediction head unless the run
  [trains it](../../agent-docs/optimization/mtp-heads.md). `reattach-vision-tower` restores the base's untrained head for a
  text-only export.

## Export and serve

Gathered saves keep the language-model experts fused, like the hub checkpoints. Transformers and vLLM
0.32.0 read that layout directly. `halo run unfuse-moe-experts` rewrites it per expert for a loader that
needs that layout.

Smoke-test with `AutoModelForImageTextToText` and `AutoTokenizer`
([snippet](README.md#smoke-test-a-checkpoint)). Serve from the [host](README.md#serve-from-the-host)
with vLLM on port 8000:

```bash
VLLM_MODEL=/data/checkpoints/qwen3.6-35b-a3b-sft VLLM_CUDA_DEVICES=0,1,2,3 VLLM_TP=4 \
VLLM_TOOL_PARSER=qwen3_xml \
  docker compose -f docker-compose.vllm.yml up -d vllm-server
```

`text_only_model: true` drops the vision tower and processor files from the export. To rebuild the
multimodal checkpoint:

```bash
halo run reattach-vision-tower --input_dir /data/checkpoints/<text-only-export> \
  --model_id Qwen/Qwen3.6-35B-A3B --output_dir /data/checkpoints/<text-only-export>-vl
```

## Reference

- [Qwen3.5 / Qwen3.6 model notes](../../agent-docs/models/qwen3_5.md) ↗: why CP is blocked, the
  text-only export, chat templates
- [Async GRPO with Environments](../training-methods/async-grpo-environments.md)
- Model cards: [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B),
  [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
