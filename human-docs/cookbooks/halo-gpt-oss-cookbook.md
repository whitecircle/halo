# GPT-OSS Cookbook

[GPT-OSS](https://huggingface.co/openai/gpt-oss-20b) is OpenAI's open-weight MoE in two sizes, 20B and
120B. Its attention adds a learned per-head "sink" logit, and its experts store gate and up projections
interleaved. Halo trains the BF16 mirrors of the release.

## Support

| FSDP | EP | CP | TP | ETP | EP+CP | EP+TP | LoRA | Online RL |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | vLLM, SGLang |

- **Checkpoints:** `unsloth/gpt-oss-20b-BF16` (32 experts, top-4) and `unsloth/gpt-oss-120b-BF16` (128
  experts, top-4).
- **GPUs:** the 20B recipe runs at EP8 on eight GPUs. The 120B trains at EP8 on eight B300s. On H100 or
  H200 it needs two nodes at EP16 (`examples/sft/gptoss/gptoss-120b-multinode-ep.yaml`,
  [Clusters](../clusters.md)).

## Recipes

Run these in the [training container](README.md#start-the-training-container).

### Full fine-tune

```bash
halo launch sft examples/sft/gptoss/gptoss-20b-multinode-ep.yaml -n 8 \
  --expert_parallel_size=8 --ep_scope=node --use_grouped_gemm=true \
  --output_dir=/data/checkpoints/gpt-oss-20b-sft
```

The shipped recipe is a two-node EP16 layout. These overrides run it on one eight-GPU node. They also
turn grouped GEMM back on: the file turns it off because at EP16, with two experts per rank, the
per-expert loop keeps up, while at EP8 grouped GEMM is faster.

### Other layouts

Add one of these to the full fine-tune command:

- EP8 + CP2, for long sequences: `--context_parallel_size=2 --packing=false`
- EP8 + TP2, when attention memory is the limit: `--tensor_parallel_size=2`

Or replace its `--expert_parallel_size=8` with one of these:

- Pure ETP8, to shard every expert instead of placing whole experts: `--expert_parallel_size=1 --expert_tensor_parallel_size=8`
- A 4-way expert split: `--expert_parallel_size=4 --expert_tensor_parallel_size=2`

### LoRA

```bash
halo launch sft examples/sft/gptoss/gptoss-20b-multinode-ep.yaml -n 8 \
  --expert_parallel_size=8 --ep_scope=node --use_grouped_gemm=true \
  --use_peft=true --lora_r=64 --lora_alpha=128 --learning_rate=1e-4 \
  --lora_target_modules=q_proj,k_proj,v_proj,o_proj,gate_up_proj,down_proj \
  --lora_modules_to_save=embed_tokens,lm_head,router \
  --output_dir=/data/checkpoints/gpt-oss-20b-lora
```

The expert targets (`gate_up_proj`, `down_proj`) go to Halo's grouped expert LoRA.

### GRPO

The shipped recipes train on code contests. Read [Before a GRPO run](README.md#before-a-grpo-run) first.
This one is a full fine-tune at EP4 with two vLLM servers on GPUs 0–3.

On the host, start the servers on the SFT checkpoint:

```bash
cp jinja-templates/gpt-oss/gpt-oss-harmony.jinja "$HALO_SCRATCH/"
export VLLM_MODEL=/data/checkpoints/gpt-oss-20b-sft
export VLLM_CHAT_TEMPLATE=/data/gpt-oss-harmony.jinja VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_TOOL_PARSER_PLUGIN=/opt/gpt_oss_text_tool_parser.py VLLM_TOOL_PARSER=gpt_oss_text
export VLLM_REASONING_PARSER_PLUGIN=/opt/gpt_oss_reasoning_parser.py VLLM_REASONING_PARSER=openai_gptoss
VLLM_CUDA_DEVICES=0,1 VLLM_TP=2 VLLM_PORT=8000 \
  docker compose -p gptoss-rollout-0 -f docker-compose.vllm.yml up -d vllm-server
VLLM_CUDA_DEVICES=2,3 VLLM_TP=2 VLLM_PORT=8001 \
  docker compose -p gptoss-rollout-1 -f docker-compose.vllm.yml up -d vllm-server
```

In the training container, copy the recipe, set its `dataset`, and launch on GPUs 4–7:

```bash
cp examples/grpo/environmental/gptoss/vllm/gptoss-20b-code-contests-full-ep4.yaml gpt-oss-grpo.yaml
CUDA_VISIBLE_DEVICES=4,5,6,7 DIST_NCCL_TIMEOUT_MINUTES=60 \
  halo launch environmental-grpo gpt-oss-grpo.yaml -n 4 \
  --model_name_or_path=/data/checkpoints/gpt-oss-20b-sft \
  --output_dir=/data/checkpoints/gpt-oss-20b-grpo
```

The other configs in `examples/grpo/environmental/gptoss/vllm/` switch to LoRA or to ep1. The ep1
configs use routing replay, so start their servers with `VLLM_ENABLE_R3=1` as well. The `sglang/` configs
run the same task on SGLang: serve them on ports 30000 and 30001 with `SGLANG_CHAT_TEMPLATE` on the same
harmony file, `SGLANG_REASONING_PARSER=gpt-oss` and `SGLANG_ENABLE_R3=1`.

## Settings that matter

- **Chat template.** SFT uses `jinja-templates/gpt-oss/gpt-oss-multiturn.jinja` with
  `force_chat_template: true`. Under `gpt-oss-harmony.jinja` the SFT completion marker matches no turn,
  so the run trains on nothing at a loss near zero. RL uses harmony, and the server must serve the same
  file.
- **Attention sinks.** SFT neutralizes them (`reset_sinks: true`, the default) and saves them that way.
  RL sets `reset_sinks: false`, which keeps the sinks live and frozen so the trainer scores what the
  server samples. A later stage runs the sinks it loads, so to carry the pretrained sinks into GRPO, run
  SFT with `--reset_sinks=false` too.
- **Router aux-loss coefficient.** The hub config ships `router_aux_loss_coef: 0.9`, which swamps the SFT
  loss. The shipped SFT recipes set it to `0.001` in `model_init_kwargs`. Do the same on any other run
  that balances with `aux_loss`, which is what `auto` picks for GPT-OSS.
- **RL rollouts.** The environment GRPO recipes set `rollout_stop_tokens: ["<|call|>"]`, without which
  the model writes past its tool call and invents the result. They also set
  `use_chunked_grpo_logprobs: true`, because the ~201k-token vocabulary makes the full logits plane too
  large.

## Limits

- The MXFP4 release (`openai/gpt-oss-*`) fails at load under EP. Train the BF16 mirror. A weight sync into
  an MXFP4 server also drops every expert weight without an error, so RL serves the BF16 mirror too.
- Live sinks (`reset_sinks: false`) need a kernel that takes a sink argument. FA2 and SDPA are refused,
  and so is CP. On Blackwell that kernel is FA4, which the GRPO recipes pin. The Hopper image's FA3 takes
  no sink, so on H100 or H200 use `--attn_implementation=flex_attention` or `eager`. Both refuse
  `packing`, so an SFT run there also needs `--packing=false`.
- `packing: true` needs a flash attention backend. It is refused on eager, SDPA and flex attention.
- ETP runs the experts through the per-expert loop instead of grouped GEMM.

## Export and serve

The gathered save re-interleaves the experts into the hub layout, so transformers, vLLM and SGLang load
it directly. Smoke-test it with `AutoModelForCausalLM`
([snippet](README.md#smoke-test-a-checkpoint)). Serve it from the
[host](README.md#serve-from-the-host) with SGLang on port 30000:

```bash
SGLANG_MODEL="$HALO_SCRATCH/checkpoints/gpt-oss-20b-sft" \
  docker compose -f docker-compose.sglang.yml up -d
```

## Reference

- [GPT-OSS model notes](../../agent-docs/models/gpt-oss.md) ↗: sink policies, the expert layout, the
  vLLM serving flags
- [Async GRPO with Environments](../training-methods/async-grpo-environments.md)
- Model cards: [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b),
  [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b)
