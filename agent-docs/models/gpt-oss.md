# GPT-OSS

Two sizes: 20B and 120B. MoE with **attention sinks** (per-head learnable scalars) and an interleaved expert layout de-interleaved at load on the grouped-GEMM and ETP paths, then re-interleaved on save.

| | EP | CP | TP | ETP | PP | EP+TP | EP+CP |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| GPT-OSS 20B / 120B | Yes | Yes | Yes | Yes | — ¹ | Yes | Yes |

`GptOssConfig` ships a native HF `base_model_ep_plan` and no `base_model_tp_plan`, so its TP runs through the Halo selective-TP path.

¹ Pipeline parallelism is [not yet available in this release](../parallelism/pipeline-parallelism.md). Its shipped split contract binds stage boundaries to multiples of the period-2 `layer_types` pattern, and its balancing gate admits only `bias_update` or `none`.

## EP wrapper

`EPGptOssMoELayer` (`src/distributed/expert_parallel/layers/gpt_oss.py`) replaces `GptOssMLP`.

- Routing: custom router returning `(logits, weights, experts)`; softmax top-k with FP32 weights for DeepEP.
- Storage: interleaved `gate_up_proj [E, H, 2M]` (`[g0, u0, g1, u1, …]`), `down_proj [E, M, H]`, plus the optional expert biases `gate_up_proj_bias [E, 2M]` (interleaved the same way) and `down_proj_bias [E, H]`.
- Compute: **Grouped GEMM** (default on SM90+) reads de-interleaved `gate_proj_gmm` / `up_proj_gmm` stored as separate contiguous tensors at init, avoiding NaN gradients on the stride-2 slice. The **loop fallback** iterates per expert with `index_add_`.
- Activation: clamped SwiGLU, `(clamp(up) + 1) * clamp(gate) * sigmoid(alpha * clamp(gate))`. Every path — grouped GEMM, ETP and the per-expert loop — runs the same fused Triton kernel (`fused_gptoss_glu`, `src/kernels/fused_glu.py`, forward + backward, fp32 accumulation) rather than `torch.compile`, whose recompiles would storm on the varying per-step routed-token count and whose float arguments go stale once shapes are dynamic; the loop path splits the halves out of its interleaved projection output first. All read `alpha`/`limit` off `GptOssExperts` and reproduce its one fixed formula, so neither needs the activation probe that arms the plain-SwiGLU kernel ([GLM-4](glm4.md#fused-swiglu)) — GptOss owns these compute paths and never reaches the base `_glu_combine` seam.

## Packing needs a flash backend

`packing: true` is **refused** on eager/SDPA/flex for this family — `select_data_collator` raises.
GPT-OSS builds its attention mask from kwargs that omit `position_ids`, so a packed row runs as one
dense causal sequence and the documents inside it attend across each other silently. The flash
interfaces do receive `position_ids` and isolate correctly, so pin `flash_attention_2` (or a later
FA) whenever packing is on, and use padded batches otherwise
([Document isolation](../data/collators.md#document-isolation-under-packing)).

## Attention sinks

GptOss attention adds a per-head learnable scalar to the attention logits before softmax, with two parallelism consequences.

**TP sharding.** `shard_sinks_param()` (`src/distributed/tensor_parallel/parallelize_attention.py`) shards sinks per rank as plain tensors, not DTensors — the forward concatenates sinks with locally-sharded logits, and a DTensor would dimension-mismatch on the concat. The slice is recorded in `model._tp_sharded_non_dtensor` so `gather_tp_sharded_non_dtensor_params()` all-gathers it at save time.

**Sink handling** (`src/models/patches/gpt_oss_sinks.py`). One `SinksPolicy` per run, resolved from `reset_sinks` / `train_sinks` at load, stamped on the model for the trainer gates, and recorded in a LoRA run's `training_provenance.json` so the merge tools re-apply it (a merge rebuilds the base from the hub, whose sinks are always live).

- **Neutralized** (`reset_sinks: true`, the SFT default): `None` under FA2 (its kernel takes no sink argument), `dtype.min` and frozen everywhere else (~0 softmax mass, zero gradient). Every checkpoint writer re-emits the FA2-removed sinks as `dtype.min`.
- **Live, frozen** (`reset_sinks: false`): the pretrained sinks stay in every softmax, `requires_grad=False`. The RL setting — the trainer scores with the sinks the rollout engine serves — and the only live policy an adapter run or weight-sync trainer accepts. Live sinks admit only sink-carrying implementations (FA4, `flex_attention`, `eager`; FA3 with an `s_aux` build) and are refused under Context Parallelism, whose kernels never see the column.
- **Trainable** (`reset_sinks: false` + `train_sinks: true`): live with gradients on; full fine-tuning under FA4 or `eager` only. The fused FA4 backward returns no sink gradient, so the loader installs a rescale on both `flash_attn.cute` entry points (varlen and dense): a grad-requiring sink runs the sink-less kernel with `return_lse=True` and applies `out * sigmoid(lse - sink)` outside it — algebraically the fused sink, with `d_sink` from the gate and complete `dq/dk/dv` through `dout` and the kernel's `dlse` path (`tests/gpu/kernels/test_fa4_trainable_sink_rescale.py` pins all four against an eager reference on both entry points). Frozen sinks and no-grad calls pass through untouched. Refused: `reset_sinks: true` (a dtype-min sink has zero gradient), `flex_attention` (NaN sink gradients in its compiled backward), FA3 (no sink gradient), adapter runs (the adapter artifact has no sink slot), and weight-sync RL (no validated end-to-end sync of a moving sink; frozen live sinks are on-policy by construction).

## CP, TP, ETP

- **CP** — `GptOssAttention` → `GptOssUlyssesAttention` (`src/distributed/context_parallel/layers/gpt_oss.py`), RoPE before all-to-all and native GQA. Needs `reset_sinks: true`: the CP kernels drop the sink column, so the wrapper raises at the first forward on live sinks rather than normalizing every softmax wrongly.
- **TP** — a DTensor plan on attention only; `embed_tokens`/`lm_head` stay replicated (sharding them faults with a cuBLAS illegal-memory error under EP+TP), MoE experts belong to EP. See [the selective-TP plan](../parallelism/tensor-parallelism.md#the-selective-tp-plan).
- **ETP** — the interleaved-GLU layout needs re-interleaving back to `[g0, u0, g1, u1, …]` before the checkpoint write. Expert compute drops to the per-expert loop at `expert_tp_size > 1` (`_grouped_mm_enabled`): once the halves are TP-sharded they cannot be de-interleaved into the contiguous `gate_proj_gmm` / `up_proj_gmm` grouped GEMM reads. See [ETP weight sharding](../parallelism/expert-tensor-parallelism.md#weight-sharding).

## Precision and kernels

- `fp32_non_ep_params: true` is safe — the router lives inside the EP wrapper (unlike Gemma 4).
- **Blackwell** auto-selects FA4. SFT runs it with sinks reset to `dtype.min`; RL runs it with sinks kept and frozen. GPT-OSS is unaffected by the Qwen3.5/3.6 FA4→SDPA fallback.
- **SFT alternatives.** With sinks reset, `flash_attention_2` (`sinks = None`) and `sdpa` (the `dtype.min` column contributes `exp(dtype.min)=0`) also run end to end, both JIT-free. RL cannot use them: a sink-dropping impl shifts every logprob ~−3 nats vs the served policy.
- **RL sink gates.** Under `reset_sinks: false`, `validate_attn_implementation` rejects any implementation whose kernel does not accept a sink argument — SDPA always, and each flash build by its `s_aux` / `learnable_sink` signature, which the shipped FA2 lacks. The inverse shape is gated too: `validate_weight_sync_support` refuses on-policy RL when the FA2 reset removed the sinks Parameter (`reset_sinks: true` + FA2), since the sync sends named parameters only and the engine would keep serving the pretrained sinks against a sink-free trainer — permanently off-policy with no error at sync time. The sink-capability matrix is in [Flash Attention](../optimization/flash-attention.md#model-specific-handling).
- **Hopper FA2** needs the split-K kernel patch shipped in the image (`docker/training/flash_attn_split_stubs_hopper.cpp`).
- **Variable-length RL** (env-GRPO) uses FA4 through a per-row dense forward (`_dense_last_hidden_state`): a padded batch would unpad to a varlen FA4 call that re-JITs per length-set. Each row is forwarded alone, trimmed to its real span, with `attention_mask=None` — dense kernel, RoPE `[0, len)`, bit-identical log-probs. Every row stays connected to the loss so per-rank backward collectives keep lockstep.

## Configs

`examples/sft/gptoss/gptoss-20b-multinode-ep.yaml` is the infra-agnostic EP example for GPT-OSS 20B and doubles as the multi-node reference; `gptoss-120b-multinode-ep.yaml` is its 120B counterpart.

A multi-stage fine-tune must agree on `reset_sinks` across stages — a continuation run keeps the sink policy of the export it starts from.

Use the BF16-dequantized mirrors `unsloth/gpt-oss-{20b,120b}-BF16`. EP materializes experts as plain parameters, so a natively MXFP4 checkpoint (the original `openai/gpt-oss-*`, uint8 expert blocks) fails fast at patching with "Expert Parallelism requires a de-quantized (BF16) checkpoint". MXFP4 still serves on vLLM — but not under weight sync: its expert loader has no branch for a bf16 expert tensor, so a synced update drops every expert weight while the biases land, silently ([Rollout Servers](../infrastructure/rollout-servers.md#weight-sync)). On-policy RL serves the BF16 mirror.

The example sets `use_grouped_gemm: false`: at its EP=16 the 20B keeps 2 experts per rank, few enough that the loop is competitive, and gpt-oss's square expert FFN (`intermediate == hidden == 2880`, not a multiple of the kernel's 128-tile K) pays a CUTLASS tail epilogue the loop avoids. Grouped GEMM remains the default and wins at low EP and at high EP through moderate batch. See [Grouped GEMM](../optimization/grouped-gemm.md#when-the-loop-path-wins).

## Chat template

One template ships for GPT-OSS: `jinja-templates/gpt-oss/gpt-oss-harmony.jinja`, selected with `chat_template:`. It is the upstream OpenAI-harmony template from `unsloth/gpt-oss-20b-BF16`, kept so the training render is **byte-identical** to vLLM's server-side render. Its one deviation: tool and parameter `description` is guarded with `is defined`, so a tool without descriptions does not crash the render. (The checkpoint's built-in template renders `param_spec.description` unconditionally and raises without it.)

For SFT, `assistant_message_template` must match the form harmony actually renders: a plain assistant message (no `thinking`, no tool calls) renders as `<|start|>assistant<|message|>…`, while a message carrying `thinking` renders an `analysis` turn followed by `<|start|>assistant<|channel|>final<|message|>…`. The shipped SFT and self-distillation examples pair harmony with the channel-less marker. On multi-turn data only the final assistant turn ends in `<|return|>` (earlier turns end in `<|end|>`, which is not an eos id), so completion-only masking trains the final assistant turn and keeps earlier turns as context.

For RL the training and serving templates must match, since log-probs are computed on the served prompt. GPT-OSS ships a built-in template, so overriding it needs **both** `chat_template: jinja-templates/gpt-oss/gpt-oss-harmony.jinja` and `force_chat_template: true`. Pass the same file to vLLM via `VLLM_CHAT_TEMPLATE`.

## Serving for GRPO (vLLM)

Use the toolkit `vllm-server:0.26.0` image, not stock upstream vLLM: on B300 the stock `vllm/vllm-openai` image produces garbage GPT-OSS output with harmony on *or* off. The toolkit image (`Dockerfile.vllm` plus `docker/vllm/patches/patch_vllm_disable_gptoss.sh`) forces vLLM's harmony pipeline off and generates coherently.

With harmony disabled, five settings are load-bearing.

- **Text tool parser.** GPT-OSS emits tool calls as plain text (`commentary to=functions.<name> json{...}`), unreadable by the stock `openai` (needs harmony token IDs) and `seed_oss` parsers. Serve with `--enable-auto-tool-choice --tool-parser-plugin /opt/gpt_oss_text_tool_parser.py --tool-call-parser gpt_oss_text` (`docker/vllm/plugins/gpt_oss_text_tool_parser.py`). It brace-balances the JSON, returns the `final`-channel text as `content` so the CoT is dropped, rewrites the name-glued token (`submit_solutionjson` → `submit_solution`) to the longest declared tool name it starts with, and coerces malformed arguments into valid JSON so a bad call never 400s the next turn. Only the **first** call is extracted — without `<|call|>` as an eos the model keeps writing `to=functions.*` text and hallucinates its own tool result — and a header with no brace-balanced JSON after it is dropped rather than rescued, so the turn scores as a no-tool turn and the policy is pushed toward valid JSON. Non-tool RLVR needs no tool parser.
- **Reasoning budget.** `--reasoning-parser-plugin /opt/gpt_oss_reasoning_parser.py --reasoning-parser openai_gptoss` re-registers the harmony-only stock parser for the harmony-disabled path, which separates the CoT from the answer and is what lets a request carry `thinking_token_budget` at all (without it every such request 400s). vLLM arms the budget on the `reasoning_start_token_ids` it tokenizes from the parser's `reasoning_start_str` and ends it by forcing `reasoning_end_str`'s ids. No channel opening can be that marker — the parser treats every channel before `final` as reasoning while the render opens `analysis` or `commentary`, and a bare `<|channel|>` also matches the tool-result turn already in the prompt, cutting before the first sampled token — so the plugin arms on `<|start|>assistant`, the generation prompt's own tail. The budget then bounds every token generated before the `final` channel, whichever channel opened them (vLLM's counter trails the marker by one, so the cut lands at `budget + 1`). `reasoning_end_str` carries the role tokens — `<|start|>assistant<|channel|>final<|message|>` — so the forced cut is byte-identical to the ending the model writes itself: the channel marker alone would decode to a bare `final` glued to the truncated CoT, indistinguishable from prose `final` whenever the cut lands on whitespace, and the whole answer stays in `reasoning` with `content` empty. `docker/vllm/plugins/verify_gptoss_plugins.py` drives vLLM's own budget state machine over a simulated harmony-disabled stream at build time and pins that the forced marker's decoded text still splits, so a render or marker drift fails the image build rather than a run. The parser keeps the commentary tool call and the `final` answer in `content` — vLLM extracts tool calls from `content` only, so a call left in `reasoning` would be dropped and the turn mis-scored as a giveup.
- **`--return-tokens-as-token-ids`** — required whenever `train_on_sampled_tokens` is on (the default); the compose file passes it ([Rollout Servers](../infrastructure/rollout-servers.md#vllm)).
- **`rollout_stop_tokens: ["<|call|>"]`** — with harmony disabled `<|call|>` is not an eos, so the model keeps generating past its tool call and hallucinates the result for ~90% of the turn. A server-side `--override-generation-config` eos does not fix it ([Environmental GRPO](../training-methods/grpo/environmental-grpo.md#stopping-a-turn-at-the-tool-call-rollout_stop_tokens)).
- **Full padded vocab.** GPT-OSS ships 201088 embedding rows against `len(tokenizer)` 200019; rows 200019–201087 are unassigned padding. Every harmony special sits below that line — `<|return|>`/EOS at 200002, `<|call|>` at 200012, highest added id 200018 — so a resize to `len(tokenizer)` keeps them all. The trap is `tokenizer.vocab_size`, which is **199998**: shrink to that and the checkpoint loses EOS and emits garbage. `scripts/before_training/patch_vocab.py` only **grows** the embedding, so a patched checkpoint is safe. Serve the full-vocab base or a grow-only-patched checkpoint.

Sinks stay **on** at serving (the default): served sinks-off, the pretrained model degenerates to repetitive garbage with zero tool calls. The trainer matches by freezing the same sinks, so recompute equals vLLM to ~0 nats (`is_ratio ~1`).

GPT-OSS also serves from **SGLang** for env-GRPO (`rollout_backend: sglang`): SGLang's own built-in detectors replace both vLLM plugins — the compose default `--tool-call-parser auto` resolves gpt-oss off the chat template, and `SGLANG_REASONING_PARSER=gpt-oss` separates the analysis channel (both registered in 0.5.17); thinking budgets are rejected at config time for this backend, and the trainer needs `fsdp_reshard_after_backward: false` or the forced-socket NCCL makes FSDP2's per-microstep reshard the dominant step cost. Flags, constraints, and the measured step-cost ratio: [Rollout Servers](../infrastructure/rollout-servers.md#sglang).

## Router balancing

GPT-OSS defaults to switch-style **aux-loss** balancing: `moe_balancing: auto` resolves to `aux_loss`, and that mode turns `output_router_logits` on itself. The coefficient a run reads is the checkpoint's, not the class default — `GptOssConfig` declares `router_aux_loss_coef: 0.001`, but every released `config.json` (`openai/gpt-oss-*` and the `unsloth/*-BF16` mirrors) ships **0.9**, large enough to dominate the SFT loss. Every shipped example that stays on `aux_loss` overrides it back down to `0.001` in `model_init_kwargs`; do the same on a new aux-loss run rather than inheriting 0.9. A `bias_update` run needs no override — that mode zeroes the coefficient itself (below).

The DeepSeek-V3 **aux-loss-free bias update** is opt-in on the EP path (`moe_balancing: bias_update`) and adopts the hub router's own **`router.bias`** as the balancing state. The Parameter is re-registered as a persistent buffer under the same key: frozen out of gradient training, sign-updated by [`RouterBiasBalancingCallback`](../training-methods/callbacks.md#routerbiasbalancingcallback) each optimizer step, and exported with every checkpoint.

vLLM and SGLang load `router.bias` and route with it (top-k on bias-inclusive logits, combine = softmax over the selected values), and `_route_with_bias` computes exactly that arithmetic, so trainer and served copy pick the same experts with the same weights. `router_aux_loss_coef` is forced to 0 for the run to avoid double-balancing (restored in the exported config). The bias lives in logit space, where the default γ is a gentle nudge — the softmax-probability scaling argument other families need does not apply.

**Not for on-policy RL.** Online and env GRPO must use `moe_balancing: none` — adoption re-registers `router.bias` as a buffer and the weight sync ships parameters only, so a synced engine routes on the pretrained bias (`build_perf_callbacks` downgrades it automatically). Bias-update needs the EP wrappers (`ep_size > 1`, or the default `use_grouped_gemm`); without them it raises at setup.
