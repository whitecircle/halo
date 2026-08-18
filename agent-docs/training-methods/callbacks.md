# Training Callbacks

Callbacks extend `transformers.TrainerCallback` and hook the HF Trainer loop to log metrics, generate samples, schedule hyperparameters, and inspect state. Source: `src/callbacks/`.

Scripts assemble them with `build_training_callbacks(...)` (`src/training/script_runner.py`): `ParameterStatsCallback`, the optional generation callback, then `build_perf_callbacks(...)` (`src/callbacks/wiring.py`), which builds the efficiency / MoE / profiler callbacks from the YAML flags and resolves `moe_balancing`. `run_trainer` then calls `reorder_integration_callbacks_last(trainer)` before training ([why](#why-reorder-integration-callbacks)).

## Built-in callbacks

### ParameterStatsCallback

Prints total/trainable parameter counts, a per-module breakdown (layer indices normalized to `X`, e.g. `model.layers.X.self_attn`, grouping identical layers), and dtype/memory stats on `on_train_begin`, main process only. No constructor parameters. Pass as a class: `callbacks = [ParameterStatsCallback]`.

Every count is **this rank's local** share: the callback runs after wrapping, where DTensor params (FSDP2/TP) and FSDP-ignored EP expert tensors report on different scopes. Multiply by the right group size for a global figure.

### GenerateExamplesCallback

Generates text samples on `on_evaluate` and logs them to W&B or ClearML as tables. Each record has `Prompt`, `Completion`, and optionally `Chosen` / `Rejected` (preference data).

```python
GenerateExamplesCallback(
    preprocessed_dataset=eval_dataset,  # tokenized: input_ids + attention_mask
    tokenizer=tokenizer,
    num_examples=5,
    max_new_tokens=256,
    logger_backend="wandb",             # "wandb" or "clearml"
)
```

`GenerateExamplesCallback.from_config(args, training_config, generate_dataset, tokenizer)` returns `None` unless `generate_eval_examples` is set and the generate split carries tokenized `input_ids` (the VLM path has none, so it is skipped there).

| Mode | Behavior |
|------|----------|
| Single GPU | All examples generated directly |
| DDP (accelerate MULTI_GPU) | Split across ranks, gathered on rank 0 |
| FSDP2 (accelerate or torchrun standard DP) | All ranks generate together (collective forward) |
| EP / ETP (torchrun) | All ranks generate together (required by all-to-all / ETP gather) |
| TP (torchrun) | **Skipped** — TP-sharded DTensor params incompatible with `generate()` |
| CP (torchrun) | **Skipped** — Ulysses attention incompatible with autoregressive decode |
| PP (torchrun) | **Skipped** — a stage holds only part of the model; generate from a saved checkpoint |

### EfficiencyCallback

Tracks token throughput, step time, and GPU memory. tokens/s/GPU is the headline; MFU / S-MFU / achieved-TFLOPS are computed every step but reach the log only under `report_mfu_diagnostics: true`. Source: `src/callbacks/efficiency.py`.

Off by default (`enable_efficiency_metrics: false`): the utilization misleads for multi-sequence trainers — DPO / SMPO / Reward forward both chosen and rejected, distillation also forwards the teacher.

Throughput is measured only where the collator emits `input_ids`. SMPO and embedding emit none, so `num_input_tokens_seen` never advances and every step falls back to a padded-length estimate (`per_device_train_batch_size × gradient_accumulation_steps × max_seq × 0.8`) behind one warning — a number that reads as measured and is not.

Auto-wiring passes the EP / TP / CP / ETP / PP sizes from `ParallelismConfig`, `max_seq_len` (`resolve_max_seq_len`), the `report_mfu_diagnostics` flag, and sets `include_num_input_tokens_seen="all"` — the field is tri-state in transformers 5 (`"no" | "all" | "non_padding"`), so a boolean is the wrong type and `"no"` is truthy. `max_seq_len` feeds the attention FLOP term and falls back to 2048 when nothing declares a bound, which is why the environmental-GRPO script passes its own multi-turn bound. The expert count and router top-k are not parameters: they come from `detect_moe_experts_topk`, the same probe the balancing wiring and the MoE load metrics read, so no consumer can classify a family as MoE while another calls it dense.

Supported GPUs: H100 (SXM/PCIe/NVL), H200, B200, B300, GB200, A100, A6000. Precision: bf16, fp16, fp32, tf32, fp8 (Hopper and Blackwell only), fp4 (B300 only). The precision is the run's `lowp_precision` wherever low-precision compute is on (`mxfp4` scores against the fp4 peak, since both fp4 recipes issue the same MMA) — the masters stay bf16 there, so the parameter dtype names a peak the GEMMs never run at — and otherwise the declared training dtype, TF32-aware for fp32 parameters. A (GPU, precision) pair outside that registry leaves peak FLOPS unknown, and MFU / S-MFU then report 0.0.

**S-MFU.** A plain MFU overstates sparse-MoE utilization — it credits FLOPs for experts that did not fire — so the callback also reports S-MFU, which scales the *expert* FLOP term by `(top_k / num_experts) × ep_size`. A rank holds `num_experts / ep_size` experts but serves its whole EP group's tokens, so the `ep_size` factor makes per-rank active FLOPs/token ep-invariant; `expert_tp_size` is absent by design, since it already divides the local expert params. With `num_full_model_params` set, the full expert bank is reconstructed as `local_expert_params × ep_size × expert_tp_size` — both axes shard the expert FFN.

`num_experts` / `top_k` come from the shared `detect_moe_experts_topk`, which reads the **text** sub-config as well as the wrapper and covers `top_k_experts` (Gemma 4) alongside `num_experts_per_tok` / `top_k` / `moe_router_topk` / `num_active_experts`. An undetected top-k gives `sparsity_factor = 1.0`, i.e. S-MFU collapses to dense MFU.

Always logged: step and average step time, per-GPU and cluster tokens/s, allocated/reserved/peak memory. Under `report_mfu_diagnostics` the log also carries MFU percent, achieved TFLOPS, distributed efficiency, plus S-MFU for MoE. After training, read `efficiency_cb.mfu` / `.smfu` / `.tps` / `.memory`.

### Token-count metrics

Three cumulative counters answer different questions and are not interchangeable. Each is a running total over the run (resets on resume).

| Metric (`train/`) | Counts | Source |
|---|---|---|
| `num_input_tokens_seen` | every element of `input_ids` (`.numel()`), summed over ranks — includes padding and prompt | HF Trainer, gated by `include_num_input_tokens_seen` (set to `"all"` when `enable_efficiency_metrics: true`) |
| `num_attended_tokens_seen` | non-padding attended tokens (`attention_mask == 1`), de-duplicated across TP/ETP ranks and reconstructed across CP ranks | `src/trainers/sft.py` (SFT only) |
| `num_unmasked_output_tokens_seen` | loss-contributing tokens only — `labels != -100` (causal LM) or the per-step completion mask (GRPO) | `TokenMetricsMixin` (`src/trainers/mixins/token_metrics.py`) |

They nest: `num_unmasked_output_tokens_seen ≤ num_attended_tokens_seen ≤ num_input_tokens_seen`. Under `packing: true` there is almost no padding, so the first two nearly coincide. `num_unmasked_output_tokens_seen` covers SFT and distillation (token-aligned `labels`), online / environmental GRPO (`completion_mask ∧ tool_mask`), and offline GRPO (`completion_attention_mask`); preference trainers keep chosen/rejected separate until `compute_loss` and sequence-level trainers (classification, reward, embedding) have no token mask, so neither logs it.

The throughput metrics (`step_tokens_per_second`, `avg_tokens_per_second`, and the `*_cluster_*` pair) derive from `num_input_tokens_seen` deltas over step compute time, so they count padding too. Per-GPU and cluster differ only by the constant `data_parallel_size × cp_size` (EP excluded, since EP ⊥ DP). HF's `train_tokens_per_second` measures the same cluster throughput over whole-session wall-clock (including dataloading / eval / checkpoint stalls), so it reads lower.

### MoEMetricsCallback

Tracks per-expert load distribution for any MoE model that declares a router. A forward hook on each router counts the experts it selected (`scatter_add_` into a per-layer counter), accumulates across micro-batches, all-reduces, and injects `moe/*` keys. Source: `src/callbacks/moe_metrics.py`.

What is counted is the router's **own** selection, not `topk(router_logits)`. DeepSeek-style routers select over `score(logits) + e_score_correction_bias` inside an `n_group` / `topk_group` mask and publish the raw pre-bias, pre-mask logits, so re-ranking those reports a distribution the model never routed: the bias flips near-ties onto another expert, and the experts of a masked-out group still hold the largest raw logits in the layer. Routers are discovered from transformers' own `_can_record_outputs["router_logits"]` declaration — the same registry that decides what `output_router_logits` captures — and the selection is read off a router's output as the integer tensor sharing the logits' token dimensions. A router returning logits alone falls back to `topk(router_logits)` and warns once, naming the model type. A router with a trailing discard slot (Zaya) is read from its logits too: its index tensor folds skipped tokens onto a real expert id, so only the logits still carry the discard decision. Gradient checkpointing re-runs a router in the recompute pass; every metric is a per-layer share, so the doubled counts cancel.

On by default (`enable_moe_metrics: true`); no-op for dense models (attaches only when `num_experts` / `num_local_experts` / `num_routed_experts` / `num_moe_experts` / `moe_num_experts` > 1). Skipped under either bias-update mode (`bias_update`, `bias_update_transient`), where `RouterBiasBalancingCallback` emits the same `moe/*` keys from `expert_load_counter` and dashboards stay identical. Also skipped under pipeline parallelism (a stage forward returns a bare tensor, no `ModelOutput`). It also skips wiring when the balancing strategy stamped `output_router_logits` as forced-off (see [MoE balancing modes](#moe-balancing-modes)) rather than re-enabling it behind the strategy's back.

Every router field — expert count, top-k, `router_aux_loss_coef`, `output_router_logits` — is read through `config_sources` (`src/models/loading/config_levels.py`), which consults `config.get_text_config()` as well as the wrapper. Composite (multimodal) MoE configs need that: Qwen3.5/3.6-MoE, Gemma 4 and Mistral-4-under-the-VLM-wrapper keep those fields on `text_config` only, and `PreTrainedConfig` defines no `__getattr__`, so a wrapper-only read yields the default and silently turns balancing and metrics into no-ops. Top-k detection spans the same spellings as above.

The hooks fire only when the HF router **module** runs. Two paths bypass it: a wrapper that routes without it (`_ep_severs_aux_loss` — Bailing/Ling, Inkling, DeepSeek-V4), and a **GRPO** trainer whose log-probs come from the backbone alone. EP by itself is not one of them — a wrapper that keeps the original router module (Qwen3 as `self.gate`, GPT-OSS as `self.router`) still reports. On the bypassing paths the callback warns once after the first optimizer step rather than leaving the keys silently absent; use `moe_balancing: bias_update` for load metrics there. A modeling that surfaces `router_logits` without declaring a router — remote code — is counted from the model output's per-layer logits instead, the fallback reading above.

The expert-load all-reduce is rank-uniform by construction: the participation gate (`global_step % log_every_n_steps`) and the payload shape (`[num_layers, num_experts]`, agreed by a `MAX` all-reduce over the local dims) are identical on every rank, so a rank whose hook captured nothing joins with zeros instead of hanging its peers. `reduce_group` narrows the reduce to a subset of the world; the PP mixin sets it to the stage group.

Cost: `output_router_logits=True` adds a `[B*S, num_experts]` tensor per MoE layer per forward — hundreds of MB on a 36-layer, 128-expert model at s=16k, growing with batch × sequence — **and** makes HF MoE forwards add `router_aux_loss_coef * aux_loss` to the loss. Neither is a side effect of asking for metrics: the callback flips the flag on only under `enable_router_logits`, which `build_perf_callbacks` passes solely for `moe_balancing: aux_loss` — the mode that already requires router logits. Otherwise it observes logits something else enabled, and when they are off it logs one line and emits no `moe/*` keys. Its own compute is one `scatter_add_` per layer — plus a `topk` on the fallback reading — negligible against the FFN.

Logged metrics, prefixed `moe/`, averaged across MoE layers unless suffixed, in units of per-expert token share normalized to uniform routing (1.0 = uniform):

| Metric | Meaning |
|---|---|
| `moe/load_max` | Max per-expert share (>1 = hot expert) |
| `moe/load_min` | Min per-expert share (→0 = dead/cold expert) |
| `moe/load_cv` | Coefficient of variation (std/mean); 0 = perfect balance |
| `moe/dead_frac` | Fraction of experts that got zero tokens this step |
| `moe/load_max_first` / `moe/load_max_last` | Max-load ratio for the first / last MoE layer |
| `moe/num_layers` | MoE layers detected (sanity check) |

Counters are zeroed every step, so each emitted sample covers exactly one optimizer step. `log_every_n_steps` is wired from `logging_steps` (a fractional `logging_steps`, which is a ratio of total steps, means every step): only `on_log` consumes the metrics, so deriving them more often buys nothing and costs a collective plus a host sync per step.

Manual construction — the defaults: `MoEMetricsCallback(topk=1, exclude_last_slot=False, log_every_n_steps=1, enable_router_logits=False)`. Pass the model's router top-k as `topk` (the default of `1` counts one expert per token); it is used only where a router exposes no selection of its own and its logits must be re-ranked. Set `exclude_last_slot=True` for routers with a trailing discard slot (Zaya).

### RouterBiasBalancingCallback

DeepSeek-V3 auxiliary-loss-free router balancing. Expert selection is steered by a per-router bias added to the *detached* top-k scores, so gradients flow through the un-biased gating weights — model quality is identical to no-balancing. Source: `src/callbacks/router_bias_balancing.py`; EP-side bias creation/injection lives in `EPRouterBalancingMixin` (`src/distributed/expert_parallel/balancing.py`, mixed into `EPMoELayerBase`) + per-family `route_tokens_to_experts`.

Update rule (DeepSeek-V3 eq. 16): `b_i ← b_i + γ · sign(mean_count − count_i)`, where `count_i` is tokens routed to expert *i* over the last optimizer step (summed across DP/EP ranks since the router is replicated), `mean_count` averages over real experts. Sign-only and tiny (γ = `router_balancing_rate`, default 1e-3), so it perturbs routing but not the gating distribution.

The strategy holds an **export contract**: `moe_balancing: bias_update` is only accepted where the trained bias lands in checkpoint-exported state a serving engine loads, so trainer and server route identically. Three ways a model satisfies it:

- **Native bias (Zaya)** — `balancing_biases` is part of `ZayaRouter` (a *persistent buffer*, round-trips through the checkpoint), so `moe_balancing: auto` resolves to `bias_update` and it is always on.
- **Adopted native slot** — where the family ships a checkpoint-persistent tensor its own selection arithmetic consults, the EP layer declares it (`_NATIVE_BALANCING_BIAS_ATTR`) and the sign-updates are applied to **that** tensor: GLM-4 and Laguna, DeepSeek-V4, Inkling, GLM-5 Next and Step-3.7 Flash (`gate.e_score_correction_bias`), Bailing (`gate.expert_bias`), LFM-2 (the block's `expert_bias`), and GPT-OSS (`router.bias` — the hub router's own logit-space bias, which vLLM and SGLang load and route with). Routing is then steered through the mechanism the model serves with, and the **final bias exports with every checkpoint**. A `nn.Parameter` slot is re-registered as a persistent buffer under the same key — controller state, which also freezes it out of gradient training.
- **Materialized slot (LFM-2 with `use_expert_bias: false`)** — the slot is config-gated, not structural, so enabling balancing creates the architecture's own zero `expert_bias` buffer (same sigmoid-score space, a semantic no-op at creation) and flips `use_expert_bias` on the wrapper **and** the exported config, so engines load and apply the trained tensor.

The contract is enforced on the **enabled tree**, not the class declaration: a renamed upstream slot, or one already FSDP-sharded as a DTensor, falls back to the side-buffer at enable time and `bias_update` then raises. Bailing is the exception with no fallback at all — its selection runs entirely inside the hub gate, so a side-buffer could never shift routing, and the layer declares `_supports_transient_balancing_bias = False`: enabling balancing without the native slot raises instead of creating a buffer selection ignores.

A slot that exports correctly can still be dropped on the way in. A layer class declaring `_SERVED_BALANCING_BIAS_DROPPED_BY` names an engine whose loader skips the exported key — Laguna under vLLM 0.26.0 — and the balancing enable warns at run start. Training and the checkpoint are correct; a transformers reload routes as trained, that engine routes on the pretrained bias.

Where the architecture has **no slot at all** — Qwen3, Qwen3.5/3.6, Mistral4, Cohere2 MoE — only a *plain-attribute side-buffer* could carry the bias, which no `state_dict()`, export, or weight-sync ever sees. An explicit `bias_update` there **raises**. `moe_balancing: bias_update_transient` opts into that trade-off deliberately: balancing works during training and resume is exact, but every exported checkpoint serves *without* the bias, so near-tied top-k picks flip between trainer and server and serving-time expert load reverts to the raw gate's. The transient spelling raises in turn on a model whose routers all adopt native slots, so a config cannot misdocument the run.

Every mode checkpoints the bias to `router_balancing_biases.pt` for resume. A resume from a checkpoint without that file keeps zero-init biases and warns; a sidecar present on some ranks and not others raises, as does a torn read; a saved bias whose shape does not match the live router **raises** rather than being `copy_`-broadcast into it — a different expert count or router layout.

A sidecar that loads but matches **no** live router is the loud case in the other direction: every trained bias is dropped and the restore warns on every rank. It means this run resolved to `moe_balancing: none`, so no module carries the balancing attribute the restore looks for — exactly what weight-sync RL does to both bias modes ([below](#no-balancing-at-all-under-weight-sync-rl)). An adopted-slot family then keeps routing on the value its safetensors already carry, now frozen; side-buffer and `bias_update_transient` families lose the bias outright, and the run emits no sidecar of its own to resume from.

The after-training merge tools (`scripts/after_training/merge_peft_adapters.py`, `scripts/after_training/convert_to_bf16.py --merge_adapter`) copy the sidecar values into the merged model's native slots — a PEFT merge starts from base weights, which never saw the sign updates.

| Family | EP class | Where the bias lands |
|---|---|---|
| Zaya | `EPZayaMoELayer` | the gate's **native** persistent `balancing_biases` buffer — hub-native, so `auto` resolves to `bias_update` in every mode |
| GPT-OSS | `EPGptOssMoELayer` | **native `router.bias`** — the hub router's logit-space bias, so selection *and* combine weights match the served model exactly (`bias_update`) |
| Qwen3 | `EPQwen3MoELayer` | no slot — selection re-derived in the wrapper, `bias_update_transient` only |
| Qwen3.5 / Qwen3.6 | `EPQwen3_5MoELayer` | no slot — routing re-derived in the wrapper (softmax), one wrapper for both families; `bias_update_transient` only |
| GLM-4 MoE Lite | `EPGlm4MoELayer` | the gate's own `e_score_correction_bias` (sigmoid + group) |
| Laguna | `EPLagunaMoELayer` | the same gate slot as GLM-4, exported under the hub key `experts.e_score_correction_bias` |
| LFM-2 MoE | `EPLfm2MoELayer` | the block's `expert_bias` buffer (sigmoid); a `use_expert_bias: false` checkpoint gets the slot materialized (zeros) and the config flag flipped for the export |
| Bailing / Ling 2 & 3 | `EPBailingMoELayer` | the hub gate's persistent `expert_bias` (sigmoid + group, selection only), adopted whole — routing stays the gate's own |
| Mistral-4 MoE | `EPMistral4MoELayer` | no slot — bias added to the softmax scores, `bias_update_transient` only |
| Inkling | `EPInklingMoELayer` | the gate's `e_score_correction_bias` (sigmoid + shared experts) |
| DeepSeek-V4 | `EPDeepseekV4MoELayer` | the gate's `e_score_correction_bias` (sqrtsoftplus scores); hash layers refuse the bias — frozen `tid2eid` selection |
| Cohere2 MoE | `EPCohere2MoELayer` | no slot — bias into the selection scores before top-k (gate weights stay the unbiased activation of the selected raw logits), `bias_update_transient` only |
| GLM-5 Next | `EPGlm5NextMoELayer` | the gate's own fp32 `e_score_correction_bias` buffer (sigmoid `noaux_tc`, selection only), the GLM-4 slot spelling |
| Step-3.7 Flash | `EPStep3p7MoELayer` | the gate's own `e_score_correction_bias` buffer (sigmoid, selection only; hub key `moe.router_bias`) |

**Requires EP patching.** `expert_parallel_size=1` still applies the wrappers when `use_grouped_gemm` is on (`needs_ep_wrappers = ep_group_size > 1 or use_grouped_gemm`). Under plain FSDP (no wrappers) an explicit `moe_balancing: bias_update` raises, rather than silently balancing nothing — except on routers with native balancing biases (Zaya), where it still applies. An explicit bias mode on a **dense** model raises the same way: there are no experts to balance, and a silently inert controller is not a configuration.

**Not supported** (no bias route): Gemma 4 — its EP wrapper receives pre-weighted indices, so no bias could shift selection ([below](#gemma-4-has-no-balancing-path)).

**Bias step size depends on the injection space.** The side-buffer families add `balancing_biases` to the **softmax probabilities**, whose scale is `1/num_experts`: scale `router_balancing_rate` down in proportion at large expert counts, since at the 1e-3 default one sign step is ~50% of the uniform probability on Qwen3.5-397B-A17B (512 experts). GPT-OSS's adopted `router.bias` lives in **logit space**, where the route reads it at the router's compute dtype and a 1e-3 step is sub-ULP in bf16 — for a more responsive controller raise the rate toward 1e-2, or set `fp32_router: true`. Watch `moe/load_cv` for oscillation.

Router contract: a router opts in by exposing `self.balancing_biases` (shape `[E]` or `[E+1]`) and `self.expert_load_counter`. Its `forward` adds `balancing_biases` to detached scores before top-k and accumulates per-step counts into `expert_load_counter` under `no_grad` when training. `expert_load_counter` must be a **plain attribute, not a registered buffer** — it has to stay out of `state_dict()` (exports, sidecars) and out of wrapper machinery, and the PP lazy loader materializes state-dict tensors only, so a non-persistent buffer would stay on meta on the shell.

The callback stamps `balancing_active = True` on every balancing router at train begin — the gate for routers whose bias state exists regardless of balancing (Zaya's hub-native buffer), so a `moe_balancing: none` run records nothing. It then all-reduces counters each step, applies the sign-step, zeros them, and emits the same `moe/*` metrics as `MoEMetricsCallback`.

The bias update runs **every** optimizer step and its all-reduce is unconditional. Only the `moe/*` summary is periodic, on the same `logging_steps` period as `MoEMetricsCallback`, because it ends in a device-to-host sync only `on_log` reads. `reduce_group` narrows the counter all-reduce; the PP mixin sets it to the stage group, since each stage holds different routers.

Manual construction — the defaults: `RouterBiasBalancingCallback(update_rate=1e-3, exclude_last_slot=False, log_every_n_steps=1)`. `exclude_last_slot` treats the trailing `balancing_biases` position as a discard slot (excluded from the update, clamped to `-1.0`) — set for Zaya. Auto-wiring derives it from the router class attribute `_has_discard_expert_slot` and nothing else — the toolkit's Zaya patch sets it on `ZayaRouter` at load, so a family opts in where its routing is defined.

## MoE balancing modes

```yaml
moe_balancing: auto              # default — bias_update where it exports, aux_loss where the forward can carry it, else none
router_balancing_rate: 1.0e-3    # gamma; only used when a bias-update mode is active
```

| `moe_balancing` | Effect | `router_aux_loss_coef` | `output_router_logits` | Callback |
|---|---|---|---|---|
| `auto` (default) | `bias_update` where the bias lands in **exported** state ([below](#auto-resolution-per-family)); `none` + warning where only a transient bias would be possible (Mistral4, Cohere2 MoE, multimodal Qwen3.5/3.6) or where nothing can balance at all ([below](#gemma-4-has-no-balancing-path)); `aux_loss` for other MoE, `none` for dense | per resolved mode | per resolved mode | per resolved mode |
| `bias_update` | DeepSeek-V3 sign update landing in the family's own checkpoint slot (native, adopted, or materialized); `enable_bias_balancing()` on each EP layer creates the state before FSDP wraps; **raises** where any router would fall back to the transient side-buffer | **forced 0** | **forced False** | `RouterBiasBalancingCallback` |
| `bias_update_transient` | The same sign update held in a trainer-only side-buffer, on families with no exportable slot — explicit opt-in; warns that every export serves without the bias; **raises** where every router exports natively | **forced 0** | **forced False** | `RouterBiasBalancingCallback` |
| `aux_loss` | Model's native switch-style aux loss — see the inertness list below | preserved | **True** only if coef `> 0`, the EP wrappers keep the aux path, and the trainer applies the term | none |
| `none` | No balancing | preserved | preserved | none |

Both bias modes force `output_router_logits=False` (overriding any `model_init_kwargs`) because the EP bias path bypasses the HF router module the `OutputRecorder` hooks: leaving it on returns an empty `router_logits` tuple and `load_balancing_loss_func` raises `IndexError`. These writes — the zeroed coefficient, the forced flag, and the forced-off stamp — are **run-scoped**: every config serialization restores the pre-run values (`config_export_ready`), so the exported `config.json` carries the hub's aux coefficient and flag, not this run's balancing state.

`aux_loss` enables `output_router_logits=True` only when the model has a usable aux term (`router_aux_loss_coef > 0`). With no usable coef — aux-loss-free routers like GLM-4 MoE Lite's `noaux_tc`, or an unset coef — it leaves the flag off and warns: enabling it would make TRL's SFTTrainer read an `outputs.aux_loss` those models never populate and raise `AttributeError`. It does the same where the EP wrappers sever the aux path (DeepSeek-V4) and `router_logits` stays an empty tuple. Use `bias_update` for real balancing on GLM-4 / LFM-2.

With a usable coef, an **explicit** `aux_loss` **raises** when the model's `forward` does not declare `output_router_logits` as a parameter (`honors_output_router_logits_config`, enforced in `src/distributed/expert_parallel/balancing_strategy.py`); `auto` never resolves there ([below](#auto-resolution-per-family)). HF's config fallback lives on that parameter, so a class reading the flag out of `**kwargs` never consults the config the mode sets: router-logit *recording* switches on — a `[tokens, num_experts]` plane per MoE layer, every forward — while the aux term never reaches the loss. Multimodal wrappers are the case that bites: `Qwen3_5MoeForConditionalGeneration` reads it from `kwargs` only, while its text-only `Qwen3_5MoeForCausalLM` sibling honors the config. On such a checkpoint use `bias_update`, which is aux-loss-free and works under EP.

Reads and writes of both fields go through `get_config_field` / `set_config_field`, which cover the wrapper **and** `config.get_text_config()`; writing only one would leave the two disagreeing about whether balancing is on. The resolved mode is then reconciled world-wide by an `all_gather_object` with precedence `bias_update > bias_update_transient > aux_loss > none` (`agree_balancing_mode`) — under PP a stage holding no MoE layer resolves `auto` out of ignorance, and a split verdict would send stages into different collectives.

### `auto` resolution per family

`auto` wants `bias_update` on three signals: a native `balancing_biases` router (Zaya); EP wrappers that sever the aux-loss path (DeepSeek-V4, Bailing, Inkling); or a `forward` declaring no `output_router_logits` parameter while an EP wrapper accepts the bias.

It **commits** to `bias_update` only where the bias would land in exported state (`accepts_native_balancing_bias`). GLM-4 MoE Lite, LFM-2 and Step-3.7 Flash qualify on the third signal (Step-3.7's modeling has no aux-loss machinery at all); GLM-5 Next does not (its forward honors the flag, so `auto` gives `aux_loss` and the exported slot serves explicit `bias_update`). **Mistral-4, Cohere2 MoE and the multimodal `Qwen3_5MoeForConditionalGeneration`** have no slot, so they resolve to `none` with a warning naming `bias_update_transient` as the deliberate opt-in — `auto` never defaults into a bias no export carries.

`aux_loss` follows for the families whose `forward` declares the parameter — GPT-OSS, Qwen3, Laguna, the text-only `Qwen3_5MoeForCausalLM` (GPT-OSS gets the native `router.bias` adoption only on an explicit `bias_update`). Where it does not, and no wrapper accepts a bias either, the model has **no balancing route at all** and `auto` resolves to `none` with a warning naming the reason: `aux_loss` there is the mode the strategy refuses, and `auto` never selects a mode the enabled tree cannot serve.

The wrapper condition holds on the default path. Without the EP wrappers (`use_grouped_gemm: false` at `ep_size = 1`) the wrapper-signal models lose their bias route: those declaring the `forward` parameter fall back to `aux_loss`, the rest (GLM-4 MoE Lite, LFM-2, Step-3.7 Flash) to `none` — the run trains unbalanced, and the warning says so.

### Gemma 4 has no balancing path

`auto` resolves Gemma 4 to `none` with a warning: its EP wrapper accepts no bias, no router ships a `balancing_biases` buffer, and its `forward` declares no `output_router_logits` parameter, so no mode can balance it. An explicit `aux_loss` finds no `router_aux_loss_coef` on the text config either, so it stamps `output_router_logits` forced-off and warns rather than raising; `MoEMetricsCallback` honors that stamp and skips wiring. Either way Gemma 4 yields no balancing **and** no `moe/*` metrics, and `output_router_logits: true` in `model_init_kwargs` cannot change that: neither `Gemma4Config` nor `Gemma4TextConfig` declares that field, so `_apply_config_overrides` rejects the override at load. Every shipped Gemma 4 config sets `moe_balancing: none`.

`bias_update` is not an option either: Gemma 4's router lives in the parent decoder layer rather than the EP module, so an explicit `bias_update` raises at setup. Do not force the aux term on by setting `router_aux_loss_coef` in `model_init_kwargs` — Gemma 4's modeling code has no aux-loss wiring, and TRL's chunked-CE path reads `text_config.num_experts_per_tok`, which Gemma 4 spells `top_k_experts`. Run Gemma 4 with `moe_balancing: none`, or with `aux_loss` accepted as observation-only.

### Where `aux_loss` is inert

`*ForCausalLM.forward` adds `router_aux_loss_coef * load_balancing_loss_func(...)` **only when `labels is not None`**. So `aux_loss` balances experts under SFT and pretraining — labels ride the batch, and TRL's chunked-CE / Liger path adds the same term. KTO reaches it by another route: TRL's `KTOTrainer` forwards `output_router_logits=True` and adds `aux_loss_coef * aux_loss` to the loss it assembles itself. Everywhere else the term never reaches the loss:

- **DPO / SMPO / Reward** — the forward is called without `labels`, and none of the three trainers add the term.
- **Distillation (teacher and self)** — both forward with `labels` stripped, to get full-vocab logits past the fused-LCE shortcut.
- **Classification / Embedding** — non-causal heads; the causal-LM forward that carries the term never runs.
- **GRPO (online / offline / environmental)** — the loss is built from per-token log-probs. The scripts pass `policy_gradient_loss=True`, so the mode warns loudly and leaves `output_router_logits` off.
- **Pipeline parallelism** (itself [not yet available](../parallelism/pipeline-parallelism.md)) — a stage would run the backbone and apply the head itself, so the term is dropped; the shipped split gate **raises** when `aux_loss` resolves with `router_aux_loss_coef > 0`, rather than collapsing routing silently.

With an EXPLICIT mode, `output_router_logits=True` stays on where inert, so `MoEMetricsCallback` still reports expert load — you observe imbalance without correcting it (the warning says the flag was left on). Use `bias_update` on a family whose EP wrappers support it, or `none` to opt out cleanly. `none` is a true no-op: it neither zeroes a `router_aux_loss_coef` set directly in `model_init_kwargs` nor turns router logits on, so a model that balances itself keeps doing so.

Inertness is a **trainer-class contract**, not a per-model table. `DistributedTrainerMixin._consumes_router_aux_loss` (default `False`; `True` on `DistributedSFTTrainer` and `DistributedKTOTrainer`, and set back to `False` by the self-distillation subclass) declares whether the objective goes through a `labels` forward, and `_validate_router_aux_loss_consumable` acts on it at construction. `src/distributed/pipeline_parallel/split.py` enforces the same declaration for a PP stage.

Both branches fire only when the model carries a live aux term — `output_router_logits` on **and** `router_aux_loss_coef > 0`. An explicit `moe_balancing: aux_loss` on a non-consuming trainer then **raises**, and an `auto` resolution landing on `aux_loss` **warns and turns `output_router_logits` back off**, since the `[batch·seq, num_experts]` plane per MoE layer would be pure waste. With no usable coef there is nothing to waste, so the mode is left inert without either.

### No balancing at all under weight-sync RL

Online and environmental GRPO set both `policy_gradient_loss=True` and `syncs_to_external_generator=True`. The first makes `aux_loss` inert (above); the second downgrades both bias modes → `none`, because the routing bias is never in the sync payload (parameters only — an adopted native slot is a buffer, a transient bias a plain attribute), and keeping it would drift trainer routing away from the generator that produced the trajectories.

The two together leave those runs with **no router balancing on any family**, `router_balancing_rate` unreachable, and Zaya / DeepSeek-V4 — which balance only via `bias_update` — training with unbalanced experts. `build_perf_callbacks` logs this once at setup, and `validate_weight_sync_support` backstops the seam at trainer construction: a hand-built driver that enabled balancing itself and then configured weight sync is refused there. Monitor `moe/*` load metrics and keep RL phases short relative to the SFT stage that established the routing.

### VariableSchedulerCallback

Schedules any numeric attribute on the model over training. Source: `src/callbacks/variable_scheduler.py`.

```python
VariableSchedulerCallback(
    attribute_name="min_log_prob",   # e.g. Offline GRPO; or "target_margin" for SMPO
    initial_value=-2.0,
    final_value=-5.0,
    schedule_type="cosine",          # "cosine" | "linear" | "exponential"
    warmup_steps=100,
)
```

The attribute lands on the model the Trainer passes into the callback (unwrapped past a DDP/FSDP
`.module`), so the value the loss reads is the scheduled one. Formulas, with `progress` 0→1 over post-warmup steps (held at `initial_value` during warmup):

- Cosine: `final + 0.5 * (initial - final) * (1 + cos(pi * progress))`
- Linear: `initial + (final - initial) * progress`
- Exponential: `initial * (final / initial) ^ progress` (requires positive values)

## Relevant YAML

```yaml
generate_eval_examples: true      # GenerateExamplesCallback during eval
num_eval_examples: 50

enable_efficiency_metrics: false  # off by default (multi-sequence trainers underreport MFU)
enable_moe_metrics: true          # on by default (no-op for dense models)
report_mfu_diagnostics: false     # MFU/S-MFU/TFLOPS to headline log; tokens/s/GPU always on
num_full_model_params: null       # optional full-model param count for distributed efficiency

moe_balancing: auto               # auto | none | aux_loss | bias_update | bias_update_transient
router_balancing_rate: 1.0e-3     # gamma; only used when a bias-update mode is active

enable_torch_profiler: false      # TorchProfilerCallback — see reference/debugging.md

report_to: wandb
logging_steps: 1                  # also the compute period for the moe/* summary (both MoE callbacks)
```

### Why reorder integration callbacks

HF Trainer puts `WandbCallback` / `TensorBoardCallback` / `MLflowCallback` / `CometCallback` / `ClearMLCallback` **before** user callbacks, and `CallbackHandler.on_log` fires in list order — so the integrations ship the `logs` snapshot to their backends before any user callback mutates it, and every key added via `logs.update(...)` is computed but never reaches the dashboards (`moe/*` from both MoE callbacks, and everything `EfficiencyCallback` adds).

`reorder_integration_callbacks_last(trainer)` moves any callback whose class lives under `transformers.integrations` to the tail, preserving relative order. Idempotent. `run_trainer` calls it for every toolkit script; add it to any new script or notebook that builds a Trainer by hand.

## Writing custom callbacks

Extend `transformers.TrainerCallback` and override hooks. Each hook receives `args`, `state`, `control`, and `**kwargs` — HF's `CallbackHandler` passes `model`, `processing_class`, `optimizer`, `lr_scheduler`, `train_dataloader`, `eval_dataloader` (no `trainer`).

```python
class LossSpikeCallback(TrainerCallback):
    def __init__(self, max_loss: float = 10.0):
        self.max_loss = max_loss

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and logs.get("loss", 0) > self.max_loss:
            control.should_training_stop = True
```

Influence the loop via `control`: `should_training_stop`, `should_evaluate`, `should_save`, `should_log`.

Distributed rules: gate prints/logs on `is_global_main_process()` (`src.distributed.runtime`); unwrap FSDP/DDP (`while hasattr(model, "module"): model = model.module`) before calling `.generate()` or custom methods; wrap collective ops in `barrier()`.

To reuse a callback across scripts, add it to `src/callbacks/` and import it from the module that defines it — package inits re-export nothing. For the built-in perf / MoE callbacks, prefer the YAML flags above over hand-wiring.
