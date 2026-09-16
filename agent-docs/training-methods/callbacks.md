# Training Callbacks

Callbacks extend `transformers.TrainerCallback` to log metrics, generate samples, schedule hyperparameters and balance MoE routers. Source: `src/callbacks/`.

Scripts assemble them in `build_training_callbacks(...)` (`src/training/script_runner.py`), which delegates the efficiency / MoE / profiler callbacks and the `moe_balancing` resolution to `build_perf_callbacks(...)` (`src/callbacks/wiring.py`). `run_trainer` then reorders HF's report-to integrations last: they fire before user callbacks, so otherwise every key added via `logs.update(...)` is lost.

## Built-in callbacks

| Callback | What it does | Enable |
|---|---|---|
| `ParameterStatsCallback` | Param counts, modules, dtypes at train begin | Always, on toolkit scripts |
| `GenerateExamplesCallback` | Eval-time sample generations | `generate_eval_examples: true` |
| `EfficiencyCallback` | tokens/s/GPU, step time, memory | `enable_efficiency_metrics: true` |
| `MoEMetricsCallback` | `moe/*` per-expert load | `enable_moe_metrics: true` (default) |
| `RouterBiasBalancingCallback` | DeepSeek-V3 bias update, plus `moe/*` | a `bias_update` mode |
| `TorchProfilerCallback` | Per-rank traces, memory timeline | `enable_torch_profiler: true` ([Debugging](../reference/debugging.md)) |
| `VariableSchedulerCallback` | Schedules a model attribute | Manual |

### ParameterStatsCallback

Prints parameter counts, a per-module breakdown (numeric layer indices collapsed to `X`, so identical layers group) and a dtype / memory table on `on_train_begin`, main process only. Every count is **this rank's local** share (`local_numel`): the callback runs after wrapping, where a plain `numel()` reports FSDP2/TP DTensors globally and FSDP-ignored EP expert tensors locally — two scopes silently summed.

### GenerateExamplesCallback

Generates text samples on `on_evaluate` and logs them as a table — `Prompt`, `Completion`, and `Chosen` / `Rejected` on preference rows — to the first backend in `report_to` (`wandb` or `clearml`).

Scripts build it through `GenerateExamplesCallback.from_config(...)`, which returns `None` unless `generate_eval_examples` is set and the generate split carries tokenized `input_ids` — the VLM path has none. That flag defaults to `true` on DPO / SMPO / offline GRPO, `false` on SFT.

DDP splits the samples across ranks and gathers on rank 0; FSDP2 and EP/ETP generate on every rank together, forced greedy, since divergent EOS timing would hang the collective. TP, CP and PP are **skipped** — sharded DTensors, Ulysses attention and partial stages cannot decode. Gradient checkpointing is restored module by module.

### EfficiencyCallback

Tracks token throughput, step time and GPU memory; tokens/s/GPU is the headline. MFU, S-MFU and achieved TFLOPS are computed every step but reach the log only under `report_mfu_diagnostics: true`, staying readable off `efficiency_cb.mfu` / `.smfu` / `.tps` / `.memory`. The first two steps are warmup.

| Key (each as `step_*` and `avg_*`) | Meaning |
|---|---|
| `tokens_per_second` | This GPU's tokens over step compute time — the headline |
| `cluster_tokens_per_second` | Unique cluster tokens: per-GPU × `data_parallel_size` × `cp_size` |
| `step_time_seconds` | Optimizer-step wall clock |
| `mfu_percent`, `tflops_per_sec` | Against the device peak in `src/hardware.py`; `0.0` for a pair it lacks |
| `smfu_percent`, `smfu_tflops_per_sec` | MoE-aware twin, crediting only the experts that fire; sparse models only |

Memory keys take no prefix: `allocated_gb`, `reserved_gb`, `peak_allocated_gb`, `training_peak_allocated_gb`; `num_full_model_params` adds `distributed_efficiency`.

Off by default: utilization misleads for multi-sequence trainers — DPO / SMPO / reward forward both chosen and rejected, distillation also the teacher. Turning it on sets `include_num_input_tokens_seen="all"`, whose exact accounting costs a world all-gather plus a host sync per micro-batch; a warning fires from world size 64 up.

The peak is keyed on `lowp_precision` where low-precision compute is on, since bf16 masters name a peak the GEMMs never run at. S-MFU scales the expert term by `(top_k / num_experts) × ep_size`; `expert_tp_size` already divides the local expert params.

Throughput is measured only where the collator emits `input_ids`. SMPO and embedding emit none, so every step falls back to a padded-length estimate (`per_device_train_batch_size × gradient_accumulation_steps × max_seq × 0.8`) behind one warning — a figure that reads as measured but is not.

Three cumulative token counters nest: `num_input_tokens_seen` (HF's, padding included), `num_attended_tokens_seen` (SFT only, de-duplicated across TP/ETP/CP) and `num_unmasked_output_tokens_seen` (`labels != -100`, or the completion mask). Only the first survives a resume, via `trainer_state.json`; it also drives throughput, so tokens/s counts padding.

### MoEMetricsCallback

Tracks per-expert load for any MoE model with a declared router. A forward hook on each router counts the experts it selected, accumulates across micro-batches, all-reduces and injects `moe/*` keys. On by default, and a no-op for dense models.

| Metric (`moe/`) | Meaning |
|---|---|
| `load_max` | Max per-expert share, normalized to uniform routing (>1 = hot expert) |
| `load_min` | Min per-expert share (→0 = dead or cold expert) |
| `load_cv` | Coefficient of variation across experts; 0 = perfect balance |
| `dead_frac` | Fraction of experts that got zero tokens this step |
| `load_max_first` / `load_max_last` | Max-load ratio for the first / last MoE layer |

Counted is the router's **own** selection, not `topk(router_logits)`: a DeepSeek-style router selects over biased scores inside an `n_group` mask while publishing raw logits. Routers come from transformers' `_can_record_outputs["router_logits"]` declaration; one returning logits alone re-ranks them and warns once. Counters zero every step, and `log_every_n_steps` follows `logging_steps`.

It emits nothing in three cases:

- Under either bias-update mode — `RouterBiasBalancingCallback` emits the same keys from its load counter.
- Under pipeline parallelism, whose stage forward returns a bare tensor, not a `ModelOutput`; `bias_update` is the only route to `moe/*` there.
- Where `output_router_logits` is off, stamped forced-off by the balancing strategy or just unset. The flag costs a `[B*S, num_experts]` tensor per MoE layer per forward **and** adds the aux loss, so metrics never enable it; opt in via `model_init_kwargs`.

Two live paths bypass the hooks even when wired: a wrapper routing without the HF router module (`_ep_severs_aux_loss` — Bailing/Ling, Inkling, DeepSeek-V4), and a GRPO trainer taking log-probs from the backbone alone. Both warn once after the first step; use `bias_update` there.

### RouterBiasBalancingCallback

DeepSeek-V3 auxiliary-loss-free router balancing (`src/callbacks/router_bias_balancing.py`). The bias is added to the *detached* top-k scores, so gradients flow through un-biased gating weights and quality matches no-balancing.

The rule is `b_i ← b_i + γ · sign(mean_count − count_i)` over tokens routed to expert *i* in the last step, summed across ranks. It runs every step; only the `moe/*` summary is periodic, since it ends in a host sync. Defaults: `update_rate=1e-3`, `log_every_n_steps=1`, `exclude_last_slot` off the router class attribute `_has_discard_expert_slot`.

The mode holds an **export contract**: `bias_update` is accepted only where the trained bias lands in checkpoint-exported state a serving engine loads, so trainer and server route identically. A family satisfies it with a native buffer, an adopted native slot (an `nn.Parameter` re-registered as a persistent buffer under the same key, frozen out of gradient training), or a materialized zero buffer plus its config flag.

| Family | Exported balancing slot |
|---|---|
| Zaya | Native persistent `balancing_biases` buffer, in every parallelism mode |
| GPT-OSS | `router.bias`, the hub router's own logit-space bias |
| GLM-4, Laguna, GLM-5 Next, Inkling, DeepSeek-V4, Step-3.7 | `gate.e_score_correction_bias` (Laguna's hub spelling: `experts.e_score_correction_bias`) |
| Bailing / Ling | The hub gate's persistent `expert_bias`, adopted whole — no side-buffer fallback |
| LFM-2 | The block's `expert_bias`, materialized with `use_expert_bias` flipped on for the export |
| Qwen3, Qwen3.5/3.6, Mistral4, Cohere2 MoE | None — `bias_update_transient` only |
| Gemma 4 | None, and no balancing route at all ([below](#gemma-4-has-no-balancing-path)) |

The contract is checked on the **enabled tree**, not the class declaration: a renamed upstream slot, or one already FSDP-sharded as a DTensor, falls back to the side-buffer and `bias_update` then raises. Bailing has no fallback at all (`_supports_transient_balancing_bias = False`). A class declaring `_SERVED_BALANCING_BIAS_DROPPED_BY` (Laguna, vLLM 0.26.0) warns that the engine's loader skips the exported key, so a copy served there routes on the pretrained bias.

> [!WARNING]
> **Scale the rate with the injection space**
>
> The side-buffer families add the bias to **softmax probabilities** of scale `1/num_experts`: the 1e-3 default is one sign step of ~50% of uniform at 512 experts, and the callback warns past 25%. GPT-OSS's adopted `router.bias` is in **logit space**, where 1e-3 is sub-ULP in bf16 — raise it toward 1e-2 and watch `moe/load_cv`.

**Requires EP patching** — `ep_group_size > 1 or use_grouped_gemm`, so `expert_parallel_size=1` still qualifies with grouped GEMM on. Without those wrappers, and on a dense model, an explicit `bias_update` raises rather than balancing nothing, except on routers with a native buffer (Zaya).

### VariableSchedulerCallback

Schedules any numeric attribute on the model over training (`src/callbacks/variable_scheduler.py`) — `min_log_prob` for offline GRPO, `target_margin` for SMPO.

```python
VariableSchedulerCallback("min_log_prob", -2.0, -5.0, schedule_type="cosine", warmup_steps=100)
```

The attribute lands on the model the Trainer passes in, unwrapped past a DDP/FSDP `.module`, so the value the loss reads is the scheduled one. It holds at `initial_value` through warmup, then runs the `cosine`, `linear` or `exponential` curve over the remaining steps.

An unknown schedule type or an exponential one with a non-positive start or end raises at construction, `warmup_steps ≥ max_steps` at train begin — never on the first scheduled step.

## MoE balancing modes

```yaml
moe_balancing: auto              # auto | none | aux_loss | bias_update | bias_update_transient
router_balancing_rate: 1.0e-3    # gamma; only used when a bias-update mode is active
```

`auto` (the default) resolves per model, [below](#auto-resolution-per-family); the four explicit modes are:

| `moe_balancing` | Effect | `router_aux_loss_coef` | `output_router_logits` | Callback |
|---|---|---|---|---|
| `bias_update` | Sign update into the family's own checkpoint slot; **raises** on any router left on the side-buffer | **forced 0** | **forced False** | `RouterBiasBalancingCallback` |
| `bias_update_transient` | The same update in a trainer-only side-buffer; **raises** where every router exports natively | **forced 0** | **forced False** | `RouterBiasBalancingCallback` |
| `aux_loss` | The model's own switch-style aux loss | preserved | **True** only where the term reaches the loss | none |
| `none` | No intervention; a self-balancing model keeps doing so | preserved | preserved | none |

Both bias modes force `output_router_logits=False`, overriding `model_init_kwargs`: the EP bias path bypasses the HF router module the recorder hooks, so leaving it on returns an empty `router_logits` tuple and `load_balancing_loss_func` raises `IndexError`. These writes are **run-scoped** — every exported `config.json` carries the hub's coefficient and flag.

`aux_loss` sets `output_router_logits=True` only where `router_aux_loss_coef > 0`; without a usable coef, or where the EP wrappers sever the aux path (DeepSeek-V4), it leaves the flag off and warns rather than letting TRL read an `outputs.aux_loss` those models never populate. With a usable coef it **raises** where the `forward` declares no `output_router_logits` parameter — HF's config fallback lives on that parameter, and `Qwen3_5MoeForConditionalGeneration` reads the flag from `kwargs` only.

The resolved mode is reconciled world-wide with precedence `bias_update > bias_update_transient > aux_loss > none`: under PP a stage holding no MoE layer resolves `auto` out of ignorance, and a split verdict would send stages into different collectives.

### `auto` resolution per family

`auto` wants `bias_update` on three signals: a native `balancing_biases` router (Zaya); EP wrappers that sever the aux-loss path (DeepSeek-V4, Bailing, Inkling); or a `forward` taking no `output_router_logits` while an EP wrapper accepts the bias.

It **commits** only where the bias would land in exported state. GLM-4 MoE Lite, LFM-2 and Step-3.7 qualify on the third signal — none of their modelings has aux-loss machinery. GLM-5 Next does not: its forward honors the flag, so `auto` gives `aux_loss` while the exported slot still serves an explicit `bias_update`, the Laguna shape.

**Mistral4, Cohere2 MoE and multimodal Qwen3.5/3.6** have no slot and resolve to `none`, warning that `bias_update_transient` is the deliberate opt-in.

`aux_loss` follows for the families whose `forward` declares the parameter: GPT-OSS, Qwen3, Laguna, GLM-5 Next, the text-only `Qwen3_5MoeForCausalLM`. Where it does not and no wrapper accepts a bias either, the model has **no balancing route at all**, `auto` resolves to `none` with the reason, and an explicit `aux_loss` raises.

That is also what `use_grouped_gemm: false` at `ep_size = 1` does to GLM-4 MoE Lite, LFM-2 and Step-3.7: with no wrappers to carry the bias, the run trains unbalanced.

### Gemma 4 has no balancing path

`auto` resolves Gemma 4 to `none` with a warning: its EP wrapper accepts no bias, no router ships a `balancing_biases` buffer, and its `forward` declares no `output_router_logits` parameter. An explicit `bias_update` raises, because the router lives in the parent decoder layer rather than the EP module. An explicit `aux_loss` finds no `router_aux_loss_coef` either, so it stamps the flag forced-off and warns rather than raising.

Either way Gemma 4 gets no balancing **and** no `moe/*` metrics, and `output_router_logits: true` in `model_init_kwargs` cannot buy them back — no Gemma 4 config level declares the field, so the override is rejected at load. Every shipped Gemma 4 config sets `moe_balancing: none`.

### Where `aux_loss` is inert

`*ForCausalLM.forward` adds `router_aux_loss_coef * load_balancing_loss_func(...)` **only when `labels is not None`**, so `aux_loss` balances experts under SFT and pretraining. KTO reaches it another way: TRL's `KTOTrainer` sets the flag and adds the term itself. Everywhere else the term never reaches the loss:

- **DPO / SMPO / reward** — the forward runs without `labels`, and none of the three add the term.
- **Distillation (teacher and self)** — both strip `labels`, for full-vocab logits.
- **Classification / embedding** — non-causal heads; the causal-LM forward never runs.
- **GRPO (offline / online / async)** — the loss is per-token log-probs, so the mode warns and leaves the flag off.
- **Pipeline parallelism** (itself [not yet available](../parallelism/pipeline-parallelism.md)) — a stage would apply the head itself; the shipped split gate **raises** when `aux_loss` resolves with a positive coefficient.

Inertness is a **trainer-class contract**: `_consumes_router_aux_loss` declares whether the objective goes through a `labels` forward (`True` only on `DistributedSFTTrainer` and `DistributedKTOTrainer`). An explicit `aux_loss` on a non-consuming trainer **raises**; an `auto` resolution landing there turns `output_router_logits` back off.

Online and async GRPO go further: syncing weights to a live rollout engine downgrades **both bias modes to `none`**, since the sync ships parameters only — an adopted native slot is a buffer, a transient bias a plain attribute — and a bias the generator never sees drifts trainer routing off the trajectories it produced.

With `aux_loss` inert under the same loss, those runs have **no router balancing on any family**, and Zaya / DeepSeek-V4 train unbalanced.

### The balancing sidecar

Every bias mode checkpoints its biases to `router_balancing_biases.pt` on the FS-aware save rank and restores them on resume; under PP the per-stage names go through `global_parameter_name` and merge on rank 0.

The restore is all-or-nothing across ranks: a missing or torn file on some ranks raises, as does a saved bias whose shape does not match the live router, rather than being `copy_`-broadcast into it. A checkpoint without the file keeps zero-init biases and warns; a sidecar matching **no** live router drops every trained bias, loudly — what a weight-sync RL leg does to a `bias_update` checkpoint.

`scripts/after_training/convert_to_bf16.py` and the `merge_peft_adapters.py` merge path apply the sidecar into the model's native slots, materializing a config-gated slot the base lacks — a PEFT merge starts from base weights that never saw the sign updates. Saves keep balancing tensors at their trained fp32; a bf16 round trip would quantize away several 1e-3 steps.

## Relevant YAML

```yaml
generate_eval_examples: true      # + num_eval_examples: 50
enable_efficiency_metrics: false  # + report_mfu_diagnostics, num_full_model_params
enable_moe_metrics: true
moe_balancing: auto
router_balancing_rate: 1.0e-3     # gamma, bias-update modes only
logging_steps: 1                  # also the moe/* summary period
```

Defaults and full descriptions: [Performance & balancing flags](../reference/configuration-reference.md#performance-balancing-flags).

## Writing custom callbacks

Extend `transformers.TrainerCallback` and override hooks. Each receives `args`, `state`, `control` and `**kwargs` — HF's `CallbackHandler` passes `model`, `processing_class`, `optimizer`, `lr_scheduler`, the two dataloaders, and no `trainer`.

```python
class LossSpikeCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and logs.get("loss", 0) > self.max_loss:
            control.should_training_stop = True   # also: should_evaluate/save/log
```

Distributed rules: gate prints on `is_global_main_process()` (`src/distributed/runtime.py`); unwrap FSDP/DDP before `.generate()`; wrap collectives in `barrier()`, keeping any participation gate rank-uniform. A reusable callback goes in `src/callbacks/` and is imported from its own module — the package init re-exports nothing.
