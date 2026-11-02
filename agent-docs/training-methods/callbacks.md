# Training Callbacks

Callbacks extend `transformers.TrainerCallback` to log metrics, generate samples, schedule hyperparameters and balance MoE routers. Source: `halo/callbacks/`.

Scripts assemble them in `configure_balancing_and_build_callbacks(...)` (`halo/entrypoints/script.py`): `configure_moe_balancing` (`halo/callbacks/balancing.py`, collective) resolves, agrees and applies `moe_balancing`, then `build_run_callbacks` (`halo/callbacks/factory.py`) builds the roster for that mode. `run_trainer` then reorders HF's report-to integrations last: they fire before user callbacks, so otherwise every key added via `logs.update(...)` is lost.

## Built-in callbacks

| Callback | What it does | Enable |
|---|---|---|
| `ParameterStatsCallback` | Param counts, modules, dtypes at train begin | Always, on toolkit scripts |
| `GenerateExamplesCallback` | Eval-time sample generations | `generate_eval_examples: true` |
| `EfficiencyCallback` | tokens/s/GPU, step time, memory | `enable_efficiency_metrics: true` |
| `MoEMetricsCallback` | `moe/*` per-expert load | `enable_moe_metrics: true` (default) |
| `BalancingBiasCallback` | DeepSeek-V3 bias update, plus `moe/*` | a `bias_update` mode |
| `TorchProfilerCallback` | Per-rank traces, memory timeline | `enable_torch_profiler: true` ([Debugging](../reference/debugging.md)) |
| `CheckpointingPlanCallback` | Relaxes gradient checkpointing to what the memory budget needs | `gradient_checkpointing_plan: auto` |
| `VariableSchedulerCallback` | Schedules a model attribute | Added by SMPO (`use_margin_schedule`) and offline GRPO (`initial_min_log_prob`) |

### ParameterStatsCallback

Prints parameter counts, a per-module breakdown (numeric layer indices collapsed to `X`, so identical layers group) and a dtype / memory table on `on_train_begin`, main process only. Every count is **this rank's local** share (`local_numel`): the callback runs after wrapping, where a plain `numel()` reports FSDP2/TP DTensors globally and FSDP-ignored EP expert tensors locally — two scopes silently summed.

### GenerateExamplesCallback

Generates text samples on `on_evaluate` and logs them as a table — `Prompt`, `Completion`, and `Chosen` / `Rejected` on preference rows — to the first backend in `report_to` (`wandb` or `clearml`).

Scripts build it through `GenerateExamplesCallback.from_config(...)`, which returns `None` unless `generate_eval_examples` is set and the generate split carries tokenized `input_ids` — the VLM path has none. That flag defaults to `true` on DPO / SMPO / offline GRPO, `false` on SFT.

DDP splits the samples across ranks and gathers on rank 0; FSDP2 and EP/ETP generate on every rank together, forced greedy, since divergent EOS timing would hang the collective. TP, CP and PP are **skipped** — sharded DTensors, Ulysses attention and partial stages cannot decode. Gradient checkpointing is restored module by module.

### EfficiencyCallback

Tracks token throughput, step time and GPU memory; tokens/s/GPU is the headline. MFU, S-MFU and achieved TFLOPS are computed every step but reach the log only under `report_mfu_diagnostics: true`, staying readable off `efficiency_cb.mfu` / `.smfu` / `.tps` / `.memory`, with `.measured_step_times` holding every measured step's unrounded time in seconds. The first two steps are warmup.

| Key (each as `step_*` and `avg_*`) | Meaning |
|---|---|
| `tokens_per_second` | `cluster_tokens_per_second` / world size — the headline, comparable across every parallel layout |
| `cluster_tokens_per_second` | Unique cluster tokens over step compute time: each data-parallel replica's batch once, however many TP, expert-TP, CP or pipeline ranks share it |
| `step_time_seconds` | Optimizer-step wall clock between two device synchronizes, so it times the step's kernels, not their launch |
| `mfu_percent`, `tflops_per_sec` | Against the device peak in `halo/hardware.py`; `0.0`, with a startup warning, for an undetected GPU or a GPU/precision pair it lacks |
| `smfu_percent`, `smfu_tflops_per_sec` | MoE-aware twin, crediting only the experts that fire; sparse models only |

Memory keys take no prefix: `allocated_gb`, `reserved_gb`, `peak_allocated_gb`, `training_peak_allocated_gb`; `num_full_model_params` adds `distributed_efficiency`.

Off by default: utilization misleads for multi-sequence trainers — DPO / SMPO / reward forward both chosen and rejected, distillation also the teacher. Turning it on sets `include_num_input_tokens_seen="all"`: each micro-batch adds its input shape on the host, and the callback's step end sums the step over the world in the same host read as the step's attention work (HF's own count gathers and syncs on every micro-batch). Without the callback the count is summed at each log and before each save, so it is exact in the logs and in `trainer_state.json` but lags between them in the state a callback sees.

The peak is keyed on `lowp_precision` where low-precision compute is on, since bf16 masters name a peak the GEMMs never run at. S-MFU scales the expert term by `(top_k / num_experts) × ep_size`; `expert_tp_size` already divides the local expert params.

Throughput is measured only where the collator emits `input_ids`. SMPO and embedding emit none, so every step falls back to a padded-length estimate (`per_device_train_batch_size × gradient_accumulation_steps × max_seq × 0.8`) behind one warning — a figure that reads as measured but is not.

Three cumulative token counters nest: `num_input_tokens_seen` (HF's, padding included), `num_attended_tokens_seen` (SFT, DPO, KTO and reward, each batch counted once across its TP/ETP/CP group) and `num_unmasked_output_tokens_seen` (`labels != -100`, or the completion mask). Only the first survives a resume, via `trainer_state.json`; it also drives throughput, so tokens/s counts padding.

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

- Under either bias-update mode — `BalancingBiasCallback` emits the same keys from its load counter.
- Under pipeline parallelism, where it is not wired: the stage forwards return bare tensors without `router_logits`, so `bias_update` is the route to load metrics there.
- Where `output_router_logits` is off, stamped forced-off by the balancing strategy or just unset. The flag costs a `[B*S, num_experts]` tensor per MoE layer per forward **and** adds the aux loss, so metrics never enable it; opt in via `model_init_kwargs`.

Two live paths bypass the hooks even when wired: an EP layer routing without the HF router module (`SEVERS_AUX_LOSS` — Bailing/Ling, Inkling, DeepSeek-V4), and a GRPO trainer taking log-probs from the backbone alone. Both warn once after the first step; use `bias_update` there.

### BalancingBiasCallback

DeepSeek-V3 auxiliary-loss-free router balancing (`halo/callbacks/balancing.py`). The bias is added to the *detached* top-k scores, so gradients flow through un-biased gating weights and quality matches no-balancing.

The rule is `b_i ← b_i + γ · sign(mean_count − count_i)` over tokens routed to expert *i* in the last step, summed across ranks. It runs every step; only the `moe/*` summary is periodic, since it ends in a host sync. Defaults: `update_rate=1e-3`, `log_every_n_steps=1`, `exclude_last_slot` off the router class attribute `_has_discard_expert_slot`.

The mode holds an **export contract**: `bias_update` is accepted only where the trained bias lands in checkpoint-exported state a serving engine loads, so trainer and server route identically. A family satisfies it with a native buffer, an adopted native slot (an `nn.Parameter` re-registered as a persistent buffer under the same key, frozen out of gradient training), or a materialized zero buffer plus its config flag.

| Family | Exported balancing slot |
|---|---|
| Zaya | Native persistent `balancing_biases` buffer, in every parallelism mode |
| GPT-OSS | `router.bias`, the hub router's own logit-space bias |
| GLM-4, Laguna, GLM-5 Next, Inkling, DeepSeek-V4, Step-3.7 | `gate.e_score_correction_bias` (Laguna's hub spelling: `experts.e_score_correction_bias`) |
| Bailing / Ling | The hub gate's persistent `expert_bias`, adopted whole — no side-buffer fallback |
| LFM-2 | The block's `expert_bias`, materialized with `use_expert_bias` flipped on for the export |
| Qwen3, Qwen3.5/3.6, Qwen3-Next, Qwen3.8-Flash-Next, Mistral4, Cohere2 MoE | None — `bias_update_transient` only |
| Gemma 4 | None, and no balancing route at all ([below](#gemma-4-has-no-balancing-path)) |

The contract is checked on the **enabled tree**, not the class declaration: a renamed upstream slot, or one already FSDP-sharded as a DTensor, falls back to the side-buffer and `bias_update` then raises. Bailing has no fallback at all (`_supports_transient_balancing_bias = False`). A family on a weight-sync client's `DROPPED_BALANCING_BIAS_MODEL_TYPES` (Laguna on vLLM) warns, naming that client's `ENGINE_RELEASE`, that the engine's loader skips the exported key, so a copy served there never routes on the trained bias.

> [!WARNING]
> **Scale the rate with the injection space**
>
> The side-buffer families add the bias to **softmax probabilities** of scale `1/num_experts`: the 1e-3 default is one sign step of ~50% of uniform at 512 experts, and the callback warns past 25%. GPT-OSS's adopted `router.bias` is in **logit space**, where 1e-3 is sub-ULP in bf16 — raise it toward 1e-2 and watch `moe/load_cv`.

**Requires EP patching** — `ep_group_size > 1 or use_grouped_gemm`, so `expert_parallel_size=1` still qualifies with grouped GEMM on. Without those EP layers, and on a dense model, an explicit `bias_update` raises rather than balancing nothing, except on routers with a native buffer (Zaya). `bias_update_transient` is refused under pipeline parallelism.

### CheckpointingPlanCallback

`gradient_checkpointing_plan: auto` (`halo/callbacks/checkpointing_plan.py`, policy math in `halo/models/checkpointing_plan.py`) starts every launch fully checkpointed and measures its step 2, the first at the steady-state footprint (step 1 allocates the optimizer moments and runs the JIT warm-ups). From that one step it predicts the peak of every lighter policy (no checkpointing, then `every_n_layers` 8, 4, 3 and 2, each checkpointing more layers than the last) and applies the lightest whose prediction fits 85% of device memory (75% under EP, whose dropless routing moves the hot rank's receive count between steps).

- **The layers** are the text decoder list. A multimodal checkpoint's vision blocks checkpoint too, but a text run never calls them, so they keep their flags and are not counted.
- **The readings:** the peak before each micro-step's first backward recompute, the peak before the optimizer step, the step peak, the allocation one layer's recompute leaves (its uncheckpointed activations) and a checkpoint's input, all host-side allocator stats. Early stop is off for that step, so a non-reentrant recompute runs to its layer's end. The optimizer phase is the same under every policy, so only the forward and backward phases grow.
- **The budget** is taken on the world's smallest device, net of the memory outside the allocator (the CUDA context, DeepEP and NCCL buffers), and scaled by the allocated-to-reserved ratio of the most fragmented rank at the measured step, since a device holds what its caching allocator reserves.
- **One plan for the world:** the ranks' fingerprints must agree at startup, the readings are reduced in one all-reduce (MAX of the peaks, the recompute, the reserved-to-allocated ratio and the non-allocator bytes, MIN of the input) before the choice, and the resolved plan is checked for agreement, raising on every rank on a divergence.
- **The switch:** only the decoder layers' `gradient_checkpointing` flags move, by transformers' `index % k == 0` rule; the installed checkpoint function (the scoped, reentrant one under EP/CP/MoE) stays. A policy replays the same forward, so losses and gradients match full checkpointing bitwise where the kernels are deterministic.
- **The record and its floor:** `_halo_run_plan.json` in `output_dir` (from the save rank or ranks, like `run.log`) holds the applied plan, whether the run finished training on it, and its fingerprint: model type, decoder-layer count, hidden and vocab size, attention backend, `max_length`, batch size and accumulation, world size and parallel layout, GPU model and memory, and the torch and transformers versions. It is written unfinished when the plan is applied and marked finished at train end. The next launch into that `output_dir`, a resume or a relaunch after a crash before the first checkpoint, still measures, but checkpoints at least one step more than an unfinished plan (an OOM, a crash or a preemption), so a plan never replays into the same OOM; full checkpointing stays full. A finished plan, or one made for other inputs, sets no floor. On a per-node filesystem each rank reads its own node's copy and the world takes the highest floor. Delete the file on every node to plan without a floor; a timestamped `output_dir` (strftime directives) starts a new directory, so it gets no floor. Exports and Hub pushes leave every top-level `_*` entry behind, this one included.
- **Rows:** only packed rows qualify, where every micro-batch carries up to `per_device_train_batch_size` × `max_length` tokens: SFT with `packing: true`, or a corpus pre-processed packed (the plan reads the rows the collator builds, not the flag). The measured step must hold the run's largest micro-batch: on rows short against `max_length` nearly every pack is full, but packs that fill unevenly (documents near `max_length` under `bfd`) can measure low over few micro-batches, and an OOM then floors the next launch.
- **Refused** at startup under pipeline parallelism, `torch_compile`, `auto_find_batch_size` (it answers an OOM by halving the batch), trainers that generate rollouts (TRL re-enables checkpointing around generation), rows the collator does not pack (every trainer but SFT), multimodal rows, activation offload, an explicit `every_n_layers`, text decoder layers that are not `GradientCheckpointingLayer` (model-level checkpointing), or with checkpointing off.

### VariableSchedulerCallback

Schedules any numeric attribute on the model over training (`halo/callbacks/variable_scheduler.py`) — `min_log_prob` for offline GRPO, `target_margin` for SMPO.

```python
VariableSchedulerCallback("min_log_prob", -2.0, -5.0)
```

The attribute lands on the model the Trainer passes in, unwrapped past a DDP/FSDP `.module`, so the value the loss reads is the scheduled one. It moves linearly from `initial_value` to `final_value` over `max_steps`, then holds; a run with no training steps raises at train begin.

## MoE balancing modes

```yaml
moe_balancing: auto              # auto | none | aux_loss | bias_update | bias_update_transient
router_balancing_rate: 1.0e-3    # gamma; only used when a bias-update mode is active
```

`auto` (the default) resolves per model, [below](#auto-resolution-per-family); the four explicit modes are:

| `moe_balancing` | Effect | `router_aux_loss_coef` | `output_router_logits` | Callback |
|---|---|---|---|---|
| `bias_update` | Sign update into the family's own checkpoint slot; **raises** on any router left on the side-buffer | **forced 0** | **forced False** | `BalancingBiasCallback` |
| `bias_update_transient` | The same update in a trainer-only side-buffer no export carries, so a served copy routes without it (near-tied top-k picks flip); **raises** where every router exports natively | **forced 0** | **forced False** | `BalancingBiasCallback` |
| `aux_loss` | The model's own switch-style aux loss | preserved | **True** only where the term reaches the loss | none |
| `none` | No intervention; a self-balancing model keeps doing so | preserved | preserved | none |

The coefficient is the model config's: a `router_aux_loss_coef` set on a TRL trainer config raises at construction (set it through `model_init_kwargs`), and TRL's own loss receives the mode's coefficient only in DPO and KTO ([Where `aux_loss` is inert](#where-aux_loss-is-inert)).

Both bias modes force `output_router_logits=False`, overriding `model_init_kwargs`: the EP bias path bypasses the HF router module the recorder hooks, so leaving it on returns an empty `router_logits` tuple and `load_balancing_loss_func` raises `IndexError`. These writes are **run-scoped** — every exported `config.json` carries the hub's coefficient and flag.

`aux_loss` sets `output_router_logits=True` only where `router_aux_loss_coef > 0`; without a usable coef, or where the EP layers sever the aux path (DeepSeek-V4), it leaves the flag off and warns rather than letting TRL read an `outputs.aux_loss` those models never populate. With a usable coef it **raises** where the `forward` declares no `output_router_logits` parameter — HF's config fallback lives on that parameter, so a forward that reads the flag from `kwargs` only never reaches the aux term. The probe reads the model under a PEFT or CP wrapper, both of which pass the flag through.

Under pipeline parallelism no stage runs the `*ForCausalLM.forward` that adds the term, so a mode resolving to `aux_loss` (`auto` included) at a positive `router_aux_loss_coef` raises at the split; balance with `bias_update` where the family exports it, or `none` ([PP + EP](../parallelism/pipeline-parallelism.md#pp--ep-moe)).

The SFT trainer weighs the term at `router_aux_loss_coef` once per optimizer step: it averages the window's micro-batch aux losses, and the gradient sync averages them over ranks. In each eval batch it adds the term once at that coefficient, so `loss` and `eval_loss` both read as the language-model mean plus `router_aux_loss_coef × aux_loss`.

HF's `num_items_in_batch` path multiplies every micro-batch loss, train and eval, by the process count and sums the accumulation window. That is right for the token-summed cross-entropy, but it would weigh the per-micro-batch aux mean `world_size × gradient_accumulation_steps` times its coefficient (`world_size` times in eval). `DistributedSFTTrainer._weigh_router_aux_loss` therefore rescales the term. When HF withholds the labels from the forward, the forward adds no aux term, so the trainer adds it itself. HF does this under a label smoother (`label_smoothing_factor`) or a TRL `compute_loss_func` (`loss_type: dft` with Liger off). Both this rescale and the CP model read the coefficient through `require_router_aux_loss_coef` (`halo/models/moe/config_fields.py`), which raises when the config declares none. CP computes its own mean and needs no rescale.

The resolved mode is reconciled world-wide with precedence `bias_update > bias_update_transient > aux_loss > none`, so every rank runs the same collectives.

### `aux_loss` under gradient checkpointing

Every MoE run with gradient checkpointing is reentrant, and a reentrant checkpoint runs a layer's original forward under `no_grad`. transformers collects `router_logits` in that pass, so on its own a checkpointed layer's aux term reaches the loss with no graph and adds no router gradient.

`aux_loss` therefore installs `install_router_aux_gradient` (`halo/models/moe/router_aux_gradient.py`) on the module collecting the logits, the innermost one declaring `router_logits` in `_can_record_outputs`. Each tensor collected under a checkpoint's `no_grad` pass becomes a carrier whose backward keeps the gradient the loss hands it. In the layer's recompute, an identity node on the output of the block owning the router passes that gradient to the router's recomputed logits.

The gradient is the family's own `load_balancing_loss_func`, differentiated by autograd: the load pooled over all layers, the attention mask, `router_aux_loss_coef` and the trainer's loss scaling apply exactly as without checkpointing, whichever consumer adds the term (the model forward, a fused-loss forward, the CP wrapper, TRL's KTO). The coefficient stays as configured, so the logged `loss` and `aux_loss` read as they do without checkpointing. Collected logits that carry a graph (no checkpointing, the layers `every_n_layers` leaves unchecked) keep the native path, so no term counts twice. EP needs no collective: each rank computes its aux loss from its own tokens, as HF does.

Every case the hooks cannot route raises instead of dropping the gradient: a kept gradient no recompute took by the end of the backward, one backward carrying the aux loss of two checkpointed forwards, a router run twice in one checkpointed forward or first seen after the model's first forward, recomputed logits of another shape, and an owning block that returns no hidden-state tensor. A model declaring no `router_logits` capture gets no hooks and a warning: its checkpointed layers' aux term reaches the logged loss without a router gradient.

### `auto` resolution per family

`auto` wants `bias_update` on three signals: a native `balancing_biases` router (Zaya); EP layers that sever the aux-loss path (DeepSeek-V4, Bailing, Inkling); or a `forward` taking no `output_router_logits` while an EP layer accepts the bias.

It **commits** only where the bias would land in exported state. GLM-4 MoE Lite, LFM-2 and Step-3.7 qualify on the third signal — none of their modelings has aux-loss machinery. GLM-5 Next does not: its forward honors the flag, so `auto` gives `aux_loss` while the exported slot still serves an explicit `bias_update`, the Laguna shape.

**Mistral4 and Cohere2 MoE** have no slot and resolve to `none`, warning that `bias_update_transient` is the deliberate opt-in.

`aux_loss` follows for the families whose `forward` declares the parameter: GPT-OSS, Qwen3, Qwen3.5/3.6 (multimodal and text-only), Qwen3-Next, Qwen3.8-Flash-Next, Laguna, GLM-5 Next. Where it does not and no EP layer accepts a bias either, the model has **no balancing route at all**, `auto` resolves to `none` with the reason, and an explicit `aux_loss` raises.

That is also what `use_grouped_gemm: false` at `ep_size = 1` does to GLM-4 MoE Lite, LFM-2 and Step-3.7: with no EP layers to carry the bias, the run trains unbalanced.

### Gemma 4 has no balancing path

`auto` resolves Gemma 4 to `none` with a warning: its EP layer accepts no bias, no router ships a `balancing_biases` buffer, and its `forward` declares no `output_router_logits` parameter. An explicit `bias_update` raises, because the router lives in the parent decoder layer rather than the EP module. An explicit `aux_loss` finds no `router_aux_loss_coef` either, so it stamps the flag forced-off and warns rather than raising.

Either way Gemma 4 gets no balancing **and** no `moe/*` metrics, and `output_router_logits: true` in `model_init_kwargs` cannot buy them back — no Gemma 4 config level declares the field, so the override is rejected at load. Every shipped Gemma 4 config sets `moe_balancing: none`.

### Where `aux_loss` is inert

`*ForCausalLM.forward` adds `router_aux_loss_coef * load_balancing_loss_func(...)` **only when `labels is not None`**, so `aux_loss` balances experts under SFT and pretraining, with or without [gradient checkpointing](#aux_loss-under-gradient-checkpointing). DPO and KTO reach it another way: TRL's `DPOTrainer` and `KTOTrainer` add the term themselves, at the coefficient the balancing mode put in effect. Everywhere else the term never reaches the loss:

- **SMPO / reward** — the forward runs without `labels`, and neither adds the term.
- **Distillation (teacher and self)** — both strip `labels`, for full-vocab logits.
- **Classification / embedding** — non-causal heads; the causal-LM forward never runs.
- **GRPO (offline / online / async)** — the loss is per-token log-probs, so the mode warns and leaves the flag off; with `output_router_logits` already on at a positive `router_aux_loss_coef`, an explicit `aux_loss` raises at construction (below).

Inertness is a **trainer-class contract**: `ROUTER_AUX_LOSS_ADDED_BY` declares what adds the aux term: the model's `labels` forward (`DistributedSFTTrainer`), TRL's own loss (`DistributedDPOTrainer`, `DistributedKTOTrainer`), or nothing (every other trainer). An explicit `aux_loss` on a non-consuming trainer **raises**; an `auto` resolution landing there turns `output_router_logits` back off.

Online and async GRPO go further: syncing weights to a live rollout engine downgrades **both bias modes to `none`**, since the sync ships parameters only — an adopted native slot is a buffer, a transient bias a plain attribute — and a bias the generator never sees drifts trainer routing off the trajectories it produced.

With `aux_loss` inert under the same loss, those runs have **no router balancing on any family**.

### The balancing sidecar

Every bias mode checkpoints its biases to `router_balancing_biases.pt` on the FS-aware save rank and restores them on resume.

The restore is all-or-nothing across ranks: a missing or torn file on some ranks raises, as does a saved bias whose shape does not match the live router, rather than being `copy_`-broadcast into it. A checkpoint without the file keeps zero-init biases and warns; a sidecar matching **no** live router drops every trained bias, loudly — what a weight-sync RL leg does to a `bias_update` checkpoint.

The PEFT merge (`merge_peft_adapters.py`) applies the sidecar into the model's native slots, materializing a config-gated slot the base lacks — a merge starts from base weights that never saw the sign updates. A full checkpoint already carries the biases in those slots, so `convert_to_bf16.py` reads them back from its shards at the trained fp32. Saves keep balancing tensors at their trained fp32; a bf16 round trip would quantize away several 1e-3 steps.

## Relevant YAML

```yaml
generate_eval_examples: true      # + num_eval_examples: 50
enable_efficiency_metrics: false  # + report_mfu_diagnostics, num_full_model_params
enable_moe_metrics: true
moe_balancing: auto
router_balancing_rate: 1.0e-3     # gamma, bias-update modes only
logging_steps: 1                  # also the moe/* summary period
```

Defaults and full descriptions: [Performance & balancing flags](../reference/configuration-reference.md#performance--balancing-flags).

## Writing custom callbacks

Extend `transformers.TrainerCallback` and override hooks. Each receives `args`, `state`, `control` and `**kwargs` — HF's `CallbackHandler` passes `model`, `processing_class`, `optimizer`, `lr_scheduler`, the two dataloaders, and no `trainer`.

```python
class LossSpikeCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and logs.get("loss", 0) > self.max_loss:
            control.should_training_stop = True   # also: should_evaluate/save/log
```

Distributed rules: gate prints on `is_global_main_process()` (`halo/runtime/ranks.py`); unwrap FSDP/DDP before `.generate()`; wrap collectives in `barrier()`, keeping any participation gate rank-uniform. A reusable callback goes in `halo/callbacks/` and is imported from its own module — the package init re-exports nothing.
