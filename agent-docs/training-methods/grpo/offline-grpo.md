# Offline GRPO

Offline GRPO trains on pre-collected, off-policy data — multiple completions per prompt with pre-computed rewards — with no generation during training. It needs at least 2 completions per prompt: a group of size 1 always yields advantage 0. `drop_degenerate_groups: true` (default off) drops such dead groups — exactly-tied rewards or fewer than two completions — at tokenization, before they spend forward compute or dilute the loss normalizer (near-ties are kept: the rank-based advantage methods train them at full scale).

| Aspect | Value |
|--------|-------|
| Trainer | `OfflineGRPOTrainer` (`src/trainers/grpo/offline.py`) |
| Config | `OfflineGRPOConfig` ([field reference](../../reference/configuration-reference.md#offlinegrpoconfig)) |
| Script | `scripts/training/offline_grpo.py` |
| Parallelism | EP, TP, EP+TP, EP+ETP, pure ETP (`ep_size=1`); **no CP** — the trainer relies on `logits_to_keep`, which CP's sequence splitting breaks, so `--context_parallel_size > 1` is rejected at config time (use [SMPO](../preference/smpo.md) for long sequences under CP). Declares `_supports_pp` — [PP](../../parallelism/pipeline-parallelism.md) is not yet available in this release |
| Reference model | Only when `kl_beta > 0` |

Use it when you have pre-collected multi-completion data with rewards, when reward computation is expensive enough to pre-compute, or when reproducibility matters. For on-policy generation use [Online GRPO](online-grpo.md); for pairwise chosen/rejected data use [SMPO](../preference/smpo.md) or DPO.

![Offline GRPO pipeline: pre-computed rewards feed advantage normalization, per-completion expansion preserves group_id, the MultiGroupSampler flattens completions in dataset order across DP ranks, then the training loop applies 1/group_size weighting and the chosen loss type](../../assets/diagrams/offline_grpo_pipeline.png)

## Objective

Advantages are normalized within each prompt's group during dataset preprocessing (`advantage_method`), raising the probability of high-reward completions and lowering it for low-reward ones. There is no `π_old` from the collection policy, so there is no importance-sampling ratio and no PPO-style clipping. `policy_gradient_formulation` sets the gradient weighting:

- `prob_weighted` (default): `loss = -(prob · advantage)` — high-probability tokens get larger gradients; conservative.
- `reinforce`: `loss = -(log_prob · advantage)` — equal weight per token; more aggressive, less stable.

### Advantage methods

| `advantage_method` | Formula | Range | Best for |
|--------------------|---------|-------|----------|
| `quantile_norm` (default) | inverse normal CDF of ranks | unbounded | outlier robustness, ordinal rewards |
| `z_norm` | `(r - mean) / (std + ε)` | unbounded | normal reward distributions |
| `minmax` | `2(r - min)/(max - min) - 1` | [-1, +1] | bounded advantages |
| `quantile_uniform` | uniform from ranks | [-1, +1] | max outlier robustness, ordinal rewards |
| `robust` | `(r - median) / IQR` | unbounded | extreme outliers |

`z_norm`'s `std` is the **sample** std (`ddof=1`) and `ε` is the `STD_EPS = 1e-4` the online and environmental z-norms use, so the same rewards produce the same advantage scale in all three trainers. The `"auto"` emphasis below reads the **population** std (numpy's default), unlike z_norm's `ddof=1` divisor — the two differ by `sqrt((n-1)/n)`, ~6% at a group of 8.

`best_completion_emphasis` (default `0.0`, off) boosts the best completion(s): a float above `1.0` (e.g. `2.0` for 2× weight), or `"auto"`, which adapts to reward variance as `3.0 + 2.0·std/(1.0 + std)` (3.0 low-variance to 5.0 high-variance). A value in `(0.0, 1.0]` raises at config time, since the consumer applies the factor only above `1.0` and it would be a silent no-op.

Every method's output — including the already-bounded `minmax` / `quantile_uniform` — is then clipped to `[-10, 10]`, because emphasis can push a boosted advantage past that range.

### Loss types

All three weight each example by `1/group_size`, so every prompt contributes equally regardless of how many completions it has. Gradient accumulation scales automatically (the trainer declares `_loss_is_own_mean`, which keeps HuggingFace's `/gradient_accumulation_steps` division); never divide by `gradient_accumulation_steps` yourself.

- `bnpo` (default): global weighted token average. Token-level like online GRPO's DAPO loss but not equivalent — this divides by the local micro-batch's group-weighted token sum, while DAPO divides by the global accumulated batch's active-token count and applies no per-group weighting.
- `grpo`: per-sequence average, then weighted mean across groups. Use when completion length varies.
- `dr_grpo`: normalized by effective group count × `max_completion_length`, a constant denominator that removes length bias. Here `max_completion_length` is a normalization constant, not a generation budget, so it must be set to a **positive** value — `dr_grpo` raises at trainer init on `null` and on any non-positive value alike.

### Stability

`min_log_prob` (default `-3.0`, ≈5% probability) clamps log-probs of low-probability tokens **only for negative-advantage examples**, preventing gradient explosion. Set `initial_min_log_prob` (e.g. `-1.5`) to linearly interpolate from a looser start to `min_log_prob` over training.

### Memory: chunked log-probs

The default log-prob path materializes `[B, T_completion, vocab]` logits (trimmed to the completion with `logits_to_keep`), and `kl_beta > 0` runs it twice per micro-batch — policy and reference. On wide vocabularies with long completions that allocation is the memory peak.

`use_chunked_grpo_logprobs: true` (default off) computes the same log-probs from the backbone's `last_hidden_state` via a vocab-chunked softmax instead, through the shared machinery of the on-policy trainers (`src/trainers/grpo/mixins/chunked_logprobs.py`) — covering the policy, the reference, and the PEFT `disable_adapter()` reference alike, with `min_log_prob` clamping applied identically on top of the resulting `(B, T)` log-probs. Mechanism and limits: [Environmental GRPO — Chunked log-probs](environmental-grpo.md#chunked-log-probs).

### Sampling

`MultiGroupSampler` (`src/trainers/grpo/mixins/dataloader.py`) yields per-completion indices for a wrapping `BatchSampler`, so `batch_size` is not a sampler parameter; the batches it feeds are padded by `OfflineGRPODataCollatorWithPadding` (`src/data/collators/offline_grpo.py`). It flattens groups in dataset order, gives each rank a contiguous slice of the flattened sequence, and re-shuffles that slice on every iteration when `shuffle=True`, seeded from `data_seed` when set and from `seed` otherwise.

The rank/batch cut is positional, not group-aware, so a group can straddle a rank or batch boundary — correctness comes from the precomputed `1/group_size` loss weight, not from co-locating a group. Per-rank batch counts are equalized by an all-reduce MIN, since an unequal count would strand ranks at the next collective; a split too small to give every rank a full batch raises at dataloader construction.

## Dataset format

The on-disk dataset is **conversational**: the script renders `template(prompt + completion)` per entry in `completions` and strips the rendered-prompt prefix. Completions are never templated standalone — strict templates (Qwen3.5) reject assistant-only message lists, and a BOS-emitting template would inject a mid-sequence BOS.

| Column | Type | Description |
|--------|------|-------------|
| `prompt` | `List[Dict]` | Conversation message list (`{"role", "content"}`) |
| `completions` | `List[List[Dict]]` | Completion message lists (typically 4–16) |
| `rewards` | `List[float]` | One reward per completion |

```jsonl
{"prompt": [{"role": "user", "content": "What is 2+2?"}], "completions": [[{"role": "assistant", "content": "4"}], [{"role": "assistant", "content": "5"}]], "rewards": [1.0, -0.5]}
```

A `tools_field` column is parsed and passed as `tools=` to the template on both renders, so a tool-use dataset trains against the same schema block it was collected under; the column name is part of the map cache key.

A group is one **row**, keyed by dataset row index — two rows carrying the same prompt stay two groups and are never merged, so put every completion of a prompt in one row. A row whose `completions` and `rewards` differ in length raises with its index.

The plain `str` / `List[str]` form applies only when calling `OfflineGRPOTrainer` directly with **pre-templated text** — `tokenize_prompt_completion` does not apply a chat template itself.

## Length budgets

`max_prompt_length` (default `512`) left-truncates the prompt, keeping the tokens nearest the completion. `max_completion_length` (default `null`) truncates the completion from the end, and doubles as `dr_grpo`'s normalization constant. For both, `null` and any non-positive value mean **no cap** — the tokenizer's window is pinned only when both halves are bounded, so an unset half is never silently capped at `model_max_length`.

At a cut the sequence gets no terminator: EOS is appended only when the completion ends inside its budget, and BOS only when the prompt does (and only for tokenizers whose post-processor emits one). A completion truncated at the cap is trained without a stop token rather than taught to stop where the data was cut.

Those two caps are the whole budget: `max_length` is a **pipeline-parallel-only** knob (the fixed shape every batch would pad to) and is rejected at trainer construction on any run without PP — today, every run, since [PP is not yet available in this release](../../parallelism/pipeline-parallelism.md).

## Usage

```bash
torchrun --nproc_per_node=8 scripts/training/offline_grpo.py \
    examples/grpo/offline/qwen3_5/offline-grpo-qwen3.6-35b-a3b-gsm8k.yaml --expert_parallel_size=8
```

From Python, load the model with `ParallelismConfig(ep_size=8)` via `load_distributed_model` and pass `parallelism_config=` to the trainer.

**Reference model at `kl_beta > 0`.** Plain dense models with full fine-tuning deepcopy the live policy (TRL's `create_reference_model`), so a resume re-anchors the KL to the resumed weights.

EP / grouped-GEMM wrapped MoE models hold live NCCL process groups `deepcopy` cannot pickle, so the reference reloads **dense** from the checkpoint path instead — anchored to the checkpoint weights, with the policy's attention implementation. Passing such a model as an object without a path raises: pass it by path, use PEFT, or set `kl_beta: 0`. A PEFT-wrapped policy builds no reference at all (`disable_adapter()` reverts to the base weights, native expert adapters included); an expert-only LoRA run is not PEFT-wrapped and takes the dense-reference path.

The k3 KL estimator is unbounded where the policy suppresses a token the reference likes, so the reference log-ratio is capped at 5 nats before the KL term (`clamp_ref_logps`, shared with the on-policy trainers — mechanism in [KL tail](environmental-grpo.md#kl-tail)). Unlike there, no `kl_clamp_frac` metric is logged: this runs per gradient-accumulation micro-batch, and the `.item()` would be a per-step host sync.

## Hyperparameters

Offline GRPO refines a tuned policy, so the learning rate sits below the SFT band — `examples/grpo/offline/qwen3_5/offline-grpo-qwen3.6-35b-a3b-gsm8k.yaml` uses `5e-6`; `1e-6` is the conservative start. Raise only if the advantage signal stalls. Advantages are estimated within each group, so a wider batch helps; effective batch follows the SFT formula (`per_device_train_batch_size × gradient_accumulation_steps × data_parallel_size`). See [SFT — Learning rate and global batch size](../sft.md#learning-rate-and-global-batch-size). Use 4+ completions per group and ensure rewards span a range; `drop_degenerate_groups: true` removes the all-identical-reward groups (advantage collapses to 0) at tokenization, and raises if that would empty the dataset.

**MoE balancing.** A policy-gradient loss never adds the router aux term, so `moe_balancing: aux_loss` is inert on every GRPO trainer (it warns and forces router logits off). Offline GRPO has no vLLM weight sync, so `bias_update` is the working choice here — unlike the on-policy trainers, where it is downgraded ([Callbacks](../callbacks.md#routerbiasbalancingcallback)).

## Comparison with KTO

[KTO](https://arxiv.org/abs/2402.01306) takes **unpaired binary feedback** — one completion per prompt labeled desirable/undesirable — and requires a reference model. It is the right choice only for truly unpaired singleton data. With 2+ completions per prompt (common from temperature > 0 sampling), Offline GRPO is strictly more expressive: continuous rewards, five advantage methods, three loss types, optional reference. To convert KTO data, group by prompt, map labels to rewards (e.g. `1.0`/`-1.0`), and keep groups of size ≥ 2.

## Troubleshooting

- **`Unknown advantage method / loss type / policy_gradient_formulation`** — set to a supported value (literals above). `loss_type` and `policy_gradient_formulation` are checked at construction on every path, so a bad value fails before the model loads rather than at the first microbatch.
- **Loss NaN / exploding** — set `min_log_prob=-3.0` and `max_grad_norm=1.0`.
- **Model diverging (rewards decreasing)** — distribution shift from offline data. Add `kl_beta=0.05`, use `policy_gradient_formulation="prob_weighted"`, lower the learning rate, or enable `min_log_prob` scheduling.
- **Slow vs SFT** — expected (heavier loss). Use larger batches, enable EP for MoE, or `optim="adamw_bnb_8bit"`.

Watch `positive/logps_mean` (stable or rising), `negative/logps_mean` (falling), and `positive/rewards_mean`.

## Related pages

- [Trainer Architecture](../../reference/trainer-architecture.md) · [Scripts Reference](../../reference/scripts-reference.md)
- [Expert Parallelism](../../parallelism/expert-parallelism.md) · [Pipeline Parallelism](../../parallelism/pipeline-parallelism.md)
- [OfflineGRPOConfig Reference](../../reference/configuration-reference.md#offlinegrpoconfig)
