# Offline GRPO

Offline GRPO trains on pre-collected, off-policy data: several completions per prompt, each with a pre-computed reward, and no generation during training. Trainer `OfflineGRPOTrainer`, script `scripts/training/offline_grpo.py`, config `OfflineGRPOConfig` ([field reference](../../reference/configuration-reference.md#offlinegrpoconfig)).

Use it when the completions already exist, when scoring them is expensive, or when a reproducible run matters. For live generation use [Online GRPO](online-grpo.md) or [Async GRPO with Environments](async-grpo/README.md); for pairwise data, [SMPO](../preference/smpo.md) or DPO ([GRPO overview](README.md) compares all three).

Parallelism: EP, TP, EP+TP, EP+ETP, pure ETP (`ep_size=1`), CP, and EP+CP. Under CP, full prompt+completion rows are right-padded to the CP degree, scored through Ulysses and the vocab-chunked head, and reduced per sequence across CP ranks. CP requires full fine-tuning and rejects PEFT and native expert LoRA, including at `kl_beta: 0`. The trainer declares `_supports_pp`, but [pipeline parallelism](../../parallelism/pipeline-parallelism.md) is not yet available in this release.

![Offline GRPO in two bands: tokenization turns one dataset row (prompt, completions, rewards) into per-group advantages and then one training row per completion carrying its advantage, group_id and group_size; each training step draws a micro-batch from MultiGroupSampler, computes the per-token loss with the min_log_prob floor and the optional k3 KL, and normalizes it with bnpo, grpo or dr_grpo, every row weighted 1/group_size](../../assets/diagrams/offline_grpo_pipeline.png)

## Dataset

Three conversational columns. A group is one **row**, keyed by row index: put every completion of a prompt in one row, since two rows with the same prompt stay two groups.

| Column | Type | Contents |
|---|---|---|
| `prompt` | `list[dict]` | Message list (`{"role", "content"}`) |
| `completions` | `list[list[dict]]` | One message list per completion (typically 4–16) |
| `rewards` | `list[float]` | One reward per completion |

```jsonl
{"prompt": [{"role": "user", "content": "What is 2+2?"}], "completions": [[{"role": "assistant", "content": "4"}], [{"role": "assistant", "content": "5"}]], "rewards": [1.0, -0.5]}
```

A singleton group, and any exactly-tied one, normalizes to advantage 0 under every method. A `completions`/`rewards` length mismatch raises with the row index, as does a prompt that tokenizes to no tokens; a non-finite reward raises with the offending group.

`scripts/inference/reward_model/rm_rejection_sampling.py` emits this shape with `--output_format offline_grpo`.

The training script renders `template(prompt + completion)` and strips the rendered-prompt prefix, so a strict template (Qwen3.5) never sees an assistant-only message list. A `tools_field` column is passed as `tools=`.

## Configuration

```yaml
model_name_or_path: Qwen/Qwen3.6-35B-A3B
dataset:
  - s3://bucket/gsm8k-qwen3.6-35b-a3b-offline-grpo
advantage_method: quantile_norm
loss_type: bnpo
kl_beta: 0.0
initial_min_log_prob: -0.5
max_prompt_length: 512
max_completion_length: 6144
expert_parallel_size: 8
learning_rate: 5.0e-06
```

| Knob | Default | Effect |
|---|---|---|
| `advantage_method` | `quantile_norm` | Reward → advantage within each group (table below) |
| `best_completion_emphasis` | `0.0` | Boost the top-reward rows by a factor above `1.0`, or `auto` |
| `loss_type` | `bnpo` | Loss normalization: `bnpo`, `grpo`, `dr_grpo` |
| `policy_gradient_formulation` | `prob_weighted` | Per token `-(π·A)`; `reinforce` uses `-(log π·A)` |
| `min_log_prob` | `-3.0` | Log-prob floor, on negative-advantage rows only |
| `initial_min_log_prob` | `null` | Linearly schedule that floor from this start |
| `kl_beta` | `0.0` | Coefficient of the k3 KL penalty to a reference policy |
| `drop_degenerate_groups` | `false` | Drop exactly-tied and singleton groups at tokenization |
| `use_chunked_grpo_logprobs` | `false` | Vocab-chunked log-probs, not full logits |
| `max_prompt_length` | `512` | Prompt budget, left-truncated; `null` = no cap |
| `max_completion_length` | `null` | Completion budget, cut from the end; `null` = no cap |
| `max_length` | `null` | Pipeline-parallel only; rejected at construction off PP — on every run, since [PP is not yet available](../../parallelism/pipeline-parallelism.md) |

The shipped recipes run `learning_rate: 5.0e-06`: GRPO refines a tuned policy, so the rate sits below the SFT band. Advantages are group-relative, so a wider batch helps ([sizing](../sft.md#learning-rate-and-global-batch-size)).

Both length knobs are truncation caps, and the tokenizer's window is pinned only when both are bounded. At a cut the sequence gets no terminator — EOS is appended only when the completion ends inside its budget — so a completion cut at the cap is trained without a stop token rather than taught to stop there.

### Advantages

Rewards map to advantages per group during preprocessing, then clip to `[-10, 10]` — emphasis can push one past a method's own range.

| `advantage_method` | Formula | Best for |
|---|---|---|
| `quantile_norm` (default) | inverse normal CDF of ranks | ordinal or outlier-heavy rewards |
| `z_norm` | `(r - mean) / (std + ε)` | normal reward distributions |
| `minmax` | `2(r - min)/(max - min) - 1` | bounded `[-1, +1]` advantages |
| `quantile_uniform` | uniform from ranks | most outlier-tolerant |
| `robust` | `(r - median) / IQR` | extreme outliers |

`best_completion_emphasis` takes a float above `1.0` (`2.0` for 2× weight) or `"auto"`, which adapts to reward variance as `3.0 + 2.0·std/(1.0 + std)`. Any other number raises — negatives and `(0.0, 1.0]` alike — since the factor applies only above `1.0`.

`drop_degenerate_groups: true` drops **exactly**-tied groups and groups of fewer than two before they spend forward compute or dilute the loss normalizer, and raises if that empties the dataset. Near-ties are kept — the rank methods train them at full scale.

### Loss types

All three weight each example by `1/group_size`, so every group contributes equally however many completions it has.

- `bnpo` (default): global weighted token average, over the micro-batch's group-weighted token sum.
- `grpo`: per-sequence average, then a weighted mean across groups. Use when completion length varies.
- `dr_grpo`: normalized by effective group count × `max_completion_length` — a constant denominator that removes length bias. It is not a generation budget here, so a `null` or non-positive value raises at trainer init.

### Memory: chunked log-probs

The default path materializes `[B, T_completion, vocab]` logits — on wide vocabularies with long completions, the memory peak. `use_chunked_grpo_logprobs: true` computes the same log-probs from the backbone's `last_hidden_state` via a vocab-chunked softmax. CP always takes the chunked path and scores only its own sequence slice, including the label at the next shard boundary. It is inert under PP and warns there. Limits: [Chunked log-probs](async-grpo/performance.md#chunked-log-probs).

### Reference model

At `kl_beta > 0`, full fine-tuning scores the run-start policy once before its first update, over the
configured train and evaluation splits. This uses the same reference lifecycle under CP, non-CP
and the PP loss seam; PP itself remains unavailable. Inputs must be finite, unsharded
`datasets.Dataset` splits. Pre-sharded KL datasets and supplied grouped `ref_per_token_logps` are
refused. No dense policy deepcopy or second resident reference model is needed. An explicit frozen
`ref_model` is accepted outside CP and PP, swept once and released.

This preparation finishes before step 1 and logs batch progress for each split. The trainer
does not resweep on resume. With KL disabled, PEFT, or expert LoRA, unused dataset reference
columns are dropped during tokenization; the finite-split and pre-sharded restrictions above
apply only to the full-finetuning sweep.

Raw completion-token scores travel with every training checkpoint in `reference_logps.pt`.
The live `min_log_prob` clamp is applied at each loss step, not stored in the reference. On resume,
the sidecar preserves the original anchor instead of scoring the trained policy. Ordered prompt
and completion token digests, row counts and reference settings must match; non-finite scores or
rank-divergent reference values raise. Pass the resolved `resume_checkpoint` at trainer
construction, before reference preparation, then pass the same path to `train()`.

The sweep streams bounded batches from one representative per DP shard to the output-filesystem
writers: global rank 0 on shared storage, each node's local rank 0 on node-local storage.
Writers merge contiguous DP shards in dataset order; all ranks attach a memory-mapped Arrow
reference column. Checkpoint serialization uses the same mapped values, and resume maps the
sidecar instead of loading a full float32 token table into each rank's heap. Unchanged
later checkpoints hardlink the immutable sidecar, or copy it when links are unavailable. DPO, KTO
and offline GRPO share that persistence and resume lifecycle
([Checkpoints](../../reference/checkpoints.md#what-gets-saved)).

The temporary `_reference_cache/<uuid>/` filenames are unlinked after every rank maps the
completed scores; those mappings stay valid through training and checkpointing. NFS may keep
hidden files until their last mapped reader closes. The scratch cache is excluded from Hub uploads.

`evaluate(new_tokenized_dataset)` reuses the original scores for token-identical rows, including
subsets, reordered rows and duplicates. Unseen rows require the exact original frozen policy:
outside CP/PP, pass `original_reference_model=original_frozen_policy`. The model must have all
parameters frozen and be in eval mode; the trainer places it temporarily for the sweep and restores
its original device. The caller owns the run-start weight identity. CP/PP evaluation splits must
be declared before training so their scores come from the original distributed policy.

An old KL checkpoint without its sidecar cannot rescore trained weights. Recover the file from a
complete checkpoint, or run the same tokenized splits and configuration for one step from the
exact original model/revision into a separate scratch output. Copy that run's
`checkpoint-1/reference_logps.pt` into the resume checkpoint on each node with node-local storage.
The refusal message gives the recovery flags. Supplied scores and pre-sharded KL datasets remain
unsupported, including during recovery.

The trainer places supplied references on the policy's device, switches them to eval mode and
freezes their parameters. Outside CP, PEFT policies retain the adapter-disabled frozen-base reference. Native expert-only
LoRA builds no `PeftModel` and requires an explicit frozen base `ref_model`, loaded by the script
through `load_frozen_reference_model` from `model_name_or_path`, never the trained resume checkpoint,
with the policy's revision, attention and sinks settings.
Those adapter paths keep their live reference behavior. A run without KL rejects an unused
`ref_model`. The reference log-ratio is capped at 5 nats to bound the k3 estimator's tail.

## Launch

```bash
torchrun --nproc_per_node=8 scripts/training/offline_grpo.py \
    examples/grpo/offline/qwen3_5/offline-grpo-qwen3.6-35b-a3b-gsm8k.yaml --expert_parallel_size=8
```

`halo launch offline-grpo <config> --nproc 8` builds the same line. From Python, load the model through `load_distributed_model` and pass the same `ParallelismConfig` to the trainer.

**MoE balancing.** A policy-gradient loss never adds the router aux term, so `moe_balancing: aux_loss` warns and does nothing — or raises at construction when `output_router_logits` is on with a positive `router_aux_loss_coef`. With no weight sync here, `bias_update` is the working choice on a family whose bias exports — unlike the on-policy trainers, where it is downgraded ([Callbacks](../callbacks.md#routerbiasbalancingcallback)); the shipped recipes set `none`.

## Testing a setup

Run a short config on a slice of the data first: `max_steps: 10` with `save_strategy: "no"` exercises tokenization, advantage normalization and the loss in minutes. A bad `loss_type`, `advantage_method` or `policy_gradient_formulation` is caught at construction.

CPU: `pytest tests/cpu/grpo -m cpu`, including CP1/CP2/CP4 equivalence of logged KL/reference metrics. GPU: `tests/gpu/trainers/grpo/test_offline_grpo.py` and its `_bnpo` / `_bs4` / `_chunked` / `_tp_resume` siblings, plus the LoRA suites. The CP suites compare CP1 with CP2/CP4 on dense Qwen3 and Qwen3-MoE with EP+CP: token log-probs, loss, per-parameter gradient direction and global norm, and an AdamWBF16 step. Boundary-only supervision makes the missing-boundary negative control fail numerically. The GPU trainer suite covers train → resume → export.

## What to watch

Per-sample diagnostics split by advantage sign, once per log step: `{positive,negative}/{logps,rewards,kl,ref_logps}_{mean,std,min,max,range}`, plus the pre-clamp `negative/*_unclamped_*`. The `kl` and `ref_logps` families exist only at `kl_beta > 0`, `min_log_prob_restriction` is the live clamp floor, and evaluation prefixes every key with `eval_`.

Read `positive/logps_mean` (stable or rising), `negative/logps_mean` (falling) and `positive/rewards_mean`.

| Symptom | Cause | Fix |
|---|---|---|
| Loss NaN or exploding | Low-probability tokens on negative-advantage rows | `min_log_prob: -3.0` and `max_grad_norm: 1.0` |
| Rewards fall through training | Distribution shift from the offline data | Add `kl_beta: 0.05`, lower the learning rate |
| `Batch-count equalization left 0 full batches` | The split is too small for this world size | Lower `per_device_train_batch_size`, shrink the world |
| OOM | Full-vocab logits over long completions | `use_chunked_grpo_logprobs: true`, lower the batch or `max_completion_length`, or raise EP size |

## Related pages

- [Online GRPO (RLVR)](online-grpo.md) · [Async GRPO with Environments](async-grpo/README.md) · [GRPO overview](README.md)
- [Expert Parallelism](../../parallelism/expert-parallelism.md) · [Pipeline Parallelism](../../parallelism/pipeline-parallelism.md)
- [Trainer Architecture](../../reference/trainer-architecture.md) · [Configuration Reference](../../reference/configuration-reference.md#offlinegrpoconfig)
