# Online SDPG

Self-Distilled Policy Gradient ([arXiv:2606.04036](https://arxiv.org/abs/2606.04036)) in its faithful on-policy form: online GRPO plus a privileged-teacher distillation term. `DistributedSDPGTrainer` subclasses `DistributedGRPOTrainer`, so it inherits vLLM server-mode rollouts, NCCL weight sync, and EP/TP from [online GRPO](../grpo/online-grpo.md). Run it through the online-GRPO script with a flag:

```bash
torchrun --nproc_per_node=4 \
    scripts/training/online_grpo/rlvr.py \
    examples/grpo/online/rlvr-online-grpo-template.yaml \
    --use_sdpg=true
```

| Aspect | Value |
|--------|-------|
| Trainer | `DistributedSDPGTrainer` (`src/trainers/distillation/sdpg.py`) |
| Script | `scripts/training/online_grpo/rlvr.py --use_sdpg=true` |
| Loss | `L_GRPO + beta(k) · L_OPD` |
| Policy | On-policy (vLLM rollouts) |
| Parallelism | EP, TP, ETP, EP+TP, EP+ETP; no CP, no PP. Adapters under FSDP2 DP, EP and pure ETP; refused under TP |
| Modality | Text only |

## How it works

Each step the student samples a group of completions from the vLLM server and the verifier scores them into group-normalized advantages — the standard online-GRPO loop. The same model is then run as a privileged teacher that additionally sees a hint revealing the gold answer, and the student is distilled toward the teacher's full-vocabulary next-token distribution on the sampled completion tokens via reverse KL `D_KL(p ‖ SG[q])`.

The OPD term is gated to positive-advantage (verifier-preferred) rollouts — a zero advantage means the group tied or the row was unscorable, so it carries no privileged supervision. The total loss is `L = L_GRPO + beta(k) · L_OPD`; reference-KL regularization is GRPO's built-in KL coefficient, not a separate term. `beta(k)` follows the SDPG warmup→decay schedule (constant `sdpg_beta_base` when `sdpg_beta_warmup_steps`/`sdpg_beta_decay_steps` are 0).

The teacher is the same policy run under `torch.no_grad()` and `eval()` (so train-mode dropout does not perturb the target), and the OPD stop-gradients it — no second model is held. Loss and schedule come from `src/trainers/distillation/losses.py`.

The OPD term costs two extra full-vocabulary forwards per microbatch (student with grad, teacher under no-grad) on top of the GRPO one, neither passing `logits_to_keep`: both materialize `[B, prompt+completion, V]` logits and the KL upcasts them to fp32. `use_chunked_grpo_logprobs` bounds the GRPO log-probs only — it does not protect OPD, so size `per_device_train_batch_size` and `max_completion_length` against that peak.

`use_liger_kernel` is forced off on every online/environmental GRPO run (`disable_trl_liger_grpo_loss`, `src/trainers/mixins/validation.py`); SDPG adds one more reason — TRL's fused GRPO-Liger loss bypasses the `_compute_loss` path and would silently drop the OPD term. Model-level Liger kernels still apply at load time.

## Dataset and generation

The dataset is the online-GRPO format `{"prompt": [...], "answer": "..."}` — the answer feeds both the reward function and the privileged hint. Generation is server-only: the training image ships without vLLM, so completions come from the separate vLLM container with NCCL weight sync. See [online GRPO](../grpo/online-grpo.md) for the vLLM setup, server flags, and parallelism rules — SDPG inherits them unchanged. With `sdpg_beta_base: 0` it reduces to plain online GRPO.

## Configuration

SDPG fields live on the RLVR script args (`src/args/rlvr_online_grpo_args.py`) and are active only with `--use_sdpg=true`. No shipped example YAML sets `use_sdpg`, so enable it on the CLI over an online-GRPO config. The remaining online-GRPO/vLLM fields are documented in [online GRPO](../grpo/online-grpo.md).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_sdpg` | `False` | Swap in `DistributedSDPGTrainer` (GRPO + privileged-teacher OPD) |
| `sdpg_hint_template` | reveals the gold answer | Hint appended to the prompt for the teacher forward (`{answer}` placeholder) |
| `sdpg_loss` | `"reverse_kl"` | OPD loss: `reverse_kl` (SDPG), `forward_kl`, or `unnormalized_kl` |
| `sdpg_temperature` | `1.0` | OPD softmax temperature |
| `sdpg_beta_base` | `1.0` | Base OPD coefficient |
| `sdpg_beta_warmup_steps` | `0` | Steps to ramp beta 0→`sdpg_beta_base` |
| `sdpg_beta_decay_steps` | `0` | Final steps over which beta decays to 0 |
| `opd_positive_advantage_only` | `True` | Restrict the OPD term to positive-advantage tokens (SDPG as published); `false` distills every completion token |

The hint reads the trainer's hard-pinned `answer` column: `process_for_rlvr` normalizes the dataset column named by `answer_field` into it, so `answer_field` picks the content, not the column the trainer reads.
