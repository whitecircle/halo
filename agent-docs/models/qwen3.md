# Qwen3 Family

Three variants: dense (`Qwen3ForCausalLM`), MoE (`Qwen3MoeForCausalLM`), and the text branch of the VL model.

| Variant | HF classes | EP | CP | TP | ETP |
|---|---|:--:|:--:|:--:|:--:|
| Qwen3 dense | `Qwen3ForCausalLM` / `Qwen3Attention` | — | Yes | Yes | — |
| Qwen3 MoE | `Qwen3MoeForCausalLM` / `Qwen3MoeSparseMoeBlock` / `Qwen3MoeAttention` | Yes | Yes | Yes | Yes |
| Qwen3-VL (text) | `Qwen3VLTextAttention` | — | Yes | **No** (dense: no `base_model_tp_plan`) | — |

Qwen3.5 / Qwen3.6 share the name prefix but use a different `Qwen3_5*` class hierarchy — see the [separate page](qwen3_5.md).

## Qwen3 dense

Standard transformer with GQA, RoPE, SwiGLU MLP. Stock `Qwen3ForCausalLM`, no patching.

- **FSDP2** — default under both launchers.
- **TP** — native HF `tp_plan="auto"` shards attention and MLP.
- **CP** — `Qwen3Attention` uses the same `Qwen3MoeUlyssesAttention` wrapper as Qwen3 MoE and Qwen3-VL.
- **PP** — [not yet available in this release](../parallelism/pipeline-parallelism.md). The shipped split contract splits the family cleanly, but the small released checkpoints (0.6B, 1.7B, 4B-Instruct-2507) ship `tie_word_embeddings=True` and would hit the tie gate; Qwen3-8B is untied.
- **LoRA** — under FSDP/DP and CP; rejected at construction under TP ([PEFT](../optimization/peft.md#parallelism-compatibility)).

### Configs

| Config | Method | Base model |
|---|---|---|
| `examples/sft/qwen3/qwen3-4b-ultrachat.yaml` | SFT (full), 4K seq | `Qwen/Qwen3-4B-Instruct-2507` |
| `examples/sft/qwen3/qwen3-4b-ultrachat-lora.yaml` | SFT (LoRA) | `Qwen/Qwen3-4B-Instruct-2507` |
| `examples/sft/qwen3/qwen3-4b-ultrachat-qlora.yaml` | SFT (QLoRA) | `Qwen/Qwen3-4B-Instruct-2507` |
| `examples/embedding/qwen3/embedding-qwen3-4b-nq.yaml` | Embedding | `Qwen/Qwen3-Embedding-4B` |
| `examples/grpo/online/qwen3/online-grpo-qwen3-{4b,8b}-smoke.yaml` | Online GRPO smoke | `Qwen/Qwen3-4B-Instruct-2507`, `Qwen/Qwen3-8B` |
| `examples/grpo/{online/rlvr-online-grpo,environmental/environmental-grpo}-template.yaml` | RLVR / env GRPO templates | `Qwen/Qwen3-4B-Instruct-2507` |

The per-method canonical examples (DPO, SMPO, KTO, reward, classification, distillation, GRPO task configs) are built on Qwen3.5/3.6 — see [Qwen3.5 / Qwen3.6](qwen3_5.md).

## Qwen3 MoE

`EPQwen3MoELayer` (`src/distributed/expert_parallel/layers/qwen3.py`) replaces `Qwen3MoeSparseMoeBlock`.

- Routing: the gate returns `(logits, weights, experts)`; softmax + top-k.
- Activation: `act(gate(x)) * up(x)` → `down`, where `act` is read off `Qwen3MoeExperts.act_fn` (the resolved `hidden_act`, SiLU on every released checkpoint) rather than assumed.
- Storage: pre-fused `gate_up_proj [E, 2M, H]` + `down_proj [E, H, M]`, split at wrapper construction into separate `gate_proj` / `up_proj` / `down_proj` for sharding.
- Compute: Grouped GEMM on SM90+, falling back to a per-expert loop with `index_add_`.
- RL weight sync: `gather_expert_state_dict` emits the per-expert layout vLLM loads. `rollout_backend:
  sglang` is **refused** — 0.5.17's `qwen3_moe` loader maps per-expert names only
  (`make_expert_params_mapping`; the fused variant is `gpt_oss`-only), so a fused pair would be
  dropped with no server-side signal, and the EP layer declares no `gather_fused_expert_state_dict`
  ([Rollout Servers](../infrastructure/rollout-servers.md#the-fused-expert-layout-is-declared-per-family)).

**Router balancing** — the wrapper re-derives selection from the router's own logits, applying the DeepSeek-V3 bias to the selection scores while the gate weights stay on the unbiased softmax (honoring the family's `norm_topk_prob`).

The architecture has no bias slot (the gate is a bare weight), so the bias is trainer-only: `moe_balancing: bias_update` **raises** and the explicit `bias_update_transient` is the opt-in, with exported checkpoints serving without the trained bias. `auto` stays on `aux_loss`, since `Qwen3MoeForCausalLM.forward` declares `output_router_logits` ([Callbacks](../training-methods/callbacks.md#moe-balancing-modes)).

**CP** — `Qwen3MoeAttention` → `Qwen3MoeUlyssesAttention` (`src/distributed/context_parallel/layers/qwen3.py`), RoPE before all-to-all and native GQA.

**TP and ETP** — selective attention-only TP (`q/k/v_proj` ColwiseParallel, `o_proj` RowwiseParallel); embeddings and `lm_head` stay replicated, expert layers belong to EP. ETP shards `gate_proj` / `up_proj` / `down_proj` along the intermediate dim; because Qwen3 already stores gate and up separately, the same compute path runs under both ETP and non-ETP with a smaller `M`.

**LoRA** on the experts works under EP through native grouped adapters (`_init_expert_lora`). It is rejected under TP and EP+TP, and at `expert_tp_size > 1` (the replicated adapter half would take partial gradients) — attention LoRA still runs there.

**Configs** — `examples/sft/qwen3/qwen3-4b-ultrachat.yaml` (plain FSDP2; `-lora` / `-qlora` variants). `Qwen/Qwen3-235B-A22B-Instruct-2507` trains at `--expert_parallel_size=8`.

## Qwen3-VL

`Qwen3VLTextAttention` reuses `Qwen3MoeUlyssesAttention` for CP and is in the selective-TP accept-list (`src/distributed/tensor_parallel/module_types.py`). That accept-list governs the attention-only DTensor path, which the loader takes for MoE and EP+TP shapes.

A **dense** Qwen3-VL goes through HF-native `tp_plan="auto"` instead, and the architecture ships no `base_model_tp_plan` — the plan is empty, nothing shards, and the load **raises** rather than running `tp_size` full replicas at `1/tp_size` throughput. Use CP or plain FSDP2 for dense VLM runs. The MoE variants get no EP: no wrapper claims `Qwen3VLMoeTextSparseMoeBlock`, and `Qwen3VLMoeTextAttention` is in neither the CP nor the selective-TP registry.

No dedicated SFT configs ship; `tests/gpu/trainers/sft/test_sft_vlm.py` exercises the path on `Qwen/Qwen3-VL-2B-Instruct`. See [Vision-language models](../training-methods/sft.md#vision-language-models).
