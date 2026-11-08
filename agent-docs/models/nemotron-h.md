# Nemotron-H

`model_type` `nemotron_h_omni` (`NemotronH_Omni_Reasoning_V3`, the release `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`) wrapping the text tower `nemotron_h` (`NemotronHForCausalLM`), both native in transformers 5.20. The text tower holds 120.7B parameters, 112.7B of them routed experts, beside a 0.84B vision tower and connector and a 2.9B multi-token head (`mtp.*`), which [`mtp_loss_weight`](#mtp-head) builds and a stock class does not. It has 88 layers: 40 Mamba-2, 40 MoE and 8 attention. A MoE layer routes 512 gateless experts at top-22 beside one shared expert, on a 1024-wide latent it projects into from hidden 4096 and back. The vision tower is C-RADIO: 32 layers, 16 heads at head_dim 80.

Nemotron 3.5 Lightning (`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`) is that text class alone, with no composite around it and no latent ([Lightning](#lightning)). Everywhere else on this page, "the release" means Super VL.

Per-axis support: [Compatibility matrix](README.md#compatibility-matrix).

## Status

The release has trained through the toolkit at `ep8` on 8× B300, with real weights and text, reentrant checkpointing and FA4. `models/test_nemotron_h_real_checkpoint.py` and the shipped recipes ran these:

- **Recipes:** full fine-tune at 8,192 tokens (20 steps) and 32,000 (5), frozen experts at 50,000 (5), LoRA at 16,384 (10), SMPO (5), offline GRPO (5) and images on the composite (5), also with half the rows text-only (5). [Memory](#memory) has their figures.
- **Load:** the lazy route, at a peak host RSS of about 10 GiB per rank on the release cut to 9 layers. The EP loss at that cut lands within 4e-4 of one rank's.
- **Packed rows:** at the cut, a document's mean logit gap against itself alone is at most 5.49e-5 with the markers across the three texts the row has read, with no position past its bar, and 1.13 without them.
- **Resume:** exact at the 9-layer cut: loss delta 0, final weights equal, optimizer state bit-exact, no warm restart.
- **Exports:** byte for byte against the release at the 9-layer cut on both routes, with an NLL gap of at most 6e-5 on reload, and at full depth: 41,643 tensors on the text-only route and 42,101 on the composite one (the release's 42,037 plus 64 `lambda1`), each with an NLL gap of 0.0013. `reattach_vision_tower.py` turns the text-only export back into the release's 43,079 tensors with no `config.json` difference.
- **Adapters:** the LoRA recipe's folded export differs from the release on its targets and the moved bias alone (41,112 tensors, 531 identical), and loads and generates in stock transformers at an NLL of 2.03296 against 2.03301. `merge_peft_adapters.py` merges attention and Mamba adapters byte for byte the same with `--trust_remote_code` on or off, changing 152 tensors of the release (80 Mamba, 32 attention, 40 bias), and streams the merge of a mixed adapter.

What these rows leave:

- **Length of run:** 5 to 20 steps each. No run has trained the release to a quality metric, so a step's loss is all they say about training.
- **Serving:** both engines have served an export, an untrained reattached one, on one image prompt and one text prompt ([Checkpoints and serving](#checkpoints-and-serving)). Engine sync stays refused ([RL weight sync](#rl-weight-sync)).
- **Mixed rows:** image and text rows ran at `ep8` on the release with no hang. That the tower trains on them rests on tiny rows.
- **Other axes:** pure ETP and EP+ETP have run on tiny models only. Pipeline parallelism ran Super VL's text tower for 13 steps with GC on.

## Lightning

The release is `NemotronHForCausalLM` itself: `model_type` `nemotron_h`, no vision tower and no `auto_map`, stored plain bf16 with its router bias in fp32. The tower holds 31.6B parameters, 29.4B of them routed experts, beside a 1.3B multi-token head (`mtp.*`), which [`mtp_loss_weight`](#mtp-head) builds and a stock class does not. It has 52 layers: 23 Mamba-2, 23 MoE and 6 attention. A MoE layer routes 128 gateless experts at top-6 beside one shared expert with no latent, so the experts and the wire run at hidden 2688, which DeepEP pads to 2816. `ep8` holds 16 experts per rank.

The family's layers, balancing and refused modes below apply, and these differ:

- **Load:** natively, with its keys at the root and no `text_only_model` to set. Upstream's ignore list drops the 270 `mtp.*` keys. On a tiny copy of the layout, each rank's lazy load holds what `from_pretrained` does.
- **Rows:** text only, as the release has no vision tower.
- **Size:** 63 GB gathered and about 40 GB static per GPU for a full fine-tune at `ep8`, by arithmetic. So FSDP2 alone is not held back by size as it is on Super VL, though it has not run on these weights, and the figures under [EP layer](#ep-layer), [Memory](#memory) and HSDP below are Super VL's.
- **Time-step limit:** the config declares none, so the class default (0, inf) applies, which both engines apply too. A stock reload keeps upstream's floor ([Mamba layers](#mamba-layers)).
- **Recipe:** `examples/sft/nemotron_h/nemotron-3.5-lightning-ep8-full.yaml`.
- **Export:** the release's own `config.json` with this run's changes, its `auto_map` absent, so there are no modules to copy ([Exported `config.json`](../reference/checkpoints.md#exported-configjson)). `layers_block_type` keeps `mamba` and `attention`, the spellings SGLang's config class requires, and no `time_step_limit` reaches the file, so it holds no JSON float tag. vLLM's and SGLang's config readers parse the file; SGLang's refuses transformers' `linear_attention` and `full_attention`. Stock `from_pretrained` reloads a tiny copy's export to the release's tensors and logits. A depth cut writes its layer list in transformers' spelling, which SGLang refuses, so serve a full-depth export.
- **Head:** a run with no `mtp_loss_weight` exports no `mtp.*`, and `reattach_vision_tower.py` cannot restore one, as it needs a composite base. A run training the [head](#mtp-head) writes the release's 270 keys.

On the real weights:

- **Load and rows at a cut:** lazy at `ep8`, peak host RSS 4.1 GiB, every MoE layer EP-wrapped, FA4. At 9 layers `ep8` differs from one rank by 2.9e-3 in loss, less than bf16 differs from fp32 on one rank (5.2e-3), and a packed document stays within 3.9e-6 of itself alone (1.01 without `seq_idx`).
- **Exports:** the full 52-layer export holds 6,243 keys, the release's 6,513 minus `mtp.*`, each byte-identical to the release, with an identical `config.json`. NLL gap 0.0042, 59 GiB on disk.
- **Resume:** exact at a 9-layer cut: loss difference 0.0, final weights rel L2 0.0.
- **Throughput per block kind:** forward and backward at 8,192 tokens run 10,694 tok/s per rank at `ep8`: Mamba 58.0%, MoE 36.2%, attention 3.2%, head 2.6%.
- **Recipe, memory and tok/s:** 20 steps at `ep8` and `max_length` 8,192 run 9,986 tok/s per rank with the optimizer and reduce, at 6.53 s a step, 47.4–47.9 GiB peak allocated and 55–60 GB reserved.
- **Serving:** vLLM and SGLang each start an untrained export with the cookbook's flags and answered one text prompt. SGLang refuses transformers' native `config.json` at start. Online weight sync stays refused, as for the family.

## Methods

- **SFT:** full fine-tuning or LoRA, on text (`text_only_model: true`, the recipes) or images (the composite).
- **SMPO:** text or images.
- **Offline GRPO:** text, on either class. It needs no server ([Offline GRPO](../training-methods/grpo/offline.md)).
- **DPO, KTO:** text, a full fine-tune with `precompute_ref_log_probs`, as a frozen reference holds 241 GB unsharded on every GPU. Image rows are refused before the load (`refuses_trl_image_rows`).
- **Self-distillation:** text or images. **Teacher distillation:** a frozen teacher of this size leaves a B300 no room for a student.
- **Online and async GRPO, online SDPG:** refused ([RL weight sync](#rl-weight-sync)).
- **Reward, classification, embedding:** no row; transformers has no sequence-classification class for the family.

## Mamba layers

`apply_nemotron_h_patches` (`halo/models/families/nemotron_h.py`) replaces upstream's module-level scan and leaves the mixer's forward in place:

- **Per-group scan.** A group's heads share B and C, which upstream copies per head. The output matches upstream's to 1e-7. On one layer and one GPU at 8,192 tokens it takes 31.8 ms against 34.7 ms and 5.23 GiB of scan memory against 8.29 GiB, by arithmetic about 1.05× faster over a step. Autocast is off inside it, as it would run this fp32 body in bf16.
- **Documents.** A row holding several documents scans each from a zero state, with its chunks counted from its first token. Documents share a padded batch while it costs at most 1.25× their tokens.
- **No fused kernel.** The patch pins `mamba2_split_conv1d_scan_combined` off, so a later `mamba-ssm` install cannot read `in_proj` and `out_proj` past their adapters or refuse `seq_idx`.
- **Time-step limit.** The mixer takes the config's `time_step_limit`: Super VL declares (0, inf), and Lightning declares none, so the class default (0, inf) applies. Both engines run their prefill at (0, inf) without reading the field. Upstream clamps at `time_step_min` 0.001 instead. In four of Super VL's Mamba layers on 8,192 tokens, 11–20% of the time steps sit below that floor, and the two clamps move a layer's output by 0.6–5.7% rel L2. Scoring the same weights on real text, one run per length, upstream's floor reads 0.05–0.09 nats higher on Super VL from 8k to 50k tokens and 0.11 higher on Lightning at 8k.
- **Radio's `lambda1`.** It stays a buffer at 1.0: the release stores none and neither engine reads one, so a trained value would serve as 1.0.

The conv runs `causal_conv1d`, which takes `seq_idx` on CUDA. Under reentrant checkpointing at `conv_dim` 10240 it ran without a fault, at +37% on a Mamba layer.

**Throughput.** The scan is the bottleneck. At `ep8` on 8× B300, the tower's forward and backward alone, with no optimizer or FSDP2 reduce, stay flat with length:

| Tokens per rank | Experts | tok/s per rank | Step |
|---:|---|---:|---:|
| 8,192 | trained | 3,226 | 2.54 s |
| 16,384 | frozen | 3,336 | 4.91 s |
| 32,768 | frozen | 3,223 | 10.17 s |
| 50,000 | frozen | 3,272 | 15.28 s |

![A forward and backward at EP8 and 8,192 tokens per rank, split by block kind: Super VL takes 2.54 s at 3,226 tok/s per rank, 64.8% Mamba and 32.6% MoE; Lightning takes 0.766 s at 10,694 tok/s per rank, 58.0% Mamba and 36.2% MoE; attention and the head take the rest](../assets/benchmarks/nemotron_block_time.png)

At 8,192 the Mamba layers take 64.8% of that step, MoE 32.6%, attention 1.6% and the head 1.0%. By arithmetic the scan is 2–2.5% of the FLOPs, as a B300's fp32 GEMM runs at 32.5T MAC/s against 726T for bf16. No native scan kernel ships. TF32 would take the step to 1.13× of upstream's body at 3e-4 output error per layer, and stays off. MFU costs a Mamba layer by its parameters (`6N`), with no term for the scan, so a low MFU here means the scan. A whole training step adds the optimizer and the reduce: [Memory](#memory) gives those rates.

## EP layer

`EPNemotronHMoELayer` (`halo/distributed/expert_parallel/families/nemotron_h.py`) over `EPGroupLimitedMoELayerBase`:

- **Wire.** The latent travels between ranks, so `fc1_latent_proj` and `fc2_latent_proj` stay replicated around the dispatch. Routing reads the hidden state, in fp32 from the router's bf16 weight.
- **Experts.** `up_proj` and `down_proj` with no gate, on the grouped kernel: two calls per layer. Autocast is off around `relu2`, which it would return in fp32 where `grouped_mm` casts nothing back. Keep `use_grouped_gemm` on: the stock per-expert loop's backward takes 2.0–2.2 s per layer at every length, against 14 ms grouped at 8,192.
- **Layouts.** `ep8` is the production shape on 8 GPUs, 64 experts per rank. `ep4` on 8 GPUs is rejected by the [single-domain rule](../parallelism/README.md#supported-combinations). By arithmetic `ep2` fits LoRA only. A dispatch tops out at 8,192 tokens per rank across domains and has no ceiling within one. Pure ETP and EP+ETP (`up_proj` split on its intermediate axis) run on tiny models; on one node EP+ETP's expert share equals `ep8`'s.

FSDP2 alone cannot hold the model: about 241 GB gathered plus 93 GB of sharded gradients and moments per GPU.

## Memory

Measured at `ep8` on 8× B300 under reentrant checkpointing, on the recipes at the lengths below. Size a run by reserved memory against the device's 288 GB (275,040 MiB). Reserved differs by rank, so each cell is the range across ranks, and tok/s per rank includes the optimizer and the FSDP2 reduce.

| Recipe | Tokens per rank | tok/s per rank | Peak allocated | Reserved |
|---|---:|---:|---:|---:|
| LoRA | 16,384 | 3,000 | 92–94 GB | 118–165 GB |
| Full fine-tune | 8,192 | 3,054 | 168–169 GB | 185–202 GB |
| Full fine-tune, `--max_length=32000` | about 31,700 | 3,221 | 227–235 GB | 268–283 GB |
| Frozen experts, `--max_length=50000` | about 48,800 | 3,246 | 188–194 GB | 241–283 GB |
| Full fine-tune on images (the cookbook's recipe) | about 6,790 | 3,198 | 168 GB | 184–188 GB |
| The same recipe, half the rows text-only | | 3,166 | 168 GB | 183–186 GB |
| SMPO, full pairs | 2 × 4,096 | | 170 GB | 178–184 GB |
| SMPO, its Tulu rows | up to 2 × 4,096 | | 157–161 GB | 165–170 GB |
| Offline GRPO | the recipe's rows | | 164–165 GB | 169–179 GB |

![Peak allocated and reserved memory per GPU for each recipe against the device's 288 GB: LoRA reserves 118–165 GB, the full fine-tune at 8,192 tokens 185–202 GB, the 32,000-token and frozen 50,000-token rows reach 283 GB, and Lightning's full fine-tune reserves 55–60 GB](../assets/benchmarks/nemotron_memory.png)

The full fine-tune at 8,192 takes 21.28 s a step over 20 steps. The 32,000 and frozen 50,000 rows are at the limit: their top ranks reached 99.4% and 99% of the device over 5 steps, so no longer row has a measured fit. Frozen experts train with `freeze_layers_patterns` on `*.mixer.up_proj` and `*.mixer.down_proj`, 7.93B trainable parameters.

The tower's forward and backward alone, with frozen experts and no optimizer state, at exact lengths:

| Tokens per rank | Peak allocated | Reserved |
|---:|---:|---:|
| 16,384 | 96 GB | 108–127 GB |
| 32,768 | 148 GB | 173–214 GB |
| 50,000 | 202 GB | 239–281 GB |

Computed, with no row behind them: 150 GB static per GPU for a full fine-tune at `ep8`, 65 GB with frozen experts and 51 GB for LoRA at r=32; on 4 GPUs at `ep4`, 84 GB static for LoRA and 269 GB for a full fine-tune, which does not fit. Without checkpointing a token holds about 66 MB, or about 1,650 tokens per rank. A full fine-tune at `ep8` needs more memory than an H100 or H200 has.

On disk the text-only export takes 241 GB, the composite one 244 GB, the reattached one 249 GB, the LoRA recipe's folded save 251 GB with its adapter state and a plain adapter save 115 MB. A full resume checkpoint comes to about 725 GB by arithmetic. The eager load holds about 249 GB of host RAM per rank, so stay on the lazy path, the default under EP (`ep_lazy_loading`).

## Rows

- **Packing and padding-free:** the collators emit `seq_idx` and `cu_seq_lens`, and each document computes what it computes alone. Measured on one Mamba layer at 8,192 tokens, 16 × 512 documents cost 1.2× one document's scan and 64 × 128 cost 1.9×.
- **Padded rows:** the text path pads right. An image batch keeps the processor's left padding, and the scan reads the attention mask, so each row scores as it does alone.
- **Tree packing:** refused, as it admits only the families verified on it.

The hub chat template renders every assistant turn with no reasoning as `<think></think>` plus its text, and the generation prompt opens `<think>`, so `assistant_message_template: "<|im_start|>assistant\n"` trains the empty block as part of the completion. The tokenizer pads with its eos while the config records `pad_token_id` 0, so the recipes set `pad_token: <unk>`, which leaves an export's config unchanged.

## Images

The composite takes images through the VLM path, with `image_grid_hw` in the bake, the collators and the forward. FA4 runs both towers ([Flash Attention](../optimization/flash-attention.md#model-specific-handling)). The fused loss stays off, as the composite scores its own loss. Image and text rows mix under `torchrun`: the embedding and the tower share the root group, whose reduce agrees across ranks ([Shard groups](../parallelism/data-parallelism.md#shard-groups)). On a training step with Liger on, the logged `mean_token_accuracy` reads 0 and `entropy` ln(vocab), which mean nothing here; loss, gradients and eval metrics that mask label -100 are right ([Metrics on fp32 logits](../optimization/liger-kernels.md#known-issue-metrics-on-fp32-logits)). Video and audio parts raise: the release has no sound tower, and transformers' video forward does not run on it.

## MTP head

`mtp_loss_weight` [trains the release's head](../optimization/mtp-heads.md#families) at its own keys: `mtp.layers.0` an attention block, `mtp.layers.1` a MoE block, under `language_model.mtp` on the composite route. Super VL's latent applies to its head, and Lightning has none. Its expert bank stores per expert, and its router bias stays fp32.

- **Axes:** EP, pure ETP, EP+ETP and FSDP2, as the trunk's. The head's MoE block takes `EPNemotronHMoELayer`, and its attention takes the backend the run selects.
- **Refused:** the [list every family shares](../optimization/mtp-heads.md#refusals), which covers the family's refused axes. `save_sharded_ep` stays refused, as for any run here, and images are refused as the head trains on text rows. A depth cut with no attention or no MoE layer refuses before the load.
- **Recipe:** `examples/sft/nemotron_h/nemotron-3.5-lightning-mtp-head-only.yaml`.
- **Memory:** by arithmetic, as no row isolates the head. At `ep8` a rank holds 218M of the head's parameters on Lightning and 476M on Super VL: 0.41 and 0.89 GiB of bf16 weights, 1.2 and 2.7 GiB with `AdamWBF16`'s state. Whole, the head's weights are 2.5 and 5.5 GiB. Its dispatch shares the trunk's DeepEP arena, and its blocks' activations are not measured here.
- **Real rows:** an untrained export holds the release's head byte for byte on both releases (270 and 1,040 tensors). A fit moves only `mtp.*`, and its acceptance is in [Hard and soft](../optimization/mtp-heads.md#hard-and-soft). Both engines draft with the head at about 0.83 to 0.84 on Lightning and on a reattached Super VL ([Serving](../optimization/mtp-heads.md#serving)), which shows the layout matches and no served gain. The Lightning recipe ran 6 steps at 9,075 tok/s per rank, at 14.1–14.5 GiB peak allocated, too few to show a fit.

The weight sync carries no head and refuses the family, so a server drafts with the head of its launch checkpoint.

## LoRA

`lora_target_modules` for the recipes: `q_proj`, `k_proj`, `v_proj`, `o_proj` on the attention layers, `in_proj`, `out_proj` on the Mamba layers, and `up_proj`, `down_proj`, which peel to the native expert adapters.

- `lora_r` must be a multiple of 8 for the expert adapters, which are refused at `expert_tp_size > 1`.
- An adapter on `conv1d` raises at the first forward, as the mixer reads its weight directly.
- The latent projections and shared experts have no route.
- A list target also matches Radio's `q_proj` on the composite. Train adapters with `text_only_model: true`, or pass a regex string, which leaves the experts out.
- `merge_peft_adapters.py` merges attention and Mamba adapters and streams the merge of expert ones; the recipe instead folds both with `merge_expert_lora_on_save: true`, a 251 GB write per save.

## Balancing

`auto` resolves to `bias_update` on the router's `gate.e_score_correction_bias` ([Callbacks](../training-methods/callbacks.md#balancingbiascallback)). The forward computes no aux loss. The release stores the bias in fp32 at 4–42, where a bf16 cast would erase its spread, so it stays fp32 through the load and every save. It routes with balancing off as well.

## Checkpoints and serving

The release is plain bf16 with no conversion step. Load it natively, `trust_remote_code: false`. Training the composite through the hub's modeling file is refused at the load (`remote_composite_refusal`): its forward hands the text tower no `seq_idx` and reads no `image_grid_hw`. That file also replaces the conversion mapping for the whole process, so a native load after a remote one is refused: load the family one way per process, a teacher beside a policy included.

- **Text-only export:** `backbone.*` and `lm_head.weight` under transformers' `nemotron_h` schema. It reloads in stock transformers through `AutoModelForCausalLM`, whose forward keeps upstream's time-step floor ([above](#mamba-layers)), so a stock reload smoke-tests the export but does not compute what the run trained.
- **Composite export:** the hub's spellings, per expert, under the release's own `config.json` and modules (`exports_source_config_schema`). It carries `language_model.mtp.*` where the run trained a [head](#mtp-head), and otherwise none, which no tool restores onto it. Stock transformers reads it whole with `trust_remote_code=True`; its native class, without this toolkit's conversion entries, leaves the vision norm at identity.
- **`reattach_vision_tower.py`:** turns a text-only export into the composite, streaming the vision tower and connector back from the base byte for byte. An export that trained a [head](#mtp-head) writes it at `language_model.mtp.*` over the base's. Otherwise the base's 5.48 GiB head streams back at the release's weights, with a warning giving its tensor count and offering a `head_only` fit.
- **Refused:** `save_sharded_ep` and `merge_ep_shards.py`, since the merge cannot respell `backbone.*` and the vision norm. `unfuse_moe_experts.py` copies a checkpoint through, as the layout stores per expert already. Gathered saves stream one 5.25 GiB layer at a time.

Serve a composite with the release's flags. Both engines boot an untrained reattached export and answer an image prompt and a text prompt. `reattach_vision_tower.py` carries the modules its processor and tokenizer configs' `auto_map` names, which SGLang's boot and `AutoProcessor` with `trust_remote_code=True` read.

- **vLLM 0.32.0:** `--trust-remote-code` (without it the boot fails on `vit_hidden_size`), `--mamba-ssm-cache-dtype float32`, `--tensor-parallel-size 2`, `--tool-call-parser qwen3_coder`, `--reasoning-parser nemotron_v3`. The row that served an export started the image directly with these. `docker-compose.vllm.yml` takes the same flags: `VLLM_TRUST_REMOTE_CODE=1`, `VLLM_TP=2`, `VLLM_TOOL_PARSER=qwen3_coder`, `VLLM_REASONING_PARSER=nemotron_v3`, `VLLM_EXTRA_ARGS="--mamba-ssm-cache-dtype float32"`.
- **SGLang 0.5.22:** `SGLANG_TRUST_REMOTE_CODE=1`, `SGLANG_TP=2`, `SGLANG_TOOL_PARSER=qwen3_coder`, `SGLANG_REASONING_PARSER=nemotron_3` on the compose file. The model card's `--mamba-ssm-dtype` flags are for its dedicated image and raise on this one.

## RL weight sync

Online and async GRPO are refused on both engines by the spec's `weight_sync_refusal`, off the checkpoint's config before the scripts read a dataset or load the model, and again at trainer construction. Both engines serve the release, and no online update into either is validated end to end.

**What runs today.** Train offline: serve an export ([above](#checkpoints-and-serving)), collect and score completions from it (`scripts/inference/reward_model/rejection_sample.py --output_format offline_grpo`), and train them with `examples/grpo/offline/nemotron_h/offline-grpo-nemotron-3.5-super-vl-gsm8k-ep8.yaml`, which needs no server ([Offline GRPO](../training-methods/grpo/offline.md)).

**What lifts the refusal.** Both engines passing `trainers/grpo/test_weight_push_exact.py` on the release at the serving `--tp-size 2`, its `--reach` rows included, which starts with a `CHECKPOINTS` entry for the family. vLLM's admission also takes a config-parity fixture under `docker/vllm/parity/fixtures/` ([Config-schema parity](../infrastructure/rollout-servers.md#config-schema-parity)), and either engine's a place in `_SYNC_EXACTNESS_SWEEP_FAMILIES` (`tests/gpu/manifest.py`). The run has to push the composite's names: vLLM's loader drops a name without `language_model.`, which is why these trainers refuse `text_only_model`. The router's `gate.e_score_correction_bias` is a buffer on the trainer and a parameter of vLLM's `GateLinear`, so the push has to carry it (`engine_loaded_buffers`), or the partly loaded gate rebuilds from uninitialized storage ([Partly loaded layers](../infrastructure/rollout-servers.md#partly-loaded-layers)).

The transport preflight's default parameter does not exist on this family: pass `--param language_model.backbone.embeddings.weight` to `weight_sync_transport.py`.

## Pipeline parallelism

`NemotronHModel` is registered on any split offset ([Model support](../parallelism/pipeline-parallelism.md#model-support)), with `text_only_model: true`. The GPU rows run pp2 × ep2 and pp2 × etp2 on tiny models. On Super VL, `pp2 × ep8` across two 8-GPU machines ran 13 steps with GC on at 13.69 s a step, 2,394 tok/s/GPU as logged (padding counted), at 100.9 / 96.8 GiB; with GC off the first step runs out of memory at about 260 GiB.

## Refused modes

- **CP:** the scan is a recurrence along the sequence axis.
- **TP:** `NemotronHAttention` is not in `TP_SHARDABLE_ATTENTION_CLASSES`. Attention is 0.24% of the parameters, and two KV heads would cap it at 2.
- **HSDP:** refused with EP, and pure DP cannot hold the model.
