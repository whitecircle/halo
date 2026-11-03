# NemotronH

NVIDIA's `NemotronHForCausalLM` (`nemotron_h`, Nemotron 3.5 Lightning): a hybrid stack in which each layer holds one mixer — Mamba-2, MoE, or GQA attention — selected by `config.layers_block_type`. The MoE layers route DeepSeek-V3 style (sigmoid scores, `e_score_correction_bias`, a shared expert) to **non-gated** relu² experts, optionally behind a latent projection. Native in transformers; no `trust_remote_code`.

| | EP | CP | TP | ETP | PP | EP+TP |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| NemotronH | Yes | **No** | **No** | Yes | — | **No** |

Pipeline parallelism is [not yet available in this release](../parallelism/pipeline-parallelism.md).

## EP wrapper

`EPNemotronHMoELayer` (`src/distributed/expert_parallel/layers/nemotron_h.py`) replaces `NemotronHMoE`.

- Routing: the shared group-limited body (`EPGroupLimitedMoELayerBase`). The router matmul runs in fp32, as `NemotronHTopkRouter` runs it; `e_score_correction_bias` biases selection only, and the gate weights are renormalized (`norm_topk_prob`) and scaled by `routed_scaling_factor`.
- Experts: `up_proj` and `down_proj` only, computed as `down(relu²(up(x)))` — two grouped-GEMM calls per layer. Under ETP each projection shards its intermediate axis directly; there is no gate half to keep aligned.
- Latent projection: where the block sets `moe_latent_size`, `fc1_latent_proj` runs before the dispatch and `fc2_latent_proj` after the combine, so DeepEP moves latent-width rows. Both projections are replicated and DP-averaged with the shared expert.
- Export: gathered saves write the hub's per-expert layout, `experts.{i}.up_proj.weight` / `experts.{i}.down_proj.weight`, which `from_pretrained` merges back. The sharded-save merge writes the same keys.

The checkpoint loads through `from_pretrained` and is patched (`_supports_lazy_loading = False`): the lazy loader's per-expert vocabulary is the gated `(gate, up, down)` triple, where a bare `up_proj` is a GLU half. RL weight sync is refused (`_supports_weight_sync = False`); no end-to-end update into a served NemotronH has been validated.

## Mamba-2 kernels

The images do not install `mamba_ssm`. On CUDA, the kernel dispatch (`src/models/patches/kernel_dispatch.py`) runs NemotronH's Mamba-2 scan (`mamba_chunk_scan_combined`, and `selective_state_update` for decoding) from the `kernels-community/mamba-ssm` Hub build, pinned to one revision, instead of transformers' torch fallback, which is several times slower, holds far more activation memory, and ignores `seq_idx`. The build is fetched into the HF cache when the model loads, so an offline run needs it cached beforehand. The build's fused `mamba_split_conv1d_scan_combined` is not used: it requires a `causal_conv1d_cuda` symbol that the installed causal-conv1d 1.6 no longer exports. The mixer therefore runs its unfused path — the installed causal-conv1d kernel for the conv, the Hub scan, then the grouped gated RMSNorm and output projection in torch.

## Packing

Documents stay isolated in every layer. Attention keys on the packed `position_ids`; for the Mamba-2 layers the collators and SMPO's padding-free forward emit `seq_idx` (`segment_markers_for` in `src/models/segment_markers.py`), and `patch_nemotron_h_mixer_segment_markers` (`src/models/patches/nemotron_h.py`) forwards it from `NemotronHBlock` to the mixer, which hands it to the conv and scan kernels.

## Why CP and TP aren't supported

The Mamba-2 mixer scans along the sequence axis, so a Ulysses sequence split would cut its recurrence; CP is refused off the `linear_attention` entries in `layer_types`. `NemotronHAttention` is not in `TP_SHARDABLE_ATTENTION_CLASSES` and the config carries no `base_model_tp_plan`.

## Router balancing

`moe_balancing: auto` resolves to `bias_update`: the forward never returns router logits (no aux-loss route), and the sign-updates land in the router's own `gate.e_score_correction_bias`, which exports with every checkpoint at fp32.

## Configs

`examples/sft/nemotron_h/nemotron-3.5-lightning-30b-a3b-ultrachat-ep8.yaml` trains `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` on UltraChat with EP8, packing and gathered Hugging Face checkpoints. The path is exercised by `tests/gpu/parallelism/ep/test_ep_vs_fsdp_nemotron_h.py` (plain and latent blocks) and `tests/gpu/trainers/sft/test_sft_nemotron_h.py`.
