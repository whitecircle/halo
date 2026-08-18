"""Model names, tiny random-init configs, and profiling benchmark configs, for every test file.

One home per constant: a checkpoint a test loads, a local patched-checkpoint path, or a tiny
config a family's branches need. A test that inlines a model name or a tiny config instead splits
the roster in two.
"""

# Dense Models

QWEN3_0_6B = "Qwen/Qwen3-0.6B"
QWEN3_4B_INSTRUCT = "Qwen/Qwen3-4B-Instruct-2507"
QWEN3_8B = "Qwen/Qwen3-8B"
QWEN3_5_2B = "Qwen/Qwen3.5-2B"

# MoE Models

GPT_OSS_20B = "unsloth/gpt-oss-20b-BF16"
GPT_OSS_120B = "unsloth/gpt-oss-120b-BF16"
QWEN3_30B_A3B = "Qwen/Qwen3-30B-A3B-Instruct-2507"
GLM4_FLASH = "zai-org/GLM-4.7-Flash"
LFM2_24B_A2B = "LiquidAI/LFM2-24B-A2B"
QWEN3_5_MOE = "Qwen/Qwen3.5-397B-A17B"
QWEN3_5_MOE_35B = "Qwen/Qwen3.5-35B-A3B"
QWEN3_5_MOE_122B = "Qwen/Qwen3.5-122B-A10B"
BAILING_MOE_RING_MINI = "inclusionAI/Ring-mini-linear-2.0"
BAILING_MOE_LING_MINI = "inclusionAI/Ling-mini-2.0"
BAILING_LING_3_TINY = "inclusionAI/Ling-3.0-tiny"
GEMMA4_26B_A4B = "google/gemma-4-26B-A4B-it"
LAGUNA_S_2_1 = "poolside/Laguna-S-2.1"
MISTRAL3_119B_MOE = "mistralai/Mistral-Small-4-119B-2603"
COMMAND_A_PLUS = "CohereLabs/command-a-plus-05-2026-bf16"
ZAYA_8B = "Zyphra/ZAYA1-8B"

# Vocab-patched local checkpoints (scripts/before_training/patch_vocab.py output). The hub copies
# carry a vocab the toolkit's tokenizer alignment rejects, so these tests need the patched dir.

GEMMA4_26B_A4B_PATCHED = "/mnt/models/gemma-4-26B-A4B-it-patched"
GPT_OSS_20B_PATCHED = "/mnt/models/gpt-oss-20b-BF16-patched"

# Tiny random-init configs (no hub checkpoint exists at test scale)

# DeepSeek-V4 tiny config exercising every architectural branch: a hash_moe bootstrap layer +
# top-k (noaux_tc-style e_score_correction_bias) layers, all three attention layer types (sliding /
# CSA+indexer / HCA), MLA-style q_lora_rank + shared-KV MQA + grouped output projection,
# sqrtsoftplus scoring, a shared expert, and the clamped SwiGLU (swiglu_limit). MTP stays inert
# (num_nextn_predict_layers is metadata only). hidden_size=256 keeps the DeepEP transport pad
# (multiple of 256) exact; tests pairing the config with a real tokenizer override vocab_size.
TINY_DSV4_CONFIG = {
    "vocab_size": 2048,
    "hidden_size": 256,
    "moe_intermediate_size": 128,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "num_key_value_heads": 1,
    "head_dim": 64,
    "q_lora_rank": 64,
    "num_experts_per_tok": 2,
    "n_routed_experts": 8,
    "scoring_func": "sqrtsoftplus",
    "routed_scaling_factor": 1.5,
    "max_position_embeddings": 4096,
    "layer_types": ["sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention"],
    "mlp_layer_types": ["hash_moe", "moe", "moe"],
    "compress_rates": {"compressed_sparse_attention": 4, "heavily_compressed_attention": 16},
    "swiglu_limit": 10.0,
    "sliding_window": 64,
    "o_groups": 2,
    "o_lora_rank": 32,
    "index_n_heads": 2,
    "index_head_dim": 32,
    "index_topk": 16,
    "hc_mult": 2,
    "hc_sinkhorn_iters": 4,
    "num_nextn_predict_layers": 1,
    "partial_rotary_factor": 0.25,
}

# Tiny gpt-oss MoE for PP+EP tests (GptOssForCausalLM): hidden 256 keeps the DeepEP transport pad
# (multiple of 256) exact; 8 layers with the family's period-2 layer_types (sliding/full alternation)
# make the pp2 stage offset (4) a whole number of periods; 8 experts top-2 split evenly at ep2/ep4.
# router_aux_loss_coef=0 because a PP stage severs the HF aux-loss path (the PP split gate rejects a
# live coefficient); attention stays eager so the tiny random model needs no FA kernels or sinks
# handling. tie_word_embeddings off — PP rejects tied embeddings.
TINY_GPTOSS_CONFIG = {
    "vocab_size": 1024,
    "hidden_size": 256,
    "intermediate_size": 256,
    "num_hidden_layers": 8,
    "num_local_experts": 8,
    "num_experts_per_tok": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "sliding_window": 64,
    "router_aux_loss_coef": 0.0,
    "tie_word_embeddings": False,
    "attn_implementation": "eager",
}

# Tiny Qwen3-MoE for the PP+EP load-equivalence matrix: transformers saves its experts as one module
# per expert (``experts.{i}.gate_proj.weight``), so a checkpoint written from it is the INDIVIDUAL
# format the lazy loader routes through ExpertFuser — the path GptOss's fused layout never reaches.
# 8 layers so a pp2 split is non-trivial on either side.
TINY_QWEN3_MOE_CONFIG = {
    "vocab_size": 1024,
    "hidden_size": 256,
    "intermediate_size": 256,
    "moe_intermediate_size": 128,
    "num_hidden_layers": 8,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "decoder_sparse_step": 1,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "max_position_embeddings": 512,
    "router_aux_loss_coef": 0.0,
    "tie_word_embeddings": False,
    "attn_implementation": "eager",
}

# Cohere2 MoE (Command A+) tiny config exercising the family's branches: interleaved sliding (RoPE)
# / full (NoPE) attention, top-k-then-sigmoid selection with top-k renorm, a fused shared expert
# combined by "average", the parallel attention+MLP residual block, LayerNorm (rms_norm_eps=None),
# logit scaling, and tied embeddings. hidden_size=256 keeps the DeepEP transport pad (multiple of
# 256) exact. vocab_size is overridden by tests that pair the config with a real tokenizer.
TINY_COHERE2_MOE_CONFIG = {
    "vocab_size": 2048,
    "hidden_size": 256,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "num_shared_experts": 2,
    "shared_expert_combination_strategy": "average",
    "expert_selection_fn": "sigmoid",
    "norm_topk_prob": True,
    "layer_types": ["sliding_attention", "full_attention"],
    "mlp_layer_types": ["sparse", "sparse"],
    "sliding_window": 64,
    "max_position_embeddings": 4096,
    "logit_scale": 0.0625,
    "tie_word_embeddings": True,
}

# Tiny Gemma 4 MoE TEXT config at the hub's attention geometry: sliding layers at ``head_dim`` /
# ``num_key_value_heads``, full-attention layers at ``global_head_dim`` / ``num_global_key_value_heads``
# under ``attention_k_eq_v`` (google/gemma-4-26B-A4B-it: 256/512 and 8/2). transformers 5.16 folds the
# two global keys into ``per_layer_config``, so this exercises the export-side flatten the rollout
# server's transformers 5.14 needs — homogeneous heads make that rewrite a no-op. ``enable_moe_block``
# is off by default (no experts to wrap without it); the per-layer-input table is shrunk from 262144x256.
TINY_GEMMA4_MOE_CONFIG = {
    "vocab_size": 128,
    "vocab_size_per_layer_input": 128,
    "hidden_size": 32,
    "hidden_size_per_layer_input": 16,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "head_dim": 8,
    "global_head_dim": 16,
    "num_global_key_value_heads": 2,
    "attention_k_eq_v": True,
    "max_position_embeddings": 128,
    "sliding_window": 16,
    "enable_moe_block": True,
    "num_experts": 4,
    "top_k_experts": 2,
    "moe_intermediate_size": 16,
}

# Tiny Qwen3.5-MoE TEXT config (the composite ``qwen3_5_moe`` wrapper is what the rollout server
# registers — a ``qwen3_5_moe_text`` config is refused there): the family's period-4 L,L,L,F
# GatedDeltaNet/full-attention interleave, a shared expert, 8 experts top-2.
TINY_QWEN35_MOE_CONFIG = {
    "vocab_size": 1024,
    "hidden_size": 128,
    "intermediate_size": 256,
    "moe_intermediate_size": 64,
    "shared_expert_intermediate_size": 64,
    "num_hidden_layers": 4,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 32,
    "linear_value_head_dim": 32,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "max_position_embeddings": 512,
    "router_aux_loss_coef": 0.0,
    "tie_word_embeddings": False,
}

# Tiny GLM-4 MoE Lite (GLM-4.7-Flash, ``glm4_moe_lite``): dense first layer + one sparse layer,
# MLA attention (``kv_lora_rank`` / rope + nope head halves), grouped top-k with one group, one
# shared expert.
TINY_GLM4_MOE_LITE_CONFIG = {
    "vocab_size": 1024,
    "hidden_size": 128,
    "intermediate_size": 256,
    "moe_intermediate_size": 64,
    "num_hidden_layers": 2,
    "first_k_dense_replace": 1,
    "n_routed_experts": 8,
    "num_experts_per_tok": 2,
    "n_shared_experts": 1,
    "n_group": 1,
    "topk_group": 1,
    "norm_topk_prob": True,
    "routed_scaling_factor": 1.0,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "kv_lora_rank": 32,
    "q_lora_rank": None,
    "qk_rope_head_dim": 16,
    "qk_nope_head_dim": 16,
    "v_head_dim": 32,
    "max_position_embeddings": 512,
    "tie_word_embeddings": False,
}

# Tiny Laguna (``poolside/Laguna-S-2.1``) at the release's period-4 shape: dense first layer + sparse
# MoE layers, one full-attention layer per period carrying FEWER heads than the sliding ones (the
# family spells that heterogeneity as a plain ``num_attention_heads_per_layer`` list, not the
# ``per_layer_config`` fold), per-head gating, and a shared expert on every token. hidden_size=256
# keeps the DeepEP transport pad (multiple of 256) exact.
TINY_LAGUNA_CONFIG = {
    "vocab_size": 1024,
    "hidden_size": 256,
    "intermediate_size": 256,
    "moe_intermediate_size": 64,
    "shared_expert_intermediate_size": 64,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "num_attention_heads_per_layer": [4, 8, 8, 8],
    "layer_types": ["full_attention", "sliding_attention", "sliding_attention", "sliding_attention"],
    "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
    "gating": "per-head",
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "moe_routed_scaling_factor": 2.5,
    "sliding_window": 64,
    "max_position_embeddings": 512,
    "router_aux_loss_coef": 0.0,
    "tie_word_embeddings": False,
}

# Tiny LFM-2 MoE (``LiquidAI/LFM2-24B-A2B``): the hybrid short-conv / full-attention stack, the first
# ``num_dense_layers`` layers dense and the rest sparse, sigmoid routing with the ``expert_bias``
# selection buffer. hidden_size=256 keeps the DeepEP transport pad (multiple of 256) exact;
# tie_word_embeddings stays True — the released checkpoints carry no ``lm_head``.
TINY_LFM2_MOE_CONFIG = {
    "vocab_size": 1024,
    "hidden_size": 256,
    "intermediate_size": 256,
    "moe_intermediate_size": 64,
    "num_hidden_layers": 4,
    "num_dense_layers": 2,
    "layer_types": ["conv", "conv", "full_attention", "conv"],
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "use_expert_bias": True,
    "norm_topk_prob": True,
    "routed_scaling_factor": 1.0,
    "conv_L_cache": 3,
    "max_position_embeddings": 512,
    "tie_word_embeddings": True,
}

# Tiny Bailing MoE V2 (``inclusionAI/Ling-mini-2.0``): dense first layer + sparse MoE layers, sigmoid
# group-limited routing (top-1 of 2 groups, then top-2 experts) with the ``expert_bias`` selection
# buffer, one shared expert. hidden_size=256 keeps the DeepEP transport pad (multiple of 256) exact;
# the vendor defaults for ``pad_token_id`` / ``eos_token_id`` exceed a tiny vocab. The family is
# remote-code — its config class comes from the module a checkpoint ships, not from transformers.
TINY_BAILING_MOE_CONFIG = {
    "vocab_size": 1024,
    "hidden_size": 256,
    "intermediate_size": 256,
    "moe_intermediate_size": 64,
    "num_hidden_layers": 4,
    "first_k_dense_replace": 1,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "num_experts": 8,
    "num_shared_experts": 1,
    "num_experts_per_tok": 2,
    "n_group": 2,
    "topk_group": 1,
    "routed_scaling_factor": 2.5,
    "moe_router_enable_expert_bias": True,
    "num_nextn_predict_layers": 0,
    "max_position_embeddings": 512,
    "pad_token_id": 0,
    "eos_token_id": 1,
    "tie_word_embeddings": False,
}

# Tiny dense Qwen3.5 (text) for pipeline-parallelism tests: 8 layers keep the family's period-4
# L,L,L,F layer_types pattern intact across a pp2 split (offset 4 = one whole period), so both
# GatedDeltaNet linear-attention layers and full-attention layers sit on every stage.
TINY_QWEN35_CONFIG = {
    "vocab_size": 1024,
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_hidden_layers": 8,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 32,
    "linear_value_head_dim": 32,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "max_position_embeddings": 512,
    "tie_word_embeddings": False,
}

# Tiny dense Qwen3 for pipeline-parallelism tests: 8 layers so a 2- or 4-stage split is exact and
# every stage still holds several layers. tie_word_embeddings is off so the embedding/head tie is
# exercised by its own dedicated test rather than confounding the base equivalence gate.
TINY_QWEN3_CONFIG = {
    "vocab_size": 1024,
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_hidden_layers": 8,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "max_position_embeddings": 512,
    "tie_word_embeddings": False,
}

# Inkling (Thinking Machines) tiny config. hidden_size=256 keeps the DeepEP transport pad (multiple
# of 256) exact; n_routed_experts must stay divisible by the EP size under test. The routed/shared
# joint normalisation needs n_shared_experts > 0 to be exercised at all.
TINY_INKLING_CONFIG = {
    "hidden_size": 256,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "vocab_size": 512,
    "n_routed_experts": 8,
    "n_shared_experts": 2,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 64,
    "dense_intermediate_size": 128,
    "local_layer_ids": [0],
    "dense_mlp_idx": 0,
    "sliding_window_size": 32,
    "max_position_embeddings": 256,
}

# Tiny Zaya (Zyphra ZAYA1) for the native-support + load-time patch tests. num_hidden_layers=4 spans
# both mixer variants (layer 0 carries no EDA state, later layers do) and the CCA convolution whose
# state crosses packed document boundaries. router_hidden_size instantiates the family's MLP gate —
# the home of the native balancing_biases buffer and its trailing discard slot — and top-1 over 2
# experts makes the expected per-token load count exact.
TINY_ZAYA_CONFIG = {
    "vocab_size": 512,
    "hidden_size": 64,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "max_position_embeddings": 64,
    "moe_intermediate_size": 128,
    "num_experts": 2,
    "num_experts_per_tok": 1,
    "router_hidden_size": 32,
    "tie_word_embeddings": True,
    "use_cache": False,
}

# GLM-5 Next (GLM-5.3-Flash) tiny TEXT-TOWER config covering every family branch: dense first layer +
# sparse MoE layers (default ``mlp_layer_types`` leaves a model this shallow all-dense), the default KDA/DSA
# interleave (3 linear-attention layers + 1 deepseek-sparse-attention layer with the indexer), sigmoid noaux-tc
# routing with ``e_score_correction_bias``, one shared expert, clamped SwiGLU and hyper-connections. hidden_size=256
# keeps the DeepEP transport pad (multiple of 256) exact; config validation requires num_key_value_heads ==
# num_attention_heads and qk_rope_head_dim 0 (NoPE DSA); pad_token_id None (default 154820 exceeds a tiny vocab).
TINY_GLM5_CONFIG = {
    "vocab_size": 2048,
    "hidden_size": 256,
    "intermediate_size": 256,
    "moe_intermediate_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "n_routed_experts": 8,
    "num_experts_per_tok": 2,
    "n_shared_experts": 1,
    "n_group": 1,
    "topk_group": 1,
    "norm_topk_prob": True,
    "routed_scaling_factor": 2.5,
    "swiglu_limit": 10.0,
    "kv_lora_rank": 64,
    "q_lora_rank": 64,
    "qk_nope_head_dim": 32,
    "v_head_dim": 32,
    "index_topk": 16,
    "index_kpool": 4,
    "index_n_heads": 2,
    "index_head_dim": 32,
    "linear_num_heads": 4,
    "linear_head_dim": 64,
    "hc_mult": 2,
    "hc_sinkhorn_iters": 4,
    "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
    "max_position_embeddings": 4096,
    "pad_token_id": None,
    "tie_word_embeddings": False,
}

# Tiny vision tower for the glm5_next composite wrapper (the family ships no text-only CausalLM, so
# every generative test builds ``Glm5NextForConditionalGeneration``). out_hidden_size must equal the
# text tower's hidden_size — it is the projector's output width.
TINY_GLM5_VISION_CONFIG = {
    "depth": 2,
    "hidden_size": 32,
    "num_heads": 2,
    "intermediate_size": 64,
    "out_hidden_size": 256,
    "projection_intermediate_size": 64,
    "patch_size": 4,
    "image_size": 32,
    "temporal_patch_size": 2,
    "spatial_merge_size": 2,
}

# Step-3.7 (Step-3.7-Flash) tiny TEXT-TOWER config covering every family branch: dense first layer + sparse MoE
# layers, PER-LAYER post-activation SwiGLU clamps on the last two MoE layers only (0 = unclamped elsewhere,
# shared bounds on a separate list), sigmoid routing with the ``e_score_correction_bias`` buffer, non-neutral
# ``moe_router_scaling_factor``, and a full/sliding attention interleave with a DIFFERENT sliding head count —
# heterogeneous, so a global ``config.num_attention_heads`` read raises AmbiguousGlobalPerLayerAttributeError
# and any path skipping ``per_layer_config`` trips. hidden_size=256 keeps the DeepEP transport pad exact.
TINY_STEP3P7_CONFIG = {
    "vocab_size": 2048,
    "hidden_size": 256,
    "intermediate_size": 256,
    "moe_intermediate_size": 128,
    "share_expert_dim": 64,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "num_sliding_attention_heads": 2,
    "layer_types": ["full_attention", "sliding_attention", "full_attention", "sliding_attention"],
    "sliding_window": 64,
    "n_routed_experts": 8,
    "num_experts_per_tok": 2,
    "moe_router_scaling_factor": 2.5,
    "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
    "swiglu_limits": [0, 0, 0.5, 0.5],
    "swiglu_limits_shared": [0, 0, 0.4, 0.4],
    "max_position_embeddings": 4096,
    "tie_word_embeddings": False,
}

# Tiny vision tower for the step3p7 composite wrapper (the family ships no text-only CausalLM, so
# every generative test builds ``Step3p7ForConditionalGeneration``). No output-width field: the
# projector maps ``hidden_size * 4`` (two stride-2 downsamplers) to the text hidden_size itself.
# ``intermediate_size`` is derived from ``mlp_ratio``; the 56/14 grid is 4x4, the smallest the two
# stride-2 downsamplers accept cleanly.
TINY_STEP3P7_VISION_CONFIG = {
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "image_size": 56,
    "patch_size": 14,
    "mlp_ratio": 2.0,
    "max_position_embeddings": 16,
}

# Special Models

QWEN3_5_VLM_4B = "Qwen/Qwen3.5-4B"  # Natively multimodal (Image-Text-to-Text)
QWEN3_VL_2B = "Qwen/Qwen3-VL-2B-Instruct"
PARAPHRASE_MINILM = "sentence-transformers/paraphrase-MiniLM-L3-v2"

# Profiling Benchmark Configs

MODEL_CONFIGS = {
    "gpt-oss-20b": {
        "hf_name": GPT_OSS_20B,
        "full_params": 20.7e9,
        "num_experts": 32,
        "top_k": 4,
    },
    "gpt-oss-120b": {
        "hf_name": GPT_OSS_120B,
        "full_params": 116.8e9,
        "num_experts": 128,
        "top_k": 4,
    },
    "qwen3-30b-a3b": {
        "hf_name": QWEN3_30B_A3B,
        "full_params": 30e9,
        "num_experts": 128,
        "top_k": 8,
    },
    "glm4-flash": {
        "hf_name": GLM4_FLASH,
        "full_params": 30e9,
        "num_experts": 32,
        "top_k": 2,
    },
    "qwen3.5-moe": {
        "hf_name": QWEN3_5_MOE,
        "full_params": 397e9,
        "num_experts": 512,
        "top_k": 10,
    },
    "qwen3.5-35b-a3b": {
        "hf_name": QWEN3_5_MOE_35B,
        "full_params": 35e9,
        "num_experts": 256,
        "top_k": 8,
    },
    "qwen3.5-122b-a10b": {
        "hf_name": QWEN3_5_MOE_122B,
        "full_params": 122e9,
        "num_experts": 256,
        "top_k": 8,
    },
    "qwen3-8b": {
        "hf_name": QWEN3_8B,
        "full_params": 8.2e9,
        "num_experts": 0,
        "top_k": 0,
    },
    "qwen3-4b": {
        "hf_name": QWEN3_4B_INSTRUCT,
        "full_params": 4.02e9,
        "num_experts": 0,
        "top_k": 0,
    },
    "qwen3-0.6b": {
        "hf_name": QWEN3_0_6B,
        "full_params": 0.6e9,
        "num_experts": 0,
        "top_k": 0,
    },
    "bailing-ring-mini": {
        "hf_name": BAILING_MOE_RING_MINI,
        "full_params": 16.4e9,
        "num_experts": 256,
        "top_k": 8,
        "trust_remote_code": True,
    },
    "laguna-s-2.1": {
        "hf_name": LAGUNA_S_2_1,
        "revision": "e80da38da3ed4c4e56888cc1ba39582946a164ba",
        "full_params": 118e9,
        "num_experts": 256,
        "top_k": 10,
        "trust_remote_code": True,
    },
    "gemma4-26b-a4b": {
        "hf_name": GEMMA4_26B_A4B,
        "full_params": 26e9,
        "num_experts": 128,
        "top_k": 8,
    },
    "mistral3-119b-moe": {
        "hf_name": MISTRAL3_119B_MOE,
        "full_params": 119e9,
        "num_experts": 128,
        "top_k": 4,
    },
}

DEFAULT_MODEL = "gpt-oss-20b"
