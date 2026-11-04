"""Model names, tiny random-init configs and profiling benchmark configs shared by the test suites. One constant per
checkpoint, local patched path and tiny config; a test inlining one splits the roster in two."""

from halo.env import data_path

QWEN3_0_6B = "Qwen/Qwen3-0.6B"
QWEN3_4B_INSTRUCT = "Qwen/Qwen3-4B-Instruct-2507"
QWEN3_8B = "Qwen/Qwen3-8B"
QWEN3_5_2B = "Qwen/Qwen3.5-2B"
QWEN3_5_9B = "Qwen/Qwen3.5-9B"

GPT_OSS_20B = "unsloth/gpt-oss-20b-BF16"
GPT_OSS_120B = "unsloth/gpt-oss-120b-BF16"
QWEN3_30B_A3B = "Qwen/Qwen3-30B-A3B-Instruct-2507"
GLM4_FLASH = "zai-org/GLM-4.7-Flash"
LFM2_24B_A2B = "LiquidAI/LFM2-24B-A2B"
QWEN3_5_MOE = "Qwen/Qwen3.5-397B-A17B"
QWEN3_5_MOE_35B = "Qwen/Qwen3.5-35B-A3B"
QWEN3_NEXT_80B_A3B = "Qwen/Qwen3-Next-80B-A3B-Instruct"
QWEN3_8_FLASH_NEXT = "Qwen/Qwen3.8-Flash-Next"
NEMOTRON_3_5_SUPER_VL = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"
NEMOTRON_3_5_LIGHTNING = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
QWEN3_5_MOE_122B = "Qwen/Qwen3.5-122B-A10B"
QWEN3_6_MOE_35B = "Qwen/Qwen3.6-35B-A3B"
BAILING_MOE_RING_MINI = "inclusionAI/Ring-mini-linear-2.0"
BAILING_MOE_LING_MINI = "inclusionAI/Ling-mini-2.0"
BAILING_LING_3_TINY = "inclusionAI/Ling-3.0-tiny"
GEMMA4_26B_A4B = "google/gemma-4-26B-A4B-it"
LAGUNA_S_2_1 = "poolside/Laguna-S-2.1"
LAGUNA_XS_2_1 = "poolside/Laguna-XS-2.1"
MISTRAL3_119B_MOE = "mistralai/Mistral-Small-4-119B-2603"
COMMAND_A_PLUS = "CohereLabs/command-a-plus-05-2026-bf16"
ZAYA_8B = "Zyphra/ZAYA1-8B"
STEP3P7_FLASH = "stepfun-ai/Step-3.7-Flash"

# Local patch_vocab.py outputs under the toolkit scratch root; a suite defaulting to one skips when it is absent.


def patched_checkpoint_dir(hub_id: str) -> str:
    """``$HALO_DATA_ROOT/models/<repo name>-patched``, where the documented command writes ``hub_id``."""
    return data_path("models", f"{hub_id.rsplit('/', 1)[-1]}-patched")


GEMMA4_26B_A4B_PATCHED = patched_checkpoint_dir(GEMMA4_26B_A4B)
GPT_OSS_20B_PATCHED = patched_checkpoint_dir(GPT_OSS_20B)

# Tiny random-init configs. hidden_size=256 keeps the DeepEP transport pad (multiple of 256) exact.

# Every architectural branch: hash_moe bootstrap + top-k layers, all three attention layer types, MLA + shared-KV MQA +
# grouped output projection, sqrtsoftplus scoring, a shared expert, clamped SwiGLU. MTP stays inert.
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

# 8 layers with period-2 layer_types make the pp2 offset a whole number of periods; 8 experts top-2 split evenly at
# ep2/ep4. router_aux_loss_coef=0 and untied since the PP split gate rejects a live coefficient and tied embeddings;
# eager attention needs no FA kernels or sinks handling.
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

# Saved one module per expert, the individual format the lazy loader routes through ExpertFuser. 8 layers for a real
# pp2 split.
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

# Interleaved sliding (RoPE) / full (NoPE) attention, top-k-then-sigmoid with renorm, a fused shared expert combined by
# "average", the parallel attention+MLP residual block, LayerNorm (rms_norm_eps=None), logit scaling, tied embeddings.
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

# At the hub's attention geometry (sliding vs full ``head_dim`` and KV heads, under ``attention_k_eq_v``), which
# transformers 5.20 folds into ``per_layer_config``. ``enable_moe_block`` is off by default: no experts to wrap without
# it.
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

# Wide global heads (``global_head_dim`` 512, past flash's 256), the shape that resolves ``sdpa`` to flex-sliding.
# Callers add ``layer_types``.
TINY_GEMMA4_WIDE_HEAD_CONFIG = {
    "vocab_size": 256,
    "hidden_size": 128,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "global_head_dim": 512,
    "num_global_key_value_heads": 1,
    "sliding_window": 32,
    "enable_moe_block": False,
    "hidden_size_per_layer_input": 0,
    "num_kv_shared_layers": 0,
}

# Text config, since the rollout server refuses ``qwen3_5_moe_text``: period-4 L,L,L,F interleave, a shared expert, 8
# experts top-2.
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
# Qwen3.5-MoE's text fields, which ``Qwen3NextConfig`` spells alike, with the router renormalizing as the 80B-A3B ships
# it.
TINY_QWEN3_NEXT_CONFIG = {**TINY_QWEN35_MOE_CONFIG, "norm_topk_prob": True}
# Cut from the release config: three GatedDeltaNet layers and a QSA layer, four hyper-connection streams, PLE on layer
# 2 over a 4-part n-gram table, a sigmoid-gated shared expert beside 8 routed (top-3), interleaved M-RoPE. An 8-token
# ``indexer_budget`` lets a short document cross it.
TINY_QWEN4_EXP_CONFIG = {
    "vocab_size": 1024,
    "hidden_size": 64,
    "num_hidden_layers": 4,
    "layer_types": ["linear_attention"] * 3 + ["full_attention"],
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "rope_parameters": {
        "rope_type": "default",
        "rope_theta": 10000000,
        "partial_rotary_factor": 0.25,
        "mrope_section": [3, 3, 2],
        "mrope_interleaved": True,
    },
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 16,
    "linear_value_head_dim": 16,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 6,
    "num_experts": 8,
    "num_experts_per_tok": 3,
    "moe_intermediate_size": 32,
    "shared_expert_intermediate_size": 32,
    "norm_topk_prob": True,
    "hc_count": 4,
    "hc_lowrank": 16,
    "ple_layer_ids": [2],
    "ple_embed_dim": 64,
    "ngram_size": 3,
    "heads_per_ngram": 2,
    "ngram_vocab_size_base": 997,
    "split_ngram_parts": 4,
    "indexer_n_heads": 2,
    "indexer_kv_heads": 1,
    "indexer_head_dim": 16,
    "indexer_budget": 8,
    "indexer_compress_ratio": 4,
    "eos_token_id": 1,
    "max_position_embeddings": 4096,
    "router_aux_loss_coef": 0.0,
    "tie_word_embeddings": False,
}
# A MoE family no EP layer class wraps.
TINY_MIXTRAL_CONFIG = {
    "vocab_size": 1024,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_local_experts": 4,
    "num_experts_per_tok": 2,
    "max_position_embeddings": 512,
    "router_aux_loss_coef": 0.0,
    "tie_word_embeddings": False,
}

# GLM-4.7-Flash: dense first layer + one sparse layer, MLA attention, grouped top-k with one group, one shared expert.
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

# At the release's period-4 shape: dense first layer + sparse layers, one full-attention layer per period with fewer
# heads (a plain ``num_attention_heads_per_layer`` list, no ``per_layer_config`` fold), per-head gating, a shared
# expert on every token.
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
# The attention pattern repeats every four layers, so a pipeline split in two needs eight to cut on a period boundary.
TINY_LAGUNA_PP_OVERRIDES = {
    "num_hidden_layers": 8,
    "num_attention_heads_per_layer": TINY_LAGUNA_CONFIG["num_attention_heads_per_layer"] * 2,
    "layer_types": TINY_LAGUNA_CONFIG["layer_types"] * 2,
    "mlp_layer_types": TINY_LAGUNA_CONFIG["mlp_layer_types"] * 2,
}

# Hybrid short-conv / full-attention stack, first layers dense, sigmoid routing with the ``expert_bias`` selection
# buffer. Tied, since the released checkpoints carry no ``lm_head``.
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

# Dense first layer + sparse layers, sigmoid group-limited routing with the ``expert_bias`` buffer, one shared expert.
# Vendor pad/eos ids exceed a tiny vocab. Remote-code: the config class comes from the module a checkpoint ships.
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

# Ring-mini-linear-2.0: Lightning Attention-2 in layers 0 and 2, softmax attention in 1 and 3.
TINY_BAILING_RING_CONFIG = {**TINY_BAILING_MOE_CONFIG, "layer_group_size": 2}

# Ling-3.0 with every layer MLA, so a forward shows what its attention mask does alone. ``qk_head_dim`` is derived at
# config init, so it is set with its parts.
TINY_BAILING_LING3_MLA_CONFIG = {
    **TINY_BAILING_MOE_CONFIG,
    "layer_group_size": 1,
    "num_key_value_heads": 4,
    "q_lora_rank": 64,
    "kv_lora_rank": 64,
    "qk_nope_head_dim": 32,
    "qk_rope_head_dim": 16,
    "qk_head_dim": 48,
    "v_head_dim": 32,
    "rope_scaling": {"rope_type": "default", "factor": 1.0, "mscale_all_dim": 0},
}

# 8 layers keep the period-4 pattern intact across a pp2 split, so both layer kinds sit on every stage.
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

# 8 layers so a 2- or 4-stage split is exact and every stage still holds several.
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

# Sliding-window layers: Mistral windows every layer from the config alone, Cohere2 and OLMo 3 alternate sliding and
# full. Packed rows and padded batches of a few hundred tokens outrun the 64-token window.
TINY_MISTRAL_SLIDING_CONFIG = {
    "vocab_size": 256,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "max_position_embeddings": 512,
    "sliding_window": 64,
    "pad_token_id": 0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "tie_word_embeddings": False,
}
TINY_ALTERNATING_SLIDING_CONFIG = {
    **TINY_MISTRAL_SLIDING_CONFIG,
    "layer_types": ["sliding_attention", "full_attention"],
}

# The optimizer-resume suite's model, dense or (with the expert fields below) Qwen3-MoE; its tokenizer sets vocab_size.
TINY_TIED_QWEN3_CONFIG = {
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "max_position_embeddings": 4096,
    "tie_word_embeddings": True,
}
TINY_TIED_QWEN3_MOE_FIELDS = {
    "moe_intermediate_size": 128,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "decoder_sparse_step": 1,
    "mlp_only_layers": [],
}

# Head-transform contract: the forward divides logits by ``logits_scaling``, so a head path skipping it scores 8x
# sharper. Four layers for an exact pp2 split; untied since PP rejects tied embeddings.
TINY_GRANITE_CONFIG = {
    "vocab_size": 256,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 256,
    "logits_scaling": 8.0,
    "tie_word_embeddings": False,
}

# Head-transform contract: the forward divides the hidden state by ``hidden_size / dim_model_base`` (64 / 24). Untied
# (PP).
TINY_MINICPM3_CONFIG = {
    "vocab_size": 256,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "qk_nope_head_dim": 16,
    "qk_rope_head_dim": 8,
    "v_head_dim": 16,
    "kv_lora_rank": 32,
    "q_lora_rank": 32,
    "dim_model_base": 24,
    "max_position_embeddings": 256,
    "tie_word_embeddings": False,
}

# Head-transform contract: ``final_logit_softcapping`` is small enough for random-init logits to saturate the cap. Four
# layers cut on a whole period of the sliding/full interleave; untied (PP).
TINY_GEMMA2_CONFIG = {
    "vocab_size": 256,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "max_position_embeddings": 256,
    "final_logit_softcapping": 0.05,
    "tie_word_embeddings": False,
}

# n_routed_experts must stay divisible by the EP size under test; the routed/shared joint normalization needs
# n_shared_experts > 0.
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
# Sliding layers come every second layer, so a pipeline split in two needs four to cut on a period boundary.
TINY_INKLING_PP_OVERRIDES = {"num_hidden_layers": 4, "local_layer_ids": [0, 2]}

# 4 layers span both mixer variants (EDA state from layer 1 on) and the CCA conv whose state crosses packed documents.
# router_hidden_size builds the MLP gate holding ``balancing_biases``; top-1 over 2 experts makes per-token load counts
# exact.
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

# Dense first layer + sparse layers (default ``mlp_layer_types`` leaves this depth all-dense), the KDA/DSA interleave,
# sigmoid noaux-tc routing, one shared expert, clamped SwiGLU, hyper-connections. Validation requires KV heads == heads
# and qk_rope_head_dim 0. pad_token_id None: the default 154820 exceeds a tiny vocab.
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

# The family ships no text-only CausalLM. out_hidden_size is the projector's output width, so it equals the text
# hidden_size.
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

# Dense first layer + sparse layers, SwiGLU clamps on the last two MoE layers only, sigmoid routing with
# ``e_score_correction_bias``, non-neutral router scaling, and a differing sliding head count, so a global
# ``num_attention_heads`` read raises and any path skipping ``per_layer_config`` fails.
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

# The family ships no text-only CausalLM. The projector maps ``hidden_size * 4`` to the text hidden_size itself; the
# 56/14 grid is 4x4, the smallest the two stride-2 downsamplers accept.
TINY_STEP3P7_VISION_CONFIG = {
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "image_size": 56,
    "patch_size": 14,
    "mlp_ratio": 2.0,
    "max_position_embeddings": 16,
}

# MLA attention, a shared expert with group routing, YARN rope with llama-4 scaling. 8 Q / 8 KV heads so cp8 and tp8
# shard evenly; 16 experts split evenly at every EP size up to 8.
TINY_MISTRAL4_CONFIG = {
    "vocab_size": 512,
    "hidden_size": 128,
    "intermediate_size": 256,
    "moe_intermediate_size": 64,
    "n_routed_experts": 16,
    "n_shared_experts": 1,
    "num_experts_per_tok": 4,
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "q_lora_rank": 64,
    "kv_lora_rank": 32,
    "qk_nope_head_dim": 16,
    "qk_rope_head_dim": 16,
    "qk_head_dim": 32,
    "v_head_dim": 32,
    "n_group": 1,
    "topk_group": 1,
    "first_k_dense_replace": 0,
    "hidden_act": "silu",
    "rope_parameters": {
        "rope_type": "yarn",
        "rope_theta": 10000.0,
        "factor": 2.0,
        "original_max_position_embeddings": 256,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "llama_4_scaling_beta": 0.1,
    },
    "rope_interleave": True,
    "max_position_embeddings": 512,
    "norm_topk_prob": True,
    "routed_scaling_factor": 1.0,
    "tie_word_embeddings": False,
}

# One layer each, since the towers carry the checkpoint namespace under test, not compute.
TINY_SIGLIP_VISION_CONFIG = {
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "image_size": 28,
    "patch_size": 14,
}
TINY_SIGLIP2_VISION_CONFIG = {
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_patches": 16,
    "patch_size": 14,
}
TINY_PIXTRAL_VISION_CONFIG = {
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "head_dim": 16,
    "image_size": 28,
    "patch_size": 14,
}
TINY_GEMMA4_VISION_CONFIG = {
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "head_dim": 16,
}
TINY_QWEN35_VISION_CONFIG = {
    "depth": 1,
    "hidden_size": 16,
    "intermediate_size": 16,
    "num_heads": 2,
    "out_hidden_size": TINY_QWEN35_MOE_CONFIG["hidden_size"],
}
# M-RoPE's three sections split head_dim / 2; a one-block vision tower with no deepstack taps.
TINY_QWEN3_VL_TEXT_CONFIG = {
    **TINY_QWEN3_CONFIG,
    "rope_parameters": {"rope_type": "default", "mrope_section": [8, 4, 4]},
}
TINY_QWEN3_VL_VISION_CONFIG = {
    **TINY_QWEN35_VISION_CONFIG,
    "out_hidden_size": TINY_QWEN3_CONFIG["hidden_size"],
    "deepstack_visual_indexes": [],
}
# Two vision blocks at the release's patch geometry.
TINY_QWEN4_EXP_VISION_CONFIG = {
    "depth": 2,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_heads": 2,
    "num_position_embeddings": 64,
    "out_hidden_size": TINY_QWEN4_EXP_CONFIG["hidden_size"],
}
# The upstream scale planner (``plan_out_scales``) rejects most tiny shapes; this one builds a valid 4-layer stack.
TINY_INKLING_VISION_CONFIG = {
    "temporal_patch_size": 2,
    "patch_size": 4,
    "n_layers": 4,
    "hidden_size": 32,
    "text_hidden_size": TINY_INKLING_CONFIG["hidden_size"],
}

# The heads stay at flash-kernel widths over the release's two KV heads; ``moe_latent_size`` differs from the hidden
# size, so a layer sizing its wire off the other one fails at its first dispatch.
TINY_NEMOTRON_H_CONFIG = {
    "model_type": "nemotron_h",
    "vocab_size": 128,
    "hidden_size": 64,
    "intermediate_size": 32,
    "max_position_embeddings": 1024,
    "layers_block_type": ["mamba", "moe", "attention", "moe", "mamba", "moe"],
    "mamba_num_heads": 8,
    "mamba_head_dim": 8,
    "ssm_state_size": 8,
    "n_groups": 2,
    "conv_kernel": 4,
    "chunk_size": 8,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "n_routed_experts": 8,
    "num_experts_per_tok": 3,
    "moe_latent_size": 16,
    "moe_intermediate_size": 32,
    "moe_shared_expert_intermediate_size": 48,
    "routed_scaling_factor": 5.0,
    "norm_topk_prob": True,
    "mlp_hidden_act": "relu2",
    "num_nextn_predict_layers": 1,
    "time_step_limit": [0.0, float("inf")],
    "tie_word_embeddings": False,
}
# The release's text and vision shape, for tests planning its checkpoint on a meta-device shell. One letter per block:
# Mamba, MoE (Expert), Attention.
NEMOTRON_H_RELEASE_BLOCKS = {"M": "mamba", "E": "moe", "A": "attention"}
NEMOTRON_H_RELEASE_LAYOUT = "MEMEMEMAEMEMEMEMAEMEMEMEMAEMEMEMEMEMAEMEMEMEMEMAEMEMEMEMEMAEMEMEMEMEMAEMEMEMEMAEMEMEMEME"
NEMOTRON_H_RELEASE_TEXT_CONFIG = {
    "model_type": "nemotron_h",
    "vocab_size": 131072,
    "hidden_size": 4096,
    "intermediate_size": 2688,
    "layers_block_type": [NEMOTRON_H_RELEASE_BLOCKS[block] for block in NEMOTRON_H_RELEASE_LAYOUT],
    "mamba_num_heads": 128,
    "mamba_head_dim": 64,
    "ssm_state_size": 128,
    "n_groups": 8,
    "conv_kernel": 4,
    "chunk_size": 128,
    "num_attention_heads": 32,
    "num_key_value_heads": 2,
    "head_dim": 128,
    "n_routed_experts": 512,
    "num_experts_per_tok": 22,
    "moe_latent_size": 1024,
    "moe_intermediate_size": 2688,
    "moe_shared_expert_intermediate_size": 5376,
    "routed_scaling_factor": 5.0,
    "norm_topk_prob": True,
    "mlp_hidden_act": "relu2",
    "num_nextn_predict_layers": 1,
    "tie_word_embeddings": False,
}
NEMOTRON_H_RELEASE_VISION_CONFIG = {
    "model_type": "radio",
    "hidden_size": 1280,
    "num_hidden_layers": 32,
    "num_attention_heads": 16,
    "max_img_size": 2048,
    "video_temporal_patch_size": 2,
    "summary_idxs": [0, 1],
}
# Its heads at the width FA4 takes, which a GPU row reaches unless it pins another backend.
TINY_RADIO_VISION_CONFIG = {
    "model_type": "radio",
    "hidden_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "max_img_size": 64,
    "video_temporal_patch_size": 2,
    "summary_idxs": [0, 1],
}
# A composite in the pinned transformers that no family spec claims. The heads stay at flash-kernel widths, its 16:1 GQA
# keeps the release's ratio, and T = index_block_size * index_topk_blocks stays under a row's length so selection runs (a
# document within T attends densely).
TINY_MINIMAX_M3_CONFIG = {
    "vocab_size": 128,
    "hidden_size": 64,
    "intermediate_size": 16,
    "shared_intermediate_size": 16,
    "dense_intermediate_size": 64,
    "max_position_embeddings": 1024,
    "num_hidden_layers": 5,
    "num_attention_heads": 64,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "rotary_dim": 64,
    "rope_parameters": {"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.5},
    "layer_types": ["minimax_m3_sparse"] * 5,
    "mlp_layer_types": ["dense"] + ["sparse"] * 4,
    "num_local_experts": 8,
    "num_experts_per_tok": 2,
    "index_n_heads": 4,
    "index_head_dim": 16,
    "index_block_size": 8,
    "index_topk_blocks": 3,
    "index_local_blocks": 1,
    "swiglu_alpha": 1.702,
    "swiglu_limit": 0.5,
}
TINY_MINIMAX_M3_VISION_CONFIG = {
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "patch_size": 14,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
}

TINY_LLAMA4_TEXT_CONFIG = {
    "vocab_size": 128,
    "hidden_size": 64,
    "intermediate_size": 128,
    "intermediate_size_mlp": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "max_position_embeddings": 128,
    "moe_layers": [],
    "no_rope_layers": [1, 1],
}
TINY_LLAMA4_VISION_CONFIG = {
    "hidden_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "intermediate_size": 128,
    "vision_output_dim": 64,
    "image_size": 28,
    "patch_size": 14,
    "projector_input_dim": 64,
    "projector_output_dim": 64,
}

QWEN3_5_VLM_4B = "Qwen/Qwen3.5-4B"  # Natively multimodal
PARAPHRASE_MINILM = "sentence-transformers/paraphrase-MiniLM-L3-v2"

# Checkpoints the CPU tests read only a tokenizer, processor or chat template from

GPT_OSS_20B_OPENAI = "openai/gpt-oss-20b"
GEMMA4_31B_QAT = "google/gemma-4-31B-it-qat-w4a16-ct"
GEMMA3_4B_IT = "google/gemma-3-4b-it"  # gated: anonymous downloads are refused
QWEN2_5_VL_3B = "Qwen/Qwen2.5-VL-3B-Instruct"
QWEN3_VL_2B = "Qwen/Qwen3-VL-2B-Instruct"

# Pinned because hub main can drift the files a test reads (a chat template, the set of processor configs).
PINNED_REVISIONS = {
    QWEN2_5_VL_3B: "66285546d2b821cf421d4f5eb2576359d3770cd3",
    QWEN3_VL_2B: "89644892e4d85e24eaac8bacfd4f463576704203",
    STEP3P7_FLASH: "5f6244077ac62e04eec3f320501ff8c2b293373a",
}

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
    "nemotron-3.5-lightning": {
        "hf_name": NEMOTRON_3_5_LIGHTNING,
        "full_params": 31.58e9,
        "num_experts": 128,
        "top_k": 6,
    },
    "glm4-flash": {
        "hf_name": GLM4_FLASH,
        "full_params": 30e9,
        "num_experts": 64,
        "top_k": 4,
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
    "qwen3-next-80b-a3b": {
        "hf_name": QWEN3_NEXT_80B_A3B,
        "full_params": 79.67e9,
        "num_experts": 512,
        "top_k": 10,
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
