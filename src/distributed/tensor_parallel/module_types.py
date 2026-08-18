"""Frozen set of attention class names the selective-TP path can shard."""

# One entry per family; transformers >= 5 dispatches backends through the attention interface.
TP_SHARDABLE_ATTENTION_CLASSES = frozenset(
    {
        "Cohere2MoeAttention",
        "GptOssAttention",
        "Qwen3MoeAttention",
        "Qwen3Attention",
        "Glm4MoeLiteAttention",
        "Lfm2MoeAttention",
        "Qwen3_5MoeAttention",
        "Qwen3_5Attention",
        # Qwen3-VL text tower.
        "Qwen3VLTextAttention",
        # Mistral4 (mistral3 VLM backbone), MLA: only q_b_proj / kv_b_proj are colwise-shardable.
        "Mistral4Attention",
    }
)
