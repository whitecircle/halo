"""Attention geometry read off a model config: head dimension and KV head count.

Kept separate because the attention patches, the RoPE buffer fixes and the pipeline split's cost
model all size something from it and none of them may import the others.
"""

from src.models.loading.config_levels import get_config_field, text_config


def resolve_head_dim(cfg) -> int:
    """The attention head dimension, falling back to ``hidden_size // num_attention_heads``.

    A declared ``head_dim`` may be decoupled from that ratio (Gemma 4's 512 against a smaller hidden
    size), so it must win at every site that sizes a RoPE table, a flash warm-up or a cost model.
    Read through :func:`get_config_field`, so a composite config resolves off its text sub-config.
    """
    # ``max``: every consumer sizes something (RoPE table, warm-up buffer, cost ceiling), so on a
    # per-layer-heterogeneous family (Gemma 4's 256/512 sliding/global split) the largest head wins.
    declared = get_config_field(cfg, "head_dim", per_layer_reduce=max)
    if declared:
        return int(declared)
    decoder = text_config(cfg)
    return decoder.hidden_size // decoder.num_attention_heads


def resolve_num_key_value_heads(cfg) -> int:
    """The KV head count, falling back to ``num_attention_heads`` (MHA when GQA is undeclared).

    Same composite-config rule as :func:`resolve_head_dim`: the field lives on the text sub-config,
    and the fallback must be spelled once, not at every consumer.
    """
    # ``max``: consumers size cost models and warm-up buffers, so on a per-layer-heterogeneous
    # family (hub Gemma 4's [2, 8] KV heads, Step-3.7's [64, 96] Q heads) the largest count wins.
    declared = get_config_field(cfg, "num_key_value_heads", per_layer_reduce=max)
    if declared:
        return int(declared)
    return int(get_config_field(cfg, "num_attention_heads", per_layer_reduce=max))
