"""Costing a sliding, chunked, sparse, compressed or linear layer as full attention over-reports MFU silently.
So does costing a packed row as one document, since the varlen kernels attend within each document."""

import pytest
import torch
from torch import nn
from transformers import CONFIG_MAPPING, NemotronHForCausalLM

from halo.models.attention.layout import (
    INDEX_PASSES,
    LAYER_TYPE_KINDS,
    SCORE_PASSES,
    AttentionLayout,
    LayerAttention,
    attention_layout,
    attention_layout_from_config,
)
from tests.common.models import NEMOTRON_H_RELEASE_TEXT_CONFIG, QWEN3_8_FLASH_NEXT, TINY_NEMOTRON_H_CONFIG
from tests.common.tokenizers import load_cached_config

# gpt-oss-120b: 18 full + 18 sliding(128) layers, 64 heads of 64.
DEPTH = 36
WINDOW = 128
HEADS, HEAD_DIM, HIDDEN = 64, 64, 2880
WIDTH = HEADS * (HEAD_DIM + HEAD_DIM)
MAX_LENGTH = 64000


class _Config:
    def __init__(self, layer_types=None, **fields):
        self.num_hidden_layers = DEPTH
        self.hidden_size = HIDDEN
        self.num_attention_heads = HEADS
        self.head_dim = HEAD_DIM
        self.sliding_window = WINDOW
        self.layer_types = layer_types
        for name, value in fields.items():
            setattr(self, name, value)

    def get_text_config(self):
        return self


def _alternating(depth=DEPTH):
    return ["sliding_attention" if i % 2 == 0 else "full_attention" for i in range(depth)]


class _Layer(nn.Module):
    def __init__(self, layer_type=None, layer_idx=None):
        super().__init__()
        if layer_type is not None:
            self.layer_type = layer_type
        if layer_idx is not None:
            self.self_attn = nn.Module()
            self.self_attn.layer_idx = layer_idx


class _Model(nn.Module):
    def __init__(self, layers, config):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(layers)
        self.config = config


def _per_token(full_layers: int, window_layers: int, seq_len: int, width=WIDTH, window=WINDOW) -> float:
    return SCORE_PASSES * width * (full_layers * seq_len + window_layers * min(seq_len, window))


def _full_attention_formula(seq_len: int) -> float:
    """PaLM's ``12·L·S·H`` full-attention accounting."""
    return 12.0 * DEPTH * seq_len * HIDDEN


def test_sliding_layers_are_costed_at_their_window_not_the_sequence():
    layout = attention_layout_from_config(_Config(_alternating()))
    got = layout.flops_per_token(MAX_LENGTH)
    assert got == pytest.approx(_per_token(18, 18, MAX_LENGTH), rel=1e-12)
    assert got < 0.75 * _full_attention_formula(MAX_LENGTH), (
        "the sliding half of the stack is still costed as full attention"
    )


def test_the_score_width_is_heads_times_head_dims_not_hidden_size():
    """gpt-oss attends at 64×64 = 4096 per projection while hidden_size is 2880; ``12·S·H`` under-sizes it."""
    layout = attention_layout_from_config(_Config(["full_attention"] * DEPTH))
    assert layout.flops_per_token(4096) == pytest.approx(SCORE_PASSES * DEPTH * 4096 * WIDTH, rel=1e-12)
    assert layout.layers[0].width == HEADS * 2 * HEAD_DIM


def test_documents_are_costed_individually_not_as_the_packed_row():
    layout = attention_layout_from_config(_Config(_alternating()))
    lengths = torch.tensor([1500, 1500, 1000, 8000, 20000, 32000], dtype=torch.long)
    assert int(lengths.sum()) == MAX_LENGTH
    got = float(layout.flops_for_documents(lengths))
    expected = sum(length * _per_token(18, 18, length) for length in lengths.tolist())
    assert got == pytest.approx(expected, rel=1e-12)
    as_one_row = MAX_LENGTH * _per_token(18, 18, MAX_LENGTH)
    assert got < 0.5 * as_one_row, "packed documents are being costed as one row-length document"


def test_per_token_and_per_document_agree_on_a_single_document():
    layout = attention_layout_from_config(_Config(_alternating()))
    for length in (1, WINDOW, 4096, MAX_LENGTH):
        per_doc = float(layout.flops_for_documents(torch.tensor([length])))
        assert per_doc == pytest.approx(length * layout.flops_per_token(length), rel=1e-12)
    assert layout.flops_per_token(0) == 0.0


def test_the_layers_own_type_attribute_wins_over_list_position():
    """A pipeline stage holding layers 5..11 re-bases list positions; the layers' own types do not."""
    config = _Config(_alternating())
    stage = _Model([_Layer(layer_type=config.layer_types[i]) for i in range(5, 11)], config)
    layout = attention_layout(stage)
    assert layout.source == "layers"
    kinds = [layer.kind for layer in layout.layers]
    assert kinds == ["full", "window", "full", "window", "full", "window"]
    assert layout.flops_per_token(4096) == pytest.approx(_per_token(3, 3, 4096), rel=1e-12)


def test_the_attention_submodules_layer_idx_indexes_layer_types():
    config = _Config(_alternating())
    stage = _Model([_Layer(layer_idx=i) for i in (7, 8, 9)], config)
    assert [layer.kind for layer in attention_layout(stage).layers] == ["full", "window", "full"]


def test_list_position_is_the_last_resort():
    config = _Config(_alternating())
    assert [layer.kind for layer in attention_layout(_Model([_Layer(), _Layer()], config)).layers] == [
        "window",
        "full",
    ]


def test_a_tree_without_layers_falls_back_to_the_configs_full_depth():
    class _Bare:
        config = _Config(_alternating())

    layout = attention_layout(_Bare())
    assert len(layout.layers) == DEPTH and layout.source == "config"


@pytest.mark.parametrize(
    ("fields", "kinds"),
    [
        ({"sliding_window": 4096}, {"window"}),
        ({"sliding_window": None}, {"full"}),
    ],
)
def test_configs_without_layer_types_follow_the_transformers_derivation(fields, kinds):
    """Mistral (and Qwen3-MoE under ``use_sliding_window``) slide every layer; otherwise every layer is full."""
    layout = attention_layout_from_config(_Config(None, **fields))
    assert {layer.kind for layer in layout.layers} == kinds and len(layout.layers) == DEPTH


def test_hybrid_groups_without_layer_types_are_refused_not_costed_as_full():
    with pytest.raises(ValueError, match="layer_group_size"):
        attention_layout_from_config(_Config(None, layer_group_size=4))


def test_a_layer_group_of_one_is_softmax_attention_throughout():
    """Each group's last layer is softmax attention, so groups of one hold no linear layer to tell apart."""
    layout = attention_layout_from_config(_Config(None, layer_group_size=1, sliding_window=None))
    assert {layer.kind for layer in layout.layers} == {"full"} and len(layout.layers) == DEPTH


def test_an_unknown_layer_type_is_refused():
    with pytest.raises(ValueError, match="mystery_attention"):
        attention_layout_from_config(_Config(["mystery_attention"] * DEPTH))


def test_a_sliding_layer_without_a_window_is_refused():
    with pytest.raises(ValueError, match="sliding_window"):
        attention_layout_from_config(_Config(_alternating(), sliding_window=None))


def test_linear_and_conv_layers_score_no_key_matrix():
    layout = attention_layout_from_config(_Config(["linear_attention", "conv", "full_attention"] * 12))
    assert layout.flops_per_token(4096) == pytest.approx(_per_token(12, 0, 4096), rel=1e-12)
    assert {layer.kind for layer in layout.layers} == {"none", "full"}


def test_chunked_layers_are_costed_at_their_chunk():
    layout = attention_layout_from_config(
        _Config(["chunked_attention", "full_attention"] * 18, attention_chunk_size=8192)
    )
    assert layout.flops_per_token(32768) == pytest.approx(_per_token(18, 18, 32768, window=8192), rel=1e-12)


def test_sparse_layers_cost_top_k_keys_plus_a_pooled_indexer():
    config = _Config(
        ["indexed_attention"] * DEPTH,
        index_topk=2048,
        index_n_heads=32,
        index_head_dim=128,
        index_kpool=16,
    )
    layout = attention_layout_from_config(config)
    seq_len = 65536
    expected = DEPTH * (SCORE_PASSES * WIDTH * 2048 + INDEX_PASSES * 32 * 128 * seq_len / 16)
    assert layout.flops_per_token(seq_len) == pytest.approx(expected, rel=1e-12)


def test_block_indexed_layers_cost_the_budget_plus_the_tail_block():
    """Qwen3.8-Flash-Next's QSA keeps ``indexer_budget`` tokens of whole blocks plus the tail's up to
    ``indexer_compress_ratio - 1`` tokens, and scores one pooled key per block."""
    config = _Config(
        ["indexed_attention"] * DEPTH,
        indexer_budget=2048,
        indexer_compress_ratio=4,
        indexer_n_heads=4,
        indexer_head_dim=128,
    )
    layout = attention_layout_from_config(config)
    seq_len = 65536
    expected = DEPTH * (SCORE_PASSES * WIDTH * 2051 + INDEX_PASSES * 4 * 128 * seq_len / 4)
    assert layout.flops_per_token(seq_len) == pytest.approx(expected, rel=1e-12)


def test_the_flash_next_release_costs_twelve_qsa_layers():
    config = load_cached_config(QWEN3_8_FLASH_NEXT)
    layers = attention_layout_from_config(config).layers
    sparse = [layer for layer in layers if layer.kind == "sparse"]
    assert len(layers) == 48 and len(sparse) == 12 and {layer.kind for layer in layers} == {"sparse", "none"}
    assert {(layer.keys, layer.index_width, layer.index_pool) for layer in sparse} == {(2051, 512.0, 4)}


def test_compressed_layers_cost_the_compressed_stream_plus_the_local_band():
    """CSA selects ``index_topk`` of its ``L/4`` compressed keys through an indexer; HCA attends all
    ``L/128`` of its keys with neither indexer nor top-k bound. Both add the local band."""
    config = _Config(
        ["compressed_sparse_attention", "heavily_compressed_attention"] * 18,
        compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 128},
        index_topk=512,
        index_n_heads=64,
        index_head_dim=128,
    )
    layout = attention_layout_from_config(config)
    seq_len = 1 << 20
    csa = SCORE_PASSES * WIDTH * (min(seq_len / 4, 512) + WINDOW) + INDEX_PASSES * 64 * 128 * seq_len / 4
    hca = SCORE_PASSES * WIDTH * (seq_len / 128 + WINDOW)
    assert seq_len / 128 > 512, "the premise: HCA's key count must exceed the top-k it does not clamp to"
    assert layout.flops_per_token(seq_len) == pytest.approx(18 * (csa + hca), rel=1e-12)


def test_mla_widths_use_the_declared_qk_and_v_dims():
    config = _Config(
        ["full_attention"] * DEPTH, head_dim=None, qk_nope_head_dim=192, qk_rope_head_dim=64, v_head_dim=256
    )
    layout = attention_layout_from_config(config)
    assert layout.layers[0].width == HEADS * (192 + 64 + 256)


def test_a_rope_slice_declared_alone_does_not_widen_the_head():
    """DeepSeek-V4 rotates the first qk_rope_head_dim of head_dim; the q/k width stays head_dim."""
    config = _Config(["full_attention"] * DEPTH, head_dim=512, qk_rope_head_dim=64)
    assert attention_layout_from_config(config).layers[0].width == HEADS * (512 + 512)


def test_per_layer_head_count_lists_are_read():
    """Laguna declares ``num_attention_heads_per_layer`` as a plain list outside the per-layer view."""
    config = _Config(["full_attention"] * 4, num_attention_heads_per_layer=[48, 72, 72, 48])
    config.num_hidden_layers = 4
    widths = [layer.width for layer in attention_layout_from_config(config).layers]
    assert widths == [48 * 2 * HEAD_DIM, 72 * 2 * HEAD_DIM, 72 * 2 * HEAD_DIM, 48 * 2 * HEAD_DIM]


def test_a_config_without_head_geometry_keeps_the_palm_width():
    """``12·S·H`` is ``6·S·(2H)``: a config declaring only hidden_size costs at width ``2·hidden``."""
    config = _Config(["full_attention"] * DEPTH, num_attention_heads=None, head_dim=None)
    layout = attention_layout_from_config(config)
    assert layout.flops_per_token(4096) == pytest.approx(12.0 * DEPTH * 4096 * HIDDEN, rel=1e-12)


def test_per_layer_head_geometry_is_read_off_the_per_layer_view():
    """Gemma 4 registers head_dim per layer."""

    class _View:
        def __init__(self, dims):
            self._dims = dims

        def __getitem__(self, index):
            layer = _Config(None)
            layer.head_dim = self._dims[index]
            return layer

    config = _Config(["full_attention"] * 4)
    config.num_hidden_layers = 4
    config.per_layer_config = _View([256, 512, 256, 512])
    layout = attention_layout_from_config(config)
    assert [layer.width for layer in layout.layers] == [HEADS * 512, HEADS * 1024, HEADS * 512, HEADS * 1024]


def test_grouping_sums_widths_per_rule_without_changing_the_total():
    layout = AttentionLayout(
        (
            LayerAttention("full", 100.0),
            LayerAttention("full", 200.0),
            LayerAttention("window", 50.0, keys=8),
            LayerAttention("none", 0.0),
        )
    )
    grouped = layout.grouped
    assert {(g.kind, g.width) for g in grouped} == {("full", 300.0), ("window", 50.0), ("none", 0.0)}
    assert layout.grouped is grouped, "every batch's costing reads the one merge, not a rebuilt one"
    assert layout.flops_per_token(16) == pytest.approx(SCORE_PASSES * (300.0 * 16 + 50.0 * 8), rel=1e-12)
    assert "2×full + 1×window(8)" in layout.describe()


# Config key -> kinds its default layer_types must map to, for every family the toolkit trains.
_ROSTER = {
    "gpt_oss": {"full", "window"},
    "cohere2_moe": {"full", "window"},
    "gemma4_text": {"full", "window"},
    "llama4_text": {"full", "chunked"},
    "qwen3": {"full"},
    "qwen3_moe": {"full"},
    "qwen3_5_moe_text": {"full", "none"},
    "qwen3_next": {"full", "none"},
    "glm4_moe": {"full"},
    "glm4_moe_lite": {"full"},
    "glm5_next_text": {"sparse", "none"},
    "deepseek_v4": {"compressed"},
    "laguna": {"full"},
    "inkling_text": {"full", "window"},
    "lfm2": {"full"},
    "mistral4": {"full"},
    "mistral": {"window"},
    "nemotron_h": {"full", "none"},
    "step3p7": {"full"},
    "zaya": {"full"},
}


@pytest.mark.parametrize("key", sorted(_ROSTER))
def test_every_roster_family_classifies_from_its_default_config(key):
    config = CONFIG_MAPPING[key]()
    layout = attention_layout_from_config(config)
    assert {layer.kind for layer in layout.layers} == _ROSTER[key]
    assert len(layout.layers) == config.get_text_config().num_hidden_layers
    assert all(layer.width > 0 for layer in layout.layers if layer.kind != "none")
    assert layout.flops_per_token(4096) > 0


def test_lfm2_moe_declares_its_conv_layers_through_layer_types():
    """The hub checkpoint carries the list; a default config leaves it None and derives all-full."""
    config = CONFIG_MAPPING["lfm2_moe"](layer_types=["conv", "full_attention"] * 20, num_hidden_layers=40)
    assert {layer.kind for layer in attention_layout_from_config(config).layers} == {"none", "full"}


def _nemotron_h_release_config():
    return CONFIG_MAPPING["nemotron_h"](
        **{k: v for k, v in NEMOTRON_H_RELEASE_TEXT_CONFIG.items() if k != "model_type"}
    )


def test_only_nemotrons_attention_layers_score_keys():
    """40 Mamba and 40 MoE blocks read no key matrix; the 8 attention blocks cost their heads at every key."""
    config = _nemotron_h_release_config()
    assert config.layer_types.count("linear_attention") == 40 and config.layer_types.count("moe") == 40
    layout = attention_layout_from_config(config)
    seq_len = 8192
    assert layout.flops_per_token(seq_len) == pytest.approx(SCORE_PASSES * 32 * (128 + 128) * seq_len * 8, rel=1e-12)
    assert layout.describe() == "80×no-attention + 8×full @ width 8192"


def test_nemotrons_blocks_read_their_kinds_off_the_module_tree():
    """A rank's layout reads the layers it holds, whose block type is what transformers keeps on them."""
    config = CONFIG_MAPPING["nemotron_h"](**{k: v for k, v in TINY_NEMOTRON_H_CONFIG.items() if k != "model_type"})
    with torch.device("meta"):
        layout = attention_layout(NemotronHForCausalLM(config))
    assert layout.source == "layers"
    assert layout.describe() == "5×no-attention + 1×full @ width 512"


@pytest.mark.parametrize("layer_type", ["moe", "mlp"])
def test_a_feed_forward_block_has_a_cost_rule_of_its_own(layer_type, monkeypatch):
    """Without its entry the block raises rather than being costed as full attention."""
    config = _nemotron_h_release_config()
    config.layer_types = [layer_type, "full_attention"]
    assert {layer.kind for layer in attention_layout_from_config(config).layers} == {"none", "full"}
    monkeypatch.delitem(LAYER_TYPE_KINDS, layer_type)
    with pytest.raises(ValueError, match=f"{layer_type!r}.*no attention cost rule"):
        attention_layout_from_config(config)


def test_the_vocabulary_covers_every_layer_type_the_roster_declares():
    declared = set()
    for key in _ROSTER:
        declared.update(getattr(CONFIG_MAPPING[key]().get_text_config(), "layer_types", None) or ())
    assert declared <= set(LAYER_TYPE_KINDS), f"unmapped layer types: {declared - set(LAYER_TYPE_KINDS)}"


def test_deepseek_v4_scores_at_head_dim_with_its_rope_slice_inside():
    config = CONFIG_MAPPING["deepseek_v4"]()
    widths = {layer.width for layer in attention_layout_from_config(config).layers}
    assert widths == {64 * (512 + 512)}


def test_gemma4_costs_its_two_head_dims_per_layer():
    """Gemma 4's sliding layers run head_dim 256 and its full layers 512, on 8 heads."""
    config = CONFIG_MAPPING["gemma4_text"]()
    widths = {layer.width for layer in attention_layout_from_config(config).layers}
    assert widths == {8 * 2 * 256, 8 * 2 * 512}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
