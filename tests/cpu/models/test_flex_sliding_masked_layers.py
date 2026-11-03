"""The flex-sliding attention serves every masked causal layer of a sliding-window model, full layers included,
and keys its warm-up on what each attention module calls with."""

import os
from unittest.mock import patch

import pytest
import torch
from torch.nn.attention import flex_attention as flex_attention_module
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from transformers import (
    AutoModelForCausalLM,
    Gemma2Config,
    Gemma3TextConfig,
    Glm4MoeLiteConfig,
    Glm4MoeLiteForCausalLM,
    MistralConfig,
    MistralForCausalLM,
    Qwen3Config,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import Glm4MoeLiteAttention
from transformers.models.step3p7.modeling_step3p7 import Step3p7Attention

from halo.models.attention import sdpa_flex_sliding
from halo.models.attention.layout import AttentionCall, attention_call, has_sliding_window_layers
from halo.models.attention.sdpa_flex_sliding import (
    SDPA_FLEX_SLIDING,
    flex_attention_calls,
    register_flex_sliding_attention,
    resolve_flex_sliding_attn_implementation,
)
from tests.common.attention_stubs import causal_attention
from tests.common.flex_sliding import flex_sliding_on_cpu, no_device_check
from tests.common.models import TINY_GLM4_MOE_LITE_CONFIG
from tests.common.numerics import fro_rel_err
from tests.common.tiny_models import (
    TINY_DENSE_FAMILIES,
    TINY_GEMMA4_FLEX_FAMILY,
    TINY_GEMMA4_TEXT_FLEX_FAMILY,
    TINY_MOE_FAMILIES,
    TINY_MOE_VLM_FAMILIES,
    TINY_SLIDING_DENSE_FAMILIES,
    tiny_family_model,
)

# Gemma 4 with its sliding heads as wide as FlexAttention runs, so its sliding layers take it as a released one's do.
ROSTER = {
    **TINY_MOE_FAMILIES,
    **TINY_MOE_VLM_FAMILIES,
    **TINY_DENSE_FAMILIES,
    **TINY_SLIDING_DENSE_FAMILIES,
    "gemma4": TINY_GEMMA4_FLEX_FAMILY,
    "gemma4_text": TINY_GEMMA4_TEXT_FLEX_FAMILY,
}
# The tiny families with sliding layers whose every call FlexAttention may take. DeepSeek-V4 resolves here too, but its
# family spec forces eager; GptOss carries sinks on every call; Inkling a dense relative bias; Bailing is remote code.
FLEX_FAMILIES = {
    "cohere2",
    "cohere2_moe",
    "cohere2_vision",
    "deepseek_v4",
    "gemma4",
    "gemma4_text",
    "laguna",
    "mistral",
    "olmo3",
    "step3p7",
}
# The families whose forward reaches the attention interface (DeepSeek-V4 runs eager).
FORWARD_FAMILIES = sorted(FLEX_FAMILIES - {"deepseek_v4"})
TILE = sdpa_flex_sliding._TILE


def _resolve(config, attn_implementation: str = "sdpa", *, cuda: bool = True) -> str:
    with patch.object(torch.cuda, "is_available", lambda: cuda):
        return resolve_flex_sliding_attn_implementation(config, attn_implementation)


@pytest.mark.parametrize("family", sorted(ROSTER))
def test_the_resolver_over_the_tiny_roster(family):
    torch.manual_seed(0)
    config = tiny_family_model(ROSTER[family]).config
    expected = SDPA_FLEX_SLIDING if family in FLEX_FAMILIES else "sdpa"
    assert _resolve(config) == expected


def test_the_resolver_keeps_sdpa_where_flex_takes_no_call():
    sliding = MistralConfig(sliding_window=64)
    assert _resolve(sliding) == SDPA_FLEX_SLIDING  # a window on every layer, no layer_types
    assert _resolve(Gemma3TextConfig()) == SDPA_FLEX_SLIDING
    assert Gemma2Config().attn_logit_softcapping  # every Gemma 2 call carries the softcap
    assert _resolve(Gemma2Config()) == "sdpa"
    assert _resolve(Qwen3Config()) == "sdpa"  # no sliding layer, narrow heads
    assert _resolve(sliding, "flash_attention_2") == "flash_attention_2"
    assert _resolve(sliding, cuda=False) == "sdpa"
    with patch.dict(os.environ, {"HALO_FLEX_SLIDING": "0"}):
        assert _resolve(sliding) == "sdpa"


@pytest.mark.parametrize("family", ["inkling_text", "inkling_mm_model"])
def test_a_class_not_declared_backend_compatible_keeps_sdpa(family):
    """Every Inkling call carries a dense relative bias flex is not given, so it would only warm unused graphs."""
    torch.manual_seed(0)
    model = tiny_family_model(ROSTER[family])
    assert has_sliding_window_layers(model.config)  # a window would select it
    assert not type(model).is_backend_compatible()
    assert _resolve(model.config) == "sdpa"


class _RemoteConfig(MistralConfig):
    pass


class _RemoteModel(MistralForCausalLM):
    config_class = _RemoteConfig


_RemoteModel.__module__ = "transformers_modules.org.repo.modeling_remote"


def test_a_remote_code_config_keeps_sdpa():
    """A remote-code model's attention may not dispatch through the registry the implementation is registered in."""
    config = _RemoteConfig(sliding_window=64)
    assert _resolve(config) == "sdpa"
    mapping = AutoModelForCausalLM._model_mapping._extra_content
    with patch.dict(mapping, {_RemoteConfig: _RemoteModel}):
        assert _RemoteModel.is_backend_compatible()  # only its remote origin keeps it off
        assert _resolve(config) == "sdpa"


def test_step3p7_calls_carry_each_layers_own_head_count():
    """The config's global head count raises, so a warm-up reading it fails the load."""
    torch.manual_seed(0)
    model = tiny_family_model(TINY_MOE_FAMILIES["step3p7"])
    attention = next(m for m in model.modules() if isinstance(m, Step3p7Attention))
    with pytest.raises(RuntimeError):  # AmbiguousGlobalPerLayerAttributeError
        _ = attention.config.num_attention_heads
    assert flex_attention_calls(model) == {
        AttentionCall(4, 2, 64, 64, None, 0.125, None),
        AttentionCall(2, 2, 64, 64, 64, 0.125, None),
    }


def _narrow_value_mla() -> Glm4MoeLiteForCausalLM:
    """A tiny GLM-4.7-Flash whose value head is narrower than its query/key head, as DeepSeek-V3's is."""
    torch.manual_seed(0)
    model = Glm4MoeLiteForCausalLM(Glm4MoeLiteConfig(**{**TINY_GLM4_MOE_LITE_CONFIG, "v_head_dim": 16}))
    attention = next(m for m in model.modules() if isinstance(m, Glm4MoeLiteAttention))
    assert attention.v_head_dim != attention.qk_head_dim
    return model


def test_the_call_check_and_the_warmup_read_one_call_on_an_mla_module():
    """An MLA module reports its query/key width instead of raising, and its value width, which flex takes unpadded."""
    model = _narrow_value_mla()
    attention = next(m for m in model.modules() if isinstance(m, Glm4MoeLiteAttention))
    assert not hasattr(attention, "head_dim")
    call = sdpa_flex_sliding._flex_call(attention)
    assert call == attention_call(attention)
    assert (call.head_dim, call.v_head_dim) == (attention.qk_head_dim, attention.v_head_dim)
    assert flex_attention_calls(model) == {call}


def test_an_mla_forward_makes_the_call_its_modules_declare():
    """A call check keyed on the query/key width would refuse every such call."""
    model = _narrow_value_mla()
    seen, refused = _runtime_calls(model, [150, 30, 90])
    assert _signatures(model) == seen
    assert not refused


def _packed(lengths: list[int], vocab: int) -> dict:
    ids = torch.randint(3, vocab, (1, sum(lengths)), generator=torch.Generator().manual_seed(0))
    position_ids = torch.cat([torch.arange(n) for n in lengths])[None]
    return {"input_ids": ids, "position_ids": position_ids, "labels": ids}


def _padded(lengths: list[int], vocab: int) -> dict:
    seq = max(lengths)
    ids = torch.randint(3, vocab, (len(lengths), seq), generator=torch.Generator().manual_seed(1))
    attention_mask = (torch.arange(seq)[None] < torch.tensor(lengths)[:, None]).long()
    return {"input_ids": ids, "attention_mask": attention_mask, "labels": ids.masked_fill(attention_mask == 0, -100)}


def _full_mask(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """The shape FlexAttention's call check admits."""
    return torch.ones(query.shape[0], 1, query.shape[-2], key.shape[-2], dtype=torch.bool)


def _runtime_calls(model, lengths: list[int]) -> tuple[set[tuple[int, int, int, int, float | None]], list]:
    """``(heads, kv_heads, head_dim, v_head_dim, scaling)`` of every causal call no wider than FLASH_MAX_HEAD_DIM a
    packed forward makes, and the calls among them FlexAttention would refuse."""
    seen, refused = set(), []
    sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]

    def spy(module, query, key, value, attention_mask, **kwargs):
        if getattr(module, "is_causal", False) and query.shape[-1] <= sdpa_flex_sliding.FLASH_MAX_HEAD_DIM:
            made = (query.shape[1], key.shape[1], query.shape[-1], value.shape[-1], kwargs.get("scaling"))
            seen.add(made)
            if not sdpa_flex_sliding._flex_serves(module, query, key, value, _full_mask(query, key), None, made[-1]):
                refused.append((type(module).__name__, made))
        return sdpa(module, query, key, value, attention_mask, **kwargs)

    with torch.no_grad(), patch.dict(ALL_ATTENTION_FUNCTIONS._global_mapping, {"sdpa": spy}):
        model(**_packed(lengths, model.config.get_text_config().vocab_size), use_cache=False)
    return seen, refused


def _signatures(model) -> set[tuple[int, int, int, int, float | None]]:
    return {
        (call.heads, call.kv_heads, call.head_dim, call.v_head_dim, call.scaling)
        for call in flex_attention_calls(model)
    }


@pytest.mark.parametrize("family", FORWARD_FAMILIES)
def test_the_warmup_calls_are_the_ones_a_forward_makes(family):
    """A call the warm-up misses compiles mid-run, and one it invents is never used."""
    torch.manual_seed(0)
    model = tiny_family_model(ROSTER[family])
    seen, refused = _runtime_calls(model, [150, 30, 90])
    assert _signatures(model) == seen
    assert not refused


def _documents(lengths: list[int], pad: int = 0) -> torch.Tensor:
    """A packed row's document-causal ``[Q, K]`` mask; a padded tail attends nothing and is never attended."""
    document = torch.cat([torch.full((n,), i) for i, n in enumerate(lengths)] + [torch.full((pad,), -1)])
    positions = torch.arange(document.numel())
    real = document >= 0
    return (document[:, None] == document[None]) & (positions[None] <= positions[:, None]) & real[:, None] & real


def _dense_blocks(counts: torch.Tensor, indices: torch.Tensor, kv_blocks: int) -> torch.Tensor:
    dense = torch.zeros(counts.shape[-1], kv_blocks, dtype=torch.bool)
    for row, (count, order) in enumerate(zip(counts[0, 0], indices[0, 0], strict=True)):
        dense[row, order[:count]] = True
    return dense


@pytest.mark.parametrize("lengths, pad", [([300, 7, 150], 0), ([200, 250], 60), ([129], 0)])
def test_a_packed_full_mask_reduces_to_exactly_its_tiles(lengths, pad):
    mask = _documents(lengths, pad)
    seq = mask.shape[0]
    (block_mask,) = sdpa_flex_sliding.row_block_masks(mask[None, None].clone())
    padded = torch.nn.functional.pad(mask, (0, -seq % TILE, 0, -seq % TILE))
    tiles = padded.view(padded.shape[0] // TILE, TILE, -1, TILE)
    kv_blocks = tiles.shape[2]
    partial = _dense_blocks(block_mask.kv_num_blocks, block_mask.kv_indices, kv_blocks)
    full = _dense_blocks(block_mask.full_kv_num_blocks, block_mask.full_kv_indices, kv_blocks)
    assert torch.equal(full, tiles.all(dim=(1, 3)))
    assert torch.equal(partial | full, tiles.any(dim=(1, 3)))
    assert not (partial & full).any()


def _flex_serves(query, key, mask, window, *, value=None, scaling=0.125, module=None) -> bool:
    value = key if value is None else value
    module = module or causal_attention(scaling=0.125, sliding_window=window)
    return sdpa_flex_sliding._flex_serves(module, query, key, value, mask, window, scaling)


def test_flex_serves_masked_rows_past_a_tile_on_narrow_heads():
    q = torch.zeros(2, 4, 300, 64)
    k = torch.zeros(2, 2, 300, 64)
    mask = torch.ones(2, 1, 300, 300, dtype=torch.bool)
    serves = _flex_serves
    assert serves(q, k, mask, None)
    assert serves(q, k, mask, 128)
    assert not serves(q, k, mask, 300)  # every query within the window: no tile skipped
    assert not serves(q, k, None, None)  # unmasked calls keep SDPA's causal kernel
    assert not serves(q[..., :TILE, :], k[..., :TILE, :], mask[..., :TILE, :TILE], None)
    assert not serves(torch.zeros(2, 4, 300, 512), k, mask, None)  # past the widest head flex runs here
    assert not serves(q, k, mask[:1], None)  # one mask for every row
    # The mask must cover exactly the call's queries and keys, the only shape the warm-up compiles.
    assert not serves(q, k[:, :, :280], mask, None)
    assert not serves(q[:, :, :299], k, mask, None)


def test_flex_takes_only_the_call_the_warmup_compiles(caplog):
    """Tensor parallelism hands each rank a share of the heads, which would compile a graph mid-run."""
    q = torch.zeros(2, 4, 300, 64)
    k = torch.zeros(2, 2, 300, 64)
    mask = torch.ones(2, 1, 300, 300, dtype=torch.bool)
    module = causal_attention(scaling=0.125)
    assert flex_attention_calls(module) == {AttentionCall(4, 2, 64, 64, None, 0.125, None)}
    assert _flex_serves(q, k, mask, None, module=module)
    with caplog.at_level("WARNING", logger=sdpa_flex_sliding.logger.name):
        for _ in range(2):
            assert not _flex_serves(q[:, :2], k[:, :1], mask, None, module=module)  # the tp=2 share
    assert len([r for r in caplog.records if "tensor parallelism" in r.getMessage()]) == 1
    assert not _flex_serves(q, k, mask, None, value=k[..., :32], module=module)  # an MLA-style value width
    assert not _flex_serves(q, k, mask, None, scaling=0.25, module=module)
    bare = torch.nn.Module()
    bare.is_causal = True  # declares no call, so the warm-up compiles none for it
    assert not _flex_serves(q, k, mask, None, module=bare)


@pytest.mark.parametrize(
    ("head_dim", "v_head_dim", "runs"),
    [(8, 8, False), (16, 8, False), (8, 16, False), (16, 16, True), (256, 32, True), (512, 512, False)],
)
def test_flex_runs_only_head_widths_its_kernel_takes(head_dim, v_head_dim, runs):
    """Inductor's kernel raises NYI at compile on a query/key or value head narrower than FLEX_MIN_HEAD_DIM."""
    call = AttentionCall(4, 2, head_dim, v_head_dim, 128, 0.125, None)
    assert sdpa_flex_sliding._flex_runs(call) is runs


def test_full_layer_flex_matches_per_document_attention():
    """Over a packed full layer's block masks, each real query must attend exactly its own document's prefix."""
    lengths, pad, heads, kv_heads, dim = [300, 7, 150], 40, 4, 2, 32
    mask = _documents(lengths, pad)
    seq = mask.shape[0]
    generator = torch.Generator().manual_seed(0)
    q = torch.randn(1, heads, seq, dim, dtype=torch.float64, generator=generator)
    k = torch.randn(1, kv_heads, seq, dim, dtype=torch.float64, generator=generator)
    v = torch.randn(1, kv_heads, seq, dim, dtype=torch.float64, generator=generator)
    with torch.no_grad(), patch.object(sdpa_flex_sliding, "_compiled_flex", flex_attention):
        out = sdpa_flex_sliding._block_sparse_attention(
            q, k, v, sdpa_flex_sliding.row_block_masks(mask[None, None]), 0.2
        )
    expected, start = [], 0
    for n in lengths:
        rows = slice(start, start + n)
        expected.append(
            torch.nn.functional.scaled_dot_product_attention(
                q[:, :, rows],
                k[:, :, rows].repeat_interleave(heads // kv_heads, 1),
                v[:, :, rows].repeat_interleave(heads // kv_heads, 1),
                is_causal=True,
                scale=0.2,
            )
        )
        start += n
    real = sum(lengths)
    assert fro_rel_err(out[:, :real], torch.cat(expected, dim=2).transpose(1, 2)) < 1e-12
    assert torch.equal(out[:, real:], torch.zeros_like(out[:, real:]))  # a padded query attends nothing


def _as_the_kernel_reads(block_mask: BlockMask) -> BlockMask:
    """As the compiled kernel reads: a full tile whole, a partial tile through the function, any other tile not at all.
    Uncompiled FlexAttention evaluates the function alone, which would hide a wrong tile list."""
    kv_blocks = block_mask.kv_indices.shape[-1]
    partial = _dense_blocks(block_mask.kv_num_blocks, block_mask.kv_indices, kv_blocks)
    full = _dense_blocks(block_mask.full_kv_num_blocks, block_mask.full_kv_indices, kv_blocks)
    mask_mod = block_mask.mask_mod

    def read(b, h, q_idx, kv_idx):
        q_tile, kv_tile = q_idx // TILE, kv_idx // TILE
        return full[q_tile, kv_tile] | (partial[q_tile, kv_tile] & mask_mod(b, h, q_idx, kv_idx))

    return BlockMask.from_kv_blocks(
        block_mask.kv_num_blocks,
        block_mask.kv_indices,
        block_mask.full_kv_num_blocks,
        block_mask.full_kv_indices,
        BLOCK_SIZE=TILE,
        mask_mod=read,
        seq_lengths=block_mask.seq_lengths,
    )


def _uncompiled_kernel(calls: list):
    """Appends to ``calls`` as proof the path ran, since a fallback to SDPA would reproduce the SDPA model exactly."""

    def kernel(query, key, value, *, block_mask, **kwargs):
        calls.append(1)
        query, key, value = (t.as_subclass(torch.Tensor) for t in (query, key, value))
        return flex_attention(query, key, value, block_mask=_as_the_kernel_reads(block_mask), **kwargs)

    return kernel


def _loss_and_grads(model, batch: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    model.zero_grad(set_to_none=True)
    out = model(**batch, use_cache=False)
    out.loss.backward()
    return out.loss.detach(), {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def _reads_rotary_positions(model, layer: int) -> bool:
    """Compares positions 0..7 against every token at position 0, where the rotation is the identity."""
    attention = model.model.layers[layer].self_attn
    hidden = torch.randn(1, 8, model.config.hidden_size, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        outputs = [
            attention(hidden, model.model.rotary_emb(hidden, positions[None]), None)[0]
            for positions in (torch.arange(8), torch.zeros(8, dtype=torch.long))
        ]
    return not torch.equal(*outputs)


def test_the_dense_sliding_families_keep_the_shapes_they_cover():
    """Dropping one of these shapes upstream would leave the parity sweep green over a shape it does not run."""
    torch.manual_seed(0)
    mistral, cohere2, olmo3 = (tiny_family_model(ROSTER[family]) for family in ("mistral", "cohere2", "olmo3"))
    assert "layer_types" not in vars(mistral.config)
    assert not any(hasattr(layer.self_attn, "sliding_window") for layer in mistral.model.layers)
    window = mistral.config.sliding_window
    unwindowed = tiny_family_model(ROSTER["mistral"], overrides={"sliding_window": None})
    unwindowed.load_state_dict(mistral.state_dict())
    ids = _packed([2 * window], mistral.config.vocab_size)["input_ids"]
    with torch.no_grad():
        windowed, full = (model(input_ids=ids, use_cache=False).logits[:, window:] for model in (mistral, unwindowed))
    assert not torch.allclose(windowed, full)

    assert cohere2.config.layer_types == ["sliding_attention", "full_attention"]
    assert [_reads_rotary_positions(cohere2, layer) for layer in range(2)] == [True, False]

    assert olmo3.config.layer_types == ["sliding_attention", "full_attention"]
    for layer in olmo3.model.layers:
        attention = layer.self_attn
        assert attention.q_norm.weight.shape == (attention.q_proj.out_features,)
        assert attention.k_norm.weight.shape == (attention.k_proj.out_features,)


@pytest.mark.parametrize("batch, lengths", [(_packed, [150, 30, 90]), (_padded, [260, 140])], ids=["packed", "padded"])
@pytest.mark.parametrize("family", FORWARD_FAMILIES)
def test_the_flex_model_matches_the_sdpa_model(family, batch, lengths):
    """Every layer of every row takes FlexAttention and reproduces the SDPA model to fp32 round-off.
    Its uncompiled math runs a CPU backward; only its input check refuses one."""
    register_flex_sliding_attention()
    torch.manual_seed(0)
    reference = tiny_family_model(ROSTER[family])
    candidate = tiny_family_model(ROSTER[family])
    candidate.load_state_dict(reference.state_dict())
    text_config = candidate.config.get_text_config()
    text_config._attn_implementation = SDPA_FLEX_SLIDING
    assert max(lengths) > text_config.sliding_window  # the window masks keys
    inputs = batch(lengths, text_config.vocab_size)
    calls = []
    with (
        patch.object(flex_attention_module, "_validate_device", no_device_check),
        patch.object(sdpa_flex_sliding, "_compiled_flex", _uncompiled_kernel(calls)),
        patch.dict(ALL_ATTENTION_FUNCTIONS._global_mapping, {SDPA_FLEX_SLIDING: flex_sliding_on_cpu}),
    ):
        loss, grads = _loss_and_grads(candidate, inputs)
    ref_loss, ref_grads = _loss_and_grads(reference, inputs)
    assert len(calls) == text_config.num_hidden_layers * inputs["input_ids"].shape[0]
    torch.testing.assert_close(loss, ref_loss, rtol=1e-5, atol=1e-6)
    assert grads.keys() == ref_grads.keys()
    for name, grad in ref_grads.items():
        assert fro_rel_err(grads[name], grad) < 1e-4, name


@pytest.mark.parametrize("head_dim", [8, sdpa_flex_sliding.FLEX_MIN_HEAD_DIM])
def test_heads_narrower_than_flex_takes_keep_sdpa(head_dim):
    """Below FLEX_MIN_HEAD_DIM there is no FlexAttention call to warm and every layer runs SDPA."""
    register_flex_sliding_attention()
    torch.manual_seed(0)
    overrides = {"head_dim": head_dim}
    reference = tiny_family_model(ROSTER["mistral"], overrides=overrides)
    candidate = tiny_family_model(ROSTER["mistral"], overrides=overrides)
    candidate.load_state_dict(reference.state_dict())
    text_config = candidate.config.get_text_config()
    text_config._attn_implementation = SDPA_FLEX_SLIDING
    inputs = _packed([150, 30, 90], text_config.vocab_size)
    calls = []
    with (
        patch.object(flex_attention_module, "_validate_device", no_device_check),
        patch.object(sdpa_flex_sliding, "_compiled_flex", _uncompiled_kernel(calls)),
        patch.dict(ALL_ATTENTION_FUNCTIONS._global_mapping, {SDPA_FLEX_SLIDING: flex_sliding_on_cpu}),
    ):
        loss, _ = _loss_and_grads(candidate, inputs)
    ref_loss, _ = _loss_and_grads(reference, inputs)
    runs = head_dim >= sdpa_flex_sliding.FLEX_MIN_HEAD_DIM
    assert bool(flex_attention_calls(candidate)) is runs
    assert len(calls) == (text_config.num_hidden_layers if runs else 0)
    torch.testing.assert_close(loss, ref_loss, rtol=1e-5, atol=1e-6)


def _batch(*rows: torch.Tensor) -> torch.Tensor:
    return torch.stack(rows)[:, None]


@pytest.mark.parametrize("rows", [[[300, 7, 150]], [[200, 257], [457]], [[1] * 5]])
def test_packed_documents_are_read_off_the_mask(rows):
    seq = sum(rows[0])
    mask = _batch(*(_documents(lengths) for lengths in rows))
    cu_seqlens, max_len = sdpa_flex_sliding.causal_documents(mask)
    lengths = [n for row in rows for n in row]
    assert cu_seqlens.dtype == torch.int32
    assert cu_seqlens.tolist() == [0, *torch.tensor(lengths).cumsum(0).tolist()]
    assert cu_seqlens[-1] == len(rows) * seq
    assert max_len == max(lengths)
    assert sdpa_flex_sliding.causal_documents(mask)[0] is cu_seqlens  # cached on the mask


def _window(lengths: list[int], window: int) -> torch.Tensor:
    positions = torch.arange(sum(lengths))
    return _documents(lengths) & (positions[:, None] - positions[None] < window)


def _bidirectional_span(lengths: list[int], span: slice) -> torch.Tensor:
    mask = _documents(lengths)
    mask[span, span] = True
    return mask


def _hole(lengths: list[int], query: int, key: int) -> torch.Tensor:
    mask = _documents(lengths)
    mask[query, key] = False
    return mask


@pytest.mark.parametrize(
    "mask",
    [
        _documents([200, 100], pad=40),  # a padded query attends nothing
        _window([300, 100], 64),
        _bidirectional_span([300, 100], slice(10, 20)),  # an image's tokens attend each other
        _hole([300, 100], 250, 3),
        _hole([300, 100], 250, 250),  # a query that does not attend itself
    ],
    ids=["padded", "window", "bidirectional", "hole", "no-diagonal"],
)
def test_masks_that_are_not_document_causal_have_no_documents(mask):
    assert sdpa_flex_sliding.causal_documents(mask[None, None]) is None


def test_one_row_off_document_causal_takes_the_batch_off():
    seq = 400
    good, bad = _documents([300, 100]), _documents([200, 100], pad=100)
    assert sdpa_flex_sliding.causal_documents(_batch(good, good)) is not None
    assert bad.shape == (seq, seq)
    assert sdpa_flex_sliding.causal_documents(_batch(good, bad)) is None
    assert sdpa_flex_sliding.causal_documents(_batch(bad, good)) is None


def test_the_varlen_path_takes_only_wide_head_self_attention():
    q = torch.zeros(2, 4, 300, 512)
    k = torch.zeros(2, 2, 300, 512)
    mask = torch.ones(2, 1, 300, 300, dtype=torch.bool)
    serves = sdpa_flex_sliding._varlen_serves
    assert serves(q, k, mask)
    assert not serves(q, k, None)
    assert not serves(q[..., :256], k[..., :256], mask)  # FlexAttention's width
    assert not serves(q[:, :, :299], k, mask[:, :, :299])  # queries other than the keys
    assert not serves(q, k, mask[:1])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
