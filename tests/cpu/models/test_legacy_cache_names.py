#!/usr/bin/env python
"""transformers 5.24 renamed ``past_key_values`` and ``cache_params`` to ``cache`` and keeps the old names on its
forwards. The toolkit's replacement forwards take them the same way, so a legacy cache is neither dropped nor let past
a refusal."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from halo.distributed.pipeline_parallel.families import qwen4_exp_stream_forward
from halo.models.attention.tree_sparse import tree_sparse_forward
from halo.models.families.inkling2 import apply_inkling2_patches
from halo.models.families.qwen4_exp.forwards import patch_qwen4_exp_document_forwards
from halo.models.upstream.upstream_guard import takes_legacy_cache_names
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model

CACHE = object()
# A continuation past a cached prefix: the cache changes what every continuation position attends to.
_PREFIX, _LENGTH = 8, 12


@pytest.mark.parametrize("old_name", ["past_key_values", "cache_params"])
def test_an_old_name_arrives_as_cache(old_name):
    @takes_legacy_cache_names
    def forward(self, hidden_states, cache=None, **kwargs):
        return cache, kwargs

    assert forward(object(), None, **{old_name: CACHE}) == (CACHE, {})


def test_both_names_together_are_refused():
    @takes_legacy_cache_names
    def forward(self, hidden_states, cache=None, **kwargs):
        return cache

    with pytest.raises(ValueError, match="both `cache` and `past_key_values`"):
        forward(object(), None, cache=CACHE, past_key_values=CACHE)


def test_the_tree_sparse_attention_hands_a_legacy_cache_to_the_upstream_forward():
    received = {}

    def upstream(self, hidden_states, attention_mask, cache=None, prev_topk_indices=None, **kwargs):
        received.update(cache=cache, **kwargs)

    tree_sparse_forward(upstream)(object(), torch.zeros(1, 2, 4), None, past_key_values=CACHE)
    assert received == {"cache": CACHE}


def test_a_pipeline_stage_refuses_a_legacy_cache():
    with pytest.raises(RuntimeError, match="called with a cache"):
        qwen4_exp_stream_forward(SimpleNamespace(config=None), past_key_values=CACHE)


def _tiny(family: str) -> torch.nn.Module:
    torch.manual_seed(0)
    model = tiny_family_model(TINY_MOE_FAMILIES[family]).eval()
    model.config._attn_implementation = "eager"
    return model


def _ids(model) -> torch.Tensor:
    return torch.randint(3, model.config.vocab_size, (1, _LENGTH), generator=torch.Generator().manual_seed(1))


def test_the_qwen4_exp_backbone_continues_a_legacy_cache_as_it_does_cache():
    """The document forward replaces the backbone's own, so the legacy name reaches it unrenamed."""
    patch_qwen4_exp_document_forwards()
    backbone = _tiny("qwen4_exp_text").model
    ids = _ids(backbone)
    with torch.no_grad():
        cache = backbone(input_ids=ids[:, :_PREFIX], use_cache=True).cache
        continuation = ids[:, _PREFIX:]
        new = backbone(input_ids=continuation, cache=copy.deepcopy(cache), use_cache=True).last_hidden_state
        legacy = backbone(input_ids=continuation, past_key_values=copy.deepcopy(cache), use_cache=True)
        uncached = backbone(input_ids=continuation, use_cache=True).last_hidden_state
    assert not torch.equal(new, uncached), "premise: the prefix's cache moves the continuation"
    torch.testing.assert_close(legacy.last_hidden_state, new, rtol=0, atol=0)


def test_the_inkling2_decoder_layer_continues_a_legacy_cache_as_it_does_cache():
    """The layer call a cached continuation makes, replayed through each name. The wrapper reads the cache itself: a
    cached call hands the short convolutions no document positions, whichever name carries it."""
    apply_inkling2_patches()
    model = _tiny("inkling2_text")
    layer = model.model.layers[0]
    ids = _ids(model)
    calls = []

    def capture(module, args, kwargs):
        if not calls:
            calls.append((args, {**kwargs, "cache": copy.deepcopy(kwargs["cache"])}))

    with torch.no_grad():
        cache = model(input_ids=ids[:, :_PREFIX], use_cache=True).cache
        handle = layer.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            model(input_ids=ids[:, _PREFIX:], cache=cache, use_cache=True)
        finally:
            handle.remove()
        args, kwargs = calls[0]
        saved = kwargs.pop("cache")
        positions = []
        convolution = layer.self_attn.k_proj.sconv
        handle = convolution.register_forward_pre_hook(lambda module, _: positions.append(module.document_positions))
        try:
            new = layer(*args, **kwargs, cache=copy.deepcopy(saved))
            legacy = layer(*args, **kwargs, past_key_values=copy.deepcopy(saved))
        finally:
            handle.remove()
    assert saved.get_seq_length() == _PREFIX, "premise: the replayed call continues the cached prefix"
    assert len(positions) == 2 and all(seen is None for seen in positions), "a cached call read document positions"
    torch.testing.assert_close(legacy, new, rtol=0, atol=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
