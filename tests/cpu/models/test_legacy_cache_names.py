#!/usr/bin/env python
"""transformers 5.24 renamed ``past_key_values`` and ``cache_params`` to ``cache`` and keeps the old names on its
forwards. The toolkit's replacement forwards take them the same way, so a legacy cache is neither dropped nor let past
a refusal."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from halo.distributed.pipeline_parallel.families import qwen4_exp_stream_forward
from halo.models.attention.tree_sparse import tree_sparse_forward
from halo.models.upstream.upstream_guard import takes_legacy_cache_names

CACHE = object()


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
