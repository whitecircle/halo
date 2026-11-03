#!/usr/bin/env python
"""``modeling_bailing_moe_v3`` (Ling 3.0) copies HF's ``eager_attention_forward`` without its dispatch, so
``attn_implementation: sdpa`` still materializes the ``[B, H, S, S]`` score plane, ~190 GiB on a packed 80k row."""

import pathlib
import sys
from types import SimpleNamespace

import pytest
import torch
import transformers.dynamic_module_utils

from halo.model_loading.remote_code import (
    _shim_dispatchless_eager_attention,
    apply_remote_code_compat_shims,
)

apply_remote_code_compat_shims()

# The published file's shape: the HF eager helper with no dispatch, so forwards resolve it as a module global per call.
DISPATCHLESS_SOURCE = '''
import torch
import torch.nn as nn


def repeat_kv2(hidden_states, n_rep):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    key_states = repeat_kv2(key, module.num_key_value_groups)
    value_states = repeat_kv2(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class LingLikeAttention:
    """Stands in for the published attention forward's global lookup."""

    def run(self, module, query, key, value, attention_mask, scaling):
        return eager_attention_forward(module, query, key, value, attention_mask, scaling=scaling)
'''

# A file with a real dispatch selects backends itself.
DISPATCHING_SOURCE = """
ALL_ATTENTION_FUNCTIONS = {}


def eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    return query, None


class DispatchingModel:
    pass
"""

# The v4-era list form that crashes 5.12+ save_pretrained (_get_tied_weight_keys calls .keys()).
TIED_LIST_SOURCE = """
class LegacyTiedModel:
    _tied_weights_keys = ["lm_head.weight"]


class DictTiedModel:
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
"""

PROBE_PREFIX = "sdpa_shim_probe_"


@pytest.fixture(autouse=True)
def _drop_probe_modules():
    yield
    for name in [n for n in sys.modules if n.startswith(PROBE_PREFIX)]:
        del sys.modules[name]


def _load(tmp_path, monkeypatch, source: str, class_name: str):
    module_name = f"{PROBE_PREFIX}{abs(hash(source)) % 10**8}"
    module_path = f"{module_name}.py"
    monkeypatch.setattr(transformers.dynamic_module_utils, "HF_MODULES_CACHE", str(tmp_path))
    (pathlib.Path(tmp_path) / module_path).write_text(source, encoding="utf-8")
    sys.modules.pop(module_name, None)
    return transformers.dynamic_module_utils.get_class_in_module(class_name, module_path)


def _attn_module(impl: str, groups: int = 1):
    return SimpleNamespace(
        config=SimpleNamespace(_attn_implementation=impl), num_key_value_groups=groups, training=False
    )


def _qkv(kv_heads: int = 4, groups: int = 1, seq: int = 32, qk_dim: int = 24, v_dim: int = 16, seed: int = 3):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(2, kv_heads * groups, seq, qk_dim, generator=g)
    k = torch.randn(2, kv_heads, seq, qk_dim, generator=g)
    v = torch.randn(2, kv_heads, seq, v_dim, generator=g)
    return q, k, v


def _causal_mask(seq: int) -> torch.Tensor:
    return torch.triu(torch.full((seq, seq), float("-inf")), 1).expand(2, 1, seq, seq)


def test_dispatchless_module_is_wrapped_through_the_funnel(tmp_path, monkeypatch):
    cls = _load(tmp_path, monkeypatch, DISPATCHLESS_SOURCE, "LingLikeAttention")
    module = sys.modules[cls.__module__]
    assert getattr(module.eager_attention_forward, "_halo_sdpa_shim", False), "the eager global was not wrapped"


def test_sdpa_route_matches_eager_on_every_geometry(tmp_path, monkeypatch):
    """fp32 throughout: the CPU SDPA math backend and the eager formula agree to float tolerance, so any drift is a
    real semantics change."""
    cls = _load(tmp_path, monkeypatch, DISPATCHLESS_SOURCE, "LingLikeAttention")
    module = sys.modules[cls.__module__]
    original = module.eager_attention_forward.__wrapped__
    scaling = 24**-0.5

    for name, groups, mask in (
        ("masked", 1, _causal_mask(32)),
        ("causal-unmasked", 1, None),
        ("gqa", 2, _causal_mask(32)),
    ):
        q, k, v = _qkv(groups=groups)
        eager_out, _ = original(_attn_module("eager", groups), q, k, v, _causal_mask(32), scaling=scaling)
        sdpa_out, sdpa_weights = module.eager_attention_forward(
            _attn_module("sdpa", groups), q, k, v, mask, scaling=scaling
        )
        assert sdpa_weights is None
        drift = (eager_out - sdpa_out).abs().max().item()
        assert drift < 1e-5, f"[{name}] SDPA route drifts {drift:.2e} from the file's own eager"


def test_non_sdpa_config_falls_through_to_the_original(tmp_path, monkeypatch):
    cls = _load(tmp_path, monkeypatch, DISPATCHLESS_SOURCE, "LingLikeAttention")
    module = sys.modules[cls.__module__]
    q, k, v = _qkv()
    _, weights = module.eager_attention_forward(_attn_module("eager"), q, k, v, _causal_mask(32), scaling=24**-0.5)
    assert weights is not None, "eager config must keep the file's own path (which returns weights)"


def test_output_attentions_keeps_the_eager_path(tmp_path, monkeypatch):
    """SDPA cannot return attention weights."""
    cls = _load(tmp_path, monkeypatch, DISPATCHLESS_SOURCE, "LingLikeAttention")
    module = sys.modules[cls.__module__]
    q, k, v = _qkv()
    _, weights = module.eager_attention_forward(
        _attn_module("sdpa"), q, k, v, _causal_mask(32), scaling=24**-0.5, output_attentions=True
    )
    assert weights is not None


def test_module_with_a_real_dispatch_is_untouched(tmp_path, monkeypatch):
    cls = _load(tmp_path, monkeypatch, DISPATCHING_SOURCE, "DispatchingModel")
    module = sys.modules[cls.__module__]
    assert not getattr(module.eager_attention_forward, "_halo_sdpa_shim", False)


def test_legacy_tied_weights_list_becomes_a_dict(tmp_path, monkeypatch):
    """Every Bailing/Ling class carries the v4 list form."""
    cls = _load(tmp_path, monkeypatch, TIED_LIST_SOURCE, "LegacyTiedModel")
    assert cls._tied_weights_keys == {}
    module = sys.modules[cls.__module__]
    assert module.DictTiedModel._tied_weights_keys == {"lm_head.weight": "model.embed_tokens.weight"}


def test_wrapping_is_idempotent(tmp_path, monkeypatch):
    cls = _load(tmp_path, monkeypatch, DISPATCHLESS_SOURCE, "LingLikeAttention")
    module = sys.modules[cls.__module__]
    wrapped = module.eager_attention_forward
    _shim_dispatchless_eager_attention(module)
    assert module.eager_attention_forward is wrapped, "a second pass re-wrapped the wrapper"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
