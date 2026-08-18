#!/usr/bin/env python
"""Pure CP on a MoE still gets the ``ep_size == 1`` grouped-GEMM expert wrappers.

``_load_cp_model`` hands ``load_model_for_cp`` an ep_config whenever ``needs_ep_wrappers`` holds
AND the model is MoE. Both conjuncts are load-bearing: without the ep_config a MoE under pure CP
pays the Liger swiglu/geglu force-off (which ``needs_ep_wrappers`` triggers on its own) while its
experts keep the stock per-expert loop — the cost with none of the speedup; with it keyed on
``needs_ep_wrappers`` alone, a DENSE model under CP would be handed an ep_config it has no experts
for. The expert wrappers this decides are observable as :class:`EPMoELayerBase` modules, which the
CP save path (``stale_dense_mlp_keys``) and the 8-GPU Mistral4 CP node both assert on.

Run: ``python tests/cpu/parallelism/test_cp_grouped_gemm_wrappers.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from accelerate import PartialState

from src.distributed.loading.model_loading import _load_cp_model

# The loader logs through accelerate's MultiProcessAdapter, which refuses to emit before the
# global state exists.
PartialState()

_SENTINEL_EP_CONFIG = object()

_MOE_CONFIG = SimpleNamespace(num_local_experts=8)
_DENSE_CONFIG = SimpleNamespace(hidden_size=128)


def _capture_ep_config(*, needs_ep_wrappers: bool, model_config) -> object | None:
    """Return the ``ep_config`` ``_load_cp_model`` would hand ``load_model_for_cp``."""
    pc = SimpleNamespace(
        cp_size=8,
        max_concurrent_loading=0,
        needs_ep_wrappers=needs_ep_wrappers,
        create_cp_config=lambda: SimpleNamespace(cp_size=8),
        create_ep_config=lambda: _SENTINEL_EP_CONFIG,
    )
    seen = {}

    def fake_load_model_for_cp(model_name_or_path, cp_config, **kwargs):
        seen["ep_config"] = kwargs["ep_config"]
        return SimpleNamespace()

    with patch("src.distributed.loading.model_loading.load_model_for_cp", fake_load_model_for_cp):
        _load_cp_model("dummy/path", pc, None, {"config": model_config})
    return seen["ep_config"]


def test_moe_under_pure_cp_gets_the_grouped_gemm_ep_config():
    assert _capture_ep_config(needs_ep_wrappers=True, model_config=_MOE_CONFIG) is _SENTINEL_EP_CONFIG


def test_dense_under_pure_cp_gets_no_ep_config():
    """``needs_ep_wrappers`` defaults True on a dense model too (grouped GEMM) — experts decide."""
    assert _capture_ep_config(needs_ep_wrappers=True, model_config=_DENSE_CONFIG) is None


def test_moe_under_pure_cp_with_grouped_gemm_off_gets_no_ep_config():
    assert _capture_ep_config(needs_ep_wrappers=False, model_config=_MOE_CONFIG) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
