#!/usr/bin/env python
"""EP+TP loading must apply the same lazy-loader layout gate as pure EP loading.

``load_ep_model`` routes a checkpoint to the lazy safetensors loader only when
``is_lazy_loadable`` accepts its expert layout; a layout nesting one module per
expert (``...experts.local_experts.N.linear_fc{1,2}``, as pre-5.14 Zaya checkpoints did) falls back
to ``from_pretrained`` + EP patch because the lazy fuser cannot map it. ``load_ep_tp_model`` must
gate identically — gating only on the safetensors probe would route a TP-capable family
with nested per-expert keys into the lazy loader and silently drop its experts.

Run: ``python tests/cpu/parallelism/test_ep_tp_lazy_gate.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from accelerate import PartialState

import src.model_loading.parallel.tp as tp_loading
from src.model_loading.parallel.model_source import ModelSource

PartialState()  # the loader's accelerate logger needs accelerate state initialized

_SUPPORTED_KEYS = ["model.layers.0.mlp.experts.gate_up_proj", "model.layers.0.self_attn.q_proj.weight"]
_UNSUPPORTED_KEYS = [
    "model.layers.0.mlp.experts.local_experts.0.linear_fc1.weight",  # nested per-expert layout
    "model.layers.0.self_attn.q_proj.weight",
]


def _write_index(tmp_path, keys: list[str]) -> str:
    index = {"metadata": {}, "weight_map": {k: "model-00001-of-00001.safetensors" for k in keys}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    return str(tmp_path)


def _stub_pc() -> SimpleNamespace:
    return SimpleNamespace(
        ep_lazy_loading=True,
        ep_size=2,
        tp_size=2,
        data_parallel_size=1,
        pp_size=1,
        max_concurrent_loading=1,
        fp32_non_ep_params=False,
        ep_config=lambda: "ep-config-sentinel",
    )


def _run_dispatch(monkeypatch, model_dir: str) -> dict:
    """Run load_ep_tp_model with every heavy step stubbed; record which loader path fired."""
    calls = {"lazy": 0, "sequential": 0}
    sentinel = torch.nn.Linear(2, 2)

    monkeypatch.setattr(
        tp_loading, "load_ep_model_lazy", lambda *a, **k: calls.__setitem__("lazy", calls["lazy"] + 1) or sentinel
    )
    monkeypatch.setattr(
        tp_loading,
        "load_whole_model",
        lambda *a, **k: calls.__setitem__("sequential", calls["sequential"] + 1) or sentinel,
    )
    monkeypatch.setattr(tp_loading, "_apply_attention_only_tp", lambda *a, **k: None)
    monkeypatch.setattr(tp_loading, "apply_ep_layers", lambda model, ep_config: model)
    monkeypatch.setattr(tp_loading, "timed_global_load", lambda tag, method, load_fn: load_fn())

    model = tp_loading.load_ep_tp_model(
        ModelSource(model_dir, None, model_dir),
        _stub_pc(),
        None,
        # The dispatcher always resolves the config before it picks a loader; the stubs above
        # never read it.
        {"config": SimpleNamespace()},
        preserve_checkpoint_precision=False,
    )
    assert model is sentinel
    return calls


def test_ep_tp_uses_lazy_loader_for_supported_layout(monkeypatch, tmp_path):
    calls = _run_dispatch(monkeypatch, _write_index(tmp_path, _SUPPORTED_KEYS))
    assert calls == {"lazy": 1, "sequential": 0}


def test_ep_tp_falls_back_on_unsupported_per_expert_layout(monkeypatch, tmp_path):
    """Same gate as load_ep_model: a per-expert (local_experts.N.*) layout must NOT go lazy."""
    calls = _run_dispatch(monkeypatch, _write_index(tmp_path, _UNSUPPORTED_KEYS))
    assert calls == {"lazy": 0, "sequential": 1}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
