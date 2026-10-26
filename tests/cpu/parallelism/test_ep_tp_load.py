#!/usr/bin/env python
"""EP+TP loads through the EP loader, then shards attention.

``load_ep_tp_model`` composes ``load_ep_model``, so it takes the same lazy-loader layout gate: a layout
nesting one module per expert (``...experts.local_experts.N.linear_fc{1,2}``, as pre-5.14 Zaya
checkpoints did) falls back to ``from_pretrained`` + EP patch, since the lazy fuser cannot map it. On
that eager path the EP patch runs before the model moves to the device, so each rank's device receives
only its own experts; Cohere2 MoE, which no lazy loader reads, always takes it.

Run: ``python tests/cpu/parallelism/test_ep_tp_load.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import datetime
import json
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoConfig

import src.model_loading.parallel.ep as ep_loading
import src.model_loading.parallel.tp as tp_loading
import src.model_loading.parallel.whole_model as whole_model
from src.distributed.expert_parallel.config import EPConfig
from src.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from src.distributed.mesh import build_dp_tp_mesh
from src.model_loading.parallel.model_source import ModelSource
from tests.common.gloo import run_gloo_ranks
from tests.common.tiny_models import TINY_MOE_FAMILIES, build_tiny_family_checkpoint

_SUPPORTED_KEYS = ["model.layers.0.mlp.experts.gate_up_proj", "model.layers.0.self_attn.q_proj.weight"]
_UNSUPPORTED_KEYS = [
    "model.layers.0.mlp.experts.local_experts.0.linear_fc1.weight",  # nested per-expert layout
    "model.layers.0.self_attn.q_proj.weight",
]
EP_SIZE = 2


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
        max_concurrent_loading=1,
        fp32_non_ep_params=False,
        ep_config=lambda: "ep-config-sentinel",
    )


def _load_steps(monkeypatch, model_dir: str) -> list[str]:
    """The steps ``load_ep_tp_model`` takes with each heavy one stubbed: which EP loader, then TP."""
    steps: list[str] = []
    sentinel = torch.nn.Linear(2, 2)
    monkeypatch.setattr(ep_loading, "load_ep_model_lazy", lambda *a, **k: steps.append("lazy") or sentinel)
    monkeypatch.setattr(
        ep_loading, "_load_ep_model_huggingface", lambda *a, **k: steps.append("from_pretrained") or sentinel
    )
    monkeypatch.setattr(tp_loading, "_apply_attention_only_tp", lambda model, *a: steps.append("attention_tp"))

    model = tp_loading.load_ep_tp_model(
        ModelSource(model_dir, None, model_dir),
        _stub_pc(),
        None,
        # The dispatcher always resolves the config before it picks a loader; the stubs never read it.
        {"config": SimpleNamespace()},
        preserve_checkpoint_precision=False,
    )
    assert model is sentinel
    return steps


def test_ep_tp_uses_lazy_loader_for_supported_layout(monkeypatch, tmp_path):
    assert _load_steps(monkeypatch, _write_index(tmp_path, _SUPPORTED_KEYS)) == ["lazy", "attention_tp"]


def test_ep_tp_falls_back_on_unsupported_per_expert_layout(monkeypatch, tmp_path):
    """Same gate as load_ep_model: a per-expert (local_experts.N.*) layout must NOT go lazy."""
    steps = _load_steps(monkeypatch, _write_index(tmp_path, _UNSUPPORTED_KEYS))
    assert steps == ["from_pretrained", "attention_tp"]


def _moved_models_rank(rank, checkpoint, family):
    """Each eager load's model at its move to the device: whether it is EP-patched, and its element count."""
    moved: list[tuple[bool, int]] = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(torch.cuda, "is_available", lambda: False)
        mp.setattr(
            whole_model,
            "move_model_to_local_device",
            lambda model: moved.append((bool(named_ep_layers(model)), sum(p.numel() for p in model.parameters())))
            or model,
        )
        mp.setattr(
            tp_loading,
            "build_dp_tp_mesh",
            lambda tp_size, dp_size: build_dp_tp_mesh(tp_size, dp_size, device_type="cpu"),
        )
        for loader, tp_size in ((ep_loading.load_ep_model, 1), (tp_loading.load_ep_tp_model, EP_SIZE)):
            ep_config = EPConfig(
                ep_size=EP_SIZE,
                stage_world_size=EP_SIZE,
                nvlink_domain_size=EP_SIZE,
                experts_fsdp_managed=False,
                use_grouped_gemm=False,
            )
            pc = SimpleNamespace(
                ep_size=EP_SIZE,
                tp_size=tp_size,
                data_parallel_size=1,
                ep_lazy_loading=False,
                max_concurrent_loading=1,
                fp32_non_ep_params=False,
                ep_config=lambda ep_config=ep_config: ep_config,
            )
            kwargs = {
                "config": AutoConfig.from_pretrained(checkpoint),
                "dtype": torch.bfloat16,
                "trust_remote_code": False,
            }
            loader(
                ModelSource(checkpoint, None, checkpoint),
                pc,
                TINY_MOE_FAMILIES[family].load_class,
                kwargs,
                preserve_checkpoint_precision=False,
            )
    (ep_patched, ep_numel), (ep_tp_patched, ep_tp_numel) = moved
    assert ep_patched and ep_tp_patched, f"rank {rank}: a model reached the device before the EP patch"
    assert ep_tp_numel == ep_numel, (
        f"rank {rank}: the eager EP+TP load moved {ep_tp_numel} elements where EP moves {ep_numel}: "
        f"the device received experts other ranks own"
    )


@pytest.mark.parametrize("family", ["qwen3_moe", "cohere2_moe"])
def test_eager_ep_tp_moves_only_this_ranks_experts(tmp_path, family):
    checkpoint = str(tmp_path / family)
    build_tiny_family_checkpoint(TINY_MOE_FAMILIES[family], checkpoint)
    run_gloo_ranks(_moved_models_rank, EP_SIZE, checkpoint, family, pg_timeout=datetime.timedelta(seconds=120))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
