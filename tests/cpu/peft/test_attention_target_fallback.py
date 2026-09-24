#!/usr/bin/env python
"""The GPU LoRA suites' attention targets fall back to the default pair only for a single-file
checkpoint, never for a hub failure.

``attention_target_modules`` reads the targets off the checkpoint's safetensors index. Only an
absent index means a single-file checkpoint; a missing or gated repo or a bad revision must surface
as itself, not as a ``q_proj``/``v_proj`` guess that PEFT later rejects as "Target modules not found"
far from the cause.

Run: pytest tests/cpu/peft/test_attention_target_fallback.py
"""

import json

import httpx
import pytest
from huggingface_hub.errors import EntryNotFoundError, GatedRepoError, RepositoryNotFoundError

from tests.common import peft_helpers


def _hub_response(status: int) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("GET", "https://huggingface.co/org/repo"))


def _raising(exc: Exception):
    def download(*_args, **_kwargs):
        raise exc

    return download


def test_absent_index_falls_back_to_default_targets(monkeypatch):
    monkeypatch.setattr(peft_helpers, "hf_hub_download", _raising(EntryNotFoundError("no index")))
    assert peft_helpers.attention_target_modules("org/single-file") == ["q_proj", "v_proj"]


@pytest.mark.parametrize(
    "exc",
    [
        RepositoryNotFoundError("no such repo", response=_hub_response(404)),
        GatedRepoError("gated", response=_hub_response(403)),
    ],
)
def test_hub_failure_raises_instead_of_default_targets(monkeypatch, exc):
    monkeypatch.setattr(peft_helpers, "hf_hub_download", _raising(exc))
    with pytest.raises(type(exc)):
        peft_helpers.attention_target_modules("org/typo")


def test_index_targets_are_read_off_the_weight_map(monkeypatch, tmp_path):
    index = tmp_path / "model.safetensors.index.json"
    weight_map = {
        "model.layers.0.self_attn.q_a_proj.weight": "a",
        "model.layers.0.self_attn.kv_b_proj.weight": "a",
        "model.layers.0.self_attn.q_norm.weight": "a",
        "model.layers.0.mlp.gate_proj.weight": "a",
    }
    index.write_text(json.dumps({"weight_map": weight_map}))
    monkeypatch.setattr(peft_helpers, "hf_hub_download", lambda *_args, **_kwargs: str(index))
    assert peft_helpers.attention_target_modules("org/mla") == ["kv_b_proj", "q_a_proj"]


def test_a_local_checkpoint_is_read_in_place(tmp_path):
    """A local directory is not a repo id: the hub client refuses it, so it must never reach it."""
    single_file = tmp_path / "single"
    single_file.mkdir()
    assert peft_helpers.attention_target_modules(str(single_file)) == ["q_proj", "v_proj"]

    sharded = tmp_path / "sharded"
    sharded.mkdir()
    weight_map = {"model.layers.0.self_attn.q_a_proj.weight": "a", "model.layers.0.self_attn.o_proj.weight": "a"}
    (sharded / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    assert peft_helpers.attention_target_modules(str(sharded)) == ["o_proj", "q_a_proj"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
