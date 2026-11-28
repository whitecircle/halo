#!/usr/bin/env python
"""A dense text tower under the ``mistral3`` composite is no Mistral 4: Ministral 3 and Devstral Small 2 wrap
``ministral3``, Mistral Small 3.1 wraps ``mistral``. Online GRPO refuses one before the load with a remedy that exists,
a llama-4 query scale counts as reading row positions, and packing isolation is the tower's own."""

from __future__ import annotations

import re
import types
from pathlib import Path

import pytest
import torch
from transformers import Ministral3Config, Ministral3ForCausalLM, Mistral3Config

from halo.models.moe.family_specs import (
    MISTRAL4,
    attention_reads_row_positions,
    find_moe_family_spec_for_tower,
    llama_4_scaling_beta,
)
from halo.models.segment_markers import packing_isolation_is_measured
from halo.trainers.grpo.rollout.weight_push import reject_unsyncable_config
from halo.weight_sync.registry import rollout_backends
from tests.common.models import TINY_MISTRAL4_CONFIG, TINY_PIXTRAL_VISION_CONFIG
from tests.common.repo import REPO_ROOT

# Where the llama-4 query scale starts: Ministral 3's original window, shrunk.
_WINDOW = 16
_DIMS = {
    "vocab_size": 64,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 16,
}
_DOC_LINK = re.compile(r"(agent-docs/[\w./-]+\.md)(?:#([\w-]+))?")


def _ministral3(beta: float = 0.1) -> Ministral3Config:
    config = Ministral3Config(**_DIMS, max_position_embeddings=16 * _WINDOW)
    config.rope_parameters = {
        **config.rope_parameters,
        "original_max_position_embeddings": _WINDOW,
        "llama_4_scaling_beta": beta,
    }
    return config


def _composite(text: dict) -> Mistral3Config:
    return Mistral3Config(text_config=text, vision_config=dict(TINY_PIXTRAL_VISION_CONFIG), spatial_merge_size=2)


def _ministral3_composite() -> Mistral3Config:
    return _composite({**_ministral3().to_dict(), "model_type": "ministral3"})


def _mistral_composite() -> Mistral3Config:
    return _composite({**_DIMS, "model_type": "mistral"})


def _mistral4_composite() -> Mistral3Config:
    return _composite({**TINY_MISTRAL4_CONFIG, "model_type": "mistral4"})


_DENSE_COMPOSITES = {"ministral3": _ministral3_composite, "mistral": _mistral_composite}


def _links_resolve(message: str) -> None:
    links = _DOC_LINK.findall(message)
    assert links, "the refusal names no page"
    for path, anchor in links:
        page = Path(REPO_ROOT, path).read_text(encoding="utf-8")
        if anchor:
            headings = {
                line.lstrip("#").strip().lower().replace(" ", "-")
                for line in page.splitlines()
                if line.startswith("#")
            }
            assert anchor in headings, f"{path} has no section #{anchor}"


@pytest.mark.parametrize("backend", rollout_backends())
@pytest.mark.parametrize("tower", sorted(_DENSE_COMPOSITES))
def test_a_dense_tower_under_the_mistral3_spelling_is_refused_before_the_load(tower, backend):
    with pytest.raises(ValueError, match=f"no routed experts: its {tower} text tower is dense") as refusal:
        reject_unsyncable_config(_DENSE_COMPOSITES[tower](), backend)
    message = str(refusal.value)
    assert "Train it offline" in message and "dense text checkpoint" not in message, (
        "the remedy must be one that exists"
    )
    _links_resolve(message)


@pytest.mark.parametrize("backend", rollout_backends())
def test_the_plain_ministral3_tower_passes_and_the_mistral4_composite_is_not_called_dense(backend):
    """Devstral 2 123B ships a bare ``ministral3``. Mistral Small 4 is refused for its own reason where it is."""
    reject_unsyncable_config(_ministral3(), backend)
    try:
        reject_unsyncable_config(_mistral4_composite(), backend)
    except ValueError as refusal:
        assert "no routed experts" not in str(refusal)


@pytest.mark.parametrize(
    ("config", "family"),
    [
        (_ministral3_composite, None),
        (_mistral_composite, None),
        (_ministral3, None),
        (_mistral4_composite, MISTRAL4),
    ],
    ids=["ministral3-composite", "mistral-composite", "ministral3", "mistral4-composite"],
)
def test_the_tower_alone_names_the_family(config, family):
    assert find_moe_family_spec_for_tower(config()) is family


@pytest.mark.parametrize(
    ("config", "reads"),
    [
        (lambda: _ministral3(0.1), True),
        (lambda: _ministral3(0.0), False),
        (_ministral3_composite, True),
        (_mistral_composite, False),
        (_mistral4_composite, True),
        (lambda: None, False),
        (lambda: types.SimpleNamespace(model_type="qwen3"), False),
    ],
    ids=[
        "ministral3",
        "ministral3-beta-0",
        "ministral3-composite",
        "mistral-composite",
        "mistral4",
        "none",
        "no-rope",
    ],
)
def test_a_llama_4_query_scale_reads_row_positions(config, reads):
    """Devstral 2 123B's beta is 0, so its tower scales no query; Ministral 3's and Mistral 4's are 0.1."""
    assert attention_reads_row_positions(config()) is reads


def _logits(model, positions: torch.Tensor) -> torch.Tensor:
    ids = torch.arange(3, 11).unsqueeze(0)
    with torch.no_grad():
        return model(input_ids=ids, position_ids=positions.unsqueeze(0)).logits


# Fifteen windows on, the scale is 1 + beta * ln(16): 1.277 at beta 0.1.
_SHIFT = 15 * _WINDOW


@pytest.mark.parametrize(("beta", "moves"), [(0.1, True), (0.0, False)], ids=["beta-0.1", "beta-0"])
def test_a_document_shifted_past_the_window_scores_differently_under_the_scale(beta, moves):
    """The premise of the predicate: RoPE alone sees only relative distance, so a shifted row scores the same unless
    the tower scales queries by absolute position. Weights drawn wide, so attention is sharp enough to show it."""
    torch.manual_seed(0)
    model = Ministral3ForCausalLM(_ministral3(beta)).eval()
    model.config._attn_implementation = "eager"
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(std=0.5)
    assert llama_4_scaling_beta(model.config) == beta
    base = torch.arange(8)
    reference = _logits(model, base)
    gap = ((_logits(model, base + _SHIFT) - reference).abs().max() / reference.abs().max()).item()
    assert (gap > 1e-2) is moves and (gap < 1e-4 or moves), f"relative logit gap {gap:.3g}"


@pytest.mark.parametrize(
    ("config", "measured"),
    [(_ministral3_composite, False), (_ministral3, False), (_mistral_composite, True), (_mistral4_composite, True)],
    ids=["ministral3-composite", "ministral3", "mistral-composite", "mistral4-composite"],
)
def test_packing_isolation_counts_only_a_measured_tower(config, measured):
    """A wrapped ``ministral3`` counted as measured through the Mistral 4 spelling while a bare one did not."""
    assert packing_isolation_is_measured(config()) is measured


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
