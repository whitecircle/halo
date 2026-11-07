#!/usr/bin/env python
"""Muon factory contract (halo/optimizers/muon.py): hidden 2D matrices orthogonalize, embeddings and heads stay on
AdamW."""

import pytest
import torch
import torch.nn as nn
from gram_newton_schulz import Muon as UpstreamMuon

from halo.models.introspection.param_roles import EMBEDDING_HEAD_MARKERS
from halo.optimizers import muon
from halo.optimizers.muon import _is_embedding_or_head, build_muon_optimizer


class _TinyMoE(nn.Module):
    def __init__(self, dtype=torch.bfloat16):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 8, dtype=dtype)
        self.q_proj = nn.Linear(8, 8, bias=True, dtype=dtype)
        self.shared_expert = nn.Linear(8, 8, bias=False, dtype=dtype)
        self.lm_head = nn.Linear(8, 16, bias=False, dtype=dtype)


def test_shared_expert_routes_to_muon():
    """``shared`` matches only as an exact component, so a shared-expert FFN is no embedding."""
    assert not _is_embedding_or_head("model.layers.0.mlp.shared_expert.gate_proj.weight")
    assert not _is_embedding_or_head("model.layers.0.mlp.shared_experts.down_proj.weight")


@pytest.mark.parametrize(
    "name",
    [
        "model.embed_tokens.weight",
        "lm_head.weight",
        "transformer.wte.weight",
        "transformer.wpe.weight",
        "model.word_embeddings.weight",
        "shared.weight",  # T5 tied embedding.
        "model.embeddings.weight",
    ],
)
def test_embeddings_and_head_excluded(name):
    assert _is_embedding_or_head(name) is True


@pytest.mark.parametrize("name", ["score.weight", "base_model.model.score.weight", "model.score.dense.weight"])
def test_a_pooled_score_head_is_excluded_by_name_too(name):
    """Without ``score`` in the markers the same head takes Newton-Schulz through a name and AdamW through the
    accessor, decided by wrapping."""
    assert _is_embedding_or_head(name) is True


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.5.mlp.experts.0.down_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
    ],
)
def test_hidden_weights_routed_to_muon(name):
    assert _is_embedding_or_head(name) is False


def test_the_marker_set_is_shared_not_copied():
    """``_peft_module_casting_to_bf16`` casts this set and Muon routes it; as two literals they drift both ways."""
    assert muon.EMBEDDING_HEAD_MARKERS is EMBEDDING_HEAD_MARKERS, "muon re-declared its own marker list"
    assert not hasattr(muon, "_EMBEDDING_HEAD_MARKERS"), "the private copy is back"
    for marker in ("embed_tokens", "lm_head", "embeddings", "word_embeddings", "wte", "wpe", "score"):
        assert marker in EMBEDDING_HEAD_MARKERS, f"{marker!r} left the shared marker set"


def test_routing_by_module():
    model = _TinyMoE()
    opt = build_muon_optimizer(model, ns_use_kernels=False)
    muon_params = {id(p) for g in opt._muon_param_groups for p in g["params"]}
    scalar_params = {id(p) for g in opt.scalar_optimizer.param_groups for p in g["params"]}
    assert id(model.shared_expert.weight) in muon_params
    assert id(model.q_proj.weight) in muon_params
    assert id(model.embed_tokens.weight) in scalar_params
    assert id(model.lm_head.weight) in scalar_params
    assert id(model.q_proj.bias) in scalar_params


class _UnconventionalNames(nn.Module):
    """``embed_in`` / ``embed_out`` are GPT-NeoX's real spellings and match no marker."""

    def __init__(self, dtype=torch.bfloat16):
        super().__init__()
        self.embed_in = nn.Embedding(16, 8, dtype=dtype)
        self.w_qkv = nn.Linear(8, 8, bias=False, dtype=dtype)
        self.embed_out = nn.Linear(8, 16, bias=False, dtype=dtype)

    def get_input_embeddings(self):
        return self.embed_in

    def get_output_embeddings(self):
        return self.embed_out


def test_embedding_and_head_resolved_structurally_for_new_family():
    """Newton-Schulz on a vocab-indexed matrix is empirically harmful and would be applied silently."""
    model = _UnconventionalNames()
    # The name-based fallback alone would misroute both.
    assert not _is_embedding_or_head("embed_in.weight")
    assert not _is_embedding_or_head("embed_out.weight")

    opt = build_muon_optimizer(model, ns_use_kernels=False)
    muon_params = {id(p) for g in opt._muon_param_groups for p in g["params"]}
    scalar_params = {id(p) for g in opt.scalar_optimizer.param_groups for p in g["params"]}
    assert id(model.embed_in.weight) in scalar_params
    assert id(model.embed_out.weight) in scalar_params
    assert id(model.w_qkv.weight) in muon_params, "hidden matrices must still take the Muon path"


def test_tied_embedding_head_routes_once_to_adamw():
    """A tied head shares storage with the embedding."""
    model = _UnconventionalNames()
    model.embed_out.weight = model.embed_in.weight
    opt = build_muon_optimizer(model, ns_use_kernels=False)
    muon_params = {id(p) for g in opt._muon_param_groups for p in g["params"]}
    assert id(model.embed_in.weight) not in muon_params


def test_non_bf16_muon_param_rejected():
    try:
        build_muon_optimizer(_TinyMoE(dtype=torch.float32), ns_use_kernels=False)
    except ValueError as e:
        assert "bf16" in str(e)
    else:
        raise AssertionError("fp32 Muon-routed params must raise (Triton step writes bf16)")


def test_the_toolkit_muon_is_a_subclass_carrying_its_own_step_and_name():
    """The fingerprint records ``type(optimizer).__name__``, so a rename warm-restarts every in-flight Muon run;
    upstream's ``torch.compile`` wrap only pays graph breaks."""
    opt = build_muon_optimizer(_TinyMoE(), ns_use_kernels=False)

    assert type(opt).__name__ == "Muon", "the fingerprint's optimizer_class changed"
    assert isinstance(opt, UpstreamMuon) and type(opt) is not UpstreamMuon
    assert type(opt)._muon_step is not UpstreamMuon._muon_step, "upstream's matrix step is back"
    assert opt._compiled_scalar_step == opt.scalar_optimizer.step, "the scalar step is not AdamWBF16's own"


def test_state_dict_roundtrips_scalar_state():
    model = _TinyMoE()
    opt = build_muon_optimizer(model, ns_use_kernels=False)
    for p in opt.scalar_optimizer.param_groups[0]["params"]:
        p.grad = torch.zeros_like(p)
    opt.scalar_optimizer.step()
    assert len(opt.scalar_optimizer.state) > 0

    sd = opt.state_dict()
    n_scalar = len(opt.scalar_optimizer.state)
    assert len(sd["state"]) >= n_scalar, "state_dict must include the scalar-optimizer entries"

    model2 = _TinyMoE()
    opt2 = build_muon_optimizer(model2, ns_use_kernels=False)
    opt2.load_state_dict(sd)
    assert len(opt2.scalar_optimizer.state) == n_scalar, "scalar state must land back on the scalar optimizer"
    scalar_ids = {id(p) for g in opt2.scalar_optimizer.param_groups for p in g["params"]}
    assert not any(id(p) in scalar_ids for p in opt2.state), "scalar entries must not linger in Muon's own state"
    restored = next(iter(opt2.scalar_optimizer.state.values()))
    assert "exp_avg" in restored and "step" in restored


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
