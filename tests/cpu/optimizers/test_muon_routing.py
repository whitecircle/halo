#!/usr/bin/env python
"""Muon factory contract tests (src/optimizers/muon.py) — CPU-only.

The standard Muon recipe orthogonalizes only the hidden 2D weight matrices and keeps the token
embedding and output head on AdamW; routing on ``ndim`` alone sends those 2D tensors through
Newton-Schulz. Both halves of that decision are pinned here: the name-based exclusion
(``_is_embedding_or_head`` — embeddings, LM head, and the vocab-free ``score`` head; ``shared``
matches only as an exact component, for T5's tied embedding) and the structural fallback that
reads the accessors a family declares, so a new spelling needs no marker edit.

Also pinned, each a silent failure mode: the marker set has ONE home shared with
``_peft_module_casting_to_bf16``, MoE shared-expert FFN matrices take the Muon path, non-bf16
Muon-routed params are rejected (the Triton step writes bf16), ``state_dict`` /
``load_state_dict`` round-trip the internal scalar-optimizer (AdamW) state, and the toolkit's
subclass keeps the class name every optimizer shard is fingerprinted with.

Run: pytest tests/cpu/optimizers/test_muon_routing.py
"""

import pytest
import torch
import torch.nn as nn
from gram_newton_schulz import Muon as UpstreamMuon

from src.models.structure import EMBEDDING_HEAD_MARKERS
from src.optimizers import muon
from src.optimizers.muon import _is_embedding_or_head, create_muon_optimizer


class _TinyMoE(nn.Module):
    """One hidden matrix, one shared-expert matrix, an embedding, and a bias."""

    def __init__(self, dtype=torch.bfloat16):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 8, dtype=dtype)
        self.q_proj = nn.Linear(8, 8, bias=True, dtype=dtype)
        self.shared_expert = nn.Linear(8, 8, bias=False, dtype=dtype)
        self.lm_head = nn.Linear(8, 16, bias=False, dtype=dtype)


def test_shared_expert_routes_to_muon():
    """``shared`` matches only as an exact component, so a shared-expert FFN is NOT an embedding."""
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
        "shared.weight",  # T5 tied embedding
        "model.embeddings.weight",
    ],
)
def test_embeddings_and_head_excluded(name):
    assert _is_embedding_or_head(name) is True


@pytest.mark.parametrize("name", ["score.weight", "base_model.model.score.weight", "model.score.dense.weight"])
def test_a_pooled_score_head_is_excluded_by_name_too(name):
    """The reward/classification head is vocab-free but still a HEAD, and the name path must say so.

    ``_embedding_and_head_param_ids`` already resolves a ``score`` module the model exposes as an
    attribute — but that is exactly the signal missing on the models this name fallback exists for
    (no accessors), and on a PEFT-wrapped tree where the head sits under ``base_model.model``. With
    ``score`` absent from the marker set those 2D rows go to Newton-Schulz while the very same head,
    reached through the accessor, goes to AdamW: one head, two optimizers, decided by wrapping.
    """
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
    """One definition of "embedding or head", or the two consumers drift apart.

    ``_peft_module_casting_to_bf16`` casts exactly this set of modules; Muon routes exactly this set
    off Newton-Schulz. Kept as two literals they drift in BOTH directions — ``score`` missing on one
    side, ``embeddings``/``word_embeddings`` on the other — so each consumer mishandles a head the
    other handles. This pins the single home.
    """
    assert muon.EMBEDDING_HEAD_MARKERS is EMBEDDING_HEAD_MARKERS, "muon re-declared its own marker list"
    assert not hasattr(muon, "_EMBEDDING_HEAD_MARKERS"), "the private copy is back"
    for marker in ("embed_tokens", "lm_head", "embeddings", "word_embeddings", "wte", "wpe", "score"):
        assert marker in EMBEDDING_HEAD_MARKERS, f"{marker!r} left the shared marker set"


def test_routing_by_module():
    model = _TinyMoE()
    opt = create_muon_optimizer(model, ns_use_kernels=False)
    muon_params = {id(p) for g in opt._muon_param_groups for p in g["params"]}
    scalar_params = {id(p) for g in opt.scalar_optimizer.param_groups for p in g["params"]}
    assert id(model.shared_expert.weight) in muon_params
    assert id(model.q_proj.weight) in muon_params
    assert id(model.embed_tokens.weight) in scalar_params
    assert id(model.lm_head.weight) in scalar_params
    assert id(model.q_proj.bias) in scalar_params


class _UnconventionalNames(nn.Module):
    """A family spelling its embedding/head outside ``_EMBEDDING_HEAD_MARKERS``.

    ``embed_in`` / ``embed_out`` are GPT-NeoX's real spellings and match none of the markers.
    The model declares them the way every ``PreTrainedModel`` does — via the accessors.
    """

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
    """A family whose embedding/head names match no marker must still land on AdamW.

    This is the point of routing on the module the model declares rather than on a name list:
    adding a family must not require editing ``_EMBEDDING_HEAD_MARKERS``. Newton-Schulz on a
    vocab-indexed matrix is empirically harmful and would be applied silently.
    """
    model = _UnconventionalNames()
    # The name-based fallback alone would misroute both — that is exactly the gap being closed.
    assert not _is_embedding_or_head("embed_in.weight")
    assert not _is_embedding_or_head("embed_out.weight")

    opt = create_muon_optimizer(model, ns_use_kernels=False)
    muon_params = {id(p) for g in opt._muon_param_groups for p in g["params"]}
    scalar_params = {id(p) for g in opt.scalar_optimizer.param_groups for p in g["params"]}
    assert id(model.embed_in.weight) in scalar_params
    assert id(model.embed_out.weight) in scalar_params
    assert id(model.w_qkv.weight) in muon_params, "hidden matrices must still take the Muon path"


def test_tied_embedding_head_routes_once_to_adamw():
    """A tied head shares storage with the embedding — it must not leak onto Muon."""
    model = _UnconventionalNames()
    model.embed_out.weight = model.embed_in.weight
    opt = create_muon_optimizer(model, ns_use_kernels=False)
    muon_params = {id(p) for g in opt._muon_param_groups for p in g["params"]}
    assert id(model.embed_in.weight) not in muon_params


def test_non_bf16_muon_param_rejected():
    try:
        create_muon_optimizer(_TinyMoE(dtype=torch.float32), ns_use_kernels=False)
    except ValueError as e:
        assert "bf16" in str(e)
    else:
        raise AssertionError("fp32 Muon-routed params must raise (Triton step writes bf16)")


def test_the_toolkit_muon_is_a_subclass_carrying_its_own_step_and_name():
    """Three properties of the subclass, each silent when it breaks.

    ``OptimizerStateFingerprint`` records ``type(optimizer).__name__`` in every optimizer shard and
    refuses a resume whose class differs, so renaming the class warm-restarts every in-flight Muon
    run. The matrix step must be the toolkit's fused Triton one, not upstream's per-batch loop. And
    the scalar leg must be AdamWBF16's own bound ``step`` — upstream wraps it in ``torch.compile``,
    which only pays graph breaks around a kernel launch.
    """
    opt = create_muon_optimizer(_TinyMoE(), ns_use_kernels=False)

    assert type(opt).__name__ == "Muon", "the fingerprint's optimizer_class changed"
    assert isinstance(opt, UpstreamMuon) and type(opt) is not UpstreamMuon
    assert type(opt)._muon_step is not UpstreamMuon._muon_step, "upstream's matrix step is back"
    assert opt._compiled_scalar_step == opt.scalar_optimizer.step, "the scalar step is not AdamWBF16's own"


def test_state_dict_roundtrips_scalar_state():
    model = _TinyMoE()
    opt = create_muon_optimizer(model, ns_use_kernels=False)
    for p in opt.scalar_optimizer.param_groups[0]["params"]:
        p.grad = torch.zeros_like(p)
    opt.scalar_optimizer.step()
    assert len(opt.scalar_optimizer.state) > 0

    sd = opt.state_dict()
    n_scalar = len(opt.scalar_optimizer.state)
    assert len(sd["state"]) >= n_scalar, "state_dict must include the scalar-optimizer entries"

    model2 = _TinyMoE()
    opt2 = create_muon_optimizer(model2, ns_use_kernels=False)
    opt2.load_state_dict(sd)
    assert len(opt2.scalar_optimizer.state) == n_scalar, "scalar state must land back on the scalar optimizer"
    scalar_ids = {id(p) for g in opt2.scalar_optimizer.param_groups for p in g["params"]}
    assert not any(id(p) in scalar_ids for p in opt2.state), "scalar entries must not linger in Muon's own state"
    restored = next(iter(opt2.scalar_optimizer.state.values()))
    assert "exp_avg" in restored and "step" in restored


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
