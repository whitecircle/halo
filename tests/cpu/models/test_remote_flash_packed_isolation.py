#!/usr/bin/env python
"""Packed-document isolation for remote modeling files built on transformers v4's Llama flash template
(Ling 2.0, ``inclusionAI/Ling-mini-2.0``).

The file's flash class calls its own mask-only ``_flash_attention_forward``, its ``ATTENTION_CLASSES``
table and support flag know only ``flash_attention_2``, and its SDPA/eager mask never reads
``position_ids``: unpatched, a packed row is one causal sequence on every backend. The flash kernels cannot
run on CPU, so a per-document SDPA reference stands in for the varlen kernel behind transformers' dispatch,
backward and gradient-checkpointing recomputes included; only the device check of transformers' flash gate is
lifted, its ``_supports_flash_attn`` check stays live.
Every Bailing file's dense mask, Ling 3.0's included, leaks the same way, so the family refuses packing there.

    python tests/cpu/models/test_remote_flash_packed_isolation.py
"""

import sys
import types

import pytest
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers import modeling_flash_attention_utils as flash_utils

from halo.data.collators.factory import select_data_collator
from halo.models.attention.remote_flash_attention import patch_v4_flash_attention_template
from halo.models.attention.selection import VARLEN_ATTN_IMPLEMENTATIONS
from tests.common.models import BAILING_MOE_RING_MINI
from tests.common.packed_flash_varlen import MULTI_DOCUMENT_ROWS, VARLEN_KEYS, padding_tokenizer
from tests.common.tiny_models import TINY_BAILING_LING3_MLA_FAMILY, TINY_MOE_FAMILIES, word_level_tokenizer
from tests.common.tokenizers import load_cached_config

SEED = 0
DOC_A, DOC_B = 12, 9
VOCAB = 1024
# The file's gradient checkpointing: off, then each recompute torch offers (use_reentrant).
GC_MODES = {"gc_off": None, "non_reentrant": False, "reentrant": True}
# One position ramp over both documents puts ~1e2 of doc B's gradient on doc A's embeddings.
LEAK_FLOOR = 1.0


class _ReferenceVarlen:
    """Per-document causal SDPA in the flash varlen call shape, counting its calls."""

    def __init__(self):
        self.calls = 0

    def __call__(
        self,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q=None,
        max_seqlen_k=None,
        causal=False,
        softmax_scale=None,
    ):
        self.calls += 1
        bounds = cu_seqlens_q.tolist()
        outputs = [
            F.scaled_dot_product_attention(
                *(x[start:end].transpose(0, 1)[None] for x in (q, k, v)),
                is_causal=causal,
                scale=softmax_scale,
                enable_gqa=True,
            )[0].transpose(0, 1)
            for start, end in zip(bounds[:-1], bounds[1:], strict=True)
        ]
        return torch.cat(outputs)


def _reference_dense(q, k, v, causal=False, softmax_scale=None):
    out = F.scaled_dot_product_attention(
        *(x.transpose(1, 2) for x in (q, k, v)), is_causal=causal, scale=softmax_scale, enable_gqa=True
    )
    return out.transpose(1, 2)


@pytest.fixture
def reference_varlen(monkeypatch):
    """The reference kernel behind transformers' flash dispatch for every label."""
    kernel = _ReferenceVarlen()
    monkeypatch.setattr(
        flash_utils,
        "_lazy_imports",
        lambda *_a, **_k: (_reference_dense, kernel, None, flash_utils._pad_input, flash_utils._unpad_input),
    )
    monkeypatch.setattr(flash_utils, "_loaded_implementation", None)
    monkeypatch.setattr(PreTrainedModel, "_flash_attn_import_error", lambda self, **_k: None)
    return kernel


def _tiny(family, attn_implementation: str):
    torch.manual_seed(SEED)
    model = family.build({"_attn_implementation": attn_implementation})
    for param in model.parameters():
        torch.nn.init.normal_(param, std=0.1)
    return model.float().eval()


def _tiny_ling(attn_implementation: str):
    return _tiny(TINY_MOE_FAMILIES["bailing_moe"], attn_implementation)


def _doc_b_logits(model, doc_a: torch.Tensor, doc_b: torch.Tensor) -> torch.Tensor:
    position_ids = torch.cat([torch.arange(doc_a.numel()), torch.arange(doc_b.numel())])[None]
    with torch.no_grad():
        logits = model(input_ids=torch.cat([doc_a, doc_b])[None], position_ids=position_ids, use_cache=False).logits
    return logits[0, doc_a.numel() :]


def _documents():
    g = torch.Generator().manual_seed(1)
    return (torch.randint(2, VOCAB, (n,), generator=g) for n in (DOC_A, DOC_A, DOC_B))


def _packing_options(model_config) -> dict:
    return {
        "tokenizer": word_level_tokenizer(),
        "packing": True,
        "train_on_completions_only": False,
        "model_config": model_config,
    }


@pytest.mark.parametrize("label", VARLEN_ATTN_IMPLEMENTATIONS)
def test_flash_label_isolates_packed_documents(reference_varlen, label):
    """Every flash label builds, and doc B of a packed row reads as doc B alone: unchanged when doc A
    changes, equal to its own forward. Unpatched, a flash label raises at build and FA2 (the one label
    the table knows) runs the packed row as one causal sequence."""
    model = _tiny_ling(label)
    attention = model.model.layers[0].attention
    assert type(attention).__name__ == "BailingMoeV2FlashAttention2"
    doc_a, doc_a_changed, doc_b = _documents()

    packed = _doc_b_logits(model, doc_a, doc_b)
    assert reference_varlen.calls > 0, "the packed row never reached the varlen kernel"
    drift = (packed - _doc_b_logits(model, doc_a_changed, doc_b)).abs().max().item()
    with torch.no_grad():
        alone = model(input_ids=doc_b[None], use_cache=False).logits[0]
    # Not 0.0: doc A changes each expert's batch, which reorders fp32 reductions (~1e-7); a leak reads ~1e-1.
    assert drift < 1e-5, f"doc B moved {drift:.2e} when doc A changed on {label}"
    assert (packed - alone).abs().max().item() < 1e-5, f"packed doc B differs from doc B alone on {label}"


@pytest.mark.parametrize("label", VARLEN_ATTN_IMPLEMENTATIONS)
def test_flash_label_matches_the_files_sdpa_on_one_document(reference_varlen, label):
    """On one document the patched flash call computes the file's own SDPA attention, with the same
    weights: the causal flag and the softmax scale reach the kernel. The packed-row tests above compare
    the flash path with itself and cannot see either."""
    flash = _tiny_ling(label)
    sdpa = _tiny_ling("sdpa")
    sdpa.load_state_dict(flash.state_dict())
    _, _, doc_b = _documents()
    with torch.no_grad():
        flash_logits, sdpa_logits = (m(input_ids=doc_b[None], use_cache=False).logits for m in (flash, sdpa))
    assert reference_varlen.calls == 0, "one document must take the dense flash call, not varlen"
    difference = (flash_logits - sdpa_logits).abs().max().item()
    assert difference < 1e-5, f"{label} differs from the file's sdpa by {difference:.2e}"


def _doc_b_gradient_on_doc_a(model, reference_varlen, position_ids) -> tuple[float, int]:
    """Max |d(doc B loss)/d(doc A embeddings)| of one packed row, and the varlen calls its backward made."""
    doc_a, _, doc_b = _documents()
    embeds = model.get_input_embeddings()(torch.cat([doc_a, doc_b])[None]).detach().requires_grad_(True)
    logits = model(inputs_embeds=embeds, position_ids=position_ids, use_cache=False).logits
    forward_calls = reference_varlen.calls
    logits[0, DOC_A:].pow(2).sum().backward()
    return embeds.grad[0, :DOC_A].abs().max().item(), reference_varlen.calls - forward_calls


@pytest.mark.parametrize("gc", GC_MODES)
@pytest.mark.parametrize("label", VARLEN_ATTN_IMPLEMENTATIONS)
def test_flash_label_isolates_packed_document_gradients(reference_varlen, label, gc):
    """Doc B's loss puts no gradient on doc A's embeddings, with gradient checkpointing off and through
    either recompute, which re-runs every layer's varlen call on the position_ids the attention forward
    stashes again. The forward checks above cannot see a recompute that runs the row as one causal
    sequence. One position ramp over both documents must leak, or the zero would be vacuous."""
    model = _tiny_ling(label).train()
    use_reentrant = GC_MODES[gc]
    if use_reentrant is not None:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": use_reentrant})
    packed = torch.cat([torch.arange(DOC_A), torch.arange(DOC_B)])[None]

    leak, recomputed = _doc_b_gradient_on_doc_a(model, reference_varlen, packed)
    expected = 0 if use_reentrant is None else len(model.model.layers)
    assert recomputed == expected, f"the backward re-ran the varlen kernel {recomputed} times, expected {expected}"
    assert leak == 0.0, f"doc B's loss reaches doc A's embeddings by {leak:.2e} on {label} ({gc})"
    ramp_leak, _ = _doc_b_gradient_on_doc_a(model, reference_varlen, torch.arange(DOC_A + DOC_B)[None])
    assert ramp_leak > LEAK_FLOOR, f"one causal sequence leaked only {ramp_leak:.2e}: the check sees no leak"


@pytest.mark.parametrize("label", VARLEN_ATTN_IMPLEMENTATIONS)
def test_a_padded_batch_matches_the_files_sdpa(reference_varlen, label):
    """A batch with a right- and a left-padded row computes the file's SDPA logits at every real token on
    each flash label: the model's padding-mask switch holds for every flash label, so the flash call unpads
    the 2D mask through the varlen kernel. The left-padded row's real tokens follow its pads, so a flash call
    that drops the mask moves their logits. The packed rows above carry no attention_mask and cannot see it."""
    flash = _tiny_ling(label)
    sdpa = _tiny_ling("sdpa")
    sdpa.load_state_dict(flash.state_dict())
    input_ids = torch.randint(2, VOCAB, (3, DOC_A), generator=torch.Generator().manual_seed(3))
    attention_mask = torch.ones_like(input_ids)
    attention_mask[1, DOC_B:] = 0
    attention_mask[2, : DOC_A - DOC_B] = 0
    with torch.no_grad():
        flash_logits, sdpa_logits = (
            m(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits for m in (flash, sdpa)
        )
    assert reference_varlen.calls > 0, "the padded batch never reached the varlen kernel"
    difference = (flash_logits - sdpa_logits)[attention_mask.bool()].abs().max().item()
    assert difference < 1e-5, f"{label} differs from the file's sdpa by {difference:.2e} on a padded batch"


def test_ling_drops_the_collators_varlen_set_and_derives_it_in_every_layer(reference_varlen, monkeypatch):
    """The packing collator hands a flash run's multi-document row the varlen set, but Ling 2.0's remote forward
    passes its layers no extra kwargs: every attention layer derives the set from the stashed position_ids, as it
    does without the set, to the same logits. The packed varlen sweep skips the family for this
    (``SET_DROPPING_FAMILIES``); a forward that starts reading the set fails here and belongs in the sweep."""
    model = _tiny_ling("flash_attention_4")
    collator = select_data_collator(
        padding_tokenizer(), train_on_completions_only=False, packing=True, model_config=model.config
    )
    handed = dict(collator.torch_call([dict(row) for row in MULTI_DOCUMENT_ROWS]))
    handed.pop("labels")
    assert all(key in handed for key in VARLEN_KEYS), sorted(handed)
    derivations = {"count": 0}
    derive = flash_utils._prepare_from_posids

    def counted(*args, **kwargs):
        derivations["count"] += 1
        return derive(*args, **kwargs)

    monkeypatch.setattr(flash_utils, "_prepare_from_posids", counted)
    with torch.no_grad():
        with_set = model(**handed, use_cache=False).logits
        with_set_derivations = derivations["count"]
        without = model(**{key: handed[key] for key in handed.keys() - set(VARLEN_KEYS)}, use_cache=False).logits
    without_derivations = derivations["count"] - with_set_derivations
    assert with_set_derivations == without_derivations == len(model.model.layers)
    assert torch.equal(with_set, without)


def test_sdpa_leaks_and_packing_is_refused_there(reference_varlen):
    """Each file's dense path is its own (its mask ignores position_ids), so packing is refused there for the
    whole family and accepted on Ling 2.0's flash label. Ling 3.0 runs all-MLA: its KDA layers cross
    documents on every backend and would hide what the mask does. Ring's forward needs CUDA, so its release
    config is checked alone. If an sdpa drift ever reads ~0, that file builds a packed mask now: revisit
    ``BAILING.dense_packing_leaks``."""
    doc_a, doc_a_changed, doc_b = _documents()
    for name, family in (("Ling 2.0", TINY_MOE_FAMILIES["bailing_moe"]), ("Ling 3.0", TINY_BAILING_LING3_MLA_FAMILY)):
        sdpa = _tiny(family, "sdpa")
        drift = (_doc_b_logits(sdpa, doc_a, doc_b) - _doc_b_logits(sdpa, doc_a_changed, doc_b)).abs().max().item()
        assert drift > 1e-2, f"{name}: doc B moved only {drift:.2e} on sdpa"
        with pytest.raises(ValueError, match="packed row runs as ONE dense"):
            select_data_collator(**_packing_options(sdpa.config))

    ring = load_cached_config(BAILING_MOE_RING_MINI, trust_remote_code=True)
    ring._attn_implementation = "sdpa"
    with pytest.raises(ValueError, match="packed row runs as ONE dense"):
        select_data_collator(**_packing_options(ring))
    select_data_collator(**_packing_options(_tiny_ling("flash_attention_4").config))


def test_patch_marks_only_the_template_and_is_idempotent(reference_varlen):
    """The loaded Ling module is patched once; a flash method with another signature is left alone."""
    model = _tiny_ling("flash_attention_2")
    module = sys.modules[type(model).__module__]
    flash_cls = module.ATTENTION_CLASSES["flash_attention_2"]
    assert all(module.ATTENTION_CLASSES[label] is flash_cls for label in VARLEN_ATTN_IMPLEMENTATIONS)
    assert type(model)._supports_flash_attn is True
    assert patch_v4_flash_attention_template(module) is False

    class _SlidingFlash:
        def _flash_attention_forward(self, q, k, v, attention_mask, query_length, dropout=0.0, use_sliding=False):
            return q

    other = types.ModuleType("other_remote_modeling")
    other.ATTENTION_CLASSES = {"flash_attention_2": _SlidingFlash}
    assert patch_v4_flash_attention_template(other) is False
    assert other.ATTENTION_CLASSES.keys() == {"flash_attention_2"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
