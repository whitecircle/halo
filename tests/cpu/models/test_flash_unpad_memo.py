"""``patch_transformers_flash_unpad_memo``: a padded batch on a flash backend derives its unpad set once per mask
kind per forward, at any depth, and its checkpoint recompute derives none, with logits and gradients bitwise equal
to transformers' per-layer derivation.

transformers' own flash dispatch runs on CPU reference kernels; every host read made outside them is counted.
"""

import inspect
from dataclasses import dataclass

import pytest
import torch
from transformers import modeling_flash_attention_utils as flash_utils

from halo.model_loading import model_preparation
from halo.models.attention.backend_setup import patch_transformers_flash_unpad_memo
from halo.models.gradient_checkpointing import enable_scoped_gradient_checkpointing
from tests.common.packed_flash_varlen import AttentionHostReads, ReferenceFlash, hide_kernels
from tests.common.tiny_models import TINY_DENSE_ATTENTION_FAMILIES, TINY_DENSE_FAMILY, tiny_family_model

FA2 = "flash_attention_2"
# The transformers module globals the flash path binds its kernels and the derivation to.
_FLASH_STATE = (
    "_loaded_implementation",
    "_flash_fn",
    "_flash_varlen_fn",
    "_flash_with_kvcache_fn",
    "_pad_fn",
    "_unpad_fn",
    "_process_flash_kwargs_fn",
    "_process_with_kvcache_kwargs_fn",
    "_get_unpad_data",
    "_upad_input",
)
_SEQ = 24
# family -> (tiny family, the mask kinds its layers read: Gemma 2 alternates sliding and full attention)
_FAMILIES = {"qwen3": (TINY_DENSE_FAMILY, 1), "gemma2": (TINY_DENSE_ATTENTION_FAMILIES["gemma2"], 2)}
# The reads one derivation makes (the nonzero and the max length's item), and the all-ones check transformers'
# flash mask makes once per mask kind.
_DERIVATION_READS = 2
_MASK_CHECK_READS = 1


@dataclass
class FlashPath:
    """The reference kernels bound into transformers' flash path, and how many unpad sets it has derived."""

    kernels: ReferenceFlash
    derivations: int = 0


@pytest.fixture
def flash_path(monkeypatch) -> FlashPath:
    """CPU reference kernels bound where transformers' flash path caches its loaded ones, with the path's padding
    helpers, and its unpad derivation counted beneath any memo installed later; every global the path and the
    memo rebind is restored afterwards."""
    for name in _FLASH_STATE:
        monkeypatch.setattr(flash_utils, name, getattr(flash_utils, name))
    path = FlashPath(ReferenceFlash())
    flash_utils._loaded_implementation = FA2
    flash_utils._flash_fn = path.kernels.dense
    flash_utils._flash_varlen_fn = path.kernels.varlen
    flash_utils._flash_with_kvcache_fn = None
    flash_utils._pad_fn = flash_utils._pad_input
    flash_utils._unpad_fn = flash_utils._unpad_input
    flash_utils._process_flash_kwargs_fn = flash_utils._lazy_define_process_function(path.kernels.varlen)
    derive = flash_utils._get_unpad_data
    # A flash load earlier in the process installs the memo process-wide; count transformers' own derivation.
    if getattr(derive, "_halo_unpad_memo", False):
        derive = inspect.unwrap(derive)

    def counted(attention_mask):
        path.derivations += 1
        return derive(attention_mask)

    flash_utils._get_unpad_data = counted
    return path


def _padded_batch(vocab_size: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    input_ids = torch.randint(3, vocab_size, (2, _SEQ))
    attention_mask = torch.ones(2, _SEQ, dtype=torch.long)
    attention_mask[1, _SEQ - 7 :] = 0
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids}


@dataclass
class Step:
    logits: torch.Tensor
    grads: dict[str, torch.Tensor]
    forward_reads: list[str]
    forward_derivations: int
    recompute_derivations: int


def _step(path: FlashPath, family, layers: int, *, reentrant: bool) -> Step:
    """One checkpointed training step of ``family``'s tiny model at ``layers`` depth on a right-padded batch, through
    transformers' flash path: its logits and gradients, the forward's host reads, and the unpad sets the forward and
    the backward's recompute derived."""
    torch.manual_seed(1234)
    model = tiny_family_model(family, overrides={"num_hidden_layers": layers}).train()
    model.config._attn_implementation = FA2
    enable_scoped_gradient_checkpointing(model, gradient_checkpointing_kwargs={"use_reentrant": reentrant})
    batch = _padded_batch(model.config.vocab_size)
    reads = AttentionHostReads()
    hide_kernels(reads)
    path.derivations = 0
    with reads:
        out = reads.scoped(model, 1)(**batch, use_cache=False)
    forward_derivations = path.derivations
    out.loss.backward()
    return Step(
        logits=out.logits.detach(),
        grads={name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None},
        forward_reads=list(reads.reads),
        forward_derivations=forward_derivations,
        recompute_derivations=path.derivations - forward_derivations,
    )


@pytest.mark.parametrize("reentrant", [True, False], ids=["reentrant", "non_reentrant"])
@pytest.mark.parametrize("family", list(_FAMILIES))
def test_a_padded_forward_derives_its_unpad_set_once_per_mask_at_any_depth(flash_path, family, reentrant):
    """With the memo, each mask kind costs one derivation and one all-ones check per forward whatever the depth, and
    the recompute derives nothing; without it, every attention layer derives the set again, forward and recompute
    (the negative control). Logits and gradients are bitwise equal either way."""
    tiny, mask_kinds = _FAMILIES[family]
    unpatched = {layers: _step(flash_path, tiny, layers, reentrant=reentrant) for layers in (2, 4)}
    patch_transformers_flash_unpad_memo()
    for layers, base in unpatched.items():
        assert base.forward_derivations == base.recompute_derivations == layers, (layers, base)
        assert len(base.forward_reads) == mask_kinds * _MASK_CHECK_READS + layers * _DERIVATION_READS
        memo = _step(flash_path, tiny, layers, reentrant=reentrant)
        assert (memo.forward_derivations, memo.recompute_derivations) == (mask_kinds, 0), (layers, memo)
        assert len(memo.forward_reads) == mask_kinds * (_MASK_CHECK_READS + _DERIVATION_READS), memo.forward_reads
        assert torch.equal(memo.logits, base.logits)
        assert memo.grads.keys() == base.grads.keys()
        assert all(torch.equal(memo.grads[name], grad) for name, grad in base.grads.items())
    assert any(call[0] == "varlen" for call in flash_path.kernels.calls), "the padded batch never reached varlen"


def test_an_in_place_write_to_the_mask_re_derives_the_set(flash_path):
    """The memo is keyed on the mask's version counter: a mask written in place after a derivation gets the set
    of its new contents. A memo keyed on the tensor alone returns the stale set (the negative control)."""
    original = flash_utils._get_unpad_data
    patch_transformers_flash_unpad_memo()
    mask = torch.ones(2, 8, dtype=torch.bool)
    first = flash_utils._get_unpad_data(mask)
    assert flash_utils._get_unpad_data(mask) is first
    mask[0, 5:] = False
    rederived = flash_utils._get_unpad_data(mask)
    expected = original(mask)
    assert torch.equal(rederived[0], expected[0]) and torch.equal(rederived[1], expected[1])
    assert rederived[2] == expected[2]
    assert not torch.equal(rederived[1], first[1]), "the write did not change the set, so the check cannot fail"


def test_an_inference_tensor_derives_on_every_call(flash_path):
    """An inference tensor keeps no version counter to key a memo on, so its set is derived on every call, equal to
    transformers' own."""
    original = flash_utils._get_unpad_data
    patch_transformers_flash_unpad_memo()
    with torch.inference_mode():
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
        derived = flash_utils._get_unpad_data(mask)
        expected = original(mask)
    assert all(torch.equal(a, b) for a, b in zip(derived[:2], expected[:2], strict=True))
    assert derived[2] == expected[2]


def test_the_patch_refuses_a_path_that_no_longer_reads_the_module_global(flash_path, monkeypatch):
    """If transformers' ``_upad_input`` stops calling the module-level ``_get_unpad_data``, the replacement would never
    run: the install raises instead of leaving every layer deriving the set unnoticed."""

    def upad_input_elsewhere(query_layer, key_layer, value_layer, attention_mask, query_length, unpad_input_func):
        return unpad_input_func(query_layer, attention_mask)

    monkeypatch.setattr(flash_utils, "_upad_input", upad_input_elsewhere)
    with pytest.raises(RuntimeError, match="_get_unpad_data"):
        patch_transformers_flash_unpad_memo()


@pytest.mark.parametrize(
    ("attn_implementation", "installs"),
    [("flash_attention_2", True), ("flash_attention_3", True), ("flash_attention_4", True), ("sdpa", False)],
)
def test_every_flash_load_installs_the_memo(flash_path, monkeypatch, attn_implementation, installs):
    """``apply_family_patches`` runs before every policy, frozen and trainer-path build, so each flash backend's load
    takes the memo and no other backend's does."""
    monkeypatch.setattr(model_preparation, "patch_transformers_flash_varlen_int_seqlen", lambda: None)
    model_preparation.apply_family_patches(tiny_family_model(TINY_DENSE_FAMILY).config, attn_implementation)
    assert getattr(flash_utils._get_unpad_data, "_halo_unpad_memo", False) is installs


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
