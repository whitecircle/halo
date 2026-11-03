#!/usr/bin/env python
"""SMPO's padding-free row hands a flash backend its varlen set, so no attention layer derives it from
``position_ids``. Transformers' flash dispatch runs on CPU reference kernels, and the per-layer derivation is counted.
"""

from types import SimpleNamespace

import pytest
import torch
from transformers import modeling_flash_attention_utils as flash_utils

from halo.configs.preference import SMPOConfig
from halo.distributed.parallelism_config import ParallelismConfig
from halo.models import segment_markers
from halo.models.segment_markers import SegmentMarkers, segment_markers_for
from halo.trainers.preference import smpo
from halo.trainers.preference.smpo import SMPOTrainer
from tests.common.packed_flash_varlen import install_reference_flash
from tests.common.segment_isolation import PAD_ID, preference_batch, tiny_model

FA2 = "flash_attention_2"
# The transformers module globals the reference kernels and the derivation counters rebind.
_FLASH_STATE = (
    "_loaded_implementation",
    "_flash_fn",
    "_flash_varlen_fn",
    "_flash_with_kvcache_fn",
    "_process_flash_kwargs_fn",
    "_is_packed_sequence",
    "_prepare_from_posids",
    "_upad_input",
)


@pytest.fixture(autouse=True)
def _flash_state(monkeypatch):
    """Restore every flash-path global a test rebinds, and report the segment-aware kernels present."""
    for name in _FLASH_STATE:
        monkeypatch.setattr(flash_utils, name, getattr(flash_utils, name))
    monkeypatch.setattr(segment_markers, "GDN_SEGMENT_AWARE_BACKENDS", (("fla", lambda: True),))


def _trainer(model, tmp_path) -> SMPOTrainer:
    return SMPOTrainer(
        model=model,
        args=SMPOConfig(output_dir=str(tmp_path), padding_free=True, bf16=False, report_to="none"),
        processing_class=SimpleNamespace(pad_token_id=PAD_ID, eos_token_id=1),
        parallelism_config=ParallelismConfig(),
        is_vlm=False,
    )


def _step(trainer: SMPOTrainer, model, markers: SegmentMarkers):
    """One padding-free forward and backward: per-document log-probs, every gradient, the derivation's call count."""
    trainer._segment_markers = markers
    model.zero_grad(set_to_none=True)
    _kernels, counts = install_reference_flash(attn_implementation=FA2)
    out = trainer.concatenated_forward(model, preference_batch())
    logps = torch.cat([out["chosen_logps"], out["rejected_logps"]])
    logps.sum().backward()
    grads = {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}
    return logps.detach(), grads, dict(counts)


@pytest.mark.parametrize(("family", "seq_idx"), [("qwen3", False), ("lfm2", True), ("qwen3_next", True)])
def test_the_policy_load_hands_a_flash_backend_the_varlen_set(tmp_path, family, seq_idx):
    """Every flash run takes ``cu_seq_lens``; the family's own ``seq_idx`` stays as it reads it."""
    markers = _trainer(tiny_model(family, FA2), tmp_path)._segment_markers
    assert markers == SegmentMarkers(seq_idx=seq_idx, cu_seq_lens=True)


def test_no_attention_layer_derives_the_set_and_the_step_is_unchanged(tmp_path):
    """Log-probs and gradients are bitwise those of the per-layer derivation that family markers alone keep running."""
    model = tiny_model("qwen3", FA2).train()
    trainer = _trainer(model, tmp_path)
    handed_logps, handed_grads, handed = _step(trainer, model, trainer._segment_markers)
    derived_logps, derived_grads, derived = _step(trainer, model, segment_markers_for(model.config))
    layers = model.config.num_hidden_layers
    assert derived["_is_packed_sequence"] == derived["_prepare_from_posids"] == layers, derived
    assert handed["_is_packed_sequence"] == handed["_prepare_from_posids"] == 0, handed
    assert torch.equal(handed_logps, derived_logps)
    assert handed_grads.keys() == derived_grads.keys()
    assert all(torch.equal(handed_grads[name], grad) for name, grad in derived_grads.items())


def test_the_kernels_read_the_handed_set(tmp_path, monkeypatch):
    """A boundary one token later moves the log-probs, so the kernels attend over the handed set, not one rebuilt from
    ``position_ids``."""
    model = tiny_model("qwen3", FA2)
    trainer = _trainer(model, tmp_path)
    baseline, _, _ = _step(trainer, model, trainer._segment_markers)
    build = smpo.segment_marker_kwargs

    def shifted(position_ids, markers):
        kwargs = build(position_ids, markers)
        cu_seq_lens = kwargs["cu_seq_lens_q"].clone()
        cu_seq_lens[1] += 1
        longest = int(cu_seq_lens.diff().max())
        return {**kwargs, "cu_seq_lens_q": cu_seq_lens, "cu_seq_lens_k": cu_seq_lens} | {
            "max_length_q": longest,
            "max_length_k": longest,
        }

    monkeypatch.setattr(smpo, "segment_marker_kwargs", shifted)
    moved, _, _ = _step(trainer, model, trainer._segment_markers)
    assert not torch.equal(moved, baseline)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
