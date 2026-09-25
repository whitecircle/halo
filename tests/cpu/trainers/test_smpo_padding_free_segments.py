#!/usr/bin/env python
"""SMPO padding-free keeps each flattened document's conv / linear-attention state its own.

``_forward_padding_free`` runs the chosen ⧺ rejected batch as one ``[1, total]`` row. Attention keeps
its documents apart through the resetting ``position_ids``; LFM2's ShortConv and the GatedDeltaNet
conv + chunked delta rule (Qwen3.5/3.6, Qwen3-Next) only through the segment markers their forward
reads from kwargs. So every document's mean log-prob from that one forward must equal the same
document run alone, and a family that reads no marker must get exactly the kwargs it always did.

The CPU forward runs the torch fallbacks, which honor the markers through the device-aware dispatch
fixups (``src/models/patches/kernel_dispatch.py``); the CUDA kernels are covered by
``tests/gpu/trainers/preference/test_smpo_padding_free_segments.py``.

    python tests/cpu/trainers/test_smpo_padding_free_segments.py
"""

from types import SimpleNamespace

import pytest
import torch

from src.configs.smpo_config import SmoothMarginPOConfig
from src.models import segment_markers
from src.models.segment_markers import GDN_SEGMENT_AWARE_BACKENDS, SegmentMarkers, segment_markers_for
from src.trainers.preference.smpo import SmoothMarginPOTrainer
from tests.common.segment_isolation import (
    FAMILIES,
    LEAK_CONTROLS,
    PAD_ID,
    lone_document_logps,
    preference_batch,
    tiny_model,
)

# Same tokens, same weights: only the matmul reduction order differs between the flattened row and a
# lone document, so fp32 agreement is at rounding level.
ISOLATION_TOL = 1e-5
# Without markers the mixers carry state across documents; the drift must clear the isolation
# tolerance by orders of magnitude, or the equality would hold for a model that leaks too.
LEAK_FLOOR = 1e-3
# The forward kwargs the flattened row always carried.
BASE_FORWARD_KWARGS = {"input_ids", "position_ids", "use_cache"}
VARLEN_KWARGS = {"cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"}


def _model(family: str):
    # SDPA builds the packed-document mask from the resetting position_ids; flash has no CPU kernel.
    return tiny_model(family, "sdpa")


def _trainer(markers: SegmentMarkers) -> SmoothMarginPOTrainer:
    """A construction-free SMPO carrying only what the padding-free forward reads."""
    trainer = object.__new__(SmoothMarginPOTrainer)
    trainer.padding_free = True
    trainer.pad_token_id = PAD_ID
    trainer.label_pad_token_id = -100
    trainer.parallelism_config = SimpleNamespace(cp_size=1)  # cp_size is a mixin property over it
    trainer.lower_clip_percentile = trainer.upper_clip_percentile = trainer.min_log_prob = None
    trainer._segment_markers = markers
    return trainer


def _padding_free_logps(model, markers: SegmentMarkers) -> tuple[torch.Tensor, set[str]]:
    """Per-document mean log-probs from SMPO's padding-free forward, plus the kwargs the model received."""
    received: set[str] = set()

    def record(_module, _args, kwargs):
        received.update(kwargs)

    handle = model.register_forward_pre_hook(record, with_kwargs=True)
    try:
        with torch.no_grad():
            out = _trainer(markers).concatenated_forward(model, preference_batch())
    finally:
        handle.remove()
    return torch.cat([out["chosen_logps"], out["rejected_logps"]]), received


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_padding_free_documents_match_their_lone_forward(family):
    model = _model(family)
    logps, _ = _padding_free_logps(model, segment_markers_for(model.config))
    drift = (logps - lone_document_logps(model)).abs().max().item()
    assert drift <= ISOLATION_TOL, (
        f"{family}: a document's padding-free log-prob moved by {drift:.3e} from the same document run "
        f"alone — state crossed a document boundary of the flattened row"
    )


@pytest.mark.parametrize(
    ("family", "control"), [(family, control) for family, controls in LEAK_CONTROLS.items() for control in controls]
)
def test_without_markers_the_mixers_leak(family, control):
    """Anti-vacuity: the isolation above is the markers' doing, not a model that ignores its context."""
    model = _model(family)
    logps, _ = _padding_free_logps(model, LEAK_CONTROLS[family][control])
    drift = (logps - lone_document_logps(model)).abs().max().item()
    assert drift > LEAK_FLOOR, (
        f"{family} ({control}): the flattened row stayed isolated (drift {drift:.3e}) — this model does "
        f"not exercise the conv / linear-attention crossing the missing markers exist to stop"
    )


@pytest.mark.parametrize(
    ("family", "extra_kwargs"),
    [("qwen3", set()), ("lfm2", {"seq_idx"}), ("qwen3_5", {"seq_idx"} | VARLEN_KWARGS)],
)
def test_forward_receives_only_the_family_markers(family, extra_kwargs):
    """Family-gated: a family reading no marker keeps the exact pre-marker call."""
    model = _model(family)
    _, received = _padding_free_logps(model, segment_markers_for(model.config))
    assert received == BASE_FORWARD_KWARGS | extra_kwargs


@pytest.mark.parametrize("missing", [name for name, _ in GDN_SEGMENT_AWARE_BACKENDS])
def test_gdn_padding_free_refused_without_segment_aware_kernels(monkeypatch, tmp_path, missing):
    """The torch fallbacks drop both markers, so the trainer refuses the run the collator refuses."""
    patched = tuple((name, (lambda ok=(name != missing): ok)) for name, _ in GDN_SEGMENT_AWARE_BACKENDS)
    monkeypatch.setattr(segment_markers, "GDN_SEGMENT_AWARE_BACKENDS", patched)
    with pytest.raises(ValueError, match="padding_free") as excinfo:
        SmoothMarginPOTrainer(
            model=tiny_model("qwen3_5", "flash_attention_2"),
            args=SmoothMarginPOConfig(output_dir=str(tmp_path), padding_free=True, bf16=False, report_to="none"),
            processing_class=SimpleNamespace(pad_token_id=PAD_ID, eos_token_id=1),
            is_vlm=False,
        )
    assert missing in str(excinfo.value)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
