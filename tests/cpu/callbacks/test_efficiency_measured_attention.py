"""Measured attention work must reach the MFU accounting exactly, and fall back honestly.

The trainer costs every micro-batch's documents through the layout the callback handed it; the
callback reads that once per optimizer step and swaps it for the config term (every token in a
``max_seq_len`` document). These tests pin the document-boundary rules, the hand-off, the divisors
the measurement must carry (TP heads, CP sequence share, PP config-depth share) and the fallback
when no batch exposed a document — a MFU that silently kept the ``max_length`` model would pass
none of them.

    python tests/cpu/callbacks/test_efficiency_measured_attention.py
"""

import pytest
import torch

from src.callbacks.efficiency import EfficiencyCallback
from src.models.attention_layout import AttentionLayout, LayerAttention
from src.trainers.mixins.token_metrics import TokenMetricsMixin
from tests.common.parallelism import make_parallelism_config

_NO_PARALLELISM = make_parallelism_config(world_size=1, gpus_per_node=1)

LINEAR_FLOPS_PER_TOKEN = 1.0e9
MAX_SEQ_LEN = 4096
PEAK_FLOPS = 1.0e15

LAYOUT = AttentionLayout((LayerAttention("full", 1024.0), LayerAttention("window", 1024.0, keys=128)))


class _Handler:
    def __init__(self, callbacks):
        self.callbacks = callbacks


class _Trainer(TokenMetricsMixin):
    def __init__(self, callbacks):
        self.callback_handler = _Handler(callbacks)
        self.parallelism_config = _NO_PARALLELISM


def _batch(lengths) -> dict:
    """A padding-free batch of documents with the given lengths."""
    lengths = torch.as_tensor(lengths)
    return {"cu_seq_lens_q": torch.cat((torch.zeros(1, dtype=torch.long), lengths.cumsum(0)))}


def _documents(lengths) -> list[int]:
    """The non-empty entries of an extracted length buffer, sorted."""
    return sorted(int(length) for length in lengths.tolist() if length)


def _callback(parallelism=_NO_PARALLELISM, layout=LAYOUT, attention_share=1.0):
    callback = EfficiencyCallback(parallelism)
    callback.state.attention_layout = layout
    callback.state.attention_share = attention_share
    fallback = layout.flops_per_token(MAX_SEQ_LEN) * attention_share / max(parallelism.tp_size, 1) if layout else 0.0
    callback.state.attention_flops_per_token = fallback
    callback.state.model_flops_per_token = LINEAR_FLOPS_PER_TOKEN + fallback
    callback.state.active_model_flops_per_token = 0.5 * LINEAR_FLOPS_PER_TOKEN + fallback
    callback.state.gpu_peak_flops = PEAK_FLOPS
    callback.state.elapsed_time = 1.0
    return callback


def _bound(parallelism=_NO_PARALLELISM, **kwargs):
    callback = _callback(parallelism, **kwargs)
    trainer = _Trainer([callback])
    return callback, trainer


# --- document lengths off the batch -------------------------------------------------------------


def test_cu_seq_lens_give_the_document_lengths():
    trainer = _Trainer([])
    lengths = trainer._extract_document_lengths({"cu_seq_lens_q": torch.tensor([0, 3, 10, 12])})
    assert lengths.tolist() == [3, 7, 2]


def test_position_id_resets_split_a_packed_row_and_skip_padding():
    inputs = {
        "input_ids": torch.zeros(2, 8, dtype=torch.long),
        "position_ids": torch.tensor([[0, 1, 2, 0, 1, 0, 1, 2], [0, 1, 2, 3, 4, 0, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0, 0]]),
    }
    assert _documents(_Trainer([])._extract_document_lengths(inputs)) == [2, 3, 3, 5]


def test_every_row_starts_a_document_even_without_a_position_reset():
    """Positions that never reset (an offset row) must not merge one row into its predecessor."""
    inputs = {
        "input_ids": torch.zeros(2, 3, dtype=torch.long),
        "position_ids": torch.tensor([[5, 6, 7], [8, 9, 10]]),
    }
    assert _documents(_Trainer([])._extract_document_lengths(inputs)) == [3, 3]


def test_left_padding_with_zero_positions_is_not_a_document():
    inputs = {
        "input_ids": torch.zeros(1, 6, dtype=torch.long),
        "position_ids": torch.tensor([[0, 0, 0, 0, 1, 2]]),
        "attention_mask": torch.tensor([[0, 0, 0, 1, 1, 1]]),
    }
    assert _documents(_Trainer([])._extract_document_lengths(inputs)) == [3]


def test_a_padded_batch_is_one_document_per_row_over_its_mask():
    inputs = {
        "input_ids": torch.zeros(2, 6, dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]]),
    }
    assert _Trainer([])._extract_document_lengths(inputs).tolist() == [4, 6]


def test_bare_input_ids_are_full_width_documents():
    assert _Trainer([])._extract_document_lengths({"input_ids": torch.zeros(3, 5, dtype=torch.long)}).tolist() == [
        5,
        5,
        5,
    ]


def test_a_batch_without_input_ids_exposes_no_documents():
    assert _Trainer([])._extract_document_lengths({"chosen_input_ids": torch.zeros(1, 4)}) is None
    assert _Trainer([])._extract_document_lengths(None) is None


# --- the hand-off ------------------------------------------------------------------------------


def test_the_first_accumulation_binds_the_callback_and_takes_its_layout():
    callback, trainer = _bound()
    trainer._accumulate_attention_flops(_batch([100, 200]))
    assert trainer._attention_layout is LAYOUT
    assert callback._attention_source() is trainer


def test_a_callback_without_a_layout_keeps_the_trainer_accumulator_off():
    callback, trainer = _bound(layout=None)
    trainer._accumulate_attention_flops(_batch([100, 200]))
    assert trainer._attention_layout is None
    assert trainer.drain_attention_flops() == (0.0, 0)
    assert callback._drain_measured_attention_flops() is None


def test_measured_work_replaces_the_config_term_in_both_estimates():
    callback, trainer = _bound()
    lengths = torch.tensor([500] * 7 + [250])
    trainer._accumulate_attention_flops(_batch(lengths[:5]))
    trainer._accumulate_attention_flops(_batch(lengths[5:]))
    measured = callback._drain_measured_attention_flops()
    assert measured == pytest.approx(float(LAYOUT.flops_for_documents(lengths)), rel=1e-12)

    tokens = int(lengths.sum())
    callback._compute_mfu(tokens, 0.5, measured)
    callback._compute_smfu(tokens, 0.5, measured)
    assert callback.state.total_flops == pytest.approx(tokens * LINEAR_FLOPS_PER_TOKEN + measured, rel=1e-12)
    assert callback.state.total_active_flops == pytest.approx(
        tokens * 0.5 * LINEAR_FLOPS_PER_TOKEN + measured, rel=1e-12
    )
    # The accumulator was reset by the drain.
    assert callback._drain_measured_attention_flops() is None


def test_the_measurement_differs_from_the_config_term_on_short_documents():
    """The premise: at these lengths the config term over-reports, so a fallback would be visible."""
    lengths = torch.tensor([500] * 7 + [250])
    tokens = int(lengths.sum())
    assert float(LAYOUT.flops_for_documents(lengths)) < 0.25 * tokens * LAYOUT.flops_per_token(MAX_SEQ_LEN)


def test_without_documents_the_step_keeps_the_config_term():
    callback, trainer = _bound()
    trainer._accumulate_attention_flops({"chosen_input_ids": torch.zeros(1, 4)})
    measured = callback._drain_measured_attention_flops()
    assert measured is None
    callback._compute_mfu(1000, 0.5, measured)
    assert callback.state.total_flops == pytest.approx(1000 * callback.state.model_flops_per_token, rel=1e-12)


def test_an_unbound_callback_keeps_the_config_term():
    callback = _callback()
    callback._compute_mfu(1000, 0.5, callback._drain_measured_attention_flops())
    assert callback.state.total_flops == pytest.approx(1000 * callback.state.model_flops_per_token, rel=1e-12)


@pytest.mark.parametrize("tp_size", (2, 4))
def test_measured_work_carries_the_tp_head_share(tp_size):
    callback, trainer = _bound(make_parallelism_config(tp_size=tp_size, world_size=tp_size, gpus_per_node=tp_size))
    lengths = torch.tensor([300, 900])
    trainer._accumulate_attention_flops(_batch(lengths))
    assert callback._drain_measured_attention_flops() == pytest.approx(
        float(LAYOUT.flops_for_documents(lengths)) / tp_size, rel=1e-12
    )


def test_measured_work_carries_the_cp_sequence_share():
    callback, trainer = _bound(make_parallelism_config(cp_size=2, world_size=2, gpus_per_node=2))
    lengths = torch.tensor([300, 900])
    trainer._accumulate_attention_flops(_batch(lengths))
    assert callback._drain_measured_attention_flops() == pytest.approx(
        float(LAYOUT.flops_for_documents(lengths)) / 2, rel=1e-12
    )


def test_measured_work_carries_the_config_depth_share_under_pp():
    """A tree exposing no layer list is costed from the config's full depth; each stage owns 1/pp of it."""
    callback, trainer = _bound(attention_share=0.25)
    lengths = torch.tensor([300, 900])
    trainer._accumulate_attention_flops(_batch(lengths))
    assert callback._drain_measured_attention_flops() == pytest.approx(
        float(LAYOUT.flops_for_documents(lengths)) * 0.25, rel=1e-12
    )


def test_the_drain_counts_documents_and_ignores_empty_slots():
    _, trainer = _bound()
    trainer._accumulate_attention_flops(_batch([10, 20, 30]))
    trainer._accumulate_attention_flops(
        {"input_ids": torch.zeros(1, 4, dtype=torch.long), "position_ids": torch.tensor([[0, 1, 2, 3]])}
    )
    assert trainer.drain_attention_flops()[1] == 4
    assert trainer.drain_attention_flops() == (0.0, 0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
