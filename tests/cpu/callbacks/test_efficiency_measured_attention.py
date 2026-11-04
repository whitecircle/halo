"""Measured attention work must reach the MFU accounting exactly: document boundaries, the hand-off, the divisors it
carries (TP heads, CP sequence share, PP config-depth share) and the fallback when no batch exposed a document."""

from types import SimpleNamespace

import pytest
import torch

from halo.callbacks import efficiency
from halo.callbacks.efficiency import EfficiencyCallback
from halo.callbacks.model_flops import RankFlopsModel
from halo.models.attention.layout import AttentionLayout, LayerAttention
from halo.models.attention.tree_plan import TREE_ATTENTION_KEY, build_node_table, build_tree_attention_plan
from halo.trainers.token_metrics import TokenMetricsMixin
from tests.common.parallelism import make_parallelism_config

_NO_PARALLELISM = make_parallelism_config(world_size=1, gpus_per_node=1)

LINEAR_FLOPS_PER_TOKEN = 1.0e9
MAX_SEQ_LEN = 4096
PEAK_FLOPS = 1.0e15
# Short enough that the attention term moves the step's TFLOP/s by many of its reported hundredths.
STEP_SECONDS = 1.0e-3

LAYOUT = AttentionLayout((LayerAttention("full", 1024.0), LayerAttention("window", 1024.0, keys=128)))


class _Handler:
    def __init__(self, callbacks):
        self.callbacks = callbacks


class _Trainer(TokenMetricsMixin):
    def __init__(self, callbacks):
        self.callback_handler = _Handler(callbacks)
        self.parallelism_config = _NO_PARALLELISM
        self.args = SimpleNamespace(include_num_input_tokens_seen="no")


def _batch(lengths) -> dict:
    lengths = torch.as_tensor(lengths)
    return {"cu_seq_lens_q": torch.cat((torch.zeros(1, dtype=torch.long), lengths.cumsum(0)))}


def _documents(lengths) -> list[int]:
    return sorted(int(length) for length in lengths.tolist() if length)


def _callback(parallelism=_NO_PARALLELISM, layout=LAYOUT, attention_share=1.0):
    callback = EfficiencyCallback(parallelism, n_warmup_steps=0)
    fallback = layout.flops_per_token(MAX_SEQ_LEN) * attention_share / max(parallelism.tp_size, 1) if layout else 0.0
    callback.flops_model = RankFlopsModel(
        model_flops_per_token=LINEAR_FLOPS_PER_TOKEN + fallback,
        active_model_flops_per_token=0.5 * LINEAR_FLOPS_PER_TOKEN + fallback,
        attention_flops_per_token=fallback,
        attention_layout=layout,
        attention_share=attention_share,
        local_params=1.0,
        full_params=1.0,
        params_ratio=1.0,
        num_experts=0,
        top_k=0,
        sparsity_factor=1.0,
        local_expert_params=0.0,
        local_active_params=1.0,
        full_active_params=1.0,
    )
    callback.gpu_peak_flops = PEAK_FLOPS
    callback.training_args = SimpleNamespace(per_device_train_batch_size=1, gradient_accumulation_steps=1)
    return callback


def _step_tflops(monkeypatch, callback, tokens: int) -> tuple[float, float]:
    clock = iter((0.0, STEP_SECONDS))
    monkeypatch.setattr(efficiency, "time", SimpleNamespace(perf_counter=lambda: next(clock)))
    callback.on_step_begin(None, SimpleNamespace(global_step=0, num_input_tokens_seen=0), None)
    callback.on_step_end(None, SimpleNamespace(global_step=1, num_input_tokens_seen=tokens), None)
    callback.on_log(None, SimpleNamespace(global_step=1), None, logs={})
    return callback.mfu.step_tflops_per_sec, callback.smfu.step_smfu_tflops_per_sec


def _tflops(step_flops: float) -> float:
    return step_flops / STEP_SECONDS / 1e12


def _bound(parallelism=_NO_PARALLELISM, **kwargs):
    callback = _callback(parallelism, **kwargs)
    trainer = _Trainer([callback])
    return callback, trainer


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


def test_the_first_accumulation_binds_the_callback_and_takes_its_layout():
    callback, trainer = _bound()
    trainer._accumulate_attention_flops(_batch([100, 200]))
    assert trainer._attention_layout is LAYOUT
    assert callback._attention_source() is trainer


def test_a_callback_without_a_layout_keeps_the_trainer_accumulator_off():
    callback, trainer = _bound(layout=None)
    trainer._accumulate_attention_flops(_batch([100, 200]))
    assert trainer._attention_layout is None
    assert trainer.drain_step_work() == (0.0, 0)
    assert callback._drain_measured_attention_flops() is None


def test_measured_work_replaces_the_config_term_in_both_estimates(monkeypatch):
    callback, trainer = _bound()
    lengths = torch.tensor([500] * 7 + [250])
    trainer._accumulate_attention_flops(_batch(lengths[:5]))
    trainer._accumulate_attention_flops(_batch(lengths[5:]))
    measured = float(LAYOUT.flops_for_documents(lengths))
    tokens = int(lengths.sum())

    dense, active = _step_tflops(monkeypatch, callback, tokens)
    assert dense == pytest.approx(_tflops(tokens * LINEAR_FLOPS_PER_TOKEN + measured), abs=0.01)
    assert active == pytest.approx(_tflops(tokens * 0.5 * LINEAR_FLOPS_PER_TOKEN + measured), abs=0.01)
    # The accumulator was reset by the step's drain.
    assert trainer.drain_step_work() == (0.0, 0)


def test_the_measurement_differs_from_the_config_term_on_short_documents():
    """At these lengths the config term over-reports, so a fallback would be visible."""
    lengths = torch.tensor([500] * 7 + [250])
    tokens = int(lengths.sum())
    assert float(LAYOUT.flops_for_documents(lengths)) < 0.25 * tokens * LAYOUT.flops_per_token(MAX_SEQ_LEN)


def test_without_documents_the_step_keeps_the_config_term(monkeypatch):
    callback, trainer = _bound()
    trainer._accumulate_attention_flops({"chosen_input_ids": torch.zeros(1, 4)})
    dense, _ = _step_tflops(monkeypatch, callback, 1000)
    assert dense == pytest.approx(_tflops(1000 * callback.flops_model.model_flops_per_token), abs=0.01)


def test_an_unbound_callback_keeps_the_config_term(monkeypatch):
    callback = _callback()
    dense, _ = _step_tflops(monkeypatch, callback, 1000)
    assert dense == pytest.approx(_tflops(1000 * callback.flops_model.model_flops_per_token), abs=0.01)


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


def test_a_tree_batch_costs_each_node_its_own_tokens_at_its_path_length():
    """Three examples over one 600-token prefix: the prefix is attended once, each branch attends its
    whole path. Read by position resets alone, the row would be one 810-token document."""
    branches = (50, 70, 90)
    position_ids = torch.cat([torch.arange(600), *(torch.arange(600, 600 + n) for n in branches)])
    depth = torch.tensor([0] * 600 + [1] * sum(branches))
    plan = build_tree_attention_plan(build_node_table(position_ids, depth), sliding_window=128, full_form="split")
    row = {"input_ids": torch.zeros(1, len(position_ids), dtype=torch.long), "position_ids": position_ids[None]}

    callback, trainer = _bound()
    trainer._accumulate_attention_flops({**row, TREE_ATTENTION_KEY: plan})
    paths, own = torch.tensor([600, *(600 + n for n in branches)]), torch.tensor([600, *branches])
    expected = float(LAYOUT.flops_for_documents(paths, own))
    assert callback._drain_measured_attention_flops() == pytest.approx(expected, rel=1e-12)

    flat = float(LAYOUT.flops_for_documents(torch.tensor([600 + n for n in branches])))
    one_document = float(LAYOUT.flops_for_documents(torch.tensor([len(position_ids)])))
    assert expected < flat and expected != pytest.approx(one_document, rel=1e-3)
    trainer._accumulate_attention_flops(row)
    assert callback._drain_measured_attention_flops() == pytest.approx(one_document, rel=1e-12)


def test_the_drain_counts_documents_and_ignores_empty_slots():
    _, trainer = _bound()
    trainer._accumulate_attention_flops(_batch([10, 20, 30]))
    trainer._accumulate_attention_flops(
        {"input_ids": torch.zeros(1, 4, dtype=torch.long), "position_ids": torch.tensor([[0, 1, 2, 3]])}
    )
    assert trainer.drain_step_work()[1] == 4
    assert trainer.drain_step_work() == (0.0, 0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
