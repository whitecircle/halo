#!/usr/bin/env python
"""TRL's real loss over packed micro-batches (filler rows, leading no-op slots) sums to the legacy one-row-per-chunk
window for every admitted loss type, because row-mean losses are rescaled by ``rows_per_bin * steps_per_generation /
round_rows`` and token-global ones are not."""

import dataclasses
import re

import pytest
import torch
from trl import GRPOConfig

from halo.trainers.grpo.environmental.packed_rounds import LOSS_SCALE_KEY, SLOT_IS_ACTIVE_KEY
from halo.trainers.grpo.environmental.row_order import (
    MICROBATCH_INVARIANT_LOSS_TYPES,
    ROW_MEAN_LOSS_TYPES,
    TOKEN_GLOBAL_LOSS_TYPES,
)
from halo.trainers.grpo.microbatch_plan import agree_microbatch_geometry
from tests.common.length_deal import (
    packed_host,
    ragged_round,
    single_process_accelerator,
    tiny_llama,
    train_window,
)

_VOCAB = 64
_ROWS, _SLOTS, _BUDGET = 16, 4, 40
# TRL's loss types that read no micro-batch's composition; bnpo reads it and vespo is refused at construction.
_ADMITTED = ("grpo", "sapo", "dr_grpo", "dapo", "cispo", "luspo")


def _round(seed: int = 9) -> dict:
    gen = torch.Generator().manual_seed(seed)
    return ragged_round(
        torch.randint(1, 6, (_ROWS,), generator=gen).tolist(),
        torch.randint(1, 9, (_ROWS,), generator=gen).tolist(),
        vocab=_VOCAB,
        seed=seed,
    )


def _plan(batch: dict):
    lengths = (batch["prompt_mask"].sum(1) + batch["completion_mask"].sum(1)).clamp(min=2).tolist()
    plan = agree_microbatch_geometry(lengths, _BUDGET, slots=_SLOTS, ceiling=None, round_rows=_ROWS)
    assert plan.num_bins < _SLOTS and plan.rows_per_bin != _ROWS // _SLOTS, "the geometry must differ from the legacy"
    return plan


def _legacy(batch: dict, loss_type: str, iterations: int) -> tuple[float, list[torch.Tensor]]:
    host = packed_host(
        tiny_llama(_VOCAB, 0),
        slots=_SLOTS,
        iterations=iterations,
        loss_type=loss_type,
        accelerator=single_process_accelerator(),
    )
    host._packs_microbatches = False
    return train_window(host, batch)


def _packed(batch: dict, loss_type: str, iterations: int, *, scaled: bool = True):
    host = packed_host(
        tiny_llama(_VOCAB, 0),
        slots=_SLOTS,
        iterations=iterations,
        loss_type=loss_type,
        accelerator=single_process_accelerator(),
    )
    if not scaled:
        host._scale_packed_loss = lambda loss, inputs: loss
    return train_window(host, batch, _plan(batch))


def _close(got: tuple[float, list[torch.Tensor]], want: tuple[float, list[torch.Tensor]]) -> bool:
    return got[0] == pytest.approx(want[0], rel=1e-5, abs=1e-6) and all(
        torch.allclose(g, w, rtol=1e-4, atol=1e-6) for g, w in zip(got[1], want[1], strict=True)
    )


@pytest.mark.parametrize("iterations", [1, 2])
@pytest.mark.parametrize("loss_type", _ADMITTED)
def test_a_packed_window_sums_to_the_legacy_window(loss_type, iterations):
    batch = _round()
    legacy = _legacy(batch, loss_type, iterations)
    assert abs(legacy[0]) > 1e-4, "a vanishing loss compares nothing"
    assert _close(_packed(batch, loss_type, iterations), legacy)


@pytest.mark.parametrize("loss_type", _ADMITTED)
def test_without_the_rescale_only_the_token_global_losses_survive_packing(loss_type):
    """Negative control: the packed loss with the scale dropped, and with ``gradient_accumulation_steps`` read in
    place of ``steps_per_generation``, equals the legacy window only where the loss reads no row count."""
    iterations = 2
    batch = _round()
    legacy = _legacy(batch, loss_type, iterations)
    raw = _packed(batch, loss_type, iterations, scaled=False)
    rows, ga = _plan(batch).rows_per_bin, _SLOTS * iterations
    if loss_type in TOKEN_GLOBAL_LOSS_TYPES:
        assert _close(raw, legacy)
        return
    assert not _close(raw, legacy)
    exact = rows * _SLOTS / _ROWS
    assert _close((raw[0] * exact, [g * exact for g in raw[1]]), legacy)
    off = rows * ga / _ROWS
    assert not _close((raw[0] * off, [g * off for g in raw[1]]), legacy), "num_iterations would multiply the gradient"


def test_bnpos_loss_reads_which_rows_share_a_bin():
    """Why bnpo is refused: its legacy window changes with the rows' grouping, packed or not."""
    batch = _round()
    legacy = _legacy(batch, "bnpo", 1)
    packed = _packed(batch, "bnpo", 1, scaled=False)
    assert abs(packed[0] - legacy[0]) > 1e-3 * abs(legacy[0])


def test_the_loss_type_sets_partition_what_the_trainer_admits():
    assert not set(ROW_MEAN_LOSS_TYPES) & set(TOKEN_GLOBAL_LOSS_TYPES)
    assert set(MICROBATCH_INVARIANT_LOSS_TYPES) == set(ROW_MEAN_LOSS_TYPES) | set(TOKEN_GLOBAL_LOSS_TYPES)
    help_text = next(f for f in dataclasses.fields(GRPOConfig) if f.name == "loss_type").metadata["help"]
    declared = re.findall(r"'(\w+)'", help_text.split("Supported values are")[1].split(". ")[0])
    assert set(declared) - set(MICROBATCH_INVARIANT_LOSS_TYPES) == {"bnpo", "vespo"}, (
        "a loss type TRL added is neither admitted nor refused for packing: classify it as row-mean or token-global"
    )
    assert set(ROW_MEAN_LOSS_TYPES) | set(TOKEN_GLOBAL_LOSS_TYPES) == set(_ADMITTED)


def test_the_view_a_micro_step_trains_on_holds_each_row_once_and_inert_filler():
    batch = _round() | {"row_id": torch.arange(1, _ROWS + 1)}
    host = packed_host(tiny_llama(_VOCAB, 0), slots=_SLOTS, loss_type="grpo", accelerator=single_process_accelerator())
    plan = _plan(batch)
    host._generate_and_score_completions = lambda _inputs: host._lay_out_packed_round(dict(batch), plan)
    host._step, host._buffered_inputs = 0, None
    seen, idle = [], 0
    for _ in range(_SLOTS):
        view = host._prepare_inputs({})
        host._step += 1
        if view[SLOT_IS_ACTIVE_KEY] is False:
            idle += 1
            assert set(view) == {SLOT_IS_ACTIVE_KEY}
            continue
        assert view["advantages"].shape == (plan.rows_per_bin,)
        assert view[LOSS_SCALE_KEY] == pytest.approx(plan.rows_per_bin * _SLOTS / _ROWS)
        filler = view["row_id"] == 0
        seen += view["row_id"][~filler].tolist()
        assert not view["completion_mask"][filler].any() and not view["tool_mask"][filler].any()
        assert (view["prompt_mask"][filler].sum(1) == 1).all() and view["prompt_mask"][filler][:, -1].all()
        assert (view["importance_sampling_ratio"][filler] == 1).all() and not view["advantages"][filler].any()
        assert (view["completion_ids"][filler] == host.pad_token_id).all()
        for key in ("advantages", "completion_ids", "importance_sampling_ratio"):
            rows = view["row_id"][~filler] - 1
            assert torch.equal(view[key][~filler], batch[key][rows]), f"{key} left its row"
    assert idle == _SLOTS - plan.num_bins and sorted(seen) == list(range(1, _ROWS + 1))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
