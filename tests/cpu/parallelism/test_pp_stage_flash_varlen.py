#!/usr/bin/env python
"""A packed pipeline microbatch on a flash run reaches its stage's attention with the varlen set already built.

The schedule microbatches only batch-dim tensors, so the varlen set the packing collator builds for a flattened
row cannot ride in with a pipeline batch; left alone, every attention layer of every stage derives it from
``position_ids`` (a packed-row check, a ``nonzero`` and a max length, each a host sync on a device), and again in
the recompute. The stage forward builds it once per microbatch from one host read of the row, with the exact
longest segment as the host-int max length, wherever transformers would take that varlen path, and hands nothing
anywhere else.

Held here: the set equals transformers' derivation on every row a pipeline batch holds (a row its documents fill, a
chunked pad tail, a per-pad zero tail, a one-document row); building it reads the device row once; the stage hands
it to the backbone only on a flash run's unmasked one-row microbatch with ``position_ids``, of a family whose own
mixers do not read it; across the pipeline-registered tiny roster the stages read it in place of the derivation with
identical kernel arguments and bit-identical outputs (:mod:`tests.common.pp_packed_flash_varlen`, one fresh
interpreter per family); and the real schedule over gloo (pp2 x dp2, FSDP2 per stage) reproduces bit for bit the
step loss, gradients, eval loss and forward-only output it gives without the set, under both schedules, activation
checkpointing and the fused head loss, with no host read left inside attention.

    python tests/cpu/parallelism/test_pp_stage_flash_varlen.py
"""

import copy
import datetime

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh

from halo.distributed.parallelism_config import ParallelismConfig
from halo.distributed.pipeline_parallel.groups import build_pipeline_group
from halo.distributed.pipeline_parallel.stage import build_pipeline_stage
from halo.models.segment_markers import packed_row_varlen_kwargs
from tests.common.gloo import run_gloo_ranks
from tests.common.packed_flash_varlen import (
    FINDINGS_MARKER,
    HOST_READS,
    PACKING_FAMILIES,
    REACH_PREFIX,
    SEED,
    SWEEP_ATTN_IMPLEMENTATION,
    VARLEN_KEYS,
    AttentionHostReads,
    assert_transformers_derivation,
    set_attn_implementation,
)
from tests.common.pp_fakes import hand_built_stage
from tests.common.pp_packed_flash_varlen import (
    MAX_LENGTH,
    PP_ROW_DOCUMENTS,
    PP_SIZE,
    ScheduleCase,
    arm_differences,
    run_schedule_arm,
    seam_findings,
    seam_sweep_skip,
    seam_withheld,
    sweep_layouts,
)
from tests.common.repo import probe_findings
from tests.common.tiny_models import TINY_DENSE_FAMILIES, tiny_family_model

WORLD = 4
SCHEDULE_CASES = (
    ScheduleCase(),
    ScheduleCase(n_microbatches=4, gc=True, fused=True),
    ScheduleCase(schedule="gpipe", n_microbatches=4, fused=True),
    ScheduleCase(family="gpt_oss", gc=True),
)


def _device_row(position_ids: torch.Tensor) -> tuple[torch.Tensor, list[str]]:
    """``position_ids`` standing in for a row on a device: every tensor derived from it stays on the "device",
    ``cpu()`` hands the host a plain copy, and each host read of a device value is recorded."""
    reads: list[str] = []

    class DeviceRow(torch.Tensor):
        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            name = getattr(func, "__name__", None)
            if name in HOST_READS:
                reads.append(name)
            out = super().__torch_function__(func, types, args, kwargs)
            return out.as_subclass(torch.Tensor) if name == "cpu" else out

    return position_ids.as_subclass(DeviceRow), reads


def _rows(layout: str) -> list[torch.Tensor]:
    config = tiny_family_model(TINY_DENSE_FAMILIES["qwen3"]).config
    return [row[None] for row in sweep_layouts(config)[layout]["position_ids"]]


def _stage(family: str = "qwen3", attn_implementation: str | None = SWEEP_ATTN_IMPLEMENTATION):
    torch.manual_seed(SEED)
    model = tiny_family_model(TINY_DENSE_FAMILIES[family])
    set_attn_implementation(model.config, attn_implementation)
    return build_pipeline_stage(model, 0, PP_SIZE)


class _Handed(Exception):
    """Stops a stage forward at its backbone, carrying the kwargs it was handed."""


def _handed(stage, position_ids: torch.Tensor | None, attention_mask: torch.Tensor | None = None) -> dict:
    """The forward kwargs ``stage`` hands its backbone for a microbatch of ``position_ids`` (without them, one row
    of :data:`MAX_LENGTH`)."""

    def capture(_module, _args, kwargs):
        raise _Handed(kwargs)

    handle = stage.model.register_forward_pre_hook(capture, with_kwargs=True)
    shape = (1, MAX_LENGTH) if position_ids is None else position_ids.shape
    input_ids = torch.full(shape, 5, dtype=torch.long)
    try:
        stage(input_ids, attention_mask=attention_mask, position_ids=position_ids)
    except _Handed as handed:
        return handed.args[0]
    finally:
        handle.remove()
    raise AssertionError("the stage forward never reached its backbone")


@pytest.mark.parametrize("layout", ["ramp_tail", "zero_tail", "one_document"])
def test_the_set_is_what_transformers_derives(layout):
    for position_ids in _rows(layout):
        assert_transformers_derivation(packed_row_varlen_kwargs(position_ids), position_ids)


def test_the_rows_cover_the_cases():
    """Anti-vacuity: every packed row's longest segment is shorter than the row, the row its documents fill ends
    on its longest (no pad tail behind it), the zero tail makes each pad a segment of its own, and the
    one-document row does not restart."""
    for position_ids in _rows("ramp_tail"):
        assert packed_row_varlen_kwargs(position_ids)["max_length_q"] < MAX_LENGTH
    filled = next(i for i, documents in enumerate(PP_ROW_DOCUMENTS) if sum(documents) == MAX_LENGTH)
    lengths = packed_row_varlen_kwargs(_rows("ramp_tail")[filled])["cu_seq_lens_q"].diff().tolist()
    assert lengths == list(PP_ROW_DOCUMENTS[filled]) and lengths[-1] > max(lengths[:-1]), lengths
    lone = PP_ROW_DOCUMENTS.index((12,))
    zero_tail = _rows("zero_tail")[lone]
    assert len(packed_row_varlen_kwargs(zero_tail)["cu_seq_lens_q"]) == MAX_LENGTH - 12 + 2
    assert packed_row_varlen_kwargs(_rows("one_document")[0]) == {}


@pytest.mark.parametrize("layout", ["ramp_tail", "one_document"])
def test_building_the_set_reads_the_device_row_once(layout):
    for position_ids in _rows(layout):
        row, reads = _device_row(position_ids)
        packed_row_varlen_kwargs(row)
        assert reads == ["cpu"], reads


def test_the_stage_hands_a_packed_flash_microbatch_its_set():
    stage = _stage()
    for position_ids in _rows("ramp_tail"):
        row, reads = _device_row(position_ids)
        handed = _handed(stage, row)
        assert_transformers_derivation({key: handed[key] for key in VARLEN_KEYS}, position_ids)
        assert reads == ["cpu"], reads


@pytest.mark.parametrize("attn_implementation", ["sdpa", "eager", "flex_attention", None])
def test_a_non_flash_stage_hands_nothing_and_reads_nothing(attn_implementation):
    row, reads = _device_row(_rows("ramp_tail")[0])
    assert not VARLEN_KEYS & _handed(_stage(attn_implementation=attn_implementation), row).keys()
    assert reads == []


def test_a_masked_microbatch_is_left_to_its_mask():
    position_ids = _rows("ramp_tail")[0]
    row, reads = _device_row(position_ids)
    handed = _handed(_stage(), row, attention_mask=torch.ones_like(position_ids))
    assert not VARLEN_KEYS & handed.keys()
    assert reads == []


def test_a_multi_row_microbatch_is_left_to_transformers():
    """transformers runs a microbatch of more rows on the dense kernel; a set spanning them would merge the rows."""
    row, reads = _device_row(torch.cat(_rows("ramp_tail")[:2]))
    assert not VARLEN_KEYS & _handed(_stage(), row).keys()
    assert reads == []


def test_a_microbatch_without_position_ids_is_left_to_transformers():
    """The backbone then numbers the row from 0 itself, which transformers runs on the dense kernel."""
    assert not VARLEN_KEYS & _handed(_stage(), None).keys()


def test_a_one_document_microbatch_keeps_the_dense_kernel():
    row, reads = _device_row(_rows("one_document")[0])
    assert not VARLEN_KEYS & _handed(_stage(), row).keys()
    assert reads == ["cpu"]


def test_a_gated_deltanet_stage_keeps_its_own_markers():
    """Its chunked delta rule reads ``cu_seq_lens_q`` too, so a handed set would change what its mixers compute."""
    row, reads = _device_row(_rows("ramp_tail")[0])
    assert not VARLEN_KEYS & _handed(_stage("qwen3_5"), row).keys()
    assert reads == []


def test_a_stage_without_a_config_hands_nothing():
    model = tiny_family_model(TINY_DENSE_FAMILIES["qwen3"])
    set_attn_implementation(model.config, SWEEP_ATTN_IMPLEMENTATION)
    stage = hand_built_stage(copy.deepcopy(model.model), model.lm_head, is_first=True, is_last=True, layer_offset=0)
    row, reads = _device_row(_rows("ramp_tail")[0])
    assert not VARLEN_KEYS & _handed(stage, row).keys()
    assert reads == []


def test_withholding_the_seam_restores_the_derivation():
    """Anti-vacuity for the sweep's and the schedule's baseline arm: without the seam nothing is handed over."""
    with seam_withheld():
        assert not VARLEN_KEYS & _handed(_stage(), _rows("ramp_tail")[0]).keys()


@pytest.mark.parametrize("family", sorted(PACKING_FAMILIES))
def test_every_pp_packing_family_reads_the_set_in_place_of_the_derivation(family):
    reason = seam_sweep_skip(family)
    if reason is not None:
        pytest.skip(reason)
    report = probe_findings(f"from tests.common.pp_packed_flash_varlen import main; main({family!r})", FINDINGS_MARKER)
    failures = [entry for entry in report if not entry.startswith(REACH_PREFIX)]
    assert not failures, "\n".join(failures)
    reach = int(next(entry for entry in report if entry.startswith(REACH_PREFIX)).removeprefix(REACH_PREFIX))
    assert reach > 0, f"no attention layer of {family}'s stages read the handed-over set"


def test_the_sweep_holds_the_flash_families_pipeline_runs():
    """Anti-vacuity: the dense and the MoE families the pipeline recipes train reach the seam, so their sweep entries
    cannot be skips."""
    assert seam_sweep_skip("dense:qwen3") is None
    assert seam_sweep_skip("gpt_oss") is None


def _schedule_worker(rank: int, case: ScheduleCase) -> None:
    torch.set_num_threads(2)
    # Bit equality needs a reference that repeats: GptOss's expert-bias gradient accumulates in a thread-dependent
    # order otherwise, a few ULPs apart from run to run with or without the seam.
    torch.use_deterministic_algorithms(True)
    config = ParallelismConfig(
        world_size=WORLD, gpus_per_node=2, nvlink_domain_size=2, pp_size=PP_SIZE, pp_schedule=case.schedule
    )
    mesh = init_device_mesh("cpu", (config.pp_size, config.stage_world_size), mesh_dim_names=("pp", "dp"))
    groups = (mesh["dp"], build_pipeline_group(config))
    reads = AttentionHostReads()
    reads.attending()
    device = torch.device("cpu")
    with seam_withheld():
        derived = run_schedule_arm(case, config, groups, device, reads)
    seam = run_schedule_arm(case, config, groups, device, reads)
    findings = [f"{name} differs by {gap:.3e}" for name, gap in sorted(arm_differences(seam, derived).items())]
    findings += seam_findings(seam, derived)
    if case.gc and not seam.checkpointed:
        findings.append("anti-vacuity: activation checkpointing never ran")
    assert not findings, f"rank {rank}: " + "; ".join(findings)


@pytest.mark.parametrize("case", SCHEDULE_CASES, ids=lambda case: repr(case).removeprefix("ScheduleCase"))
def test_the_schedule_reproduces_the_derivation_bit_for_bit(case):
    run_gloo_ranks(_schedule_worker, WORLD, case, pg_timeout=datetime.timedelta(seconds=300))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
