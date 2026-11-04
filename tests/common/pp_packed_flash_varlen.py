"""Packed pipeline microbatches on transformers' flash dispatch: stages handed the varlen set vs left to derive it vs
the unsplit model. The sweep rebinds process-global state, so it runs in a fresh interpreter per family (:func:`main`)
and reports through one ``FINDINGS:`` line."""

import contextlib
import copy
import functools
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from unittest import mock

import torch

from halo.data.collators.factory import select_data_collator
from halo.data.collators.fixed_shape import FixedShapeCollator
from halo.distributed.parallelism_config import ParallelismConfig
from halo.distributed.pipeline_parallel import stage as stage_module
from halo.distributed.pipeline_parallel.families import resolve_pp_spec
from halo.distributed.pipeline_parallel.model_rules import reject_unsupported_pp_structure
from halo.distributed.pipeline_parallel.runner import resolve_batch_pad_values
from halo.distributed.pipeline_parallel.stage import build_pipeline_stage
from halo.distributed.pipeline_parallel.stage_losses import FusedHeadLoss, TokenLoss
from halo.models.module_tree import backbone_with_layers
from halo.models.segment_markers import SegmentMarkers, packed_row_segment_kwargs, segment_markers_for
from tests.common.packed_flash_varlen import (
    PACKING_FAMILIES,
    PAD_ID,
    REACH_PREFIX,
    SEED,
    SWEEP_ATTN_IMPLEMENTATION,
    AttentionHostReads,
    install_loaded_flash,
    install_reference_flash,
    padding_tokenizer,
    report,
    use_flash,
)
from tests.common.pp_equivalence import build_runner_stage, gather_stage_grads
from tests.common.tiny_models import TINY_PP_SPLIT_OVERRIDES, tiny_family_model

PP_SIZE = 2
# Past the tiny GptOss sliding window, so its windowed layers take the window.
MAX_LENGTH = 96
# Document lengths, one row per microbatch: a filled row second so every replica's first two microbatches hold it, the
# rest leaving pad tails; a one-document row's tail outgrows its document, and two documents outgrow the window.
PP_ROW_DOCUMENTS = ((9, 14), (20, MAX_LENGTH - 20), (3, 20, 8), (12,), (70, 17))
# No restart, so transformers runs it on the dense kernel.
FULL_ROW_DOCUMENTS = ((MAX_LENGTH,),)
VOCAB_FLOOR = 4
VOCAB_CEIL = 1000


@functools.cache
def seam_sweep_skip(name: str) -> str | None:
    """Why ``name`` has no seam to sweep, or None: PP does not register its backbone or refuses its class, its mixers
    read the markers, or it never resolves to flash."""
    model = tiny_family_model(PACKING_FAMILIES[name])
    spec = resolve_pp_spec(backbone_with_layers(model))
    if not spec.validated or spec.refusal is not None:
        return "pipeline parallelism does not register its backbone"
    try:
        reject_unsupported_pp_structure(model)
    except ValueError as refused:
        return str(refused)
    if segment_markers_for(model.config).cu_seq_lens:
        return "its GatedDeltaNet mixers read the markers the seam hands them, so no arm runs without it"
    if not getattr(model, "_supports_flash_attn", False):
        return "its model class declares no flash support, so its runs never resolve to flash"
    return None


def packed_rows(documents: tuple[tuple[int, ...], ...], seed: int = SEED) -> list[dict]:
    """One packed row per entry of ``documents``, as TRL's ``pack_dataset`` emits it."""
    generator = torch.Generator().manual_seed(seed)
    return [
        {
            "input_ids": torch.randint(VOCAB_FLOOR, VOCAB_CEIL, (sum(lengths),), generator=generator).tolist(),
            "seq_lengths": list(lengths),
        }
        for lengths in documents
    ]


def pp_packed_batch(model_config, rows: list[dict]) -> dict[str, torch.Tensor]:
    """``rows`` through the collator a pipeline trains on, padded to :data:`MAX_LENGTH` by the fixed-shape wrapper."""
    collator = select_data_collator(
        padding_tokenizer(),
        train_on_completions_only=False,
        packing=True,
        model_config=model_config,
        per_device_train_batch_size=len(rows),
        keeps_packed_rows=True,
    )
    collator.flatten_to_single_row = False
    fixed = FixedShapeCollator(collator, MAX_LENGTH, resolve_batch_pad_values(PAD_ID))
    return fixed([dict(row) for row in rows])


def zero_pad_tails(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Each pad tail's positions set to 0, the constant fill making every pad its own segment."""
    return {**batch, "position_ids": batch["position_ids"].masked_fill(batch["input_ids"] == PAD_ID, 0)}


def sweep_layouts(model_config) -> dict[str, dict[str, torch.Tensor]]:
    ramp = pp_packed_batch(model_config, packed_rows(PP_ROW_DOCUMENTS))
    return {
        "ramp_tail": ramp,
        "zero_tail": zero_pad_tails(ramp),
        "one_document": pp_packed_batch(model_config, packed_rows(FULL_ROW_DOCUMENTS)),
    }


def _document_markers(position_ids: torch.Tensor, markers: SegmentMarkers) -> dict:
    """What a mixer outside attention reads to keep documents apart, which no arm runs without."""
    return packed_row_segment_kwargs(position_ids, SegmentMarkers(seq_idx=True)) if markers.seq_idx else {}


@contextlib.contextmanager
def seam_withheld() -> Iterator[None]:
    """The stage forward with no varlen set handed to the backbone."""
    with mock.patch.object(stage_module, "packed_row_segment_kwargs", _document_markers):
        yield


@dataclass
class Forward:
    """One no-grad forward on the reference kernels: output, kernel calls, derivation counts and host reads inside
    attention."""

    logits: torch.Tensor
    calls: list[tuple]
    counts: dict[str, int]
    reads: list[str]


def _forward(reads: AttentionHostReads, run: Callable[..., torch.Tensor], *args) -> Forward:
    kernels, counts = install_reference_flash(reads)
    reads.reads.clear()
    with torch.no_grad(), reads:
        logits = run(*args)
    return Forward(logits, kernels.calls, dict(counts), list(reads.reads))


def _chain(stages: list, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    hidden = input_ids
    for stage in stages:
        hidden = stage(hidden, position_ids=position_ids)
    return hidden


def _unsplit(model, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    markers = _document_markers(position_ids, segment_markers_for(model.config))
    return model(input_ids=input_ids, position_ids=position_ids, use_cache=False, **markers).logits


def _compare(where: str, seam: Forward, derived: Forward, unsplit: Forward, width: int) -> list[str]:
    findings = []
    if not torch.equal(seam.logits, derived.logits):
        findings.append(f"{where}: the seam changed the stages' output")
    if not torch.equal(derived.logits, unsplit.logits):
        findings.append(f"{where}: the stages' output differs from the unsplit model's")
    if not seam.calls == derived.calls == unsplit.calls:
        findings.append(f"{where}: the kernels saw different arguments: {seam.calls} vs {derived.calls}")
    varlen = [call for call in seam.calls if call[0] == "varlen"]
    if not varlen:
        if seam.counts != derived.counts:
            findings.append(f"{where}: a row on the dense kernel was handed a set ({seam.counts} vs {derived.counts})")
        return findings
    if seam.counts["_is_packed_sequence"] or seam.counts["_prepare_from_posids"]:
        findings.append(f"{where}: a layer still derived the set from position_ids: {seam.counts}")
    if seam.reads:
        findings.append(f"{where}: attention still read back to the host: {seam.reads}")
    if not derived.reads:
        findings.append(f"{where}: anti-vacuity: the derivation read nothing back, so no read was removed")
    for _, cu_seqlens, _, max_q, max_k, _window, _softcap in varlen:
        longest = max(b - a for a, b in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True))
        if not (type(max_q) is type(max_k) is int and max_q == max_k == longest):
            findings.append(f"{where}: max lengths {max_q!r}/{max_k!r}, the longest segment is {longest}")
        if longest == width:
            findings.append(
                f"{where}: anti-vacuity: the longest segment spans the row, so its width passes as the max"
            )
    return findings


def family_findings(name: str) -> list[str]:
    """What breaks when ``name``'s stages get their varlen set, plus a ``reach:`` entry. For a family
    :func:`seam_sweep_skip` lets through."""
    torch.manual_seed(SEED)
    model = tiny_family_model(PACKING_FAMILIES[name], overrides=TINY_PP_SPLIT_OVERRIDES.get(name)).eval()
    config = model.config
    use_flash(config)
    stages = [build_pipeline_stage(copy.deepcopy(model), rank, PP_SIZE) for rank in range(PP_SIZE)]
    reads = AttentionHostReads()
    reads.attending()

    findings, reach = [], 0
    for layout, batch in sweep_layouts(config).items():
        for row, (input_ids, position_ids) in enumerate(zip(batch["input_ids"], batch["position_ids"], strict=True)):
            microbatch = (input_ids[None], position_ids[None])
            seam = _forward(reads, _chain, stages, *microbatch)
            with seam_withheld():
                derived = _forward(reads, _chain, stages, *microbatch)
            unsplit = _forward(reads, _unsplit, model, *microbatch)
            findings += _compare(f"{layout}[{row}]", seam, derived, unsplit, MAX_LENGTH)
            reach += sum(call[0] == "varlen" for call in seam.calls)
    return [*findings, f"{REACH_PREFIX}{reach}"]


def main(name: str) -> None:
    report(family_findings(name))


@dataclass(frozen=True)
class ScheduleCase:
    """One schedule drive of a packed batch: family, schedule, microbatch count, GC and the fused last-stage loss."""

    family: str = "dense:qwen3"
    schedule: str = "1f1b"
    n_microbatches: int = 2
    gc: bool = False
    fused: bool = False


@dataclass
class ScheduleArm:
    """A drive's step loss, step-1 gradients, eval loss and forward-only output, with derivation counts and host
    reads."""

    loss: torch.Tensor | None
    grads: dict[str, torch.Tensor]
    eval_loss: torch.Tensor | None
    outputs: torch.Tensor | None
    counts: dict[str, int]
    reads: list[str]
    checkpointed: int


def dp_rows(case: ScheduleCase, dp_rank: int) -> list[dict]:
    """This DP replica's packed rows, one per microbatch, distinct across replicas."""
    documents = [PP_ROW_DOCUMENTS[(dp_rank + i) % len(PP_ROW_DOCUMENTS)] for i in range(case.n_microbatches)]
    return packed_rows(tuple(documents), seed=SEED + dp_rank)


def run_schedule_arm(
    case: ScheduleCase,
    config: ParallelismConfig,
    groups,
    device: torch.device,
    reads: AttentionHostReads,
    *,
    attn_implementation: str = SWEEP_ATTN_IMPLEMENTATION,
    dtype: torch.dtype = torch.float32,
    reference_kernels: bool = True,
    step_scope: Callable[[], contextlib.AbstractContextManager] = contextlib.nullcontext,
) -> ScheduleArm:
    """COLLECTIVE over the chain and stage group: one training step, one eval-loss drive and one forward-only drive
    through the real runner."""
    torch.manual_seed(SEED)
    model = tiny_family_model(PACKING_FAMILIES[case.family])
    use_flash(model.config, attn_implementation)
    model = model.to(device=device, dtype=dtype)
    loss = FusedHeadLoss() if case.fused else TokenLoss()
    stage, runner, checkpointed = build_runner_stage(
        model, config, groups, device, case.n_microbatches, loss, gc=case.gc
    )
    batch = pp_packed_batch(model.config, dp_rows(case, config.get_data_parallel_rank()))
    input_ids, labels, position_ids = (batch[key].to(device) for key in ("input_ids", "labels", "position_ids"))

    counts = (
        install_reference_flash(reads)[1] if reference_kernels else install_loaded_flash(reads, attn_implementation)
    )
    reads.reads.clear()
    with step_scope(), reads:
        step_loss = runner.step(input_ids, labels, position_ids=position_ids)
    grads = gather_stage_grads(stage, to_cpu=True)
    with reads:
        eval_loss = runner.eval_loss(input_ids, labels, position_ids=position_ids)
        outputs = runner.forward_only(input_ids, position_ids=position_ids)
    return ScheduleArm(
        loss=None if step_loss is None else step_loss.detach().cpu(),
        grads=grads,
        eval_loss=None if eval_loss is None else eval_loss.detach().cpu(),
        outputs=None if outputs is None else outputs.detach().cpu(),
        counts=dict(counts),
        reads=list(reads.reads),
        checkpointed=checkpointed[0],
    )


def _arm_tensors(arm: ScheduleArm) -> dict[str, torch.Tensor | None]:
    return {"loss": arm.loss, "eval_loss": arm.eval_loss, "outputs": arm.outputs, **arm.grads}


def arm_differences(a: ScheduleArm, b: ScheduleArm) -> dict[str, float]:
    """Each tensor two drives disagree on in any bit, with its largest difference; a tensor only one holds is
    infinitely far."""
    ours, theirs = _arm_tensors(a), _arm_tensors(b)
    differences = {}
    for name in ours.keys() | theirs.keys():
        x, y = ours.get(name), theirs.get(name)
        if x is None and y is None:
            continue
        if x is None or y is None or x.shape != y.shape:
            differences[name] = float("inf")
        elif not torch.equal(x, y):
            differences[name] = float((x.float() - y.float()).abs().max())
    return differences


def seam_findings(seam: ScheduleArm, derived: ScheduleArm) -> list[str]:
    """What the seam left derived or read back inside attention, and whether the drive without it derived anything."""
    findings = []
    if not seam.grads:
        findings.append("anti-vacuity: the drive produced no gradient")
    if seam.counts["_is_packed_sequence"] or seam.counts["_prepare_from_posids"]:
        findings.append(f"a layer still derived the set from position_ids: {seam.counts}")
    if seam.reads:
        findings.append(f"attention still read back to the host: {sorted(set(seam.reads))} x{len(seam.reads)}")
    if not derived.counts["_prepare_from_posids"] or not derived.reads:
        findings.append(f"anti-vacuity: the drive without the seam derived nothing ({derived.counts})")
    return findings
