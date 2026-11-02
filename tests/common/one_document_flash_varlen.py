"""The one-document varlen sweep: a tiny family's forward fed an unpacked one-row batch with the varlen set the
collator hands it, against the same batch without the set (so every attention layer tests the row for packing and
runs the dense kernel), both through transformers' own flash dispatch on the CPU reference kernels of
:mod:`tests.common.packed_flash_varlen`; a training step under gradient checkpointing, whose recompute must read the
handed set; and a host-read probe over a model at the depth of the profiled dense row.

Each entry point rebinds process-global state (the reference kernels, the family patches), so it runs in a fresh
interpreter per case and reports through one ``FINDINGS:`` line (:func:`tests.common.packed_flash_varlen.report`).
"""

import torch
from torch.utils.checkpoint import CheckpointError
from transformers import Qwen3Config, Qwen3ForCausalLM
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling as TRLDataCollatorForLanguageModeling

from halo.data.collators.collator_rules import supports_one_document_varlen_set
from halo.data.collators.factory import select_data_collator
from halo.models.gradient_checkpointing import enable_scoped_gradient_checkpointing
from tests.common.models import TINY_QWEN3_CONFIG
from tests.common.packed_flash_varlen import (
    GC_MODES,
    PACKED_ATTN_IMPLEMENTATIONS,
    PACKING_FAMILIES,
    PAD_ID,
    REACH_PREFIX,
    SEED,
    SET_DROPPING_FAMILIES,
    SKIP_PREFIX,
    SWEEP_ATTN_IMPLEMENTATION,
    VARLEN_KEYS,
    AttentionHostReads,
    install_reference_flash,
    padding_tokenizer,
    use_flash,
)
from tests.common.tiny_models import tiny_family_model

# One unpacked training row, as a tokenized dataset hands it to the collator.
ONE_DOCUMENT_ROW = {"input_ids": [5, 6, 7, 8, 9, 10, 11, 12, 13]}
# The decoder depth of the dense benchmark model (Qwen3-8B).
PROFILED_DEPTH = 36


def one_document_batch(config, attn_implementation: str) -> dict:
    """Resolve ``config`` to ``attn_implementation`` with its production family patches, and collate
    :data:`ONE_DOCUMENT_ROW` through the collator an SFT run builds for a full-sequence row (TRL's own where the
    factory leaves it to TRL)."""
    use_flash(config, attn_implementation)
    collator = select_data_collator(
        padding_tokenizer(), train_on_completions_only=False, model_config=config, emit_varlen_one_document=True
    ) or TRLDataCollatorForLanguageModeling(pad_token_id=PAD_ID)
    return collator([dict(ONE_DOCUMENT_ROW)])


def without_set(batch: dict) -> dict:
    """``batch`` without the varlen set, as a caller that does not opt in collates it."""
    return {key: value for key, value in batch.items() if key not in VARLEN_KEYS}


def wrong_boundary(batch: dict) -> dict:
    """``batch`` with a varlen set that splits the row in two: a forward that reads the set trains another loss."""
    width = batch["input_ids"].shape[1]
    cu_seq_lens = torch.tensor([0, width // 2, width], dtype=torch.int32)
    longest = width - width // 2
    return {
        **batch,
        "cu_seq_lens_q": cu_seq_lens,
        "cu_seq_lens_k": cu_seq_lens,
        "max_length_q": longest,
        "max_length_k": longest,
    }


def _flash_label(name: str) -> str:
    return PACKED_ATTN_IMPLEMENTATIONS.get(name, SWEEP_ATTN_IMPLEMENTATION)


def family_findings(name: str) -> list[str]:
    """What breaks when ``name``'s one-document row hands flash its varlen set, then a ``reach:`` entry counting the
    attention calls that read it; a lone ``skip:`` entry for a family the change does not reach."""
    if name in SET_DROPPING_FAMILIES:
        return [f"{SKIP_PREFIX} its model forward drops the varlen set, so every attention layer tests the row"]
    torch.manual_seed(SEED)
    model = tiny_family_model(PACKING_FAMILIES[name]).eval()
    if not getattr(model, "_supports_flash_attn", False):
        return [f"{SKIP_PREFIX} its model class declares no flash support, so its runs never resolve to flash"]
    config = model.config
    batch = one_document_batch(config, _flash_label(name))
    if not supports_one_document_varlen_set(config):
        if any(key in batch for key in VARLEN_KEYS):
            return ["a family whose one-document rows keep the dense kernel was handed the varlen set"]
        return [f"{SKIP_PREFIX} another of its mixers reads the varlen set, so its one-document rows carry none"]
    if not all(key in batch for key in VARLEN_KEYS):
        return ["the flash run's one-document batch carried no varlen set"]
    width = batch["input_ids"].shape[1]

    def run(inputs):
        kernels, counts = install_reference_flash(attn_implementation=_flash_label(name))
        with torch.no_grad():
            logits = model(**{k: v for k, v in inputs.items() if k != "labels"}, use_cache=False).logits
        return logits, kernels.calls, counts

    handed, handed_calls, handed_counts = run(batch)
    bare, bare_calls, bare_counts = run(without_set(batch))

    findings = []
    if not torch.equal(handed, bare):
        findings.append("the varlen set changed the logits")
    if handed_counts["_is_packed_sequence"] or handed_counts["_prepare_from_posids"] or handed_counts["_upad_input"]:
        findings.append(f"a layer still tested or unpadded the row: {handed_counts}")
    # Anti-vacuity: without the set, every dense call follows a packed-row test.
    dense = [call for call in bare_calls if call[0] == "dense"]
    if not dense or bare_counts["_is_packed_sequence"] != len(dense) or len(dense) != len(bare_calls):
        findings.append(f"the batch without the set did not test every layer: {bare_calls} / {bare_counts}")
    varlen = [call for call in handed_calls if call[0] == "varlen"]
    if len(varlen) != len(handed_calls) or len(varlen) != len(dense):
        findings.append(f"{len(varlen)} varlen calls of {len(handed_calls)} where {len(dense)} layers attended")
    for (_, cu_q, cu_k, max_q, max_k, window, softcap), (_, _, dense_window, dense_softcap) in zip(
        varlen, dense, strict=False
    ):
        if cu_q != [0, width] or cu_k != [0, width] or max_q != width or max_k != width:
            findings.append(f"a varlen call got {cu_q}, {cu_k}, {max_q}, {max_k} for a {width}-token row")
        if type(max_q) is not int or type(max_k) is not int:
            findings.append("a varlen kernel got a tensor max length")
        if (window, softcap) != (dense_window, dense_softcap):
            findings.append(
                f"window/softcap {window}/{softcap} where the dense call had {dense_window}/{dense_softcap}"
            )
    return [*findings, f"{REACH_PREFIX}{len(varlen)}"]


def step_findings(name: str, gc_mode: str) -> list[str]:
    """What breaks when a training step on ``name``'s one-document row, checkpointed per ``gc_mode``
    (:data:`~tests.common.packed_flash_varlen.GC_MODES`), hands flash its varlen set: the loss and every gradient
    must match the step without it bit for bit, no recompute may test the row again, and a set with a wrong boundary
    must train another loss, the same with checkpointing as without (the recompute reads the handed set)."""
    torch.manual_seed(SEED)
    model = tiny_family_model(PACKING_FAMILIES[name]).train()
    batch = one_document_batch(model.config, _flash_label(name))
    if not all(key in batch for key in VARLEN_KEYS):
        return ["the flash run's one-document batch carried no varlen set"]
    reentrant = GC_MODES[gc_mode]
    if reentrant is not None:
        enable_scoped_gradient_checkpointing(model, gradient_checkpointing_kwargs={"use_reentrant": reentrant})

    def step(inputs):
        model.zero_grad(set_to_none=True)
        _kernels, counts = install_reference_flash(attn_implementation=_flash_label(name))
        loss = model(**inputs, use_cache=False).loss
        forward = dict(counts)
        loss.backward()
        trained = {"loss": loss.detach()} | {
            param_name: param.grad.clone() for param_name, param in model.named_parameters() if param.grad is not None
        }
        return trained, forward, {key: counts[key] - forward[key] for key in counts}

    handed, handed_forward, handed_backward = step(batch)
    bare, bare_forward, bare_backward = step(without_set(batch))
    wrong_error = None
    try:
        wrong = step(wrong_boundary(batch))[0]
    except CheckpointError as error:
        wrong, wrong_error = None, str(error).partition("\n")[0]
    model.gradient_checkpointing_disable()
    wrong_plain = step(wrong_boundary(batch))[0]

    def same(a, b):
        return a is not None and a.keys() == b.keys() and all(torch.equal(a[key], b[key]) for key in a)

    findings = []
    if not same(handed, bare):
        findings.append("the varlen set changed the loss or a gradient")
    tests_forward, tests_backward = bare_forward["_is_packed_sequence"], bare_backward["_is_packed_sequence"]
    if not tests_forward or tests_backward != (tests_forward if reentrant is not None else 0):
        findings.append(
            f"without the set the backward tested the row in {tests_backward} of the {tests_forward} layers whose "
            f"forward did, with gradient checkpointing {gc_mode}"
        )
    if handed_forward["_is_packed_sequence"] or handed_backward["_is_packed_sequence"]:
        findings.append(f"a layer tested the row despite the set: {handed_forward} / {handed_backward}")
    if wrong_error is not None or not same(wrong, wrong_plain):
        findings.append(
            "the recompute attended over other boundaries than the set the forward was handed"
            + (f" ({wrong_error})" if wrong_error else "")
        )
    if same(wrong_plain, handed):
        findings.append("a wrong boundary trained the same step, so the set is not what attention reads")
    return findings


def host_read_findings() -> list[str]:
    """The packed-row tests and host reads one reentrant-checkpointed training step of a :data:`PROFILED_DEPTH`-layer
    Qwen3 makes on its one-document row, without the set and with it, as ``reads:<without>:<with>`` and
    ``tests:<without>:<with>`` entries, after any failure."""
    torch.manual_seed(SEED)
    config = Qwen3Config(**{**TINY_QWEN3_CONFIG, "num_hidden_layers": PROFILED_DEPTH})
    model = Qwen3ForCausalLM(config).train()
    batch = one_document_batch(model.config, SWEEP_ATTN_IMPLEMENTATION)
    enable_scoped_gradient_checkpointing(model, gradient_checkpointing_kwargs={"use_reentrant": True})

    def step(inputs) -> tuple[int, int]:
        reads = AttentionHostReads()
        reads.attending()
        _kernels, counts = install_reference_flash(reads)
        with reads:
            loss = model(**inputs, use_cache=False).loss
        loss.backward()
        return len(reads.reads), counts["_is_packed_sequence"]

    bare_reads, bare_tests = step(without_set(batch))
    handed_reads, handed_tests = step(batch)
    findings = []
    if bare_tests != 2 * PROFILED_DEPTH or bare_reads < PROFILED_DEPTH:
        findings.append(f"without the set: {bare_tests} packed-row tests and {bare_reads} forward host reads")
    if handed_tests or handed_reads:
        findings.append(f"with the set: {handed_tests} packed-row tests and {handed_reads} forward host reads")
    return [*findings, f"reads:{bare_reads}:{handed_reads}", f"tests:{bare_tests}:{handed_tests}"]
