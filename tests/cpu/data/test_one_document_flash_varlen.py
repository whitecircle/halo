#!/usr/bin/env python
"""On a flash run an SFT batch that is one document hands attention its varlen set, as packed rows do.

Without ``cu_seq_lens_q/k`` and ``max_length_q/k`` in the forward kwargs, transformers' flash path tests every
attention layer's ``position_ids`` for a packed row (``_is_packed_sequence``, a device sync, again in each
checkpoint recompute) before running the dense kernel on an unpacked batch-1 row. The SFT collators hand that row
the set ``[0, S]``, ``S`` that transformers derives from its positions, so the kernel runs varlen over one segment
and no layer tests the row. The set follows the resolved backend and the model: FA2/FA3/FA4, not where another
mixer reads the set (GatedDeltaNet) or FA4's varlen kernels refuse the attention (Gemma 2's softcap), never under CP
or PP, and only for a caller that opts in (the SFT script, whose batches reach one model). Several rows, a padded
row and a multi-document packed row keep their own path. The set is all a route adds to its batch, TRL's own
collator included, so the logits, the loss and every gradient stay bit-identical on the CPU reference kernels
across the tiny roster of packing families (:mod:`tests.common.one_document_flash_varlen`, one fresh interpreter
each), and a set with a wrong boundary trains another loss.

    python tests/cpu/data/test_one_document_flash_varlen.py
"""

import contextlib
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset
from transformers import Gemma2Config, Qwen3_5TextConfig, Qwen3Config, Qwen3ForCausalLM, Qwen3Model
from transformers import modeling_flash_attention_utils as flash_utils
from trl import SFTConfig
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling as TRLDataCollatorForLanguageModeling

from halo.data.collators.factory import select_data_collator
from halo.data.collators.packing import DataCollatorWithPacking
from halo.data.collators.padded import (
    DataCollatorForCausalLMWithPadding,
    DataCollatorForCompletionOnlyLM,
    DataCollatorForLanguageModelingWithVarlen,
)
from halo.models.attention.selection import VARLEN_ATTN_IMPLEMENTATIONS
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.models import TINY_QWEN3_CONFIG, TINY_QWEN35_CONFIG
from tests.common.offline_grpo import make_offline_tokenizer
from tests.common.one_document_flash_varlen import PROFILED_DEPTH, without_set
from tests.common.packed_flash_varlen import (
    EOS_ID,
    FINDINGS_MARKER,
    GC_MODES,
    PACKING_FAMILIES,
    PAD_ID,
    REACH_PREFIX,
    SKIP_PREFIX,
    VARLEN_KEYS,
    padding_tokenizer,
)
from tests.common.parallelism import make_parallelism_config
from tests.common.repo import probe_findings
from tests.common.sft_collator import build_sft_collator

RESPONSE_TEMPLATE = [30]
# One unpacked row whose completion opens on the response template and closes on EOS.
COMPLETION_ROW = {"input_ids": [5, 6, RESPONSE_TEMPLATE[0], 8, 9, 10, EOS_ID]}
# Two rows of different lengths: the batch pads, and transformers unpads it by its mask.
TWO_ROWS = ({"input_ids": [5, 6, 7, 8]}, {"input_ids": [9, 10]})
# Packed batches of several documents, as (completion masking, rows): one row of three, two one-document rows the
# collator flattens into one row, and one row of two completions.
MULTI_DOCUMENT_BATCHES = {
    "one_row": (False, [{"input_ids": [5, 6, 7, 8, 9, 10, 11, 12, 13], "seq_lengths": [3, 4, 2]}]),
    "flattened_rows": (
        False,
        [{"input_ids": [5, 6, 7, 8], "seq_lengths": [4]}, {"input_ids": [9, 10, 11], "seq_lengths": [3]}],
    ),
    "completions": (True, [{"input_ids": COMPLETION_ROW["input_ids"] * 2, "seq_lengths": [7, 7]}]),
}
# The attention variants a checkpointed one-document step trains: dense, sliding window, softcap (FA2), gpt-oss
# sinks, a routed MoE, Mistral 4's MLA.
STEP_FAMILIES = ("dense:qwen3", "dense:mistral", "dense:gemma2", "gpt_oss", "qwen3_moe", "mistral4")
# The SFT collator routes a one-document batch takes, as (completion masking, packing).
ROUTES = [(False, False), (True, False), (False, True), (True, True)]
ROUTE_IDS = ("full_sequence", "completion_only", "packing", "completion_only_packing")


def _config(attn_implementation: str | None, config_cls=Qwen3Config, **fields):
    config = config_cls(**{**TINY_QWEN3_CONFIG, **fields})
    config._attn_implementation = attn_implementation
    return config


def _collator(config, *, completion_only: bool = False, packing: bool = False, **kwargs):
    return select_data_collator(
        padding_tokenizer(),
        train_on_completions_only=completion_only,
        assistant_message_template=RESPONSE_TEMPLATE if completion_only else None,
        packing=packing,
        model_config=config,
        **{"one_document_varlen": True, **kwargs},
    )


def _row(completion_only: bool = False, packing: bool = False) -> dict:
    row = dict(COMPLETION_ROW) if completion_only else {"input_ids": [5, 6, 7, 8, 9, 10, 11]}
    if packing:
        row["seq_lengths"] = [len(row["input_ids"])]
    return row


def _assert_transformers_derivation(batch: dict) -> None:
    """The set equals what transformers derives from the row's positions (its ``position_ids``, else ``0..S-1``):
    int32 offsets and host-int lengths."""
    positions = batch.get("position_ids", torch.arange(batch["input_ids"].shape[1]).unsqueeze(0))
    (cu_q, cu_k), (max_q, max_k) = flash_utils.prepare_fa_kwargs_from_position_ids(positions)
    for key, derived in (("cu_seq_lens_q", cu_q), ("cu_seq_lens_k", cu_k)):
        assert batch[key].dtype == derived.dtype == torch.int32, key
        assert torch.equal(batch[key], derived), (key, batch[key], derived)
    for key, derived in (("max_length_q", max_q), ("max_length_k", max_k)):
        assert type(batch[key]) is int and batch[key] == int(derived), (key, batch[key], derived)


def _equal_batches(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(
        torch.equal(a[key], b[key]) if torch.is_tensor(a[key]) else a[key] == b[key] for key in a
    )


@pytest.mark.parametrize("attn_implementation", VARLEN_ATTN_IMPLEMENTATIONS)
@pytest.mark.parametrize(("completion_only", "packing"), ROUTES, ids=ROUTE_IDS)
def test_a_flash_run_hands_a_one_document_batch_its_varlen_set(attn_implementation, completion_only, packing):
    batch = _collator(_config(attn_implementation), completion_only=completion_only, packing=packing)(
        [_row(completion_only, packing)]
    )
    assert all(key in batch for key in VARLEN_KEYS), sorted(batch)
    _assert_transformers_derivation(batch)
    # Anti-vacuity: transformers would test exactly this row in every layer and run its dense kernel.
    positions = batch.get("position_ids", torch.arange(batch["input_ids"].shape[1]).unsqueeze(0))
    assert not flash_utils._is_packed_sequence(positions, batch_size=1)


@pytest.mark.parametrize(("completion_only", "packing"), ROUTES, ids=ROUTE_IDS)
def test_the_set_is_the_only_change_to_the_batch(completion_only, packing):
    """Each route's batch is the one it built without the set, which is all the collator adds."""
    rows = [_row(completion_only, packing)]
    handed = _collator(_config("flash_attention_2"), completion_only=completion_only, packing=packing)(
        [dict(row) for row in rows]
    )
    # Without the opt-in the full-sequence route is TRL's own collator, which TRL builds from the pad id.
    bare = _collator(
        _config("flash_attention_2"), completion_only=completion_only, packing=packing, one_document_varlen=False
    ) or TRLDataCollatorForLanguageModeling(pad_token_id=PAD_ID)
    assert all(key in handed for key in VARLEN_KEYS)
    assert _equal_batches(without_set(handed), bare([dict(row) for row in rows]))


@pytest.mark.parametrize("labelled", [False, True], ids=("raw_rows", "precomputed_labels"))
@pytest.mark.parametrize("pad_to_multiple_of", [None, 8])
@pytest.mark.parametrize("rows", [(TWO_ROWS[0],), TWO_ROWS], ids=("one_row", "two_rows"))
def test_the_full_sequence_route_is_trls_collator(rows, pad_to_multiple_of, labelled):
    """Where an SFT run used to leave its collator to TRL, the factory's collator is TRL's, built with the arguments
    TRL builds its own with, plus the set on an unpadded one-row batch only."""
    rows = [{**row, "labels": [-100, *row["input_ids"][1:]]} if labelled else dict(row) for row in rows]
    collator = _collator(_config("flash_attention_2"), pad_to_multiple_of=pad_to_multiple_of)
    assert type(collator) is DataCollatorForLanguageModelingWithVarlen
    assert isinstance(collator, TRLDataCollatorForLanguageModeling)
    handed = collator([dict(row) for row in rows])
    trl_batch = TRLDataCollatorForLanguageModeling(pad_token_id=PAD_ID, pad_to_multiple_of=pad_to_multiple_of)(
        [dict(row) for row in rows]
    )
    assert _equal_batches(without_set(handed), trl_batch)
    unpadded_row = len(rows) == 1 and bool(trl_batch["attention_mask"].all())
    assert all((key in handed) is unpadded_row for key in VARLEN_KEYS), sorted(handed)
    if unpadded_row:
        _assert_transformers_derivation(handed)
    if pad_to_multiple_of is not None and len(rows) == 1:
        assert not unpadded_row, "anti-vacuity: the padded one-row batch was not padded"


def test_a_wrapped_packed_row_gets_the_set():
    """``wrapped`` packing emits no ``seq_lengths``, so the packing collator keeps each row whole, mask included: one
    such row is one unpadded document."""
    with pytest.warns(UserWarning, match="seq_lengths"):
        batch = _collator(_config("flash_attention_2"), packing=True)([{"input_ids": [5, 6, 7, 8]}])
    assert "position_ids" not in batch and batch["attention_mask"].all()
    _assert_transformers_derivation(batch)


def test_a_multi_row_batch_gets_no_set():
    """Rows transformers unpads by their mask keep that path (the padded one-row batch is TRL's route's case)."""
    collator = _collator(_config("flash_attention_2"), completion_only=True)
    two = collator([dict(COMPLETION_ROW), {"input_ids": [5, RESPONSE_TEMPLATE[0], 8, EOS_ID]}])
    assert two["input_ids"].shape[0] == 2 and not two["attention_mask"].all()
    assert not any(key in two for key in VARLEN_KEYS)


@pytest.mark.parametrize(
    ("completion_only", "packing"),
    [(False, False), (True, False), (False, True)],
    ids=("trl", "completion_only", "wrapped"),
)
def test_equal_length_rows_get_no_set(completion_only, packing):
    """Two unpadded rows of one length (rows cut at ``max_length``, ``wrapped`` packing above batch 1) carry an
    all-ones mask: one ``[0, S]`` set would have flash attend the first row alone and never write the second."""
    rows = [_row(completion_only), _row(completion_only)]
    collator = _collator(_config("flash_attention_2"), completion_only=completion_only, packing=packing)
    with pytest.warns(UserWarning, match="seq_lengths") if packing else contextlib.nullcontext():
        batch = collator(rows)
    # Anti-vacuity: no padding tells these rows apart from one document; only their count does.
    assert batch["input_ids"].shape[0] == 2 and batch["attention_mask"].all()
    assert not any(key in batch for key in VARLEN_KEYS), sorted(batch)


@pytest.mark.parametrize("name", MULTI_DOCUMENT_BATCHES)
def test_a_multi_document_packed_batch_keeps_the_set_of_its_documents(name):
    """The opt-in adds nothing to a packed row of several documents: it carries the set of its documents, which
    transformers derives from its restarting positions, never the one-document ``[0, S]``."""
    completion_only, rows = MULTI_DOCUMENT_BATCHES[name]
    batch = _collator(_config("flash_attention_2"), completion_only=completion_only, packing=True)(
        [dict(row) for row in rows]
    )
    assert batch["input_ids"].shape[0] == 1 and (batch["position_ids"][0, 1:] == 0).any(), batch["position_ids"]
    _assert_transformers_derivation(batch)
    assert len(batch["cu_seq_lens_q"]) == sum(len(row["seq_lengths"]) for row in rows) + 1


@pytest.mark.parametrize("attn_implementation", ["sdpa", "eager", "flex_attention", None])
def test_a_non_flash_run_keeps_trls_collator_and_gets_no_set(attn_implementation):
    config = _config(attn_implementation)
    assert _collator(config) is None
    for completion_only, packing in ((True, False), (False, True), (True, True)):
        batch = _collator(config, completion_only=completion_only, packing=packing)([_row(completion_only, packing)])
        assert not any(key in batch for key in VARLEN_KEYS)


@pytest.mark.parametrize(
    "kwargs",
    [{"one_document_varlen": False}, {"keeps_packed_rows": True}],
    ids=("caller_did_not_opt_in", "pipeline_parallel"),
)
def test_a_run_outside_the_opt_in_keeps_its_collators(kwargs):
    """A caller whose batches reach another model too (teacher distillation) does not opt in, and PP builds the set
    per stage microbatch: TRL keeps its collator and no route hands a one-document batch the set."""
    config = _config("flash_attention_2")
    assert _collator(config, **kwargs) is None
    for completion_only, packing in ((True, False), (False, True), (True, True)):
        batch = _collator(config, completion_only=completion_only, packing=packing, **kwargs)(
            [_row(completion_only, packing)]
        )
        assert not any(key in batch for key in VARLEN_KEYS)


def test_context_parallel_builds_its_own_collator():
    """Ulysses passes the kernel its own ``cu_seqlens``; the CP collators hand nothing."""
    config = _config("flash_attention_2")
    for completion_only in (False, True):
        collator = _collator(config, completion_only=completion_only, use_context_parallel=True, pad_to_multiple_of=2)
        assert isinstance(collator, DataCollatorForCompletionOnlyLM | DataCollatorForCausalLMWithPadding)
        batch = collator([_row(completion_only)])
        assert batch["input_ids"].shape[0] == 1 and not any(key in batch for key in VARLEN_KEYS)


def test_fa4_keeps_the_dense_kernel_where_its_varlen_kernels_refuse_the_attention():
    """Gemma 2's softcap: FA4's varlen backward takes none, so its one-document rows stay dense on FA4, while the
    FA2 and FA3 varlen kernels take it."""
    softcapped = {"config_cls": Gemma2Config, "attn_logit_softcapping": 50.0}
    assert _collator(_config("flash_attention_4", **softcapped)) is None
    for attn_implementation in ("flash_attention_2", "flash_attention_3"):
        batch = _collator(_config(attn_implementation, **softcapped))([_row()])
        assert all(key in batch for key in VARLEN_KEYS)


def test_a_gated_delta_net_decoder_keeps_its_one_document_path():
    """Its delta rule reads ``cu_seq_lens_q`` too and would take its varlen path; packed rows keep the set they
    always carry."""
    config = Qwen3_5TextConfig(**TINY_QWEN35_CONFIG)
    config._attn_implementation = "flash_attention_2"
    assert _collator(config) is None
    batch = _collator(config, completion_only=True)([dict(COMPLETION_ROW)])
    assert not any(key in batch for key in VARLEN_KEYS)


def test_directly_built_collators_hand_no_set():
    """The set follows the resolved backend, which only the factory reads."""
    tokenizer = padding_tokenizer()
    completion_only = DataCollatorForCompletionOnlyLM(RESPONSE_TEMPLATE, tokenizer=tokenizer)
    packing = DataCollatorWithPacking(tokenizer=tokenizer)
    for batch in (completion_only([_row(completion_only=True)]), packing([_row(packing=True)])):
        assert not any(key in batch for key in VARLEN_KEYS)


@pytest.mark.parametrize("family", sorted(PACKING_FAMILIES))
def test_every_packing_family_reads_the_set_in_place_of_the_test(family):
    report = probe_findings(
        f"from tests.common.one_document_flash_varlen import family_findings; "
        f"from tests.common.packed_flash_varlen import report; report(family_findings({family!r}))",
        FINDINGS_MARKER,
    )
    if report and report[0].startswith(SKIP_PREFIX):
        pytest.skip(report[0].removeprefix(SKIP_PREFIX).strip())
    failures = [entry for entry in report if not entry.startswith(REACH_PREFIX)]
    assert not failures, "\n".join(failures)
    reach = int(next(entry for entry in report if entry.startswith(REACH_PREFIX)).removeprefix(REACH_PREFIX))
    assert reach > 0, f"no attention layer of {family} read the handed-over set"


@pytest.mark.parametrize("gc_mode", list(GC_MODES))
@pytest.mark.parametrize("family", STEP_FAMILIES)
def test_a_training_step_is_bit_identical_and_reads_the_handed_set(family, gc_mode):
    failures = probe_findings(
        "from tests.common.one_document_flash_varlen import step_findings; "
        f"from tests.common.packed_flash_varlen import report; report(step_findings({family!r}, {gc_mode!r}))",
        FINDINGS_MARKER,
    )
    assert not failures, "\n".join(failures)


def test_a_profiled_depth_step_makes_no_per_layer_read():
    """The profiled dense row's depth under reentrant checkpointing: without the set every layer tests the row in
    the forward and again in the recompute (two per layer and step), with it none does and attention reads nothing
    back to the host."""
    report = probe_findings(
        "from tests.common.one_document_flash_varlen import host_read_findings; "
        "from tests.common.packed_flash_varlen import report; report(host_read_findings())",
        FINDINGS_MARKER,
    )
    failures = [entry for entry in report if not entry.startswith(("reads:", "tests:"))]
    assert not failures, "\n".join(failures)
    assert f"tests:{2 * PROFILED_DEPTH}:0" in report, report


def _sft_trainer(tmp_path, model, collator) -> DistributedSFTTrainer:
    args = SFTConfig(
        output_dir=str(tmp_path),
        use_cpu=True,
        bf16=False,
        per_device_train_batch_size=1,
        gradient_checkpointing=False,
        use_liger_kernel=False,
        dataloader_num_workers=0,
        save_strategy="no",
        report_to="none",
        disable_tqdm=True,
    )
    return DistributedSFTTrainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=Dataset.from_dict({"input_ids": [[5, 6, 7]]}),
        processing_class=make_offline_tokenizer(),
        parallelism_config=make_parallelism_config(world_size=1, gpus_per_node=1),
    )


def test_the_trainer_hands_the_set_to_the_decoder(tmp_path):
    """The SFT trainer's input preparation and loss pass the full-sequence route's set to the decoder unchanged."""
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(_config("sdpa"))
    decoder = next(module for module in model.modules() if isinstance(module, Qwen3Model))
    received = []
    decoder.register_forward_pre_hook(lambda _module, _args, kwargs: received.append(kwargs), with_kwargs=True)
    collator = _collator(_config("flash_attention_2"))
    trainer = _sft_trainer(tmp_path, model, collator)
    batch = collator([_row()])

    trainer.compute_loss(trainer.model, trainer._prepare_inputs(dict(batch)))

    assert len(received) == 1 and all(key in received[0] for key in VARLEN_KEYS), sorted(received[0])
    for key in VARLEN_KEYS:
        handed, expected = received[0][key], batch[key]
        assert torch.equal(handed, expected) if torch.is_tensor(expected) else handed == expected, key


def test_a_decoder_without_kwargs_refuses_the_set(tmp_path):
    """A remote-code forward declaring no ``**kwargs`` fails loud on the set at the first step: nothing between the
    collator and the model filters the batch by signature, so no layer silently goes back to testing the row."""

    class NoKwargsLM(Qwen3ForCausalLM):
        def forward(self, input_ids=None, attention_mask=None, labels=None, use_cache=None):
            return super().forward(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=use_cache
            )

    collator = _collator(_config("flash_attention_2"))
    trainer = _sft_trainer(tmp_path, NoKwargsLM(_config("sdpa")), collator)
    batch = trainer._prepare_inputs(collator([_row()]))
    with pytest.raises(TypeError, match="cu_seq_lens"):
        trainer.compute_loss(trainer.model, batch)
    trainer.compute_loss(trainer.model, without_set(batch))


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ({}, DataCollatorForLanguageModelingWithVarlen),
        ({"packing": True}, type(None)),
        ({"padding_free": True}, type(None)),
        ({"cp_size": 2}, DataCollatorForCausalLMWithPadding),
        ({"pp_size": 2}, type(None)),
    ],
    ids=("standard", "packing", "padding_free", "cp", "pp"),
)
def test_the_suites_and_benchmarks_build_the_sft_scripts_collator(mode, expected):
    """The GPU SFT suites and the profiling benchmarks train the collator the SFT script selects; TRL keeps the rows
    it packs itself and PP's."""
    sft_config = SimpleNamespace(
        packing=mode.get("packing", False),
        padding_free=mode.get("padding_free", False),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=8,
        eval_strategy="no",
        pad_to_multiple_of=8,
    )
    cp_size, pp_size = mode.get("cp_size", 1), mode.get("pp_size", 1)
    # The parallelism view the helper reads, without the topology a real CP or PP config needs.
    parallelism = SimpleNamespace(is_cp_mode=cp_size > 1, cp_size=cp_size, pp_size=pp_size)
    collator = build_sft_collator(padding_tokenizer(), sft_config, _config("flash_attention_4"), parallelism)
    assert type(collator) is expected
    if collator is not None:
        # The run's pad_to_multiple_of outside CP, as TRL builds its own collator with; cp_size under CP.
        assert collator.pad_to_multiple_of == (cp_size if cp_size > 1 else sft_config.pad_to_multiple_of)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
