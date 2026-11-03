#!/usr/bin/env python
"""On a flash run the packing collators hand attention the varlen set transformers would derive, sparing each layer
a device sync and a re-derivation. Across the roster (:mod:`tests.common.packed_flash_varlen`) kernel arguments
and logits stay identical and the recompute re-runs on the same set. Non-flash and PP carry nothing."""

from itertools import accumulate

import pytest
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen3Model
from transformers import modeling_flash_attention_utils as flash_utils
from trl import SFTConfig

from halo.data.collators.factory import select_data_collator
from halo.data.collators.packing import DataCollatorForCompletionOnlyLMWithPacking, DataCollatorWithPacking
from halo.kernels.logprobs import LABEL_IGNORE_INDEX
from halo.models.attention.selection import VARLEN_ATTN_IMPLEMENTATIONS
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.models import TINY_QWEN3_CONFIG
from tests.common.offline_grpo import make_offline_tokenizer
from tests.common.packed_flash_varlen import (
    EOS_ID,
    FINDINGS_MARKER,
    GC_MODES,
    MULTI_DOCUMENT_ROWS,
    PACKING_FAMILIES,
    PAD_ID,
    REACH_PREFIX,
    SKIP_PREFIX,
    VARLEN_KEYS,
    assert_transformers_derivation,
    padding_tokenizer,
)
from tests.common.parallelism import make_parallelism_config
from tests.common.repo import probe_findings

# A row without ``seq_lengths`` keeps its right padding inside its one document (a trailing pad document);
# a row whose documents end before the batch width has its pads dropped by the flatten.
MIXED_ROWS = (
    {"input_ids": [5, 6, 7, 8, 9, 10, 11, 12, 13], "seq_lengths": [4, 5]},
    {"input_ids": [14, 15, 16]},
    {"input_ids": [17, 18, 19, 20], "seq_lengths": [4]},
)
ONE_DOCUMENT_ROWS = ({"input_ids": [5, 6, 7, 8, 9], "seq_lengths": [5]},)
RESPONSE_TEMPLATE = [30]
# The attention variants a checkpointed packed step trains: dense, gpt-oss sinks, a routed MoE, Mistral 4's MLA.
RECOMPUTE_FAMILIES = ("dense:qwen3", "gpt_oss", "qwen3_moe", "mistral4")


def _config(attn_implementation: str | None):
    config = Qwen3Config(**TINY_QWEN3_CONFIG)
    config._attn_implementation = attn_implementation
    return config


def _collator(attn_implementation: str | None, *, completion_only: bool = False):
    return select_data_collator(
        padding_tokenizer(),
        train_on_completions_only=completion_only,
        assistant_message_template=RESPONSE_TEMPLATE if completion_only else None,
        packing=True,
        model_config=_config(attn_implementation),
    )


def _rows(rows):
    return [dict(row) for row in rows]


def _completion_rows(rows):
    """``rows`` with each document of three or more tokens opening on the response template and closing on
    EOS, so completion-only masking keeps the span between."""
    marked = []
    for row in rows:
        input_ids = list(row["input_ids"])
        lengths = row.get("seq_lengths", (len(input_ids),))
        for start, length in zip(accumulate((0, *lengths[:-1])), lengths, strict=True):
            if length >= 3:
                input_ids[start], input_ids[start + length - 1] = RESPONSE_TEMPLATE[0], EOS_ID
        marked.append({**row, "input_ids": input_ids})
    return marked


@pytest.mark.parametrize("completion_only", [False, True], ids=("packing", "completion_only_packing"))
@pytest.mark.parametrize("attn_implementation", VARLEN_ATTN_IMPLEMENTATIONS)
@pytest.mark.parametrize(
    "rows", [MULTI_DOCUMENT_ROWS, MIXED_ROWS, ONE_DOCUMENT_ROWS], ids=("multi_document", "mixed", "one_document")
)
def test_a_flash_run_gets_the_varlen_set_transformers_would_derive(rows, attn_implementation, completion_only):
    rows = _completion_rows(rows) if completion_only else _rows(rows)
    batch = _collator(attn_implementation, completion_only=completion_only).torch_call(rows)
    assert batch["position_ids"].shape[0] == 1
    assert_transformers_derivation(batch, batch["position_ids"])
    if completion_only:
        assert (batch["labels"] != LABEL_IGNORE_INDEX).any(), "the response template matched no document"


def test_the_batches_cover_the_cases():
    """The multi-document and mixed batches emit the set and the one-document batch does not; the mixed batch's
    unsegmented row keeps its trailing pads."""
    collator = _collator("flash_attention_2")
    assert "cu_seq_lens_q" in collator.torch_call(_rows(MULTI_DOCUMENT_ROWS))
    mixed = collator.torch_call(_rows(MIXED_ROWS))
    assert mixed["cu_seq_lens_q"].tolist() == [0, 4, 9, 18, 22]
    assert mixed["input_ids"][0, 9:18].tolist() == [14, 15, 16] + [PAD_ID] * 6
    one_document = collator.torch_call(_rows(ONE_DOCUMENT_ROWS))
    assert not flash_utils._is_packed_sequence(one_document["position_ids"], batch_size=1)
    assert "cu_seq_lens_q" not in one_document


@pytest.mark.parametrize("attn_implementation", ["sdpa", "eager", "flex_attention", None])
def test_a_non_flash_run_gets_no_varlen_set(attn_implementation):
    batch = _collator(attn_implementation).torch_call(_rows(MULTI_DOCUMENT_ROWS))
    assert not any(key in batch for key in VARLEN_KEYS)


def test_kept_rows_get_no_varlen_set():
    """PP keeps the packed rows apart, where the set has no per-row convention."""
    collator = _collator("flash_attention_2")
    collator.flatten_to_single_row = False
    batch = collator.torch_call(_rows(MULTI_DOCUMENT_ROWS))
    assert batch["position_ids"].shape[0] == 2
    assert not any(key in batch for key in VARLEN_KEYS)


@pytest.mark.parametrize(
    ("cls", "extra"),
    [
        (DataCollatorWithPacking, {}),
        (DataCollatorForCompletionOnlyLMWithPacking, {"response_prompt_template": RESPONSE_TEMPLATE}),
    ],
    ids=("packing", "completion_only_packing"),
)
def test_a_directly_built_collator_emits_no_varlen_set(cls, extra):
    """The set follows the resolved backend, which only the factory reads."""
    batch = cls(tokenizer=padding_tokenizer(), **extra).torch_call(_rows(MULTI_DOCUMENT_ROWS))
    assert not any(key in batch for key in VARLEN_KEYS)


@pytest.mark.parametrize("family", sorted(PACKING_FAMILIES))
def test_every_packing_family_reads_the_set_in_place_of_the_derivation(family):
    report = probe_findings(f"from tests.common.packed_flash_varlen import main; main({family!r})", FINDINGS_MARKER)
    if report and report[0].startswith(SKIP_PREFIX):
        pytest.skip(report[0].removeprefix(SKIP_PREFIX).strip())
    failures = [entry for entry in report if not entry.startswith(REACH_PREFIX)]
    assert not failures, "\n".join(failures)
    reach = int(next(entry for entry in report if entry.startswith(REACH_PREFIX)).removeprefix(REACH_PREFIX))
    assert reach > 0, f"no attention layer of {family} read the handed-over set"


@pytest.mark.parametrize("gc_mode", list(GC_MODES))
@pytest.mark.parametrize("family", RECOMPUTE_FAMILIES)
def test_the_recompute_reads_the_handed_set(family, gc_mode):
    """No layer derives it again, and a set with a shifted boundary trains the same with checkpointing as without;
    set-vs-derived equality alone cannot see a recompute that re-derives."""
    failures = probe_findings(
        "from tests.common.packed_flash_varlen import recompute_findings, report; "
        f"report(recompute_findings({family!r}, {gc_mode!r}))",
        FINDINGS_MARKER,
    )
    assert not failures, "\n".join(failures)


def test_the_sweep_holds_a_flash_family():
    """The dense roster family runs flash and reads no GatedDeltaNet marker, so its sweep entry cannot be a skip."""
    assert "dense:qwen3" in PACKING_FAMILIES and Qwen3ForCausalLM._supports_flash_attn


def _sft_trainer(tmp_path, model) -> DistributedSFTTrainer:
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
        train_dataset=Dataset.from_dict({"input_ids": [[5, 6, 7]]}),
        processing_class=make_offline_tokenizer(),
        parallelism_config=make_parallelism_config(world_size=1, gpus_per_node=1),
    )


@pytest.mark.parametrize("adapter", [False, True], ids=("full", "lora"))
def test_the_trainer_hands_the_set_to_the_decoder(tmp_path, adapter):
    """The trainer's input preparation and loss, and a LoRA wrapper over the model, pass the set through to
    the decoder backbone unchanged; a filter on the way would put the derivation back in every layer."""
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(_config("sdpa"))
    if adapter:
        model = get_peft_model(model, LoraConfig(r=4, target_modules=["q_proj", "v_proj"]))
    decoder = next(module for module in model.modules() if isinstance(module, Qwen3Model))
    received = []
    decoder.register_forward_pre_hook(lambda _module, _args, kwargs: received.append(kwargs), with_kwargs=True)
    trainer = _sft_trainer(tmp_path, model)
    batch = _collator("flash_attention_2").torch_call(_rows(MULTI_DOCUMENT_ROWS))
    assert all(key in batch for key in VARLEN_KEYS)

    trainer.compute_loss(trainer.model, trainer._prepare_inputs(dict(batch)))

    assert len(received) == 1, len(received)
    for key in VARLEN_KEYS:
        assert key in received[0], f"{key} never reached the decoder: {sorted(received[0])}"
        handed, expected = received[0][key], batch[key]
        assert torch.equal(handed, expected) if torch.is_tensor(expected) else handed == expected, key


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
