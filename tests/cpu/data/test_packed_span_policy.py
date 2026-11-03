#!/usr/bin/env python
"""The packed routes and the offline bake resolve completion spans under one named policy,
:data:`PACKED_SPAN_POLICY`. The packed collators add an end-of-sequence fallback for a row with no terminator, so
baking the padded policy into a packed artifact makes one YAML train different tokens when pre-processed."""

import pytest
import torch
from datasets import Dataset

from halo.data.collators.packing import (
    DataCollatorForCompletionOnlyLMWithPacking,
    DataCollatorWithFlatteningAndCompletionMask,
)
from halo.data.completion_spans import PACKED_SPAN_POLICY, PADDED_SPAN_POLICY, mask_batch_to_completion_spans
from halo.data.preprocessed import tokenization as tokenization_mod
from halo.data.preprocessed.metadata import PreprocessingConfig
from halo.data.preprocessed.tokenization import _completion_only_labels, tokenize_dataset
from tests.common.models import QWEN3_0_6B
from tests.common.tokenizers import load_cached_tokenizer

PAD, EOS, IGNORE = 0, 1, -100
RESP_1, RESP_2 = 10, 11
TEMPLATE_IDS = [RESP_1, RESP_2]

# Rows whose every assistant turn closes with a terminator: the policies must agree here, or the one
# named flag has quietly changed the padded route too.
TERMINATED_ROWS = [
    pytest.param([20, RESP_1, RESP_2, 30, EOS], id="one-turn"),
    pytest.param([20, RESP_1, RESP_2, 30, EOS, 21, RESP_1, RESP_2, 31, EOS], id="two-turns"),
    pytest.param([20, 21, 22, EOS], id="no-marker"),
    pytest.param([20, RESP_1, RESP_2, 30, EOS, 40], id="trailing-context"),
]

# The divergent shape, a marker with no terminator anywhere in the sequence: a chunk whose terminator landed in the
# next one, or a final turn the length filter clipped.
UNTERMINATED_ROW = [20, RESP_1, RESP_2, 30, 31]


class _Tokenizer:
    """The surface the collators read: ``encode`` returns the pre-tokenized marker, ``decode`` feeds
    the terminator walk and the span-miss warning."""

    pad_token_id = PAD
    eos_token_id = EOS
    padding_side = "right"

    def encode(self, *_args, **_kwargs):
        return TEMPLATE_IDS

    def decode(self, token_ids, **_kwargs):
        return " ".join(f"t{token_id}" for token_id in token_ids)


@pytest.fixture
def _isolated_datasets_cache(tmp_path, monkeypatch):
    """Fresh coordinated-map cache: a hit from an earlier run would skip the bake under test."""
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "hf_datasets"))


def _bake(input_ids: list[int], span_policy) -> list[int]:
    return _completion_only_labels(
        input_ids,
        _Tokenizer(),
        assistant_template="<resp>",
        response_token_ids=TEMPLATE_IDS,
        eos_token_ids=frozenset({EOS}),
        span_policy=span_policy,
    )


def _packed_collator_labels(input_ids: list[int]) -> list[int]:
    collator = DataCollatorForCompletionOnlyLMWithPacking(
        tokenizer=_Tokenizer(),
        response_prompt_template=TEMPLATE_IDS,
        eos_token_ids=frozenset({EOS}),
    )
    ids = torch.tensor(input_ids)
    return collator._mask_sequence(ids.clone(), input_ids=ids).tolist()


def _flattened_collator_labels(input_ids: list[int]) -> list[int]:
    collator = DataCollatorWithFlatteningAndCompletionMask(
        tokenizer=_Tokenizer(),
        response_prompt_template=TEMPLATE_IDS,
        eos_token_ids=frozenset({EOS}),
        emit_varlen_always=True,
    )
    return collator([{"input_ids": list(input_ids)}])["labels"][0].tolist()


def _padded_route_labels(input_ids: list[int]) -> list[int]:
    """The batch masker under its default (padded) policy, the route that must not gain a fallback."""
    ids = torch.tensor([input_ids])
    batch = mask_batch_to_completion_spans(
        {"input_ids": ids, "labels": ids.clone()},
        TEMPLATE_IDS,
        frozenset({EOS}),
        train_on_last_assistant_only=False,
        response_prompt_template="<resp>",
        tokenizer=None,
    )
    return batch["labels"][0].tolist()


@pytest.mark.parametrize("input_ids", TERMINATED_ROWS)
def test_every_route_masks_a_terminated_row_identically(input_ids):
    expected = _padded_route_labels(input_ids)

    assert _packed_collator_labels(input_ids) == expected
    assert _flattened_collator_labels(input_ids) == expected
    assert _bake(input_ids, PADDED_SPAN_POLICY) == expected
    assert _bake(input_ids, PACKED_SPAN_POLICY) == expected


def test_the_packed_bake_and_the_packed_collators_agree_on_a_terminator_less_turn():
    """A packed artifact's baked labels equal what the packed collator would produce: the turn trains to the
    sequence end on both paths."""
    packed_runtime = _packed_collator_labels(UNTERMINATED_ROW)

    assert packed_runtime == [IGNORE, RESP_1, RESP_2, 30, 31], packed_runtime
    assert _bake(UNTERMINATED_ROW, PACKED_SPAN_POLICY) == packed_runtime
    assert _flattened_collator_labels(UNTERMINATED_ROW) == packed_runtime


def test_the_padded_route_still_drops_a_terminator_less_turn():
    """On a padded row the sequence end is pad, so the padded route masks the turn out. The fallback cannot become
    universal."""
    padded = _padded_route_labels(UNTERMINATED_ROW)

    assert padded == [IGNORE] * len(UNTERMINATED_ROW), padded
    assert _bake(UNTERMINATED_ROW, PADDED_SPAN_POLICY) == padded
    assert _bake(UNTERMINATED_ROW, None) == padded, "the bake's default stays the padded policy"


@pytest.mark.parametrize("pack_sequences", [False, True], ids=["unpacked", "packed"])
def test_the_bake_selects_the_policy_its_artifact_will_be_collated_under(
    monkeypatch, pack_sequences, _isolated_datasets_cache
):
    """``tokenize_dataset`` hands the bake the policy of the collator that will read the artifact."""
    tokenizer = load_cached_tokenizer(QWEN3_0_6B)

    seen = []
    real_bake = tokenization_mod._completion_only_labels

    def recording(*args, span_policy=None, **kwargs):
        seen.append(span_policy)
        return real_bake(*args, span_policy=span_policy, **kwargs)

    monkeypatch.setattr(tokenization_mod, "_completion_only_labels", recording)

    dataset = Dataset.from_dict(
        {
            "conversation": [
                [
                    {"role": "user", "content": f"pack={pack_sequences}?"},
                    {"role": "assistant", "content": "Sure."},
                ]
            ]
        }
    )
    config = PreprocessingConfig(
        model_name_or_path="Qwen/Qwen3-0.6B",
        max_length=512,
        conversation_field="conversation",
        train_on_completions_only=True,
        assistant_message_template="<|im_start|>assistant",
        pack_sequences=pack_sequences,
        num_proc=1,
    )

    tokenize_dataset(dataset, tokenizer, config, split_name=f"pack_{pack_sequences}")

    expected = PACKED_SPAN_POLICY if pack_sequences else PADDED_SPAN_POLICY
    assert seen and all(policy == expected for policy in seen), f"baked under {seen}, expected {expected}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
