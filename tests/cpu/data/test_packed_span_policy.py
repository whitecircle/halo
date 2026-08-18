#!/usr/bin/env python
"""CPU tests: the packed routes and the offline bake resolve completion spans under ONE named policy.

The packed collator, the padding-free collator and the offline label bake all mask assistant
completions. They agree on every row that carries a turn terminator; the row that does not is where
they diverge — an unnamed third policy in the packed runtime collators (the collator policy plus an
end-of-sequence fallback) against the padded one in the bake makes the SAME YAML train different
tokens depending on whether its dataset was pre-processed.

:data:`PACKED_SPAN_POLICY` names that third policy, both packed collators take it, and the bake takes
it exactly when the artifact it writes will be packed — which is exactly when the packing collator is
the one that would have masked those rows at runtime.

Run: python tests/cpu/data/test_packed_span_policy.py  (or pytest)
"""

import pytest
import torch
from datasets import Dataset
from transformers import AutoTokenizer

from src.data.collators.packing import (
    DataCollatorForCompletionOnlyLMWithPacking,
    DataCollatorWithFlatteningAndCompletionMask,
)
from src.data.pipeline import preprocessing as preprocessing_mod
from src.data.pipeline.preprocessed_metadata import PreprocessingConfig
from src.data.pipeline.preprocessing import _completion_only_labels, tokenize_dataset
from src.data.spans import COLLATOR_SPAN_POLICY, PACKED_SPAN_POLICY, mask_batch_to_completion_spans

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

# The divergent shape: a marker whose turn has no terminator anywhere in the sequence —
# a chunk whose terminator landed in the next one, or a final turn the length filter clipped.
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
    """What the packed SFT collator trains for one document."""
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
        return_flash_attn_kwargs=True,
    )
    return collator([{"input_ids": list(input_ids)}])["labels"][0].tolist()


def _padded_route_labels(input_ids: list[int]) -> list[int]:
    """The batch masker under its default (padded) policy — the route that must NOT gain a fallback."""
    ids = torch.tensor([input_ids])
    batch = mask_batch_to_completion_spans(
        {"input_ids": ids, "labels": ids.clone()},
        TEMPLATE_IDS,
        frozenset({EOS}),
        ignore_index=IGNORE,
        train_on_last_assistant_only=False,
        response_prompt_template="<resp>",
        tokenizer=None,
    )
    return batch["labels"][0].tolist()


@pytest.mark.parametrize("input_ids", TERMINATED_ROWS)
def test_every_route_masks_a_terminated_row_identically(input_ids):
    """Equivalence where the policies must agree: naming the packed policy changes nothing for a row
    whose turns all close, on any of the four routes."""
    expected = _padded_route_labels(input_ids)

    assert _packed_collator_labels(input_ids) == expected
    assert _flattened_collator_labels(input_ids) == expected
    assert _bake(input_ids, COLLATOR_SPAN_POLICY) == expected
    assert _bake(input_ids, PACKED_SPAN_POLICY) == expected


def test_the_packed_bake_and_the_packed_collators_agree_on_a_terminator_less_turn():
    """The divergent case, now agreed: a packed artifact's baked labels equal what the packed
    collator would have produced for the same row — the turn trains to the sequence end instead of
    dropping out of the loss on one of the two paths."""
    packed_runtime = _packed_collator_labels(UNTERMINATED_ROW)

    assert packed_runtime == [IGNORE, RESP_1, RESP_2, 30, 31], packed_runtime
    assert _bake(UNTERMINATED_ROW, PACKED_SPAN_POLICY) == packed_runtime
    assert _flattened_collator_labels(UNTERMINATED_ROW) == packed_runtime


def test_the_padded_route_still_drops_a_terminator_less_turn():
    """Anti-vacuity, and the reason the fallback cannot simply become universal: on a padded row the
    sequence end is pad, so the padded route masks the turn out rather than training to it. The two
    policies genuinely differ on this row — which is what makes the agreement above load-bearing."""
    padded = _padded_route_labels(UNTERMINATED_ROW)

    assert padded == [IGNORE] * len(UNTERMINATED_ROW), padded
    assert _bake(UNTERMINATED_ROW, COLLATOR_SPAN_POLICY) == padded
    assert _bake(UNTERMINATED_ROW, None) == padded, "the bake's default stays the padded policy"


@pytest.mark.parametrize("pack_sequences", [False, True], ids=["unpacked", "packed"])
def test_the_bake_selects_the_policy_its_artifact_will_be_collated_under(
    monkeypatch, pack_sequences, _isolated_datasets_cache
):
    """Wiring: ``tokenize_dataset`` hands the bake the policy of the collator that will read the
    artifact. Baking one policy into an artifact collated under the other is the divergence itself."""
    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    except Exception as e:
        pytest.skip(f"tokenizer unavailable offline: {e}")

    seen = []
    real_bake = preprocessing_mod._completion_only_labels

    def recording(*args, span_policy=None, **kwargs):
        seen.append(span_policy)
        return real_bake(*args, span_policy=span_policy, **kwargs)

    monkeypatch.setattr(preprocessing_mod, "_completion_only_labels", recording)

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

    expected = PACKED_SPAN_POLICY if pack_sequences else COLLATOR_SPAN_POLICY
    assert seen and all(policy == expected for policy in seen), f"baked under {seen}, expected {expected}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
