#!/usr/bin/env python
"""CPU tests: tree packing trains exactly the rows it is given.

Tree packing is an optimization, so its contract is an equality: every root-to-leaf path of a
processed tree must carry the tokens the flat pipeline tokenizes for that row, and the labels the flat
packing collator gives it. These tests run both pipelines over the same conversations with the
production GPT-OSS tokenizer and chat template and compare example by example. They also pin what
decides which rows are examined together (the share key and the ordering it induces) and the
collator's refusals — the cases where a tree row would otherwise reach the model in a shape that
attends across branches. The last test drives the SFT script's data path both ways over one shuffled
dataset, so the ordering, the typed map, the filter and the packing between the two are covered too.

Run: python tests/cpu/data/test_tree_rows.py  (or pytest)
"""

import random
import sys
from types import SimpleNamespace
from unittest import mock

import datasets
import numpy as np
import pytest
import torch
from accelerate import PartialState
from datasets import Dataset, DatasetDict
from transformers import GptOssConfig, Qwen3Config

PartialState()  # the dataset stages log through accelerate, which refuses to log without it

from scripts.training import sft as sft_script
from scripts.training.sft import _prepare_text_data
from src.data.collators.packing import DataCollatorForCompletionOnlyLMWithPacking
from src.data.collators.tree_packing import DataCollatorWithTreePacking
from src.data.pipeline.text_rows import build_chat_row_processor
from src.data.pipeline.tree_rows import (
    SHARE_KEY_COLUMN,
    TREE_ROW_FEATURES,
    _trained_tokens,
    build_share_key_map,
    build_tree_row_processor,
    is_kept_tree,
    prepare_tree_dataset,
    share_key,
)
from src.data.tree_layout import (
    DEPTH_COLUMN,
    POSITION_COLUMN,
    TRAINED_COLUMN,
    build_trie,
    serialize_tree,
    share_limit,
)
from src.data.turn_markers import resolve_eos_token_ids, tokenize_response_template
from src.kernels.logprobs import LABEL_IGNORE_INDEX
from src.models.attention import selection
from src.models.attention.tree_plan import TREE_ATTENTION_KEY, build_node_table
from tests.common.models import GPT_OSS_20B
from tests.common.repo import REPO_ROOT
from tests.common.tokenizers import load_cached_tokenizer
from tests.common.trees import example_rows

RESPONSE = "<|start|>assistant<|channel|>final<|message|>"
MAX_LENGTH = 4096
WINDOW = 128
HEADER = "You are given a document and a set of questions about it.\n"
DOCUMENT = "\nDocument:\n" + "The forecast says it will rain in the old city tomorrow. " * 16
MENU = "\nQuestions:\n"
MARKERS = (MENU, "\nAnswer options:\n")
REQUIRES = "\nDocument:\n"
WEATHER = (
    "Weather: does the document say it will rain, snow or blow in any city? Use the document alone, never memory."
)


def _question_row(menu: str, answer: str, *, document: str = DOCUMENT, menu_first: bool = False) -> list[dict]:
    body = MENU + menu + document if menu_first else document + MENU + menu
    return [{"role": "user", "content": HEADER + body + "\nAnswer in JSON."}, {"role": "assistant", "content": answer}]


def _conversations() -> list[list[dict]]:
    """Four rows over one document (two questions, the first of them alone, and one prompt with two
    answers), a row over another document, a questions-first row, and chat."""
    return [
        _question_row(f'{{\n  "1": "{WEATHER}",\n  "2": "Sports."\n}}', '{"1": "yes", "2": "no"}'),
        _question_row(f'{{\n  "1": "{WEATHER}"\n}}', '{"1": "yes"}'),
        _question_row('{\n  "1": "Sports."\n}', '{"1": "no"}'),
        _question_row('{\n  "1": "Sports."\n}', '{"1": "unsure"}'),
        _question_row('{\n  "1": "Sports."\n}', '{"1": "no"}', document=DOCUMENT.replace("rain", "snow")),
        _question_row('{\n  "1": "Sports."\n}', '{"1": "no"}', menu_first=True),
        [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi, how can I help?"}],
    ]


@pytest.fixture(scope="module")
def tokenizer():
    tokenizer = load_cached_tokenizer(GPT_OSS_20B)
    tokenizer.chat_template = (REPO_ROOT / "jinja-templates" / "gpt-oss" / "gpt-oss-multiturn.jinja").read_text()
    tokenizer.padding_side = "right"
    return tokenizer


def _process(tokenizer, conversations, *, max_length=MAX_LENGTH, min_shared_tokens=16, last_only=True) -> list[dict]:
    """Processed tree rows, read back from a dataset of the declared column types as the trainer reads them."""
    processor = build_tree_row_processor(
        tokenizer,
        max_length,
        conversation_field="conversation",
        share_markers=MARKERS,
        share_requires=REQUIRES,
        eos_token_ids=resolve_eos_token_ids(tokenizer),
        assistant_message_template=RESPONSE,
        train_on_last_assistant_only=last_only,
        min_shared_tokens=min_shared_tokens,
    )
    return list(Dataset.from_dict(processor({"conversation": conversations}), features=TREE_ROW_FEATURES))


def _flat_examples(tokenizer, conversations, *, last_only=True) -> list[tuple[list[int], list[int]]]:
    """(tokens, labels) of every row by the flat packed pipeline."""
    tokenize = build_chat_row_processor(tokenizer, MAX_LENGTH, conversation_field="conversation", use_padding=False)
    collator = DataCollatorForCompletionOnlyLMWithPacking(
        response_prompt_template=RESPONSE,
        tokenizer=tokenizer,
        train_on_last_assistant_only=last_only,
        eos_token_ids=resolve_eos_token_ids(tokenizer),
    )
    examples = []
    for conversation in conversations:
        input_ids = tokenize({"conversation": conversation})["input_ids"]
        batch = collator([{"input_ids": input_ids, "seq_lengths": [len(input_ids)]}])
        examples.append((input_ids, batch["labels"][0].tolist()))
    return examples


def _tree_examples(batch: dict) -> list[tuple[list[int], list[int]]]:
    """(tokens, labels) along every root-to-leaf path of a collated tree batch."""
    table = build_node_table(batch[POSITION_COLUMN], batch[DEPTH_COLUMN])
    return [(batch["input_ids"][0, rows].tolist(), batch["labels"][0, rows].tolist()) for rows in example_rows(table)]


def _collate(tokenizer, tree_rows) -> dict:
    batch = DataCollatorWithTreePacking(tokenizer, sliding_window=WINDOW, full_form="split")(tree_rows)
    batch[DEPTH_COLUMN] = torch.tensor([depth for row in tree_rows for depth in row[DEPTH_COLUMN]])
    return batch


def test_rows_key_on_what_they_share_up_to_the_marker():
    rows = _conversations()
    keys = [share_key(row, MARKERS, REQUIRES) for row in rows]
    assert len(set(keys[:4])) == 1, "rows over one document differ only after the marker"
    assert len(set(keys)) == 4, "another document, a questions-first row and chat each stand alone"
    # Without the requirement a questions-first row keys on its short header, as every such row would.
    hail = _question_row('{\n  "1": "Other."\n}', "{}", document=DOCUMENT.replace("rain", "hail"), menu_first=True)
    assert share_key(rows[5], MARKERS, None) == share_key(hail, MARKERS, None)
    assert share_key(rows[5], MARKERS, REQUIRES) != share_key(hail, MARKERS, REQUIRES)
    # The marker is part of what is shared: the same conversation under the other marker stands alone.
    other = [dict(rows[0][0], content=rows[0][0]["content"].replace(MENU, MARKERS[1])), rows[0][1]]
    assert share_key(other, MARKERS, REQUIRES) != keys[0]
    # A marker that occurs twice names no single cut, so the row keys on its whole content.
    twice = _question_row('{\n  "1": "a' + MENU + 'b"\n}', "{}")
    assert share_key(twice, MARKERS, REQUIRES) != keys[0]
    assert share_key(twice, MARKERS, REQUIRES) == share_key([dict(message) for message in twice], MARKERS, REQUIRES)


def test_the_key_map_keys_every_row_of_a_batch_as_share_key_does():
    rows = _conversations()
    keys = build_share_key_map("conversation", MARKERS, REQUIRES)({"conversation": rows})[SHARE_KEY_COLUMN]
    assert keys == [share_key(row, MARKERS, REQUIRES) for row in rows]


def test_tree_paths_are_the_flat_examples_token_for_token(tokenizer):
    rows = _conversations()
    flat = _flat_examples(tokenizer, rows)
    trees = _process(tokenizer, rows)
    assert len(trees) == 4, "one tree row per run of rows with one key"
    batch = _collate(tokenizer, trees)

    assert sorted(_tree_examples(batch)) == sorted(flat)
    assert all(any(label != LABEL_IGNORE_INDEX for label in labels) for _, labels in flat)
    # The four rows over one document store their ~250 prefix tokens once instead of four times.
    assert batch["input_ids"].shape[1] < sum(len(tokens) for tokens, _ in flat) - 500
    assert TREE_ATTENTION_KEY in batch
    # The plan is the one for this collator's window and form: a node's sliding keys are its own tokens
    # plus the window - 1 path tokens before its first, and the split form gathers queries per parent.
    plan, table = batch[TREE_ATTENTION_KEY], build_node_table(batch[POSITION_COLUMN], batch[DEPTH_COLUMN])
    reach = batch[POSITION_COLUMN][0, table.starts].clamp(max=WINDOW - 1) + torch.from_numpy(table.lengths)
    assert plan["sliding"]["cu_seqlens_k"].diff().tolist() == reach.tolist()
    assert "q_index" in plan["full"]
    assert batch[POSITION_COLUMN].shape == batch["input_ids"].shape == batch["labels"].shape
    assert "attention_mask" not in batch


@pytest.mark.parametrize("last_only", [True, False])
def test_multi_turn_rows_train_the_turns_the_flat_pipeline_trains(tokenizer, last_only):
    history = [{"role": "user", "content": "Is rain likely tomorrow?"}, {"role": "assistant", "content": "Probably."}]
    rows = [history + _question_row(f'{{\n  "1": "Question {p}."\n}}', f'{{"1": "no {p}"}}') for p in range(3)]
    flat = _flat_examples(tokenizer, rows, last_only=last_only)
    trained = {
        flag: [sum(label != LABEL_IGNORE_INDEX for label in labels) for _, labels in examples]
        for flag, examples in (
            (last_only, flat),
            (not last_only, _flat_examples(tokenizer, rows, last_only=not last_only)),
        )
    }
    assert all(both > last for both, last in zip(trained[False], trained[True], strict=True)), (
        "the flag changes nothing"
    )
    batch = _collate(tokenizer, _process(tokenizer, rows, last_only=last_only))
    assert sorted(_tree_examples(batch)) == sorted(flat)


def test_a_row_opening_on_the_response_marker_leaves_its_first_token_untrained_as_flat_packing_does(tokenizer):
    """A rendered row may begin with the assistant marker (a template with no system turn or BOS). The marker
    trains, but the flat packing collator never trains a document's first token: the tree row leaves it
    untrained too, and its root then collates."""
    input_ids = tokenizer.encode(RESPONSE + "Hello there.<|return|>", add_special_tokens=False)
    trained = _trained_tokens(
        input_ids,
        tokenizer=tokenizer,
        response_token_ids=tokenize_response_template(RESPONSE, tokenizer),
        eos_token_ids=resolve_eos_token_ids(tokenizer),
        assistant_message_template=RESPONSE,
        train_on_last_assistant_only=True,
    )
    flat = DataCollatorForCompletionOnlyLMWithPacking(
        response_prompt_template=RESPONSE, tokenizer=tokenizer, eos_token_ids=resolve_eos_token_ids(tokenizer)
    )([{"input_ids": input_ids, "seq_lengths": [len(input_ids)]}])["labels"][0]
    assert trained == (flat != LABEL_IGNORE_INDEX).tolist() and trained[1] and not trained[0]
    row = serialize_tree(build_trie([input_ids], [share_limit(trained)], 16), [input_ids], [trained])
    assert _tree_examples(_collate(tokenizer, [row])) == [(input_ids, flat.tolist())]


def _run_sft_main(tmp_path, yaml_body: str) -> None:
    """The SFT script's ``main`` over a tree-packing config, its log tee kept out of the test process."""
    config = tmp_path / "config.yaml"
    config.write_text(
        f"model_name_or_path: dummy/model\noutput_dir: {tmp_path / 'out'}\nbf16: false\nuse_cpu: true\n"
        "packing: true\nmax_length: 64\ntree_packing: true\ntree_share_markers: ['x']\n" + yaml_body
    )
    with mock.patch("src.entrypoints.script.install_log_tee"), mock.patch.object(sys, "argv", ["prog", str(config)]):
        sft_script.main()


@pytest.mark.parametrize(
    ("backend", "match"),
    [
        ("tree_full_attention: gather\n", "tree packing is verified on"),
        ("attn_implementation: flash_attention_2\ntree_full_attention: split\n", "tree_full_attention='split'"),
    ],
    ids=["auto-fa3", "split-on-fa2"],
)
def test_an_unsupported_backend_is_refused_before_any_weight_loads(tmp_path, monkeypatch, backend, match):
    """Auto-selection picks FA3 on Hopper, and the split form needs FA4: the run stops at the backend the loader
    would resolve for the config, dtype and GPU, with the message the loaded model's install gives, before the
    datasets or the weights load. Pinned to FA2 with the gather form, the same run gets past the check."""
    monkeypatch.setattr(selection, "_detect_attention_impl", lambda: "flash_attention_3")
    monkeypatch.setattr(sft_script, "resolve_training_dtype", lambda config: torch.bfloat16)
    monkeypatch.setattr(sft_script, "probe_checkpoint", lambda *args, **kwargs: (Qwen3Config(), False))
    monkeypatch.setattr(sft_script, "init_training_script", mock.MagicMock())
    monkeypatch.setattr(
        sft_script, "load_script_datasets", mock.Mock(side_effect=AssertionError("past the backend check"))
    )
    monkeypatch.setattr(sft_script, "load_policy_model", mock.Mock(side_effect=AssertionError("the model load ran")))
    with pytest.raises(ValueError, match=match):
        _run_sft_main(tmp_path, backend)
    with pytest.raises(AssertionError, match="past the backend check"):
        _run_sft_main(tmp_path, "attn_implementation: flash_attention_2\ntree_full_attention: gather\n")


def test_a_pad_valued_token_inside_an_answer_stays_untrained_as_the_packing_collator_leaves_it(tokenizer):
    """The LM collator masks every pad-valued label, the one inside a completion included; the span's
    closing token is the only one the packing collator restores."""
    pad = tokenizer.pad_token
    rows = [_question_row(f'{{\n  "1": "Question {p}."\n}}', f'{{"1": "no{pad}{p}"}}') for p in range(2)]
    flat = _flat_examples(tokenizer, rows)
    for tokens, labels in flat:
        inside = [i for i, token in enumerate(tokens) if token == tokenizer.pad_token_id]
        assert inside and all(labels[i] == LABEL_IGNORE_INDEX for i in inside), "the premise: flat leaves it untrained"
        assert labels[inside[0] - 1] != LABEL_IGNORE_INDEX and labels[inside[0] + 1] != LABEL_IGNORE_INDEX
    assert sorted(_tree_examples(_collate(tokenizer, _process(tokenizer, rows)))) == sorted(flat)


def test_rows_that_are_not_adjacent_do_not_share(tokenizer):
    """Sharing is decided per run of adjacent rows, which is why the dataset is ordered by key first."""
    rows = _conversations()
    scattered = [rows[0], rows[4], rows[1], rows[6], rows[2]]
    apart = sum(len(tree["input_ids"]) for tree in _process(tokenizer, scattered))
    assert apart == sum(len(tokens) for tokens, _ in _flat_examples(tokenizer, scattered))
    together = sum(len(tree["input_ids"]) for tree in _process(tokenizer, rows[:4]))
    assert together < sum(len(tokens) for tokens, _ in _flat_examples(tokenizer, rows[:4])) - 500


def test_two_questions_and_the_first_alone_share_a_second_level(tokenizer):
    trees = _process(tokenizer, _conversations()[:4])
    assert len(trees) == 1 and max(trees[0][DEPTH_COLUMN]) == 2
    unshared = _process(tokenizer, _conversations()[:4], min_shared_tokens=10**6)
    assert len(unshared) == 1 and max(unshared[0][DEPTH_COLUMN]) == 0
    assert unshared[0][POSITION_COLUMN].count(0) == 4


def test_a_row_over_the_length_budget_is_dropped_and_counted(tokenizer):
    rows = [*_conversations()[:2], _question_row("{" + " very long question text" * 400 + "}", "{}")]
    budget = len(_process(tokenizer, rows)[0]["input_ids"]) - 500
    out = _process(tokenizer, rows, max_length=budget)
    kept = [tree for tree in out if is_kept_tree(tree[DEPTH_COLUMN])]
    assert len(out) - len(kept) == 1
    assert len(_tree_examples(_collate(tokenizer, kept))) == 2
    assert all(len(tree["input_ids"]) <= budget for tree in kept)


def test_a_batch_of_plain_documents_carries_no_plan(tokenizer):
    trees = _process(tokenizer, [_conversations()[6], _conversations()[5]])
    batch = _collate(tokenizer, trees)
    assert TREE_ATTENTION_KEY not in batch
    assert batch[POSITION_COLUMN][0].tolist().count(0) == 2


def test_a_stateful_family_gets_its_segments_on_a_batch_of_plain_documents(tokenizer):
    """Attention isolates plain documents by their position resets; a convolution and a recurrence
    read their boundaries off the plan, so it comes with every batch."""
    trees = _process(tokenizer, [_conversations()[6], _conversations()[5]])
    collator = DataCollatorWithTreePacking(tokenizer, sliding_window=None, full_form="gather", conv_kernel=4)
    plan = collator(trees)[TREE_ATTENTION_KEY]
    assert "full" not in plan
    assert plan["levels"][0]["cu_seqlens"].tolist() == [
        0,
        len(trees[0]["input_ids"]),
        sum(len(t["input_ids"]) for t in trees),
    ]
    assert plan["conv"]["seq_idx"][0].unique().tolist() == [0, 1] and len(plan["levels"]) == 1


def test_the_collator_refuses_rows_it_cannot_keep_apart(tokenizer):
    trees = _process(tokenizer, _conversations()[:4])
    collator = DataCollatorWithTreePacking(tokenizer, sliding_window=WINDOW, full_form="split")
    collator.flatten_to_single_row = False
    with pytest.raises(ValueError, match="pipeline parallelism"):
        collator(trees[:1])


def test_the_collator_refuses_a_form_it_has_no_plan_for(tokenizer):
    with pytest.raises(ValueError, match="full_form must be one of"):
        DataCollatorWithTreePacking(tokenizer, sliding_window=WINDOW, full_form="dense")


def test_an_empty_split_is_refused_by_name(tokenizer):
    """A map over no rows returns them untouched, so the tree columns the later stages read would
    be missing; the split is named instead."""
    splits = DatasetDict(
        {
            "train": Dataset.from_dict({"conversation": _conversations()[:2]}),
            "test": Dataset.from_dict({"conversation": []}),
        }
    )
    with pytest.raises(ValueError, match="empty test split"):
        prepare_tree_dataset(
            splits,
            tokenizer,
            MAX_LENGTH,
            conversation_field="conversation",
            share_markers=MARKERS,
            share_requires=REQUIRES,
            num_proc=None,
        )


def test_the_collator_refuses_a_node_that_starts_on_a_trained_token(tokenizer):
    tree = dict(_process(tokenizer, _conversations()[:4])[0])
    tree[TRAINED_COLUMN] = list(tree[TRAINED_COLUMN])
    tree[TRAINED_COLUMN][tree[DEPTH_COLUMN].index(1)] = True
    with pytest.raises(ValueError, match="starts on a trained token"):
        _collate(tokenizer, [tree])


def test_a_truncated_tree_row_is_refused_at_collation(tokenizer):
    tree = _process(tokenizer, _conversations()[:4])[0]
    cut = tree[DEPTH_COLUMN].index(1) + 3
    tail = {column: values[cut:] for column, values in tree.items()}
    with pytest.raises(ValueError, match="not a depth-first tree layout"):
        _collate(tokenizer, [tail])


def _packed_examples(dataset, collator) -> list[tuple[list[int], list[int]]]:
    """(tokens, labels) of every example in a packed split, whichever way it was packed."""
    examples = []
    for row in dataset:
        batch = collator([row])
        if DEPTH_COLUMN in row:
            examples += _tree_examples({**batch, DEPTH_COLUMN: torch.tensor(row[DEPTH_COLUMN])})
            continue
        bounds = np.cumsum([0, *row["seq_lengths"]])
        examples += [
            (batch["input_ids"][0, a:b].tolist(), batch["labels"][0, a:b].tolist())
            for a, b in zip(bounds[:-1], bounds[1:], strict=True)
        ]
    return examples


def test_the_sft_data_path_packs_the_same_examples_as_trees_and_flat(tokenizer, tmp_path, monkeypatch):
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path)
    rows = []
    for conversation in range(12):
        document = DOCUMENT.replace("rain", f"rain in city {conversation}")
        rows += [
            _question_row(f'{{\n  "1": "Question {p}."\n}}', f'{{"1": "no {p}"}}', document=document) for p in range(5)
        ]
    rows += [
        _conversations()[5],
        _conversations()[6],
        _question_row("{" + " very long question text" * 2000 + "}", "{}"),
    ]
    random.Random(0).shuffle(rows)
    splits = DatasetDict(
        {
            "train": Dataset.from_dict({"conversation": rows[:48]}),
            "test": Dataset.from_dict({"conversation": rows[48:]}),
        }
    )
    hf_config = GptOssConfig()
    hf_config._attn_implementation = "flash_attention_4"

    reports = []
    monkeypatch.setattr(
        "src.data.pipeline.tree_rows.report_rejected_rows", lambda total, kept, context: reports.append((total, kept))
    )

    def prepare(tree_packing: bool, template: str = RESPONSE):
        args = SimpleNamespace(
            dataset=None,
            conversation_field="conversation",
            system_prompt="You answer questions about documents.",
            model_supports_system_role=False,
            interleaved_thinking=False,
            tools_field=None,
            train_on_completions_only=True,
            assistant_message_template=template,
            train_on_last_assistant_only=True,
            tree_packing=tree_packing,
            tree_share_markers=list(MARKERS),
            tree_share_requires=REQUIRES,
            tree_min_shared_tokens=16,
            tree_full_attention="split",
        )
        sft_config = SimpleNamespace(
            max_length=MAX_LENGTH,
            packing=True,
            eval_packing=True,
            packing_strategy="bfd",
            padding_free=False,
            dataset_num_proc=None,
            per_device_train_batch_size=1,
        )
        train, test, _, collator = _prepare_text_data(
            splits,
            False,
            args,
            sft_config,
            SimpleNamespace(model_name_or_path=GPT_OSS_20B),
            tokenizer,
            SimpleNamespace(is_cp_mode=False, cp_size=1, pp_size=1),
            hf_config=hf_config,
        )
        return train, test, collator

    flat_train, flat_test, flat_collator = prepare(tree_packing=False)
    tree_train, tree_test, tree_collator = prepare(tree_packing=True)
    assert isinstance(tree_collator, DataCollatorWithTreePacking)
    assert reports == [(63, 62)], "the over-length row is dropped and reported"
    plan = tree_collator([tree_train[0]])[TREE_ATTENTION_KEY]
    assert "sliding" in plan and "q_index" in plan["full"], "the collator was not built for this model and form"
    assert all(length > 0 for split in (tree_train, tree_test) for row in split for length in row["seq_lengths"])
    with pytest.raises(ValueError, match="does not occur in this tokenizer's rendered chat template"):
        prepare(tree_packing=True, template="<|start|>assistant<|message|>")
    for flat, tree in ((flat_train, tree_train), (flat_test, tree_test)):
        flat_examples = _packed_examples(flat, flat_collator)
        assert sorted(_packed_examples(tree, tree_collator)) == sorted(flat_examples)
        assert all(any(label != LABEL_IGNORE_INDEX for label in labels) for _, labels in flat_examples)
    # 63 rows, one over the length budget; the shuffled rows of a document found each other again.
    assert len(_packed_examples(tree_train, tree_collator)) + len(_packed_examples(tree_test, tree_collator)) == 62
    packed_tokens = [sum(sum(row["seq_lengths"]) for row in split) for split in (tree_train, flat_train)]
    assert packed_tokens[0] < 0.5 * packed_tokens[1]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
