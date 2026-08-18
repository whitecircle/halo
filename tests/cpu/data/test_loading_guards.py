#!/usr/bin/env python3
"""Load-time guards in ``src/data/sources/loading.py``.

Two guards pinned here:

* a **declared** render column (``conversation_field``) the dataset does not carry must not be
  accepted silently: ``_filter_empty_conversations`` no-ops on a missing column, so a typo surfaces
  much later as a bare ``KeyError`` inside an HF ``map`` worker — after the (possibly multi-node)
  model load and naming neither the knob nor the offending dataset. It fails loud at load, and *only*
  where the column is consumed: the callers that declare nothing (DPO/SMPO/KTO/reward/classification)
  and the pre-processed path (rows are already tokenized) must still load.
* ``test_size`` is inert for sharded and pre-processed loads, whose train/test split is fixed at
  preparation time. Exactly one warning must say so — and none when ``test_size`` is unset, which is
  the normal case.
* a corpus entry whose source ships no test split contributes training rows only: the placeholder
  test split is the last resort for a corpus with no held-out data anywhere, never a per-entry filler
  that dilutes a sibling pool's real one.

Run: pytest tests/cpu/data/test_loading_guards.py
"""

import json
import logging
import os
import shutil
import sys
import tempfile

import pytest
from accelerate import PartialState
from datasets import Dataset, DatasetDict

# The module logs through the accelerate logger, which requires an initialized state.
PartialState()

from src.data.pipeline.preprocessed_metadata import PreprocessedDatasetMetadata
from src.data.pipeline.preprocessing import shard_dataset
from src.data.shard_index import SHARD_INDEX_FILE
from src.data.sources.loading import alias_tools_column, load_datasets, load_datasets_auto
from src.data.sources.paths import METADATA_FILE

_LOADING_LOGGER = "src.data.sources.loading"

_CONVERSATION = [{"role": "user", "content": "hi"}]


def _conversation_dataset(num_rows: int, *, empty_rows: int = 0, field: str = "messages") -> Dataset:
    """A raw conversational split; ``empty_rows`` trailing rows carry an empty conversation."""
    rows = [_CONVERSATION] * (num_rows - empty_rows) + [[]] * empty_rows
    return Dataset.from_dict({field: rows, "row_id": list(range(num_rows))})


def _pool(num_rows: int, first_id: int) -> Dataset:
    """A raw conversational split whose ``row_id`` values identify the pool the rows came from."""
    return Dataset.from_dict(
        {"messages": [_CONVERSATION] * num_rows, "row_id": list(range(first_id, first_id + num_rows))}
    )


def _save_raw_dataset(directory: str, train: Dataset, test: Dataset) -> str:
    """Write a ``save_to_disk`` DatasetDict — the local raw (non-preprocessed) load path."""
    DatasetDict({"train": train, "test": test}).save_to_disk(directory)
    return directory


def _save_train_only_dataset(directory: str, train: Dataset) -> str:
    """Write a pool whose source ships no test split (``agentica-org/DeepCoder-Preview-Dataset``'s
    ``taco`` config, prepared as ``deepcoder-taco-rl``)."""
    DatasetDict({"train": train}).save_to_disk(directory)
    return directory


def _save_sharded_dataset(directory: str, train_rows: int = 20, test_rows: int = 6) -> str:
    """Write a sharded (prepared) dataset carrying both splits."""
    for split, num_rows in (("train", train_rows), ("test", test_rows)):
        dataset = Dataset.from_dict(
            {
                "input_ids": [[i, i + 1, i + 2] for i in range(num_rows)],
                "attention_mask": [[1, 1, 1]] * num_rows,
            }
        )
        index = shard_dataset(dataset, output_dir=directory, split_name=split, num_shards=2)
        index.save(os.path.join(directory, split, SHARD_INDEX_FILE))
    return directory


def _save_preprocessed_dataset(directory: str, train_rows: int = 8, test_rows: int = 4) -> str:
    """Write a non-sharded pre-processed dataset (tokenized rows + the ``preprocessed`` stamp)."""
    splits = {
        name: Dataset.from_dict(
            {
                "input_ids": [[i, i + 1] for i in range(num_rows)],
                "attention_mask": [[1, 1]] * num_rows,
                "labels": [[i, i + 1] for i in range(num_rows)],
            }
        )
        for name, num_rows in (("train", train_rows), ("test", test_rows))
    }
    DatasetDict(splits).save_to_disk(directory)
    PreprocessedDatasetMetadata(
        max_length=2,
        total_train_examples=train_rows,
        total_test_examples=test_rows,
    ).save(os.path.join(directory, METADATA_FILE))
    return directory


def _test_size_warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING and "test_size=" in r.getMessage()]


# --------------------------------------------------------------------------------------------- #
# Declared render column must exist
# --------------------------------------------------------------------------------------------- #


def test_declared_conversation_field_typo_fails_loud():
    """A misspelled conversation_field must raise at load, naming the knob, the dataset and the
    columns that are actually there — not KeyError inside a map worker after the model load."""
    temp_dir = tempfile.mkdtemp()
    try:
        _save_raw_dataset(temp_dir, _conversation_dataset(4), _conversation_dataset(2))
        with pytest.raises(ValueError, match="conversation_field='mesages'") as excinfo:
            load_datasets(temp_dir, test_size=None, dataset_ratio=1, conversation_field="mesages")
        message = str(excinfo.value)
        assert temp_dir in message, message
        assert "messages" in message, message  # the available columns are reported
    finally:
        shutil.rmtree(temp_dir)


def test_declared_tools_field_typo_fails_loud():
    """``tools_field`` is optional at render time, so a typo NEVER surfaces downstream — rows just
    render without tools and the run trains tool-calling data with none. The load must refuse it."""
    temp_dir = tempfile.mkdtemp()
    try:
        _save_raw_dataset(temp_dir, _conversation_dataset(4), _conversation_dataset(2))
        with pytest.raises(ValueError, match="tools_field='tols'"):
            load_datasets(temp_dir, test_size=None, dataset_ratio=1, conversation_field="messages", tools_field="tols")
    finally:
        shutil.rmtree(temp_dir)


def test_declared_tools_field_present_loads():
    """Anti-over-rejection: a tools column that exists loads; unset tools_field checks nothing."""
    temp_dir = tempfile.mkdtemp()
    try:
        train = _conversation_dataset(4).add_column("tools", [[]] * 4)
        test = _conversation_dataset(2).add_column("tools", [[]] * 2)
        _save_raw_dataset(temp_dir, train, test)
        ds = load_datasets(
            temp_dir, test_size=None, dataset_ratio=1, conversation_field="messages", tools_field="tools"
        )
        assert len(ds["train"]) == 4
        ds = load_datasets(temp_dir, test_size=None, dataset_ratio=1, conversation_field="messages")
        assert len(ds["train"]) == 4
    finally:
        shutil.rmtree(temp_dir)


def test_tools_column_is_aliased_onto_the_spelling_the_reward_trainer_renders():
    """TRL's RewardTrainer reads the column literally named ``tools``: a declared column under any
    other name is renamed in every split, the canonical name passes through, and a dataset that
    already carries ``tools`` beside the declared column is refused rather than overwritten."""
    train = _conversation_dataset(4).add_column("fn_schemas", [[]] * 4)
    test = _conversation_dataset(2).add_column("fn_schemas", [[]] * 2)
    aliased = alias_tools_column(DatasetDict({"train": train, "test": test}), "fn_schemas", "ds")
    assert all("tools" in split.column_names and "fn_schemas" not in split.column_names for split in aliased.values())

    assert alias_tools_column(aliased, "tools", "ds") is aliased

    with pytest.raises(ValueError, match="tools_field='fn_schemas'"):
        alias_tools_column(
            DatasetDict({"train": train.add_column("tools", [[]] * 4), "test": test}), "fn_schemas", "ds"
        )
    with pytest.raises(ValueError, match="tools_field='tols'"):
        alias_tools_column(DatasetDict({"train": train, "test": test}), "tols", "ds")


def test_declared_conversation_field_present_loads_and_still_filters():
    """Anti-over-rejection, plus the filter the guard sits in front of: a correct column loads, and
    rows whose conversation is empty are still dropped."""
    temp_dir = tempfile.mkdtemp()
    try:
        _save_raw_dataset(temp_dir, _conversation_dataset(4, empty_rows=1), _conversation_dataset(2))
        ds = load_datasets(temp_dir, test_size=None, dataset_ratio=1, conversation_field="messages")
        assert len(ds["train"]) == 3, "the empty-conversation row must still be filtered out"
        assert len(ds["test"]) == 2
    finally:
        shutil.rmtree(temp_dir)


def test_extra_split_without_the_column_is_not_rejected():
    """Anti-over-rejection: the loader renders train and test only, so a `validation` split a source
    happens to carry (with its own schema) is not a render contract and must not fail the load."""
    temp_dir = tempfile.mkdtemp()
    try:
        DatasetDict(
            {
                "train": _conversation_dataset(4),
                "test": _conversation_dataset(2),
                "validation": Dataset.from_dict({"something_else": ["x"]}),
            }
        ).save_to_disk(temp_dir)
        ds = load_datasets(temp_dir, test_size=None, dataset_ratio=1, conversation_field="messages")
        assert len(ds["train"]) == 4
    finally:
        shutil.rmtree(temp_dir)


def test_conversation_field_none_is_unaffected():
    """RLVR / embedding pass conversation_field=None: nothing is declared, nothing is checked."""
    temp_dir = tempfile.mkdtemp()
    try:
        _save_raw_dataset(temp_dir, _conversation_dataset(4, field="prompt"), _conversation_dataset(2, field="prompt"))
        ds = load_datasets(temp_dir, test_size=None, dataset_ratio=1, conversation_field=None)
        assert len(ds["train"]) == 4
    finally:
        shutil.rmtree(temp_dir)


def test_undeclared_conversation_field_does_not_reject_preference_data():
    """The DPO/SMPO/KTO/reward/classification scripts declare no render column at all and their rows
    carry none — the historical "conversation" fallback stays an opportunistic filter for them.

    Holding an undeclared fallback to the existence check would reject every one of those runs.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        preference = Dataset.from_dict(
            {
                "prompt": [_CONVERSATION] * 4,
                "chosen": [_CONVERSATION] * 4,
                "rejected": [_CONVERSATION] * 4,
            }
        )
        _save_raw_dataset(temp_dir, preference, preference)
        ds = load_datasets(temp_dir, test_size=None, dataset_ratio=1)
        assert len(ds["train"]) == 4
    finally:
        shutil.rmtree(temp_dir)


def test_dataset_list_names_the_offending_entry():
    """With a list path the concatenation would drop the column entirely; the raise must identify
    WHICH entry lacks it, not just report that the column went missing."""
    good_dir = tempfile.mkdtemp()
    bad_dir = tempfile.mkdtemp()
    try:
        _save_raw_dataset(good_dir, _conversation_dataset(4), _conversation_dataset(2))
        _save_raw_dataset(bad_dir, _conversation_dataset(4, field="prompt"), _conversation_dataset(2, field="prompt"))
        with pytest.raises(ValueError, match="conversation_field='messages'") as excinfo:
            load_datasets([good_dir, bad_dir], test_size=None, dataset_ratio=1, conversation_field="messages")
        message = str(excinfo.value)
        assert bad_dir in message, message
        assert good_dir not in message, message
    finally:
        shutil.rmtree(good_dir)
        shutil.rmtree(bad_dir)


def test_mixed_corpus_keeps_the_tools_column_through_the_concatenation():
    """The mixed-corpus shape the tools warning promises: only the rows of the source WITHOUT the
    column render toolless.

    The schema intersection dropped a column missing from any entry, so the whole corpus — the
    tool-use half included — rendered without tools, and nothing downstream ever said so
    (``tools_field`` is optional at render time).
    """
    with_tools = tempfile.mkdtemp()
    without_tools = tempfile.mkdtemp()
    try:
        tools = [{"name": "search", "description": "the web"}]
        _save_raw_dataset(
            with_tools,
            _conversation_dataset(4).add_column("tools", [tools] * 4),
            _conversation_dataset(2).add_column("tools", [tools] * 2),
        )
        _save_raw_dataset(without_tools, _conversation_dataset(4), _conversation_dataset(2))

        ds = load_datasets(
            [with_tools, without_tools],
            test_size=None,
            dataset_ratio=1,
            conversation_field="messages",
            tools_field="tools",
        )

        for split in ("train", "test"):
            assert "tools" in ds[split].column_names, f"the {split} split lost the declared tools column"
        carried = ds["train"]["tools"]
        assert carried.count(None) == 4, carried
        assert [row for row in carried if row is not None] == [tools] * 4, carried
    finally:
        shutil.rmtree(with_tools)
        shutil.rmtree(without_tools)


def test_tools_column_lost_to_a_type_mismatch_fails_loud():
    """A declared column no null fill can reconcile (list-of-struct in one entry, a JSON string in
    the other) is dropped by the intersection — and that must raise, not train toolless in silence."""
    struct_tools = tempfile.mkdtemp()
    string_tools = tempfile.mkdtemp()
    try:
        _save_raw_dataset(
            struct_tools,
            _conversation_dataset(4).add_column("tools", [[{"name": "search"}]] * 4),
            _conversation_dataset(2).add_column("tools", [[{"name": "search"}]] * 2),
        )
        _save_raw_dataset(
            string_tools,
            _conversation_dataset(4).add_column("tools", ['[{"name": "search"}]'] * 4),
            _conversation_dataset(2).add_column("tools", ['[{"name": "search"}]'] * 2),
        )
        with pytest.raises(ValueError, match="tools_field='tools'"):
            load_datasets(
                [struct_tools, string_tools],
                test_size=None,
                dataset_ratio=1,
                conversation_field="messages",
                tools_field="tools",
            )
    finally:
        shutil.rmtree(struct_tools)
        shutil.rmtree(string_tools)


def test_preprocessed_load_ignores_the_render_column():
    """Pre-processed rows are already tokenized — the render column is never consumed there, so a
    declared conversation_field must NOT be required of them."""
    temp_dir = tempfile.mkdtemp()
    try:
        _save_preprocessed_dataset(temp_dir)
        ds, is_preprocessed = load_datasets_auto(
            temp_dir, test_size=None, dataset_ratio=1, conversation_field="messages"
        )
        assert is_preprocessed
        assert len(ds["train"]) == 8
    finally:
        shutil.rmtree(temp_dir)


# --------------------------------------------------------------------------------------------- #
# test_size cannot re-split an already-prepared dataset
# --------------------------------------------------------------------------------------------- #


def test_test_size_warns_once_for_a_sharded_load(caplog):
    """A sharded dataset's split was fixed at preparation time, so a YAML test_size does nothing.
    Exactly one warning must name the cause and the re-prepare fix."""
    temp_dir = tempfile.mkdtemp()
    try:
        _save_sharded_dataset(temp_dir)
        with caplog.at_level(logging.WARNING, logger=_LOADING_LOGGER):
            ds = load_datasets(temp_dir, test_size=0.5, dataset_ratio=1, conversation_field=None)

        warnings = _test_size_warnings(caplog)
        assert len(warnings) == 1, warnings
        assert "prepare_dataset.py --test-size" in warnings[0], warnings[0]
        # And the reason the warning exists: the prepared split is what actually loaded.
        assert (len(ds["train"]), len(ds["test"])) == (20, 6)
    finally:
        shutil.rmtree(temp_dir)


def test_no_test_size_warning_when_unset(caplog):
    """Anti-over-rejection: the normal run leaves test_size unset, and a warning that fires every
    time only trains operators to ignore this module's warnings."""
    temp_dir = tempfile.mkdtemp()
    try:
        _save_sharded_dataset(temp_dir)
        with caplog.at_level(logging.WARNING, logger=_LOADING_LOGGER):
            load_datasets(temp_dir, test_size=None, dataset_ratio=1, conversation_field=None)

        assert not _test_size_warnings(caplog), _test_size_warnings(caplog)
    finally:
        shutil.rmtree(temp_dir)


def test_test_size_warns_once_for_a_preprocessed_load(caplog):
    """The pre-processed branch never even receives test_size (``load_preprocessed_dataset`` takes
    none), so the auto entry point is the only place that can report it as ignored."""
    temp_dir = tempfile.mkdtemp()
    try:
        _save_preprocessed_dataset(temp_dir)
        with caplog.at_level(logging.WARNING, logger=_LOADING_LOGGER):
            ds, is_preprocessed = load_datasets_auto(temp_dir, test_size=0.25, dataset_ratio=1)

        warnings = _test_size_warnings(caplog)
        assert len(warnings) == 1, warnings
        assert is_preprocessed
        assert (len(ds["train"]), len(ds["test"])) == (8, 4)
    finally:
        shutil.rmtree(temp_dir)


def test_test_size_still_splits_a_plain_dataset(caplog):
    """Anti-over-rejection: on the raw non-sharded path test_size is honored, so nothing warns."""
    temp_dir = tempfile.mkdtemp()
    try:
        DatasetDict({"train": _conversation_dataset(10)}).save_to_disk(temp_dir)
        with caplog.at_level(logging.WARNING, logger=_LOADING_LOGGER):
            ds = load_datasets(temp_dir, test_size=0.2, dataset_ratio=1, conversation_field="messages")

        assert not _test_size_warnings(caplog), _test_size_warnings(caplog)
        assert (len(ds["train"]), len(ds["test"])) == (8, 2)
    finally:
        shutil.rmtree(temp_dir)


def test_preprocessed_metadata_fixture_is_detected():
    """Guards the two pre-processed tests above: a fixture that stopped being detected as
    pre-processed would send them down the raw path and certify nothing."""
    temp_dir = tempfile.mkdtemp()
    try:
        _save_preprocessed_dataset(temp_dir)
        with open(os.path.join(temp_dir, METADATA_FILE)) as f:
            assert json.load(f)["preprocessed"] is True
        _, is_preprocessed = load_datasets_auto(temp_dir, test_size=None, dataset_ratio=1)
        assert is_preprocessed
    finally:
        shutil.rmtree(temp_dir)


def test_train_only_list_entry_contributes_no_rows_to_the_held_out_split(caplog):
    """A pool whose source ships no test split must contribute TRAINING rows only.

    Filling its missing split with the first 100 of its own train rows — the last resort a
    single-path load takes so eval machinery has something — silently seeds a corpus whose sibling
    pool DOES ship a held-out split with rows the model trains on, and the concatenated split then
    reports a size that hides the contamination.
    """
    with_test = tempfile.mkdtemp()
    train_only = tempfile.mkdtemp()
    try:
        _save_raw_dataset(with_test, _pool(6, 0), _pool(3, 100))
        _save_train_only_dataset(train_only, _pool(5, 200))

        with caplog.at_level(logging.WARNING, logger=_LOADING_LOGGER):
            ds = load_datasets([with_test, train_only], test_size=None, dataset_ratio=1, conversation_field="messages")

        assert sorted(ds["test"]["row_id"]) == [100, 101, 102], ds["test"]["row_id"]
        assert sorted(ds["train"]["row_id"]) == [0, 1, 2, 3, 4, 5, 200, 201, 202, 203, 204]
        warned = [r.getMessage() for r in caplog.records if train_only in r.getMessage()]
        assert len(warned) == 1, [r.getMessage() for r in caplog.records]
        assert "no test split" in warned[0], warned[0]
    finally:
        shutil.rmtree(with_test)
        shutil.rmtree(train_only)


def test_corpus_with_no_test_split_anywhere_keeps_the_placeholder_fallback():
    """With no held-out split anywhere there is nothing to protect, so the placeholder still applies
    — once, over the concatenated train — because every consumer indexes ``ds["test"]``.

    Per entry it would instead take the first 100 rows of EACH pool, so the fallback's size is what
    separates the corpus-wide last resort from the per-entry filler.
    """
    first = tempfile.mkdtemp()
    second = tempfile.mkdtemp()
    try:
        _save_train_only_dataset(first, _pool(150, 0))
        _save_train_only_dataset(second, _pool(60, 1000))

        ds = load_datasets([first, second], test_size=None, dataset_ratio=1, conversation_field="messages")

        assert len(ds["train"]) == 210
        assert sorted(ds["test"]["row_id"]) == list(range(100)), len(ds["test"])
    finally:
        shutil.rmtree(first)
        shutil.rmtree(second)


def test_test_size_carves_a_held_out_split_from_a_train_only_entry():
    """The remedy the warning names: ``test_size`` splits every entry, so a train-only pool
    contributes genuinely held-out rows rather than none."""
    with_test = tempfile.mkdtemp()
    train_only = tempfile.mkdtemp()
    try:
        _save_raw_dataset(with_test, _pool(6, 0), _pool(2, 100))
        _save_train_only_dataset(train_only, _pool(6, 200))

        ds = load_datasets([with_test, train_only], test_size=0.5, dataset_ratio=1, conversation_field="messages")

        assert not set(ds["test"]["row_id"]) & set(ds["train"]["row_id"])
        assert any(row_id >= 200 for row_id in ds["test"]["row_id"]), ds["test"]["row_id"]
    finally:
        shutil.rmtree(with_test)
        shutil.rmtree(train_only)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
