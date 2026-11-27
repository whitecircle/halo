#!/usr/bin/env python3
"""Load-time guards in ``halo/data/sources/loading.py``: a declared render column must exist, ``test_size`` warns
where inert, a train-only corpus entry adds no held-out rows, and ``validation`` serves as the held-out split."""

import logging
import os
import shutil
import tempfile
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
from datasets import Dataset, DatasetDict

from halo.data.conversation import TEMPLATE_COLUMN_KNOBS
from halo.data.preprocessed.metadata import PreprocessedDatasetMetadata
from halo.data.preprocessed.preparation import shard_dataset
from halo.data.preprocessed.shard_index import SHARD_INDEX_FILE, write_stamped_sidecar
from halo.data.render_columns import alias_tools_column
from halo.data.sources.loading import load_dataset_from_source, load_datasets, load_datasets_auto
from halo.data.sources.paths import METADATA_FILE
from halo.entrypoints.data_setup import load_script_datasets

_LOADING_LOGGER = "halo.data.sources.loading"

_CONVERSATION = [{"role": "user", "content": "hi"}]


def _conversation_dataset(num_rows: int, *, empty_rows: int = 0, field: str = "messages") -> Dataset:
    rows = [_CONVERSATION] * (num_rows - empty_rows) + [[]] * empty_rows
    return Dataset.from_dict({field: rows, "row_id": list(range(num_rows))})


def _pool(num_rows: int, first_id: int) -> Dataset:
    """A raw conversational split whose ``row_id`` values identify the pool the rows came from."""
    return Dataset.from_dict(
        {"messages": [_CONVERSATION] * num_rows, "row_id": list(range(first_id, first_id + num_rows))}
    )


def _save_raw_dataset(directory: str, train: Dataset, test: Dataset) -> str:
    DatasetDict({"train": train, "test": test}).save_to_disk(directory)
    return directory


def _save_train_only_dataset(directory: str, train: Dataset) -> str:
    """Write a pool whose source ships no test split (``agentica-org/DeepCoder-Preview-Dataset``'s
    ``taco`` config, prepared as ``deepcoder-taco-rl``)."""
    DatasetDict({"train": train}).save_to_disk(directory)
    return directory


def _save_sharded_dataset(directory: str, train_rows: int = 20, test_rows: int = 6) -> str:
    for split, num_rows in (("train", train_rows), ("test", test_rows)):
        dataset = Dataset.from_dict(
            {
                "input_ids": [[i, i + 1, i + 2] for i in range(num_rows)],
                "attention_mask": [[1, 1, 1]] * num_rows,
            }
        )
        index = shard_dataset(dataset, output_dir=directory, split_name=split, num_shards=2)
        write_stamped_sidecar(os.path.join(directory, split, SHARD_INDEX_FILE), index)
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
    write_stamped_sidecar(
        os.path.join(directory, METADATA_FILE),
        PreprocessedDatasetMetadata(max_length=2, total_train_examples=train_rows, total_test_examples=test_rows),
    )
    return directory


def _test_size_warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING and "test_size=" in r.getMessage()]


def test_declared_conversation_field_typo_fails_loud():
    """A misspelled column raises at load, naming the knob, the dataset and the columns present; it would otherwise
    be a KeyError inside a map worker after the model load."""
    temp_dir = tempfile.mkdtemp()
    try:
        _save_raw_dataset(temp_dir, _conversation_dataset(4), _conversation_dataset(2))
        with pytest.raises(ValueError, match="conversation_field='mesages'") as excinfo:
            load_datasets(temp_dir, test_size=None, dataset_ratio=1, conversation_field="mesages")
        message = str(excinfo.value)
        assert temp_dir in message, message
        assert "messages" in message, message
    finally:
        shutil.rmtree(temp_dir)


def test_the_refusal_names_the_knob_the_caller_read_the_column_from(tmp_path):
    """RLVR and env GRPO declare ``prompt_field``, so telling them to set ``conversation_field`` points at a knob
    they lack."""
    source = _save_raw_dataset(str(tmp_path / "raw"), _conversation_dataset(4), _conversation_dataset(2))
    for path in (source, [source]):
        with pytest.raises(ValueError, match=r"^prompt_field='question' .*Point prompt_field at") as excinfo:
            load_datasets(
                path, test_size=None, dataset_ratio=1, conversation_field="question", conversation_knob="prompt_field"
            )
        assert "conversation_field" not in str(excinfo.value), excinfo.value


@pytest.mark.parametrize("knob", TEMPLATE_COLUMN_KNOBS)
def test_a_declared_template_column_typo_fails_loud(tmp_path, knob):
    """Optional at render time, so a typo never surfaces downstream: every row would train without its tools or its
    level."""
    source = _save_raw_dataset(
        str(tmp_path / "raw"),
        _conversation_dataset(4).add_column("column", ["x"] * 4),
        _conversation_dataset(2).add_column("column", ["y"] * 2),
    )
    load = partial(load_datasets, source, test_size=None, dataset_ratio=1, conversation_field="messages")
    with pytest.raises(ValueError, match=rf"{knob}='colum' names a column"):
        load(template_columns={knob: "colum"})
    assert load(template_columns={knob: "column"})["train"]["column"] == ["x"] * 4


def test_load_script_datasets_hands_the_loader_every_template_column():
    """Each script's loader is where a declared column is checked, so one the call leaves out goes unchecked."""
    seen = {}

    def loader(*args, **kwargs):
        seen.update(kwargs)
        return "loaded"

    columns = {knob: f"{knob}_column" for knob in TEMPLATE_COLUMN_KNOBS}
    args = SimpleNamespace(dataset="/data/raw", test_size=None, dataset_ratio=None, **columns)
    parallelism = SimpleNamespace(get_data_parallel_rank=lambda: 0, data_parallel_size=1)
    assert load_script_datasets(args, parallelism, loader=loader) == ("loaded", False)
    assert seen["template_columns"] == columns


def test_tools_column_is_aliased_onto_the_spelling_the_reward_trainer_renders():
    """TRL's RewardTrainer reads the column literally named ``tools``; a dataset already carrying ``tools`` beside
    the declared column is refused."""
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
    """A correct column loads, and rows whose conversation is empty are still dropped."""
    temp_dir = tempfile.mkdtemp()
    try:
        _save_raw_dataset(temp_dir, _conversation_dataset(4, empty_rows=1), _conversation_dataset(2))
        ds = load_datasets(temp_dir, test_size=None, dataset_ratio=1, conversation_field="messages")
        assert len(ds["train"]) == 3, "the empty-conversation row must still be filtered out"
        assert len(ds["test"]) == 2
    finally:
        shutil.rmtree(temp_dir)


def test_extra_split_without_the_column_is_not_rejected():
    """The loader renders train and test only, so a ``validation`` split with its own schema is no render contract."""
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
    """These scripts declare no render column, and the conversation fallback stays an opportunistic filter; an
    existence check would reject every such run."""
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
    """The concatenation would drop the column entirely, so the raise identifies which entry lacks it."""
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
    """A schema intersection would drop a column missing from any entry, so the tool-use half would render without
    tools and nothing downstream would say so."""
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
            template_columns={"tools_field": "tools"},
        )

        for split in ("train", "test"):
            assert "tools" in ds[split].column_names, f"the {split} split lost the declared tools column"
        carried = ds["train"]["tools"]
        assert carried.count(None) == 4, carried
        assert [row for row in carried if row is not None] == [tools] * 4, carried
    finally:
        shutil.rmtree(with_tools)
        shutil.rmtree(without_tools)


@pytest.mark.parametrize("knob", TEMPLATE_COLUMN_KNOBS)
def test_a_corpus_checks_each_template_column_across_its_entries(tmp_path, caplog, knob):
    """A column no entry carries is a typo, refused; one an entry lacks leaves that entry's rows rendering as if the
    knob were unset, which the load says, and one every entry carries loads without a word."""
    carrying = _save_raw_dataset(
        str(tmp_path / "carrying"),
        _conversation_dataset(4).add_column("extra", ["x"] * 4),
        _conversation_dataset(2).add_column("extra", ["x"] * 2),
    )
    plain = _save_raw_dataset(str(tmp_path / "plain"), _conversation_dataset(4), _conversation_dataset(2))
    load = partial(load_datasets, test_size=None, dataset_ratio=1, conversation_field="messages")

    with pytest.raises(ValueError, match=rf"{knob}='missing' names a column NONE of the 2"):
        load([carrying, plain], template_columns={knob: "missing"})
    with caplog.at_level(logging.WARNING, logger="halo.data.sources.corpus"):
        load([carrying, carrying], template_columns={knob: "extra"})
        assert not [record for record in caplog.records if knob in record.getMessage()], caplog.records
        load([carrying, plain], template_columns={knob: "extra"})
    absent = {plain: ["test", "train"]}
    assert any(f"{knob}='extra' is absent from {absent}" in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize("knob", TEMPLATE_COLUMN_KNOBS)
def test_a_template_column_one_entry_carries_in_train_only_is_no_typo(tmp_path, caplog, knob):
    """The column exists, so the corpus refuses nothing whatever the other entries carry: the splits lacking it render
    as if the knob were unset, which the load names split by split."""
    train_only = _save_raw_dataset(
        str(tmp_path / "train_only"), _conversation_dataset(4).add_column("extra", ["x"] * 4), _conversation_dataset(2)
    )
    plain = _save_raw_dataset(str(tmp_path / "plain"), _conversation_dataset(4), _conversation_dataset(2))

    with caplog.at_level(logging.WARNING, logger="halo.data.sources.corpus"):
        ds = load_datasets(
            [train_only, plain],
            test_size=None,
            dataset_ratio=1,
            conversation_field="messages",
            template_columns={knob: "extra"},
        )

    assert ds["train"]["extra"] == ["x"] * 4 + [None] * 4
    assert ds["test"]["extra"] == [None] * 4
    absent = {train_only: ["test"], plain: ["test", "train"]}
    assert any(f"{knob}='extra' is absent from {absent}" in record.getMessage() for record in caplog.records), [
        record.getMessage() for record in caplog.records
    ]


@pytest.mark.parametrize("knob", TEMPLATE_COLUMN_KNOBS)
def test_a_pinned_template_column_keys_the_loaded_fingerprint(tmp_path, knob):
    """Pinned, the column survives the concatenation; unpinned, the schema intersection drops it. A map keyed by the
    loaded fingerprint (TRL's own) would otherwise serve one load's rows to the other."""
    carrying = _save_raw_dataset(
        str(tmp_path / "carrying"),
        _conversation_dataset(4).add_column("extra", ["x"] * 4),
        _conversation_dataset(2).add_column("extra", ["x"] * 2),
    )
    plain = _save_raw_dataset(str(tmp_path / "plain"), _conversation_dataset(4), _conversation_dataset(2))
    load = partial(load_datasets, [carrying, plain], test_size=None, dataset_ratio=1, conversation_field="messages")

    pinned = load(template_columns={knob: "extra"})
    unpinned = load()
    pinned_again = load(template_columns={knob: "extra"})

    assert "extra" in pinned["train"].column_names and "extra" not in unpinned["train"].column_names
    for split in ("train", "test"):
        assert pinned[split]._fingerprint != unpinned[split]._fingerprint, split
        assert pinned[split]._fingerprint == pinned_again[split]._fingerprint, split


def test_a_tools_column_typed_differently_across_sources_is_refused():
    """Two sources typing a declared column differently leave no null fill to reconcile, so the load refuses the
    mismatch instead of dropping the column from every row."""
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
        with pytest.raises(ValueError, match="tools_field='tools' is a type mismatch") as excinfo:
            load_datasets(
                [struct_tools, string_tools],
                test_size=None,
                dataset_ratio=1,
                conversation_field="messages",
                template_columns={"tools_field": "tools"},
            )
        held = str(excinfo.value)
        assert f"{struct_tools} (train)" in held and f"{string_tools} (test)" in held, held
    finally:
        shutil.rmtree(struct_tools)
        shutil.rmtree(string_tools)


@pytest.mark.parametrize("unlabelled_first", [True, False], ids=["unlabelled-source-first", "labelled-source-first"])
@pytest.mark.parametrize("null_column", [False, True], ids=["column-absent", "column-all-null"])
def test_a_mixed_corpus_keeps_every_level_through_the_concatenation(tmp_path, null_column, unlabelled_first):
    """A source collected without levels renders its rows with none, whether it lacks the column or holds it as nulls,
    the no-level cell, which Arrow types ``null``: beside a typed source the schema intersection would drop either."""
    labelled = _save_raw_dataset(
        str(tmp_path / "labelled"),
        _conversation_dataset(4).add_column("effort", ["low", "medium", "high", "low"]),
        _conversation_dataset(2).add_column("effort", ["high"] * 2),
    )

    def unlabelled_split(rows: int) -> Dataset:
        split = _conversation_dataset(rows)
        return split.add_column("effort", [None] * rows) if null_column else split

    unlabelled = _save_raw_dataset(str(tmp_path / "unlabelled"), unlabelled_split(4), unlabelled_split(2))

    ds = load_datasets(
        [unlabelled, labelled] if unlabelled_first else [labelled, unlabelled],
        test_size=None,
        dataset_ratio=1,
        conversation_field="messages",
        template_columns={"reasoning_effort_field": "effort"},
    )

    carried = ds["train"]["effort"]
    assert carried.count(None) == 4, carried
    assert sorted(level for level in carried if level is not None) == ["high", "low", "low", "medium"], carried


def test_preprocessed_load_ignores_the_render_column():
    """Pre-processed rows are already tokenized, so the render column is never consumed."""
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


def test_test_size_warns_once_for_a_sharded_load(caplog):
    """The split was fixed at preparation time, so a YAML test_size does nothing."""
    temp_dir = tempfile.mkdtemp()
    try:
        _save_sharded_dataset(temp_dir)
        with caplog.at_level(logging.WARNING, logger=_LOADING_LOGGER):
            ds = load_datasets(temp_dir, test_size=0.5, dataset_ratio=1, conversation_field=None)

        warnings = _test_size_warnings(caplog)
        assert len(warnings) == 1, warnings
        assert "prepare_dataset.py --test_size" in warnings[0], warnings[0]
        # And the reason the warning exists: the prepared split is what actually loaded.
        assert (len(ds["train"]), len(ds["test"])) == (20, 6)
    finally:
        shutil.rmtree(temp_dir)


def test_no_test_size_warning_when_unset(caplog):
    """A warning firing every run only trains operators to ignore this module's warnings."""
    temp_dir = tempfile.mkdtemp()
    try:
        _save_sharded_dataset(temp_dir)
        with caplog.at_level(logging.WARNING, logger=_LOADING_LOGGER):
            load_datasets(temp_dir, test_size=None, dataset_ratio=1, conversation_field=None)

        assert not _test_size_warnings(caplog), _test_size_warnings(caplog)
    finally:
        shutil.rmtree(temp_dir)


def test_test_size_warns_once_for_a_preprocessed_load(caplog):
    """``load_preprocessed_dataset`` takes no test_size, so the auto entry point is the only place to report it as
    ignored."""
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


def test_a_sub_one_ratio_is_refused_on_a_preprocessed_dataset(tmp_path):
    """The rows were fixed at preparation, so the run would otherwise train every row the YAML says to drop."""
    source = _save_preprocessed_dataset(str(tmp_path / "prepared"))
    with pytest.raises(ValueError, match=r"dataset_ratio=0\.1 .*pre-processed"):
        load_datasets_auto(source, test_size=None, dataset_ratio=0.1)


def test_a_ratio_list_on_a_preprocessed_string_path_takes_the_raw_path_refusal(tmp_path):
    source = _save_preprocessed_dataset(str(tmp_path / "prepared"))
    with pytest.raises(ValueError, match="all of your dataset ratios"):
        load_datasets_auto(source, test_size=None, dataset_ratio=[0.1])


@pytest.mark.parametrize("ratio", [None, 1.0])
def test_a_ratio_keeping_every_row_loads_a_preprocessed_dataset(tmp_path, ratio):
    source = _save_preprocessed_dataset(str(tmp_path / "prepared"))
    ds, is_preprocessed = load_datasets_auto(source, test_size=None, dataset_ratio=ratio)
    assert is_preprocessed
    assert (len(ds["train"]), len(ds["test"])) == (8, 4)


def test_a_file_uri_reads_a_preprocessed_dataset_as_its_plain_path_does(tmp_path):
    source = _save_preprocessed_dataset(str(tmp_path / "prepared"))
    ds, is_preprocessed = load_datasets_auto(f"file://{source}", test_size=None, dataset_ratio=None)
    assert is_preprocessed
    assert (len(ds["train"]), len(ds["test"])) == (8, 4)


def test_a_file_uri_reads_a_sharded_dataset_as_its_plain_path_does(tmp_path):
    source = _save_sharded_dataset(str(tmp_path / "sharded"))
    ds, is_preprocessed = load_datasets_auto(f"file://{source}", test_size=None, dataset_ratio=None)
    assert not is_preprocessed
    assert (len(ds["train"]), len(ds["test"])) == (20, 6)
    assert sorted(ds["train"]["input_ids"][0]) == [0, 1, 2]


def test_a_file_uri_reads_a_raw_directory_and_a_data_file_as_their_plain_paths_do(tmp_path):
    directory = _save_raw_dataset(str(tmp_path / "raw"), _conversation_dataset(4), _conversation_dataset(2))
    data_file = tmp_path / "rows.jsonl"
    data_file.write_text('{"row_id": 7}\n')

    assert load_dataset_from_source(f"file://{directory}")["train"]["row_id"] == list(range(4))
    assert load_dataset_from_source(f"file://{data_file}")["train"]["row_id"] == [7]


def test_test_size_still_splits_a_plain_dataset(caplog):
    temp_dir = tempfile.mkdtemp()
    try:
        DatasetDict({"train": _conversation_dataset(10)}).save_to_disk(temp_dir)
        with caplog.at_level(logging.WARNING, logger=_LOADING_LOGGER):
            ds = load_datasets(temp_dir, test_size=0.2, dataset_ratio=1, conversation_field="messages")

        assert not _test_size_warnings(caplog), _test_size_warnings(caplog)
        assert (len(ds["train"]), len(ds["test"])) == (8, 2)
    finally:
        shutil.rmtree(temp_dir)


@pytest.mark.parametrize("layout", ["bare", "train_only", "train_and_test"])
def test_test_size_writes_its_split_under_the_datasets_cache(tmp_path, monkeypatch, layout):
    """``datasets`` writes a split's index files beside the source's Arrow files: a read-only input fails, and every
    node's load rank writes into a shared one."""
    cache = tmp_path / "datasets_cache"
    monkeypatch.setenv("HF_DATASETS_CACHE", str(cache))
    source = str(tmp_path / "source")
    if layout == "bare":
        _conversation_dataset(10).save_to_disk(source)
    elif layout == "train_only":
        _save_train_only_dataset(source, _conversation_dataset(10))
    else:
        _save_raw_dataset(source, _conversation_dataset(6), _conversation_dataset(4))
    before = sorted(str(p.relative_to(source)) for p in Path(source).rglob("*"))

    ds = load_datasets(source, test_size=0.2, dataset_ratio=1, conversation_field="messages")

    assert (len(ds["train"]), len(ds["test"])) == (8, 2)
    assert sorted(str(p.relative_to(source)) for p in Path(source).rglob("*")) == before
    assert list(cache.glob("split-*.arrow")), "the split's index went nowhere under the datasets cache"


def test_a_directory_mixing_two_bakes_is_refused(tmp_path):
    """A copy that adds files without removing the old ones leaves a sharded bake's index beside an unsharded bake's
    rows and stamp, and the index decides the load."""
    mixed = str(tmp_path / "mixed")
    _save_sharded_dataset(mixed, train_rows=20, test_rows=6)
    _save_preprocessed_dataset(mixed, train_rows=8, test_rows=4)
    with pytest.raises(ValueError, match=r"has shard_index.json while its metadata.json records num_shards=1"):
        load_datasets_auto(mixed, test_size=None, dataset_ratio=None)

    sharded_stamp_over_rows = str(tmp_path / "shards")
    _save_preprocessed_dataset(sharded_stamp_over_rows)
    write_stamped_sidecar(
        os.path.join(sharded_stamp_over_rows, METADATA_FILE),
        PreprocessedDatasetMetadata(max_length=2, num_shards=2),
    )
    with pytest.raises(
        ValueError, match=r"has no readable shard_index.json while its metadata.json records num_shards=2"
    ):
        load_datasets_auto(sharded_stamp_over_rows, test_size=None, dataset_ratio=None)


def test_train_only_list_entry_contributes_no_rows_to_the_held_out_split(caplog):
    """A filler of its first 100 train rows would seed a sibling's held-out split with rows the model trains on."""
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
    """Every consumer indexes ``ds["test"]``, so the placeholder applies once over the concatenated train. Per entry
    it would take 100 rows of each pool, so its size separates the two."""
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
    """``test_size`` splits every entry, so a train-only pool contributes held-out rows."""
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


def _save_train_validation_dataset(directory: str, train: Dataset, validation: Dataset) -> str:
    """Write a source whose held-out split is named ``validation`` (``rajpurkar/squad``'s layout)."""
    DatasetDict({"train": train, "validation": validation}).save_to_disk(directory)
    return directory


def test_a_source_with_no_test_split_evaluates_on_its_validation_split(caplog):
    temp_dir = tempfile.mkdtemp()
    try:
        _save_train_validation_dataset(temp_dir, _pool(6, 0), _pool(3, 100))

        with caplog.at_level(logging.INFO, logger=_LOADING_LOGGER):
            ds = load_datasets(temp_dir, test_size=None, dataset_ratio=1, conversation_field="messages")

        assert sorted(ds["test"]["row_id"]) == [100, 101, 102], "the test split must be the validation rows"
        assert sorted(ds["train"]["row_id"]) == list(range(6))
        assert "validation" not in ds
        assert any("'validation' split is the test split" in r.getMessage() for r in caplog.records)
    finally:
        shutil.rmtree(temp_dir)


def test_a_validation_only_list_entry_contributes_its_validation_rows_as_test():
    with_test = tempfile.mkdtemp()
    with_validation = tempfile.mkdtemp()
    try:
        _save_raw_dataset(with_test, _pool(4, 0), _pool(2, 100))
        _save_train_validation_dataset(with_validation, _pool(4, 200), _pool(2, 300))

        ds = load_datasets(
            [with_test, with_validation], test_size=None, dataset_ratio=1, conversation_field="messages"
        )

        assert sorted(ds["test"]["row_id"]) == [100, 101, 300, 301], ds["test"]["row_id"]
        assert sorted(ds["train"]["row_id"]) == [0, 1, 2, 3, 200, 201, 202, 203]
    finally:
        shutil.rmtree(with_test)
        shutil.rmtree(with_validation)


def test_a_test_split_is_read_before_a_validation_split():
    temp_dir = tempfile.mkdtemp()
    try:
        DatasetDict({"train": _pool(4, 0), "test": _pool(2, 100), "validation": _pool(2, 200)}).save_to_disk(temp_dir)

        ds = load_datasets(temp_dir, test_size=None, dataset_ratio=1, conversation_field="messages")

        assert sorted(ds["test"]["row_id"]) == [100, 101], ds["test"]["row_id"]
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
