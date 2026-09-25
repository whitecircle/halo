#!/usr/bin/env python
"""CPU tests for the contest selection a LiveCodeBench run scores: the date window, the platform
filter, their validation, and the re-grader rebuilding the same selection.

A LiveCodeBench release is cumulative, so an eval without a window scores every problem since 2023,
most of them older than a current model's training cutoff. The window is what makes a run
contamination-clean, and the offline re-grader rebuilds its payloads by example index: a selection it
did not re-apply would grade every episode against another problem's tests.

The release files are synthetic ``test*.jsonl`` written to a temp dir, served in place of the Hub
download (no network).

Run: python tests/cpu/environments/test_livecodebench_selection.py  (or pytest)
"""

import json
from types import SimpleNamespace

import pytest

from scripts.environments.inference import regrade_trajectories
from scripts.environments.inference.run_code_contests import build_examples, contest_meta, resolve_selection
from src.environments.envs.tasks.coding import datasets as coding_datasets
from src.environments.envs.tasks.coding.datasets import CODE_DATASET_ADAPTERS, CodeDatasetAdapter, ContestSelection
from src.environments.registry import resolve_environment

_DATASET = "livecodebench/code_generation_lite"
_LCB = CODE_DATASET_ADAPTERS["livecodebench"]


def _row(qid: str, platform: str, contest_date: str | None, testtype: str = "stdin") -> dict:
    row = {
        "question_title": qid,
        "question_content": f"Solve {qid}.",
        "platform": platform,
        "question_id": qid,
        "contest_id": qid,
        "difficulty": "easy",
        "public_test_cases": json.dumps([{"input": f"{qid}\n", "output": "1\n", "testtype": testtype}]),
        "private_test_cases": json.dumps([]),
        "metadata": "{}",
    }
    if contest_date is not None:
        row["contest_date"] = contest_date
    return row


# release_v2 = test.jsonl (the older contests) + test2.jsonl, read newest file first.
_RELEASE = {
    "test.jsonl": [
        _row("cf_2024_11_30", "codeforces", "2024-11-30T00:00:00"),
        _row("ac_2025_01_03", "atcoder", "2025-01-03T00:00:00"),
    ],
    "test2.jsonl": [
        _row("ac_2025_01_04", "atcoder", "2025-01-04T00:00:00"),
        _row("lc_2025_02_09", "leetcode", "2025-02-09T00:00:00", testtype="functional"),
        _row("cf_2025_03_01", "codeforces", "2025-03-01T00:00:00"),
        # Stamped past midnight: the window compares calendar days, so the end day holds it.
        _row("ac_2025_04_06", "atcoder", "2025-04-06T18:30:00"),
        _row("ac_2025_04_07", "atcoder", "2025-04-07T00:00:00"),
    ],
}
_WINDOW = ContestSelection.parse("2025-01-04", "2025-04-06")


@pytest.fixture
def release(tmp_path, monkeypatch):
    """The synthetic release on disk, served where the loader downloads from the Hub."""
    for name, rows in _RELEASE.items():
        (tmp_path / name).write_text("".join(json.dumps(row) + "\n" for row in rows))

    def download(repo_id, filename, repo_type):
        assert (repo_id, repo_type) == (_DATASET, "dataset")
        return str(tmp_path / filename)

    monkeypatch.setattr(coding_datasets, "hf_hub_download", download)
    regrade_trajectories._load_payloads.cache_clear()
    yield
    regrade_trajectories._load_payloads.cache_clear()


def _scored(selection: ContestSelection) -> list[str]:
    rows = _LCB.load(_DATASET, "release_v2", "test")
    return [row["question_id"] for row in _LCB.scored_rows(rows, selection)]


def _args(**overrides) -> SimpleNamespace:
    flags = {
        "dataset": _DATASET,
        "config": "release_v2",
        "split": "test",
        "num_examples": 0,
        "adapter": "livecodebench",
        "start_date": None,
        "end_date": None,
        "platform": None,
    }
    return SimpleNamespace(**{**flags, **overrides})


# --- The window and the platform filter ---


def test_the_window_holds_both_edge_days_and_drops_the_days_outside(release):
    assert _scored(_WINDOW) == ["ac_2025_01_04", "cf_2025_03_01", "ac_2025_04_06"]


def test_no_selection_scores_every_gradable_row_newest_release_first(release):
    assert _scored(ContestSelection()) == [
        "ac_2025_01_04",
        "cf_2025_03_01",
        "ac_2025_04_06",
        "ac_2025_04_07",
        "cf_2024_11_30",
        "ac_2025_01_03",
    ]


def test_either_end_of_the_window_may_stay_open(release):
    assert _scored(ContestSelection.parse(start_date="2025-04-06")) == ["ac_2025_04_06", "ac_2025_04_07"]
    assert _scored(ContestSelection.parse(end_date="2025-01-03")) == ["cf_2024_11_30", "ac_2025_01_03"]


def test_the_platform_filter_takes_the_dataset_spellings(release):
    assert _scored(ContestSelection.parse(platforms=["codeforces"])) == ["cf_2025_03_01", "cf_2024_11_30"]
    assert _scored(ContestSelection.parse("2025-01-04", "2025-04-06", ["atcoder"])) == [
        "ac_2025_01_04",
        "ac_2025_04_06",
    ]


def test_a_platform_the_adapter_cannot_grade_is_refused():
    """LeetCode rows are functional, which the stdin/stdout env drops: selecting them would download the
    whole release to score nothing."""
    assert "leetcode" not in _LCB.platforms
    with pytest.raises(ValueError, match=r"platform\(s\) \['leetcode'\] are not ones this adapter grades"):
        _scored(ContestSelection.parse(platforms=["leetcode"]))


# --- Validation ---


@pytest.mark.parametrize(
    ("start", "end", "match"),
    [
        ("2025/01/04", None, "start_date must be a YYYY-MM-DD day"),
        ("20250104", None, "start_date must be a YYYY-MM-DD day"),
        (None, "2025-02-30", "end_date must be a YYYY-MM-DD day"),
        (None, "2025-04-06T00:00:00", "end_date must be a YYYY-MM-DD day"),
        ("2025-04-07", "2025-04-06", "is after end_date"),
    ],
)
def test_a_malformed_or_empty_window_is_refused(start, end, match):
    with pytest.raises(ValueError, match=match):
        ContestSelection.parse(start, end)


def test_a_one_day_window_is_not_empty(release):
    assert _scored(ContestSelection.parse("2025-04-06", "2025-04-06")) == ["ac_2025_04_06"]


def test_a_selection_the_dataset_cannot_apply_is_refused_before_any_row_is_read():
    def unreadable():
        raise AssertionError("the selection must be validated before the rows are read")
        yield

    with pytest.raises(ValueError, match=r"platform\(s\) \['AtCoder'\] are not ones this adapter grades"):
        _LCB.scored_rows(unreadable(), ContestSelection.parse(platforms=["AtCoder"]))
    codeforces = CODE_DATASET_ADAPTERS["codeforces"]
    with pytest.raises(ValueError, match="stamps no contest date"):
        codeforces.scored_rows(unreadable(), _WINDOW)
    with pytest.raises(ValueError, match="records no platform"):
        codeforces.scored_rows(unreadable(), ContestSelection.parse(platforms=["codeforces"]))
    # An empty selection applies to any dataset.
    codeforces.require_selectable(ContestSelection())


def test_a_row_without_a_contest_date_is_refused_under_a_window_only():
    undated = _row("undated", "atcoder", None)
    with pytest.raises(ValueError, match="'undated' carries no ISO contest_date"):
        list(_LCB.scored_rows([undated], _WINDOW))
    assert list(_LCB.scored_rows([undated], ContestSelection())) == [undated]
    assert list(_LCB.scored_rows([undated], ContestSelection.parse(platforms=["atcoder"]))) == [undated]


def test_the_eval_flags_exit_on_a_selection_the_adapter_refuses():
    assert resolve_selection(_args(start_date="2025-01-04", platform="codeforces, atcoder"), _LCB) == (
        ContestSelection.parse("2025-01-04", None, ["atcoder", "codeforces"])
    )
    for platform in ("LeetCode", "leetcode"):
        with pytest.raises(SystemExit, match="--platform on --adapter livecodebench: platform"):
            resolve_selection(_args(platform=platform), _LCB)
    with pytest.raises(SystemExit, match="--platform names no platform"):
        resolve_selection(_args(platform=" , "), _LCB)
    with pytest.raises(SystemExit, match="--start_date/--end_date/--platform on --adapter livecodebench: start_date"):
        resolve_selection(_args(start_date="2025-1-4"), _LCB)
    with pytest.raises(SystemExit, match="--adapter codeforces: this dataset stamps no contest date"):
        resolve_selection(_args(adapter="codeforces", end_date="2025-04-06"), CODE_DATASET_ADAPTERS["codeforces"])


# --- The record a re-grade reads back ---


def test_the_selection_round_trips_through_the_meta_line():
    selection = ContestSelection.parse("2025-01-04", "2025-04-06", ["codeforces", "atcoder"])
    meta = json.loads(json.dumps(selection.to_meta()))
    assert meta == {"start_date": "2025-01-04", "end_date": "2025-04-06", "platforms": ["atcoder", "codeforces"]}
    assert ContestSelection.from_meta(meta) == selection
    assert selection.label == "2025-01-04..2025-04-06_atcoder+codeforces"
    # A meta line recording no selection scored every row.
    assert ContestSelection.from_meta(None) == ContestSelection()
    assert ContestSelection().label == ""


def test_a_selection_has_one_spelling_however_it_is_built():
    """It keys the re-grader's payload cache: a list or an unsorted tuple would fail to hash or miss."""
    built = ContestSelection(platforms=["codeforces", "atcoder", "atcoder"])
    assert built == ContestSelection.parse(platforms=("atcoder", "codeforces"))
    assert built.platforms == ("atcoder", "codeforces") and hash(built) == hash(
        ContestSelection.parse(platforms=["atcoder", "codeforces"])
    )


def test_an_adapter_declaring_platforms_must_name_their_field():
    with pytest.raises(ValueError, match="must name the platform_field"):
        CodeDatasetAdapter(str, dict, lambda row: True, platforms=("atcoder",))


def test_the_regrader_rebuilds_the_windowed_examples_from_the_meta_line_the_eval_writes(release):
    """Round trip through the writer (``contest_meta``) and the reader: the same problems in the same
    order, so every episode's index still names its own problem."""
    selection = ContestSelection.parse("2025-01-04", "2025-04-06", ["atcoder"])
    env = resolve_environment("codeforces", {})
    generic = {"env_type": "codeforces", "model": "m", "dataset": _DATASET, "config": "release_v2", "split": "test"}
    meta = json.loads(json.dumps({**generic, **contest_meta("livecodebench", selection, env, "medium", {})}))
    regrade_trajectories.validate_meta("f.jsonl", meta)

    examples = build_examples(_args(), _LCB, selection)
    payloads = regrade_trajectories.build_payloads(meta)

    assert [json.loads(example["context"]["answer"]) for example in examples] == list(payloads)
    assert [example["id"] for example in examples] == ["ac_2025_01_04", "ac_2025_04_06"]
    # Without the recorded selection, episode 1 would be graded against another problem's tests.
    unselected = regrade_trajectories.build_payloads({k: v for k, v in meta.items() if k != "selection"})
    assert unselected[1] != payloads[1]


def test_an_example_is_named_by_its_adapters_id_field():
    """Each source names its problems in its own column; the adapter declares which, so a row carrying
    another source's column as well is still named by its own."""
    rows = [{"id": "other", "question_id": "q1"}, {"question_id": "q2"}]

    def ids(**fields):
        adapter = CodeDatasetAdapter(str, dict, lambda row: True, load=lambda *_: rows, **fields)
        return [example["id"] for example in build_examples(_args(adapter="stub"), adapter, ContestSelection())]

    assert ids(id_field="question_id") == ["q1", "q2"]
    assert ids() == ["other", None]
    assert CODE_DATASET_ADAPTERS["livecodebench"].id_field == CODE_DATASET_ADAPTERS["hlce"].id_field == "question_id"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
