#!/usr/bin/env python
"""CPU tests for the code-contests pool preparation: the HardTests adapter, the generated-tests merge on
the Codeforces adapter, the test-suite caps and compaction order, and the preparation script's exclusion,
tests-table, checker-soundness, hold-out and band-config steps. A Codeforces problem whose only tests are its
statement's examples leaves the eval, the re-grade and the pool unless asked for, and a tests table covering
one split prepares every split.

Run: python tests/cpu/environments/test_code_pool_preparation.py  (or pytest)
"""

import base64
import dataclasses
import json
import logging
import pickle
import subprocess
import sys
import zlib
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from datasets import Dataset, DatasetDict, load_from_disk

from scripts.environments.inference import regrade_trajectories
from scripts.environments.inference.run_code_contests import build_examples, resolve_selection
from scripts.environments.preparation import compact_code_tests as cct
from scripts.environments.preparation import prepare_code_dataset
from scripts.environments.preparation.compact_code_tests import _flatten_indices, cap_tests
from scripts.environments.preparation.prepare_code_dataset import (
    RATING_BANDS,
    band_configs,
    band_of,
    carve_holdout,
    checker_is_sound,
    excluded_by,
    load_exclusions,
    load_tests_table,
)
from src.args.mixins import DEFAULT_ANSWER_FIELD
from src.environments.envs.tasks.coding.datasets import (
    CODE_DATASET_ADAPTERS,
    CodeDatasetAdapter,
    ContestSelection,
    _hardtests_memory_limit_mb,
    codeforces_examples_only,
    decode_test_payload,
    format_hardtests_prompt,
    hardtests_checker,
    hardtests_rating,
    keep_hardtests,
    normalize_hardtests,
    pack_codeforces_verification,
    pack_hardtests_verification,
)

_SUITE = json.dumps([{"input": "1 2\n", "output": "3\n"}, {"input": "5 5\n", "output": "10\n"}])


def _hardtests_row(**overrides):
    row = {
        "pid": "codeforces_1208_f",
        "question_title": "Bits And Pieces",
        "question_content": "## Problem Description\n\n[problemUrl]: https://codeforces.com/x\n\nGiven n numbers.\n\n"
        "## Input Format\n\nn then the numbers.\n\n### Sample Input #1\n\n3\n1 2 3\n",
        "difficulty_ratings": [{"level": "very hard", "score": 2600, "source": "codeforces"}],
        "time_limit": "2000 s",
        "memory_limit": "256.0 MB",
        "tags": [{"content": ["bitmasks", "dp"], "source": "codeforces"}, {"content": ["dp"], "source": "luogu"}],
        "starter_code": None,
        "joined_tests": _SUITE,
        "joined_checker": None,
    }
    row.update(overrides)
    return row


def test_hardtests_rating_prefers_codeforces_then_luogu_then_levels():
    cf_and_luogu = [{"source": "luogu", "score": 7, "level": "very hard"}, {"source": "codeforces", "score": 1500}]
    assert hardtests_rating({"difficulty_ratings": cf_and_luogu}) == 1500
    assert hardtests_rating({"difficulty_ratings": [{"source": "luogu", "score": 5, "level": "hard"}]}) == 2000
    # Luogu level 0 is "unknown" and must not become a rating; the TACO label decides instead.
    unknown_luogu = [{"source": "luogu", "score": 0, "level": "unknown"}, {"source": "taco", "level": "hard"}]
    assert hardtests_rating({"difficulty_ratings": unknown_luogu}) == 2100
    assert hardtests_rating({"difficulty_ratings": [{"source": "taco", "level": "unknown_difficulty"}]}) is None
    assert hardtests_rating({"difficulty_ratings": []}) is None
    out_of_table = [{"source": "luogu", "score": 9}, {"source": "atcoder", "level": "hard"}]
    assert hardtests_rating({"difficulty_ratings": out_of_table}) == 1800, "an unmapped Luogu level falls through"


def test_hardtests_time_limit_reads_milliseconds_and_seconds():
    assert pack_hardtests_verification(_hardtests_row(time_limit="2000 s"))["time_limit"] == 2.0
    assert pack_hardtests_verification(_hardtests_row(time_limit="1 s"))["time_limit"] == 1.0
    assert pack_hardtests_verification(_hardtests_row(time_limit="1 - 2 s"))["time_limit"] == 2.0
    assert pack_hardtests_verification(_hardtests_row(time_limit=None))["time_limit"] is None


def test_hardtests_memory_limit_reads_units_and_drops_scraped_byte_counts():
    limit = _hardtests_memory_limit_mb
    assert limit("256 MB") == 256.0
    assert limit("1024.0MB") == 1024.0
    assert limit("1 GB") == 1024.0
    assert limit("262144 KB") == 256.0
    assert limit("512") == 512.0
    assert limit("50000 bytes") is None, "CodeChef's scraped byte counts are not judge limits"
    assert limit("512 KiB") is None and limit("1 GiB") == 1024.0 and limit("1,024 MB") == 1024.0
    assert limit("9000 MB") is None, "above any judge's limit, so not a limit"
    assert limit(None) is None
    assert "memory limit per test: 134 MB" in format_hardtests_prompt(_hardtests_row(memory_limit="134.217728MB"))
    assert "memory limit" not in format_hardtests_prompt(_hardtests_row(memory_limit="50000 bytes"))


def test_hardtests_prompt_has_title_limits_body_and_no_judge_url():
    prompt = format_hardtests_prompt(_hardtests_row())
    assert prompt.startswith("# Bits And Pieces\n\ntime limit per test: 2 s\nmemory limit per test: 256 MB")
    assert "problemUrl" not in prompt
    assert "### Sample Input #1" in prompt


def test_hardtests_normalize_sets_id_rating_and_flat_tags():
    fields = normalize_hardtests(_hardtests_row())
    assert fields == {"id": "codeforces_1208_f", "rating": 2600, "tags": ["bitmasks", "dp"]}


def test_hardtests_keep_drops_functional_unrated_and_suiteless_rows():
    assert keep_hardtests({**_hardtests_row(), "rating": 2600})
    assert not keep_hardtests({**_hardtests_row(), "rating": 2600, "starter_code": "def solve(a):"})
    assert not keep_hardtests({**_hardtests_row(), "rating": None})
    assert not keep_hardtests({**_hardtests_row(), "rating": 2600, "joined_tests": None})


def test_hardtests_pack_uses_the_joined_suite():
    payload = pack_hardtests_verification(_hardtests_row())
    assert payload["tests"] == [{"input": "1 2\n", "output": "3\n"}, {"input": "5 5\n", "output": "10\n"}]
    assert payload["checker"] is None


def test_hardtests_checker_runs_the_judging_function_under_the_env_contract(tmp_path):
    """The env calls ``python checker.py input expected got``; the judging function takes
    ``(input, candidate, reference)``, so the two output paths must be swapped into it."""
    judge = (
        "def output_judging_function(input_str, candidate_output, reference_output):\n"
        "    return candidate_output.split() == list(reversed(reference_output.split()))\n"
    )
    checker = tmp_path / "checker.py"
    checker.write_text(hardtests_checker(judge))
    (tmp_path / "in.txt").write_text("3\n")
    (tmp_path / "exp.txt").write_text("1 2 3\n")
    (tmp_path / "got.txt").write_text("3 2 1\n")
    (tmp_path / "bad.txt").write_text("1 2 3\n")
    argv = [sys.executable, str(checker), str(tmp_path / "in.txt"), str(tmp_path / "exp.txt")]
    assert subprocess.run([*argv, str(tmp_path / "got.txt")], capture_output=True, text=True).stdout.split()[-1] == "1"
    assert subprocess.run([*argv, str(tmp_path / "bad.txt")], capture_output=True, text=True).stdout.split()[-1] == "0"
    assert hardtests_checker(None) is None and hardtests_checker("   ") is None


def test_hardtests_checker_rejects_when_the_judging_function_raises(tmp_path):
    checker = tmp_path / "checker.py"
    checker.write_text(hardtests_checker("def output_judging_function(i, c, r):\n    return int(c) > 0\n"))
    for name, text in (("in.txt", ""), ("exp.txt", "1\n"), ("got.txt", "not a number\n")):
        (tmp_path / name).write_text(text)
    out = subprocess.run(
        [sys.executable, str(checker), str(tmp_path / "in.txt"), str(tmp_path / "exp.txt"), str(tmp_path / "got.txt")],
        capture_output=True,
        text=True,
    )
    assert out.stdout.split()[-1] == "0"


def test_codeforces_pack_appends_joined_generated_tests_after_official_ones():
    row = {
        "official_tests": [{"input": "1\n", "output": "1\n"}],
        "examples": [{"input": "9\n", "output": "9\n"}],
        "joined_tests": json.dumps([{"input": "2\n", "output": "2\n"}]),
        "generated_checker": None,
        "time_limit": 1.0,
    }
    assert pack_codeforces_verification(row)["tests"] == [
        {"input": "1\n", "output": "1\n"},
        {"input": "2\n", "output": "2\n"},
    ]
    assert not codeforces_examples_only(row)
    assert not codeforces_examples_only({**row, "official_tests": []}), "a joined suite is a hidden test"
    sample_only = {**row, "official_tests": [{"input": "9", "output": " 9\n"}], "joined_tests": None}
    assert codeforces_examples_only(sample_only), "official tests that are the statement's sample hide nothing"
    other_answer = {**sample_only, "official_tests": [{"input": "9\n", "output": "another valid answer\n"}]}
    assert codeforces_examples_only(other_answer), "a special judge accepts the shown answer to the shown input"
    # Without either, the statement examples remain the fallback, and the row says so.
    row.update(official_tests=[], joined_tests=None)
    assert pack_codeforces_verification(row)["tests"] == [{"input": "9\n", "output": "9\n"}]
    assert codeforces_examples_only(row) and CODE_DATASET_ADAPTERS["codeforces"].is_examples_only(row)


def test_cap_tests_keeps_order_size_caps_and_large_slots():
    small = [{"input": str(i), "output": "x"} for i in range(5)]
    large = {"input": "a" * 3000, "output": ""}
    huge = {"input": "a" * 20_000, "output": ""}
    caps = {"max_tests": 4, "max_test_bytes": 1000, "max_large_tests": 1, "max_large_bytes": 5000}
    kept = cap_tests([small[0], huge, large, large, *small[1:]], **caps)
    assert kept == [small[0], large, small[1], small[2]], "one large slot, huge dropped, order kept, cap at 4"
    assert cap_tests(small, max_tests=10, max_test_bytes=1, max_large_tests=0, max_large_bytes=1) == []
    wide = {"input": "é" * 3, "output": ""}  # 3 characters, 6 bytes: the caps are byte caps
    assert cap_tests([wide], max_tests=10, max_test_bytes=4, max_large_tests=0, max_large_bytes=4) == []
    assert cap_tests([wide], max_tests=10, max_test_bytes=6, max_large_tests=0, max_large_bytes=6) == [wide]


def _encoded_payload(obj) -> str:
    return base64.b64encode(zlib.compress(pickle.dumps(obj))).decode("ascii")


def test_decode_test_payload_reads_json_strings_and_refuses_pickled_globals():
    suite = [{"input": "1\n", "output": "2\n"}]
    assert decode_test_payload(json.dumps(suite)) == suite
    assert decode_test_payload(_encoded_payload(json.dumps(suite))) == suite
    with pytest.raises(pickle.UnpicklingError, match="only a JSON string"):
        decode_test_payload(_encoded_payload(subprocess.check_output))


def _prepared_row(checker):
    return {
        "answer": json.dumps({"tests": [{"input": "1\n", "output": "2\n"}], "checker": checker, "time_limit": 1.0})
    }


def test_checker_soundness_gate_keeps_real_judges_and_drops_broken_ones():
    assert checker_is_sound(_prepared_row(None)), "token comparison needs no gate"
    sound = "import sys\nprint(1 if open(sys.argv[3]).read().split() == open(sys.argv[2]).read().split() else 0)\n"
    assert checker_is_sound(_prepared_row(sound))
    assert not checker_is_sound(_prepared_row("print(0)\n")), "rejects its own reference output"
    assert not checker_is_sound(_prepared_row("print(100)\n")), "a score is not the contract's 1"
    assert not checker_is_sound(_prepared_row("print(1)\n")), "accepts garbage"
    # Oriented: argv[3] is the candidate. A judge reading the reference slot instead would pass garbage too.
    sees_candidate = "import sys\nprint(1 if open(sys.argv[3]).read().strip() == '2' else 0)\n"
    assert checker_is_sound(_prepared_row(sees_candidate))
    assert not checker_is_sound(_prepared_row("raise RuntimeError('broken judge')\n"))
    raising_judge = hardtests_checker("def output_judging_function(i, c, r):\n    raise ValueError(c)\n")
    assert not checker_is_sound(_prepared_row(raising_judge))


def test_exclusions_match_exact_ids_and_prefixes(tmp_path):
    listing = tmp_path / "exclude.txt"
    listing.write_text("codeforces_2063_a\ncodeforces_1873_*\n\n# comment\n")
    exact, prefixes = load_exclusions(str(listing))
    assert excluded_by("codeforces_2063_a", exact, prefixes)
    assert excluded_by("codeforces_1873_d", exact, prefixes)
    assert not excluded_by("codeforces_2063_b", exact, prefixes)
    # The prefix is literal ("codeforces_1873_"), so a longer contest id does not match it.
    assert not excluded_by("codeforces_18730_a", exact, prefixes)


def test_hardtests_rows_are_gradable_only_after_normalize():
    raw = {k: v for k, v in _hardtests_row().items() if k != "rating"}
    assert not keep_hardtests(raw), "the raw row carries no rating"
    assert keep_hardtests({**raw, **normalize_hardtests(raw)})
    assert (
        not CODE_DATASET_ADAPTERS["hardtests"].scores_raw_rows and CODE_DATASET_ADAPTERS["codeforces"].scores_raw_rows
    )


def test_eval_examples_refuse_an_empty_yield():
    adapter = CodeDatasetAdapter(str, dict, lambda row: False, load=lambda dataset, config, split: [{"id": "x"}])
    args = type("Args", (), {"dataset": "d", "config": None, "split": "test", "num_examples": 0, "adapter": "stub"})()
    with pytest.raises(SystemExit, match="no gradable problem"):
        build_examples(args, adapter, ContestSelection())


def test_hardtests_adapter_is_registered_with_its_normalizer():
    adapter = CODE_DATASET_ADAPTERS["hardtests"]
    assert adapter.normalize is normalize_hardtests and adapter.load is None


def test_flatten_indices_orders_hardtests_groups_and_nests():
    mapping = {"SPGen": [7], "RPGen": [[5, 6]], "HackGen": [3, 4], "LLMGen": [[0, 1], [2]]}
    assert _flatten_indices(mapping) == [0, 1, 2, 3, 4, 5, 6, 7]
    assert _flatten_indices({}) == []


def _write_part(path, keys):
    table = pa.table({"key": keys, "tests": ["[]"] * len(keys), "checker": [None] * len(keys)})
    pq.write_table(table, path)


def test_tests_table_refuses_a_key_carried_twice(tmp_path):
    _write_part(tmp_path / "a.parquet", ["p1", "p2"])
    _write_part(tmp_path / "b.parquet", ["p3"])
    table, index = load_tests_table(str(tmp_path))
    assert len(table) == 3 and set(index) == {"p1", "p2", "p3"}
    _write_part(tmp_path / "c.parquet", ["p2"])
    with pytest.raises(SystemExit, match="'p2' twice"):
        load_tests_table(str(tmp_path))


def _rated(ratings):
    return Dataset.from_dict({"id": [f"p{i}" for i in range(len(ratings))], "rating": ratings})


def test_holdout_takes_the_same_rows_from_every_band_and_the_rows_below():
    ratings = [1000, 1100, 1200] + [1500, 1600, 1700] + [2000, 2100, 2200] + [2600, 2700, 2800]
    train, test = carve_holdout(_rated(ratings), per_band=1)
    assert len(test) == 4 and len(train) == 8
    assert sorted(band_of_all(test["rating"])) == ["below", "extra-hard", "hard", "medium"]
    again_train, again_test = carve_holdout(_rated(ratings), per_band=1)
    assert again_test["id"] == test["id"], "deterministic"
    assert set(train["id"]).isdisjoint(test["id"])
    with pytest.raises(SystemExit, match="fewer than the 2"):
        carve_holdout(_rated([1500, 2000, 2600, 1000]), per_band=2)
    with pytest.raises(SystemExit, match="band 'hard' has 0 rows"):
        carve_holdout(_rated([1000, 1500, 2600]), per_band=1)


def test_recompaction_that_keeps_no_rows_removes_the_old_part(tmp_path, monkeypatch):
    """A part written under wider caps must not survive a rebuild that empties the shard, or the next
    tests-table join reads tests the current caps excluded."""
    part = tmp_path / "shard-0.parquet"
    stale = pa.Table.from_pylist([], schema=cct._OUTPUT_SCHEMA).replace_schema_metadata(
        {cct._CAPS_METADATA_KEY: json.dumps({"max_tests": 40}).encode()}
    )
    pq.write_table(stale, part)
    monkeypatch.setitem(cct._COMPACTORS, "empty", lambda path, caps: cct.ShardResult([], 0, 1))
    assert cct._compact_one(("empty", "shard-0", str(part), {"max_tests": 1}))[1] == 0
    assert not part.exists(), "the stale part must be removed, not left for the next join"


def band_of_all(ratings):
    return [band_of(r) or "below" for r in ratings]


def test_band_configs_share_the_test_split_and_cut_train_by_band():
    pool = DatasetDict({"train": _rated([1000, 1500, 1999, 2000, 2599, 2600, 3500]), "test": _rated([1200, 2100])})
    configs = band_configs(pool)
    assert set(configs) == {"full", *RATING_BANDS}
    assert configs["full"]["train"]["rating"] == [1000, 1500, 1999, 2000, 2599, 2600, 3500]
    assert configs["medium"]["train"]["rating"] == [1500, 1999]
    assert configs["hard"]["train"]["rating"] == [2000, 2599]
    assert configs["extra-hard"]["train"]["rating"] == [2600, 3500]
    assert all(cfg["test"]["rating"] == [1200, 2100] for cfg in configs.values())


# --- Problems graded on their statement's examples alone ---

_SAMPLE = [{"input": "1\n", "output": "1\n"}]
_ECHO = [{"input": "2\n", "output": "2\n"}]


def _codeforces_row(problem_id: str, official: list[dict[str, str]]) -> dict:
    """A gradable ``open-r1/codeforces`` row whose statement shows :data:`_SAMPLE`, graded on ``official``."""
    return {
        "id": problem_id,
        "title": "Echo",
        "index": "A",
        "rating": 1200,
        "tags": ["implementation"],
        "executable": True,
        "input_mode": "stdio",
        "description": "Print n.",
        "input_format": "n",
        "output_format": "n",
        "time_limit": 1.0,
        "memory_limit": 256.0,
        "generated_checker": None,
        "examples": _SAMPLE,
        "official_tests": official,
    }


# No official test (the payload falls back to the sample), official tests that are the sample spaced otherwise (the
# verifiable config's shape for 206 of its 422 test problems), and a hidden test.
_ROWS = [
    _codeforces_row("cf-0", []),
    _codeforces_row("cf-1", [{"input": "1 \n", "output": "1"}]),
    _codeforces_row("cf-2", _ECHO),
]


def test_the_eval_leaves_out_and_counts_examples_only_problems_unless_asked(caplog):
    """A problem graded on the samples its prompt shows is passed by a program printing a constant: scored, such
    problems inflate success@k."""
    adapter = dataclasses.replace(CODE_DATASET_ADAPTERS["codeforces"], load=lambda *_: list(_ROWS))
    args = SimpleNamespace(
        dataset="d",
        config=None,
        split="test",
        num_examples=0,
        adapter="codeforces",
        start_date=None,
        end_date=None,
        platform=None,
        include_examples_only=False,
    )
    with caplog.at_level(logging.INFO):
        default = build_examples(args, adapter, resolve_selection(args, adapter))
    assert [example["id"] for example in default] == ["cf-2"]
    assert "Left out 2 problems graded only on their statement's examples" in caplog.text
    args.include_examples_only = True
    included = build_examples(args, adapter, resolve_selection(args, adapter))
    assert [example["id"] for example in included] == ["cf-0", "cf-1", "cf-2"]


def test_the_regrader_rebuilds_the_rows_each_run_scored(monkeypatch):
    """The re-grade reads an episode's problem by index, so it rebuilds the eval's sequence from the meta line: a
    run recording the exclusion scored without the examples-only problems, one predating it with them."""
    monkeypatch.setattr(regrade_trajectories, "load_hf_split", lambda *_: list(_ROWS))
    regrade_trajectories._load_payloads.cache_clear()
    meta = {"adapter": "codeforces", "dataset": "examples-only/now", "split": "test"}
    current = regrade_trajectories.build_payloads({**meta, "selection": ContestSelection().to_meta()})
    older = {**meta, "dataset": "examples-only/older", "selection": {"start_date": None, "end_date": None}}
    assert [payload["tests"] for payload in current] == [_ECHO]
    assert len(regrade_trajectories.build_payloads(older)) == len(_ROWS)
    regrade_trajectories._load_payloads.cache_clear()


def _prepare(monkeypatch, tmp_path, source: DatasetDict, name: str, *flags: str) -> DatasetDict:
    """``prepare_code_dataset.py --adapter codeforces`` over ``source``, saved under ``tmp_path / name``."""
    load = prepare_code_dataset.load_dataset
    monkeypatch.setattr(
        prepare_code_dataset, "load_dataset", lambda path, *a, **k: source if path == "src" else load(path, *a, **k)
    )
    monkeypatch.setattr(prepare_code_dataset, "disable_caching", lambda: None)
    out = tmp_path / name
    argv = ["prepare_code_dataset.py", "--adapter", "codeforces", "--dataset", "src", "--output_dir", str(out)]
    monkeypatch.setattr(sys, "argv", [*argv, "--num_proc", "1", "--no-verify_checkers", *flags])
    prepare_code_dataset.main()
    return load_from_disk(str(out))


def test_the_pool_drops_examples_only_problems_unless_asked(monkeypatch, tmp_path):
    source = DatasetDict({"train": Dataset.from_list(_ROWS)})
    assert _prepare(monkeypatch, tmp_path, source, "default")["train"]["id"] == ["cf-2"]
    included = _prepare(monkeypatch, tmp_path, source, "included", "--include_examples_only")
    assert included["train"]["id"] == ["cf-0", "cf-1", "cf-2"]


def _tests_table(tmp_path, key: str) -> str:
    table = tmp_path / "tests-table"
    table.mkdir()
    pq.write_table(pa.table({"key": [key], "tests": [json.dumps(_ECHO)], "checker": [None]}), table / "part-0.parquet")
    return str(table)


def test_a_tests_table_covering_one_split_prepares_every_split(monkeypatch, tmp_path):
    """Generated tests that cover train alone are a valid table: the test split prepares from its own official
    tests, and the train row the table covers is graded on its joined suite, no longer on its sample alone."""
    source = DatasetDict(
        {"train": Dataset.from_list(_ROWS), "test": Dataset.from_list([_codeforces_row("cf-9", _ECHO)])}
    )
    pool = _prepare(monkeypatch, tmp_path, source, "pool", "--tests_table", _tests_table(tmp_path, "cf-1"))
    assert pool["train"]["id"] == ["cf-1", "cf-2"] and pool["test"]["id"] == ["cf-9"]
    assert json.loads(pool["train"][0][DEFAULT_ANSWER_FIELD])["tests"][-1:] == _ECHO


def test_a_tests_table_matching_no_split_exits(monkeypatch, tmp_path):
    source = DatasetDict({"train": Dataset.from_list(_ROWS)})
    with pytest.raises(SystemExit, match="no row of any split matched a key of the tests table"):
        _prepare(monkeypatch, tmp_path, source, "pool", "--tests_table", _tests_table(tmp_path, "other"))
    assert not (tmp_path / "pool").exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
