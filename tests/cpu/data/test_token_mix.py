#!/usr/bin/env python
"""The weighted token mix: every 4,096 consecutive samples hold each quota exactly and shorter runs (a step's rows)
stay within about two samples of their share, passes revisit every window in a new order, the samples are a pure
function of the config so a loader skip resumes the uninterrupted stream on any data-parallel layout of the same rows
per step, a changed mix or rank that sees other files (held-out folders too) is refused naming what differs, and the
open-file limit is raised to hold the open files or the mix is refused."""

import contextlib
import datetime
import json
import logging
import math
import resource
import shutil
from unittest import mock

import numpy as np
import pytest
import torch
from accelerate.data_loader import prepare_data_loader, skip_first_batches
from torch.utils.data import DataLoader, Dataset, SequentialSampler

from halo.data.collators.packing import DataCollatorWithPacking
from halo.data.sources import token_mix, token_shards
from halo.data.sources.token_mix import (
    ORDER_VERSION,
    PATTERN_PERIOD,
    TokenMix,
    TokenSource,
    build_held_out_shards_across_ranks,
    build_token_mix_across_ranks,
    keyed_permutation,
    verify_resumed_token_mix,
)
from halo.data.sources.token_shards import TokenShards
from tests.common.datatrove_shards import write_token_folder
from tests.common.gloo import run_gloo_ranks
from tests.common.goldens.token_mix_stream import GOLDEN_PATH, render
from tests.common.offline_grpo import make_offline_tokenizer

MAX_LENGTH = 8
# Windows of MAX_LENGTH per file of each synthetic source: 37, 101 and 5 in all.
SOURCE_WINDOWS = {"a": (20, 17), "b": (60, 41), "c": (3, 2)}
NATURAL = {"a": None, "b": None, "c": None}
WORLD_SIZE = 2
PG_TIMEOUT = datetime.timedelta(seconds=60)


def _write_source(folder, windows_per_file, rng):
    """Files of the given window counts plus a tail, each split into documents of random lengths."""
    files = {}
    for index, windows in enumerate(windows_per_file):
        tokens = windows * MAX_LENGTH + int(rng.integers(0, MAX_LENGTH))
        cuts = np.unique(np.concatenate(([0, tokens], rng.integers(1, tokens, size=tokens // 5))))
        files[f"{index:05d}.ds"] = [
            rng.integers(3, 1000, size=end - begin).tolist() for begin, end in zip(cuts[:-1], cuts[1:], strict=True)
        ]
    return write_token_folder(folder, files, token_bytes=2)


@pytest.fixture(scope="module")
def folders(tmp_path_factory):
    root = tmp_path_factory.mktemp("token_mix")
    rng = np.random.default_rng(0)
    return {name: _write_source(root / name, windows, rng) for name, windows in SOURCE_WINDOWS.items()}


def _mix(folders, weights, *, num_steps=3 * PATTERN_PERIOD, rows_per_step=1, seed=0, max_length=MAX_LENGTH):
    sources = [TokenSource(name, str(folders[name]), weight) for name, weight in weights.items()]
    return TokenMix(sources, max_length=max_length, num_steps=num_steps, rows_per_step=rows_per_step, seed=seed)


def _share_violations(sources: np.ndarray, quotas: list[int], run: int, tolerance: float) -> list[tuple[int, int]]:
    """``(source, start)`` of each run of ``run`` consecutive samples whose count of the source is off its share
    ``run * quota / PATTERN_PERIOD`` by more than ``tolerance``."""
    violations = []
    for source, quota in enumerate(quotas):
        held = np.concatenate(([0], np.cumsum(sources == source)))
        counts = held[run:] - held[:-run]
        off = np.flatnonzero(np.abs(counts - run * quota / PATTERN_PERIOD) > tolerance)
        violations += [(source, int(start)) for start in off[:3]]
    return violations


def _source_sequence(mix) -> np.ndarray:
    return np.array([mix.locate(index)[0] for index in range(len(mix))])


_WEIGHT_SETS = {
    "7:3": ({"a": 7, "b": 3}, [2867, 1229]),
    # 4096 * (37, 101, 5) / 143 = 1059.80, 2892.98, 143.22: the two spare samples go to b, then a.
    "natural shares": (NATURAL, [1060, 2893, 143]),
    "1:1:1": ({"a": 1, "b": 1, "c": 1}, [1366, 1365, 1365]),
    "skewed": ({"a": 1, "b": 50, "c": 0.5}, [79, 3977, 40]),
}


def test_each_period_holds_its_quota_and_shorter_runs_stay_near_their_share(folders, subtests):
    for label, (weights, quotas) in _WEIGHT_SETS.items():
        with subtests.test(label):
            mix = _mix(folders, weights)
            assert mix.quotas == quotas and sum(quotas) == PATTERN_PERIOD
            sources = _source_sequence(mix)
            assert not _share_violations(sources, quotas, PATTERN_PERIOD, tolerance=0)
            for run in (8, 64, 512):
                assert not _share_violations(sources, quotas, run, tolerance=2), run


def test_the_share_check_fails_a_swapped_weights_control(folders):
    mix = _mix(folders, {"a": 7, "b": 3})
    swapped = list(reversed(mix.quotas))
    sources = _source_sequence(mix)
    assert _share_violations(sources, swapped, PATTERN_PERIOD, tolerance=0)
    assert _share_violations(sources, swapped, 64, tolerance=2)


def test_the_same_config_draws_the_same_stream_and_a_new_seed_another(folders):
    first, again, reseeded = (_mix(folders, NATURAL, seed=seed) for seed in (0, 0, 1))
    located = [first.locate(index) for index in range(len(first))]
    assert located == [again.locate(index) for index in range(len(again))]
    assert [first[index] for index in range(0, len(first), 97)] == [again[index] for index in range(0, len(again), 97)]
    moved = [reseeded.locate(index) for index in range(len(reseeded))]
    assert [source for source, _ in moved] == [source for source, _ in located], "the pattern ignores the seed"
    assert sum(old != new for old, new in zip(located, moved, strict=True)) > 0.9 * len(located)


def test_each_pass_visits_every_window_once_in_a_new_order(folders, subtests):
    mix = _mix(folders, {"a": 1, "b": 1, "c": 1})
    located = [mix.locate(index) for index in range(len(mix))]
    for source, name in enumerate(SOURCE_WINDOWS):
        with subtests.test(name):
            windows = mix.shards[source].num_windows
            draws = [window for drawn, window in located if drawn == source]
            passes = [tuple(draws[start : start + windows]) for start in range(0, len(draws) - windows + 1, windows)]
            assert len(passes) >= 3
            assert all(sorted(order) == list(range(windows)) for order in passes)
            if windows >= 37:
                assert all(first != second for first, second in zip(passes[:-1], passes[1:], strict=True)), name
            else:
                assert len(set(passes)) > math.factorial(windows) // 2, "the key changes the order every pass"


def test_the_keyed_permutation_is_a_bijection_that_the_key_changes(subtests):
    for size in (1, 2, 3, 5, 16, 17, 255, 256, 1000, 4097):
        with subtests.test(size):
            image = [keyed_permutation(index, size, key=11) for index in range(size)]
            assert sorted(image) == list(range(size))
            if size >= 16:
                assert image != [keyed_permutation(index, size, key=12) for index in range(size)]
    with pytest.raises(IndexError):
        keyed_permutation(5, 5, key=0)


class _Indexed(Dataset):
    """The mix with each row's sample index, so a test can tell which samples a loader handed out."""

    def __init__(self, mix):
        self.mix = mix

    def __len__(self):
        return len(self.mix)

    def __getitem__(self, index):
        return {"index": index, **self.mix[index]}


def _keep_rows(rows):
    return rows


def _rank_batches(mix, *, batch: int, dp_size: int, dp_rank: int, skip: int = 0) -> list[list[dict]]:
    """The batches a data-parallel rank draws as the trainer's loader does: sequential, sharded by accelerate across
    ``dp_size`` ranks, with a resume's skip of ``skip`` batches."""
    loader = DataLoader(_Indexed(mix), batch_size=batch, sampler=SequentialSampler(mix), collate_fn=_keep_rows)
    prepared = prepare_data_loader(
        loader, torch.device("cpu"), num_processes=dp_size, process_index=dp_rank, put_on_device=False
    )
    return list(skip_first_batches(prepared, skip) if skip else prepared)


@pytest.mark.parametrize(("batch", "accumulation", "dp_size"), [(4, 2, 3), (6, 2, 2), (24, 1, 1)])
def test_dp_ranks_split_each_step_and_a_skip_resumes_the_same_rows(folders, batch, accumulation, dp_size):
    """Every layout keeps a global batch of 24 rows: step t takes samples [24t, 24t + 24) whatever the DP size, and a
    fresh mix skipped past t steps hands each rank the rows of the uninterrupted run."""
    steps, resumed_at, global_batch = 10, 4, batch * accumulation * dp_size
    mix = _mix(folders, NATURAL, num_steps=steps, rows_per_step=global_batch)
    ranks = [_rank_batches(mix, batch=batch, dp_size=dp_size, dp_rank=rank) for rank in range(dp_size)]
    for step in range(steps):
        held = [
            row["index"]
            for batches in ranks
            for rows in batches[step * accumulation : (step + 1) * accumulation]
            for row in rows
        ]
        assert sorted(held) == list(range(step * global_batch, (step + 1) * global_batch)), step
    fresh = _mix(folders, NATURAL, num_steps=steps, rows_per_step=global_batch)
    for rank in range(dp_size):
        resumed = _rank_batches(fresh, batch=batch, dp_size=dp_size, dp_rank=rank, skip=resumed_at * accumulation)
        assert resumed == ranks[rank][resumed_at * accumulation :], rank


def test_mix_rows_collate_with_positions_restarting_at_each_document(folders):
    mix = _mix(folders, NATURAL)
    rows = [mix[index] for index in range(6)]
    assert all(tuple(row) == mix.column_names and row["labels"] == row["input_ids"] for row in rows)
    assert any(len(row["seq_lengths"]) > 1 for row in rows)
    batch = DataCollatorWithPacking(make_offline_tokenizer())(rows)
    positions = [position for row in rows for length in row["seq_lengths"] for position in range(length)]
    ids = [token for row in rows for token in row["input_ids"]]
    assert batch["input_ids"].tolist() == [ids]
    assert batch["position_ids"].tolist() == [positions]
    assert batch["labels"].tolist() == [[-100 if p == 0 else t for t, p in zip(ids, positions, strict=True)]]


def _saved(mix) -> dict:
    """The fingerprint as a checkpoint holds it, through JSON."""
    return json.loads(json.dumps(mix.fingerprint()))


def _copy_with(folders, root, name, files) -> dict:
    """``folders`` with source ``name`` replaced by a copy holding ``files``."""
    write_token_folder(root / name, files, token_bytes=2)
    return {**folders, name: root / name}


def test_an_unchanged_mix_resumes_wherever_its_files_sit(folders, tmp_path, subtests):
    saved = _saved(_mix(folders, {"a": 7, "b": 3}))
    moved = {name: shutil.copytree(path, tmp_path / name) for name, path in folders.items()}
    same = {
        "rebuilt": _mix(folders, {"a": 7, "b": 3}),
        "copied elsewhere": _mix(moved, {"a": 7, "b": 3}),
        "weights scaled alike": _mix(folders, {"a": 14, "b": 6}),
        "a longer run": _mix(folders, {"a": 7, "b": 3}, num_steps=5 * PATTERN_PERIOD),
    }
    for label, live in same.items():
        with subtests.test(label):
            verify_resumed_token_mix(saved, live.fingerprint())


def _read_documents(folder) -> dict[str, list[np.ndarray]]:
    """A synthetic folder's documents, split at its index."""
    files = {}
    for path in sorted(folder.glob("*.ds")):
        tokens = np.fromfile(path, dtype="<u2")
        ends = np.fromfile(f"{path}.index", dtype="<u8").astype(np.int64)
        files[path.name] = np.split(tokens, ends[:-1])
    return files


def _documents(folder) -> dict[str, list[list[int]]]:
    """A synthetic folder's documents as id lists, to write again."""
    return {
        name: [[int(token) for token in document] for document in documents]
        for name, documents in _read_documents(folder).items()
    }


def test_a_changed_mix_is_refused_naming_the_changed_field(folders, tmp_path, subtests):
    base = {"a": 7, "b": 3}
    saved = _saved(_mix(folders, base))
    a_files = _documents(folders["a"])
    grown = {**a_files, "00000.ds": [*a_files["00000.ds"], [7, 7, 7]]}
    added = {**a_files, "00002.ds": [[8] * 20]}
    reshuffled = {name: documents[::-1] for name, documents in a_files.items()}

    def rewritten(label, names=("a",), **layout):
        root = tmp_path / label
        for name in names:
            write_token_folder(root / name, _documents(folders[name]), **layout)
        return _mix({**folders, **{name: root / name for name in names}}, base)

    changed = {
        "weight": (lambda: _mix(folders, {"a": 6, "b": 4}), "source 'a' quota 2867 → 2458"),
        "max_length": (lambda: _mix(folders, base, max_length=4), "max_length 8 → 4"),
        "seed": (lambda: _mix(folders, base, seed=1), "seed 0 → 1"),
        "rows per step": (lambda: _mix(folders, base, rows_per_step=2), "rows_per_step 1 → 2.*halve one factor"),
        # Every source at once: a mix holds one width.
        "same ids at 4 bytes": (
            lambda: rewritten("wide", ("a", "b"), token_bytes=4),
            "source 'a' token_bytes 2 → 4.*source 'b' token_bytes 2 → 4",
        ),
        "another tokenizer": (
            lambda: rewritten("retokenized", token_bytes=2, tokenizer="other"),
            "source 'a' file 00000.ds names tokenizer 'other', not 'synthetic'",
        ),
        "source order": (lambda: _mix(folders, {"b": 3, "a": 7}), r"sources \['a', 'b'\] → \['b', 'a'\]"),
        "source added": (lambda: _mix(folders, {**base, "c": 1}), r"sources \['a', 'b'\] → \['a', 'b', 'c'\]"),
        "file grew": (
            lambda: _mix(_copy_with(folders, tmp_path / "grown", "a", grown), base),
            f"source 'a' file 00000.ds holds {sum(map(len, grown['00000.ds']))} tokens",
        ),
        "file added": (
            lambda: _mix(_copy_with(folders, tmp_path / "added", "a", added), base),
            r"source 'a' gained files \['00002.ds'\]",
        ),
        "same sizes, other tokens": (
            lambda: _mix(_copy_with(folders, tmp_path / "reshuffled", "a", reshuffled), base),
            "source 'a' content",
        ),
    }
    for label, (build, message) in changed.items():
        with subtests.test(label):
            with pytest.raises(ValueError, match=f"differs from the one the checkpoint was trained on: .*{message}"):
                verify_resumed_token_mix(saved, build().fingerprint())
    with subtests.test("order_version"), pytest.raises(ValueError, match=f"order_version 0 → {ORDER_VERSION}"):
        verify_resumed_token_mix({**saved, "order_version": 0}, _mix(folders, base).fingerprint())


def test_a_source_too_light_to_be_drawn_is_refused(folders):
    with pytest.raises(ValueError, match=r"Token sources \['c'\] would never be drawn: its weight is under one"):
        _mix(folders, {"a": 1, "c": 1e-4})


def test_each_source_logs_the_share_of_its_tokens_its_windows_hold(folders, caplog):
    """The tail of each file shorter than a window never trains; many small files can drop a large share."""
    with caplog.at_level(logging.INFO, logger=token_mix.logger.name):
        mix = _mix(folders, NATURAL)
    for name, shards in zip(SOURCE_WINDOWS, mix.shards, strict=True):
        covered = shards.num_windows * MAX_LENGTH / sum(file.num_tokens for file in shards.files)
        (line,) = [record.getMessage() for record in caplog.records if f"Token source {name!r}" in record.getMessage()]
        assert f"windows of {MAX_LENGTH} tokens ({covered:.1%} of its tokens" in line, line


def test_the_stream_matches_its_golden():
    assert render() == GOLDEN_PATH.read_text(encoding="utf-8")


def _rank_worker(rank: int, case: str, paths: list[str], out: str) -> None:
    """Build the mix from ``paths[rank]``; under ``held out``, the mix from ``paths[0]`` and the held-out folder from
    ``paths[rank]``; under ``rank 1 raises``, rank 1's digest read fails as an index shrinking mid-read would."""
    failing = (
        mock.patch.object(TokenShards, "content_digest", side_effect=IndexError("index 0 is out of bounds"))
        if case == "rank 1 raises" and rank == 1
        else contextlib.nullcontext()
    )
    layout = {"max_length": MAX_LENGTH, "num_steps": 64, "rows_per_step": 1, "seed": 0}
    try:
        with failing:
            if case == "held out":
                mix = build_token_mix_across_ranks([TokenSource("a", paths[0])], **layout)
                build_held_out_shards_across_ranks([TokenSource("a-val", paths[rank])], mix=mix)
            else:
                build_token_mix_across_ranks([TokenSource("a", paths[rank])], **layout)
        verdict = "built"
    except ValueError as error:
        verdict = str(error)
    with open(f"{out}/rank{rank}.txt", "w") as handle:
        handle.write(verdict)


def _verdicts(tmp_path, paths, case: str = "mix") -> list[str]:
    run_gloo_ranks(_rank_worker, WORLD_SIZE, case, [str(path) for path in paths], str(tmp_path), pg_timeout=PG_TIMEOUT)
    return [(tmp_path / f"rank{rank}.txt").read_text() for rank in range(WORLD_SIZE)]


# Per case, the knob the folders sit under and the name of the source each rank reads.
_RANK_CASES = {"mix": ("token_sources", "a"), "held out": ("eval_token_sources", "a-val")}


def test_ranks_that_see_the_same_files_build_together(folders, tmp_path):
    """Each rank reads its own copy, as on nodes with a node-local cache."""
    copy = shutil.copytree(folders["a"], tmp_path / "node1")
    assert _verdicts(tmp_path, [folders["a"], copy]) == ["built", "built"]


@pytest.mark.parametrize("case", list(_RANK_CASES))
def test_a_rank_that_sees_other_files_fails_every_rank(folders, tmp_path, case):
    knob, name = _RANK_CASES[case]
    files = {
        name: [document.tolist() for document in documents]
        for name, documents in _read_documents(folders["a"]).items()
    }
    stale = write_token_folder(tmp_path / "stale", {"00000.ds": files["00000.ds"]}, token_bytes=2)
    verdicts = _verdicts(tmp_path, [folders["a"], stale], case)
    assert verdicts[0] == verdicts[1]
    assert f"Ranks see different files or tokens under {knob}: ranks [1] (nodes [0])" in verdicts[0]
    assert f"differ from rank 0 in source '{name}' content" in verdicts[0]
    assert f"source '{name}' lost files ['00001.ds']" in verdicts[0]


def test_a_rank_whose_build_fails_fails_every_rank(folders, tmp_path):
    verdicts = _verdicts(tmp_path, [folders["a"], tmp_path / "absent"])
    assert verdicts[0] == verdicts[1]
    assert "Reading token_sources failed on 1 of 2 rank(s) [1]" in verdicts[0]
    assert "does not exist" in verdicts[0]


def test_a_rank_raising_any_error_fails_every_rank(folders, tmp_path):
    """Not only a refusal: a peer left waiting in the agreement's all-reduce would hang until the group times out."""
    verdicts = _verdicts(tmp_path, [folders["a"], folders["a"]], "rank 1 raises")
    assert verdicts[0] == verdicts[1]
    assert "Reading token_sources failed on 1 of 2 rank(s) [1]. First (rank 1): IndexError: index 0" in verdicts[0]


def test_the_open_file_limit_is_raised_to_hold_the_open_files(folders):
    """The three sources' six files hold up to twelve descriptors; the soft limit is lowered under what they and the
    reserve need for this test only."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    needed = 12 + token_shards._RESERVED_OPEN_FILES
    if hard != resource.RLIM_INFINITY and hard < needed:
        pytest.skip(f"RLIMIT_NOFILE hard limit {hard} is below {needed}")
    resource.setrlimit(resource.RLIMIT_NOFILE, (needed - 1, hard))
    try:
        mix = _mix(folders, NATURAL)
        raised = resource.getrlimit(resource.RLIMIT_NOFILE)
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
    assert sum(shards.max_open_descriptors for shards in mix.shards) == 12
    assert raised == (needed, hard)


def test_the_held_out_folders_raise_the_limit_to_hold_the_mix_files_beside_theirs(folders):
    """The mix's two sources hold up to eight descriptors and the held-out folder four. The soft limit is lowered to one
    under what all twelve and the reserve need, which the mix's alone or the held-out folder's alone do not reach."""
    mix = _mix(folders, {"a": 7, "b": 3})
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    needed = 12 + token_shards._RESERVED_OPEN_FILES
    if hard != resource.RLIM_INFINITY and hard < needed:
        pytest.skip(f"RLIMIT_NOFILE hard limit {hard} is below {needed}")
    resource.setrlimit(resource.RLIMIT_NOFILE, (needed - 1, hard))
    try:
        held_out = build_held_out_shards_across_ranks([TokenSource("c-val", str(folders["c"]))], mix=mix)
        raised = resource.getrlimit(resource.RLIMIT_NOFILE)
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
    assert sum(shards.max_open_descriptors for shards in mix.shards) == 8
    assert held_out["c-val"].max_open_descriptors == 4
    assert raised == (needed, hard)


def test_a_mix_past_the_hard_open_file_limit_is_refused(folders, monkeypatch):
    """The process reports a hard limit of 64; the real limit is never read or set."""
    lowered = []
    monkeypatch.setattr(resource, "getrlimit", lambda _which: (64, 64))
    monkeypatch.setattr(resource, "setrlimit", lambda *limits: lowered.append(limits))
    needed = 12 + token_shards._RESERVED_OPEN_FILES
    with pytest.raises(
        ValueError,
        match=(
            rf"can hold 12 file descriptors open .* needs an open-file limit of {needed}, past this process's hard "
            rf"limit of 64\. .*--ulimit nofile={needed}:{needed}.* DocumentTokenizerMerger"
        ),
    ):
        _mix(folders, NATURAL)
    assert lowered == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
