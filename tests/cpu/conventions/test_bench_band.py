"""The bench band passes a run inside its baselines and fails each way a run can fall out of them; ``record``
writes baselines only from repeats that hold together, and touches no other file in the store.

Logs are written as ``make bench`` leaves them: every rank of a row prints one ``__HALO_BENCH__`` line per key,
among the rows' other output. Two cases emit through the benchmarks' own emitters, so the band reads what they
print, from one process and from ranks printing at once.

Run: python tests/cpu/conventions/test_bench_band.py
"""

import datetime
import json
import os
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.common.equivalence import bench_band
from tests.common.equivalence.bench_band import check, load_baselines, read_logs, record
from tests.common.repo import REPO_ROOT
from tests.common.reporting import BENCH_SENTINEL, emit_benchmark, emit_save_benchmark, parse_benchmarks

TRAIN = "sft_ep_gpt-oss-20b_ep8_s4096"
SAVE = "sft_ep_gpt-oss-20b_ep8_gathered_save"
COMMAND = "tests/gpu/profiling/benchmark_sft_ep.py --ep 8 --measure_save --seq 4096 --steps 20 --warmup 3"
COMMIT = "0123456789abcdef0123456789abcdef01234567"
IMAGE = "halo:blackwell@sha256:feed"
MEASURED = datetime.date(2026, 10, 15)
WORLD_SIZE = 2
CONCURRENT_RANKS = 8
LINES_PER_RANK = 1000
# A worker as torchrun starts it (``python -u``): it waits for stdin to close, then prints its save lines.
_EMITTING_RANK = (
    "import sys\n"
    "from tests.common.reporting import emit_save_benchmark\n"
    "sys.stdin.read()\n"
    f"for _ in range({LINES_PER_RANK}):\n"
    f"    emit_save_benchmark({SAVE!r}, 41.234, 11.004)\n"
)


def _train_metrics(tokens_per_second: float, peak: float, gpu: str = "B300") -> dict:
    return {
        "tokens_per_second": tokens_per_second,
        "cluster_tokens_per_second": tokens_per_second * WORLD_SIZE,
        "peak_allocated_gb": peak,
        "training_peak_allocated_gb": peak,
        "avg_step_time_seconds": 0.38,
        "diagnostics": {"gpu_model": gpu, "precision": "bf16", "mfu_percent": 30.1},
    }


def _save_metrics(seconds: float, peak_rss: float) -> dict:
    return {"save_seconds": seconds, "save_peak_rss_gib": peak_rss}


def _write_log(
    path: Path, rows: dict[str, list[dict]], *, command: str = COMMAND, world_size: int | None = None
) -> Path:
    """A log in which rank ``i`` of each key prints ``rows[key][i]``, among unrelated output."""
    lines = ["=== gpt-oss-20b EP8 + gathered save ===", "[Rank 0] loading model"]
    for key, per_rank in rows.items():
        for rank, metrics in enumerate(per_rank):
            line = {
                "key": key,
                "rank": rank,
                "world_size": world_size or len(per_rank),
                "command": command,
                "metrics": metrics,
            }
            lines.append(f"[noise] {BENCH_SENTINEL} {json.dumps(line)}")
    path.write_text("\n".join([*lines, "PASS: gpt-oss-20b EP8 + gathered save", ""]), encoding="utf-8")
    return path


def _repeat_logs(
    tmp_path: Path, tokens: list[float], peaks: list[float], saves: list[float], save_peaks: list[float] | None = None
) -> list[Path]:
    """One log per repeat, each repeat's metrics identical on both ranks; save peaks of 12 GiB unless given."""
    save_peaks = save_peaks or [12.0] * len(tokens)
    return [
        _write_log(
            tmp_path / f"bench-{index}.log",
            {
                TRAIN: [_train_metrics(tps, peak)] * WORLD_SIZE,
                SAVE: [_save_metrics(seconds, save_peak)] * WORLD_SIZE,
            },
        )
        for index, (tps, peak, seconds, save_peak) in enumerate(zip(tokens, peaks, saves, save_peaks, strict=True))
    ]


@pytest.fixture
def baselines(tmp_path):
    """Recorded from tokens/s 1000 to 1010 around 1005 (band 984.9 .. 1025.1), peaks 49.98 to 50.02 (ceiling
    50.25), saves of 40 to 42 s around 41 (band 37.0 .. 45.0) and save peaks of 12 GiB (ceiling 12.12)."""
    logs = _repeat_logs(
        tmp_path,
        [1000.0, 1010.0, 1005.0, 1002.0, 1008.0],
        [50.0, 50.02, 49.98, 50.01, 49.99],
        [40.0, 42.0, 41.0, 40.5, 41.5],
    )
    directory = tmp_path / "baselines"
    record(read_logs(logs), commit=COMMIT, image=IMAGE, measured=MEASURED, directory=directory)
    return load_baselines(directory)


def _run(tmp_path, *, tokens=1005.0, peaks=(49.0, 50.0), save=40.5, save_rss=12.0, **log_kwargs):
    log = _write_log(
        tmp_path / "run.log",
        {
            TRAIN: [_train_metrics(tokens, peak) for peak in peaks],
            SAVE: [_save_metrics(save, save_rss)] * len(peaks),
        },
        **log_kwargs,
    )
    return read_logs([log])


def test_emitted_lines_read_back_as_one_measurement_per_key(tmp_path, monkeypatch, capsys):
    argv = ["tests/gpu/profiling/benchmark_sft_ep.py", "--ep", "8", "--measure_save"]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setenv("WORLD_SIZE", "3")
    ranks = [(1000.0, 30.0, 41.234, 11.004), (1000.0, 32.5, 41.0, 9.0), (1030.0, 31.0, 41.5, 10.0)]
    for rank, (tokens, peak, seconds, rss) in enumerate(ranks):
        monkeypatch.setenv("RANK", str(rank))
        callback = SimpleNamespace(
            tps=SimpleNamespace(avg_tokens_per_second=tokens, avg_cluster_tokens_per_second=3 * tokens),
            memory=SimpleNamespace(peak_allocated_gb=peak, training_peak_allocated_gb=peak),
            time=SimpleNamespace(avg_step_time_seconds=0.4),
            gpu_model="B300",
            precision="bf16",
            flops_model=SimpleNamespace(local_params=11e9, sparsity_factor=0.125),
            mfu=SimpleNamespace(avg_mfu_percent=30.0, avg_tflops_per_sec=700.0),
            smfu=SimpleNamespace(avg_smfu_percent=12.0, avg_smfu_tflops_per_sec=300.0),
        )
        emit_benchmark(TRAIN, callback)
        emit_save_benchmark(SAVE, seconds, rss)
    log = tmp_path / "bench.log"
    log.write_text("unrelated output\n" + capsys.readouterr().out, encoding="utf-8")

    run = read_logs([log])

    assert sorted(run) == [SAVE, TRAIN]
    train, save = run[TRAIN][0], run[SAVE][0]
    assert (train.command, train.world_size) == (shlex.join(argv), 3)
    assert train.metrics["tokens_per_second"] == 1010.0, "a throughput is the mean over ranks"
    assert train.metrics["peak_allocated_gb"] == 32.5, "a peak is the highest rank's"
    assert train.metrics["diagnostics"]["gpu_model"] == "B300"
    assert save.metrics == {"save_seconds": 41.24, "save_peak_rss_gib": 11.0}, "a save line carries two decimals"


def test_lines_of_ranks_printing_at_once_read_back_whole():
    """torchrun's workers print unbuffered into one stream, and every rank prints its lines as training ends."""
    start_read, start_write = os.pipe()
    out_read, out_write = os.pipe()
    workers = [
        subprocess.Popen(
            [sys.executable, "-u", "-c", _EMITTING_RANK],
            stdin=start_read,
            stdout=out_write,
            cwd=REPO_ROOT,
            env={**os.environ, "RANK": str(rank), "WORLD_SIZE": str(CONCURRENT_RANKS)},
        )
        for rank in range(CONCURRENT_RANKS)
    ]
    os.close(start_read)
    os.close(out_write)
    os.close(start_write)
    with os.fdopen(out_read, encoding="utf-8") as stream:
        printed = stream.read()
    assert [worker.wait() for worker in workers] == [0] * CONCURRENT_RANKS

    lines = parse_benchmarks(printed)

    assert Counter(line["rank"] for line in lines) == dict.fromkeys(range(CONCURRENT_RANKS), LINES_PER_RANK)
    assert all(line["metrics"] == {"save_seconds": 41.23, "save_peak_rss_gib": 11.0} for line in lines)


def test_a_run_inside_the_band_passes(tmp_path, baselines):
    verdict = check(_run(tmp_path, tokens=986.0, peaks=(49.0, 50.24), save=44.9, save_rss=12.12), baselines)

    assert verdict["verdict"] == "pass", verdict["failures"]
    assert verdict["keys"][TRAIN]["metrics"]["tokens_per_second"] == {
        "run": 986.0,
        "band": [984.9, 1025.1],
        "verdict": "within",
    }
    assert verdict["keys"][TRAIN]["metrics"]["peak_allocated_gb"] == {
        "run": 50.24,
        "ceiling": 50.25,
        "verdict": "within",
    }
    assert verdict["keys"][SAVE]["metrics"]["save_seconds"]["band"] == [37.0, 45.0]
    assert verdict["keys"][SAVE]["metrics"]["save_peak_rss_gib"]["ceiling"] == 12.12


def test_a_slower_run_fails(tmp_path, baselines):
    verdict = check(_run(tmp_path, tokens=983.0), baselines)

    assert verdict["verdict"] == "fail"
    assert verdict["failures"] == [f"{TRAIN}: tokens_per_second 983.0 is below its band 984.9 .. 1025.1"]


def test_a_run_past_the_faster_side_fails_too(tmp_path, baselines):
    verdict = check(_run(tmp_path, tokens=1026.0, save=36.9), baselines)

    assert verdict["verdict"] == "fail"
    assert verdict["failures"] == [
        f"{SAVE}: save_seconds 36.9 is below its band 37.0 .. 45.0",
        f"{TRAIN}: tokens_per_second 1026.0 is above its band 984.9 .. 1025.1",
    ]


def test_the_band_and_the_ceiling_widen_with_the_baseline_spread(tmp_path):
    logs = _repeat_logs(tmp_path, [1000.0, 1040.0, 1020.0, 1010.0, 1030.0], [50.0, 50.4, 50.2, 50.1, 50.3], [40.0] * 5)
    directory = tmp_path / "baselines"
    record(read_logs(logs), commit=COMMIT, image=IMAGE, measured=MEASURED, directory=directory)
    widened = load_baselines(directory)

    assert check(_run(tmp_path, tokens=945.0, peaks=(51.0, 50.0), save=40.0), widened)["verdict"] == "pass"
    failing = check(_run(tmp_path, tokens=935.0, peaks=(51.01, 50.0), save=40.0), widened)
    assert failing["failures"] == [
        f"{TRAIN}: tokens_per_second 935.0 is below its band 940.0 .. 1100.0",
        f"{TRAIN}: peak_allocated_gb 51.01 is above its ceiling 51.0",
    ]


def test_a_slower_save_fails(tmp_path, baselines):
    verdict = check(_run(tmp_path, save=45.1), baselines)

    assert verdict["failures"] == [f"{SAVE}: save_seconds 45.1 is above its band 37.0 .. 45.0"]


def test_a_peak_above_the_ceiling_fails_on_any_rank(tmp_path, baselines):
    assert check(_run(tmp_path, peaks=(50.25, 40.0)), baselines)["verdict"] == "pass"

    higher_rank = check(_run(tmp_path, peaks=(40.0, 50.26)), baselines)
    assert higher_rank["failures"] == [f"{TRAIN}: peak_allocated_gb 50.26 is above its ceiling 50.25"]

    hungrier_save = check(_run(tmp_path, save_rss=12.13), baselines)
    assert hungrier_save["failures"] == [f"{SAVE}: save_peak_rss_gib 12.13 is above its ceiling 12.12"]


def test_an_unmeasured_baseline_and_an_unbaselined_key_fail(tmp_path, baselines):
    log = _write_log(tmp_path / "run.log", {TRAIN: [_train_metrics(1005.0, 50.0)] * 2, "sft_new_s4096": [{}] * 2})

    verdict = check(read_logs([log]), baselines)

    assert verdict["failures"] == [
        f"{SAVE}: has a baseline the run did not measure",
        "sft_new_s4096: measured, but has no baseline",
    ]


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"command": COMMAND.replace("--steps 20", "--steps 10")}, "the run's command"),
        ({"world_size": 4}, "the run spans 4 ranks, the baseline 2"),
        ({"gpu": "H100"}, "the run measured on H100, the baseline on B300"),
    ],
)
def test_a_run_of_another_setup_is_not_comparable(tmp_path, baselines, change, reason):
    ranks = change.get("world_size", WORLD_SIZE)
    rows = {TRAIN: [_train_metrics(1005.0, 50.0, change.get("gpu", "B300"))] * ranks}
    log = _write_log(tmp_path / "run.log", rows, command=change.get("command", COMMAND))
    baselines = {TRAIN: baselines[TRAIN]}

    verdict = check(read_logs([log]), baselines)

    assert verdict["verdict"] == "fail"
    assert len(verdict["failures"]) == 1
    assert verdict["failures"][0].startswith(f"{TRAIN}: not comparable with its baseline: {reason}")


def test_a_gated_metric_the_run_drops_fails(tmp_path, baselines):
    log = _write_log(tmp_path / "run.log", {SAVE: [{"save_seconds": 40.0}] * WORLD_SIZE})

    verdict = check(read_logs([log]), {SAVE: baselines[SAVE]})

    assert verdict["failures"] == [f"{SAVE}: save_peak_rss_gib is in the baseline only"]


def test_a_key_printed_twice_by_one_rank_raises(tmp_path):
    log = _write_log(tmp_path / "run.log", {TRAIN: [_train_metrics(1005.0, 50.0)] * 2})
    log.write_text(log.read_text() + log.read_text(), encoding="utf-8")

    with pytest.raises(ValueError, match="rank 0 printed .* twice: two rows emit one key"):
        read_logs([log])


def test_an_unreadable_line_raises(tmp_path):
    log = _write_log(tmp_path / "run.log", {TRAIN: [_train_metrics(1005.0, 50.0)] * 2})
    log.write_text(log.read_text() + f'{BENCH_SENTINEL} {{"key": "{TRAIN}", "rank": 1\n', encoding="utf-8")

    with pytest.raises(ValueError, match="unreadable benchmark line"):
        read_logs([log])


def test_a_key_missing_a_rank_raises(tmp_path):
    log = _write_log(tmp_path / "run.log", {TRAIN: [_train_metrics(1005.0, 50.0)]}, world_size=2)

    with pytest.raises(ValueError, match="not one launch's ranks: ranks \\[0\\], world sizes \\[2\\]"):
        read_logs([log])


def test_record_writes_each_key_from_its_repeats(tmp_path):
    logs = _repeat_logs(
        tmp_path,
        [1000.0, 1010.0, 1005.0, 1002.0, 1008.0],
        [50.0, 50.02, 49.98, 50.01, 49.99],
        [40.0, 42.0, 41.0, 40.5, 41.5],
    )
    directory = tmp_path / "baselines"

    written = record(read_logs(logs), commit=COMMIT, image=IMAGE, measured=MEASURED, directory=directory)

    assert sorted(path.name for path in written) == [f"{SAVE}.json", f"{TRAIN}.json"]
    stored = json.loads((directory / f"{TRAIN}.json").read_text())
    assert stored["provenance"] == {
        "commit": COMMIT,
        "image": IMAGE,
        "measured": "2026-10-15",
        "world_size": WORLD_SIZE,
        "command": COMMAND,
    }
    assert [repeat["tokens_per_second"] for repeat in stored["repeats"]] == [1000.0, 1010.0, 1005.0, 1002.0, 1008.0]
    recorded = load_baselines(directory)
    for log in logs:
        assert check(read_logs([log]), recorded)["verdict"] == "pass", "a recorded repeat falls outside its band"


@pytest.mark.parametrize(
    ("tokens", "peaks", "commit", "refusal"),
    [
        (
            [1000.0, 1005.0, 1002.0, 1003.0, 1100.0],
            [50.0] * 5,
            COMMIT,
            "repeat 5's tokens_per_second 1100.0 lies outside the band of the others",
        ),
        (
            [1000.0, 1005.0, 1002.0, 1003.0, 1001.0],
            [50.0] * 4 + [51.0],
            COMMIT,
            "repeat 5's peak_allocated_gb 51.0 lies above the ceiling of the others",
        ),
        ([1000.0, 1005.0, 1002.0, 1003.0], [50.0] * 4, COMMIT, "4 repeats, fewer than 5"),
        ([1000.0, 1005.0, 1002.0, 1003.0, 1001.0], [50.0] * 5, f"{COMMIT}-dirty", "is not one"),
    ],
    ids=["outlying throughput", "outlying peak", "too few repeats", "modified tree"],
)
def test_record_refuses_and_writes_nothing(tmp_path, tokens, peaks, commit, refusal):
    directory = tmp_path / "baselines"
    directory.mkdir()
    kept = directory / "sft_kept_s4096.json"
    kept.write_text("{}", encoding="utf-8")
    run = read_logs(_repeat_logs(tmp_path, tokens, peaks, [40.0] * len(tokens)))

    with pytest.raises(ValueError, match=refusal):
        record(run, commit=commit, image=IMAGE, measured=MEASURED, directory=directory)

    assert [path.name for path in directory.iterdir()] == [kept.name]


@pytest.mark.parametrize(
    ("tokens", "peaks", "saves", "save_peaks"),
    [
        ([1000.0] * 5, [50.0] * 4 + [48.0], [40.0] * 5, [12.0] * 5),
        ([1000.0, 1005.0, 1010.0, 1002.0, 1040.0], [50.0] * 5, [40.0] * 5, [12.0] * 5),
        ([1000.0] * 5, [50.0] * 5, [40.0] * 4 + [42.0], [12.0] * 5),
        ([1000.0] * 5, [50.0] * 5, [40.0] * 5, [12.0] * 4 + [12.1]),
    ],
    ids=[
        "a lower peak",
        "a throughput past three times the others' spread, within four times",
        "a save 5% slower than the others",
        "a save peak 0.8% over the others",
    ],
)
def test_record_keeps_a_repeat_that_varies_as_runs_do(tmp_path, tokens, peaks, saves, save_peaks):
    """A peak below the others widens the spread its ceiling is drawn from, so the ceiling still clears it; a
    rate is refused only past :data:`bench_band.OUTLIER_FACTOR` times the others' spread (1040 against 1000 to
    1010 lies 35.75 off their mean: past three times their spread of 10, within four times); and the save's host
    metrics carry their own margins (6% on its time, 1% on its peak)."""
    directory = tmp_path / "baselines"
    logs = _repeat_logs(tmp_path, tokens, peaks, saves, save_peaks)

    record(read_logs(logs), commit=COMMIT, image=IMAGE, measured=MEASURED, directory=directory)

    recorded = load_baselines(directory)
    for log in logs:
        assert check(read_logs([log]), recorded)["verdict"] == "pass", "a recorded repeat falls outside its band"


def test_record_replaces_every_other_baseline_and_nothing_else(tmp_path):
    directory = tmp_path / "baselines"
    (directory / "corpus").mkdir(parents=True)
    (directory / "corpus" / "manifest.json").write_text("{}", encoding="utf-8")
    (directory / "notes.txt").write_text("kept", encoding="utf-8")
    (directory / "sft_dropped_row_s4096.json").write_text("{}", encoding="utf-8")
    (directory / bench_band.STORE_PLACEHOLDER).touch()
    run = read_logs(_repeat_logs(tmp_path, [1000.0, 1010.0, 1005.0, 1002.0, 1008.0], [50.0] * 5, [40.0] * 5))

    record(run, commit=COMMIT, image=IMAGE, measured=MEASURED, directory=directory)

    assert sorted(path.name for path in directory.iterdir()) == [
        "corpus",
        "notes.txt",
        f"{SAVE}.json",
        f"{TRAIN}.json",
    ]
    assert (directory / "corpus" / "manifest.json").read_text(encoding="utf-8") == "{}"
    assert sorted(load_baselines(directory)) == [SAVE, TRAIN], "a file beside the baselines is read as one"


@pytest.mark.parametrize(
    ("edit", "problem"),
    [
        (lambda data: data.update(key="sft_other_s4096"), "its key 'sft_other_s4096' is not its file name"),
        (lambda data: data["provenance"].pop("command"), "its provenance lacks \\['command'\\]"),
        (lambda data: data["repeats"].pop(), "it holds 4 repeats, fewer than 5"),
        (lambda data: data["repeats"][0].update(tokens_per_second=900.0), "repeat 1's tokens_per_second 900.0"),
    ],
    ids=["misnamed", "incomplete provenance", "too few repeats", "outlying repeat"],
)
def test_a_malformed_baseline_raises(tmp_path, edit, problem):
    directory = tmp_path / "baselines"
    record(
        read_logs(_repeat_logs(tmp_path, [1000.0, 1010.0, 1005.0, 1002.0, 1008.0], [50.0] * 5, [40.0] * 5)),
        commit=COMMIT,
        image=IMAGE,
        measured=MEASURED,
        directory=directory,
    )
    path = directory / f"{TRAIN}.json"
    data = json.loads(path.read_text())
    edit(data)
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match=problem):
        load_baselines(directory)


def test_check_exits_non_zero_on_a_failure_and_writes_its_evidence(tmp_path, baselines, monkeypatch):
    monkeypatch.setattr(bench_band, "BASELINE_DIR", tmp_path / "baselines")
    monkeypatch.setattr(bench_band, "EVIDENCE_DIR", tmp_path / "evidence")
    within = _write_log(
        tmp_path / "within.log", {TRAIN: [_train_metrics(1000.0, 50.0)] * 2, SAVE: [_save_metrics(41.0, 12.0)] * 2}
    )
    slower = _write_log(
        tmp_path / "slower.log", {TRAIN: [_train_metrics(900.0, 50.0)] * 2, SAVE: [_save_metrics(41.0, 12.0)] * 2}
    )
    arguments = ["--revision", "feedface-dirty", "--image", IMAGE]

    assert bench_band.main(["check", str(within), *arguments]) == 0
    assert bench_band.main(["check", str(slower), *arguments]) == 1

    evidence = json.loads((tmp_path / "evidence" / "bench_band-feedface-dirty.json").read_text())
    assert (evidence["verdict"], evidence["base"], evidence["head"]) == ("fail", COMMIT, "feedface-dirty")
    assert evidence["failures"] == [f"{TRAIN}: tokens_per_second 900.0 is below its band 984.9 .. 1025.1"]
    assert evidence["logs"] == [str(slower)]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
