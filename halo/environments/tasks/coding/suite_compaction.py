"""Test-suite compaction for the code-contest pools: a bulky corpus (HardTests' encoded suites, open-r1/codeforces'
generated tests) reduced to one capped row per problem, written as parquet parts (``key``, ``tests``, ``checker``)
stamped with the caps that built them, and the parquet-part listing both the compaction and the pool preparation's
tests-table join read. Behind ``scripts/environments/preparation/compact_code_tests.py``."""

import json
import logging
import os
import zlib
from collections import defaultdict
from collections.abc import Iterator
from multiprocessing import Pool
from pathlib import Path
from typing import NamedTuple

import pyarrow as pa
import pyarrow.parquet as pq

from halo.data.pipeline.coordinated_map import resolve_map_num_proc
from halo.environments.tasks.coding.contest_datasets import decode_test_payload

logger = logging.getLogger(__name__)

_HARDTESTS_GROUP_ORDER = ("LLMGen", "HackGen", "RPGen", "SPGen")
_OUTPUT_SCHEMA = pa.schema([("key", pa.string()), ("tests", pa.string()), ("checker", pa.string())])
CAP_DEFAULTS = {"max_tests": 40, "max_test_bytes": 262_144, "max_large_tests": 2, "max_large_bytes": 4_000_000}
# Parquet key-value metadata slot holding the caps a part was written with.
_CAPS_METADATA_KEY = b"caps"
# A suite that fails to decode is skipped and counted; a payload naming a class or callable (the
# unpickler's refusal) is not a decode failure and propagates.
_UNDECODABLE = (ValueError, TypeError, zlib.error, EOFError)


def parquet_parts(directory: str | Path, *, non_empty: bool = False) -> list[str]:
    """The parquet files under ``directory`` in sorted order; ``non_empty`` leaves out zero-row parts.
    Raises when none remain, since every caller needs at least one."""
    parts = sorted(
        str(p) for p in Path(directory).glob("*.parquet") if not non_empty or pq.read_metadata(p).num_rows > 0
    )
    if not parts:
        raise SystemExit(f"no{' non-empty' if non_empty else ''} parquet parts under {directory}")
    return parts


def cap_tests(
    tests: list[dict[str, str]],
    *,
    max_tests: int,
    max_test_bytes: int,
    max_large_tests: int,
    max_large_bytes: int,
) -> list[dict[str, str]]:
    """The first ``max_tests`` tests within the size caps, in the given order. A test above
    ``max_test_bytes`` takes one of the ``max_large_tests`` slots while it stays under
    ``max_large_bytes``, and is dropped otherwise."""
    kept: list[dict[str, str]] = []
    large_slots = max_large_tests
    for test in tests:
        inp, out = test.get("input") or "", test.get("output") or ""
        size = len(inp.encode("utf-8")) + len(out.encode("utf-8"))
        if size > max_test_bytes:
            if large_slots == 0 or size > max_large_bytes:
                continue
            large_slots -= 1
        kept.append({"input": inp, "output": out})
        if len(kept) >= max_tests:
            break
    return kept


def _flatten_indices(mapping: dict) -> list[int]:
    """HardTests ``mapping`` positions in group order; nested groups (one list per generator) flattened."""
    order: list[int] = []
    for group in _HARDTESTS_GROUP_ORDER:
        entries = mapping.get(group) or []
        for entry in entries:
            order.extend(entry if isinstance(entry, list) else [entry])
    return order


def _iter_rows(path: str, columns: list[str], batch_size: int) -> Iterator[dict]:
    """Rows of a parquet shard a few at a time: a suite can decode to hundreds of megabytes, so a whole
    shard in memory per worker is what an out-of-memory host looks like."""
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=columns):
        yield from batch.to_pylist()


class ShardResult(NamedTuple):
    """One compacted shard: its rows, the suites that failed to decode, the suites the caps emptied."""

    rows: list[dict]
    undecodable: int
    emptied: int


def compact_hardtests_shard(path: str, caps: dict) -> ShardResult:
    """One HardTests shard: decode each problem's suite, order it by generator group, cap it."""
    rows: list[dict] = []
    undecodable = emptied = 0
    for row in _iter_rows(path, ["pid", "test_cases_kit", "mapping", "test_cases"], batch_size=2):
        try:
            decoded = decode_test_payload(row["test_cases"])
        except _UNDECODABLE as exc:
            undecodable += 1
            logger.warning("%s: undecodable suite (%s), skipped", row["pid"], type(exc).__name__)
            continue
        order = _flatten_indices(row.get("mapping") or {}) or list(range(len(decoded)))
        ordered = [decoded[i] for i in order if 0 <= i < len(decoded)]
        tests = cap_tests(ordered, **caps)
        if not tests:
            emptied += 1
            continue
        kit = row.get("test_cases_kit") or {}
        judge = (kit.get("output_judging_function") or "").strip() or None
        rows.append({"key": row["pid"], "tests": json.dumps(tests), "checker": judge})
    return ShardResult(rows, undecodable, emptied)


def compact_codeforces_generated_shard(path: str, caps: dict) -> ShardResult:
    """One open-r1 ``generated_tests`` shard (a contest): group its rows per problem in ``test_i`` order."""
    per_problem: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for row in _iter_rows(path, ["problem_id", "input", "output", "test_i"], batch_size=64):
        per_problem[row["problem_id"]].append((int(row["test_i"]), {"input": row["input"], "output": row["output"]}))
    rows: list[dict] = []
    emptied = 0
    for key, entries in per_problem.items():
        tests = cap_tests([t for _, t in sorted(entries, key=lambda e: e[0])], **caps)
        if tests:
            rows.append({"key": key, "tests": json.dumps(tests), "checker": None})
        else:
            emptied += 1
    return ShardResult(rows, 0, emptied)


COMPACTORS = {"hardtests": compact_hardtests_shard, "codeforces_generated": compact_codeforces_generated_shard}


def _part_caps(out_path: str) -> dict | None:
    """The caps stamped on an existing part, or ``None`` when the file carries none."""
    raw = (pq.read_metadata(out_path).metadata or {}).get(_CAPS_METADATA_KEY)
    return json.loads(raw) if raw else None


def _compact_one(args: tuple[str, str, str, dict]) -> tuple[str, int, int, int]:
    """Compact one shard into its part: ``(shard, problems kept, undecodable suites, emptied suites)``."""
    source, path, out_path, caps = args
    if Path(out_path).exists():
        if _part_caps(out_path) == caps:
            return path, pq.read_metadata(out_path).num_rows, 0, 0
        logger.info("%s: rebuilt, its caps differ from this run's", Path(out_path).name)
    result = COMPACTORS[source](path, caps)
    # No part for a shard that kept nothing: a zero-row parquet file breaks the reader downstream, and a
    # part left from a run with wider caps would join tests this run's caps excluded.
    if not result.rows:
        Path(out_path).unlink(missing_ok=True)
    else:
        table = pa.Table.from_pylist(result.rows, schema=_OUTPUT_SCHEMA)
        table = table.replace_schema_metadata({_CAPS_METADATA_KEY: json.dumps(caps).encode()})
        # Written beside, then moved in: a part is either complete or absent, never a truncated file.
        tmp_path = f"{out_path}.tmp"
        pq.write_table(table, tmp_path, compression="zstd")
        os.replace(tmp_path, out_path)
    return path, len(result.rows), result.undecodable, result.emptied


def compact_tests_table(source: str, input_dir: str, output_dir: str, caps: dict, num_proc: int | None) -> None:
    """Compact every parquet shard under ``input_dir`` of the ``source`` corpus into its part under ``output_dir``,
    one worker per shard, keeping a part already stamped with ``caps`` and logging the totals."""
    shards = parquet_parts(input_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(source, shard, str(out / f"{Path(shard).stem}.parquet"), caps) for shard in shards]
    total = undecodable = emptied = 0
    with Pool(resolve_map_num_proc(num_proc)) as pool:
        for path, kept, skipped, dropped in pool.imap_unordered(_compact_one, jobs):
            total += kept
            undecodable += skipped
            emptied += dropped
            logger.info("%s: %d problems", Path(path).name, kept)
    logger.info(
        "compacted %d problems from %d shards into %s (%d undecodable suites skipped, %d suites emptied by the caps)",
        total,
        len(shards),
        out,
        undecodable,
        emptied,
    )
