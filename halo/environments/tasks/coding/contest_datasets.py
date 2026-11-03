"""Contest dataset adapters: row transforms into the code-contests env's ``{prompt, answer}`` shape, plus a ``keep``
predicate for rows a stdin/stdout grader cannot grade, registered in :data:`CODE_DATASET_ADAPTERS`."""

import base64
import io
import json
import pickle
import re
import zlib
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from datasets import load_dataset
from huggingface_hub import hf_hub_download

# Cumulative releases (test.jsonl = v1, each later appends); the newest file is read first.
_LCB_RELEASE_FILES: dict[str, list[str]] = {
    f"release_v{v}": [f"test{'' if i == 1 else i}.jsonl" for i in range(1, v + 1)] for v in range(1, 7)
}
# LiveCodeBench's ``leetcode`` rows are functional and ``keep_livecodebench`` drops them, so a selection naming that
# platform would score nothing.
_LCB_GRADABLE_PLATFORMS = ("atcoder", "codeforces")
# Columns ``prepare_code_dataset.py`` joins onto a row from a compacted tests table.
JOINED_TESTS_FIELD = "joined_tests"
JOINED_CHECKER_FIELD = "joined_checker"
# The one split HLCE ships; its loader reads it whatever split it is passed.
_HLCE_SPLIT = "train"

# HardTests difficulty on the Codeforces rating scale, so one rating bound cuts the whole pool. Luogu's levels and the
# coarse AtCoder/TACO labels map to a representative rating. Luogu level 0 is "unknown" and falls through.
_LUOGU_LEVEL_RATING = {1: 800, 2: 1000, 3: 1300, 4: 1600, 5: 2000, 6: 2400, 7: 2900}
_HARDTESTS_LEVEL_RATING: dict[str, dict[str, int]] = {
    "atcoder": {"easy": 1000, "medium": 1400, "hard": 1800, "very hard": 2400},
    "taco": {"easy": 1000, "medium": 1400, "medium_hard": 1700, "hard": 2100, "very_hard": 2500},
}
_LIMIT_NUMBER = re.compile(r"[0-9]+(?:\.[0-9]+)?")
# A time limit this large is milliseconds on the judges that write them as ``N s``.
_MS_THRESHOLD = 50.0
_MEMORY_LIMIT = re.compile(r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*(GI?B|MI?B|KI?B|BYTES?|B)?\b", re.IGNORECASE)
_MEMORY_UNIT_MB = {
    "GB": 1024.0,
    "GIB": 1024.0,
    "MB": 1.0,
    "MIB": 1.0,
    "KB": 1 / 1024,
    "KIB": 1 / 1024,
    "BYTE": 1 / 1024**2,
    "BYTES": 1 / 1024**2,
    "B": 1 / 1024**2,
}
# Outside this range a scraped limit is not a judge's (CodeChef rows carry ``50000 bytes``); the prompt omits it.
_MIN_MEMORY_LIMIT_MB = 1.0
_MAX_MEMORY_LIMIT_MB = 8192.0
# Runs HardTests' judging function under the env's checker contract: file paths on argv (input, expected, got), a
# trailing 1/0 on stdout. A judging function that raises rejects.
_HARDTESTS_CHECKER_DRIVER = """

if __name__ == "__main__":
    import sys

    _inp, _exp, _got = (open(p, encoding="utf-8", errors="replace").read() for p in sys.argv[1:4])
    try:
        _ok = bool(output_judging_function(_inp, _got, _exp))
    except Exception:
        _ok = False
    print(1 if _ok else 0)
"""


def _format_problem_statement(
    label: str,
    title: str,
    *,
    time_limit_s: float | None = None,
    memory_limit_mb: float | None = None,
    description: str | None = None,
    input_spec: str | None = None,
    output_spec: str | None = None,
    examples: Iterable[tuple[int, str, str]] = (),
    note: str | None = None,
) -> str:
    """``examples`` are pre-numbered ``(index, input, output)`` triples, so a skipped malformed example keeps the
    others' numbering."""
    heading = (
        f"# {label}. {title}" if (label and title) else (f"# {label or title}" if (label or title) else "# Problem")
    )
    parts: list[str] = [heading]

    limits = []
    if time_limit_s:
        limits.append(f"time limit per test: {time_limit_s:g} s")
    if memory_limit_mb:
        limits.append(f"memory limit per test: {memory_limit_mb:.0f} MB")
    if limits:
        parts.append("\n".join(limits))

    if description:
        parts.append(description.strip())
    if input_spec:
        parts.append("## Input\n" + input_spec.strip())
    if output_spec:
        parts.append("## Output\n" + output_spec.strip())

    for i, example_input, example_output in examples:
        parts.append(
            f"## Example {i}\nInput:\n```\n{example_input.rstrip()}\n```\nOutput:\n```\n{example_output.rstrip()}\n```"
        )

    if note:
        parts.append("## Note\n" + note.strip())

    return "\n\n".join(parts)


def format_codeforces_prompt(row: dict[str, Any]) -> str:
    return _format_problem_statement(
        (row.get("index") or "").strip(),
        (row.get("title") or "").strip(),
        time_limit_s=row.get("time_limit"),
        memory_limit_mb=row.get("memory_limit"),
        description=row.get("description"),
        input_spec=row.get("input_format"),
        output_spec=row.get("output_format"),
        examples=[(i, ex.get("input", ""), ex.get("output", "")) for i, ex in enumerate(row.get("examples") or [], 1)],
        note=row.get("note"),
    )


def _stdin_tests(items: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    return [{"input": t.get("input", ""), "output": t.get("output", "")} for t in items]


def joined_tests(row: dict[str, Any]) -> list[dict[str, str]]:
    """Tests joined onto the row from a compacted tests table; empty when the row had no match."""
    raw = row.get(JOINED_TESTS_FIELD)
    if not raw:
        return []
    tests = json.loads(raw) if isinstance(raw, str) else raw
    return _stdin_tests(tests)


def _test_input(test: dict[str, Any]) -> str:
    """Whitespace runs collapsed: one input however a source spaces it."""
    return " ".join((test.get("input") or "").split())


def _codeforces_graded_tests(row: dict[str, Any]) -> list[dict[str, Any]]:
    """``official_tests`` (only the ones Codeforces shows untruncated) plus the joined generated tests."""
    return list(row.get("official_tests") or []) + joined_tests(row)


def codeforces_examples_only(row: dict[str, Any]) -> bool:
    """Whether a row is graded only on its statement's example inputs: its official tests are the samples (half the
    ``verifiable`` test split) or absent, and no joined suite adds more. The prompt shows those answers, so a program
    printing them passes, under a special judge too where the official output is another valid answer."""
    examples = {_test_input(test) for test in row.get("examples") or []}
    return all(_test_input(test) in examples for test in _codeforces_graded_tests(row))


def pack_codeforces_verification(row: dict[str, Any]) -> dict[str, Any]:
    """Falls back to ``examples`` where the row ships no tests. ``None`` checker => token grading."""
    tests = _codeforces_graded_tests(row) or _stdin_tests(row.get("examples") or [])
    return {
        "tests": tests,
        "checker": row.get("generated_checker") or None,
        "time_limit": float(row.get("time_limit") or 0.0) or None,
    }


def _keep_codeforces(row: dict[str, Any]) -> bool:
    if not row.get("executable"):
        return False
    if (row.get("input_mode") or "stdio") != "stdio":
        return False
    if row.get("interaction_format"):
        return False
    return len(pack_codeforces_verification(row)["tests"]) > 0


def _format_deepcoder_prompt(row: dict[str, Any]) -> str:
    return (row.get("problem") or "").strip()


def _parse_deepcoder_tests(row: dict[str, Any]) -> list[dict[str, str]]:
    """Functional (``fn_name``) specs yield no tests."""
    raw = row.get("tests")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return []
    if not isinstance(raw, dict) or raw.get("fn_name"):
        return []

    inputs = raw.get("inputs") or []
    outputs = raw.get("outputs") or []

    def _stringify(value: Any) -> str:
        if isinstance(value, list):
            return "\n".join(str(v) for v in value)
        return str(value)

    return [
        {"input": _stringify(inp), "output": _stringify(out)}
        for inp, out in zip(inputs, outputs, strict=False)
        if isinstance(inp, str)
    ]


def _pack_deepcoder_verification(row: dict[str, Any]) -> dict[str, Any]:
    return {"tests": _parse_deepcoder_tests(row), "checker": None, "time_limit": None}


def _keep_deepcoder(row: dict[str, Any]) -> bool:
    return len(_parse_deepcoder_tests(row)) > 0


class _StringOnlyUnpickler(pickle.Unpickler):
    """The test payloads hold one JSON string, so a payload that references a class or callable is refused."""

    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(f"test payload references {module}.{name}; only a JSON string is expected")


def decode_test_payload(raw: str) -> list[dict[str, Any]]:
    """Plain JSON, or base64-of-zlib-of-pickle around a JSON string (LiveCodeBench private tests, HardTests suites)."""
    try:
        return json.loads(raw)
    except ValueError:
        payload = zlib.decompress(base64.b64decode(raw.encode("utf-8")))
        return json.loads(_StringOnlyUnpickler(io.BytesIO(payload)).load())


def _lcb_stdin_tests(row: dict[str, Any]) -> list[dict[str, str]]:
    return _stdin_tests(
        t
        for field in ("public_test_cases", "private_test_cases")
        for t in decode_test_payload(row.get(field) or "[]")
        if t.get("testtype") == "stdin"
    )


def format_titled_statement(row: dict[str, Any]) -> str:
    """LiveCodeBench and HLCE share this row shape; ``question_content`` already carries the I/O spec."""
    title = (row.get("question_title") or "").strip()
    body = (row.get("question_content") or "").strip()
    return f"# {title}\n\n{body}" if title else body


def pack_livecodebench_verification(row: dict[str, Any]) -> dict[str, Any]:
    return {"tests": _lcb_stdin_tests(row), "checker": None, "time_limit": None}


def keep_livecodebench(row: dict[str, Any]) -> bool:
    return len(_lcb_stdin_tests(row)) > 0


def _load_livecodebench(dataset: str, config: str | None, split: str) -> Iterator[dict[str, Any]]:
    """Raw rows from release ``config``'s files, newest first, each in stored order: the order the eval and the
    re-grader index. ``split`` is ignored. Bypasses ``load_dataset``, whose loading script datasets 4.x won't run."""
    files = _LCB_RELEASE_FILES.get(config or "release_v6")
    if files is None:
        raise ValueError(f"unknown LiveCodeBench release {config!r}; expected one of {sorted(_LCB_RELEASE_FILES)}")
    for filename in reversed(files):
        path = hf_hub_download(dataset, filename, repo_type="dataset")
        with open(path) as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def _livecodebench_contest_date(row: dict[str, Any]) -> date:
    """A row without a date raises: a window cannot place it, and dropping it would shrink the window unannounced."""
    raw = row.get("contest_date")
    try:
        return datetime.fromisoformat(raw).date()
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"LiveCodeBench row {row.get('question_id')!r} carries no ISO contest_date (got {raw!r})"
        ) from exc


def _icpc_time_limit_s(row: dict[str, Any]) -> float | None:
    return (row.get("time_limit_ms") or 0) / 1000 or None


def format_icpc_prompt(row: dict[str, Any]) -> str:
    return _format_problem_statement(
        (row.get("problem_label") or "").strip(),
        (row.get("title") or "").strip(),
        time_limit_s=_icpc_time_limit_s(row),
        memory_limit_mb=row.get("memory_limit_mb"),
        description=row.get("description"),
        input_spec=row.get("input"),
        output_spec=row.get("output"),
        # Malformed pairs are dropped but still consume their number, as in the raw row.
        examples=[
            (i, str(ex[0]), str(ex[1]))
            for i, ex in enumerate(row.get("examples") or [], 1)
            if isinstance(ex, (list, tuple)) and len(ex) == 2
        ],
        note=row.get("note"),
    )


def _icpc_tests(row: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"input": str(tc[0]), "output": str(tc[1])}
        for tc in (row.get("test_cases") or [])
        if isinstance(tc, (list, tuple)) and len(tc) == 2
    ]


def pack_icpc_verification(row: dict[str, Any]) -> dict[str, Any]:
    return {"tests": _icpc_tests(row), "checker": None, "time_limit": _icpc_time_limit_s(row)}


def keep_icpc(row: dict[str, Any]) -> bool:
    """Drops ``spj`` rows: C++ special judges we can't run."""
    return row.get("type") == "traditional" and len(_icpc_tests(row)) > 0


def _load_icpc(dataset: str, config: str | None, split: str) -> Iterator[dict[str, Any]]:
    """Streaming avoids pulling the full multi-GB test payload for a small probe."""
    yield from load_dataset(dataset, config, split=split, streaming=True)


def hardtests_rating(row: dict[str, Any]) -> int | None:
    by_source = {r.get("source"): r for r in row.get("difficulty_ratings") or [] if r.get("source")}
    codeforces = by_source.get("codeforces")
    if codeforces and codeforces.get("score"):
        return int(codeforces["score"])
    luogu = by_source.get("luogu")
    if luogu and luogu.get("score") and (rating := _LUOGU_LEVEL_RATING.get(int(luogu["score"]))) is not None:
        return rating
    for source, levels in _HARDTESTS_LEVEL_RATING.items():
        rating = by_source.get(source)
        if rating and rating.get("level") in levels:
            return levels[rating["level"]]
    return None


def _hardtests_time_limit_s(raw: str | None) -> float | None:
    """Some judges spell ``N s`` in milliseconds (``2000 s``), others in seconds (``1 s``); a value of
    :data:`_MS_THRESHOLD` or more is milliseconds. A range (``1 - 2 s``) keeps its upper bound."""
    numbers = [float(x) for x in _LIMIT_NUMBER.findall(raw or "")]
    if not numbers:
        return None
    value = max(numbers)
    return value / 1000 if value >= _MS_THRESHOLD else value


def _hardtests_memory_limit_mb(raw: str | None) -> float | None:
    """The largest ``N <unit>`` in the field in MB (a bare number is MB); out-of-bounds values are dropped."""
    limits = [
        float(number.replace(",", "")) * _MEMORY_UNIT_MB[(unit or "MB").upper()]
        for number, unit in _MEMORY_LIMIT.findall(raw or "")
    ]
    value = max(limits, default=None)
    return value if value is not None and _MIN_MEMORY_LIMIT_MB <= value <= _MAX_MEMORY_LIMIT_MB else None


def normalize_hardtests(row: dict[str, Any]) -> dict[str, Any]:
    """The prepared-row fields HardTests spells differently; ``tags`` flatten across its rating sources."""
    tags = sorted({tag for entry in row.get("tags") or [] for tag in (entry.get("content") or []) if tag})
    return {"id": row.get("pid") or "", "rating": hardtests_rating(row), "tags": tags}


def format_hardtests_prompt(row: dict[str, Any]) -> str:
    """The body already carries the I/O spec and samples; its ``[problemUrl]:`` line names the judge and is dropped."""
    body = "\n".join(
        line for line in (row.get("question_content") or "").splitlines() if not line.startswith("[problemUrl]:")
    )
    return _format_problem_statement(
        "",
        (row.get("question_title") or "").strip(),
        time_limit_s=_hardtests_time_limit_s(row.get("time_limit")),
        memory_limit_mb=_hardtests_memory_limit_mb(row.get("memory_limit")),
        description=body,
    )


def hardtests_checker(judging_function: str | None) -> str | None:
    """Wrap ``output_judging_function(input_str, candidate_output, reference_output) -> bool`` as a ``checker.py``."""
    source = (judging_function or "").strip()
    return source + _HARDTESTS_CHECKER_DRIVER if source else None


def pack_hardtests_verification(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "tests": joined_tests(row),
        "checker": hardtests_checker(row.get(JOINED_CHECKER_FIELD)),
        "time_limit": _hardtests_time_limit_s(row.get("time_limit")),
    }


def keep_hardtests(row: dict[str, Any]) -> bool:
    """Drops functional problems (``starter_code``) and problems no source rates."""
    if row.get("starter_code"):
        return False
    if row.get("rating") is None:
        return False
    return len(joined_tests(row)) > 0


def _hlce_tests(row: dict[str, Any]) -> list[dict[str, str]]:
    return _stdin_tests(t for t in (row.get("test_cases") or []) if isinstance(t, dict))


def pack_hlce_verification(row: dict[str, Any]) -> dict[str, Any]:
    return {"tests": _hlce_tests(row), "checker": None, "time_limit": None}


def keep_hlce(row: dict[str, Any]) -> bool:
    return len(_hlce_tests(row)) > 0


def _load_hlce(dataset: str, config: str | None, split: str) -> Iterator[dict[str, Any]]:
    """``split`` is ignored: the dataset is a single ``train`` set. Only the icpc-world-finals subset is gradable; the
    sibling ioi set has no hidden tests."""
    yield from load_dataset(dataset, config, split=_HLCE_SPLIT, streaming=True)


def _parse_day(name: str, value: str | None) -> date | None:
    """Strict ``YYYY-MM-DD``; ``None`` stays open. The other ISO forms :meth:`date.fromisoformat` accepts raise."""
    if value is None:
        return None
    try:
        day = date.fromisoformat(value)
    except (TypeError, ValueError):
        day = None
    if day is None or day.isoformat() != value:
        raise ValueError(f"{name} must be a YYYY-MM-DD day, got {value!r}")
    return day


@dataclass(frozen=True)
class ContestSelection:
    """The rows a run scores: contests from ``start_date`` through ``end_date`` inclusive by calendar day (either may
    be open), on ``platforms`` as the source spells them (all when empty). Examples-only problems are scored only
    under ``include_examples_only``."""

    start_date: date | None = None
    end_date: date | None = None
    platforms: tuple[str, ...] = ()
    include_examples_only: bool = False

    def __post_init__(self) -> None:
        if self.start_date is not None and self.end_date is not None and self.start_date > self.end_date:
            raise ValueError(f"start_date {self.start_date} is after end_date {self.end_date}; the window is empty")
        # One spelling per selection: it keys the re-grader's payload cache and names the run's files.
        object.__setattr__(self, "platforms", tuple(sorted(set(self.platforms))))

    @classmethod
    def parse(
        cls,
        start_date: str | None = None,
        end_date: str | None = None,
        platforms: Iterable[str] = (),
        include_examples_only: bool = False,
    ) -> "ContestSelection":
        return cls(
            _parse_day("start_date", start_date),
            _parse_day("end_date", end_date),
            tuple(platforms),
            include_examples_only,
        )

    @classmethod
    def from_meta(cls, meta: dict[str, Any] | None) -> "ContestSelection":
        """A line recording no ``include_examples_only`` predates the exclusion, so its run scored those problems."""
        return cls.parse(**{"include_examples_only": True, **(meta or {})})

    def to_meta(self) -> dict[str, Any]:
        return {
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "platforms": list(self.platforms),
            "include_examples_only": self.include_examples_only,
        }

    @property
    def dated(self) -> bool:
        return self.start_date is not None or self.end_date is not None

    @property
    def label(self) -> str:
        """E.g. ``2025-01-04..2025-04-06_atcoder``; empty for the default selection."""
        window = f"{self.start_date or ''}..{self.end_date or ''}" if self.dated else ""
        examples = "with-examples-only" if self.include_examples_only else ""
        return "_".join(part for part in (window, "+".join(self.platforms), examples) if part)


@dataclass(frozen=True)
class CodeDatasetAdapter:
    """How a dataset's rows become the env's ``{prompt, answer}`` shape. ``group_field``/``group_label`` name what the
    eval report buckets by, ``id_field`` what names a problem; ``load`` replaces the standard HF load."""

    format_prompt: Callable[[dict[str, Any]], str]
    pack_verification: Callable[[dict[str, Any]], dict[str, Any]]
    keep: Callable[[dict[str, Any]], bool]
    group_field: str = "rating"
    group_label: str = "rating"
    id_field: str = "id"
    load: Callable[[str, str | None, str], Iterable[dict[str, Any]]] | None = None
    # The one split a benchmark ships; ``None`` => the split the run names.
    split: str | None = None
    # Prepared-row fields a source spells differently, added before the preparation filters run; ``None`` => the raw
    # row already carries them.
    normalize: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    # What a ContestSelection selects on: contest day, the platform field, and the platforms ``keep`` can grade.
    # Undeclared => a selection bounding that axis is refused.
    contest_date: Callable[[dict[str, Any]], date] | None = None
    platform_field: str | None = None
    platforms: tuple[str, ...] = ()
    # Whether a row's only tests are its statement's examples, which a constant print can pass: scored only under
    # include_examples_only, joins a training pool only when asked. ``None`` => every row has hidden tests.
    examples_only: Callable[[dict[str, Any]], bool] | None = None

    def __post_init__(self) -> None:
        if self.platforms and self.platform_field is None:
            raise ValueError(f"an adapter declaring platforms {self.platforms} must name the platform_field")

    @property
    def scores_raw_rows(self) -> bool:
        """A source whose prepared fields need ``normalize`` (and a joined tests table) is scored from its prepared
        pool instead."""
        return self.normalize is None

    def resolve_split(self, requested: str | None, default: str) -> str:
        """The source's own split where it ships one, else ``requested``, else ``default``. A requested split the
        source does not ship raises, rather than being recorded while its own is scored."""
        if self.split is None:
            return default if requested is None else requested
        if requested not in (None, self.split):
            raise ValueError(f"this dataset ships only the {self.split!r} split, not {requested!r}")
        return self.split

    def reject_unselectable(self, selection: ContestSelection) -> None:
        """A date window needs a stamped contest date, a platform filter a platform this adapter grades."""
        if selection.dated and self.contest_date is None:
            raise ValueError("this dataset stamps no contest date, so a start_date/end_date window cannot apply")
        unknown = sorted(set(selection.platforms) - set(self.platforms))
        if unknown and not self.platforms:
            raise ValueError("this dataset records no platform, so a platform filter cannot apply")
        if unknown:
            raise ValueError(
                f"platform(s) {unknown} are not ones this adapter grades; it grades {list(self.platforms)}"
            )

    def is_examples_only(self, row: dict[str, Any]) -> bool:
        return self.examples_only is not None and self.examples_only(row)

    def scored_rows(
        self, rows: Iterable[dict[str, Any]], selection: ContestSelection, left_out: list[Any] | None = None
    ) -> Iterator[dict[str, Any]]:
        """The rows a run scores, in source order; the eval and the offline re-grader both index this sequence.
        Examples-only ids left out are appended to ``left_out``. The selection is validated before a row is read."""
        self.reject_unselectable(selection)
        return self._scored(rows, selection, left_out)

    def _scored(
        self, rows: Iterable[dict[str, Any]], selection: ContestSelection, left_out: list[Any] | None
    ) -> Iterator[dict[str, Any]]:
        for row in rows:
            if not (self._selects(row, selection) and self.keep(row)):
                continue
            if not selection.include_examples_only and self.is_examples_only(row):
                if left_out is not None:
                    left_out.append(row.get(self.id_field))
                continue
            yield row

    def _selects(self, row: dict[str, Any], selection: ContestSelection) -> bool:
        if selection.platforms and row.get(self.platform_field) not in selection.platforms:
            return False
        if not selection.dated:
            return True
        day = self.contest_date(row)
        if selection.start_date is not None and day < selection.start_date:
            return False
        return selection.end_date is None or day <= selection.end_date


CODE_DATASET_ADAPTERS: dict[str, CodeDatasetAdapter] = {
    "codeforces": CodeDatasetAdapter(
        format_codeforces_prompt,
        pack_codeforces_verification,
        _keep_codeforces,
        examples_only=codeforces_examples_only,
    ),
    "hardtests": CodeDatasetAdapter(
        format_hardtests_prompt, pack_hardtests_verification, keep_hardtests, normalize=normalize_hardtests
    ),
    # No rating/difficulty column, so overall metrics only; no id column, so its examples go unnamed.
    "deepcoder": CodeDatasetAdapter(_format_deepcoder_prompt, _pack_deepcoder_verification, _keep_deepcoder),
    "livecodebench": CodeDatasetAdapter(
        format_titled_statement,
        pack_livecodebench_verification,
        keep_livecodebench,
        group_field="difficulty",
        group_label="difficulty",
        id_field="question_id",
        load=_load_livecodebench,
        split="test",
        contest_date=_livecodebench_contest_date,
        platform_field="platform",
        platforms=_LCB_GRADABLE_PLATFORMS,
    ),
    "icpc": CodeDatasetAdapter(
        format_icpc_prompt,
        pack_icpc_verification,
        keep_icpc,
        group_field="source",
        group_label="contest",
        load=_load_icpc,
    ),
    "hlce": CodeDatasetAdapter(
        format_titled_statement,
        pack_hlce_verification,
        keep_hlce,
        group_field="platform",
        group_label="contest",
        id_field="question_id",
        load=_load_hlce,
        split=_HLCE_SPLIT,
    ),
}
