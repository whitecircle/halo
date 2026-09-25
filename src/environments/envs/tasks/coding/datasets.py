"""Dataset adapters for the coding task environments.

Stateless row transforms into the env's ``{prompt, answer}`` shape, plus a ``keep`` predicate dropping
rows the stdin/stdout env cannot grade. Each dataset registers a :class:`CodeDatasetAdapter` in
:data:`CODE_DATASET_ADAPTERS` via a ``format_*`` / ``pack_*`` / ``keep_*`` trio (+ optional ``load_*``).
A benchmark that stamps its rows' contest date and platform also declares them, so a run can score a
:class:`ContestSelection` of it.
"""

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
# The LiveCodeBench platforms whose rows carry stdin tests, as the rows spell them. Its ``leetcode``
# rows are functional, which ``keep_livecodebench`` drops, so a selection naming it would score nothing.
_LCB_GRADABLE_PLATFORMS = ("atcoder", "codeforces")

# HardTests difficulty on the Codeforces rating scale. A Codeforces rating is used as is; Luogu's
# seven levels and the coarse AtCoder/TACO labels map to a representative rating, so one rating
# bound cuts the whole pool. Luogu level 0 is "unknown" and falls through to the other sources.
_LUOGU_LEVEL_RATING = {1: 800, 2: 1000, 3: 1300, 4: 1600, 5: 2000, 6: 2400, 7: 2900}
_HARDTESTS_LEVEL_RATING: dict[str, dict[str, int]] = {
    "atcoder": {"easy": 1000, "medium": 1400, "hard": 1800, "very hard": 2400},
    "taco": {"easy": 1000, "medium": 1400, "medium_hard": 1700, "hard": 2100, "very_hard": 2500},
}
_LIMIT_NUMBER = re.compile(r"[0-9]+(?:\.[0-9]+)?")
# A time limit this large is spelled in milliseconds on the judges that write them as ``N s``.
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
# Outside this range a scraped limit is not a judge's (CodeChef rows carry ``50000 bytes``); the prompt
# omits it.
_MIN_MEMORY_LIMIT_MB = 1.0
_MAX_MEMORY_LIMIT_MB = 8192.0
# Runs HardTests' judging function under the env's checker contract: three file paths on argv
# (input, expected, got), a trailing 1/0 on stdout. A judging function that raises rejects.
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
    """Render a contest problem statement, including only the sections the row provides.

    ``examples`` carries pre-numbered ``(index, input, output)`` triples so a dataset that skips a
    malformed example keeps the surviving ones' original numbering.
    """
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
    """Compose a full problem statement from an ``open-r1/codeforces`` row (only present sections included)."""
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


def stdin_tests(items: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    """``{input, output}`` pairs from test dicts, missing halves as empty strings — the env's test shape."""
    return [{"input": t.get("input", ""), "output": t.get("output", "")} for t in items]


def joined_tests(row: dict[str, Any]) -> list[dict[str, str]]:
    """Tests the preparation script joined onto the row from a compacted tests table (``joined_tests``,
    a JSON list of ``{input, output}``); empty when the row had no match."""
    raw = row.get("joined_tests")
    if not raw:
        return []
    tests = json.loads(raw) if isinstance(raw, str) else raw
    return stdin_tests(tests)


def pack_codeforces_verification(row: dict[str, Any]) -> dict[str, Any]:
    """Extract the grading payload from an ``open-r1/codeforces`` row.

    ``official_tests`` (only the ones Codeforces shows untruncated) plus the generated tests the
    preparation script joined onto the row, falling back to ``examples``. ``None`` checker => token
    grading.
    """
    tests = list(row.get("official_tests") or []) + joined_tests(row)
    if not tests:
        tests = stdin_tests(row.get("examples") or [])
    return {
        "tests": tests,
        "checker": row.get("generated_checker") or None,
        "time_limit": float(row.get("time_limit") or 0.0) or None,
    }


def keep_codeforces(row: dict[str, Any]) -> bool:
    """Keep executable, stdin/stdout Codeforces problems with tests; drop interactive and empty-test rows."""
    if not row.get("executable"):
        return False
    if (row.get("input_mode") or "stdio") != "stdio":
        return False
    if row.get("interaction_format"):
        return False
    return len(pack_codeforces_verification(row)["tests"]) > 0


def format_deepcoder_prompt(row: dict[str, Any]) -> str:
    """Return the DeepCoder problem statement, which already includes the I/O spec and examples."""
    return (row.get("problem") or "").strip()


def _parse_deepcoder_tests(row: dict[str, Any]) -> list[dict[str, str]]:
    """Parse DeepCoder ``tests`` JSON into stdin/stdout pairs; functional (``fn_name``) specs yield none."""
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


def pack_deepcoder_verification(row: dict[str, Any]) -> dict[str, Any]:
    """Pack a DeepCoder row's stdin/stdout tests into the env payload (no checker, default time limit)."""
    return {"tests": _parse_deepcoder_tests(row), "checker": None, "time_limit": None}


def keep_deepcoder(row: dict[str, Any]) -> bool:
    """Keep DeepCoder rows that have at least one stdin/stdout test (drops functional-only specs)."""
    return len(_parse_deepcoder_tests(row)) > 0


class _StringOnlyUnpickler(pickle.Unpickler):
    """Unpickler for the encoded test payloads, which hold one JSON string: no class or callable is ever
    needed, so a payload that references one is refused instead of imported."""

    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(f"test payload references {module}.{name}; only a JSON string is expected")


def decode_test_payload(raw: str) -> list[dict[str, Any]]:
    """Decode a test-case field into ``{input, output, ...}`` dicts.

    Plain JSON, or base64-of-zlib-of-pickle around a JSON string: the encoding LiveCodeBench uses for
    its private tests and HardTests for its whole suites.
    """
    try:
        return json.loads(raw)
    except ValueError:
        payload = zlib.decompress(base64.b64decode(raw.encode("utf-8")))
        return json.loads(_StringOnlyUnpickler(io.BytesIO(payload)).load())


def _lcb_stdin_tests(row: dict[str, Any]) -> list[dict[str, str]]:
    """Collect a LiveCodeBench row's stdin/stdout tests; LeetCode ``functional`` problems are skipped."""
    return stdin_tests(
        t
        for field in ("public_test_cases", "private_test_cases")
        for t in decode_test_payload(row.get(field) or "[]")
        if t.get("testtype") == "stdin"
    )


def format_titled_statement(row: dict[str, Any]) -> str:
    """Return a ``question_title`` + ``question_content`` statement (LiveCodeBench and HLCE share this
    row shape; ``question_content`` already carries the I/O spec)."""
    title = (row.get("question_title") or "").strip()
    body = (row.get("question_content") or "").strip()
    return f"# {title}\n\n{body}" if title else body


def pack_livecodebench_verification(row: dict[str, Any]) -> dict[str, Any]:
    """Pack a LiveCodeBench row's stdin tests into the env payload (no checker, default time limit)."""
    return {"tests": _lcb_stdin_tests(row), "checker": None, "time_limit": None}


def keep_livecodebench(row: dict[str, Any]) -> bool:
    """Keep LiveCodeBench problems with at least one stdin test (drops LeetCode functional problems)."""
    return len(_lcb_stdin_tests(row)) > 0


def load_livecodebench(dataset: str, config: str | None, split: str) -> Iterator[dict[str, Any]]:
    """Yield raw LiveCodeBench rows from a release's ``test*.jsonl`` files: the newest file first, each
    file in its stored row order, which is not newest-first (``test6.jsonl`` opens on its 2025-01-04
    contests). The eval's example indices, and the re-grader's, are this order.

    ``config`` is the release tag (default ``release_v6``); ``split`` ignored. Bypasses ``load_dataset``,
    whose loading script datasets 4.x does not execute. A release is cumulative, so a contamination-clean
    subset is a :class:`ContestSelection` window over it, not a release tag.
    """
    files = _LCB_RELEASE_FILES.get(config or "release_v6")
    if files is None:
        raise ValueError(f"unknown LiveCodeBench release {config!r}; expected one of {sorted(_LCB_RELEASE_FILES)}")
    for filename in reversed(files):
        path = hf_hub_download(dataset, filename, repo_type="dataset")
        with open(path) as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def livecodebench_contest_date(row: dict[str, Any]) -> date:
    """The calendar day of a LiveCodeBench row's ``contest_date`` (an ISO datetime). A row without one
    raises: a date window cannot place it, and dropping it would shrink the window unannounced."""
    raw = row.get("contest_date")
    try:
        return datetime.fromisoformat(raw).date()
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"LiveCodeBench row {row.get('question_id')!r} carries no ISO contest_date (got {raw!r})"
        ) from exc


def _icpc_time_limit_s(row: dict[str, Any]) -> float | None:
    """ICPC-Eval's ``time_limit_ms`` in seconds; ``None`` when the row carries none."""
    return (row.get("time_limit_ms") or 0) / 1000 or None


def format_icpc_prompt(row: dict[str, Any]) -> str:
    """Compose an ICPC-Eval problem statement from its title, description, I/O spec, and examples."""
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
    """Parse ICPC-Eval ``test_cases`` (a list of ``[input, output]`` pairs) into stdin/stdout dicts."""
    return [
        {"input": str(tc[0]), "output": str(tc[1])}
        for tc in (row.get("test_cases") or [])
        if isinstance(tc, (list, tuple)) and len(tc) == 2
    ]


def pack_icpc_verification(row: dict[str, Any]) -> dict[str, Any]:
    """Pack an ICPC-Eval row's tests into the env payload (no checker, per-problem time limit)."""
    return {"tests": _icpc_tests(row), "checker": None, "time_limit": _icpc_time_limit_s(row)}


def keep_icpc(row: dict[str, Any]) -> bool:
    """Keep ``traditional`` ICPC-Eval problems with tests; drop ``spj`` (C++ special judges we can't run)."""
    return row.get("type") == "traditional" and len(_icpc_tests(row)) > 0


def load_icpc(dataset: str, config: str | None, split: str) -> Iterator[dict[str, Any]]:
    """Stream ICPC-Eval rows. Streaming avoids pulling the full multi-GB test payload for a small probe."""
    yield from load_dataset(dataset, config, split=split or "test", streaming=True)


def hardtests_rating(row: dict[str, Any]) -> int | None:
    """A HardTests problem's difficulty on the Codeforces rating scale, or ``None`` when no source rates it."""
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
    """HardTests spells limits as ``N s`` with N in milliseconds on some judges (``2000 s``) and seconds
    on others (``1 s``); a value of :data:`_MS_THRESHOLD` or more is milliseconds. A range (``1 - 2 s``)
    keeps its upper bound."""
    numbers = [float(x) for x in _LIMIT_NUMBER.findall(raw or "")]
    if not numbers:
        return None
    value = max(numbers)
    return value / 1000 if value >= _MS_THRESHOLD else value


def _hardtests_memory_limit_mb(raw: str | None) -> float | None:
    """The largest ``N <unit>`` in the field in megabytes (a bare number is megabytes); a range keeps its
    upper bound, a value outside :data:`_MIN_MEMORY_LIMIT_MB`..:data:`_MAX_MEMORY_LIMIT_MB` is dropped."""
    limits = [
        float(number.replace(",", "")) * _MEMORY_UNIT_MB[(unit or "MB").upper()]
        for number, unit in _MEMORY_LIMIT.findall(raw or "")
    ]
    value = max(limits, default=None)
    return value if value is not None and _MIN_MEMORY_LIMIT_MB <= value <= _MAX_MEMORY_LIMIT_MB else None


def normalize_hardtests(row: dict[str, Any]) -> dict[str, Any]:
    """The prepared-row fields HardTests spells differently: ``id`` from ``pid``, ``rating`` on the
    Codeforces scale, ``tags`` flattened across its rating sources."""
    tags = sorted({tag for entry in row.get("tags") or [] for tag in (entry.get("content") or []) if tag})
    return {"id": row.get("pid") or "", "rating": hardtests_rating(row), "tags": tags}


def format_hardtests_prompt(row: dict[str, Any]) -> str:
    """Title, limits and the statement body; the body already carries the I/O spec and the samples.
    The ``[problemUrl]:`` line names the judge and is dropped."""
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
    """Wrap HardTests' ``output_judging_function(input_str, candidate_output, reference_output) -> bool``
    into the env's ``checker.py`` contract; ``None`` when the problem has no special judge."""
    source = (judging_function or "").strip()
    return source + _HARDTESTS_CHECKER_DRIVER if source else None


def pack_hardtests_verification(row: dict[str, Any]) -> dict[str, Any]:
    """Pack the joined HardTests suite, its judging function as a checker, and the per-problem time limit."""
    return {
        "tests": joined_tests(row),
        "checker": hardtests_checker(row.get("joined_checker")),
        "time_limit": _hardtests_time_limit_s(row.get("time_limit")),
    }


def keep_hardtests(row: dict[str, Any]) -> bool:
    """Keep rated stdin/stdout HardTests problems that received a suite; functional problems
    (``starter_code``) and problems no source rates are dropped."""
    if row.get("starter_code"):
        return False
    if row.get("rating") is None:
        return False
    return len(joined_tests(row)) > 0


def _hlce_tests(row: dict[str, Any]) -> list[dict[str, str]]:
    """Collect an HLCE row's stdin/stdout ``test_cases`` (a list of ``{input, output}`` dicts)."""
    return stdin_tests(t for t in (row.get("test_cases") or []) if isinstance(t, dict))


def pack_hlce_verification(row: dict[str, Any]) -> dict[str, Any]:
    """Pack an HLCE row's tests into the env payload (no checker, no time limit)."""
    return {"tests": _hlce_tests(row), "checker": None, "time_limit": None}


def keep_hlce(row: dict[str, Any]) -> bool:
    """Keep HLCE problems with at least one stdin/stdout test."""
    return len(_hlce_tests(row)) > 0


def load_hlce(dataset: str, config: str | None, split: str) -> Iterator[dict[str, Any]]:
    """Stream HLCE ICPC World Finals rows. ``split`` is ignored — the dataset is a single ``train`` set.
    Only the icpc-world-finals subset is gradable; the sibling ioi set has no hidden tests."""
    yield from load_dataset(dataset, config, split="train", streaming=True)


def _parse_day(name: str, value: str | None) -> date | None:
    """``value`` as a calendar day, spelled ``YYYY-MM-DD``; ``None`` stays open. Any other spelling
    raises, the other ISO forms :meth:`date.fromisoformat` accepts (``20250104``, week dates) included."""
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
    """The rows of a benchmark a run scores: contests dated ``start_date`` through ``end_date``, both
    ends inclusive and compared by calendar day (either may be open), on ``platforms`` as the source
    spells them (every platform when empty). The empty selection scores every row."""

    start_date: date | None = None
    end_date: date | None = None
    platforms: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.start_date is not None and self.end_date is not None and self.start_date > self.end_date:
            raise ValueError(f"start_date {self.start_date} is after end_date {self.end_date}; the window is empty")
        # One spelling per selection: it keys the re-grader's payload cache and names the run's files.
        object.__setattr__(self, "platforms", tuple(sorted(set(self.platforms))))

    @classmethod
    def parse(
        cls, start_date: str | None = None, end_date: str | None = None, platforms: Iterable[str] = ()
    ) -> "ContestSelection":
        """A selection from its spelled form (the CLI flags, a trajectory meta line)."""
        return cls(_parse_day("start_date", start_date), _parse_day("end_date", end_date), tuple(platforms))

    @classmethod
    def from_meta(cls, meta: dict[str, Any] | None) -> "ContestSelection":
        """The selection a trajectory meta line records; a line recording none selected every row."""
        return cls.parse(**(meta or {}))

    def to_meta(self) -> dict[str, Any]:
        """The spelled form :meth:`from_meta` reads back."""
        return {
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "platforms": list(self.platforms),
        }

    @property
    def dated(self) -> bool:
        """Whether the selection bounds the contest date on either end."""
        return self.start_date is not None or self.end_date is not None

    @property
    def label(self) -> str:
        """A short name for the selection (``2025-01-04..2025-04-06_atcoder``); empty when it selects everything."""
        window = f"{self.start_date or ''}..{self.end_date or ''}" if self.dated else ""
        return "_".join(part for part in (window, "+".join(self.platforms)) if part)


@dataclass(frozen=True)
class CodeDatasetAdapter:
    """How to turn a contest dataset's rows into the env's ``{prompt, answer}`` shape.

    ``format_prompt`` builds the prompt, ``pack_verification`` the grading payload, ``keep`` drops
    ungradable rows. ``group_field``/``group_label`` name the raw-row field the eval report buckets by,
    ``id_field`` the one naming a problem in the eval's results and trajectories. ``load`` is an
    optional custom row iterator; ``None`` => standard HF load.
    """

    format_prompt: Callable[[dict[str, Any]], str]
    pack_verification: Callable[[dict[str, Any]], dict[str, Any]]
    keep: Callable[[dict[str, Any]], bool]
    group_field: str = "rating"
    group_label: str = "rating"
    id_field: str = "id"
    load: Callable[[str, str | None, str], Iterable[dict[str, Any]]] | None = None
    # Prepared-row fields a source spells differently (``id``/``rating``/``tags``), added by the
    # preparation script before its filters run; ``None`` => the raw row already carries them.
    normalize: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    # What a ContestSelection selects on: a row's contest day, and the field naming its platform with
    # the platforms whose rows ``keep`` can grade, as the source spells them. Undeclared => a selection
    # bounding that axis is refused.
    contest_date: Callable[[dict[str, Any]], date] | None = None
    platform_field: str | None = None
    platforms: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.platforms and self.platform_field is None:
            raise ValueError(f"an adapter declaring platforms {self.platforms} must name the platform_field")

    @property
    def scores_raw_rows(self) -> bool:
        """Whether the eval scripts can score the source's raw rows. A source whose prepared fields need
        ``normalize`` (and a joined tests table) is scored from its prepared pool instead."""
        return self.normalize is None

    def require_selectable(self, selection: ContestSelection) -> None:
        """Raise unless this source can apply ``selection``: a date window needs a stamped contest date,
        a platform filter a platform this adapter grades, spelled as the source spells it."""
        if selection.dated and self.contest_date is None:
            raise ValueError("this dataset stamps no contest date, so a start_date/end_date window cannot apply")
        unknown = sorted(set(selection.platforms) - set(self.platforms))
        if unknown and not self.platforms:
            raise ValueError("this dataset records no platform, so a platform filter cannot apply")
        if unknown:
            raise ValueError(
                f"platform(s) {unknown} are not ones this adapter grades; it grades {list(self.platforms)}"
            )

    def scored_rows(self, rows: Iterable[dict[str, Any]], selection: ContestSelection) -> Iterator[dict[str, Any]]:
        """The rows a run scores, in source order: inside ``selection`` and gradable (``keep``). The eval
        builds its examples and the offline re-grader rebuilds their payloads by index from this one
        sequence. The selection is validated here, before a row is read."""
        self.require_selectable(selection)
        return (row for row in rows if self._selects(row, selection) and self.keep(row))

    def _selects(self, row: dict[str, Any], selection: ContestSelection) -> bool:
        """Whether ``row`` falls inside ``selection``; the contest date is read only under a window."""
        if selection.platforms and row.get(self.platform_field) not in selection.platforms:
            return False
        if not selection.dated:
            return True
        day = self.contest_date(row)
        if selection.start_date is not None and day < selection.start_date:
            return False
        return selection.end_date is None or day <= selection.end_date


CODE_DATASET_ADAPTERS: dict[str, CodeDatasetAdapter] = {
    "codeforces": CodeDatasetAdapter(format_codeforces_prompt, pack_codeforces_verification, keep_codeforces),
    "hardtests": CodeDatasetAdapter(
        format_hardtests_prompt, pack_hardtests_verification, keep_hardtests, normalize=normalize_hardtests
    ),
    # DeepCoder rows have no rating/difficulty column, so no report bucket (overall metrics only), and
    # no id column, so its examples go unnamed.
    "deepcoder": CodeDatasetAdapter(format_deepcoder_prompt, pack_deepcoder_verification, keep_deepcoder),
    "livecodebench": CodeDatasetAdapter(
        format_titled_statement,
        pack_livecodebench_verification,
        keep_livecodebench,
        group_field="difficulty",
        group_label="difficulty",
        id_field="question_id",
        load=load_livecodebench,
        contest_date=livecodebench_contest_date,
        platform_field="platform",
        platforms=_LCB_GRADABLE_PLATFORMS,
    ),
    "icpc": CodeDatasetAdapter(
        format_icpc_prompt,
        pack_icpc_verification,
        keep_icpc,
        group_field="source",
        group_label="contest",
        load=load_icpc,
    ),
    "hlce": CodeDatasetAdapter(
        format_titled_statement,
        pack_hlce_verification,
        keep_hlce,
        group_field="platform",
        group_label="contest",
        id_field="question_id",
        load=load_hlce,
    ),
}
