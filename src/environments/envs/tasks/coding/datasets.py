"""Dataset adapters for the coding task environments.

Stateless row transforms into the env's ``{prompt, answer}`` shape, plus a ``keep`` predicate dropping
rows the stdin/stdout env cannot grade. Each dataset registers a :class:`CodeDatasetAdapter` in
:data:`CODE_DATASET_ADAPTERS` via a ``format_*`` / ``pack_*`` / ``keep_*`` trio (+ optional ``load_*``).
"""

import base64
import json
import pickle
import zlib
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset
from huggingface_hub import hf_hub_download

# Cumulative releases (test.jsonl = v1, each later appends); read newest-first for least-contaminated contests.
_LCB_RELEASE_FILES: dict[str, list[str]] = {
    f"release_v{v}": [f"test{'' if i == 1 else i}.jsonl" for i in range(1, v + 1)] for v in range(1, 7)
}


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
    parts: list[str] = [f"# {label}. {title}".strip(". ") if (label or title) else "# Problem"]

    limits = []
    if time_limit_s:
        limits.append(f"time limit per test: {time_limit_s:g} s")
    if memory_limit_mb:
        limits.append(f"memory limit per test: {memory_limit_mb:g} MB")
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


def pack_codeforces_verification(row: dict[str, Any]) -> dict[str, Any]:
    """Extract the grading payload from an ``open-r1/codeforces`` row.

    Prefers ``official_tests``, falls back to ``examples``. ``None`` checker => token grading.
    """
    tests = list(row.get("official_tests") or [])
    if not tests:
        tests = [{"input": ex.get("input", ""), "output": ex.get("output", "")} for ex in row.get("examples") or []]
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


def _decode_lcb_tests(raw: str) -> list[dict[str, Any]]:
    """Decode a LiveCodeBench ``*_test_cases`` field into ``{input, output, testtype}`` dicts.

    Public tests are plain JSON; private tests are base64-of-zlib-of-pickle (official-data-only path).
    """
    try:
        return json.loads(raw)
    except ValueError:
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(raw.encode("utf-8")))))


def _lcb_stdin_tests(row: dict[str, Any]) -> list[dict[str, str]]:
    """Collect a LiveCodeBench row's stdin/stdout tests; LeetCode ``functional`` problems are skipped."""
    tests: list[dict[str, str]] = []
    for field in ("public_test_cases", "private_test_cases"):
        for t in _decode_lcb_tests(row.get(field) or "[]"):
            if t.get("testtype") == "stdin":
                tests.append({"input": t.get("input", ""), "output": t.get("output", "")})
    return tests


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
    """Yield raw LiveCodeBench rows from a release's ``test*.jsonl`` files, newest contests first.

    ``config`` is the release tag (default ``release_v6``); ``split`` ignored. Bypasses ``load_dataset``,
    whose loading script datasets 4.x no longer executes.
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


def format_icpc_prompt(row: dict[str, Any]) -> str:
    """Compose an ICPC-Eval problem statement from its title, description, I/O spec, and examples."""
    return _format_problem_statement(
        (row.get("problem_label") or "").strip(),
        (row.get("title") or "").strip(),
        time_limit_s=(row.get("time_limit_ms") or 0) / 1000 or None,
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
    time_limit = (row.get("time_limit_ms") or 0) / 1000 or None
    return {"tests": _icpc_tests(row), "checker": None, "time_limit": time_limit}


def keep_icpc(row: dict[str, Any]) -> bool:
    """Keep ``traditional`` ICPC-Eval problems with tests; drop ``spj`` (C++ special judges we can't run)."""
    return row.get("type") == "traditional" and len(_icpc_tests(row)) > 0


def load_icpc(dataset: str, config: str | None, split: str) -> Iterator[dict[str, Any]]:
    """Stream ICPC-Eval rows. Streaming avoids pulling the full multi-GB test payload for a small probe."""
    yield from load_dataset(dataset, config, split=split or "test", streaming=True)


# Of HLCE only the icpc-world-finals subset is gradable; the sibling ioi set has no hidden tests.


def _hlce_tests(row: dict[str, Any]) -> list[dict[str, str]]:
    """Collect an HLCE row's stdin/stdout ``test_cases`` (a list of ``{input, output}`` dicts)."""
    return [
        {"input": t.get("input", ""), "output": t.get("output", "")}
        for t in (row.get("test_cases") or [])
        if isinstance(t, dict)
    ]


def pack_hlce_verification(row: dict[str, Any]) -> dict[str, Any]:
    """Pack an HLCE row's tests into the env payload (no checker, no time limit)."""
    return {"tests": _hlce_tests(row), "checker": None, "time_limit": None}


def keep_hlce(row: dict[str, Any]) -> bool:
    """Keep HLCE problems with at least one stdin/stdout test."""
    return len(_hlce_tests(row)) > 0


def load_hlce(dataset: str, config: str | None, split: str) -> Iterator[dict[str, Any]]:
    """Stream HLCE ICPC World Finals rows. ``split`` is ignored — the dataset is a single ``train`` set."""
    yield from load_dataset(dataset, config, split="train", streaming=True)


@dataclass(frozen=True)
class CodeDatasetAdapter:
    """How to turn a contest dataset's rows into the env's ``{prompt, answer}`` shape.

    ``format_prompt`` builds the prompt, ``pack_verification`` the grading payload, ``keep`` drops
    ungradable rows. ``group_field``/``group_label`` name the raw-row field the eval report buckets by.
    ``load`` is an optional custom row iterator; ``None`` => standard HF load.
    """

    format_prompt: Callable[[dict[str, Any]], str]
    pack_verification: Callable[[dict[str, Any]], dict[str, Any]]
    keep: Callable[[dict[str, Any]], bool]
    group_field: str = "rating"
    group_label: str = "rating"
    load: Callable[[str, str | None, str], Iterable[dict[str, Any]]] | None = None


CODE_DATASET_ADAPTERS: dict[str, CodeDatasetAdapter] = {
    "codeforces": CodeDatasetAdapter(format_codeforces_prompt, pack_codeforces_verification, keep_codeforces),
    # DeepCoder rows have no rating/difficulty column, so no report bucket (overall metrics only).
    "deepcoder": CodeDatasetAdapter(format_deepcoder_prompt, pack_deepcoder_verification, keep_deepcoder),
    "livecodebench": CodeDatasetAdapter(
        format_titled_statement,
        pack_livecodebench_verification,
        keep_livecodebench,
        group_field="difficulty",
        group_label="difficulty",
        load=load_livecodebench,
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
        load=load_hlce,
    ),
}
