"""A contest program's source as the code-contests environment reads it: comments stripped, normalized (comments,
blank lines and trailing whitespace outside string literals gone) for the resubmission identity check, its evident
language, and whether it compiles."""

import re
import warnings
from collections.abc import Sequence

from halo.environments.sandbox.base import SandboxExecutor
from halo.environments.sandbox.languages import find_language, require_language, utf8_encodable

# A preprocessor block the compiler drops, the C family's other comment.
_C_DISABLED_BLOCK = ("#if 0", "#endif")
# A line a program sheds between its last run and its submission without becoming another program: an output
# statement (a debug print), whose removal is no change to what the run tested.
_OUTPUT_STATEMENT = re.compile(
    r"^\s*(?:print\s*\(|sys\.std(?:out|err)\.write\s*\(|(?:std::)?(?:cout|cerr|clog)\s*<<|f?printf\s*\(|puts\s*\()"
)
# Marks of C or C++ source: an include, a main function, a namespace directive or a std-qualified name; else statements
# ending in ``;`` on this share of the non-blank lines.
_C_FAMILY_MARKS = re.compile(r"^\s*#\s*include\s*[<\"]|\bint\s+main\s*\(|\busing\s+namespace\b|\bstd::", re.MULTILINE)
_C_FAMILY_STATEMENT_SHARE = 0.3
# Marks of C++ over C: a namespace directive, a std-qualified name, a C++ standard header (a name with
# no ``.h``, or ``bits/stdc++.h``), the iostream objects, a template. A C header (``<stdio.h>``) marks C.
_CPP_MARKS = re.compile(
    r"\busing\s+namespace\b|\bstd::|^\s*#\s*include\s*<(?:bits/stdc\+\+\.h|\w+)>|\bcin\b|\bcout\b|\btemplate\s*<",
    re.MULTILINE,
)
_C_MARKS = re.compile(r"^\s*#\s*include\s*<[\w/]+\.h>", re.MULTILINE)
_PYTHON_MARKS = re.compile(r"^\s*(?:def\s+\w+\s*\(|import\s+\w|from\s+[\w.]+\s+import\b)|\bprint\s*\(", re.MULTILINE)


def _end_of(code: str, start: int, closer: str) -> int:
    end = code.find(closer, start)
    return len(code) if end < 0 else end + len(closer)


def _skip_string(code: str, i: int) -> int:
    """Honours backslash escapes; an unterminated literal runs to the end of the line (triple-quoted: of the code)."""
    quote = next((q for q in ('"""', "'''") if code.startswith(q, i)), code[i])
    j = i + len(quote)
    while j < len(code):
        if code[j] == "\\":
            j += 2
            continue
        if code.startswith(quote, j):
            return j + len(quote)
        if code[j] == "\n" and len(quote) == 1:
            return j
        j += 1
    return len(code)


def _comment_spans(code: str, line_marker: str, block: tuple[str, str] | None) -> list[tuple[int, int]]:
    n = len(code)
    i = 0
    spans: list[tuple[int, int]] = []
    line_start = True
    while i < n:
        if block is not None and line_start and code.startswith(_C_DISABLED_BLOCK[0], i):
            end = _end_of(code, i + len(_C_DISABLED_BLOCK[0]), _C_DISABLED_BLOCK[1])
        elif code.startswith(line_marker, i):
            end = code.find("\n", i)
            end = n if end < 0 else end
        elif block is not None and code.startswith(block[0], i):
            end = _end_of(code, i + len(block[0]), block[1])
        elif code[i] in "\"'":
            i = _skip_string(code, i)
            line_start = False
            continue
        else:
            line_start = code[i] == "\n" or (line_start and code[i] in " \t")
            i += 1
            continue
        spans.append((i, end))
        i = end
    return spans


def strip_comments(code: str, language: str) -> str:
    """Line comments, block comments and a C-family ``#if 0`` block go, string literals stay. A resubmission that
    differs only in them is the same program."""
    spec = require_language(language)
    out, last = [], 0
    for start, end in _comment_spans(code, spec.line_comment, spec.block_comment):
        out.append(code[last:start])
        last = end
    out.append(code[last:])
    return "".join(out)


def trim_outside_strings(code: str) -> str:
    """``code`` without blank lines and trailing whitespace, string literals untouched (their whitespace is program
    data), leading indentation kept."""
    out: list[str] = []
    pending: list[str] = []  # whitespace since the last content, written only when content follows
    newline_pending = False
    line_has_content = False
    i = 0
    while i < len(code):
        char = code[i]
        if char in "\"'":
            end = _skip_string(code, i)
            piece = code[i:end]
        elif char == "\n":
            pending.clear()
            newline_pending = newline_pending or line_has_content
            line_has_content = False
            i += 1
            continue
        elif char.isspace():
            pending.append(char)
            i += 1
            continue
        else:
            end = i + 1
            piece = char
        if newline_pending:
            out.append("\n")
            newline_pending = False
        out.extend(pending)
        pending.clear()
        out.append(piece)
        line_has_content = True
        i = end
    return "".join(out)


def normalized_program(code: str, language: str) -> str:
    """Language first (the same text is another program in another language), then the code without comments, blank
    lines or trailing whitespace outside string literals (:func:`trim_outside_strings`); indentation stays (it is the
    program under Python)."""
    return f"{language}\n{trim_outside_strings(strip_comments(code, language))}"


def is_output_only_change(ran: str, submitted: str) -> bool:
    """Whether ``submitted`` is ``ran`` with only output statements dropped (:data:`_OUTPUT_STATEMENT`). A walk in
    order, since matching an equal line is never worse than skipping it."""
    submitted_lines = submitted.split("\n")
    matched = 0
    for line in ran.split("\n"):
        if matched < len(submitted_lines) and line == submitted_lines[matched]:
            matched += 1
        elif not _OUTPUT_STATEMENT.match(line):
            return False
    return matched == len(submitted_lines)


def python_syntax_error(code: str) -> str | None:
    """The interpreter's error line for source that does not compile, else ``None``: the program fails on it before
    reading input, so the line quotes the source alone. A compile the host cannot finish is left to the sandbox."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        try:
            compile(utf8_encodable(code).encode("utf-8"), "main.py", "exec")
        except SyntaxError as exc:  # null bytes included, which name no line
            return f"{type(exc).__name__}: {exc.msg}" + (f" (line {exc.lineno})" if exc.lineno else "")
        except (RecursionError, MemoryError):
            return None
    return None


def host_build_error(code: str, language: str, sandbox: SandboxExecutor) -> str | None:
    """Python source that does not compile, on a backend running this process's interpreter (one stating its Python
    toolchain); ``None`` otherwise, the sandbox's to say."""
    resolved = find_language(language)
    if resolved is None or resolved.name != "python" or sandbox.toolchain(resolved.name) is None:
        return None
    return python_syntax_error(code)


def _looks_like_c_family(code: str) -> bool:
    if _C_FAMILY_MARKS.search(code):
        return True
    lines = [line.rstrip() for line in code.splitlines() if line.strip()]
    return bool(lines) and sum(line.endswith(";") for line in lines) >= _C_FAMILY_STATEMENT_SHARE * len(lines)


def _c_family_language(code: str, compiled: Sequence[str]) -> str | None:
    """``cpp`` on a C++ mark (none when the run lists no ``cpp``), ``c`` on a C header without one; source with
    neither, or C on a run listing no ``c``, is ambiguous and takes the first listed."""
    if _CPP_MARKS.search(code):
        return "cpp" if "cpp" in compiled else None
    if "c" in compiled and _C_MARKS.search(code):
        return "c"
    return compiled[0] if compiled else None


def find_evident_language(code: str, language: str, offered: Sequence[str]) -> str | None:
    """C or C++ sent as Python fails Python's parse and carries a C-family mark but no Python one; Python sent as a
    compiled language parses and carries a Python mark but no C-family one. Parse and marks together keep a valid
    program quoting the other language on its label."""
    spec = require_language(language)
    if spec.name == "python":
        if not _looks_like_c_family(code) or _PYTHON_MARKS.search(code) or python_syntax_error(code) is None:
            return None
        return _c_family_language(code, [name for name in offered if require_language(name).is_compiled])
    if (
        spec.is_compiled
        and "python" in offered
        and _PYTHON_MARKS.search(code)
        and not _looks_like_c_family(code)
        and python_syntax_error(code) is None
    ):
        return "python"
    return None
