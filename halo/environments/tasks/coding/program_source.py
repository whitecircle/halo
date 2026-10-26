"""A contest program's source as the code-contests environment reads it: without its comments, normalized into
the identity a resubmission is checked by, the language it is evidently written in, and whether it compiles."""

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
# Marks of C or C++ source for the language-label check: an include directive, a main function, a
# namespace directive or a std-qualified name; else statements ending in ``;`` on this share of the
# non-blank lines.
_C_FAMILY_MARKS = re.compile(r"^\s*#\s*include\s*[<\"]|\bint\s+main\s*\(|\busing\s+namespace\b|\bstd::", re.MULTILINE)
_C_FAMILY_STATEMENT_SHARE = 0.3
# Marks of C++ over C: a namespace directive, a std-qualified name, a C++ standard header (a name with
# no ``.h``, or ``bits/stdc++.h``), the iostream objects, a template. A C header (``<stdio.h>``) marks C.
_CPP_MARKS = re.compile(
    r"\busing\s+namespace\b|\bstd::|^\s*#\s*include\s*<(?:bits/stdc\+\+\.h|\w+)>|\bcin\b|\bcout\b|\btemplate\s*<",
    re.MULTILINE,
)
_C_MARKS = re.compile(r"^\s*#\s*include\s*<[\w/]+\.h>", re.MULTILINE)
# Marks of Python source: a function definition, an import, a print call.
_PYTHON_MARKS = re.compile(r"^\s*(?:def\s+\w+\s*\(|import\s+\w|from\s+[\w.]+\s+import\b)|\bprint\s*\(", re.MULTILINE)


def _end_of(code: str, start: int, closer: str) -> int:
    """The index just past ``closer`` found at or after ``start``, or the end of ``code``."""
    end = code.find(closer, start)
    return len(code) if end < 0 else end + len(closer)


def _skip_string(code: str, i: int) -> int:
    """The index just past the string literal opening at ``i``: triple-quoted or single-quoted, backslash
    escapes honoured, an unterminated one running to the end of the line (triple-quoted: of the code)."""
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
    """The spans of ``code`` inside line and block comments (and a C-family ``#if 0`` block), string
    literals skipped, in order."""
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
    """``code`` without its comments, by ``language`` (its registry entry's syntax; a language the registry
    does not hold raises): line comments wherever they start, block comments anywhere, a C-family ``#if 0``
    block whole, string literals kept. A resubmission that differs only in them is the same program."""
    spec = require_language(language)
    out, last = [], 0
    for start, end in _comment_spans(code, spec.line_comment, spec.block_comment):
        out.append(code[last:start])
        last = end
    out.append(code[last:])
    return "".join(out)


def normalized_program(code: str, language: str) -> str:
    """``code`` in ``language`` as the identity check reads it: the language first (the same text is another
    program in another language), then the code without its comments (:func:`strip_comments`), trailing
    whitespace off every line and every blank line dropped, indentation kept (it is the program under Python)."""
    lines = [line.rstrip() for line in strip_comments(code, language).splitlines()]
    return f"{language}\n" + "\n".join(line for line in lines if line)


def is_output_only_change(ran: str, submitted: str) -> bool:
    """Whether normalized program ``submitted`` is normalized program ``ran`` with nothing changed but output
    statements dropped (:data:`_OUTPUT_STATEMENT`): the same program, less its debug prints. A walk in order, since
    matching an equal line is never worse than skipping it."""
    submitted_lines = submitted.split("\n")
    matched = 0
    for line in ran.split("\n"):
        if matched < len(submitted_lines) and line == submitted_lines[matched]:
            matched += 1
        elif not _OUTPUT_STATEMENT.match(line):
            return False
    return matched == len(submitted_lines)


def python_syntax_error(code: str) -> str | None:
    """The interpreter's error line for ``code`` that does not compile as Python (``SyntaxError: invalid syntax (line
    3)``), else ``None``: the program fails on it before reading input, so the line quotes the source alone. Compiled
    from the bytes a backend writes (:func:`utf8_encodable`); a compile the host cannot finish is left to the sandbox.
    """
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
    """The error a ``language`` program fails to build with, where the host can tell before any run:
    Python source that does not compile (:func:`python_syntax_error`), on a backend running this
    process's interpreter (one stating its Python toolchain); ``None`` otherwise, the sandbox's to say."""
    resolved = find_language(language)
    if resolved is None or resolved.name != "python" or sandbox.toolchain(resolved.name) is None:
        return None
    return python_syntax_error(code)


def _looks_like_c_family(code: str) -> bool:
    """Whether ``code`` carries a mark of C or C++ source (:data:`_C_FAMILY_MARKS`, or the statement share)."""
    if _C_FAMILY_MARKS.search(code):
        return True
    lines = [line.rstrip() for line in code.splitlines() if line.strip()]
    return bool(lines) and sum(line.endswith(";") for line in lines) >= _C_FAMILY_STATEMENT_SHARE * len(lines)


def _c_family_language(code: str, compiled: Sequence[str]) -> str | None:
    """The language among ``compiled`` that C or C++ source is written in: ``cpp`` on a C++ mark (none
    when the run lists no ``cpp``), ``c`` on a C header without one; source with neither, or C on a run
    listing no ``c``, is ambiguous and takes the first listed."""
    if _CPP_MARKS.search(code):
        return "cpp" if "cpp" in compiled else None
    if "c" in compiled and _C_MARKS.search(code):
        return "c"
    return compiled[0] if compiled else None


def find_evident_language(code: str, language: str, offered: Sequence[str]) -> str | None:
    """The language among ``offered`` that ``code``, sent as ``language``, is evidently written in instead, else
    ``None``: C or C++ sent as Python fails Python's parse and carries a C-family mark but no Python one (C++ told by
    its own marks); Python sent as a compiled language parses and carries a Python mark but no C-family one. Parse and
    marks together keep a valid program quoting the other language in a string or comment on its label.
    """
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
