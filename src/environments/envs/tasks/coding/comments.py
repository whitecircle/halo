"""Comment accounting for the coding environments: how many characters of a program sit in comments, by the
language's registered syntax, the verdict the reasoning-in-comments guard reads off that count, and the
program with its comments removed, what the identity of a resubmission is read on."""

import ast
import warnings

from src.environments.sandbox.base import require_language

# A program whose comments reach this many characters and outweigh its code carries the reasoning the
# thinking cap closed, not documentation.
REASONING_IN_COMMENTS_MIN_CHARS = 16384

_TRIPLE_QUOTES = ('"""', "'''")
# A preprocessor block the compiler drops, the C family's other comment.
_C_DISABLED_BLOCK = ("#if 0", "#endif")


def comment_chars(code: str, language: str) -> tuple[int, int]:
    """Characters of ``code`` inside comments and outside them, by ``language`` (its registry entry's
    syntax; a language the registry does not hold raises): line comments wherever they start, block
    comments anywhere, string literals skipped, a C-family ``#if 0`` block counted whole, and under
    Python a bare string statement (a docstring) counted as a comment."""
    spec = require_language(language)
    comments = sum(end - start for start, end in _comment_spans(code, spec.line_comment, spec.block_comment))
    if spec.name == "python":
        comments += _bare_string_chars(code)
    return comments, len(code) - comments


def strip_comments(code: str, language: str) -> str:
    """``code`` without its comments (the same spans :func:`comment_chars` counts, bare strings kept): a
    resubmission that differs only in them is the same program."""
    spec = require_language(language)
    out, last = [], 0
    for start, end in _comment_spans(code, spec.line_comment, spec.block_comment):
        out.append(code[last:start])
        last = end
    out.append(code[last:])
    return "".join(out)


def reasoning_in_comments(comments: int, rest: int) -> bool:
    """The guard's verdict on a program's counts: comments at :data:`REASONING_IN_COMMENTS_MIN_CHARS` or
    more, outweighing everything else."""
    return comments >= REASONING_IN_COMMENTS_MIN_CHARS and comments > rest


def carries_reasoning_in_comments(code: str, language: str) -> bool:
    """Whether the coding environments refuse ``code`` in ``language`` for the reasoning in its comments."""
    return reasoning_in_comments(*comment_chars(code, language))


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


def _end_of(code: str, start: int, closer: str) -> int:
    """The index just past ``closer`` found at or after ``start``, or the end of ``code``."""
    end = code.find(closer, start)
    return len(code) if end < 0 else end + len(closer)


def _skip_string(code: str, i: int) -> int:
    """The index just past the string literal opening at ``i``: triple-quoted or single-quoted, backslash
    escapes honoured, an unterminated one running to the end of the line (triple-quoted: of the code)."""
    quote = next((q for q in _TRIPLE_QUOTES if code.startswith(q, i)), code[i])
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


def _bare_string_chars(code: str) -> int:
    """Characters of Python string statements (docstrings and bare strings), 0 for code that does not parse."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError):
            return 0
    return sum(
        len(ast.get_source_segment(code, node) or "")
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
    )
