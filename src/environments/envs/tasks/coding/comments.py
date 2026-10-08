"""Comment accounting for the coding environments: how many characters of a program sit in comments, by the
language's registered syntax, the verdict the reasoning-in-comments guard reads off that count, the
program with its comments removed, what the identity of a resubmission is read on, and whether it reads
standard input outside them and its string literals."""

import ast
import bisect
import itertools
import re
import warnings
from collections.abc import Callable

from src.environments.sandbox.base import require_language

# A program whose comments reach this many characters and outweigh its code carries the reasoning the
# thinking cap closed, not documentation.
REASONING_IN_COMMENTS_MIN_CHARS = 16384

_TRIPLE_QUOTES = ('"""', "'''")
# A preprocessor block the compiler drops, the C family's other comment.
_C_DISABLED_BLOCK = ("#if 0", "#endif")
# The line ends the parser and ``ast.get_source_segment`` split source on (a form feed is not one).
_LINE_BREAK = re.compile(r"\r\n|\r|\n")


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
    return _without(code, _comment_spans(code, spec.line_comment, spec.block_comment))


def reads_stdin(code: str, language: str) -> bool:
    """Whether ``code`` reads its standard input: its language's ``stdin_read`` pattern found outside its
    comments and string literals, with no ``stdin_redirect`` there swapping that input for an in-memory buffer."""
    spec = require_language(language)
    bare = _without(code, _comment_spans(code, spec.line_comment, spec.block_comment, strings=True))
    if spec.stdin_redirect is not None and re.search(spec.stdin_redirect, bare):
        return False
    return re.search(spec.stdin_read, bare) is not None


def reasoning_in_comments(comments: int, rest: int) -> bool:
    """The guard's verdict on a program's counts: comments at :data:`REASONING_IN_COMMENTS_MIN_CHARS` or
    more, outweighing everything else."""
    return comments >= REASONING_IN_COMMENTS_MIN_CHARS and comments > rest


def carries_reasoning_in_comments(code: str, language: str) -> bool:
    """Whether the coding environments refuse ``code`` in ``language`` for the reasoning in its comments."""
    return reasoning_in_comments(*comment_chars(code, language))


def _without(code: str, spans: list[tuple[int, int]]) -> str:
    """``code`` with the ordered, disjoint ``spans`` cut out."""
    out, last = [], 0
    for start, end in spans:
        out.append(code[last:start])
        last = end
    out.append(code[last:])
    return "".join(out)


def _comment_spans(
    code: str, line_marker: str, block: tuple[str, str] | None, *, strings: bool = False
) -> list[tuple[int, int]]:
    """The spans of ``code`` inside line and block comments (and a C-family ``#if 0`` block), in order.
    String literals are skipped, or with ``strings`` returned among them."""
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
            end = _skip_string(code, i)
            line_start = False
            if not strings:
                i = end
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
    """Characters of Python string statements (docstrings and bare strings), 0 for code that does not parse:
    the length of each statement's source segment, read off positions mapped in one pass over ``code``
    (``ast.get_source_segment`` re-splits the source up to each node, quadratic over a program)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError):
            return 0
    offset = _char_offsets(code)
    return sum(
        offset(node.end_lineno, node.end_col_offset) - offset(node.lineno, node.col_offset)
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
    )


def _char_offsets(code: str) -> Callable[[int, int], int]:
    """The character offset in ``code`` of an ast position: a 1-based line and a UTF-8 byte column. A
    non-ASCII line's byte-to-character table is built the first time a position on it is read."""
    starts = [0, *(match.end() for match in _LINE_BREAK.finditer(code)), len(code)]
    if code.isascii():
        return lambda line, col: starts[line - 1] + col
    tables: dict[int, list[int] | None] = {}

    def offset(line: int, col: int) -> int:
        start = starts[line - 1]
        if line not in tables:
            text = code[start : starts[line]]
            tables[line] = (
                None if text.isascii() else list(itertools.accumulate((len(c.encode()) for c in text), initial=0))
            )
        table = tables[line]
        return start + (col if table is None else bisect.bisect_left(table, col))

    return offset
