"""Comment stripping for the coding environments: a program with its comments removed, by the language's
registered syntax, what the identity of a resubmission is read on."""

from src.environments.sandbox.base import require_language

_TRIPLE_QUOTES = ('"""', "'''")
# A preprocessor block the compiler drops, the C family's other comment.
_C_DISABLED_BLOCK = ("#if 0", "#endif")


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
