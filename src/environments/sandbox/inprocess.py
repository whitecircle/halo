"""In-process restricted-builtins evaluation: the lightest sandbox tier.

Runs math expressions and Python snippets in-process under restricted builtins with an AST
escape-guard and a wall-clock timeout — no OS isolation, no imports. For complete untrusted programs
use the subprocess / bubblewrap / remote backends.
"""

import ast
import math
import re
import threading
from collections.abc import Callable

from src.environments.sandbox.base import REPL_NO_OUTPUT_MESSAGE, SANDBOX_DEFAULT_TIMEOUT, repl_timeout_message

# Builtins that defeat the sandbox even without imports (introspection/exec/IO); blocked by name.
_FORBIDDEN_SANDBOX_NAMES = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "open",
        "input",
        "__import__",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "breakpoint",
        "memoryview",
        "exit",
        "quit",
    }
)

# No ``__`` prefix, so the dunder guard misses ``gen.gi_frame.f_back.f_globals['__builtins__']``.
_INTROSPECTION_ATTR_PREFIXES = ("f_", "gi_", "cr_", "ag_", "tb_")

# ``"{0.__class__.__mro__}".format(obj)`` walks dunders at runtime, invisible to the AST guards.
_FORMAT_DUNDER_RE = re.compile(r"\{[^{}]*[.\[]__")


# Big-int ``**``/``<<``/``*``/``factorial`` compute in a single GIL-holding C call, so the wall-clock
# timeout cannot preempt them (``9**9**9`` blocks the worker indefinitely); bound the result size.
_BIGINT_MAX_RESULT_BITS = 1_000_000
_FACTORIAL_MAX_N = 10_000
# Same problem, allocation-shaped: ``[0] * 10**9`` and ``sum(range(10**18))`` never return to
# bytecode. Python-level ``for`` loops run bytecode between items and stay preemptible.
_SEQUENCE_MAX_RESULT_ELEMENTS = 100_000_000
_RANGE_MAX_LENGTH = 100_000_000
# ``pow(b, e, m)`` does ~bits(e) modmuls in one C call; cap bits(e)·bits(m) (RSA-scale is far below).
_MODPOW_MAX_COST = 1_000_000_000


def _guarded_pow(base, exp, mod=None):
    """``pow`` with size/cost bounds on the unpreemptible big-int paths (plain pow's result size,
    modpow's bits(exp)·bits(mod) work)."""
    is_bigint_pow = mod is None and isinstance(base, int) and isinstance(exp, int) and exp > 0 and abs(base) > 1
    if is_bigint_pow and abs(base).bit_length() * exp > _BIGINT_MAX_RESULT_BITS:
        raise ValueError("exponentiation result too large for the sandbox")
    if (
        mod is not None
        and isinstance(exp, int)
        and isinstance(mod, int)
        and exp.bit_length() * mod.bit_length() > _MODPOW_MAX_COST
    ):
        raise ValueError("modular exponentiation too expensive for the sandbox")
    return pow(base, exp) if mod is None else pow(base, exp, mod)


def _guarded_lshift(value, shift):
    """``<<`` with a result-size bound: ``x << n`` allocates ``bits(x) + n`` bits in one C call."""
    if (
        isinstance(value, int)
        and isinstance(shift, int)
        and shift > 0
        and abs(value).bit_length() + shift > _BIGINT_MAX_RESULT_BITS
    ):
        raise ValueError("shift result too large for the sandbox")
    return value << shift


def _guarded_mul(left, right):
    """``*`` with result-size bounds on the two unpreemptible-C-call shapes: big-int products
    (bit_length probe) and sequence repetition (len × count probe, either operand order). Floats
    and other pairs pass through without a check."""
    if isinstance(left, int) and isinstance(right, int):
        if left.bit_length() + right.bit_length() > _BIGINT_MAX_RESULT_BITS:
            raise ValueError("multiplication result too large for the sandbox")
    else:
        count, sized = (left, right) if isinstance(left, int) else (right, left)
        if (
            isinstance(count, int)
            and count > 0
            and hasattr(sized, "__len__")
            and len(sized) * count > _SEQUENCE_MAX_RESULT_ELEMENTS
        ):
            raise ValueError("sequence repetition result too large for the sandbox")
    return left * right


def _guarded_range(*args):
    """``range`` with a length cap: the C-iterator builtins (``sum``/``min``/``max``/``sorted``/
    ``list``/``any``/``all``) drain a range inside one GIL-holding C loop, so an unbounded range
    blocks the worker like a big-int op. The cap leaves realistic loops (10**6-scale) unaffected."""
    result = range(*args)
    try:
        too_large = len(result) > _RANGE_MAX_LENGTH
    except OverflowError:  # len() past sys.maxsize — categorically too large
        too_large = True
    if too_large:
        raise ValueError(f"range too large for the sandbox (max {_RANGE_MAX_LENGTH} elements)")
    return result


def _guarded_factorial(n):
    """``math.factorial`` with an argument bound (the C loop holds the GIL for its whole runtime)."""
    if isinstance(n, int) and n > _FACTORIAL_MAX_N:
        raise ValueError(f"factorial argument too large for the sandbox (max {_FACTORIAL_MAX_N})")
    return math.factorial(n)


_POW_GUARD_NAME = "_sandbox_pow"
_OPERATOR_GUARDS: dict[str, Callable] = {
    _POW_GUARD_NAME: _guarded_pow,
    "_sandbox_lshift": _guarded_lshift,
    "_sandbox_mul": _guarded_mul,
}
_GUARDED_OPS: dict[type, tuple[str, str]] = {
    ast.Pow: ("**", _POW_GUARD_NAME),
    ast.LShift: ("<<", "_sandbox_lshift"),
    ast.Mult: ("*", "_sandbox_mul"),
}


class _GuardedOpTransformer(ast.NodeTransformer):
    """Rewrite the unpreemptible-big-int operators (``**``, ``<<``, ``*``) into guard calls.

    Python offers no operator hook for int-int operators, so the bounds are applied at the AST level.
    """

    def visit_BinOp(self, node: ast.BinOp):
        self.generic_visit(node)
        guarded = _GUARDED_OPS.get(type(node.op))
        if guarded is None:
            return node
        guard = ast.Name(id=guarded[1], ctx=ast.Load())
        return ast.copy_location(ast.Call(func=guard, args=[node.left, node.right], keywords=[]), node)

    def visit_AugAssign(self, node: ast.AugAssign):
        self.generic_visit(node)
        guarded = _GUARDED_OPS.get(type(node.op))
        if guarded is None:
            return node
        symbol, guard_name = guarded
        if not isinstance(node.target, ast.Name):
            # Leaving x[i] **= y (or <<=, *=) unguarded would bypass the result-size bounds.
            raise ValueError(f"{symbol}= on a non-name target is not supported in the sandbox")
        guard = ast.Name(id=guard_name, ctx=ast.Load())
        value = ast.Call(func=guard, args=[ast.Name(id=node.target.id, ctx=ast.Load()), node.value], keywords=[])
        return ast.copy_location(ast.Assign(targets=[node.target], value=value), node)


def _compile_sandboxed(code: str, mode: str):
    """Parse, apply the operator-guard transform, and compile for ``eval``/``exec``."""
    tree = _GuardedOpTransformer().visit(ast.parse(code, mode=mode))
    return compile(ast.fix_missing_locations(tree), "<sandbox>", mode)


SAFE_MATH_BUILTINS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": _guarded_pow,
    "len": len,
    "int": int,
    "float": float,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
    "pi": math.pi,
    "e": math.e,
    "factorial": _guarded_factorial,
    "gcd": math.gcd,
}

# Invariant: no entry may allocate or compute unbounded work in one C call (hence guarded ``range``).
SAFE_PYTHON_BUILTINS = {
    **SAFE_MATH_BUILTINS,
    "str": str,
    "bool": bool,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "range": _guarded_range,
    "enumerate": enumerate,
    "zip": zip,
    "sorted": sorted,
    "reversed": reversed,
    "any": any,
    "all": all,
    "map": map,
    "filter": filter,
    "divmod": divmod,
}


def _run_with_timeout(fn: Callable[[], str], timeout: float) -> str:
    """Run ``fn`` on a daemon thread bounded by a wall-clock timeout, returning its result string.

    The thread is a daemon joined with a timeout, so it is abandoned at interpreter exit and a runaway
    ``while True:`` cannot block process shutdown (``ThreadPoolExecutor.shutdown(wait=True)`` would).
    ``fn`` maps its own exceptions to an ``Error: ...`` string; one escaping the thread surfaces as
    the produced-no-result error.
    """
    result: dict[str, str] = {}

    def _worker() -> None:
        result["value"] = fn()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        return repl_timeout_message(timeout)
    return result.get("value", "Error: execution produced no result")


def safe_calculate(expression: str, timeout: float = SANDBOX_DEFAULT_TIMEOUT) -> str:
    """Evaluate a math expression under restricted builtins, bounded by a wall-clock timeout.

    An empty ``__builtins__`` does not block attribute access, so the AST escape-guard runs before
    ``eval``. The timeout bounds an expression the guard admits but that runs indefinitely (a large
    comprehension), which would otherwise block the caller: sync tool execution runs on a Ray actor's
    event loop.
    """
    try:
        expr = expression.strip().replace("^", "**").replace("\u00d7", "*").replace("\u00f7", "/")
        _validate_sandbox_ast(expr, allow_imports=False)
    except ValueError as e:
        return f"Error: {str(e)}"

    def _evaluate() -> str:
        try:
            code_obj = _compile_sandboxed(expr, "eval")
            result = eval(code_obj, {"__builtins__": {}, **_OPERATOR_GUARDS}, SAFE_MATH_BUILTINS)
            return str(result)
        except ZeroDivisionError:
            return "Error: Division by zero"
        except Exception as e:
            return f"Error: {str(e) or type(e).__name__}"

    return _run_with_timeout(_evaluate, timeout)


def _validate_sandbox_ast(code: str, allow_imports: bool = False) -> None:
    """Reject sandbox-escape constructs before execution. Raises ValueError on a violation.

    Overriding ``__builtins__`` does not contain ``eval``/``exec``: ``().__class__.__bases__[0].
    __subclasses__()`` walks the dunder graph to arbitrary classes (file IO, subprocess). Dunder
    attribute/name access, frame/generator introspection attrs, dunder subscripts, and introspection/IO
    builtins are therefore all forbidden. ``import`` is blocked unless ``allow_imports``, which is safe
    only under external isolation. This is defense in depth, not a hard boundary (see the module
    docstring).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if not allow_imports and isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("imports are not allowed in the sandbox")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError(f"access to dunder attribute '{node.attr}' is not allowed in the sandbox")
        if isinstance(node, ast.Attribute) and node.attr.startswith(_INTROSPECTION_ATTR_PREFIXES):
            raise ValueError(f"access to introspection attribute '{node.attr}' is not allowed in the sandbox")
        # ``mapping['__dunder__']`` reaches the same dunders the attribute guard blocks.
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
            and node.slice.value.startswith("__")
        ):
            raise ValueError(f"subscripting with dunder key '{node.slice.value}' is not allowed in the sandbox")
        # ``_sandbox_*`` blocks user code from shadowing an injected operator guard.
        if isinstance(node, ast.Name) and (
            node.id.startswith(("__", "_sandbox_")) or node.id in _FORBIDDEN_SANDBOX_NAMES
        ):
            raise ValueError(f"use of '{node.id}' is not allowed in the sandbox")
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and _FORMAT_DUNDER_RE.search(node.value):
            raise ValueError("format string with dunder field access is not allowed in the sandbox")


def _run_python_sandboxed_inner(code: str) -> str:
    """Run validated code under restricted builtins (no timeout — see run_python_sandboxed).

    Every exception the submitted code raises is part of its result and is rendered as ``Error: ...``
    so the model gets an actionable observation. Catching only an enumerated subset would let common
    failures (a ``SyntaxError`` from the exec compile, an ``AssertionError``) escape the worker thread
    and surface as "produced no result" plus a traceback in the actor log. ``BaseException``
    (``SystemExit``, ``KeyboardInterrupt``) still propagates.
    """
    output_lines: list[str] = []

    def capture_print(*args, sep=" ", end="\n"):
        output_lines.append(sep.join(str(a) for a in args))

    # A bare ``print(...)`` takes the eval path, so the capture must be in scope there too.
    sandbox_globals = {
        "__builtins__": SAFE_PYTHON_BUILTINS,
        "print": capture_print,
        **_OPERATOR_GUARDS,
    }
    local_vars: dict = {}

    # Captured print output takes precedence over the expression value, so ``print(x)`` returns x.
    try:
        result = eval(_compile_sandboxed(code, "eval"), sandbox_globals, local_vars)
        return "\n".join(output_lines) if output_lines else str(result)
    except SyntaxError:
        # Not an expression — fall through to the statement (exec) path.
        pass
    except Exception as e:
        return f"Error: {str(e) or type(e).__name__}"

    try:
        exec(_compile_sandboxed(code, "exec"), sandbox_globals, local_vars)
        if output_lines:
            return "\n".join(output_lines)
        elif local_vars:
            return str(list(local_vars.values())[-1])
        else:
            return REPL_NO_OUTPUT_MESSAGE
    except Exception as e:
        return f"Error: {str(e) or type(e).__name__}"


def run_python_sandboxed(
    code: str,
    timeout: float = SANDBOX_DEFAULT_TIMEOUT,
    allow_imports: bool = False,
) -> str:
    """Execute Python in a restricted-builtins sandbox with an escape guard + wall-clock timeout.

    Returns an ``Error: ...`` string on violation, exception, or timeout.
    """
    try:
        _validate_sandbox_ast(code, allow_imports=allow_imports)
    except ValueError as e:
        return f"Error: {str(e)}"

    return _run_with_timeout(lambda: _run_python_sandboxed_inner(code), timeout)
