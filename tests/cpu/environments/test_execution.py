#!/usr/bin/env python
"""
Tests for safe execution helpers (sandbox).

Run: python tests/cpu/environments/test_execution.py
"""

import math
import time

import pytest

from src.environments.sandbox.inprocess import (
    SAFE_MATH_BUILTINS,
    SAFE_PYTHON_BUILTINS,
    run_python_sandboxed,
    safe_calculate,
)

# safe_calculate


def test_calc_basic_arithmetic():
    assert safe_calculate("2 + 3") == "5"
    assert safe_calculate("10 - 4") == "6"
    assert safe_calculate("3 * 7") == "21"
    assert safe_calculate("15 / 3") == "5.0"
    assert safe_calculate("10 % 3") == "1"


def test_calc_exponentiation():
    assert safe_calculate("2^3") == "8", "Caret should be converted to **"
    assert safe_calculate("2**3") == "8"


def test_calc_unicode_operators():
    assert safe_calculate("6\u00d73") == "18", "Unicode multiply \u00d7"
    assert safe_calculate("10\u00f72") == "5.0", "Unicode divide \u00f7"


def test_calc_math_functions():
    assert safe_calculate("sqrt(16)") == "4.0"
    assert safe_calculate("abs(-5)") == "5"
    assert safe_calculate("floor(3.7)") == "3"
    assert safe_calculate("ceil(3.2)") == "4"
    result = float(safe_calculate("sin(0)"))
    assert abs(result) < 1e-10, f"sin(0) should be ~0, got {result}"


def test_calc_constants():
    result = float(safe_calculate("pi"))
    assert abs(result - math.pi) < 1e-10
    result = float(safe_calculate("e"))
    assert abs(result - math.e) < 1e-10


def test_calc_division_by_zero():
    assert "Error" in safe_calculate("1/0")
    assert "zero" in safe_calculate("1/0").lower()


def test_calc_invalid_expression():
    result = safe_calculate("abc def ghi")
    assert "Error" in result


def test_calc_no_builtins_access():
    result = safe_calculate("__import__('os')")
    assert "Error" in result


def test_calc_no_open():
    result = safe_calculate("open('/etc/passwd')")
    assert "Error" in result


def test_calc_whitespace():
    assert safe_calculate("  2 + 3  ") == "5"


# run_python_sandboxed


def test_sandbox_expression():
    assert run_python_sandboxed("2 + 3") == "5"


def test_sandbox_statement():
    result = run_python_sandboxed("x = 10\nprint(x)")
    assert "10" in result


def test_sandbox_print_capture():
    result = run_python_sandboxed("print('hello')\nprint('world')")
    assert "hello" in result
    assert "world" in result


def test_sandbox_no_output():
    """No print, two locals bound -> the LAST bound local (y = sum(x) = 6) is reported.

    The previous `"6" in result or "success" in result.lower()` passed regardless of which
    branch held, so a regression that returned the success line (or the wrong variable)
    would not be caught. Pin the exact last-variable value.
    """
    result = run_python_sandboxed("x = [1, 2, 3]\ny = sum(x)")
    assert result == "6", f"expected last-variable value '6', got {result!r}"


def test_sandbox_list_operations():
    assert run_python_sandboxed("sorted([3, 1, 2])") == "[1, 2, 3]"


def test_sandbox_no_import():
    result = run_python_sandboxed("import os")
    assert "Error" in result


def test_sandbox_no_file_access():
    result = run_python_sandboxed("open('test.txt')")
    assert "Error" in result


def test_sandbox_error_handling():
    result = run_python_sandboxed("1/0")
    assert "Error" in result


def test_sandbox_blocks_dunder_escape():
    """The classic ().__class__.__bases__[0].__subclasses__() escape must be rejected.

    Overriding __builtins__ alone does not contain it; AST validation must block dunder
    attribute access before execution.
    """
    result = run_python_sandboxed("().__class__.__bases__[0].__subclasses__()")
    assert "Error" in result
    assert "dunder" in result.lower()


def test_sandbox_blocks_dunder_globals_access():
    result = run_python_sandboxed("(lambda: None).__globals__")
    assert "Error" in result


def test_sandbox_blocks_getattr_builtin():
    # getattr() can reach attributes a literal cannot; it is forbidden by name.
    result = run_python_sandboxed("getattr((), '__class__')")
    assert "Error" in result


def test_sandbox_blocks_frame_introspection_escape():
    """A generator frame walk reaches the REAL builtins dict and __import__ — a confirmed RCE.

    ``gi_frame``/``f_back``/``f_globals`` are not dunders, so the dunder-attr guard misses them;
    the introspection-prefix guard must reject them. Without that guard this returns ``/`` (real cwd);
    with it, an Error string.
    """
    escape = (
        "def f(b):\n"
        "    yield b[0].gi_frame.f_back\n"
        "b=[None]; g=f(b); b[0]=g\n"
        "fr=list(g)[0]\n"
        "fr.f_back.f_globals['__builtins__']['__import__']('os').getcwd()\n"
    )
    result = run_python_sandboxed(escape)
    assert "Error" in result
    assert "introspection" in result.lower() or "dunder" in result.lower()


def test_sandbox_blocks_dunder_subscript():
    """Subscripting with a dunder string key (``d['__builtins__']``) is rejected — the general form
    of the frame-escape's final hop."""
    result = run_python_sandboxed("{}['__class__']")
    assert "Error" in result
    assert "dunder" in result.lower()


def test_sandbox_allows_normal_attributes_and_subscripts():
    """The introspection guard must not break legitimate method calls / subscripts."""
    assert run_python_sandboxed("'hello world'.upper()") == "HELLO WORLD"
    assert run_python_sandboxed("{'a': 1, 'b': 2}['b']") == "2"
    assert run_python_sandboxed("[10, 20, 30][-1]") == "30"


def test_sandbox_times_out_on_infinite_loop():
    """A runaway loop must not hang the caller — it returns a timeout error."""
    result = run_python_sandboxed("while True:\n    pass", timeout=0.5)
    assert "Error" in result
    assert "timeout" in result.lower()


def test_sandbox_multiline():
    code = "result = 0\nfor i in range(5):\n    result += i\nprint(result)"
    result = run_python_sandboxed(code)
    assert "10" in result


def test_sandbox_no_output_no_vars_success_message():
    """A statement that produces neither print output nor a bound local returns the success line."""
    # A bare for-loop would bind its loop variable, so use a no-op that binds nothing.
    result = run_python_sandboxed("pass")
    assert "executed successfully" in result.lower()


def test_sandbox_last_var_is_returned_not_first():
    """When several locals are bound (no print), the LAST one is reported."""
    result = run_python_sandboxed("a = 1\nb = 2\nc = 99")
    assert result == "99"


def test_sandbox_statement_nameerror_caught():
    """A NameError in statement form is caught and returned as an Error string (not raised)."""
    # Multi-line forces the exec path.
    result = run_python_sandboxed("y = 1\nz = undefined_name + y")
    assert "Error" in result
    assert "undefined_name" in result


def test_sandbox_renders_the_real_message_for_every_user_error():
    """Whatever the submitted code raises must come back as an actionable ``Error: <message>``.

    An enumerated except tuple on the exec branch that omits ``SyntaxError`` (the exec compile
    re-raises it) or ``AssertionError`` lets the two most common failures escape the sandbox worker
    thread: the model sees the opaque "execution produced no result" plus a raw traceback in the
    actor log, no signal it can act on.
    """
    syntax = run_python_sandboxed("def f(:\n    pass")
    assert syntax.startswith("Error:") and "produced no result" not in syntax
    assert "syntax" in syntax.lower() or "invalid" in syntax.lower()

    # A bare assert has no message, so the class name must stand in.
    assertion = run_python_sandboxed("x = 1\nassert x == 2")
    assert assertion.startswith("Error:") and "produced no result" not in assertion
    assert assertion.strip() != "Error:"


def test_sandbox_print_with_separator():
    """capture_print joins multiple args with the default space separator."""
    result = run_python_sandboxed("print('a', 'b', 'c')")
    assert result == "a b c"


def test_sandbox_bare_print_outputs():
    """A bare ``print(...)`` expression captures its output instead of erroring.

    The tool's own description tells the model to "use print() for output", so the eval path (which a
    single ``print(...)`` expression takes) must define the capturing ``print`` too: without it the
    call returns ``Error: name 'print' is not defined``.
    """
    assert run_python_sandboxed("print('hi')") == "hi"


def test_calc_negative_result():
    """Negative arithmetic results are formatted as strings, not errored."""
    assert safe_calculate("3 - 10") == "-7"


# Builtins completeness


def test_math_builtins_subset_of_python():
    """SAFE_PYTHON_BUILTINS is built as {**SAFE_MATH_BUILTINS, ...}; the spread must hold.

    This is a real invariant (the python sandbox must offer everything the calculator does); a
    refactor that drops the spread would regress it. The reverse (extra data-structure builtins)
    is exercised behaviorally below rather than by constant membership.
    """
    for key in SAFE_MATH_BUILTINS:
        assert key in SAFE_PYTHON_BUILTINS, f"{key} in SAFE_MATH_BUILTINS but not in SAFE_PYTHON_BUILTINS"
    assert set(SAFE_MATH_BUILTINS).issubset(SAFE_PYTHON_BUILTINS)
    assert len(SAFE_PYTHON_BUILTINS) > len(SAFE_MATH_BUILTINS)


def test_calc_factorial_and_gcd_execute():
    """Essential math builtins are not just present — they evaluate correctly through safe_calculate."""
    assert safe_calculate("factorial(5)") == "120"
    assert safe_calculate("gcd(12, 18)") == "6"
    assert safe_calculate("log2(8)") == "3.0"


def test_sandbox_data_structure_builtins_execute():
    """Data-structure builtins (zip/enumerate/dict/list) actually run inside the sandbox."""
    assert run_python_sandboxed("dict(zip(['a', 'b'], [1, 2]))") == "{'a': 1, 'b': 2}"
    result = run_python_sandboxed("list(enumerate(['x', 'y']))")
    assert result == "[(0, 'x'), (1, 'y')]"


def test_calc_timeout_on_slow_evaluation():
    """A slow evaluation must hit the wall-clock timeout instead of blocking the caller — sync tool
    execution runs on a Ray actor's event loop, so an unbounded eval would wedge every episode on the
    actor. Driven via an injected slow builtin: it yields the GIL (like any Python-level loop), the
    class of runaway the daemon-thread timeout can preempt."""
    SAFE_MATH_BUILTINS["hang"] = lambda: time.sleep(30) or 0
    try:
        result = safe_calculate("hang() + 1", timeout=0.2)
    finally:
        del SAFE_MATH_BUILTINS["hang"]
    assert result == "Error: execution exceeded 0.2s timeout"


def test_calc_fast_expression_unaffected_by_timeout():
    assert safe_calculate("2 + 3", timeout=5.0) == "5"


def test_pow_bomb_rejected_not_hung():
    """``9**9**9`` passes the AST escape-guard but is a SINGLE GIL-holding C-level pow — the timeout
    thread can never preempt it, so it would wedge the worker (and its Ray actor) forever. The pow
    guard must reject it immediately, in both the calculator and the python sandbox."""
    assert safe_calculate("9**9**9", timeout=5.0) == "Error: exponentiation result too large for the sandbox"
    assert run_python_sandboxed("9**9**9", timeout=5.0) == "Error: exponentiation result too large for the sandbox"
    assert "too large" in safe_calculate("pow(9, 387420489)", timeout=5.0)
    assert "too large" in run_python_sandboxed("x = 2\nx **= 10**9", timeout=5.0)


def test_factorial_bomb_rejected_not_hung():
    """``factorial(10**9)`` is the same unpreemptible C-level class; the argument bound rejects it."""
    assert "too large" in safe_calculate("factorial(10**9)", timeout=5.0)


def test_legitimate_pow_and_factorial_still_work():
    assert safe_calculate("2**10") == "1024"
    assert safe_calculate("2**1000") == str(2**1000)  # big but bounded results stay exact
    assert safe_calculate("pow(2, 10, 1000)") == "24"  # realistic modpow stays well under the cost bound
    assert safe_calculate("factorial(20)") == str(math.factorial(20))
    assert safe_calculate("(-2)**3") == "-8"
    assert safe_calculate("2**-2") == "0.25"
    assert run_python_sandboxed("x = 3\nx **= 4\nprint(x)") == "81"


def test_shift_bomb_rejected_not_hung():
    """``1 << 268435456`` builds a 32 MB int in one unpreemptible C call and any use of it (str,
    ``*``) then holds the GIL for minutes — same wedge class as the pow bomb. The shift guard must
    reject it immediately in both sandboxes, statement/augmented forms included."""
    assert safe_calculate("1 << 268435456", timeout=5.0) == "Error: shift result too large for the sandbox"
    assert run_python_sandboxed("1 << 268435456", timeout=5.0) == "Error: shift result too large for the sandbox"
    assert "too large" in run_python_sandboxed("x = 1\nx <<= 268435456", timeout=5.0)


def test_mul_bomb_rejected_not_hung():
    """Big-int ``*`` (x*x on a pow-bound-sized int) is a single GIL-holding C multiplication the
    timeout can never preempt; the product-size guard must reject it immediately."""
    assert (
        safe_calculate("(1 << 900000) * (1 << 900000)", timeout=5.0)
        == "Error: multiplication result too large for the sandbox"
    )
    assert (
        run_python_sandboxed("a = 1 << 900000\nb = a * a", timeout=5.0)
        == "Error: multiplication result too large for the sandbox"
    )
    assert "too large" in run_python_sandboxed("a = 1 << 900000\na *= a", timeout=5.0)


def test_range_c_iterator_bomb_rejected_not_hung():
    """``sum(range(10**18))`` (also min/max/sorted/list/any/all over a range) drains the range inside
    ONE GIL-holding C loop, never returning to bytecode — pre-guard, a 1s-timeout ``sum(range(8*10**7))``
    was empirically observed to run to completion (1.2s) instead of timing out, so a 10**18 range wedged
    the worker forever. The guarded ``range`` must reject construction immediately."""
    start = time.monotonic()
    assert "range too large" in run_python_sandboxed("sum(range(10**18))", timeout=1.0)
    assert time.monotonic() - start < 1.0, "must be rejected by the guard, not saved by the bypassable timeout"
    assert "range too large" in run_python_sandboxed("min(range(10**12))", timeout=1.0)
    assert "range too large" in run_python_sandboxed("list(range(10**10))", timeout=1.0)
    assert "range too large" in run_python_sandboxed("x = range(10**14)\nsorted(x)", timeout=1.0)


def test_sequence_repetition_bomb_rejected_not_hung():
    """``[0] * 10**9`` / ``'a' * 10**9`` allocate the whole multi-GB result in one C call — the old
    int×int-only mul guard passed them through untouched (empirically confirmed). The extended guard
    bounds len × count for any sized operand, both operand orders and the augmented form."""
    assert "sequence repetition" in run_python_sandboxed("[0] * 10**9", timeout=5.0)
    assert "sequence repetition" in run_python_sandboxed("10**9 * [0]", timeout=5.0)
    assert "sequence repetition" in run_python_sandboxed("'a' * 10**9", timeout=5.0)
    assert "sequence repetition" in run_python_sandboxed("b'a' * 10**9", timeout=5.0)
    assert "sequence repetition" in run_python_sandboxed("(1, 2) * 10**9", timeout=5.0)
    assert "sequence repetition" in run_python_sandboxed("x = [1, 2]\nx *= 10**9", timeout=5.0)
    assert "sequence repetition" in safe_calculate("(1, 2) * 10**9", timeout=5.0)


def test_modpow_cost_bomb_rejected_not_hung():
    """Guard-admitted ``pow(3, 1 << 999999, (1 << 999999) + 1)`` runs ~1e6 modmuls of 1e6-bit numbers
    inside one C call (minutes of GIL hold); the bits(exp)·bits(mod) cost bound rejects it."""
    assert "too expensive" in run_python_sandboxed("pow(3, 1 << 999999, (1 << 999999) + 1)", timeout=5.0)
    assert "too expensive" in safe_calculate("pow(3, 1 << 999999, (1 << 999999) + 1)", timeout=5.0)


def test_realistic_model_code_unaffected_by_new_bounds():
    """Model-realistic snippets far below the caps must pass: 10**6-scale loops and C-iterator
    builtins, list building, string join, comprehensions, modest repetition, small modpow."""
    expected = str(sum(range(10**6)))
    assert run_python_sandboxed("print(sum(range(10**6)))", timeout=15.0) == expected
    assert run_python_sandboxed("t = 0\nfor i in range(10**6):\n    t = t + i\nprint(t)", timeout=15.0) == expected
    assert run_python_sandboxed("x = [0] * 1000\nprint(len(x))") == "1000"
    assert run_python_sandboxed("print(len('ab' * 500))") == "1000"
    assert run_python_sandboxed("words = ['w'] * 100\nprint(len(''.join(words)))") == "100"
    assert run_python_sandboxed("print([i * i for i in range(5)])") == "[0, 1, 4, 9, 16]"
    assert run_python_sandboxed("print(max(range(100)))") == "99"
    assert run_python_sandboxed("print(pow(2, 10, 1000))") == "24"


def test_guard_names_not_shadowable():
    """User code must not shadow an injected operator guard before the transform routes through it."""
    assert "not allowed" in run_python_sandboxed("_sandbox_mul = 1\nx = 3 * 3")
    assert "not allowed" in safe_calculate("_sandbox_lshift")


def test_legitimate_shift_and_mul_still_work():
    """Everyday math must pass the guards untouched: float mul, small-int shifts, 2**64-scale
    products, loops, and sequence replication (int * list is not int * int)."""
    assert safe_calculate("1 << 8") == "256"
    assert safe_calculate("3 * 7.5") == "22.5"
    assert safe_calculate("2**64 * 2**64") == str(2**128)
    assert safe_calculate("6 * -7") == "-42"
    assert run_python_sandboxed("x = 1\nx <<= 10\nprint(x)") == "1024"
    assert run_python_sandboxed("t = 0\nfor i in range(100):\n    t = t + i * i\nprint(t)") == str(
        sum(i * i for i in range(100))
    )
    assert run_python_sandboxed("m = 2\nm *= 8\nprint(m)") == "16"
    assert run_python_sandboxed("print([0, 1] * 3)") == "[0, 1, 0, 1, 0, 1]"
    assert run_python_sandboxed("print('ab' * 2)") == "abab"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
