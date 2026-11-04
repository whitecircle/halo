#!/usr/bin/env python
import math
import time

import pytest

from halo.environments.sandbox.inprocess import (
    SAFE_MATH_BUILTINS,
    SAFE_PYTHON_BUILTINS,
    calculate_sandboxed,
    run_python_sandboxed,
)
from halo.environments.sandbox.repl import REPL_NO_OUTPUT_MESSAGE


def test_calc_basic_arithmetic():
    assert calculate_sandboxed("2 + 3") == "5"
    assert calculate_sandboxed("10 - 4") == "6"
    assert calculate_sandboxed("3 * 7") == "21"
    assert calculate_sandboxed("15 / 3") == "5.0"
    assert calculate_sandboxed("10 % 3") == "1"


def test_calc_exponentiation():
    assert calculate_sandboxed("2^3") == "8", "Caret should be converted to **"
    assert calculate_sandboxed("2**3") == "8"


def test_calc_unicode_operators():
    assert calculate_sandboxed("6\u00d73") == "18", "Unicode multiply \u00d7"
    assert calculate_sandboxed("10\u00f72") == "5.0", "Unicode divide \u00f7"


def test_calc_math_functions():
    assert calculate_sandboxed("sqrt(16)") == "4.0"
    assert calculate_sandboxed("abs(-5)") == "5"
    assert calculate_sandboxed("floor(3.7)") == "3"
    assert calculate_sandboxed("ceil(3.2)") == "4"
    result = float(calculate_sandboxed("sin(0)"))
    assert abs(result) < 1e-10, f"sin(0) should be ~0, got {result}"


def test_calc_constants():
    result = float(calculate_sandboxed("pi"))
    assert abs(result - math.pi) < 1e-10
    result = float(calculate_sandboxed("e"))
    assert abs(result - math.e) < 1e-10


def test_calc_division_by_zero():
    assert "Error" in calculate_sandboxed("1/0")
    assert "zero" in calculate_sandboxed("1/0").lower()


def test_calc_invalid_expression():
    result = calculate_sandboxed("abc def ghi")
    assert "Error" in result


def test_calc_no_builtins_access():
    result = calculate_sandboxed("__import__('os')")
    assert "Error" in result


def test_calc_no_open():
    result = calculate_sandboxed("open('/etc/passwd')")
    assert "Error" in result


def test_calc_whitespace():
    assert calculate_sandboxed("  2 + 3  ") == "5"


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
    """The last bound local is reported; a ``"6" in result or "success" in result.lower()`` check would pass on
    either branch."""
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
    """Overriding __builtins__ alone does not contain it, so AST validation blocks dunder access before execution."""
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
    """A generator frame walk reaches the real builtins dict and __import__. ``gi_frame``/``f_back``/``f_globals``
    are no dunders, so the introspection-prefix guard must reject them."""
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
    """The general form of the frame-escape's final hop."""
    result = run_python_sandboxed("{}['__class__']")
    assert "Error" in result
    assert "dunder" in result.lower()


def test_sandbox_allows_normal_attributes_and_subscripts():
    """The introspection guard must not break legitimate method calls / subscripts."""
    assert run_python_sandboxed("'hello world'.upper()") == "HELLO WORLD"
    assert run_python_sandboxed("{'a': 1, 'b': 2}['b']") == "2"
    assert run_python_sandboxed("[10, 20, 30][-1]") == "30"


def test_sandbox_times_out_on_infinite_loop():
    result = run_python_sandboxed("while True:\n    pass", timeout=0.5)
    assert "Error" in result
    assert "timeout" in result.lower()


def test_sandbox_multiline():
    code = "result = 0\nfor i in range(5):\n    result += i\nprint(result)"
    result = run_python_sandboxed(code)
    assert "10" in result


def test_sandbox_no_output_no_vars_success_message():
    # A bare for-loop would bind its loop variable, so use a no-op that binds nothing.
    result = run_python_sandboxed("pass")
    assert "executed successfully" in result.lower()


def test_sandbox_last_var_is_returned_not_first():
    result = run_python_sandboxed("a = 1\nb = 2\nc = 99")
    assert result == "99"


def test_sandbox_statement_nameerror_caught():
    result = run_python_sandboxed("y = 1\nz = undefined_name + y")
    assert "Error" in result
    assert "undefined_name" in result


def test_sandbox_functions_see_their_own_name_and_the_names_beside_them():
    """Code runs in one namespace, as a module does; class-body scoping raises NameError on recursion and on a
    top-level read."""
    recursive = "def fact(n):\n    return 1 if n < 2 else n * fact(n - 1)\nprint(fact(5))"
    assert run_python_sandboxed(recursive) == "120"
    assert run_python_sandboxed("k = 3\ndef f(x):\n    return x * k\nprint(f(2))") == "6"


def test_calc_walrus_binding_stays_in_its_own_call():
    """On a table shared across calls, one episode's ``pi := 3`` would answer every later calculation in the
    process."""
    assert calculate_sandboxed("(pi := 3)") == "3"
    assert calculate_sandboxed("pi") == str(math.pi)
    assert calculate_sandboxed("(sqrt := abs)(-4)") == "4"
    assert "Error" in calculate_sandboxed("sqrt(-4)")


def test_calc_function_bodies_see_the_math_names():
    assert calculate_sandboxed("(lambda x: sqrt(x))(16)") == "4.0"


def test_sandbox_renders_the_real_message_for_every_user_error():
    """An except tuple omitting ``SyntaxError`` or ``AssertionError`` lets them escape the worker thread, and the
    model sees only an opaque no-result message."""
    syntax = run_python_sandboxed("def f(:\n    pass")
    assert syntax.startswith("Error:") and "produced no result" not in syntax
    assert "syntax" in syntax.lower() or "invalid" in syntax.lower()

    # A bare assert has no message, so the class name must stand in.
    assertion = run_python_sandboxed("x = 1\nassert x == 2")
    assert assertion.startswith("Error:") and "produced no result" not in assertion
    assert assertion.strip() != "Error:"


def test_sandbox_print_with_separator():
    result = run_python_sandboxed("print('a', 'b', 'c')")
    assert result == "a b c"


def test_sandbox_bare_print_outputs():
    """The tool's description says to use print(), and a single ``print(...)`` expression takes the eval path, which
    must define it too."""
    assert run_python_sandboxed("print('hi')") == "hi"


@pytest.mark.parametrize(
    ("code", "printed"),
    [
        ("for i in range(3):\n    print(i, end='')", "012"),
        ("print('a', end=', ')\nprint('b')", "a, b"),
        ("print('a', 'b', sep='-', end='!\\n')\nprint('c')", "a-b!\nc"),
        ("print('row', end='\\n\\n')\nprint('next')", "row\n\nnext"),
        ("print('x', sep=None, end=None, flush=True)", "x"),
    ],
)
def test_sandbox_print_writes_what_python_s_print_writes(code, printed):
    """A capture joining one line per call turns ``end=''`` output into one value per line."""
    assert run_python_sandboxed(code) == printed


def test_sandbox_print_of_nothing_reads_as_no_output():
    """What the program printed is read like the subprocess REPL reads stdout: trailing newlines dropped,
    and an empty print is no output rather than an empty observation."""
    assert run_python_sandboxed("print('a')\nprint()") == "a"
    assert run_python_sandboxed("print('', end='')") == REPL_NO_OUTPUT_MESSAGE


def test_calc_negative_result():
    assert calculate_sandboxed("3 - 10") == "-7"


def test_math_builtins_subset_of_python():
    """The python sandbox must offer everything the calculator does; a refactor dropping the spread would regress
    it."""
    for key in SAFE_MATH_BUILTINS:
        assert key in SAFE_PYTHON_BUILTINS, f"{key} in SAFE_MATH_BUILTINS but not in SAFE_PYTHON_BUILTINS"
    assert set(SAFE_MATH_BUILTINS).issubset(SAFE_PYTHON_BUILTINS)
    assert len(SAFE_PYTHON_BUILTINS) > len(SAFE_MATH_BUILTINS)


def test_calc_factorial_and_gcd_execute():
    assert calculate_sandboxed("factorial(5)") == "120"
    assert calculate_sandboxed("gcd(12, 18)") == "6"
    assert calculate_sandboxed("log2(8)") == "3.0"


def test_sandbox_data_structure_builtins_execute():
    assert run_python_sandboxed("dict(zip(['a', 'b'], [1, 2]))") == "{'a': 1, 'b': 2}"
    result = run_python_sandboxed("list(enumerate(['x', 'y']))")
    assert result == "[(0, 'x'), (1, 'y')]"


def test_calc_timeout_on_slow_evaluation():
    """Sync tool execution runs on a Ray actor's event loop, so an unbounded eval wedges every episode there. The
    injected builtin yields the GIL, the class the daemon-thread timeout can preempt."""
    SAFE_MATH_BUILTINS["hang"] = lambda: time.sleep(30) or 0
    try:
        result = calculate_sandboxed("hang() + 1", timeout=0.2)
    finally:
        del SAFE_MATH_BUILTINS["hang"]
    assert result == "Error: execution exceeded 0.2s timeout"


def test_calc_fast_expression_unaffected_by_timeout():
    assert calculate_sandboxed("2 + 3", timeout=5.0) == "5"


def test_pow_bomb_rejected_not_hung():
    """A single GIL-holding C-level pow that the timeout thread can never preempt, which would wedge the worker and
    its actor."""
    assert calculate_sandboxed("9**9**9", timeout=5.0) == "Error: exponentiation result too large for the sandbox"
    assert run_python_sandboxed("9**9**9", timeout=5.0) == "Error: exponentiation result too large for the sandbox"
    assert "too large" in calculate_sandboxed("pow(9, 387420489)", timeout=5.0)
    assert "too large" in run_python_sandboxed("x = 2\nx **= 10**9", timeout=5.0)


def test_factorial_bomb_rejected_not_hung():
    """``factorial(10**9)`` is the same unpreemptible C-level class; the argument bound rejects it."""
    assert "too large" in calculate_sandboxed("factorial(10**9)", timeout=5.0)


def test_legitimate_pow_and_factorial_still_work():
    assert calculate_sandboxed("2**10") == "1024"
    assert calculate_sandboxed("2**1000") == str(2**1000)  # big but bounded results stay exact
    assert calculate_sandboxed("pow(2, 10, 1000)") == "24"  # realistic modpow stays well under the cost bound
    assert calculate_sandboxed("factorial(20)") == str(math.factorial(20))
    assert calculate_sandboxed("(-2)**3") == "-8"
    assert calculate_sandboxed("2**-2") == "0.25"
    assert run_python_sandboxed("x = 3\nx **= 4\nprint(x)") == "81"


def test_shift_bomb_rejected_not_hung():
    """Builds a 32 MB int in one unpreemptible C call, and any use then holds the GIL for minutes."""
    assert calculate_sandboxed("1 << 268435456", timeout=5.0) == "Error: shift result too large for the sandbox"
    assert run_python_sandboxed("1 << 268435456", timeout=5.0) == "Error: shift result too large for the sandbox"
    assert "too large" in run_python_sandboxed("x = 1\nx <<= 268435456", timeout=5.0)


def test_mul_bomb_rejected_not_hung():
    """A single GIL-holding C multiplication the timeout can never preempt."""
    assert (
        calculate_sandboxed("(1 << 900000) * (1 << 900000)", timeout=5.0)
        == "Error: multiplication result too large for the sandbox"
    )
    assert (
        run_python_sandboxed("a = 1 << 900000\nb = a * a", timeout=5.0)
        == "Error: multiplication result too large for the sandbox"
    )
    assert "too large" in run_python_sandboxed("a = 1 << 900000\na *= a", timeout=5.0)


def test_range_c_iterator_bomb_rejected_not_hung():
    """The range drains inside one GIL-holding C loop: with a 1s timeout ``sum(range(8*10**7))`` runs to completion
    (1.2s), so a 10**18 range would wedge the worker."""
    start = time.monotonic()
    assert "range too large" in run_python_sandboxed("sum(range(10**18))", timeout=1.0)
    assert time.monotonic() - start < 1.0, "must be rejected by the guard, not saved by the bypassable timeout"
    assert "range too large" in run_python_sandboxed("min(range(10**12))", timeout=1.0)
    assert "range too large" in run_python_sandboxed("list(range(10**10))", timeout=1.0)
    assert "range too large" in run_python_sandboxed("x = range(10**14)\nsorted(x)", timeout=1.0)


def test_sequence_repetition_bomb_rejected_not_hung():
    """These allocate the whole multi-GB result in one C call; an int-only mul guard passes them. The guard bounds
    len x count, both orders and the augmented form."""
    assert "sequence repetition" in run_python_sandboxed("[0] * 10**9", timeout=5.0)
    assert "sequence repetition" in run_python_sandboxed("10**9 * [0]", timeout=5.0)
    assert "sequence repetition" in run_python_sandboxed("'a' * 10**9", timeout=5.0)
    assert "sequence repetition" in run_python_sandboxed("b'a' * 10**9", timeout=5.0)
    assert "sequence repetition" in run_python_sandboxed("(1, 2) * 10**9", timeout=5.0)
    assert "sequence repetition" in run_python_sandboxed("x = [1, 2]\nx *= 10**9", timeout=5.0)
    assert "sequence repetition" in calculate_sandboxed("(1, 2) * 10**9", timeout=5.0)


def test_modpow_cost_bomb_rejected_not_hung():
    """~1e6 modmuls of 1e6-bit numbers inside one C call (minutes of GIL hold); the bits(exp)*bits(mod) cost bound
    rejects it."""
    assert "too expensive" in run_python_sandboxed("pow(3, 1 << 999999, (1 << 999999) + 1)", timeout=5.0)
    assert "too expensive" in calculate_sandboxed("pow(3, 1 << 999999, (1 << 999999) + 1)", timeout=5.0)


def test_realistic_model_code_unaffected_by_new_bounds():
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
    assert "not allowed" in calculate_sandboxed("_sandbox_lshift")


def test_legitimate_shift_and_mul_still_work():
    """Everyday math must pass the guards untouched: float mul, small-int shifts, 2**64-scale
    products, loops, and sequence replication (int * list is not int * int)."""
    assert calculate_sandboxed("1 << 8") == "256"
    assert calculate_sandboxed("3 * 7.5") == "22.5"
    assert calculate_sandboxed("2**64 * 2**64") == str(2**128)
    assert calculate_sandboxed("6 * -7") == "-42"
    assert run_python_sandboxed("x = 1\nx <<= 10\nprint(x)") == "1024"
    assert run_python_sandboxed("t = 0\nfor i in range(100):\n    t = t + i * i\nprint(t)") == str(
        sum(i * i for i in range(100))
    )
    assert run_python_sandboxed("m = 2\nm *= 8\nprint(m)") == "16"
    assert run_python_sandboxed("print([0, 1] * 3)") == "[0, 1, 0, 1, 0, 1]"
    assert run_python_sandboxed("print('ab' * 2)") == "abab"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
