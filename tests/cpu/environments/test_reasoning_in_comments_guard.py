"""The code-contests guard on reasoning carried into a program's comments: the comment accounting by the
language's registered syntax, a program whose comments reach the bound and outweigh the rest refused unrun
on both tools with the call returned to its budget and the turn flagged untrainable, an honestly documented
program run, the offline re-grader skipping what the environment refused, and the metrics naming it."""

import ast
import json
import time

import pytest

from scripts.environments.inference.regrade_trajectories import submitted_solutions
from src.environments.base import REWARD_COMPONENTS_KEY
from src.environments.envs.tasks.coding.code_contests import REASONING_IN_COMMENTS_REPLY, CodeContestsEnvironment
from src.environments.envs.tasks.coding.comments import (
    REASONING_IN_COMMENTS_MIN_CHARS,
    _bare_string_chars,
    carries_reasoning_in_comments,
    comment_chars,
    reads_stdin,
)
from src.environments.episode import recovering_turn
from src.environments.sandbox.base import SandboxResult
from tests.common.code_contests import SINGLE_TEST_ANSWER, StubSandbox

MIN = REASONING_IN_COMMENTS_MIN_CHARS
# Comment lines past the bound beside eight characters of code: reasoning, not documentation.
LAUNDERED = "\n".join("# " + "r" * 100 for _ in range(MIN // 100)) + "\nprint(1)\n"
# 40 characters of comments beside 28 of code, far under the bound: documentation.
DOCUMENTED = "# reads n, prints it doubled\nn = int(input())\n# the answer\nprint(2 * n)\n"


def _call(name: str, **arguments) -> dict:
    return {"id": f"call_{name}", "function": {"name": name, "arguments": json.dumps(arguments)}}


def _env(**kwargs) -> CodeContestsEnvironment:
    return CodeContestsEnvironment(
        sandbox=StubSandbox(SandboxResult(stdout="X\n", returncode=0)), tool_error_penalty=0.05, **kwargs
    )


def test_comment_chars_reads_each_languages_registered_syntax():
    """Characters inside comments, and every other character of the program, newlines included."""
    assert comment_chars("# a\nx = 1\n", "python") == (3, 7)
    assert comment_chars("x = 1  # trailing reasoning\n", "python") == (20, 8)
    assert comment_chars("// a\nint x;\n/* b\n c */\ny;", "cpp") == (14, 11)
    assert comment_chars("int x; /* mid-line */ y;", "c") == (14, 10)
    assert comment_chars("#if 0\nold code\n#endif\nz;", "cpp") == (21, 3)
    assert comment_chars("# a\necho hi\n", "bash") == (3, 9)
    assert comment_chars("", "python") == (0, 0)


def test_comment_chars_skips_string_literals_and_counts_python_bare_strings():
    grid = 'grid = """' + "\n".join("#....#" for _ in range(3000)) + '"""\nprint(grid)\n'
    assert comment_chars(grid, "python") == (0, len(grid))
    assert not carries_reasoning_in_comments(grid, "python")
    c_strings = "s = \"// not a comment\"; t = '# nor this';"
    assert comment_chars(c_strings, "cpp") == (0, len(c_strings))
    py_string = 's = "# not a comment"\nprint(s)\n'
    assert comment_chars(py_string, "python") == (0, len(py_string))
    docstring = '"""' + "d" * 50 + '"""'
    assert comment_chars(f"{docstring}\nx = 1\n", "python") == (56, 7)


def _segment_chars(code: str) -> int:
    """The bare-string count as ``ast.get_source_segment`` slices each string statement."""
    return sum(
        len(ast.get_source_segment(code, node))
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
    )


def _seconds(fn, code: str) -> float:
    start = time.perf_counter()
    fn(code)
    return time.perf_counter() - start


@pytest.mark.parametrize(
    "code",
    [
        '"""Module docstring."""\nx = 1\n',
        'def f():\n    """Ünïcode docstring — 字"""\n    return "é"\n',
        'x = "ü"; "after a non-ASCII string on its line"; y = 2\n',
        '"""first\nsecond — ü\nthird 字 🙂"""\nz = 3',
        'class C:\n    "🙂 a four-byte character"\n    def m(self):\n        f"ü {self!r:>{4}} 字"\n        "ü {x}"\n',
        'b"bytes are no string statement"\n"ascii"\n',
        '"crlf"\r\nx = 1\r\n"after crlf ü"\r\n',
        '"cr"\rx = 1\r"after cr ü"\r',
        '\f"a form feed is no line break"\nx = "ü"; "ü"\n',
        '("implicit"\n "concatenation ü")\n',
        'x = 1 + \\\n  2; "after a continuation ü"\n',
    ],
    ids=[
        "docstring",
        "non-ascii-docstring",
        "shares-a-non-ascii-line",
        "multi-line",
        "four-byte-and-f-string",
        "bytes",
        "crlf",
        "cr",
        "form-feed",
        "implicit-concatenation",
        "continuation",
    ],
)
def test_bare_string_chars_match_the_segments_ast_slices(code):
    """The bare-string count reads each statement's span as ``ast.get_source_segment`` does: UTF-8 byte
    columns on non-ASCII lines, the parser's line breaks, f-strings and bytes left out."""
    assert _bare_string_chars(code) == _segment_chars(code) > 0


def test_bare_string_chars_grow_linearly_with_the_program():
    """Every scratchpad call and submission is counted inside a rollout actor. Re-splitting the source per
    string statement, as ``ast.get_source_segment`` does, costs hundreds of parses on 3,000 of them; one
    pass costs about two. Bounded against the parse, so a slow or loaded host moves both sides."""
    code = "".join(f'"bare string {i}: ü"\nx{i} = {i}\n' for i in range(3000))
    parse = min(_seconds(ast.parse, code) for _ in range(3))
    count = min(_seconds(_bare_string_chars, code) for _ in range(3))
    assert count < 20 * parse + 0.25, f"{count:.3f}s against a {parse:.3f}s parse"
    assert _bare_string_chars(code) == _segment_chars(code)


@pytest.mark.parametrize(
    ("code", "language", "reads"),
    [
        ("int n; std::cin >> n;", "cpp", True),
        ('int n; scanf("%d", &n);', "c", True),
        ("int main() { assert(f(1) == 2); }", "cpp", False),
        ("// cin >> n;\nint main() { return 0; }", "cpp", False),
        ("/* scanf later */ int main() { return 0; }", "c", False),
        ("n = int(input())", "python", True),
        ("assert sorted([2, 1]) == [1, 2]", "python", False),
        ('n = int(input("n? "))', "python", True),
        ("import sys\ndata = sys.stdin.read().split()", "python", True),
        ("data = open(0).read()", "python", True),
        ("import fileinput\nfor line in fileinput.input():\n    pass", "python", True),
        ("import sys\ninput = sys.stdin.readline\nn = int(input())", "python", True),
        ("import sys\nfor line in sys.stdin:\n    pass", "python", True),
        ("from sys import stdin\nn = int(stdin.readline())", "python", True),
        ("import sys\ndata = sys.stdin.buffer.read()", "python", True),
        ("import sys\nnums = list(map(int, sys.stdin))", "python", True),
        ("import os\ndata = os.read(0, 1 << 20)", "python", True),
        ("std::string s; std::getline(std::cin, s);", "cpp", True),
        ("char b[64]; std::cin.getline(b, 64);", "cpp", True),
        ("std::ios::sync_with_stdio(false), std::cin.tie(nullptr);\nint n; std::cin >> n;", "cpp", True),
        ("std::istream& in = std::cin; int n; in >> n;", "cpp", True),
        ('std::istringstream in("1"); int n; std::cin >> n;', "cpp", True),
        ("int c = getchar();", "cpp", True),
        ("char buf[64]; fgets(buf, 64, stdin);", "c", True),
        ("char *line = 0; size_t cap = 0; getline(&line, &cap, stdin);", "c", True),
        ("fread(buf, 1, n, stdin);", "c", True),
        ("read(0, buf, sizeof buf);", "c", True),
        ("read(STDIN_FILENO, buf, sizeof buf);", "c", True),
        ("read n", "bash", True),
        ("x=$(cat /dev/stdin)", "bash", True),
    ],
)
def test_reads_stdin_finds_each_languages_input_calls_outside_comments(code, language, reads):
    assert reads_stdin(code, language) is reads


@pytest.mark.parametrize(
    ("code", "language"),
    [
        (
            'import io, sys\nsys.stdin = io.StringIO("1 2\\n")\na, b = map(int, input().split())\nassert a + b == 3',
            "python",
        ),
        ("from io import StringIO\nimport sys\nsys.stdin = StringIO(SAMPLE)\nprint(sys.stdin.read())", "python"),
        ('sample_stdin = "1 2"\nassert solve(sample_stdin) == 3', "python"),
        (
            'def solve(s):\n    """Parse what input() returns off stdin."""\n    return s.split()\nassert solve("1")',
            "python",
        ),
        ('print("read it with input( or sys.stdin")', "python"),
        ("from sys import stdin\nprint(1)", "python"),
        ("import sys\ninput = sys.stdin.readline\nassert sorted([2, 1]) == [1, 2]", "python"),
        ("def run(stdin):\n    return stdin.split()\nassert run('1 2') == ['1', '2']", "python"),
        ('std::istringstream in("1 2\\n");\nstd::string line;\nstd::getline(in, line);', "cpp"),
        ('std::istringstream in("3"); std::cin.rdbuf(in.rdbuf()); int n; std::cin >> n;', "cpp"),
        (
            'std::ios::sync_with_stdio(false), std::cin.tie(nullptr);\nstd::istringstream in("3");\n'
            "int n; in >> n; assert(n == 3);",
            "cpp",
        ),
        ('FILE *f = fopen("in.txt", "r"); char buf[64]; fgets(buf, 64, f);', "c"),
        ('puts("scanf from stdin");', "c"),
        ("echo 'read the stdin'", "bash"),
    ],
    ids=[
        "python-stringio-redirect",
        "python-stringio-redirect-imported",
        "python-name-ending-in-stdin",
        "python-docstring",
        "python-string-literal",
        "python-import-alone",
        "python-fast-io-template-unused",
        "python-parameter-named-stdin",
        "cpp-istringstream-getline",
        "cpp-cin-rdbuf-redirect",
        "cpp-fast-io-boilerplate",
        "c-fgets-on-a-file",
        "c-string-literal",
        "bash-string-literal",
    ],
)
def test_a_program_that_embeds_its_input_reads_no_stdin(code, language):
    """A self-test that feeds itself from a string, names a variable after stdin, keeps a fast-IO template it never
    reads through, or mentions an input call in a string or docstring reads no standard input: flagged, its silent
    run would be booked as starved."""
    assert not reads_stdin(code, language)


def test_an_unregistered_language_has_no_comment_syntax():
    with pytest.raises(ValueError, match="unsupported language 'fortran'"):
        comment_chars("x", "fortran")


def test_the_bound_is_pinned_at_the_constant():
    assert carries_reasoning_in_comments("#" + "r" * (MIN - 1), "python")
    assert not carries_reasoning_in_comments("#" + "r" * (MIN - 2), "python")
    assert not carries_reasoning_in_comments("#" + "r" * MIN + "\n" + "x" * (MIN + 1), "python")


def test_a_laundered_program_is_refused_on_both_tools_and_the_budget_returned():
    env = _env(max_test_calls=1, max_submissions=1)
    ids, _ = env.reset(["solve it"], [SINGLE_TEST_ANSWER])
    step = env.step(ids, [""], [{"tool_calls": [_call(env.test_tool_name, code=LAUNDERED)]}])[0]
    traj = step.trajectory
    reply = traj.messages[-1]
    assert reply.role == "tool" and reply.content == "Error: " + REASONING_IN_COMMENTS_REPLY.format(verb="run")
    assert step.reward == pytest.approx(-0.05) and not step.done
    assert traj.info["reasoning_in_comments_calls"] == 1 and env._test_calls(traj) == 0
    # A turn of nothing but refusals trains only on a negative advantage, and the next turn is a retry.
    assert traj.messages[-2].role == "assistant" and traj.messages[-2].untrainable and recovering_turn(traj)

    step = env.step(ids, [""], [{"tool_calls": [_call("submit_solution", code=LAUNDERED)]}])[0]
    assert traj.messages[-1].content == "Error: " + REASONING_IN_COMMENTS_REPLY.format(verb="graded")
    assert not step.done and env._submissions(traj) == 0 and traj.info["reasoning_in_comments_calls"] == 2

    # The budget it returned is still there, and the accounting books the later calls as their own.
    step = env.step(ids, [""], [{"tool_calls": [_call(env.test_tool_name, code=DOCUMENTED, stdin="3\n")]}])[0]
    assert traj.messages[-1].content.startswith("X") and traj.info["successful_tool_calls"] == 1
    assert not traj.messages[-2].untrainable and not recovering_turn(traj)
    step = env.step(ids, [""], [{"tool_calls": [_call("submit_solution", code=DOCUMENTED)]}])[0]
    assert step.done and traj.info["submission_result"].startswith("Passed")
    metrics = env.rollout_metrics(traj)
    assert metrics["episode/reasoning_in_comments_calls"] == 2.0
    laundered = comment_chars(LAUNDERED, "python")
    documented = comment_chars(DOCUMENTED, "python")
    comments, rest = 2 * laundered[0] + 2 * documented[0], 2 * laundered[1] + 2 * documented[1]
    assert metrics["episode/code_comment_share"] == pytest.approx(comments / (comments + rest))
    assert traj.info[REWARD_COMPONENTS_KEY]["reward/turn_shaping"] == pytest.approx(-0.1)


def test_a_c_family_program_is_read_by_its_own_syntax():
    env = _env(language=["python", "cpp"])
    ids, _ = env.reset(["solve it"], [SINGLE_TEST_ANSWER])
    laundered = "\n".join("// " + "r" * 100 for _ in range(MIN // 100)) + "\nint main(){}\n"
    step = env.step(ids, [""], [{"tool_calls": [_call("run_code", code=laundered, language="cpp")]}])[0]
    assert step.trajectory.messages[-1].content == "Error: " + REASONING_IN_COMMENTS_REPLY.format(verb="run")
    block = "/*" + "\n".join("r" * 100 for _ in range(MIN // 100)) + "*/\nint main(){}\n"
    step = env.step(ids, [""], [{"tool_calls": [_call("run_code", code=block, language="cpp")]}])[0]
    assert step.trajectory.messages[-1].content == "Error: " + REASONING_IN_COMMENTS_REPLY.format(verb="run")
    assert step.trajectory.info["reasoning_in_comments_calls"] == 2


def test_comments_under_the_bound_or_under_the_code_run():
    env = _env()
    ids, _ = env.reset(["solve it"], [SINGLE_TEST_ANSWER])
    step = env.step(ids, [""], [{"tool_calls": [_call(env.test_tool_name, code=DOCUMENTED, stdin="3\n")]}])[0]
    assert step.trajectory.messages[-1].content.startswith("X") and step.reward == 0.0
    # Many comments beside more code than that: documentation at scale, not laundering.
    heavy = (
        "\n".join(f"# note {i}" for i in range(2000))
        + "\n"
        + "\n".join(f"x{i} = {i}" for i in range(4000))
        + "\nprint(1)\n"
    )
    assert comment_chars(heavy, "python")[0] >= MIN
    step = env.step(ids, [""], [{"tool_calls": [_call(env.test_tool_name, code=heavy, stdin="3\n")]}])[0]
    assert step.trajectory.messages[-1].content.startswith("X")
    assert "reasoning_in_comments_calls" not in step.trajectory.info


def test_a_laundered_call_after_the_accept_takes_the_free_post_accept_reply():
    env = _env(max_submissions=3)
    ids, _ = env.reset(["solve it"], [SINGLE_TEST_ANSWER])
    calls = [_call("submit_solution", code=DOCUMENTED), {**_call("submit_solution", code=LAUNDERED), "id": "call_2"}]
    step = env.step(ids, [""], [{"tool_calls": calls}])[0]
    assert step.done and step.trajectory.info["submission_result"].startswith("Passed")
    assert "already passed" in step.trajectory.messages[-1].content and step.reward >= 0.0
    assert "reasoning_in_comments_calls" not in step.trajectory.info


def test_the_regrader_takes_no_slot_for_a_refused_program():
    env = _env(max_submissions=1)
    episode = {
        "messages": [
            {"role": "assistant", "tool_calls": [_call("submit_solution", code=LAUNDERED)]},
            {"role": "assistant", "tool_calls": [_call("submit_solution", code=DOCUMENTED)]},
        ]
    }
    assert submitted_solutions(episode, env) == [(DOCUMENTED, None)]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
