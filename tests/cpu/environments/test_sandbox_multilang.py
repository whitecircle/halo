#!/usr/bin/env python
"""
Tests for multi-language sandbox execution, persistent sessions, and the bubblewrap backend.

Covers the behaviour layered over the python-only backend (see test_sandbox.py for the
python/remote/grading basics):

- Language registry: name/alias resolution, the compiled-vs-interpreted distinction.
- bash: a command run in the session working directory — it sees the session's files, its own writes
  stay there, a non-zero exit is its own verdict — on the local backend and in a bwrap jail; on the
  remote backend the shell tool puts the canonical ``bash`` on the wire.
- C/C++ compile-and-run: stdout, stdin plumbing, a compile error reported as a *program* fault
  (returncode set, ``error`` unset) so grading buckets it as a failed solution, a runtime timeout,
  and the memory cap on a compiled binary.
- Missing-compiler handling (a backend error) via a sandbox pointed at a bogus toolchain.
- Sessions: state persists across runs (write a file, read it next run; cross-language sharing),
  two sessions are isolated, concurrent sessions on one shared instance don't cross-contaminate, and
  the host never follows a link the program planted (staging, ``read_file``, ``reset_to_staged``).
- Session build reuse: a compiled source is built once across runs with different stdin (the grader's
  one-session-per-submission pattern), rebuilt when the source, an auxiliary header, a session-written
  file or an interpreted run changes the working dir, and a rejected compile is cached as
  ``compile_failed`` instead of recompiling per test.
- RemoteSession: client-side file accumulation resent on every request (no network — fake session).
- BubblewrapSandbox: construction guard when bwrap is absent, a probe that names the clean-proc
  remedy, the init reap and status read-back against a stand-in ``bwrap`` (no namespaces needed), and —
  when bwrap IS present — network is unshared and the host filesystem is hidden while the working dir
  stays writable, and no run leaves a process behind.

Run: python tests/cpu/environments/test_sandbox_multilang.py
"""

import os
import shutil
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.environments.sandbox import base as _base
from src.environments.sandbox import bubblewrap as _bubblewrap
from src.environments.sandbox import local as local_sandbox
from src.environments.sandbox.base import LANGUAGES, resolve_language, supported_languages
from src.environments.sandbox.bubblewrap import BubblewrapSandbox
from src.environments.sandbox.local import TAMPERED_WORKDIR_RETURNCODE, LocalSubprocessSandbox
from src.environments.sandbox.remote import RemoteSandbox
from src.environments.sandbox.resolve import resolve_sandbox
from src.environments.tools.factories import create_session_bash_tools
from tests.common.code_contests import RecordingSandboxSession
from tests.common.utils import probe_findings

_HAS_GPP = shutil.which("g++") is not None
_HAS_GCC = shutil.which("gcc") is not None


def _usable_bubblewrap():
    """Return a working BubblewrapSandbox, or None if bwrap is absent OR can't create a sandbox here.

    ``BubblewrapSandbox`` probes at construction and raises when user/mount namespaces are blocked
    (e.g. under Docker's default seccomp) — so functional bubblewrap tests skip unless a real,
    working jail is available (not merely the binary).
    """
    if shutil.which("bwrap") is None:
        return None
    try:
        return BubblewrapSandbox()
    except RuntimeError:
        return None


_BWRAP = _usable_bubblewrap()

_CPP_DOUBLE = """
#include <iostream>
int main() { long n; std::cin >> n; std::cout << n * 2 << std::endl; return 0; }
"""


def _skip(reason: str):
    """Register a genuine skip (not a silent pass).

    ``unittest.SkipTest`` is what pytest treats as a skip, so a bubblewrap functional test that
    cannot run here is reported as skipped, never as a passing test that exercised nothing.
    """
    raise unittest.SkipTest(reason)


# Language registry


def test_language_registry_resolves_aliases():
    assert resolve_language("python").name == "python"
    assert resolve_language("py").name == "python"
    assert resolve_language("c++").name == "cpp"
    assert resolve_language("CXX").name == "cpp"  # case-insensitive
    assert resolve_language("c").name == "c"
    assert resolve_language("sh").name == "bash"
    assert resolve_language("SHELL").name == "bash"  # case-insensitive
    assert resolve_language("ruby") is None
    assert set(supported_languages()) == set(LANGUAGES.keys()) == {"python", "bash", "cpp", "c"}


def test_language_compiled_flag():
    assert resolve_language("python").is_compiled is False
    assert resolve_language("bash").is_compiled is False
    assert resolve_language("cpp").is_compiled is True
    assert resolve_language("c").is_compiled is True


def test_bash_runs_in_the_session_working_directory():
    """The shell sees the session's files and its own writes stay there, so a workspace tool set can
    mix file tools and shell commands."""
    with LocalSubprocessSandbox().open_session() as session:
        session.write_file("notes.txt", "remember me\n")
        res = session.run("cat notes.txt; echo made > made.txt", language="sh")
        assert res.ok, f"expected clean exit, got {res}"
        assert res.stdout.strip() == "remember me"
        assert session.read_file("made.txt").strip() == "made"


def test_bash_nonzero_exit_is_the_command_s_own_verdict():
    res = LocalSubprocessSandbox().run("exit 3", language="bash")
    assert res.returncode == 3
    assert res.error is None and not res.compile_failed and not res.timed_out


# C/C++ compile-and-run (local backend)


def test_cpp_compiles_and_runs_with_stdin():
    if not _HAS_GPP:
        return _skip("g++ not installed")
    res = LocalSubprocessSandbox().run(_CPP_DOUBLE, stdin="21", language="cpp")
    assert res.ok, f"expected clean exit, got {res}"
    assert res.stdout.strip() == "42"
    assert res.returncode == 0
    assert not res.compile_failed


def test_cpp_alias_runs():
    if not _HAS_GPP:
        return _skip("g++ not installed")
    res = LocalSubprocessSandbox().run(_CPP_DOUBLE, stdin="5", language="c++")
    assert res.ok and res.stdout.strip() == "10"


def test_c_compiles_and_runs():
    if not _HAS_GCC:
        return _skip("gcc not installed")
    code = '#include <stdio.h>\nint main(){int n;scanf("%d",&n);printf("%d",n+1);return 0;}'
    res = LocalSubprocessSandbox().run(code, stdin="41", language="c")
    assert res.ok and res.stdout.strip() == "42"


def test_cpp_compile_error_is_program_fault_not_backend_error():
    """A compile error must read as the solution's fault (returncode set, error unset),
    so grading counts it as a failed test rather than a sandbox outage."""
    if not _HAS_GPP:
        return _skip("g++ not installed")
    res = LocalSubprocessSandbox().run("int main(){ this is not valid c++ }", language="cpp")
    assert not res.ok
    assert res.compile_failed, "a rejected compile must be flagged so the grader can name the verdict"
    assert res.error is None, "compile error must NOT set the backend-error flag"
    assert res.returncode not in (0, None), "compile failure should carry the compiler's exit code"
    assert "error" in res.stderr.lower(), "compiler diagnostics should be surfaced on stderr"
    assert not res.timed_out


def test_python_runs_never_set_compile_failed():
    """``compile_failed`` is a compile-step verdict; an interpreted run has no compile step, so neither
    a clean nor a crashing Python program may carry it (a grader would misname the crash)."""
    sb = LocalSubprocessSandbox()
    clean = sb.run("print(1)")
    assert clean.ok and not clean.compile_failed
    crashed = sb.run("raise ValueError('boom')")
    assert not crashed.ok and crashed.returncode not in (0, None)
    assert not crashed.compile_failed


def test_cpp_runtime_timeout():
    if not _HAS_GPP:
        return _skip("g++ not installed")
    code = "int main(){ while(true){} }"
    res = LocalSubprocessSandbox().run(code, timeout=0.5, language="cpp")
    assert res.timed_out, f"expected timeout, got {res}"
    assert not res.ok


def test_cpp_runtime_nonzero_exit():
    if not _HAS_GPP:
        return _skip("g++ not installed")
    res = LocalSubprocessSandbox().run("int main(){ return 3; }", language="cpp")
    assert not res.ok
    assert res.returncode == 3
    assert res.error is None and not res.timed_out


def test_cpp_memory_cap_applies_to_binary():
    """RLIMIT_AS is inherited by the compiled binary: a huge allocation fails the child, not host."""
    if not _HAS_GPP:
        return _skip("g++ not installed")
    code = (
        "#include <vector>\n#include <cstdio>\n"
        'int main(){ std::vector<char> v(2L*1024*1024*1024, 1); printf("%zu", v.size()); }'
    )
    res = LocalSubprocessSandbox(memory_limit_mb=256).run(code, language="cpp")
    assert not res.ok, "a 2GiB allocation past the 256MiB cap must not succeed"
    assert not res.timed_out


def test_missing_compiler_is_backend_error():
    """Pointing the spec at a non-existent compiler surfaces a backend error, not a crash."""
    sb = LocalSubprocessSandbox()

    original = _base.LANGUAGES["cpp"]
    _base.LANGUAGES["cpp"] = _base.LanguageSpec(
        name="cpp",
        source_name="main.cpp",
        compile_argv=("g++_does_not_exist_xyz", "-o", "main", "main.cpp"),
        run_argv=("./main",),
        aliases=("c++",),
    )
    try:
        res = sb.run("int main(){}", language="cpp")
        assert res.error is not None
        assert "compiler not found" in res.error.lower()
        assert not res.compile_failed, "a missing compiler is a backend failure, not the source's fault"
    finally:
        _base.LANGUAGES["cpp"] = original


def test_cpp_auxiliary_header_file():
    if not _HAS_GPP:
        return _skip("g++ not installed")
    prog = '#include <iostream>\n#include "h.h"\nint main(){ std::cout << val(); }'
    res = LocalSubprocessSandbox().run(prog, language="cpp", files={"h.h": "inline int val(){return 7;}"})
    assert res.ok and res.stdout.strip() == "7"


# Parallel safety (one shared instance, concurrent runs)


def test_parallel_runs_do_not_cross_contaminate():
    """Concurrent executions on a single shared sandbox must not see each other's output."""
    sb = LocalSubprocessSandbox()
    results = {}

    def worker(i: int):
        res = sb.run(f"print({i} * {i})", language="python")
        results[i] = res.stdout.strip()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == {i: str(i * i) for i in range(16)}


# Sessions (persistent state across runs)


def test_session_persists_files_across_runs():
    sb = LocalSubprocessSandbox()
    with sb.open_session() as session:
        first = session.run("with open('state.txt', 'w') as f: f.write('carried')\nprint('wrote')")
        assert first.ok and first.stdout.strip() == "wrote"
        second = session.run("print(open('state.txt').read())")
        assert second.ok and second.stdout.strip() == "carried"


def test_session_write_read_list_files():
    sb = LocalSubprocessSandbox()
    with sb.open_session() as session:
        session.write_file("a.txt", "alpha")
        session.write_file("pkg/b.txt", "beta")
        assert session.read_file("a.txt") == "alpha"
        assert session.read_file("pkg/b.txt") == "beta"
        assert session.read_file("missing.txt") is None
        assert session.list_files() == ["a.txt", "pkg/b.txt"]


def test_session_reads_a_file_the_program_wrote_in_bytes_that_are_not_utf8():
    """A program's output file is the program's own bytes: read back under the strict locale codec, a
    stray ``\\xff`` raises out of ``read_file`` and turns the model's file read into a broken tool."""
    with LocalSubprocessSandbox().open_session() as session:
        res = session.run("open('out.bin', 'wb').write(b'ok\\xff\\n')")
        assert res.ok, res
        assert session.read_file("out.bin") == "ok\ufffd\n"


def test_session_cross_language_sharing():
    """A file written via the session is visible to BOTH a python run and a compiled-cpp run."""
    if not _HAS_GPP:
        return _skip("g++ not installed")
    sb = LocalSubprocessSandbox()
    with sb.open_session() as session:
        session.write_file("inc.h", "inline int f(){return 9;}")
        cpp = '#include <iostream>\n#include "inc.h"\nint main(){ std::cout << f(); }'
        rc = session.run(cpp, language="cpp")
        assert rc.ok and rc.stdout.strip() == "9"
        rp = session.run("print(open('inc.h').read().strip())", language="python")
        assert rp.ok and "f()" in rp.stdout


def test_sessions_are_isolated_from_each_other():
    sb = LocalSubprocessSandbox()
    s1, s2 = sb.open_session(), sb.open_session()
    try:
        s1.write_file("secret.txt", "one")
        assert s2.read_file("secret.txt") is None, "sessions must not share a working directory"
        assert s2.list_files() == []
    finally:
        s1.close()
        s2.close()


def test_session_close_removes_workdir():
    sb = LocalSubprocessSandbox()
    session = sb.open_session()
    session.write_file("x.txt", "y")
    workdir = session.workdir
    assert os.path.isdir(workdir)
    session.close()
    assert not os.path.exists(workdir), "close() must delete the working directory"


def test_session_rejects_unsafe_paths():
    sb = LocalSubprocessSandbox()
    with sb.open_session() as session:
        raised = False
        try:
            session.write_file("../escape.txt", "x")
        except ValueError:
            raised = True
        assert raised, "a traversal path must be rejected"


def _plant_link(workdir: str, name: str, target: str) -> None:
    """What a program can do to its own working directory between runs."""
    path = os.path.join(workdir, name)
    if os.path.lexists(path):
        os.remove(path)
    os.symlink(target, path)


def test_staging_never_writes_through_a_link_the_program_planted(tmp_path):
    """Under bubblewrap the working directory is a plain rw bind: a program can replace the staged
    ``main.py`` with a symlink to a host file, and the next run's staging — the host process, as
    root — would write the new source through it. The program is judged for it (a runtime error of
    its own), never handed an infra error it could void its episode with."""
    host_file = tmp_path / "credentials"
    host_file.write_text("SECRET")
    sb = LocalSubprocessSandbox()
    with sb.open_session() as session:
        assert session.run("print(1)").ok
        _plant_link(session.workdir, "main.py", str(host_file))

        result = session.run("print(2)")

        assert host_file.read_text() == "SECRET"
        assert result.error is None, "tampering must not read as an infra fault"
        assert result.returncode == TAMPERED_WORKDIR_RETURNCODE and not result.ok
        assert "main.py" in result.stderr
        # A verdict the model can recover from (a link at a staged name may be a benign in-workspace
        # one), not an agent fault that ends the episode.
        assert result.agent_fault is None


def test_reset_to_staged_drops_a_staged_entry_whose_type_changed(tmp_path):
    """The grader resets the session between hidden tests, so a link planted over a staged entry by
    test 1 must be gone before test 2 is staged — a name-only diff keeps it."""
    host_file = tmp_path / "credentials"
    host_file.write_text("SECRET")
    sb = LocalSubprocessSandbox()
    with sb.open_session() as session:
        assert session.run("print(1)").ok
        _plant_link(session.workdir, "main.py", str(host_file))

        session.reset_to_staged()

        assert not os.path.lexists(os.path.join(session.workdir, "main.py"))
        second = session.run("print(2)")
        assert second.ok and second.stdout.strip() == "2"
        assert host_file.read_text() == "SECRET"


def test_read_file_returns_none_for_a_link_and_never_reads_the_host_through_it(tmp_path):
    host_file = tmp_path / "credentials"
    host_file.write_text("SECRET")
    host_dir = tmp_path / "home"
    host_dir.mkdir()
    (host_dir / "secret.txt").write_text("SECRET")
    sb = LocalSubprocessSandbox()
    with sb.open_session() as session:
        session.write_file("a.txt", "alpha")
        _plant_link(session.workdir, "leak", str(host_file))
        _plant_link(session.workdir, "pkg", str(host_dir))

        assert session.read_file("leak") is None
        assert session.read_file("pkg/secret.txt") is None
        assert session.read_file("a.txt") == "alpha"
        with pytest.raises(ValueError):
            session.write_file("pkg/planted.txt", "x")
        assert not (host_dir / "planted.txt").exists()


def test_concurrent_sessions_keep_separate_state():
    """One shared sandbox, many sessions stepped from threads: each keeps its own files."""
    sb = LocalSubprocessSandbox()
    sessions = {i: sb.open_session() for i in range(8)}
    try:

        def worker(i: int):
            sessions[i].write_file("id.txt", str(i))
            sessions[i].run("print(open('id.txt').read())")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for i, session in sessions.items():
            assert session.read_file("id.txt") == str(i)
    finally:
        for session in sessions.values():
            session.close()


# Session build reuse (one compile per distinct source across a session's runs)


class _CountingSandbox(LocalSubprocessSandbox):
    """Counts compiler launches so a test fails the moment the session stops reusing its build."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compiles = 0

    def _compile(self, workdir, spec):
        self.compiles += 1
        return super()._compile(workdir, spec)


_CPP_HEADER_VAL = '#include <iostream>\n#include "h.h"\nint main(){ std::cout << val(); }'


def _header(value: int) -> str:
    return f"inline int val(){{return {value};}}"


def test_session_compiles_same_source_once_across_runs():
    """The grader's pattern: one session per submission, one run per hidden test. The source must be
    built once and rerun from the binary, not recompiled per test."""
    if not _HAS_GPP:
        return _skip("g++ not installed")
    sb = _CountingSandbox()
    with sb.open_session() as session:
        for n in range(1, 6):
            res = session.run(_CPP_DOUBLE, stdin=str(n), language="cpp")
            assert res.ok and res.stdout.strip() == str(2 * n), f"run {n}: {res}"
            assert not res.compile_failed
    assert sb.compiles == 1, f"expected one compile across 5 runs, got {sb.compiles}"


def test_session_rebuilds_when_source_changes():
    """A different source must not run the previous build's binary (the output check would read the
    stale program's answer); the same source again rebuilds too, since a session holds one build."""
    if not _HAS_GPP:
        return _skip("g++ not installed")
    triple = "#include <iostream>\nint main() { long n; std::cin >> n; std::cout << n * 3; return 0; }"
    sb = _CountingSandbox()
    with sb.open_session() as session:
        assert session.run(_CPP_DOUBLE, stdin="5", language="cpp").stdout.strip() == "10"
        assert session.run(triple, stdin="5", language="cpp").stdout.strip() == "15"
        assert session.run(_CPP_DOUBLE, stdin="5", language="cpp").stdout.strip() == "10"
    assert sb.compiles == 3


def test_session_rebuilds_when_auxiliary_header_changes():
    """``files`` are part of the build's identity: a changed header with byte-identical source rebuilds."""
    if not _HAS_GPP:
        return _skip("g++ not installed")
    sb = _CountingSandbox()
    with sb.open_session() as session:
        first = session.run(_CPP_HEADER_VAL, language="cpp", files={"h.h": _header(7)})
        assert first.ok and first.stdout.strip() == "7"
        again = session.run(_CPP_HEADER_VAL, language="cpp", files={"h.h": _header(7)})
        assert again.ok and again.stdout.strip() == "7"
        changed = session.run(_CPP_HEADER_VAL, language="cpp", files={"h.h": _header(8)})
        assert changed.ok and changed.stdout.strip() == "8", f"stale binary served: {changed}"
    assert sb.compiles == 2


def test_session_write_file_and_interpreted_run_drop_the_build():
    """A header rewritten outside the compiled run's own ``files`` — via ``write_file`` or by an
    interpreted run — must invalidate the build, or the same source keeps answering from a stale binary."""
    if not _HAS_GPP:
        return _skip("g++ not installed")
    sb = _CountingSandbox()
    with sb.open_session() as session:
        session.write_file("h.h", _header(1))
        assert session.run(_CPP_HEADER_VAL, language="cpp").stdout.strip() == "1"
        session.write_file("h.h", _header(2))
        assert session.run(_CPP_HEADER_VAL, language="cpp").stdout.strip() == "2", "write_file kept a stale build"
        rewrite = session.run(f"open('h.h', 'w').write({_header(3)!r})", language="python")
        assert rewrite.ok
        assert session.run(_CPP_HEADER_VAL, language="cpp").stdout.strip() == "3", "python run kept a stale build"
    assert sb.compiles == 3


def test_session_caches_a_rejected_compile():
    """A source the compiler rejects is compiled once; every later run of it returns the cached
    ``compile_failed`` verdict with the diagnostics, never a backend error."""
    if not _HAS_GPP:
        return _skip("g++ not installed")
    sb = _CountingSandbox()
    with sb.open_session() as session:
        results = [session.run("int main(){ this is not valid c++ }", stdin=str(n), language="cpp") for n in range(3)]
    assert sb.compiles == 1, f"a rejected source must not be recompiled per test, got {sb.compiles}"
    for res in results:
        assert res.compile_failed
        assert not res.ok
        assert res.error is None
        assert res.returncode not in (0, None)
        assert "error" in res.stderr.lower(), "diagnostics must survive the cache"


def test_one_shot_run_builds_every_time():
    """``SandboxExecutor.run`` is a throwaway session: back-to-back one-shot runs of one source each
    compile (nothing persists across them)."""
    if not _HAS_GPP:
        return _skip("g++ not installed")
    sb = _CountingSandbox()
    assert sb.run(_CPP_DOUBLE, stdin="1", language="cpp").stdout.strip() == "2"
    assert sb.run(_CPP_DOUBLE, stdin="2", language="cpp").stdout.strip() == "4"
    assert sb.compiles == 2


# RemoteSession (client-side file accumulation, no network)


def test_remote_session_resends_accumulated_files():
    sess = RecordingSandboxSession()
    sb = RemoteSandbox("http://sandbox:8080", session=sess)
    rsession = sb.open_session()
    rsession.run("print(0)")
    rsession.write_file("util.py", "X = 1")
    rsession.run("import util; print(util.X)", language="python")
    rsession.write_file("util2.py", "Y = 2")
    rsession.run("print('again')", files={"adhoc.py": "Z=3"})

    assert "files" not in sess.posts[0].payload, "a session holding no files sends none"
    assert sess.posts[1].payload["files"] == {"util.py": "X = 1"}
    assert sess.posts[2].payload["files"] == {"util.py": "X = 1", "util2.py": "Y = 2", "adhoc.py": "Z=3"}
    assert rsession.list_files() == ["util.py", "util2.py"]


def test_remote_shell_tool_sends_the_command_as_a_bash_program():
    """The service, not the registry, runs the program on this backend: the shell tool has to put the
    canonical ``bash`` on the wire, or a SandboxFusion service runs the command through its Python
    runner and every call comes back a syntax error."""
    sess = RecordingSandboxSession()
    remote_session = RemoteSandbox("http://sandbox:8080", session=sess).open_session()
    create_session_bash_tools(lambda: remote_session).get("run_bash_command").execute(command="echo hi")
    assert sess.posts[0].payload["language"] == "bash"
    assert sess.posts[0].payload["code"] == "echo hi"


# BubblewrapSandbox


def test_bubblewrap_missing_binary_raises_clearly():
    """A bogus explicit bwrap_path must fail construction with a clear error (never silently pass)."""
    raised = False
    try:
        BubblewrapSandbox(bwrap_path="/nonexistent/bwrap_xyz")
    except RuntimeError as exc:
        raised = "bwrap" in str(exc).lower()
    assert raised, "construction must fail clearly when the bwrap binary is unavailable"


def test_bubblewrap_probe_fails_fast_when_cannot_sandbox(monkeypatch, tmp_path):
    """A bwrap that cannot create its namespaces fails construction with an actionable error, never a
    per-run failure grading would read as the program's."""
    blocked = "#!/bin/sh\necho 'bwrap: No permissions to create new namespace' >&2\nexit 1\n"
    _use_stand_in_jail(monkeypatch, tmp_path, bwrap=blocked)
    with pytest.raises(RuntimeError, match="cannot create a sandbox in this environment .*namespaces"):
        BubblewrapSandbox()


def test_bubblewrap_resolve_backend_when_present():
    if _BWRAP is None:
        return _skip("bubblewrap not usable here (binary absent or namespaces blocked)")
    sb = resolve_sandbox(backend="bubblewrap")
    assert isinstance(sb, BubblewrapSandbox)


def test_bubblewrap_runs_python_and_unshares_network():
    if _BWRAP is None:
        return _skip("bubblewrap not usable here")
    ok = _BWRAP.run("print(6 * 7)", language="python")
    assert ok.ok and ok.stdout.strip() == "42"
    # Network namespace unshared: only loopback exists, so a routable connect must fail.
    netcode = (
        "import socket\n"
        "s = socket.socket()\n"
        "s.settimeout(1)\n"
        "try:\n"
        "    s.connect(('8.8.8.8', 53)); print('CONNECTED')\n"
        "except OSError as e:\n"
        "    print('BLOCKED')\n"
    )
    res = _BWRAP.run(netcode, language="python")
    assert "BLOCKED" in res.stdout, f"network should be unreachable in the jail, got {res.stdout!r}"


def test_bubblewrap_hides_host_filesystem():
    if _BWRAP is None:
        return _skip("bubblewrap not usable here")
    res = _BWRAP.run("import os; print('VISIBLE' if os.path.exists('/workspace/src') else 'HIDDEN')")
    assert "HIDDEN" in res.stdout, f"host workspace must be invisible in the jail, got {res.stdout!r}"


def test_bubblewrap_runs_bash():
    """bash is the one registered language resolved off PATH inside the jail (no interpreter
    placeholder), so it depends on the read-only binds covering the system directories."""
    if _BWRAP is None:
        return _skip("bubblewrap not usable here")
    res = _BWRAP.run("echo 21 | awk '{print $1 * 2}'", language="bash")
    assert res.ok and res.stdout.strip() == "42", f"bash in jail failed: {res}"


def test_bubblewrap_compiles_and_runs_cpp():
    if _BWRAP is None or not _HAS_GPP:
        return _skip("bubblewrap not usable here or g++ absent")
    res = _BWRAP.run(_CPP_DOUBLE, stdin="21", language="cpp")
    assert res.ok and res.stdout.strip() == "42", f"cpp in jail failed: {res}"


def test_bubblewrap_session_persists_and_isolates():
    if _BWRAP is None:
        return _skip("bubblewrap not usable here")
    with _BWRAP.open_session() as session:
        session.write_file("s.txt", "kept")
        res = session.run("print(open('s.txt').read())")
        assert res.ok and res.stdout.strip() == "kept"


def test_bubblewrap_parallel_safe():
    if _BWRAP is None:
        return _skip("bubblewrap not usable here")
    results = {}

    def worker(i: int):
        results[i] = _BWRAP.run(f"print({i} * {i})").stdout.strip()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == {i: str(i * i) for i in range(8)}, "concurrent jailed runs cross-contaminated"


_LEFT_BEHIND = "LEFT_BEHIND:"
# <linux/mount.h>
_MS_BIND = 4096
_MS_REMOUNT = 32
# A clean exit, a failing one, a signal, a timeout, a compile error, a compiled run.
_JAILED_RUNS = [
    ("print(42)", "python", 10.0),
    ("raise SystemExit(3)", "python", 10.0),
    ("import os; os.abort()", "python", 10.0),
    ("while True: pass", "python", 0.3),
    ("int main() { syntax error }", "cpp", 10.0),
    ("int main() { return 0; }", "cpp", 10.0),
]
_JAILED_HOST = f"""
from src.environments.sandbox.bubblewrap import BubblewrapSandbox
sandbox = BubblewrapSandbox()
for code, language, timeout in {_JAILED_RUNS!r} * 2:
    sandbox.run(code, language=language, timeout=timeout)
"""

# Stands in for bwrap without namespaces, as bwrap behaves on exit: its child (the jail's init) runs the
# program and reports its status, 128 + n for a signal; bwrap exits on that status, never reaping it.
# The program stays in the run's group: its own session needs the jail's PID namespace to end with a
# killed run.
_STAND_IN_BWRAP = """#!{python}
import os, sys
argv = sys.argv[sys.argv.index("--") + 1:]
argv = argv[1:] if argv[0] == "setsid" else argv
report_r, report_w = os.pipe()
if os.fork() == 0:
    program = os.fork()
    if program == 0:
        os.execvp(argv[0], argv)
    code = os.waitstatus_to_exitcode(os.waitpid(program, 0)[1])
    os.write(report_w, bytes([128 - code if code < 0 else code]))
    os._exit(0)
os.close(report_w)
status = os.read(report_r, 1)
os._exit(status[0] if status else 1)
"""
# Runs what follows its ``--`` (namespaces are the jail's, which the stand-in bwrap does without).
_STAND_IN_UNSHARE = '#!/bin/sh\nwhile [ "$1" != "--" ]; do shift; done\nshift\nexec "$@"\n'
# The jail ids the stand-in subordinate ranges grant.
_STAND_IN_JAIL_ID = 165536


def _stand_in_jail(tmp_path, bwrap: str = "") -> tuple[str, tuple[str, str]]:
    """A ``PATH`` directory of stand-in jail launchers — ``bwrap`` (``_STAND_IN_BWRAP`` unless given),
    ``unshare``, and the id-map helpers, which need only exist — and subordinate-id files granting this
    user a range at :data:`_STAND_IN_JAIL_ID`."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    scripts = {
        "bwrap": bwrap or _STAND_IN_BWRAP.format(python=sys.executable),
        "unshare": _STAND_IN_UNSHARE,
        "newuidmap": "#!/bin/sh\n",
        "newgidmap": "#!/bin/sh\n",
    }
    for name, text in scripts.items():
        (bin_dir / name).write_text(text)
        (bin_dir / name).chmod(0o755)
    subids = []
    for name in ("subuid", "subgid"):
        (tmp_path / name).write_text(f"{os.getuid()}:{_STAND_IN_JAIL_ID}:65536\n")
        subids.append(str(tmp_path / name))
    return str(bin_dir), (subids[0], subids[1])


def _use_stand_in_jail(monkeypatch, tmp_path, bwrap: str = "") -> None:
    bin_dir, subids = _stand_in_jail(tmp_path, bwrap)
    monkeypatch.setenv("PATH", bin_dir + os.pathsep + os.environ["PATH"])
    monkeypatch.setattr(_bubblewrap, "_SUBORDINATE_ID_FILES", subids)


_STAND_IN_HOST = """
import os, signal, threading
from concurrent.futures import ThreadPoolExecutor
os.environ["PATH"] = {bin_dir!r} + os.pathsep + os.environ["PATH"]
from src.environments.sandbox import bubblewrap
from src.environments.sandbox.bubblewrap import BubblewrapSandbox, _is_child_subreaper, _set_child_subreaper

bubblewrap._SUBORDINATE_ID_FILES = {subids!r}
sandbox = BubblewrapSandbox()
assert not _is_child_subreaper(), "construction left the process a subreaper"
assert sandbox.run("print(42)").stdout.strip() == "42"
assert sandbox.run("raise SystemExit(3)").returncode == 3
assert sandbox.run("import os; os.abort()").returncode == -signal.SIGABRT
assert sandbox.run("while True: pass", timeout=0.5).timed_out
with ThreadPoolExecutor(max_workers=8) as pool:
    codes = list(pool.map(lambda code: sandbox.run(f"raise SystemExit({{code}})").returncode, range(24)))
assert codes == list(range(24)), codes
# A fork while a run is in flight: the child holds no window, whatever count it copied.
in_flight = threading.Thread(target=sandbox.run, args=("import time; time.sleep(1)",))
in_flight.start()
threading.Event().wait(0.3)
child = os.fork()
if child == 0:
    os._exit(0 if sandbox.run("print(1)").ok else 1)
assert os.waitstatus_to_exitcode(os.waitpid(child, 0)[1]) == 0, "the forked child's run failed"
in_flight.join()
assert not _is_child_subreaper(), "the process stayed a subreaper after its runs"
_set_child_subreaper(1)
assert sandbox.run("print(1)").ok and _is_child_subreaper(), "a run cleared a flag the process held already"
_set_child_subreaper(0)
"""
# Ignores SIGTERM and sends it to its own group, as a pool's shutdown can; unjailed, that reaches only it.
_GROUP_SIGNAL = (
    "import os, signal\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\nos.killpg(0, signal.SIGTERM)\nprint('done')"
)


def _non_reaping_ancestor(host: str) -> str:
    """A probe standing in for a container PID 1 that never reaps: as a subreaper, it adopts whatever
    the ``host`` script's runs orphan, and whatever the host adopted and left unreaped when it exits.
    It reports those, or the host's failure."""
    return f"""
import ctypes, os, subprocess, sys
assert ctypes.CDLL(None, use_errno=True).prctl({_bubblewrap._PR_SET_CHILD_SUBREAPER}, 1, 0, 0, 0) == 0
host = subprocess.run([sys.executable, "-c", {host!r}], capture_output=True, text=True)
left = open(f"/proc/self/task/{{os.getpid()}}/children").read().split()
findings = [open(f"/proc/{{pid}}/comm").read().strip() + " " + pid for pid in left]
if host.returncode:
    findings.append("host failed: " + host.stderr.strip().splitlines()[-1])
print({_LEFT_BEHIND!r} + "|".join(findings))
"""


def test_bubblewrap_runs_leave_no_process_behind():
    """``bwrap`` exits without reaping the jail's init, so each run would leave a ``[bwrap] <defunct>`` to
    the container's PID 1 (a ``torchrun`` never reaps it) unless the backend reaps it itself.

    Needs a working jail, skipped otherwise; to run it in the image:

        docker run --rm --cap-add SYS_ADMIN --security-opt seccomp=unconfined \\
            --security-opt apparmor=unconfined -v "$(pwd)":/workspace -w /workspace halo:blackwell \\
            bash -c "mkdir -p /run/fullproc && mount -t proc proc /run/fullproc && \\
                pytest tests/cpu/environments/test_sandbox_multilang.py -k bubblewrap"
    """
    if _BWRAP is None or not _HAS_GPP:
        return _skip("bubblewrap not usable here or g++ absent")
    left = probe_findings(_non_reaping_ancestor(_JAILED_HOST), _LEFT_BEHIND)
    assert left == [], f"jailed runs left processes to the host's ancestor: {left}"


def test_bubblewrap_keeps_a_group_signal_inside_the_jail():
    """A program that ignores SIGTERM and sends it to its own group finishes, as on ``local``: the signal
    reaches neither ``bwrap`` nor the jail's init, whose death would tear the jail down mid-run."""
    if _BWRAP is None:
        return _skip("bubblewrap not usable here")
    result = _BWRAP.run(_GROUP_SIGNAL)
    assert result.ok and result.stdout.strip() == "done", result


def test_bubblewrap_bounds_a_jailed_programs_process_count(monkeypatch):
    """The jail's root is a subordinate uid, which RLIMIT_NPROC binds (uid 0 it never does), counted
    in the run's own user namespace: a fork loop is stopped at the limit."""
    if _BWRAP is None:
        return _skip("bubblewrap not usable here")
    monkeypatch.setattr(local_sandbox, "LOCAL_NPROC_LIMIT", 32)
    forks = (
        "import os, time\nn = 0\nfor _ in range(64):\n    try:\n        pid = os.fork()\n    except OSError:\n"
        "        break\n    if pid == 0:\n        time.sleep(1); os._exit(0)\n    n += 1\nprint(n)"
    )
    result = _BWRAP.run(forks, timeout=10)
    assert result.ok and int(result.stdout) < 32, result


def test_bubblewrap_jail_holds_no_capability():
    """The jail's root keeps no capability in its namespace, so it cannot remount a read-only bind
    writable and reach the host files under it (a world-writable directory on the trainer's library
    path among them)."""
    if _BWRAP is None:
        return _skip("bubblewrap not usable here")
    remount = (
        "import ctypes\nlibc = ctypes.CDLL(None, use_errno=True)\n"
        f"rc = libc.mount(None, b'/usr', None, {_MS_REMOUNT | _MS_BIND}, None)\n"
        "print(open('/proc/self/status').read().split('CapEff:')[1].split()[0], rc)"
    )
    result = _BWRAP.run(remount)
    assert result.stdout.split() == ["0000000000000000", "-1"], result


def test_bubblewrap_counts_each_runs_processes_alone(monkeypatch):
    """The limit counts in the run's own user namespace (Linux 5.14+): two concurrent runs each hold
    more than half of it, which one count shared by the jail's uid would refuse."""
    if _BWRAP is None:
        return _skip("bubblewrap not usable here")
    monkeypatch.setattr(local_sandbox, "LOCAL_NPROC_LIMIT", 32)
    hold = (
        "import os, time\nn = 0\nfor _ in range(20):\n    try:\n        pid = os.fork()\n    except OSError:\n"
        "        break\n    if pid == 0:\n        time.sleep(3); os._exit(0)\n    n += 1\nprint(n)"
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        held = [int(r.stdout) for r in pool.map(lambda _: _BWRAP.run(hold, timeout=15), range(2))]
    assert held == [20, 20], held


def test_bubblewrap_program_owns_its_workdir_at_a_fixed_path():
    """The program works in a directory it owns, at a fixed path: it can extend a file the host wrote,
    what it writes is the jail ids' on the host, and no host path reaches a traceback."""
    if _BWRAP is None:
        return _skip("bubblewrap not usable here")
    with _BWRAP.open_session() as session:
        session.write_file("data.txt", "a")
        result = session.run(
            "import os\nopen('data.txt', 'a').write('b')\nopen('new.txt', 'w').close()\nprint(os.getcwd())"
        )
        assert result.ok and result.stdout.strip() == "/sandbox", result
        assert session.read_file("data.txt") == "ab"
        assert os.stat(os.path.join(session.workdir, "new.txt")).st_uid == _BWRAP.program_owner[0]
        failed = session.run("raise ValueError('x')")
    assert 'File "/sandbox/main.py"' in failed.stderr and session.workdir not in failed.stderr, failed


def test_bubblewrap_reaps_the_init_bwrap_leaves_and_reads_its_signal_status(tmp_path):
    """Against a stand-in ``bwrap`` (no namespaces needed): a process is a subreaper only while a run is
    in flight, sequential, concurrent and forked runs leave no init behind (a timed-out one included),
    ``bwrap``'s ``128 + n`` reads as the signal it stands for, and a flag the process held already stays."""
    bin_dir, subids = _stand_in_jail(tmp_path)
    host = _STAND_IN_HOST.format(bin_dir=bin_dir, subids=subids)
    left = probe_findings(_non_reaping_ancestor(host), _LEFT_BEHIND)
    assert left == [], f"runs left processes to the host's ancestor, or the host failed: {left}"


def test_a_group_that_outlives_its_wait_is_reaped_at_a_later_run(monkeypatch):
    """A jail's init exits only once every task in it has, which a fork bomb's teardown can drag past the
    run's bounded wait; the run then gives the group up to a later run's reap rather than leak it."""
    monkeypatch.setattr(_bubblewrap, "_JAIL_TEARDOWN_SECONDS", 0.05)
    reaper = _bubblewrap._JailReaper()
    slow = os.posix_spawnp("sleep", ["sleep", "0.5"], os.environ, setsid=True)
    reaper.reap(slow)
    assert os.path.exists(f"/proc/{slow}"), "the wait was not bounded"
    time.sleep(0.6)
    fast = os.posix_spawnp("true", ["true"], os.environ, setsid=True)
    reaper.reap(fast)
    assert [pid for pid in (slow, fast) if os.path.exists(f"/proc/{pid}")] == []


def test_bubblewrap_probe_names_a_bind_path_its_root_cannot_reach(monkeypatch, tmp_path):
    """The jail's root is not this process's uid, so a working directory under a private directory is
    out of its reach: the probe names the search permission it needs, not namespace rights."""
    unreachable = '#!/bin/sh\necho "bwrap: Can\'t find source path /x/halo_sandbox_1: Permission denied" >&2\nexit 1\n'
    _use_stand_in_jail(monkeypatch, tmp_path, bwrap=unreachable)
    with pytest.raises(RuntimeError, match=r"searchable by other users.*point TMPDIR"):
        BubblewrapSandbox()


def test_bubblewrap_probe_names_the_clean_proc_remedy(monkeypatch, tmp_path):
    """While parts of the container's ``/proc`` are masked (Docker's masks, the NVIDIA runtime's in a GPU
    container) the kernel refuses the jail a fresh one; the probe names the clean mount beside it, not
    namespace rights the container already has, and leaves the process no subreaper."""
    refused = '#!/bin/sh\necho "bwrap: Can\'t mount proc on /newroot/proc: Operation not permitted" >&2\nexit 1\n'
    _use_stand_in_jail(monkeypatch, tmp_path, bwrap=refused)
    with pytest.raises(RuntimeError, match="mount -t proc proc /run/fullproc"):
        BubblewrapSandbox()
    assert not _bubblewrap._is_child_subreaper()


@pytest.mark.parametrize("missing", ["newuidmap", "subuid"])
def test_bubblewrap_refuses_a_host_that_cannot_map_the_jail_root(monkeypatch, tmp_path, missing):
    """Without the id-map helpers or a subordinate range the jail's root would be this process's uid,
    which RLIMIT_NPROC never binds: construction refuses, naming what is missing."""
    _use_stand_in_jail(monkeypatch, tmp_path)
    if missing == "newuidmap":
        (tmp_path / "bin" / "newuidmap").unlink()
        monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    else:
        (tmp_path / "subuid").write_text("")
    with pytest.raises(RuntimeError, match=r"uidmap package .* /etc/subuid"):
        BubblewrapSandbox()


def test_bubblewrap_hands_the_working_directory_and_host_writes_to_the_jail_ids(monkeypatch, tmp_path):
    """The jail's root is a subordinate uid, so the working directory and every entry the host writes
    there are made its own; the program can then change what the host staged (an edited source)."""
    if os.geteuid() != 0:
        return _skip("handing entries to another uid needs root")
    _use_stand_in_jail(monkeypatch, tmp_path)
    with BubblewrapSandbox().open_session() as session:
        session.write_file("pkg/data.txt", "a")
        assert session.run("print(1)").ok
        owners = {
            name: os.stat(os.path.join(session.workdir, name)).st_uid
            for name in (".", "pkg", "pkg/data.txt", "main.py")
        }
    assert owners == dict.fromkeys(owners, _STAND_IN_JAIL_ID)


def test_bubblewrap_hands_a_recreated_working_directory_to_the_jail_ids(monkeypatch, tmp_path):
    """A working directory the program removed is recreated for the next run, the jail's again."""
    if os.geteuid() != 0:
        return _skip("handing entries to another uid needs root")
    _use_stand_in_jail(monkeypatch, tmp_path)
    with BubblewrapSandbox().open_session() as session:
        shutil.rmtree(session.workdir)
        assert session.run("print(1)").ok
        assert os.stat(session.workdir).st_uid == _STAND_IN_JAIL_ID


@pytest.mark.parametrize("allow_network", [False, True])
def test_bubblewrap_unshares_the_network_unless_the_executor_allows_it(monkeypatch, tmp_path, allow_network):
    """Every jailed run, a session's included, gets a network namespace of its own unless the executor
    was built with ``allow_network``."""
    if os.geteuid() != 0:
        return _skip("handing entries to another uid needs root")
    log = tmp_path / "unshare_net.log"
    recording = _STAND_IN_BWRAP.format(python=sys.executable).replace(
        "import os, sys\n",
        f"import os, sys\nopen({str(log)!r}, 'a').write(str('--unshare-net' in sys.argv) + '\\n')\n",
    )
    _use_stand_in_jail(monkeypatch, tmp_path, bwrap=recording)
    with BubblewrapSandbox(allow_network=allow_network).open_session() as session:
        assert session.run("print(1)").ok
    assert set(log.read_text().split()) == {str(not allow_network)}


def test_session_reset_to_staged_drops_run_output_and_keeps_the_build():
    """``reset_to_staged`` removes what a run produced (files, directories) and leaves the staged source
    and the built binary, so the next run of the same program starts clean without recompiling."""
    if not _HAS_GPP:
        return _skip("g++ not installed")

    class _Counting(LocalSubprocessSandbox):
        compiles = 0

        def _compile(self, *args, **kwargs):
            type(self).compiles += 1
            return super()._compile(*args, **kwargs)

    src = (
        '#include <cstdio>\n#include <sys/stat.h>\nint main(){ FILE* f = fopen("marker", "r"); '
        'puts(f ? "seen" : "fresh"); if (f) fclose(f); fopen("marker", "w"); mkdir("out", 0700); return 0; }'
    )
    with _Counting().open_session() as session:
        assert session.run(src, language="cpp").stdout.strip() == "fresh"
        assert "marker" in session.list_files() and os.path.isdir(os.path.join(session.workdir, "out"))
        session.reset_to_staged()
        assert "marker" not in session.list_files() and not os.path.exists(os.path.join(session.workdir, "out"))
        assert session.read_file("main.cpp") == src
        assert session.run(src, language="cpp").stdout.strip() == "fresh"
    assert _Counting.compiles == 1, "the reset keeps the build slot: no recompile for the same source"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
