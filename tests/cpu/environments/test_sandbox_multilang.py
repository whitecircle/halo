#!/usr/bin/env python
"""
Tests for multi-language sandbox execution, persistent sessions, and the bubblewrap backend.

Covers the behaviour layered over the python-only backend (see test_sandbox.py for the
python/remote/grading basics):

- Language registry: name/alias resolution, the compiled-vs-interpreted distinction.
- C/C++ compile-and-run: stdout, stdin plumbing, a compile error reported as a *program* fault
  (returncode set, ``error`` unset) so grading buckets it as a failed solution, a runtime timeout,
  and the memory cap on a compiled binary.
- Missing-compiler handling (a backend error) via a sandbox pointed at a bogus toolchain.
- Sessions: state persists across runs (write a file, read it next run; cross-language sharing),
  two sessions are isolated, and concurrent sessions on one shared instance don't cross-contaminate.
- Session build reuse: a compiled source is built once across runs with different stdin (the grader's
  one-session-per-submission pattern), rebuilt when the source, an auxiliary header, a session-written
  file or an interpreted run changes the working dir, and a rejected compile is cached as
  ``compile_failed`` instead of recompiling per test.
- RemoteSession: client-side file accumulation resent on every request (no network — fake session).
- BubblewrapSandbox: construction guard when bwrap is absent, and — when bwrap IS present —
  network is unshared and the host filesystem is hidden while the working dir stays writable.

Run: python tests/cpu/environments/test_sandbox_multilang.py
"""

import os
import shutil
import threading

import pytest

from src.environments.sandbox.base import LANGUAGES, resolve_language, supported_languages
from src.environments.sandbox.bubblewrap import BubblewrapSandbox
from src.environments.sandbox.local import LocalSubprocessSandbox
from src.environments.sandbox.remote import RemoteSandbox
from src.environments.sandbox.resolve import resolve_sandbox

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
    import unittest

    raise unittest.SkipTest(reason)


# Language registry


def test_language_registry_resolves_aliases():
    assert resolve_language("python").name == "python"
    assert resolve_language("py").name == "python"
    assert resolve_language("c++").name == "cpp"
    assert resolve_language("CXX").name == "cpp"  # case-insensitive
    assert resolve_language("c").name == "c"
    assert resolve_language("ruby") is None
    assert set(supported_languages()) == set(LANGUAGES.keys()) == {"python", "cpp", "c"}


def test_language_compiled_flag():
    assert resolve_language("python").is_compiled is False
    assert resolve_language("cpp").is_compiled is True
    assert resolve_language("c").is_compiled is True


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
    from src.environments.sandbox import base as _base

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

    def _compile(self, workdir, spec, *, allow_network):
        self.compiles += 1
        return super()._compile(workdir, spec, allow_network=allow_network)


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


class _CapturingSession:
    """Fake requests.Session that records each request's JSON and returns a canned success."""

    def __init__(self):
        self.payloads = []

    def post(self, url, json=None, timeout=None):
        self.payloads.append(json)

        class _Resp:
            def raise_for_status(self_inner):
                pass

            def json(self_inner):
                return {"status": "Success", "run_result": {"status": "Finished", "stdout": "ok", "return_code": 0}}

        return _Resp()


def test_remote_session_resends_accumulated_files():
    sess = _CapturingSession()
    sb = RemoteSandbox("http://sandbox:8080", session=sess)
    rsession = sb.open_session()
    rsession.write_file("util.py", "X = 1")
    rsession.run("import util; print(util.X)", language="python")
    rsession.write_file("util2.py", "Y = 2")
    rsession.run("print('again')", files={"adhoc.py": "Z=3"})

    assert sess.payloads[0]["files"] == {"util.py": "X = 1"}
    assert sess.payloads[1]["files"] == {"util.py": "X = 1", "util2.py": "Y = 2", "adhoc.py": "Z=3"}
    assert rsession.list_files() == ["util.py", "util2.py"]


# BubblewrapSandbox


def test_bubblewrap_missing_binary_raises_clearly():
    """A bogus explicit bwrap_path must fail construction with a clear error (never silently pass)."""
    raised = False
    try:
        BubblewrapSandbox(bwrap_path="/nonexistent/bwrap_xyz")
    except RuntimeError as exc:
        raised = "bwrap" in str(exc).lower()
    assert raised, "construction must fail clearly when the bwrap binary is unavailable"


def test_bubblewrap_probe_fails_fast_when_cannot_sandbox(tmp_path=None):
    """If bwrap is present but can't create a namespace, construction must raise an actionable error.

    Simulated with a fake 'bwrap' that is executable but exits non-zero (like a blocked user
    namespace), so the construction probe detects it deterministically without needing a real
    namespace failure.
    """
    import os
    import tempfile

    d = tempfile.mkdtemp()
    fake = os.path.join(d, "bwrap")
    with open(fake, "w") as fh:
        fh.write("#!/bin/sh\necho 'bwrap: No permissions to create new namespace' >&2\nexit 1\n")
    os.chmod(fake, 0o755)
    try:
        raised_msg = ""
        try:
            BubblewrapSandbox(bwrap_path=fake)
        except RuntimeError as exc:
            raised_msg = str(exc).lower()
        assert "cannot create a sandbox" in raised_msg, "probe must fail fast with an actionable message"
        assert "namespace" in raised_msg
    finally:
        shutil.rmtree(d, ignore_errors=True)


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
