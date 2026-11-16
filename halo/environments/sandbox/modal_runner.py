"""The program-side half of the Modal backend, run inside the sandbox as ``python3 -c`` with one JSON request on
stdin and one JSON reply on stdout. Stdlib only. Every limit and the program environment arrive in the request, so
the verdicts stay on the client, read off the local backend's helpers."""

import contextlib
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time


def _apply_limits(limits):
    def apply():
        os.setsid()
        for name, value in limits.items():
            kind = getattr(resource, name)
            if name == "RLIMIT_STACK":
                hard = resource.getrlimit(kind)[1]
                value = value if hard == resource.RLIM_INFINITY else min(value, hard)
            resource.setrlimit(kind, (value, value))

    return apply


def _read(capture):
    capture.seek(0)
    return capture.read().decode("utf-8", errors="replace")


def _execute(argv, req, *, stdin, timeout, limits):
    """``{stdout, stderr, returncode, timed_out}``, the clock started at exec; a missing binary is returncode 127."""
    env = {"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"), **req["env"]}
    with tempfile.TemporaryFile() as feed, tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        feed.write(stdin.encode("utf-8"))
        feed.seek(0)
        try:
            proc = subprocess.Popen(
                argv, stdin=feed, stdout=out, stderr=err, cwd=req["workdir"], env=env, preexec_fn=_apply_limits(limits)
            )
        except FileNotFoundError:
            return {"stdout": "", "stderr": f"{argv[0]}: not found", "returncode": 127, "timed_out": False}
        deadline = time.monotonic() + timeout
        timed_out = False
        try:
            proc.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out = True
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        return {"stdout": _read(out), "stderr": _read(err), "returncode": proc.returncode, "timed_out": timed_out}


def _write(workdir, files):
    for name, content in files.items():
        dest = os.path.join(workdir, name)
        os.makedirs(os.path.dirname(dest) or workdir, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as fh:
            fh.write(content)


def _reset(workdir, keep):
    for name in set(os.listdir(workdir)) - set(keep):
        path = os.path.join(workdir, name)
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            with contextlib.suppress(OSError):
                os.remove(path)


def _run(req, fresh):
    """Builds when asked or when the directory is new (a replaced sandbox lost the build)."""
    workdir = req["workdir"]
    if req.get("reset_to") is not None:
        _reset(workdir, req["reset_to"])
    _write(workdir, {**req["files"], req["source_name"]: req["code"]})
    reply = {}
    if req.get("compile_argv") and (req["build"] or fresh):
        reply["compile"] = _execute(
            req["compile_argv"], req, stdin="", timeout=req["compile_timeout"], limits=req["compile_limits"]
        )
        if reply["compile"]["timed_out"] or reply["compile"]["returncode"] != 0:
            return reply
    reply["entries"] = sorted(os.listdir(workdir))
    reply["run"] = _execute(req["run_argv"], req, stdin=req["stdin"], timeout=req["timeout"], limits=req["limits"])
    return reply


def _list(workdir):
    paths = []
    for root, _dirs, names in os.walk(workdir):
        paths.extend(os.path.relpath(os.path.join(root, name), workdir) for name in names)
    return {"files": sorted(paths)}


def _read_file(workdir, path):
    full = os.path.join(workdir, path)
    if not os.path.isfile(full):
        return {"content": None}
    with open(full, encoding="utf-8", errors="replace") as fh:
        return {"content": fh.read()}


def main():
    req = json.load(sys.stdin)
    for stale in req.get("remove", ()):
        shutil.rmtree(stale, ignore_errors=True)
    workdir = req["workdir"]
    fresh = not os.path.isdir(workdir)
    os.makedirs(workdir, exist_ok=True)
    op = req["op"]
    if op == "run":
        reply = _run(req, fresh)
    elif op == "write":
        _write(workdir, req["files"])
        reply = {}
    elif op == "read":
        reply = _read_file(workdir, req["path"])
    elif op == "list":
        reply = _list(workdir)
    else:
        reply = {"error": f"unknown op {op!r}"}
    sys.stdout.write(json.dumps(reply))


if __name__ == "__main__":
    main()
