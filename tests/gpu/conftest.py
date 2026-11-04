"""pytest launcher for the manifest-driven GPU suite: one torchrun launch per ``(script, args)`` row.

ERROR is a non-zero exit with no result line. FAIL is a ``__HALO_TEST_RESULT__`` failure, or a non-zero exit whose line
says ``pass``: only rank 0 emits it, so a non-zero rank failed. SKIP is too few GPUs or an OOM on the 8-GPU ``full``
tier; zero GPUs is a usage error. Other ``test_*.py`` here are torchrun scripts pytest must never import: one acting at
import takes the whole session down.
"""

import contextlib
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from functools import cache
from pathlib import Path

import pytest

from tests.common.ports import free_port
from tests.common.reporting import parse_result
from tests.gpu.manifest import (
    GPU_DIR,
    LAUNCHER_ENTRYPOINTS,
    MANIFEST,
    TestSpec,
    script_path,
    stale_entries,
    unregistered_scripts,
)

_TRANSIENT = ("OSError", "Timeout", "HTTPError", "Connection", "NCCL", "ECONNRESET")
_OOM = ("OutOfMemoryError", "CUDA out of memory", "CUDA error: out of memory")

# Identifies surviving workers from /proc without matching a script name a co-tenant might also run.
_LAUNCH_ID_VAR = "HALO_TEST_LAUNCH_ID"
_NPROC_VAR = "HALO_TEST_NPROC"
# Seconds for torchrun to reap its own workers after SIGTERM before SIGKILL.
_SHUTDOWN_GRACE = 10
# Orphan sweep: kill deadline, consecutive clean surveys required, /proc passes per survey.
_ORPHAN_SWEEP_TIMEOUT = 30
_CLEAN_SCANS_REQUIRED = 2
_PASSES_PER_SURVEY = 2
# A worker still holding the stdout pipe would block the drain forever.
_OUTPUT_DRAIN_TIMEOUT = 30

# Marker -> (env var with the base URL, default URL, start command). Raw os.environ: importing halo pulls
# torch+transformers (~4s, ~500MB) and a root logging handler into the launcher.
_SERVER_TIERS = {
    "vllm_server": (
        "VLLM_SERVER_URL",
        "http://localhost:8000",
        "make build-vllm (or set VLLM_IMAGE), then "
        "HALO_SCRATCH=<large volume> VLLM_CUDA_DEVICES=7 docker compose -f docker-compose.vllm.yml up -d vllm-server",
    ),
    "sglang_server": (
        "SGLANG_SERVER_URL",
        "http://localhost:30000",
        "make build-sglang (or set SGLANG_IMAGE), then "
        "SGLANG_CUDA_DEVICES=7 SGLANG_MODEL=<hub id> SGLANG_MODEL_DIR=<any dir> "
        "docker compose -f docker-compose.sglang.yml up -d",
    ),
}
# One probe per URL per session: N nodes asking one server is N chances of a different answer mid-collection.
_SERVER_PROBE_TIMEOUT_S = 5
# Set by the make server tiers: an unreachable server is then a usage error, like the zero-GPU refusal below.
_REQUIRE_SERVER_VAR = "HALO_TEST_REQUIRE_SERVER"

# AF_UNIX paths cap at 108 bytes and `datasets` binds a SyncManager socket under TMPDIR; pytest's nodeid-derived
# tmp_path overflows it and the test dies after its assertions pass. Budget covers `pymp-XXXXXXXX/listener-XXXXXXXX`.
_AF_UNIX_MAX = 108
_SOCKET_SUFFIX_BUDGET = 40

_ENTRYPOINT_PATHS = frozenset(script_path(rel) for rel in LAUNCHER_ENTRYPOINTS)


def pytest_ignore_collect(collection_path: Path, config) -> bool | None:
    if collection_path.is_dir() and collection_path != GPU_DIR:
        return True
    if collection_path.suffix == ".py" and collection_path not in _ENTRYPOINT_PATHS:
        return True
    return None


def _gpu_count() -> int:
    """Visible GPUs, or 0 with no driver. A hanging or erroring ``nvidia-smi`` raises: it signals a wedged GPU, which a
    killed hang leaves behind, and answering 0 would skip the whole tier green."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return len([d for d in visible.split(",") if d.strip() != ""])
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=15)
    except FileNotFoundError:
        return 0
    except subprocess.TimeoutExpired as e:
        raise pytest.UsageError(
            "nvidia-smi -L did not answer within 15s — the driver is wedged (a stuck process still "
            "holding a GPU?). Refusing to run the GPU tier, which would otherwise report every node "
            "as skipped."
        ) from e
    if out.returncode != 0:
        raise pytest.UsageError(f"nvidia-smi -L failed ({out.returncode}): {out.stderr.strip() or out.stdout.strip()}")
    return len([ln for ln in out.stdout.splitlines() if ln.startswith("GPU ")])


def _launch_survivors(launch_id: str) -> list[int]:
    """PIDs still carrying ``launch_id`` in their environment. torchrun gives each worker its own session, so
    ``killpg`` on the agent misses them. Passes are unioned: one ``/proc`` readdir drops a live entry ~15% of the time
    under churn, and under-reporting would let a wedged worker keep its GPU."""
    marker = f"{_LAUNCH_ID_VAR}={launch_id}".encode()
    survivors: set[int] = set()
    for _ in range(_PASSES_PER_SURVEY):
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                if marker in (entry / "environ").read_bytes():
                    survivors.add(int(entry.name))
            except OSError:
                continue  # Exited between iterdir and read, or not ours to read.
    return sorted(survivors)


def _kill_launch(proc: subprocess.Popen, launch_id: str) -> list[int]:
    """Kill the agent and every worker it left; return what refused to die. A wedged NCCL rank never exits on its own
    and holds its GPU and this launch's stdout pipe open, stalling the next node and the output drain."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(os.getpgid(proc.pid), sig)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=_SHUTDOWN_GRACE)
            break
    # A worker can exit on its own between surveys, so one empty survey does not mean swept.
    deadline = time.time() + _ORPHAN_SWEEP_TIMEOUT
    clean = 0
    while clean < _CLEAN_SCANS_REQUIRED and time.time() < deadline:
        survivors = _launch_survivors(launch_id)
        if survivors:
            clean = 0
            for pid in survivors:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.kill(pid, signal.SIGKILL)
        else:
            clean += 1
        time.sleep(0.5)
    return _launch_survivors(launch_id)


def _socket_safe_tmpdir(tmp_path: Path) -> Path:
    """``tmp_path``, or a short sibling when it leaves no room for an AF_UNIX socket; a usage error when even that
    leaves none (``datasets`` would fail every map with a bare ``EOFError``). Stays on pytest's basetemp volume,
    per-run."""
    if len(str(tmp_path)) + _SOCKET_SUFFIX_BUDGET <= _AF_UNIX_MAX:
        return tmp_path
    # Still inside basetemp, so the retention policy reclaims it.
    short = Path(tempfile.mkdtemp(prefix="h", dir=tmp_path.parent))
    if len(str(short)) + _SOCKET_SUFFIX_BUDGET > _AF_UNIX_MAX:
        short.rmdir()
        raise pytest.UsageError(
            f"pytest's basetemp is too deep for the AF_UNIX sockets the GPU tests bind: {short} leaves "
            f"{_AF_UNIX_MAX - len(str(short))} of the {_SOCKET_SUFFIX_BUDGET} bytes they need. Point TMPDIR "
            f"(or --basetemp) at a shorter path (agent-docs/reference/troubleshooting.md, AF_UNIX path too long)."
        )
    return short


class GPUCase:
    def __init__(self, rel: str, spec: TestSpec, args: str):
        self.rel = rel
        self.spec = spec
        self.args = args

    def _launch_once(self, tmp_path: Path):
        launch_id = uuid.uuid4().hex
        # TORCHELASTIC_ERROR_FILE stays unset: on one shared path the agent's collateral SIGTERMs overwrite the real
        # cause.
        env = {
            **os.environ,
            _LAUNCH_ID_VAR: launch_id,
            # A crash can't poison the next node, and ranks never share a tokenizer-cache dir (token-ID collisions).
            "TMPDIR": str(_socket_safe_tmpdir(tmp_path)),
        }
        script = [str(script_path(self.rel)), *(self.args.split() if self.args else [])]
        if self.spec.launcher == "python":
            # The script's own launches inherit the launch id, so the timeout sweep reaches their ranks.
            env[_NPROC_VAR] = str(self.spec.nproc)
            cmd = [sys.executable, *script]
        else:
            port = free_port()
            env["MASTER_PORT"] = str(port)
            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nproc_per_node={self.spec.nproc}",
                f"--master_port={port}",
                *script,
            ]
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,  # killpg on timeout.
        )
        try:
            stdout, _ = proc.communicate(timeout=self.spec.timeout)
            return proc.returncode, stdout, []
        except subprocess.TimeoutExpired:
            leaked = _kill_launch(proc, launch_id)
            # Re-``communicate``: the first call discards its buffer on the raise, so a bare read() returns only the
            # tail. Bounded, since an unkilled worker holds the write end open indefinitely.
            try:
                stdout, _ = proc.communicate(timeout=_OUTPUT_DRAIN_TIMEOUT)
            except subprocess.TimeoutExpired:
                stdout = ""
            return None, stdout, leaked

    def run(self, tmp_path: Path):
        attempts = 3 if self.spec.flaky else 1
        last = None
        for attempt in range(attempts):
            code, stdout, leaked = self._launch_once(tmp_path)
            print(stdout)

            if code is None:
                last = f"TIMEOUT after {self.spec.timeout}s (agent and workers killed)"
                if leaked:
                    last += (
                        f"; {len(leaked)} worker(s) survived SIGKILL and still hold GPUs: {leaked} "
                        "— later nodes in this run are unreliable"
                    )
                if not (self.spec.flaky and attempt < attempts - 1):
                    pytest.fail(f"ERROR: {self.rel} [{self.args}] — {last}", pytrace=False)
                continue

            matched_oom = next((tok for tok in _OOM if tok in stdout), None)
            if matched_oom is not None:
                if "full" in self.spec.markers and self.spec.nproc >= 8:
                    pytest.skip(f"OOM on 8-GPU full tier: {self.rel}")
                # Quote the match and the tail: this keys on a substring, so a script printing an OOM string is
                # labelled one.
                pytest.fail(
                    f"ERROR: {self.rel} — OOM at nproc={self.spec.nproc} (this config must fit). "
                    f"Matched {matched_oom!r}; last output:\n" + "\n".join(stdout.splitlines()[-15:]),
                    pytrace=False,
                )

            result = parse_result(stdout)
            # A script declining to run prints ``SKIP:``; exit 0 with no result line still passes, since exit-code-only
            # scripts have no other channel. Substring: ``log()`` stamps ``[Rank N] `` and torchrun interleaves
            # mid-line.
            if code == 0 and result is None:
                for line in stdout.splitlines():
                    marker = line.find("SKIP:")
                    if marker != -1:
                        pytest.skip(f"{self.rel}: {line[marker + len('SKIP:') :].strip()}")
            if code == 0 and (result is None or result.get("status") == "pass"):
                return

            transient = any(tok in stdout for tok in _TRANSIENT)
            if self.spec.flaky and transient and attempt < attempts - 1:
                last = "transient error, retrying"
                time.sleep(5)
                continue

            if result is not None and result.get("status") in ("fail", "error"):
                failed = [k for k, v in result.get("checks", {}).items() if not v]
                err = result.get("error")
                pytest.fail(
                    f"FAIL: {self.rel} [{self.args}] status={result['status']} "
                    f"failed_checks={failed}" + (f" error={err}" if err else ""),
                    pytrace=False,
                )
            if result is not None:
                # Only rank 0 emits the result line, so pass plus a non-zero exit means a non-zero rank failed.
                pytest.fail(
                    f"FAIL: {self.rel} [{self.args}] — rank 0 reported "
                    f"status={result['status']} but the launch exited {code}: a non-zero rank failed "
                    "(see the '[Rank N] FAILED CHECKS' / FATAL lines above)",
                    pytrace=False,
                )
            pytest.fail(
                f"ERROR: {self.rel} [{self.args}] exited {code} with no result line (infra / hang / import crash)",
                pytrace=False,
            )

        pytest.fail(f"ERROR: {self.rel} [{self.args}] — exhausted retries ({last})", pytrace=False)

    def __repr__(self):
        return f"{self.rel}[{self.args}]" if self.args else self.rel


def pytest_generate_tests(metafunc):
    if "gpu_case" not in metafunc.fixturenames:
        return
    params = []
    for rel, script_spec in MANIFEST.items():
        for args, spec in script_spec.launches():
            node_id = rel if not args else f"{rel}[{args}]"
            marks = [getattr(pytest.mark, m) for m in spec.markers]
            params.append(pytest.param(GPUCase(rel, spec, args), marks=marks, id=node_id))
    metafunc.parametrize("gpu_case", params)


@cache
def _server_is_up(url: str) -> bool:
    """Any failure (refused, DNS, timeout, 5xx) reads as down."""
    if "://" not in url:
        url = f"http://{url}"  # A scheme-less URL raises ValueError below, which would read as down.
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=_SERVER_PROBE_TIMEOUT_S) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _absent_server(markers: tuple) -> str | None:
    """Skip reason when this node's engine is unreachable, else ``None``. With ``HALO_TEST_REQUIRE_SERVER`` set (the
    make server tiers) the caller turns the reason into a ``UsageError``."""
    for marker, (env_var, default_url, start_cmd) in _SERVER_TIERS.items():
        if marker not in markers:
            continue
        url = os.environ.get(env_var) or default_url
        if not _server_is_up(url):
            return f"no {marker.split('_')[0]} server at {url} (start it: {start_cmd})"
    return None


# trylast: after ``-m``/``-k`` deselection, so a CPU-tier run from the repo root is not refused for having no GPU.
@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    drift_un, drift_stale = unregistered_scripts(), stale_entries()
    if drift_un or drift_stale:
        raise pytest.UsageError(
            "GPU test manifest is out of sync with tests/gpu/. "
            f"Unregistered scripts (add a TestSpec): {drift_un}. "
            f"Stale entries (script deleted): {drift_stale}."
        )
    cases = []
    for item in items:
        case = getattr(item, "callspec", None)
        case = case.params.get("gpu_case") if case else None
        if isinstance(case, GPUCase):
            cases.append((item, case))
    if not cases:
        return
    have = _gpu_count()
    # Zero GPUs would skip every node and exit 0, reporting a whole tier green without running it.
    if have == 0:
        raise pytest.UsageError(
            f"{len(cases)} GPU test(s) selected but no GPU is visible — check `nvidia-smi`, that the "
            "container was started with `--gpus all` (see the Makefile's DOCKER_RUN), and that "
            "CUDA_VISIBLE_DEVICES is not empty. Refusing to report a skipped tier as success."
        )
    for item, case in cases:
        if have < case.spec.nproc:
            item.add_marker(pytest.mark.skip(reason=f"needs {case.spec.nproc} GPUs, {have} visible"))
            continue
        absent = _absent_server(case.spec.markers)
        if absent:
            required = os.environ.get(_REQUIRE_SERVER_VAR, "")
            if required and f"{required}_server" in case.spec.markers:
                raise pytest.UsageError(
                    f"{absent} — this run explicitly selected the {required} server tier "
                    f"({_REQUIRE_SERVER_VAR}={required}), so skipping it would report an untested "
                    "tier as success."
                )
            item.add_marker(pytest.mark.skip(reason=absent))
