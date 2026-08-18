"""pytest launcher for the manifest-driven GPU test suite.

GPU tests are torchrun scripts, so this conftest does not import them. It reads
:data:`tests.gpu.manifest.MANIFEST`, generates one pytest node per ``(script, args)``,
and each node shells out::

    python -m torch.distributed.run --nproc_per_node=<nproc> --master_port=<free> <script> <args>

with the spec's ``timeout``. On expiry the agent and its workers are killed, since torchrun starts
every worker in its own session and killing the agent's process group alone leaves them running.
It then classifies:

* ERROR  — non-zero exit with no structured result line: infra, hang or import crash.
* FAIL   — a ``__HALO_TEST_RESULT__`` line with ``status="fail"``/``"error"`` (an assertion
           failure), or a non-zero exit whose result line says ``pass``; only rank 0 emits it, so
           that combination means a non-zero rank failed.
* SKIP   — fewer GPUs than ``nproc`` available, or an OOM on the 8-GPU ``full`` tier (an OOM on a
           2-GPU ``core`` smoke is an ERROR, since that config must fit). Zero GPUs is never a
           skip: it would report the whole tier green, so it is a usage error.
* PASS   — exit 0 (and ``status="pass"`` when the script uses ``gpu_test_main``).

Preferred over ``torchrun -m pytest`` (N-rank double collection, racing JUnit, duplicated
fixtures): one OS process owns the rank group, one node owns the verdict.
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
from tests.gpu.manifest import MANIFEST, TestSpec, script_path, stale_entries, unregistered_scripts

_TRANSIENT = ("OSError", "Timeout", "HTTPError", "Connection", "NCCL", "ECONNRESET")
_OOM = ("OutOfMemoryError", "CUDA out of memory", "CUDA error: out of memory")

# Stamped into every launch's env so surviving workers are identifiable from /proc without matching
# on a script name a co-tenant might also be running.
_LAUNCH_ID_VAR = "HALO_TEST_LAUNCH_ID"
# Seconds to let torchrun reap its own workers after SIGTERM before escalating to SIGKILL.
_SHUTDOWN_GRACE = 10
# Orphan sweep: kill deadline, consecutive clean surveys required, /proc passes per survey.
_ORPHAN_SWEEP_TIMEOUT = 30
_CLEAN_SCANS_REQUIRED = 2
_PASSES_PER_SURVEY = 2
# A worker that still holds the stdout pipe would block the drain forever; bound it.
_OUTPUT_DRAIN_TIMEOUT = 30

# Marker → (env var holding the base URL, default URL, compose command). Raw os.environ rather than
# src.env: importing src pulls torch+transformers (~4s, ~500MB) and a root logging handler into the
# launcher process.
_SERVER_TIERS = {
    "vllm_server": (
        "VLLM_SERVER_URL",
        "http://localhost:8000",
        "VLLM_CUDA_DEVICES=7 docker compose -f docker-compose.vllm.yml up -d vllm-server",
    ),
    "sglang_server": (
        "SGLANG_SERVER_URL",
        "http://localhost:30000",
        "SGLANG_CUDA_DEVICES=7 SGLANG_MODEL=<hub id> docker compose -f docker-compose.sglang.yml up -d",
    ),
}
# One probe per URL per session: N nodes asking the same server is N chances to be told something
# different mid-collection.
_SERVER_PROBE_TIMEOUT_S = 5
# Set by the make server tiers: the run asked for this engine explicitly, so an unreachable server is
# a usage error rather than a green tier of skips (mirrors the zero-GPU refusal below).
_REQUIRE_SERVER_VAR = "HALO_TEST_REQUIRE_SERVER"

# AF_UNIX paths cap at 108 bytes and `datasets` binds a SyncManager socket under TMPDIR while
# tokenizing; pytest's nodeid-derived tmp_path overflows it and the test dies after its assertions
# pass. Budget covers `pymp-XXXXXXXX/listener-XXXXXXXX`.
_AF_UNIX_MAX = 108
_SOCKET_SUFFIX_BUDGET = 40


def _gpu_count() -> int:
    """Number of visible GPUs (honors CUDA_VISIBLE_DEVICES), or 0 when there is no driver.

    A broken driver is not 0: a hanging or erroring ``nvidia-smi`` is the signature of a wedged GPU,
    which is what a killed hang leaves behind, and answering 0 there would skip the whole tier green.
    Raise instead.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return len([d for d in visible.split(",") if d.strip() != ""])
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=15)
    except FileNotFoundError:
        return 0  # no driver stack at all
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
    """PIDs still carrying ``launch_id`` in their environment.

    torchrun starts every worker with ``start_new_session=True``, so the workers sit in their own
    process groups and a ``killpg`` on the agent never reaches them. Matching the stamped launch id
    rather than the script path targets only processes this launcher spawned, not a co-tenant's job on
    a shared box.

    One pass is not a survey: a single ``/proc`` readdir drops a live entry whenever the process table
    churns underneath it (~15% against a process both the preceding and following pass saw), and
    under-reporting would let a wedged worker keep its GPU. The passes are unioned so a miss has to
    happen twice in a row.
    """
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
                continue  # exited between iterdir and read, or not ours to read
    return sorted(survivors)


def _kill_launch(proc: subprocess.Popen, launch_id: str) -> list[int]:
    """Kill the elastic agent and every worker it left behind; return whatever refused to die.

    SIGTERM first so torchrun runs its own shutdown (it reaps its workers), then SIGKILL the agent's
    group, then sweep the workers that outlive it: a wedged NCCL rank never exits on its own and holds
    both its GPU and this launch's stdout pipe open, stalling the next node and the output drain.
    """
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
    """``tmp_path``, or a short unique sibling when it leaves no room for an AF_UNIX socket.

    Keeps the scratch dir on whatever volume pytest's basetemp lives on (never the small root FS),
    and stays per-run isolated either way.
    """
    if len(str(tmp_path)) + _SOCKET_SUFFIX_BUDGET <= _AF_UNIX_MAX:
        return tmp_path
    # Sibling of the nodeid-named dir: still inside basetemp, so the retention policy reclaims it.
    return Path(tempfile.mkdtemp(prefix="h", dir=tmp_path.parent))


class GPUCase:
    """One launchable manifest node: a script + a single args variant."""

    def __init__(self, rel: str, spec: TestSpec, args: str):
        self.rel = rel
        self.spec = spec
        self.args = args

    def _launch_once(self, tmp_path: Path):
        port = free_port()
        launch_id = uuid.uuid4().hex
        # TORCHELASTIC_ERROR_FILE is left unset: one shared path makes the last writer win, and the
        # agent's collateral SIGTERMs would overwrite the real cause.
        env = {
            **os.environ,
            "MASTER_PORT": str(port),
            _LAUNCH_ID_VAR: launch_id,
            # Per-run isolated temp: a crash can't poison the next node and ranks never share a
            # tokenizer-cache dir (cross-model token-ID collisions).
            "TMPDIR": str(_socket_safe_tmpdir(tmp_path)),
        }
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={self.spec.nproc}",
            f"--master_port={port}",
            str(script_path(self.rel)),
            *(self.args.split() if self.args else []),
        ]
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,  # own process group -> killpg on timeout
        )
        try:
            stdout, _ = proc.communicate(timeout=self.spec.timeout)
            return proc.returncode, stdout, []
        except subprocess.TimeoutExpired:
            leaked = _kill_launch(proc, launch_id)
            # Re-``communicate`` rather than ``proc.stdout.read()``: the first call discards its
            # buffer on the raise, so a bare read() returns only the tail. Bounded, since an unkilled
            # worker holds the write end open indefinitely.
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
            print(stdout)  # surfaced under pytest -s / on failure

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
                # Quote the match and the tail: this keys on a substring, so a script that merely
                # prints an OOM string is labelled one.
                pytest.fail(
                    f"ERROR: {self.rel} — OOM at nproc={self.spec.nproc} (this config must fit). "
                    f"Matched {matched_oom!r}; last output:\n" + "\n".join(stdout.splitlines()[-15:]),
                    pytrace=False,
                )

            result = parse_result(stdout)
            # A script declining to run prints ``SKIP:`` to be reported as one; exit 0 with no result
            # line still counts as a pass below, since exit-code-only scripts have no other channel.
            # Substring rather than prefix: ``log()`` stamps ``[Rank N] `` and torchrun interleaves
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
                # Only rank 0 emits the result line, so pass plus a non-zero exit means a non-zero
                # rank failed: a correctness failure rather than an infra error.
                pytest.fail(
                    f"FAIL: {self.rel} [{self.args}] — rank 0 reported "
                    f"status={result['status']} but the launch exited {code}: a non-zero rank failed "
                    "(see the '[Rank N] FAILED CHECKS' / FATAL lines above)",
                    pytrace=False,
                )
            # Exit non-zero with no parseable result: infra error.
            pytest.fail(
                f"ERROR: {self.rel} [{self.args}] exited {code} with no result line (infra / hang / import crash)",
                pytrace=False,
            )

        pytest.fail(f"ERROR: {self.rel} [{self.args}] — exhausted retries ({last})", pytrace=False)

    def __repr__(self):
        return f"{self.rel}[{self.args}]" if self.args else self.rel


def pytest_generate_tests(metafunc):
    """Expand the manifest into one parametrized ``test_gpu_script`` node per (script, args)."""
    if "gpu_case" not in metafunc.fixturenames:
        return
    params = []
    for rel, spec in MANIFEST.items():
        marks = [getattr(pytest.mark, m) for m in spec.markers]
        for args in spec.args_matrix:
            node_id = rel if not args else f"{rel}[{args}]"
            params.append(pytest.param(GPUCase(rel, spec, args), marks=marks, id=node_id))
    metafunc.parametrize("gpu_case", params)


@cache
def _server_is_up(url: str) -> bool:
    """Whether ``<url>/health`` answers 2xx. Any failure (refused, DNS, timeout, 5xx) reads as down."""
    if "://" not in url:
        url = f"http://{url}"  # a scheme-less URL raises ValueError below, which would read as down
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=_SERVER_PROBE_TIMEOUT_S) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _absent_server(markers: tuple) -> str | None:
    """Skip reason when this node's engine is not reachable, else ``None``.

    A server-tier node drives an engine the launcher does not own; on a host where it is down there
    is nothing to test rather than something broken, so the node skips like one needing more GPUs than
    are present. Unless the run set ``HALO_TEST_REQUIRE_SERVER`` (the make server tiers do), where the
    caller turns this reason into a ``UsageError``.
    """
    for marker, (env_var, default_url, start_cmd) in _SERVER_TIERS.items():
        if marker not in markers:
            continue
        url = os.environ.get(env_var) or default_url
        if not _server_is_up(url):
            return f"no {marker.split('_')[0]} server at {url} (start it: {start_cmd})"
    return None


def pytest_collection_modifyitems(config, items):
    """Fail fast on manifest drift and skip nodes that need more GPUs than are present."""
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
