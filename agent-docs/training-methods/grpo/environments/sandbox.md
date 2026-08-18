# Code Execution Sandboxes

Two paths run model-generated code during RL rollouts, both in `src/environments/sandbox/`:

- **In-process restricted REPL** (`inprocess.py`, `run_python_sandboxed` / `safe_calculate`) — restricted-builtins Python in-process for math and logic. No imports, no OS isolation; deterministic and dependency-free. Backs the calculator and the default `python` REPL tool.
- **`SandboxExecutor`** — runs a complete, untrusted program (Python, C, or C++) against stdin in real OS isolation. Backs competitive-programming grading and multi-turn SWE / tool-use rollouts.

This page covers `SandboxExecutor`.

## Backends

Three backends implement one interface (`src/environments/sandbox/base.py`) and return the same `SandboxResult` (`stdout`, `stderr`, `returncode`, `timed_out`, `error`), so environment code is backend-agnostic.

| Backend | Class | Isolation | Languages |
|---------|-------|-----------|-----------|
| Local subprocess (default) | `LocalSubprocessSandbox` (`local.py`) | POSIX `RLIMIT_CPU` / `RLIMIT_AS` / `RLIMIT_FSIZE` / `RLIMIT_NPROC`, throwaway working dir, minimal child env, own process group. Bounds CPU/memory/disk; does **not** namespace the network or host filesystem | Python, C, C++ |
| Bubblewrap jail | `BubblewrapSandbox` (`bubblewrap.py`) | The local backend's compile/run core under a `bwrap` jail: fresh user/mount/PID/net namespaces (`--unshare-all`, no network unless `allow_network`), read-only system, writable working dir only. Strongest local isolation, unprivileged, parallel-safe | Python, C, C++ |
| Remote SandboxFusion | `RemoteSandbox` (`remote.py`) | OS/container, in the service; wire-compatible with a [SandboxFusion](https://github.com/bytedance/SandboxFusion) `/run_code` endpoint you host yourself — no sandbox service ships with Halo | Any the service exposes |

`BubblewrapSandbox` needs the `bwrap` binary (the `bubblewrap` apt package, installed in the training `Dockerfile`) **and** the right to create user + mount namespaces, which Docker's default seccomp profile denies. The constructor probes once and raises a `RuntimeError` if the jail can't be created, so the failure lands at construction rather than on every run. Grant the namespaces with `--privileged` (or an equivalent seccomp/AppArmor relaxation) on a host that permits unprivileged user namespaces (`kernel.unprivileged_userns_clone=1`).

An error the *program* caused (non-zero exit, compile error, timeout) leaves `error` unset, so grading counts it as a failed solution. A missing compiler, compile timeout, or transport failure sets `error` — a backend outage, not a verdict.

### Selection

`resolve_sandbox()` (`src/environments/sandbox/resolve.py`) picks the backend in precedence order **explicit config > env var > default**. The coding environments (`code_contests`, `codeforces`, `swe`) take `sandbox_backend` / `sandbox_url` in their `environment_kwargs`:

```yaml
environment_kwargs:
  sandbox_backend: remote          # local | bubblewrap | remote
  sandbox_url: http://sandbox-host:8080   # required when backend is remote
```

| Env var | Values | Default |
|---------|--------|---------|
| `HALO_SANDBOX_BACKEND` | `local`, `bubblewrap`, `remote` | `local` |
| `HALO_SANDBOX_URL` | endpoint for the remote backend | — (required when backend is `remote`) |
| `HALO_SANDBOX_MAX_CONCURRENCY` | execution slots in the [concurrency gate](#concurrency-gate) | host CPU count |

A backend/URL mismatch fails loud: `remote` without a URL raises, and so does a URL set against `local` or `bubblewrap`.

Pass `sandbox=` to an environment to inject a pre-built executor. That is also the only way to set the executor's own constructor arguments — `memory_limit_mb`, `compile_timeout`, `compile_memory_limit_mb`, and bubblewrap's `allow_network` / `extra_ro_binds` are not reachable from `environment_kwargs`, which forwards only `sandbox_backend` and `sandbox_url`.

## Language registry

`LanguageSpec` entries in `LANGUAGES` (`base.py`) define how each language is compiled and run; the same registry serves every backend. `resolve_language(name)` looks up a spec by canonical name or alias, `supported_languages()` lists the canonical names.

| Language | Aliases | Compile | Run |
|----------|---------|---------|-----|
| `python` | `py`, `python3` | — (interpreted) | the backend's interpreter with `-s -E` (the working dir stays on `sys.path`, so a session can import files written on earlier turns) |
| `cpp` | `c++`, `cxx`, `cc` | `g++ -O2 -pipe -std=c++17` | `./main` |
| `c` | — | `gcc -O2 -pipe -std=c11` | `./main` |

`SANDBOX_DEFAULT_TIMEOUT` (15 s) is the single source of truth for every code-execution wall-clock limit — the REPL/test tools, the SWE shell, and competitive-programming grading all default to it.

## Sessions (persistent multi-turn state)

`SandboxExecutor.open_session()` returns a `SandboxSession` — a working directory that survives across `run()` calls, which is what a multi-turn SWE agent or tool-use rollout needs. Its API: `run(code, *, stdin, timeout, language, files)`, `write_file`, `read_file` (`None` if absent), `list_files`, and an idempotent `close()` that deletes the working dir (also via context-manager exit). On the local backends a path that would escape the working directory is rejected.

The local and bubblewrap backends back a session with a `LocalSession` over the real working directory (the jail wraps each command). `RemoteSession` accumulates written files client-side and resends them with every request, since `/run_code` is stateless. One session per episode keeps concurrent rollouts isolated. `SandboxExecutor.run()` is the one-shot path used for stateless grading.

## Performance and parallelism

Every execution gets its own subprocess and (for bubblewrap) its own namespaces, and every session its own working directory, with no shared mutable state, so one backend instance is safe to share across threads and Ray rollout actors. Compile and run happen in child processes that release the GIL, so Python-side launch overhead never serializes the workload; thread concurrency scales to roughly the gate's slot count.

### Resource limits per execution

Each `run()` on the local and bubblewrap backends is bounded by per-run RLIMITs applied through a `ulimit` shell wrapper (`LocalSubprocessSandbox._limit_wrap`; constants in `base.py`). The wrapper is used instead of a `preexec_fn` so launches stay on the `vfork`/`exec` path and don't copy a resident model's page tables.

| Resource | Run step | Compile step | Constant |
|----------|----------|--------------|----------|
| Wall-clock (`timeout`) | 15 s | 30 s | `SANDBOX_DEFAULT_TIMEOUT` / `SANDBOX_DEFAULT_COMPILE_TIMEOUT` |
| CPU (`RLIMIT_CPU`) | `ceil(timeout + 1)` s | `ceil(compile_timeout + 1)` s | derived from the timeout |
| Address space (`RLIMIT_AS`) | 1024 MiB | 2048 MiB | `SANDBOX_DEFAULT_MEMORY_MB` / `SANDBOX_DEFAULT_COMPILE_MEMORY_MB` |
| File size (`RLIMIT_FSIZE`) | 64 MiB | 64 MiB | `LOCAL_FSIZE_LIMIT` |
| Processes/threads (`RLIMIT_NPROC`) | 4096 | not applied | `LOCAL_NPROC_LIMIT` |

The `RLIMIT_CPU` backstop kills a busy loop that outruns delayed timeout delivery; `SIGXCPU` is reported as `timed_out=True`, not a generic non-zero exit. The address-space cap is a ceiling, not a reservation — the headroom keeps a memory-heavy-but-correct CPython solution from being falsely killed.

`RLIMIT_NPROC` binds only unprivileged runs (the kernel skips it for root, how the training containers run), so on the plain local backend the timeout's process-group kill is the effective fork-bomb defense and the cap matters inside the bubblewrap jail. It counts the user's whole task set, threads included, so it must clear the host baseline. The remote backend enforces its own limits in the service.

### Concurrency gate

A per-test time limit is wall-clock, so it is only meaningful when an execution runs on an effectively dedicated core; without that, a correct, fast solution can be timed out purely from scheduling starvation. In `swe` and other tool-use sessions the same starvation surfaces as episodes hitting `episode_timeout` rather than as false verdicts.

The local and bubblewrap backends route every execution through a process-global `ExecutionGate` (`base.py`). At most `HALO_SANDBOX_MAX_CONCURRENCY` (default: host CPU count) runs proceed at once; the rest queue. The queue wait is **outside** the measured run — a run acquires its slot before its timeout starts.

The gate is **per process**, and its slot count is resolved once when `src/environments/sandbox/base.py` is imported — set the env var before the process starts, not from Python afterwards. When several grader processes share one host (parallel model sweeps, or Ray actors on one node), set `HALO_SANDBOX_MAX_CONCURRENCY` so the slots **sum** to the core count (e.g. 8 processes on a 240-core host → ~28 each).

The remote backend needs no gate — it offloads to a service that owns its own scheduling. For an eval sweep, record trajectories and [re-grade offline](benchmarks.md#offline-re-grading-contention-free) in one process instead of grading inline.

### Sizing at scale

One execution = one subprocess on one core + one throwaway working directory in `TMPDIR`. Point `TMPDIR` at a large volume: grading writes test inputs/outputs there, and a `swe` session's whole working tree lives there for the episode.

Budget by peak concurrent executions `C` (rollout actors × in-flight episodes × executions per episode): ~`C` cores, worst-case `C` × 1 GB of RAM (× 2 GB while a C/C++ compile is in flight), `C` × ≤64 MB of scratch.

Each in-flight execution also holds several file descriptors, so a high `C` exhausts the default 1024-fd limit and launches fail with `OSError: Too many open files` — raise it (`ulimit -n`, or `docker run --ulimit nofile=1048576`). During grading such an executor fault is bucketed as a grading infra error (the episode is marked invalid), never scored as the submission's failure.

### Ray rollout actors

Each `EnvironmentActor` builds its environment in its own process on its first episode and resolves the backend once there, so `HALO_SANDBOX_BACKEND` / `HALO_SANDBOX_URL` must be present in the *actor's* environment. Actors inherit the env of whatever started their raylet — the trainer process in local mode, the `ray start` daemon on a shared cluster. Nothing forwards the trainer's env, so export the vars before `ray start` on each node ([Ray Cluster](../../../infrastructure/ray.md#multi-node)).

Under `bubblewrap`, an actor whose container forbids user namespaces fails the jail probe during that construction, and every rollout it takes returns an error result.

The rollout calls `env.cleanup([episode_id])` in a `finally` after each episode, so a stateful env's per-episode session never leaks across a long-lived actor.
