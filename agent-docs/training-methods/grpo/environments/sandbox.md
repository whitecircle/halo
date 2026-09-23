# Code Execution Sandboxes

`SandboxExecutor` (`src/environments/sandbox/`) runs a complete untrusted program — Python, bash, C, or C++ — against stdin in a subprocess: under rlimits only on `local`, confined on `bubblewrap` and `remote`. It backs `submit_solution` hidden-test grading and the scratchpad test tool (`python_repl` for a Python-only run, else `run_code`) in `code_contests` / `codeforces`, the `swe` environment's `run_code`, `run_bash_command` and file tools, and checker verification in `scripts/environments/preparation/prepare_code_dataset.py`.

The in-process restricted REPL (`inprocess.py`) — restricted builtins, no imports, no OS isolation — is the other path, behind `calculate` and a standalone `python` / `python_repl` tool with no `sandbox=` executor.

## Choosing a backend

| Backend | Isolation | Languages | Needs |
|---|---|---|---|
| `local` (default) | rlimits, throwaway working dir, stripped child env, own process group; no namespaces | python, bash, cpp, c | `gcc` / `g++` for C/C++ |
| `bubblewrap` | that core in a `bwrap` jail (`--unshare-all`): read-only system, only the working dir writable | python, bash, cpp, c | `bwrap` plus user-namespace rights |
| `remote` | in the service, behind a SandboxFusion-compatible `/run_code` endpoint | whatever it exposes | network to it |

Coding environments select it in `environment_kwargs`:

```yaml
environment_kwargs:
  sandbox_backend: remote                 # local (default) | bubblewrap | remote
  sandbox_url: http://sandbox-host:8080   # remote only
```

`resolve_sandbox()` (`src/environments/sandbox/resolve.py`) takes **explicit argument > env var > default** for `HALO_SANDBOX_BACKEND` (`local`) and `HALO_SANDBOX_URL` (unset). `remote` with no URL raises; so does a URL against `local` or `bubblewrap`, which would be ignored while untrusted code ran here.

`bubblewrap` needs the `bwrap` binary (the `bubblewrap` apt package, in the training image) **and** the right to create user and mount namespaces, which Docker's default seccomp denies — run the container `--privileged` on a host allowing them (`kernel.unprivileged_userns_clone=1`). Its constructor probes once, so a blocked jail fails at construction, not per run.

## Using it from Python

```python
from src.environments.sandbox.resolve import resolve_sandbox

sandbox = resolve_sandbox(backend="bubblewrap")   # args beat HALO_SANDBOX_*
result = sandbox.run("print(input())", stdin="hi\n", language="python", timeout=15.0)
```

`run()` is one-shot: a throwaway working directory, so a compiled program rebuilds every call. A session keeps that directory across calls.

```python
with sandbox.open_session() as session:
    session.write_file("lib.py", "X = 41\n")
    session.run("import lib; print(lib.X + 1)")   # "42"
    session.list_files()                          # ['lib.py', 'main.py']
    session.read_file("lib.py")                   # None if absent or not a regular file
```

`close()` deletes the local working directory and is idempotent; the context manager calls it.

The host never follows a link into the session: every staged or read path must resolve to itself
under the real working directory, and the open carries `O_NOFOLLOW`, so a name that escapes or is
a link is refused (`SessionPathError`), as is a FIFO or device, which the host opens non-blocking —
`read_file` returns `None` for anything but a regular file, `write_file` raises on an unsafe name.
A program that swaps a staged entry for a link is booked as its **own** runtime error (`returncode`
set, `working directory tampered`), never an infra fault it could void its episode with. A working
directory the program removed is recreated empty on the next use. One it replaced with a link or a
file breaks the session: the run's result carries `agent_fault` with a non-zero `returncode`, every
later run reports the same without staging, the file operations raise `SandboxAgentFault`, and
`close()` unlinks what took its place. On `local` another episode's program can do the same to a
session, so the attribution holds only where the backend isolates episodes.

Read the result in this order:

- `error` — a backend or transport fault (missing compiler, HTTP failure, a remote request timeout, a SandboxFusion `SandboxError`). Never a verdict on the code.
- `agent_fault` — the program replaced its working directory, as above; grading reads it as a runtime error.
- `compile_failed` — the source never built: the compiler rejected it (`returncode` is the compiler's, `stderr` its diagnostics) or the build ran past the compile limit (an `#include` bomb).
- `timed_out`, then a non-zero `returncode` — the program's own failure, `error` unset; `remote` reads a SandboxFusion `Failed` response off its run block the same way. `ok` is True only when it built and exited zero.

`environment_kwargs` carries only `sandbox_backend` and `sandbox_url`, so `sandbox=` is the only way to set the executor's constructor arguments (`memory_limit_mb`, `compile_timeout`, `compile_memory_limit_mb`, bubblewrap's `allow_network` / `extra_ro_binds`):

```python
env = SweEnvironment(sandbox=resolve_sandbox("bubblewrap", memory_limit_mb=2048, allow_network=True))
```

## Sandbox faults

`run_code_via_sandbox` raises the two non-verdict results as typed exceptions, and every protocol's tool dispatch (native, async native, ReAct) books them by type:

| Exception | Raised for | Price | GRPO baseline |
|---|---|---|---|
| `SandboxInfraError` | `error` | none: the call is unpriced | out: the episode is marked `episode_invalid`, the fault as its `episode_invalid_reason` |
| `SandboxAgentFault` | `agent_fault` | a failed call (`tool_error_penalty`) | in: scored against its group like any other episode |

A sandbox-backed tool of your own raises them the same way (a missing session is a `SandboxInfraError`). Either fault ends the episode uncompleted and not truncated (so `mask_truncated_completions` keeps an agent fault in the loss): a completion-graded environment grades it 0, while `code_contests` keeps an earlier graded submission. The episode is never sent to an external scorer and logs no `episode/reward_scored`. `info["sandbox_fault"]` names the class (`infra` or `agent`); `episode/sandbox_infra_fault` and `episode/sandbox_agent_fault` log the per-episode rates, and both count toward `episode/natural_termination_rate`. Any other exception a tool raises is an ordinary tool error, priced `tool_error_penalty`.

Voiding is only as sound as the backend's containment of the program: whatever the program can drive into an `error` voids its own episode. The routes left to it:

- `remote`: a response the service fails to produce (a huge output, the service's own OOM) or one past the client deadline, `run_timeout` + 30 s.
- Grading on `local` / `bubblewrap`: any host-side exception during a test is an infra error for that test (`_run_in_sandbox`) — on `local`, the `EAGAIN` of a process table the program's leftover processes filled; on either, the `ENOSPC` of a `TMPDIR` it filled.

An output flood is not one: output is captured in files under the child's `RLIMIT_FSIZE`, so it ends as the program's own failure at the file-size limit. Nor is a lone surrogate in the program's source, stdin or files: every backend replaces it with `?`, as a text-mode pipe writes it. Where faults are frequent, dropping them is a selection — the episodes that call the sandbox most drop most. Watch `episode/sandbox_infra_fault`; `remote` retries nothing.

## Languages

`LANGUAGES` (`src/environments/sandbox/base.py`) is the registry `local` and `bubblewrap` read, and every construction-time caller validates against; an unknown name returns an `error` result. `remote` forwards the name to the service, which answers for it.

| Language | Aliases | Compile | Run |
|---|---|---|---|
| `python` | `py`, `python3` | — | the backend's interpreter, `-s -E`; the working dir stays on `sys.path` |
| `bash` | `sh`, `shell` | — | `bash main.sh` — the submitted text is the script, run from the working dir |
| `cpp` | `c++`, `cxx`, `cc` | `g++ -O2 -pipe -std=c++17` | `./main` |
| `c` | — | `gcc -O2 -pipe -std=c11` | `./main` |

`SANDBOX_DEFAULT_TIMEOUT` is the single default wall-clock limit: the REPL and test tools, the `swe` session and hidden-test grading all start from it.

## Limits

Per-run rlimits bound each `local` / `bubblewrap` execution; `remote` enforces its own in the service.

| Resource | Run step | Compile step | Constant |
|---|---|---|---|
| Wall-clock | 15 s | 30 s | `SANDBOX_DEFAULT_TIMEOUT` / `SANDBOX_DEFAULT_COMPILE_TIMEOUT` |
| CPU (`RLIMIT_CPU`) | wall-clock + 1 s | compile timeout + 1 s | `RLIMIT_CPU_SLACK_SECONDS` |
| Address space (`RLIMIT_AS`) | 1024 MiB | 2048 MiB | `SANDBOX_DEFAULT_MEMORY_MB` / `SANDBOX_DEFAULT_COMPILE_MEMORY_MB` |
| File size (`RLIMIT_FSIZE`), captured stdout / stderr included | 64 MiB | 64 MiB | `LOCAL_FSIZE_LIMIT` |
| Processes (`RLIMIT_NPROC`) | 4096 | not applied | `LOCAL_NPROC_LIMIT` |

The `RLIMIT_CPU` backstop kills a busy loop that outruns timeout delivery, reporting `SIGXCPU` as `timed_out=True` — a spin still reads as a time limit. `RLIMIT_NPROC` does not bind a root process (how the containers run), so the process-group kill is `local`'s real fork-bomb defense. The group is killed when the leader exits too, so a run is judged on the leader's exit and output, and a child left in its process group neither outlives it nor holds it open; one that `setsid()`s out of the group escapes on `local`.

## Concurrency and sizing

Every `local` / `bubblewrap` execution takes a process-global `ExecutionGate` slot before its timeout starts; `HALO_SANDBOX_MAX_CONCURRENCY` (default: host CPU count) sets the slot count and the rest queue. A wall-clock limit measures the program only on an effectively dedicated core — oversubscribe and a correct, fast solution times out from starvation, while `swe` episodes hit `episode_timeout`. Remote needs no gate.

The gate is **per process**, its slot count fixed at import: set the variable before the process starts, and when several processes share a host size the slots so they **sum** to the core count.

One execution is a child process in a working directory under `TMPDIR`, its stdout and stderr captured in two more files there (up to 64 MiB each, so 2 × 64 MiB per concurrent run). Point `TMPDIR` at a large volume: a `swe` session's whole tree lives there for the episode. Several file descriptors go with each execution, so a high slot count exhausts the default 1024-fd limit (`docker run --ulimit nofile=1048576`).

Under Ray the backend resolves inside each `EnvironmentActor`'s own process, so these variables must be in the *actor's* environment ([actor runtime](README.md#actor-runtime)). `env.cleanup([episode_id])` runs in a `finally` after every episode, so no session leaks across a long-lived actor.

## Related pages

- [Code Contests](code-contests.md) · [SWE](swe-environment.md) · [Environments](README.md) — the callers
- [Memory and Throughput](../async-grpo/performance.md#sizing-a-run) — sizing a rollout round
- [Ray Cluster](../../../infrastructure/ray.md#multi-node) — actor environments across nodes
