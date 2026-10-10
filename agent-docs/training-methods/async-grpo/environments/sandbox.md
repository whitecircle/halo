# Code Execution Sandboxes

`SandboxExecutor` (`src/environments/sandbox/`) runs a complete untrusted program — Python, bash, C or C++ — against stdin in a subprocess. It backs:

- code-contests grading and its scratchpad test tool;
- the `swe` environment's `run_code`, `run_bash_command` and file tools;
- checker verification in `scripts/environments/preparation/prepare_code_dataset.py`.

`calculate` and a standalone `python` / `python_repl` tool with no `sandbox=` executor run instead in the in-process restricted REPL (`inprocess.py`): restricted builtins, no imports, no OS isolation.

## Choosing a backend

| Backend | Isolation | Languages | Needs |
|---|---|---|---|
| `local` (default) | rlimits, throwaway working dir, stripped child env, own process group; no namespaces | python, bash, cpp, c | `pidfd_open` (Linux 5.3+, checked at construction); `gcc` / `g++` for C/C++ |
| `bubblewrap` | `local` inside a `bwrap` jail: own user, mount, PID, IPC, UTS and cgroup namespaces and, without `allow_network`, network; read-only system, only the working dir writable, no capabilities, jail root a subordinate uid | python, bash, cpp, c | What `local` needs, plus root, `bwrap`, util-linux `unshare`, `uidmap` and namespace rights ([Bubblewrap setup](#bubblewrap-setup)) |
| `remote` | In the service, behind a SandboxFusion-compatible `/run_code` endpoint | Whatever it exposes | Network to it |

Coding environments select it in `environment_kwargs`:

```yaml
environment_kwargs:
  sandbox_backend: remote                 # local (default) | bubblewrap | remote
  sandbox_url: http://sandbox-host:8080   # remote only
```

Precedence is explicit argument, then `HALO_SANDBOX_BACKEND` / `HALO_SANDBOX_URL`, then `local` (`resolve_sandbox()` in `resolve.py`). `remote` with no URL raises, and so does a URL with `local` or `bubblewrap`.

### Confinement

A confined program cannot write the host filesystem or reach the network, but can read what the backend exposes (`bubblewrap`'s read-only system paths and `extra_ro_binds`).

- `remote` confines; `bubblewrap` confines unless `allow_network`; `local` does not (`SandboxExecutor.isolated`).
- On `local` the program can rewrite what grades it, fetch a solution, and read the trainer's `--env-file` secrets from `/proc/<pid>/environ`.
- `swe` and `code_contests` log a warning when built on an executor that does not confine.

Use `bubblewrap` without network, or `remote`, for RL on untrusted code and whenever the trainer's environment carries secrets.

Only `bubblewrap` lets a compile error show under code contests' `verdict_detail: outcome` (`compiles_without_test_input`): its build runs before any test and its program cannot force a rebuild ([Grading rules](tasks/code-contests/grading.md#grading-rules)).

### Bubblewrap setup

The training image ships the `bwrap` binary, util-linux `unshare`, `uidmap`'s `newuidmap` / `newgidmap` and a subordinate id range for root in `/etc/subuid` and `/etc/subgid`; a custom image must add them.

The container needs the right to create namespaces, which Docker's default seccomp denies: run it `--privileged`, or with the narrower set below. Under either, mount a clean proc beside the container's own before the trainer starts. The kernel grants the jail a fresh `/proc` only while a proc mount in the container is fully visible, and the container's is not: Docker masks parts of it under the narrower set, and in a GPU container the NVIDIA runtime mounts over parts of `/proc/driver/nvidia` under both.

```bash
docker run --gpus all --init --cap-add SYS_ADMIN --security-opt seccomp=unconfined \
    --security-opt apparmor=unconfined ... halo:blackwell \
    bash -lc "mkdir -p /run/fullproc && mount -t proc proc /run/fullproc && exec torchrun ..."
```

- Without the clean mount the jail fails with `Can't mount proc on /newroot/proc`; without `SYS_ADMIN`, with `loopback: Failed RTM_NEWADDR`.
- The constructor probes once, so a blocked jail fails at construction.
- On several nodes, every container that hosts environment actors needs the same setup.

### Inside the jail

- The jail's root is root's first subordinate uid, so the trainer must run as root.
- The program owns its working directory, mounted at `/sandbox`, and no other host file.
- `RLIMIT_NPROC` binds per run on Linux 5.14+; an older kernel counts every jail against one uid.
- Every directory above a bound path, `TMPDIR` included, must be searchable by other users, as `/tmp` is.
- Keep `--init`, so the container's PID 1 reaps the processes of an actor that dies mid-run.
- A program that exits `128 + n` reads as killed by signal `n`.
- While a jailed run is in flight the host process is a child subreaper, so orphans of anything else it started are reparented to it until the run ends.

## Using it from Python

```python
from src.environments.sandbox.resolve import resolve_sandbox

sandbox = resolve_sandbox(backend="bubblewrap")   # args beat HALO_SANDBOX_*
result = sandbox.run("print(input())", stdin="hi\n", language="python", timeout=15.0)
```

`run()` is one-shot: a throwaway working directory, so a compiled program rebuilds every call. On `local` / `bubblewrap` a session keeps the directory across calls; a `remote` session resends only what `write_file` wrote and rebuilds every call:

```python
with sandbox.open_session() as session:
    session.write_file("lib.py", "X = 41\n")
    session.run("import lib; print(lib.X + 1)")   # "42"
    session.list_files()
    session.read_file("lib.py")                   # None if absent or not a regular file
```

`close()` deletes the working directory (`remote`: drops the written files), is idempotent and never raises; the context manager calls it.

`environment_kwargs` carries only `sandbox_backend` and `sandbox_url`. The executor's constructor arguments (`memory_limit_mb`, `compile_timeout`, `compile_memory_limit_mb`, bubblewrap's `allow_network` / `extra_ro_binds`) go through `sandbox=`:

```python
env = SweEnvironment(sandbox=resolve_sandbox("bubblewrap", memory_limit_mb=2048, allow_network=True))
```

### Session paths

- A path that leaves the working directory by name (empty, `..`, absolute) or that UTF-8 cannot encode is refused as a bad argument (`ValueError`). One that is a link, resolves outside through one, or names a FIFO or device is refused on write (`SessionPathError`); `read_file` returns `None` for anything but a regular file.
- A program that swaps a staged file for a link gets its own runtime error (`working directory tampered`). A removed working directory is recreated empty.
- A working directory replaced with a link or a file breaks the session: later runs report `agent_fault` ([Reading a result](#reading-a-result)), file operations raise `SandboxAgentFault`. On `local` another episode's program can cause this.

### Reading a result

Read the result in this order:

- `error` — a backend or transport fault (missing compiler, HTTP failure, remote timeout, SandboxFusion `SandboxError`). Never a verdict on the code.
- `agent_fault` — the program replaced its working directory; grading reads it as a runtime error.
- `compile_failed` — the compiler rejected the source (`returncode` and `stderr` are its), or the build passed the compile limit (`returncode` unset).
- `timed_out`, then a non-zero `returncode` — the program's own failure. A negative `returncode` is the signal that killed it.

`ok` is True only when the program built and exited zero. `format_sandbox_repl_output` (`repl.py`) renders a result for a tool reply, error first, so a truncated observation keeps it.

## Sandbox faults

`format_sandbox_repl_output` raises the two non-verdict results as typed exceptions, and every protocol's tool dispatch books them by type:

| Exception | Raised for | Price | GRPO baseline |
|---|---|---|---|
| `SandboxInfraError` | `error` | None: the call is unpriced | Out: the episode is marked `episode_invalid`, the fault as its `episode_invalid_reason` |
| `SandboxAgentFault` | `agent_fault` | A failed call (`tool_error_penalty`) | In: scored against its group like any other episode |

A sandbox-backed tool of your own raises them the same way (a missing session is a `SandboxInfraError`). Either fault:

- ends the episode uncompleted and not truncated, so `mask_truncated_completions` keeps an agent fault in the loss;
- grades 0 on an environment graded only when the episode completes, while `code_contests` keeps an earlier graded submission;
- skips the external scorers ([Settlement](../../grpo/rewards/README.md#settlement));
- sets `info["sandbox_fault"]` (`infra` or `agent`) and logs as `episode/sandbox_infra_fault` / `episode/sandbox_agent_fault`. Both count toward `episode/natural_termination_rate`.

Any other exception a tool raises is an ordinary tool error, priced `tool_error_penalty`. That includes a host-side `OSError` from a `local` / `bubblewrap` run, such as `ENOSPC` (full `TMPDIR`) or `EAGAIN` (full process table).

A grader of your own that calls `sandbox.run()` must handle its host-side exceptions itself. [Code-contests grading](tasks/code-contests/grading.md) books a sandbox infra error per test; a grade that lost tests to it and had none pass or fail marks the episode `episode_invalid` (`episode/grading_infra_outage`).

### What a program can void

A program that drives its own run into an `error` voids its episode. The routes left:

- `remote`: a response the service fails to produce (huge output, service OOM), or one past the client deadline (run timeout + 30 s).
- `local` / `bubblewrap` grading: a host-side exception during a test, such as a `TMPDIR` the program filled.
- `local` only: a `setsid()` child that outlives the run fills the process table, or, under a root grader, the program deletes the compiler (`compiler not found` for every later build on that host).

On `local` / `bubblewrap`, output floods, escaping session paths and removed permissions are not routes.

Frequent faults bias the batch: the episodes that call the sandbox most drop most. Watch `episode/sandbox_infra_fault` for the tools; code contests also logs `episode/grading_infra_outage` and `episode/grade_inconclusive` (a grade that ended with no test failed and not every test passed) for grading. `remote` retries nothing.

## Languages

`LANGUAGES` (`src/environments/sandbox/base.py`) is the registry `local` and `bubblewrap` run and constructors validate against; an unknown name returns an `error` result. `remote` forwards the name to the service.

| Language | Aliases | Compile | Run |
|---|---|---|---|
| `python` | `py`, `python3` | — | The backend's interpreter, `-s -E`; the working dir stays on `sys.path` |
| `bash` | `sh`, `shell` | — | `bash main.sh` from the working dir |
| `cpp` | `c++`, `cxx`, `cc` | `g++ -O2 -pipe -std=c++17` | `./main` |
| `c` | — | `gcc -O2 -pipe -std=c11` | `./main` |

`SandboxExecutor.toolchain(language)` describes how a backend builds or runs a language (compile flags, `CPython 3.12`) for a tool description to name; `remote` returns nothing.

## Limits

Rlimits bound each `local` / `bubblewrap` run; `remote` enforces its own.

| Resource | Run step | Compile step | Set by |
|---|---|---|---|
| Wall-clock | 15 s | 30 s | `run(timeout=)` (`SANDBOX_DEFAULT_TIMEOUT`, every tool's default) / `compile_timeout` |
| CPU (`RLIMIT_CPU`) | wall-clock + 1 s | compile timeout + 1 s | fixed slack |
| Address space (`RLIMIT_AS`) | 1024 MiB | 2048 MiB | `memory_limit_mb` / `compile_memory_limit_mb` |
| File size (`RLIMIT_FSIZE`), captured stdout / stderr included | 64 MiB | 64 MiB | fixed |
| Processes (`RLIMIT_NPROC`) | 4096 per run on `bubblewrap`; on `local`, none for uid 0 | not applied | fixed |
| Stack (`RLIMIT_STACK`) | the address-space limit, compiled languages only | inherited | `memory_limit_mb` |
| Core dumps (`RLIMIT_CORE`) | 0 | 0 | fixed |

- **Stack.** A compiled program's stack is capped at the process's hard stack limit, which `--ulimit stack=67108864` holds at 64 MiB. An interpreter keeps the inherited stack.
- **Math threads.** `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` and `MKL_NUM_THREADS` are `1` in the child env; a host-sized pool would pass the address-space limit and abort `import numpy`.
- **CPU backstop.** The CPU limit is set soft and hard alike, so a program that reaches it before the wall-clock timeout (threads spinning in parallel) is killed with `SIGKILL` and reads as a runtime error, not as `timed_out`.
- **Fork bombs.** `RLIMIT_NPROC` does not bind uid 0, so on `local` in a root container no rlimit bounds a fork bomb's process count; the run's end bounds only how long they live.
- **Process group.** A run is judged on the leader's exit and output; its process group is killed when the leader exits. On `local` a child that `setsid()`s out escapes.

## Concurrency and sizing

- Every `local` / `bubblewrap` execution takes a slot of a process-global `ExecutionGate` before its timeout starts. `HALO_SANDBOX_MAX_CONCURRENCY` sets the slot count (default: the CPUs in the process's affinity set); the rest queue.
- Oversubscribed cores make a correct solution time out from starvation, and `swe` episodes hit `episode_timeout`.
- The gate is **per process**, fixed at import: set the variable before the process starts. When several processes share a host, size the slots to **sum** to the core count.
- Each execution writes a working directory plus stdout and stderr files of up to 64 MiB each under `TMPDIR`. Point `TMPDIR` at a large volume; a `swe` session's whole tree lives there for the episode.
- A high slot count exhausts the default 1024-fd limit (`docker run --ulimit nofile=1048576`).
- Under Ray the backend resolves inside each `EnvironmentActor`, so these variables must be in the *actor's* environment ([Actor runtime](README.md#actor-runtime)).

## Related pages

- [Code Contests](tasks/code-contests/README.md) · [SWE](tasks/swe.md) · [Environments](README.md) — the callers.
- [Memory and Throughput](../performance.md#sizing-a-run) — sizing a rollout round.
- [Ray Cluster](../../../infrastructure/ray.md#multi-node) — actor environments across nodes.
