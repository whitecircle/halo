# Code Execution Sandboxes

`SandboxExecutor` (`src/environments/sandbox/`) runs a complete untrusted program — Python, bash, C, or C++ — against stdin in a subprocess: under rlimits only on `local`, confined on `remote` and on `bubblewrap` without `allow_network`. It backs `submit_solution` hidden-test grading and the scratchpad test tool (`python_repl` for a Python-only run, else `run_code`) in `code_contests` / `codeforces`, the `swe` environment's `run_code`, `run_bash_command` and file tools, and checker verification in `scripts/environments/preparation/prepare_code_dataset.py`.

The in-process restricted REPL (`inprocess.py`) — restricted builtins, no imports, no OS isolation — is the other path, behind `calculate` and a standalone `python` / `python_repl` tool with no `sandbox=` executor.

## Choosing a backend

| Backend | Isolation | Languages | Needs |
|---|---|---|---|
| `local` (default) | rlimits, throwaway working dir, stripped child env, own process group; no namespaces | python, bash, cpp, c | `pidfd_open` (Linux 5.3+, allowed by the seccomp profile; checked at construction); `gcc` / `g++` for C/C++ |
| `bubblewrap` | that core in a `bwrap` jail (`--unshare-all`): read-only system, only the working dir writable | python, bash, cpp, c | what `local` needs, plus `bwrap` and user-namespace rights |
| `remote` | in the service, behind a SandboxFusion-compatible `/run_code` endpoint | whatever it exposes | network to it |

Coding environments select it in `environment_kwargs`:

```yaml
environment_kwargs:
  sandbox_backend: remote                 # local (default) | bubblewrap | remote
  sandbox_url: http://sandbox-host:8080   # remote only
```

`resolve_sandbox()` (`src/environments/sandbox/resolve.py`) takes **explicit argument > env var > default** for `HALO_SANDBOX_BACKEND` (`local`) and `HALO_SANDBOX_URL` (unset). `remote` with no URL raises; so does a URL against `local` or `bubblewrap`, which would be ignored while untrusted code ran here.

`bubblewrap` needs the `bwrap` binary (the `bubblewrap` apt package, in the training image) **and** the right to create user and mount namespaces, which Docker's default seccomp denies — run the container `--privileged` on a host allowing them (`kernel.unprivileged_userns_clone=1`). Its constructor probes once, so a blocked jail fails at construction, not per run.

An executor declares whether it confines the program (`SandboxExecutor.isolated`, `False` unless declared): kept from writing the host's filesystem and from its network, though it may still read what the backend exposes (`bubblewrap`'s read-only system paths and `extra_ro_binds`). `remote` does, `bubblewrap` does unless `allow_network`, `local` does not. On `local` the program has this process's filesystem and network, where a policy can read or rewrite what grades it and fetch a solution, and it reads the grader's launch environment (`/proc/<pid>/environ`): the secrets the trainer was started with (`--env-file`) reach it. `swe` and `code_contests` log one warning per process per backend class when built on an executor that does not confine it. Use `bubblewrap` without network, or `remote`, for RL on untrusted code, and whenever the trainer's environment carries secrets.

An executor also declares whether a compile error may show under `verdict_detail: outcome` (`compiles_without_test_input`, `False` unless declared): only `bubblewrap` does, whose build runs before any test on an empty stdin and whose program can force no rebuild ([Grading rules](code-contests.md#grading-rules)).

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

`close()` deletes the local working directory, is idempotent and never raises; the context manager calls it.

The host never follows a link into the session: every staged or read path must resolve to itself
under the real working directory, and the open carries `O_NOFOLLOW`, so a name that escapes or is
a link is refused (`SessionPathError`), as is a FIFO or device, which the host opens non-blocking —
`read_file` returns `None` for anything but a regular file.
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

A sandbox-backed tool of your own raises them the same way (a missing session is a `SandboxInfraError`). Either fault ends the episode uncompleted and not truncated (so `mask_truncated_completions` keeps an agent fault in the loss): a completion-graded environment grades it 0, while `code_contests` keeps an earlier graded submission. It skips the external scorers ([Reward Terms](../rewards.md#environment-arm)). `info["sandbox_fault"]` names the class (`infra` or `agent`); `episode/sandbox_infra_fault` and `episode/sandbox_agent_fault` log the per-episode rates, and both count toward `episode/natural_termination_rate`. Any other exception a tool raises is an ordinary tool error, priced `tool_error_penalty`, and that includes a host-side `OSError` out of a `local` / `bubblewrap` run (the `ENOSPC` of a full `TMPDIR`, the `EAGAIN` of a full process table) in the scratchpad tools and the `swe` session. Grading books the same exception the other way: it is an infra error for the test (`_run_in_sandbox`, below), and a grade with one in which no test ran cleanly or passed marks the episode `episode_invalid` (`episode/grading_infra_outage`).

Voiding is only as sound as the backend's containment of the program: whatever the program can drive into an `error` can void its own episode. The routes left to it:

- `remote`: a response the service fails to produce (a huge output, the service's own OOM) or one past the client deadline, the run's timeout + 30 s.
- Grading on `local` / `bubblewrap`: any host-side exception during a test is an infra error for that test (`_run_in_sandbox`), such as the `ENOSPC` of a `TMPDIR` the program filled. On `local`, where a child that `setsid()`s out of its process group outlives the run (`bubblewrap`'s PID namespace and `--die-with-parent` end it with the jail), also the `EAGAIN` of a process table such children filled and, under a grader that is not root, one locking the working directory again just before the host enters or lists it.
- `local`: the program has the grader's uid and no namespaces, so under a root grader it can delete the compiler, and every later build on that host reports `compiler not found`.

On `local` / `bubblewrap` an output flood is not one: output is captured in files under the child's `RLIMIT_FSIZE`, so it ends as the program's own failure at the file-size limit. Nor is a lone surrogate in the program's source, stdin or files: every backend replaces it with `?`, as a text-mode pipe writes it. Every backend refuses a session file path UTF-8 cannot encode or one that leaves the working directory (empty, `.`, `..`, absolute, through `..`) as a bad argument (`ValueError`, a priced tool error). Nor, on `local` / `bubblewrap`, are permissions the program took away: under a grader that is not root the program shares its uid and can make its working directory, a directory in it or a staged file read-only, so before staging, before each reset and while removing the directory the host gives the owner back access to the whole tree, walked one descriptor at a time without recursion, so no link is followed and no depth of nesting defeats it. Only a child that escaped the process group can take it again after that: a staging write it refuses is the program's own runtime error, and an entry it keeps the reset from removing stays. Where faults are frequent, dropping them is a selection — the episodes that call the sandbox most drop most. Watch `episode/sandbox_infra_fault` for the tools and `episode/grading_infra_outage` for grading; `remote` retries nothing.

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

The `RLIMIT_CPU` backstop kills a busy loop that outruns timeout delivery, reporting `SIGXCPU` as `timed_out=True` — a spin still reads as a time limit. `RLIMIT_NPROC` does not bind a root process (how the containers run), so the process-group kill is `local`'s real fork-bomb defense. The group is killed when the leader exits too, before the leader is reaped, so a run is judged on the leader's exit and output and a child left in its process group does not outlive it; one that `setsid()`s out of the group escapes on `local`, without holding the run open.

## Concurrency and sizing

Every `local` / `bubblewrap` execution takes a process-global `ExecutionGate` slot before its timeout starts; `HALO_SANDBOX_MAX_CONCURRENCY` (default: host CPU count) sets the slot count and the rest queue. A wall-clock limit measures the program only on an effectively dedicated core — oversubscribe and a correct, fast solution times out from starvation, while `swe` episodes hit `episode_timeout`. Remote needs no gate.

The gate is **per process**, its slot count fixed at import: set the variable before the process starts, and when several processes share a host size the slots so they **sum** to the core count.

One execution is a child process in a working directory under `TMPDIR`, with its stdin, stdout and stderr in three more files there (the output ones up to 64 MiB each, so 2 × 64 MiB per concurrent run on top of its stdin). Point `TMPDIR` at a large volume: a `swe` session's whole tree lives there for the episode. Several file descriptors go with each execution, so a high slot count exhausts the default 1024-fd limit (`docker run --ulimit nofile=1048576`).

Under Ray the backend resolves inside each `EnvironmentActor`'s own process, so these variables must be in the *actor's* environment ([actor runtime](README.md#actor-runtime)). `env.cleanup([episode_id])` runs in a `finally` after every episode, so no session leaks across a long-lived actor.

## Related pages

- [Code Contests](code-contests.md) · [SWE](swe-environment.md) · [Environments](README.md) — the callers
- [Memory and Throughput](../async-grpo/performance.md#sizing-a-run) — sizing a rollout round
- [Ray Cluster](../../../infrastructure/ray.md#multi-node) — actor environments across nodes
