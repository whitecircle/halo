"""Execution of untrusted programs (Python, C, C++, bash) against stdin: three ``SandboxExecutor`` backends (local
subprocess, bubblewrap namespace jail, remote SandboxFusion), ``factory`` to pick one from args or env, and the
lighter in-process restricted REPL in ``inprocess`` / ``repl``."""
