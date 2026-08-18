"""Execution of untrusted programs (Python, C, C++) against stdin: three ``SandboxExecutor`` backends (local
subprocess jail, bubblewrap namespace jail, remote SandboxFusion), ``resolve`` to pick one from args or env, and
the lighter in-process restricted REPL in ``inprocess``/``repl``."""
