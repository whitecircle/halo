"""Execution of untrusted programs (Python, C, C++, bash) against stdin: three ``SandboxExecutor`` backends (local
subprocess, bubblewrap namespace jail, remote SandboxFusion), ``factory`` to pick one from args or env, the lighter
in-process restricted evaluator in ``inprocess``, and in ``repl`` the reply a REPL tool returns for a run plus
``run_code_via_sandbox``."""
