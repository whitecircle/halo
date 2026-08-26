# allowlist

This branch holds exactly one thing: `.github/APPROVED_CONTRIBUTORS`, the contributor allowlist
that `pr-gate.yml` reads and the approval workflows write. It lives here — outside `main`'s
ruleset — because GitHub refuses the Actions app as a ruleset bypass actor by design; keeping the
list off `main` lets the workflow token maintain it while `main`'s reviewed-PR rule stays
exception-free. Do not merge this branch anywhere, and do not delete it: the gate reads it live.
