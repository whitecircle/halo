# Contributing

The full contribution policy lives in the root
[`CONTRIBUTING.md`](../CONTRIBUTING.md); this page is the short version.

Halo welcomes bug reports, proposals, and focused pull requests. Because AI
makes plausible-looking-but-wrong contributions cheap to produce, code changes
go through an accepted issue and a maintainer approval first.

## Start in the right place

| You have | Where it goes |
| --- | --- |
| Reproducible bug | Issue — **Bug report** template |
| Feature idea / proposal | Issue — **Feature request / proposal** template |
| Usage or design question | Issue — **Question / support** template |
| Docs error or gap | Issue — **Documentation** template |
| Security issue | Private report — see [`SECURITY.md`](../SECURITY.md) |
| Code change | Accepted issue + maintainer `/approve` first |

Don't open a PR before a maintainer approves you on an accepted issue —
unapproved PRs are closed automatically. Once you land one PR, you're added
to the approved list and skip the gate next time.

Work from a **fork**: fork the repo, clone your fork, branch off `main`, push
to your fork, and open the PR against `whitecircle/halo`.

## The one rule

**You must understand and own every line you submit.** Using AI to write code
is fine — Halo itself is built with AI. Submitting code you can't explain is
not, and the PR template asks you to disclose the scaffold and models you
used.

## The short checklist

1. Issue → approval → focused PR (keep diffs under ~2,000 lines).
2. Build the image and run the gates: `make lint`, `make format`,
   `make test-cpu`, `make docs` — plus `make test-gpu-core` for GPU-affecting
   changes. Lint/format run on the host; tests and the strict docs build run inside the image.
3. Ship tests that **fail when the behavior breaks** — no smoke-only or
   `assert x is not None` tests. The anti-slop test guide is in
   [`agent-docs/contributing/`](../agent-docs/contributing/README.md) ↗.
4. Sign your commits (SSH or GPG). The repo requires signed commits on every
   branch; an unsigned PR can still land, but only by squash merge. Never
   commit secrets, `.env`, or keys.

The dev-environment guide (building images, running tests, docs tooling) is
[`agent-docs/contributing/development-environment.md`](../agent-docs/contributing/development-environment.md) ↗.
