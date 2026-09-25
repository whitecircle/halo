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
| Code change | Accepted issue + a maintainer's `/approve @your-handle` on it first |

Don't open a PR before a maintainer approves you on an accepted issue —
unapproved PRs are closed automatically. After approval, reopen a PR closed
this way, or ask on the issue and a maintainer will. Once you land one PR,
you're added to the approved list and skip the gate next time.

Work from a **fork**: fork the repo, clone your fork, branch off `main`, push
to your fork, and open the PR against `whitecircle/halo`.

## The one rule

**You must understand and own every line you submit.** Using AI to write code
is fine — Halo itself is built with AI, and the images ship
[skills](ai-tooling.md) that teach an agent this codebase. Submitting code you
can't explain is not, and the PR template asks you to disclose the scaffold and
models you used.

## The short checklist

1. Issue → approval → focused PR (keep diffs under ~2,000 lines).
2. Pull (or build) the image and run the gates: `make lint`, `make format`,
   `make seed-hf-cache` (the Hub configs and tokenizers the CPU tests read;
   again when `tests/common/models.py` or `examples/` gain a repo),
   `make test-cpu`, `make docs` — plus `make test-gpu-core` for GPU-affecting
   changes. Lint/format and the docs link check run on the host; tests run
   inside the image (`make test-cpu` needs no GPU). Hosted CI runs only lint and
   the docs checks, so report the test results in the PR.
3. Ship tests that **fail when the behavior breaks** — no smoke-only or
   `assert x is not None` tests. The anti-slop test guide is in
   [`agent-docs/contributing/`](../agent-docs/contributing/README.md) ↗.
4. Every PR is squash-merged, and every commit in it must carry a verified
   signature (SSH or GPG), forks included: GitHub does not merge a PR while any
   of its commits lacks one. Signing setup and re-signing earlier commits:
   [`CONTRIBUTING.md`](../CONTRIBUTING.md). Never commit secrets, `.env`, or
   keys.

The dev-environment guide (building images, running tests, docs tooling) is
[`agent-docs/contributing/development-environment.md`](../agent-docs/contributing/development-environment.md) ↗.
