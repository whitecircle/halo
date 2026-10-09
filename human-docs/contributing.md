# Contributing

This is the short version. The full policy is in the root
[`CONTRIBUTING.md`](../CONTRIBUTING.md).

Bug reports, proposals and focused pull requests are welcome. Code changes need
an accepted issue and a maintainer's approval first, because AI makes plausible
but wrong contributions cheap to produce.

## Start in the right place

| You have | Where it goes |
| --- | --- |
| Reproducible bug | Issue, **Bug report** template |
| Feature idea or proposal | Issue, **Feature request / proposal** template |
| Usage or design question | Issue, **Question / support** template |
| Docs error or gap | Issue, **Documentation** template |
| Security issue | Private report, see [`SECURITY.md`](../SECURITY.md) |
| Code change | An accepted issue, then a maintainer's `/approve @your-handle` on it |

Don't open a PR until a maintainer approves you on an accepted issue. PRs from
unapproved contributors are closed automatically.

- Once approved, reopen a PR the gate closed, or ask on the issue and a
  maintainer will. The gate runs again on reopen.
- After your first merged PR you're on the approved list and skip the gate.

Work from a fork: fork the repo, clone your fork, branch off `main`, push to your
fork, and open the PR against `whitecircle/halo`.

## The one rule

You must understand and own every line you submit. Writing code with AI is fine:
Halo itself is built with AI, and the images ship [skills](ai-tooling.md) that
teach an agent this codebase. Submitting code you can't explain is not. The PR
template asks you to disclose the scaffold and models you used.

## Checklist

1. Issue, approval, then a focused PR. Keep the diff under about 2,000 lines.
2. Pull or build the image and run the gates:

   ```bash
   make lint && make format   # host
   make docs                  # host (stdlib python3): link and anchor check
   make seed-hf-cache         # image: Hub configs and tokenizers the CPU tests read
   make test-cpu              # image, no GPU needed
   make test-gpu-core         # image, for GPU-affecting changes
   ```

   Re-run `make seed-hf-cache` when `tests/common/models.py` or `examples/` add
   a Hub repo. Hosted CI runs only lint and the docs checks, so report your test
   results in the PR.
3. Ship tests that fail when the behavior breaks. Smoke-only tests and
   `assert x is not None` don't count. See the test guide in
   [`agent-docs/contributing/`](../agent-docs/contributing/README.md) ↗.
4. Sign every commit (SSH or GPG), forks included. PRs are squash-merged, and
   GitHub won't merge one while any commit lacks a verified signature. Setup and
   re-signing are in [`CONTRIBUTING.md`](../CONTRIBUTING.md).
5. Never commit secrets, `.env` or keys.

Building images, running tests and the docs tooling are covered in
[Development environment](../agent-docs/contributing/development-environment.md) ↗.
