# Contributing to Halo

Halo is actively developed and maintained by **White Circle**, and we genuinely want your bug
reports, ideas, and pull requests. Because Halo is itself built with AI — and AI makes plausible-looking
but ill-fitting contributions cheap to produce — **contributions are gated**: a light process that keeps
quality high, keeps AI-slop out, and keeps direction under maintainer control. Think of it as routing,
not a wall. Read this before opening an issue or PR; the full dev guide (build, test, docs) lives in
[`agent-docs/contributing/README.md`](agent-docs/contributing/README.md).

## The gate (read first)

1. **Open an issue first.** Everything starts as an issue — pick the right template: **Bug report** for a
   reproducible bug, **Feature request / proposal** to propose a change, **Question / support** for a
   usage or design question, **Documentation** for a docs error or gap. A maintainer may reject it, build
   it directly, ask for detail, or accept it as work to be done.
2. **Issues are the front door and the maintainer's work queue.** Anyone can open one via a template;
   maintainers triage. A proposal being accepted is what greenlights the work.
3. **Don't open a PR without an accepted issue and explicit approval.** A maintainer adds you to
   [`.github/APPROVED_CONTRIBUTORS`](https://github.com/whitecircle/halo/blob/allowlist/.github/APPROVED_CONTRIBUTORS)
   (kept on the `allowlist` branch) by commenting `/approve @your-handle` on an accepted issue. **PRs from anyone not on that list (and without write access) are auto-commented
   and closed.** A reaction, comment, branch, or draft does **not** reserve the work or approve your
   PR path. Land one PR and you're added automatically — repeat contributors skip the gate.
4. **Demand ≠ implementation.** Reactions and comments show interest; they don't guarantee implementation,
   priority, or maintainer attention.

The gate exists to keep review focused on work that fits and to keep AI-slop out of the tree — it's
quality control, not a barrier. Good bug reports, well-argued proposals, and code you understand are all
welcome.

## The one rule

**You must understand your code.** If you can't explain what your change does, how it behaves at the
edges, and how it fits Halo's design, the PR is closed. **Using AI to write code is fine — submitting
code you don't understand is not.** Halo is itself built with AI (see the README's *Built with AI*); the
bar is comprehension and ownership, not avoidance of AI.

If you used AI, **disclose it** in the PR (the template has a checkbox): the scaffold and the models.
Coding agents working in this repo must respect this gate — see [`AGENTS.md`](AGENTS.md) and
[`CLAUDE.md`](CLAUDE.md).

## If you're approved — the engineering bar

Once your PR path is approved, the normal bar applies (full detail in
[`agent-docs/contributing/README.md`](agent-docs/contributing/README.md)):

- **Everything runs in the Docker image.** The host has no usable Python; use the `make` targets
  (`make build-blackwell`, `make install`, `make test-cpu`, `make docs`); only `make lint` / `make format` run on the host (`uvx ruff`).
- **Pass the gates.** `make lint`, `make format`, `make test-cpu` (and `make test-gpu-core` for
  GPU-affecting changes), `make docs`.
- **Tests ship with behavior — and must not be slop.** A test must *fail when the behavior breaks*; no
  tautologies, smoke-only "didn't raise" checks, vacuous `assert x is not None`, or mock-the-thing-under-test.
  See the anti-slop test guide in [`agent-docs/contributing/README.md`](agent-docs/contributing/README.md).
- **Proof of value.** No-op/refactor → bitwise-identical loss (fixed seed); behavior change → an e2e
  test; perf → before/after tokens/s/GPU + peak memory. The PR template has a checkbox per case.
- **Keep it focused.** A diff over **~2,000 lines** is hard to review and will be deferred — split it.
- **Work from a fork.** Outside contributors have no push access here: fork the repo, clone your
  fork, branch off `main`, push to your fork, and open the PR against `whitecircle/halo`'s `main`.
- **Sign your commits** (SSH or GPG — the *Verified* badge): this repository requires signed
  commits on every branch. A PR whose commits are unsigned can still land, but only by **squash
  merge** (GitHub signs the squash commit it creates) — sign your commits if you want your commit
  structure preserved. One-time setup: `git config --global gpg.format ssh`,
  `git config --global user.signingkey ~/.ssh/id_ed25519.pub`, `git config --global commit.gpgsign true`.

## Markdown locations

Source-tree markdown belongs in `agent-docs/` (the reference), `human-docs/` (the user guide), or
`skills/` (the agent skills). The GitHub-convention root files (`README`, `CONTRIBUTING`, `SECURITY`,
`AGENTS.md`, `.github/*`) are the documented exception.

## Security

Never commit `keys/`, `.env`, or `*.pem`. Report vulnerabilities **privately** — see
[`SECURITY.md`](SECURITY.md), never a public issue.

## Code of conduct

Participation in issues, PRs, and reviews falls under the
[Code of Conduct](CODE_OF_CONDUCT.md) (Contributor Covenant).
