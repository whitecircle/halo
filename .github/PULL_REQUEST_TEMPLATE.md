<!-- Heads up: PRs without an accepted issue + maintainer approval are auto-closed — to keep review
     focused and AI-slop out, not to reject you. Before opening, both should be true:
       1. an accepted issue exists for this work, and
       2. a maintainer commented `/approve @your-handle` on it (you're on .github/APPROVED_CONTRIBUTORS).
     Not there yet? Open an issue first (Feature request / proposal) — see CONTRIBUTING.md. Thanks for contributing to Halo! -->

## Approval

- [ ] This PR addresses an **accepted issue** (link it below) and my PR path was **approved** (I'm on `.github/APPROVED_CONTRIBUTORS`, or I have write access).
- [ ] I understand and can explain every line of this change (no unreviewed AI output).

## What & why

<!-- What does this change do, and why? Link any issue. -->

## Type of change

- [ ] Bug fix
- [ ] New feature / behaviour
- [ ] Performance
- [ ] Docs
- [ ] Refactor / no behaviour change

## Proof of Value

<!-- Pick what applies (see CONTRIBUTING.md). -->

- [ ] **No-op / refactor:** loss is bitwise-identical (fixed seed, deterministic) and the
      original checkpoint still loads.
- [ ] **Behaviour change:** added/updated a test demonstrating the new behaviour.
- [ ] **Performance:** before/after **tokens/s/GPU** and **peak memory** below.

```
<!-- paste tokens/s/GPU + peak mem before/after, or the test that proves the change -->
```

## Environment (optional)

<!-- Helpful for perf numbers and reproduction. Skip for docs-only / pure-refactor PRs. -->

- GPU(s): <!-- e.g. 8x B300 / 8x H200 -->
- Image tag: <!-- halo:blackwell / hopper, or public.ecr.aws/whitecircle/halo:blackwell-1.0.0 -->
- Parallelism / model: <!-- FSDP / EP=? / CP=? / TP=? / ETP=?  +  HF id + config -->

## AI assistance

<!-- AI-assisted PRs are welcome — see CONTRIBUTING.md. Disclose and own the output. -->

- [ ] This PR was written with AI assistance
  - **Scaffold(s):** <!-- e.g. Claude Code, PI-Agent, Codex (version optional) -->
  - **Model(s):** <!-- e.g. Claude Fable 5, GPT-5.3 -->
- [ ] I reviewed, ran, and understand every line; it follows `CLAUDE.md` and the project standards (no unreviewed AI-slop).

## Checklist

- [ ] `make lint` and `make format` pass
- [ ] Tests added/updated; `make test-cpu` (and `make test-gpu-core` if GPU-affecting) pass
- [ ] Docs updated for any user-facing change (`make docs` is green)
- [ ] No internal infra / secrets / credentials added
- [ ] Commits are signed (*Verified* badge) — unsigned PRs are squash-merged
- [ ] Focused diff (large PRs ~2,000+ lines are reviewed later — consider splitting)
