---
name: Bug report
about: Report a reproducible bug
title: "[bug] "
labels: bug
---

<!-- Bug reports are for reproducible defects. Not a bug? Use Feature request / proposal (a change),
     Question / support (a how-to), or Documentation (a docs error). Opening a PR needs an accepted
     issue + maintainer approval — see CONTRIBUTING.md; unapproved PRs are auto-closed. -->

## Summary

<!-- One or two sentences. -->

## Environment

- GPU(s): <!-- e.g. 8x B300 / 8x H200 -->
- Image tag: <!-- halo:blackwell / hopper, or public.ecr.aws/whitecircle/halo:blackwell-1.0.0 -->
- Halo version / commit: <!-- git rev-parse --short HEAD, or the release tag -->
- Parallelism: <!-- FSDP / EP=? / CP=? / TP=? / ETP=? -->
- Model + config: <!-- HF id + the YAML / command -->

## Reproduction

```bash
# the exact command (prefer a `make` target or `torchrun ...` line)
```

## Expected vs actual

<!-- What you expected, and what happened (paste the traceback / NCCL / DeepEP error). -->

## Logs

<!-- Relevant log lines. For hangs, include the rank that stalled and any NCCL timeout. -->

## Checklist

- [ ] Searched existing issues for a duplicate
- [ ] Reproduced inside the prebuilt Docker image (not a host-Python environment)
- [ ] Included the exact command and the full traceback / hang location
