# Continuous Integration

Halo's CI is GitHub Actions (`.github/workflows/`), split into two tiers by where the runner lives, plus the contribution gate.

| Tier | Runner | Workflows | Triggers | Status |
|------|--------|-----------|----------|--------|
| Hosted | GitHub `ubuntu-latest` | `.github/workflows/lint.yml`, `.github/workflows/docs.yml` | every PR + push `main` | active |
| Self-hosted | GPU box `[self-hosted, halo]` | `.github/workflows/cpu-tests.yml`, `.github/workflows/gpu-tests.yml` | `workflow_dispatch` | dispatch-only |

Hosted jobs are pure lint/link checks (ruff, actionlint, the docs link check) and need no image. The test tiers import torch and run inside the prebuilt image, so they cannot run on a hosted runner — and GPU tests additionally need Blackwell (SM100) or Hopper (SM90) for FA3/FA4 + DeepEP, which no hosted runner provides.

## Hosted tier

`.github/workflows/lint.yml`: `ruff format --check` plus the full pyproject rule set (including `PLC0415`, the no-inline-imports house rule) block the PR, and `actionlint` checks the workflow files themselves. Self-hosted runner labels are declared in `.github/actionlint.yaml` so actionlint does not flag them as unknown.

`.github/workflows/docs.yml`: `scripts/docs/check_links.sh` blocks on any broken relative link across `agent-docs/`, `human-docs/`, `skills/` and the root markdown. A `diagrams` job re-runs `scripts/diagrams/` in a `python:3.12-slim` container at `uv.lock`'s matplotlib pin and byte-compares the result against the committed figures under `agent-docs/assets/`, so a generator edited without re-committing its figure goes red — run `make diagrams` and commit. `diagrams` and `markdownlint` are advisory: blocking is enforced by the `main` ruleset's **required status checks** (`ruff`, `actionlint`, `build`), and a red check outside that set still lets a PR merge.

`.pre-commit-config.yaml` mirrors these gates on the host at commit time — the same pinned ruff (lint + format), plus `nbstripout`, whitespace fixers and a 1 MB file-size cap. It is optional: `pipx install pre-commit`, then `pre-commit install`.

## Self-hosted tier

`.github/workflows/cpu-tests.yml` asserts the image is present, then runs `make test-cpu` — the image without `--gpus`, so it does not contend with GPU jobs sharing the runner — under a 75-minute cap, and uploads its JUnit XML.

`.github/workflows/gpu-tests.yml` asserts the image is present (`halo:blackwell` by default, never rebuilt by CI), then runs `make test-gpu-core ENV_FILE= AWS_DIR=` — creds-free, `-m "gpu and core"` — and uploads the JUnit XML as an artifact. Once its `pull_request` trigger is enabled a PR run requires **both** a non-draft PR and the `run-ci-gpu` label. It deliberately has no `push` trigger: that would fire the tier on every merge with no label gate; post-merge runs go through `workflow_dispatch`.

The job's `timeout-minutes` is the budget the `core` tier has to fit inside; a new core entry that pushes the tier past it belongs in `full` ([tier composition](../contributing/README.md#tests) owns both numbers).

The full GPU tier (`make test-gpu-full`) and the two inference-server tiers (`make test-gpu-vllm` / `make test-gpu-sglang`, each needing its server already running on a GPU outside `TRAINER_CUDA_DEVICES`) have no workflow — run them by hand.

### Enabling the test tiers (repo admin)

1. Register a self-hosted runner on the GPU box (repo → Settings → Actions → Runners → New self-hosted runner) with labels `self-hosted` and `halo`. Run it as a systemd service under a dedicated non-root user in the `docker` group.
2. Build the image on that box (`make build-blackwell`). CI reuses it and never rebuilds per run; refresh it when the `Dockerfile` or deps change.
3. Repo → Settings → Actions → General: set fork-PR runs to require approval for **all outside collaborators** — the *first-time contributors* setting does not gate returning contributors. This must precede step 4: the CPU tier has no label gate, so per-run approval is its only maintainer opt-in.
4. Uncomment the `push` / `pull_request` triggers in `cpu-tests.yml` and the `pull_request` trigger in `gpu-tests.yml`.
5. Create the `run-ci-gpu` label.

## Contribution gate

Three workflows implement the issue-first gate described in `CONTRIBUTING.md`:

- `.github/workflows/pr-gate.yml` (`pull_request_target`, opened/reopened): a PR whose author is neither a bot, nor a write-access maintainer, nor listed in `.github/APPROVED_CONTRIBUTORS` gets an explanatory comment and is closed. It reads the allowlist via the API from the `allowlist` branch and checks out nothing, so PR code never executes.
- `.github/workflows/approve-contributor.yml`: a maintainer commenting `/approve @username` appends that user to `.github/APPROVED_CONTRIBUTORS`. The gate is the commenter's write access, verified first — the `issue_comment` trigger fires on PR comments too.
- `.github/workflows/approve-merged-contributor.yml`: merging a PR adds its author to the allowlist, so repeat contributors skip the gate.

The allowlist lives on the **`allowlist` branch**, not `main` — GitHub refuses the Actions app as a ruleset bypass actor by design (any collaborator could otherwise push anywhere via a workflow), so the file sits on a branch outside `main`'s ruleset where the workflow token can write it. `pr-gate.yml` reads it from that branch via the API (no checkout at all); the approval workflows commit to it with `createOrUpdateFileContents`, and those API commits arrive GitHub-signed, satisfying the org-wide signed-commit rule. `main` keeps a pointer stub at the same path, and its reviewed-PR rule stays exception-free. On failure (missing branch, permissions) the workflows say so on the issue/PR rather than erroring invisibly; concurrent approvals race on the file sha, and each workflow refetches and retries once. If the branch is ever deleted, recreate it: a single signed commit whose tree holds `.github/APPROVED_CONTRIBUTORS` (one username per line, `#` comments), pushed to `refs/heads/allowlist`.

## Security

The self-hosted runner executes contributor code on your hardware, beside training and secrets. The controls:

- **GPU tier is creds-free but mounts the scratch volume.** `make test-gpu-core ENV_FILE= AWS_DIR=` drops the `.env` (WANDB/HF/AWS keys) and `~/.aws` mounts, but the default `MNT_MOUNT` still bind-mounts all of `HALO_SCRATCH` (default `/mnt`) read-write — HF cache, dataset caches, checkpoints. `HALO_SCRATCH` is the one home for that volume: the bind mount and the in-container `HF_HOME` / `HF_DATASETS_CACHE` / `TMPDIR` / `HALO_DATA_ROOT` all derive from it, so pointing the tier at another disk is one override. Narrowing `MNT_MOUNT` alone is not, since those env vars still resolve under `HALO_SCRATCH`. The label gate is the primary control. Inject `HF_TOKEN` from a repo secret only when a gated model is needed.
- **The CPU tier mounts the HF cache.** `DOCKER_RUN_CPU` bind-mounts `HF_CACHE` (default `$(HALO_SCRATCH)/hf`) read-write and points `HF_HOME` at it, because CPU tests that call `from_pretrained` directly hard-fail when the cache is missing and the Hub is unreachable — a state a self-hosted runner can be in. Tests going through `tests/common/tokenizers.py` skip instead. Override `HF_CACHE=` to run genuinely cache-less and accept those failures.
- **No repository secrets to fork PR code.** The commented-out test triggers are `pull_request`, not `pull_request_target`, so enabling them still keeps repo secrets away from fork PR code (a same-repo branch is a maintainer's, and gets them). Whatever sits on the bind-mounted scratch volume is a separate matter — see the first bullet. `pr-gate.yml` uses `pull_request_target` on purpose and checks out nothing.
- **Label gate (GPU tier).** A GPU PR run requires a maintainer to add `run-ci-gpu` — a per-PR opt-in, not the boundary: a `pull_request` run executes the PR's own copy of the workflow and the Makefile, and the label survives later pushes, so the fork-approval setting below is what keeps unreviewed code off the box.
- **Fork approval (both tiers, and the only gate on the CPU tier).** Require approval for **all outside collaborators** (not just first-time) *before* enabling the test triggers — the CPU tier has no label gate, and `pr-gate` closing an unapproved PR does not stop workflows the same `opened` event already started.

## Repo hygiene

`.github/workflows/stale.yml` (daily, 01:30 UTC): issues idle for 30 days are marked stale and closed 7 days later. A `keep-open` or `help wanted` label, an assignee, or a milestone exempts them; a closed issue is reopened manually. PRs are left alone — `pr-gate.yml` already curates those.

`.github/workflows/branch-cleanup.yml` (weekly, Monday 02:00 UTC): remote branches older than 90 days with no open PR are deleted. The default branch, `allowlist` (the contribution-gate roster lives there), `gh-pages`, and `release-*` branches are never touched.
