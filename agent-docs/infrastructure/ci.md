# Continuous Integration

Halo's CI is GitHub Actions (`.github/workflows/`), split into tiers by where the runner lives, plus the contribution gate.

| Tier | Runner | Workflows | Triggers | Status |
|------|--------|-----------|----------|--------|
| Hosted | GitHub `ubuntu-latest` | `.github/workflows/lint.yml`, `.github/workflows/docs.yml`, `.github/workflows/cpu-tests-hosted.yml` | every PR + push `main` (CPU tests: non-draft PRs) | active |
| Self-hosted | GPU box `[self-hosted, halo]` | `.github/workflows/cpu-tests.yml`, `.github/workflows/gpu-tests.yml` | `workflow_dispatch` | dispatch-only |

The lint and docs jobs need no image. The CPU tests import torch, so both CPU workflows run them inside the image: the hosted one pulls the published image, the self-hosted one uses the image built on the box. GPU tests need Blackwell (SM100) or Hopper (SM90) for FA3/FA4 + DeepEP, which no hosted runner provides. CodeQL (the default code-scanning setup) and the GitGuardian app also check every PR; neither lives in `.github/workflows/`.

## Hosted tier

`.github/workflows/lint.yml`: `ruff format --check` plus the full pyproject rule set (including `PLC0415`, the no-inline-imports house rule) block the PR, and `actionlint` checks the workflow files themselves. Self-hosted runner labels are declared in `.github/actionlint.yaml` so actionlint does not flag them as unknown.

`.github/workflows/docs.yml`: `scripts/docs/check_links.sh` blocks on any broken relative link across `agent-docs/`, `human-docs/`, `skills/` and the root markdown. `markdownlint` covers the same doc trees.

A `diagrams` job re-runs `scripts/diagrams/` in a `python:3.12-slim` container at `uv.lock`'s matplotlib pin and byte-compares the result against the committed figures under `agent-docs/assets/`, so a generator edited without re-committing its figure goes red — run `make diagrams` and commit.

`diagrams`, `markdownlint` and the CPU tests are advisory: blocking is enforced by the `main` ruleset's **required status checks** (`ruff`, `actionlint`, `build`), and a red check outside that set still lets a PR merge.

`.pre-commit-config.yaml` mirrors these gates on the host at commit time — the same pinned ruff (lint + format), plus `nbstripout`, whitespace fixers and a 1 MB file-size cap. It is optional: `pipx install pre-commit`, then `pre-commit install`.

### CPU tests

`.github/workflows/cpu-tests-hosted.yml` splits the CPU tier across four `ubuntu-latest` runners (4 vCPU, 16 GB RAM) on every push to `main` and every non-draft PR; a draft runs once it is marked ready for review. Each shard job:

1. Moves Docker's storage to the runner's `/mnt` scratch disk, and fails unless 45 GB are free there. The image extracts to 30.3 GB from a 14.2 GB download, and the root disk keeps about 20 GB free.
2. Pulls `public.ecr.aws/whitecircle/halo:blackwell` anonymously.
3. Restores the HF cache from the Actions cache, keyed on the seed list (`python -m tests.common.hub_seed --list`) and `tests/common/hub_seed.py`. On a miss it runs `make seed-hf-cache`, which fetches the seed from the Hub and names any repo it cannot fetch, and saves the cache before testing.
4. Runs `make test-cpu` on its quarter of the tests (`pytest-shard`, split by test id) with four `pytest-xdist` workers, offline (`HF_HUB_OFFLINE=1`) and with `HALO_TEST_REQUIRE_HUB_CACHE=1`, so a test that reads a Hub repo outside the seed fails instead of skipping. The target mounts the checkout over the source baked into the image.

The seed is every Hub id an `examples/` config trains, every checkpoint constant in `tests/common/models.py`, and the revisions its `PINNED_REVISIONS` pins. Of each it holds the configs, tokenizers, chat templates and remote-code `*.py` files, never weights (about 0.6 GB). A test that loads a new repo names it in `tests/common/models.py`; the tier then fetches it.

The `cpu-tier` job fails unless every shard passed, including when the run was cancelled, so a required-check rule can name that one job. A shard's tests take 3–4.5 minutes pinned to 4 cores and 16 GB of a B300 node, with container memory (page cache included) peaking under 12 GB; the runner's pull and extract come on top. Reproduce one shard:

```bash
make seed-hf-cache
make test-cpu EXTRA_DOCKER_ENV="-e HF_HUB_OFFLINE=1 -e HALO_TEST_REQUIRE_HUB_CACHE=1" \
    PYTEST_ARGS="-n 4 --num-shards 4 --shard-id 0"
```

Limits:

- **It tests the released dependencies.** A PR that changes `pyproject.toml`, `uv.lock` or a `Dockerfile` also needs `make test-cpu` on a rebuilt image, locally or on the self-hosted tier.
- **Some checks skip under a config-only seed.** Gated repos (`GATED_REPOS` in `tests/common/hub_seed.py`) cannot be fetched anonymously, so their cases skip. The Liger coverage checks for the remote-code Bailing V2/V3 norms skip too: they need the family's modeling module imported, which a config load never does.
- **Anonymous pull quota.** AWS limits anonymous ECR Public pulls to 500 GB a month per source IP, and GitHub-hosted runners share IPs; each shard pulls 14.2 GB. A pull past the limit fails the `Pull the image` step. Re-run the job; a registry without a per-IP quota (such as a GHCR mirror) would remove the limit, and none is configured.

## Self-hosted tier

`.github/workflows/cpu-tests.yml` asserts the image is present, then runs `make test-cpu` — the image without `--gpus`, so it does not contend with GPU jobs sharing the runner — in one pytest process (about 1.5 hours) under a 120-minute cap, and uploads its JUnit XML. It tests the image built on the box, so it covers a dependency change the hosted tier cannot.

`.github/workflows/gpu-tests.yml` asserts the image is present (`halo:blackwell` by default, never rebuilt by CI), then runs `make test-gpu-core ENV_FILE= AWS_DIR=` — creds-free, `-m "gpu and core"` — and uploads the JUnit XML as an artifact. Once its `pull_request` trigger is enabled a PR run requires **both** a non-draft PR and the `run-ci-gpu` label. It deliberately has no `push` trigger: that would fire the tier on every merge with no label gate; post-merge runs go through `workflow_dispatch`.

The job's `timeout-minutes` is the budget the `core` tier has to fit inside; a new core entry that pushes the tier past it belongs in `full` ([tier composition](../contributing/README.md#tests) owns both numbers).

The full GPU tier (`make test-gpu-full`) and the two inference-server tiers (`make test-gpu-vllm` / `make test-gpu-sglang`, each needing its server already running on a GPU outside `TRAINER_CUDA_DEVICES`) have no workflow — run them by hand.

A per-family pass of either server tier serves the family's checkpoint and points the async GRPO wrapper at it: `HALO_TEST_ENV_GRPO_MODEL` for the vLLM tier, `HALO_TEST_ENV_GRPO_SGLANG_MODEL` for the SGLang tier. The rows move the served policy with an expert-only perturbation as well as a dense one.

### Enabling the self-hosted tiers (repo admin)

1. Register a self-hosted runner on the GPU box (repo → Settings → Actions → Runners → New self-hosted runner) with labels `self-hosted` and `halo`. Use runner version 2.327.1 or newer: the pinned `actions/checkout` and `actions/upload-artifact` run on Node 24. Run it as a systemd service under a dedicated non-root user in the `docker` group.
2. Build the image on that box (`make build-blackwell`). CI reuses it and never rebuilds per run; refresh it when the `Dockerfile` or deps change.
3. Repo → Settings → Actions → General: set fork-PR runs to require approval for **all outside collaborators** — the *first-time contributors* setting does not gate returning contributors. This must precede step 4: the self-hosted CPU tier has no label gate, so per-run approval is its only maintainer opt-in.
4. Uncomment the `push` / `pull_request` triggers in `cpu-tests.yml` and the `pull_request` trigger in `gpu-tests.yml`.
5. Create the `run-ci-gpu` label.

## Contribution gate

Three workflows implement the issue-first gate described in `CONTRIBUTING.md`:

- `.github/workflows/pr-gate.yml` (`pull_request_target`, opened/reopened): a PR whose author is neither a bot, nor a write-access maintainer, nor listed in `.github/APPROVED_CONTRIBUTORS` gets an explanatory comment and is closed. It reads the allowlist via the API from the `allowlist` branch and checks out nothing, so PR code never executes.
- `.github/workflows/approve-contributor.yml`: a maintainer commenting `/approve @username` appends that user to `.github/APPROVED_CONTRIBUTORS` and assigns them to the issue (`stale.yml` exempts assigned issues; a user GitHub refuses to assign — one who never commented on the issue — is reported with a `keep-open` hint). Several `/approve @handle` lines in one comment approve each named user. The gate is the commenter's write access, verified first — the `issue_comment` trigger fires on PR comments too, and a permission check that errors is reported on the issue and fails the run rather than approving nobody in silence. Bot comments are skipped.
- `.github/workflows/approve-merged-contributor.yml`: merging a PR adds its author to the allowlist, so repeat contributors skip the gate. Authors with write access are not listed — they pass `pr-gate.yml` on that access.

The allowlist lives on the **`allowlist` branch**, not `main` — GitHub refuses the Actions app as a ruleset bypass actor by design (any collaborator could otherwise push anywhere via a workflow), so the file sits on a branch outside `main`'s ruleset where the workflow token can write it. `pr-gate.yml` reads it from that branch via the API (no checkout at all); the approval workflows commit to it with `createOrUpdateFileContents`, and those API commits arrive GitHub-signed, satisfying the org-wide signed-commit rule. `main` keeps a pointer stub at the same path, and its reviewed-PR rule stays exception-free. On failure (missing branch, permissions) the workflows say so on the issue/PR rather than erroring invisibly; concurrent approvals race on the file sha, and each workflow refetches and retries once. `pr-gate.yml` fails open on a transient API error (a maintainer's own PR must never be auto-closed by a 500) but treats a missing branch or file as an empty list and gates every outsider, so the branch must survive: a repository ruleset on `refs/heads/allowlist` with the `deletion` and `non_fast_forward` rules (repo admin) blocks a stray delete or force-push while the workflows' fast-forward API commits still land. If the branch is ever deleted, recreate it: a single signed commit whose tree holds `.github/APPROVED_CONTRIBUTORS` (one username per line, `#` comments), pushed to `refs/heads/allowlist`.

## Security

The hosted CPU tier runs PR code on a GitHub-hosted VM discarded after the job, with a `contents: read` token, no secrets and a checkout that does not persist the token; it needs no gate of its own.

The self-hosted runner executes contributor code on your hardware, beside training and secrets. The controls:

- **GPU tier is creds-free but mounts the scratch volume.** `make test-gpu-core ENV_FILE= AWS_DIR=` drops the `.env` (WANDB/HF/AWS keys) and `~/.aws` mounts, but the default `MNT_MOUNT` still bind-mounts all of `HALO_SCRATCH` (default `/mnt`) read-write: HF cache, dataset caches, checkpoints. Fork-PR approval is the boundary; the label only picks which approved PRs run.

    `HALO_SCRATCH` is the one home for that volume: the bind mount and the in-container `HF_HOME` / `HF_DATASETS_CACHE` / `TMPDIR` / `HALO_DATA_ROOT` all derive from it, so pointing the tier at another disk is one override. Narrowing `MNT_MOUNT` alone is not, since those env vars still resolve under `HALO_SCRATCH`. Inject `HF_TOKEN` from a repo secret only when a gated model is needed.

- **The CPU tier mounts the host's HF cache.** `DOCKER_RUN_CPU` bind-mounts `HF_CACHE` (default `$(HALO_SCRATCH)/hf`) read-write and points `HF_HOME` at it, so PR code can write into the cache every later job reads. CPU tests that call `from_pretrained` directly hard-fail when the cache is missing and the Hub is unreachable — a state a self-hosted runner can be in.

    Tests going through `tests/common/tokenizers.py` skip instead. `HF_CACHE=` runs cache-less: those tests skip, and the guards that refuse an all-skipped file fail.

- **No repository secrets to fork PR code.** Every PR trigger of a test workflow is `pull_request`, not `pull_request_target` — the hosted CPU tier's and the commented-out self-hosted ones — so fork PR code never gets repo secrets (a same-repo branch is a maintainer's, and gets them). Whatever sits on the bind-mounted scratch volume is a separate matter — see the first bullet. `pr-gate.yml` uses `pull_request_target` on purpose and checks out nothing.
- **Label gate (GPU tier).** A GPU PR run requires a maintainer to add `run-ci-gpu` — a per-PR opt-in, not the boundary: a `pull_request` run executes the PR's own copy of the workflow and the Makefile, and the label survives later pushes, so the fork-approval setting below is what keeps unreviewed code off the box.
- **Fork approval (both self-hosted tiers, and the only gate on the self-hosted CPU tier).** Require approval for **all outside collaborators** (not just first-time) *before* enabling the self-hosted test triggers — that CPU tier has no label gate, and `pr-gate` closing an unapproved PR does not stop workflows the same `opened` event already started.

## Repo hygiene

`.github/workflows/stale.yml` (daily, 01:30 UTC): issues idle for 30 days are marked stale and closed 7 days later. A `keep-open` or `help wanted` label, an assignee, or a milestone exempts them — `/approve` assigns the contributor, so an accepted issue stays open while it is worked; a closed issue is reopened manually. PRs are left alone — `pr-gate.yml` already curates those.

`.github/workflows/branch-cleanup.yml` (weekly, Monday 02:00 UTC): remote branches older than 90 days with no open PR are deleted. The default branch, `allowlist` (the contribution-gate roster lives there), `gh-pages`, and `release-*` branches are never touched; a branch a ruleset shields from deletion fails its delete call and is logged.

Every deleted tip SHA is logged (recoverable by SHA), and `workflow_dispatch` defaults to a dry run that only prints the kill list.
