# Continuous Integration

Halo's CI is GitHub Actions (`.github/workflows/`), in two tiers by where the runner lives, plus the
contribution gate.

| Tier | Runner | Workflows | Triggers | Status |
|------|--------|-----------|----------|--------|
| Hosted | GitHub `ubuntu-latest` | `.github/workflows/lint.yml`, `.github/workflows/docs.yml`, `.github/workflows/sensitive-label.yml` | every PR + push `main` (`sensitive-label`: PR only) | active |
| Self-hosted | GPU box `[self-hosted, halo]` | `.github/workflows/cpu-tests.yml`, `.github/workflows/gpu-tests.yml` | `workflow_dispatch` | dispatch-only |

Hosted jobs need no image. The test tiers import torch and run inside the prebuilt image, built on the
self-hosted box and never pulled or rebuilt by CI; GPU tests also need Blackwell (SM100) or Hopper
(SM90) for FA3/FA4 and DeepEP, which no hosted runner provides. CodeQL (the default code-scanning
setup), GitGuardian and CodeRabbit report on every PR from outside `.github/workflows/`.

## Hosted tier

- **`lint.yml`.** `ruff format --check` plus the full pyproject rule set (`PLC0415`, the no-inline-imports
  rule, included), and `actionlint` over the workflows (self-hosted labels declared in
  `.github/actionlint.yaml`). Its `conventions` job runs the convention tests it lists, the ones that
  read the tree without importing it, on Python 3.12 with pytest alone;
  `tests/cpu/conventions/test_hosted_conventions_are_stdlib.py` fails, in the image too, when a listed
  file, a package `__init__.py` or conftest pytest runs for it, or a file one of them imports or loads
  reaches torch, `halo` or any other non-stdlib import. `.github/scripts/locked_version.py` reads each
  job's tool pin (ruff, pytest, matplotlib) out of `uv.lock`.
- **`docs.yml`.** `scripts/docs/check_links.sh` blocks on a broken relative link or heading anchor across
  `agent-docs/`, `human-docs/`, `skills/` and the root markdown (an anchor must be GitHub's slug, so a
  `{#id}` heading fails), and resolves their code: a repo-root path must exist, and a dotted first-party
  name must name a module and a name it binds, read off the source. Its `diagrams` job re-runs
  `scripts/diagrams/` in `python:3.12-slim` at `uv.lock`'s matplotlib and byte-compares the result with
  the committed figures under `agent-docs/assets/`: run `make diagrams` and commit after editing a
  generator.
- **`sensitive-label.yml`.** A PR whose own commits change a path listed in `.github/sensitive-paths` (a
  moved file counts under its old path) fails until it carries the `sensitive` label, which a maintainer
  adds once the change is approved as `CLAUDE.md` requires; a label change re-runs it. It runs on
  `pull_request_target`, so the workflow, the check (`.github/scripts/check_sensitive_label.py`) and the
  list are the base branch's, and a PR that edits any of them is still judged by the base's copy.
  `tests/cpu/conventions/test_sensitive_label.py` also fails when a listed glob matches no file.

Blocking is the `main` ruleset's **required status checks** (`ruff`, `actionlint`, `build`); a red check
outside that set, `diagrams` included, still lets a PR merge. `conventions` and `sensitive-label` block
once a repo admin adds them to the set and creates the `sensitive` label.

`.pre-commit-config.yaml` mirrors the ruff gates on the host at commit time, plus whitespace fixers and
a 1 MB file-size cap (optional: `pipx install pre-commit`, then `pre-commit install`).

## Self-hosted tier

`cpu-tests.yml` asserts the image is present and runs `make test-cpu` (the image without `--gpus`, so it
does not contend with GPU jobs on the runner) in one pytest process, about 1.5 hours under a 120-minute
cap, uploading its JUnit XML.

`gpu-tests.yml` asserts the image (`halo:blackwell` by default) and runs
`make test-gpu-core ENV_FILE= AWS_DIR=`: creds-free, `-m "gpu and core"`, JUnit XML as an artifact. Once
its `pull_request` trigger is enabled a PR run needs **both** a non-draft PR and the `run-ci-gpu` label;
it has no `push` trigger, which would fire the tier on every merge with no label gate, so post-merge
runs go through `workflow_dispatch`. Its `timeout-minutes` (360) is the budget the `core` tier must fit
in; a new core entry that pushes the tier past it belongs in `full`
([tier composition](../contributing/README.md#tests)).

The full GPU tier (`make test-gpu-full`) and the two server tiers (`make test-gpu-vllm` /
`make test-gpu-sglang`, each needing its server running on a GPU outside `TRAINER_CUDA_DEVICES`) have no
workflow: run them by hand. A per-family pass of a server tier serves the family's checkpoint and
points the async GRPO rows at it (`HALO_TEST_ENV_GRPO_MODEL` for vLLM, `HALO_TEST_ENV_GRPO_SGLANG_MODEL`
for SGLang); the rows move the served policy with an expert-only perturbation as well as a dense one.

### Hub seed

`make seed-hf-cache` (`tests/common/hub_seed.py`) fetches what the CPU tier reads from the Hub into
`HF_CACHE`: for every Hub id an `examples/` config trains, every checkpoint constant in
`tests/common/models.py` and the revisions its `PINNED_REVISIONS` pins, the configs, tokenizers, chat
templates and remote-code `*.py` files, never weights (about 0.6 GB). It names any repo it cannot
fetch. Over a seeded cache the tier runs offline, and `HALO_TEST_REQUIRE_HUB_CACHE=1` turns a Hub miss
into a failure instead of a skip:

```bash
make seed-hf-cache
make test-cpu EXTRA_DOCKER_ENV="-e HF_HUB_OFFLINE=1 -e HALO_TEST_REQUIRE_HUB_CACHE=1" PYTEST_ARGS="-n 8"
```

Gated repos (`GATED_REPOS` in `tests/common/hub_seed.py`) cannot be fetched anonymously, so their cases
still skip, as do the Liger coverage checks for the remote-code Bailing V2/V3 norms (they need the
family's modeling module imported, which a config load never does). A test that loads a new repo names
it in `tests/common/models.py` so the seed carries it.

### Enabling the test tiers (repo admin)

1. Register a self-hosted runner on the GPU box (repo → Settings → Actions → Runners) with labels
   `self-hosted` and `halo`, version 2.327.1 or newer (the pinned actions run on Node 24), as a systemd
   service under a dedicated non-root user in the `docker` group.
2. Build the image on that box (`make build-blackwell`); CI reuses it, so refresh it when the
   `Dockerfile` or the dependencies change.
3. Repo → Settings → Actions → General: require approval of fork-PR runs for **all outside
   collaborators** (the *first-time contributors* setting does not gate returning ones). This must
   precede step 4: the CPU tier has no label gate, so per-run approval is its only opt-in.
4. Uncomment the `push` / `pull_request` triggers in `cpu-tests.yml` and the `pull_request` trigger in
   `gpu-tests.yml`.
5. Create the `run-ci-gpu` label.

## Contribution gate

Three workflows implement the issue-first gate [`CONTRIBUTING.md`](../../CONTRIBUTING.md#the-gate-read-first)
states:

- `.github/workflows/pr-gate.yml` (`pull_request_target`, opened/reopened) comments on and closes a PR
  whose author is neither a bot, nor a write-access maintainer, nor listed in
  `.github/APPROVED_CONTRIBUTORS`. It reads the allowlist through the API and checks out nothing, so PR
  code never runs.
- `.github/workflows/approve-contributor.yml`: a maintainer's `/approve @username` comment (several lines
  approve several users) appends each user to the allowlist and assigns them to the issue (`stale.yml`
  exempts assigned issues; a user GitHub refuses to assign, one who never commented there, is reported
  with a `keep-open` hint). The commenter's write access is verified first, since `issue_comment` fires
  on PR comments too; a permission check that errors fails the run on the issue rather than approving
  nobody in silence. Bot comments are skipped.
- `.github/workflows/approve-merged-contributor.yml` adds a merged PR's author to the allowlist, so
  repeat contributors skip the gate (write-access authors pass on that access and are not listed).

The org's signed-commit ruleset covers every branch with no bypass actor, so a PR merges only once every
commit on its head, a fork's included, is signed.

The allowlist lives on the **`allowlist` branch**, outside `main`'s ruleset, because GitHub refuses the
Actions app as a ruleset bypass actor. The approval workflows commit to it through
`createOrUpdateFileContents`, whose commits arrive GitHub-signed; `main` keeps a pointer stub at the
same path. Failures (missing branch, permissions) are reported on the issue or PR, and concurrent
approvals refetch and retry once on a file-sha race. `pr-gate.yml` fails open on a transient API error
(a maintainer's PR must never be auto-closed by a 500) but treats a missing branch or file as an empty
list and gates every outsider, so protect the branch with a repository ruleset on
`refs/heads/allowlist` carrying the `deletion` and `non_fast_forward` rules. If it is ever deleted,
recreate it as a single signed commit whose tree holds `.github/APPROVED_CONTRIBUTORS` (one username per
line, `#` comments).

## Security

The self-hosted runner executes contributor code on your hardware, beside training and secrets:

- **The GPU tier is creds-free but mounts the scratch volume.** `make test-gpu-core ENV_FILE= AWS_DIR=`
  drops the `.env` and `~/.aws` mounts, but `MNT_MOUNT` still bind-mounts all of `HALO_SCRATCH`
  read-write: HF cache, dataset caches, checkpoints. Fork-PR approval is the boundary; the label only
  picks which approved PRs run. The mount and the in-container `HF_HOME` / `HF_DATASETS_CACHE` /
  `TMPDIR` / `HALO_DATA_ROOT` all derive from `HALO_SCRATCH`, so export it in the runner's environment:
  unset, it resolves to the local mount with the most free space, which can move the seeded HF cache
  between runs. Inject `HF_TOKEN` from a repo secret only when a gated model is needed.
- **The CPU tier mounts the host's HF cache.** `DOCKER_RUN_CPU` bind-mounts `HF_CACHE` (default
  `$(HALO_SCRATCH)/hf`) read-write, so PR code can write into the cache every later job reads. CPU tests
  that call `from_pretrained` directly hard-fail when the cache is missing and the Hub is unreachable;
  tests going through `tests/common/tokenizers.py` skip. `HF_CACHE=` runs cache-less: those tests skip,
  and the guards that refuse an all-skipped file fail.
- **No repository secrets to fork code.** The test triggers are `pull_request`, not
  `pull_request_target`, so repo secrets stay away from fork PR code (a same-repo branch is a
  maintainer's and gets them). `pr-gate.yml` uses `pull_request_target` and checks out nothing;
  `sensitive-label.yml` uses it so a PR cannot edit its own gate, checks out the base branch, and only
  fetches and diffs the PR's commits with a read-only token.
- **The label gate is not the boundary.** A `pull_request` run executes the PR's own copy of the workflow
  and the Makefile, and the label survives later pushes, so fork approval is what keeps unreviewed code
  off the box. Require it for **all outside collaborators** *before* enabling the triggers: `pr-gate`
  closing an unapproved PR does not stop workflows the same `opened` event already started.

## Repo hygiene

`.github/workflows/stale.yml` (daily, 01:30 UTC) marks issues idle for 30 days stale and closes them 7
days later. A `keep-open` or `help wanted` label, an assignee or a milestone exempts an issue, so an
accepted issue (`/approve` assigns its contributor) stays open while it is worked. PRs are left alone.

`.github/workflows/branch-cleanup.yml` (weekly, Monday 02:00 UTC) deletes remote branches older than 90
days with no open PR, never the default branch, `allowlist` or `release/*`; a branch a ruleset shields
fails its delete and is logged. Every deleted tip SHA is logged (recoverable by SHA), and
`workflow_dispatch` defaults to a dry run that prints the list.
