---
name: docs
description: >-
  Use when writing, editing, or reviewing any page under agent-docs/, human-docs/
  or skills/, or when a code change in halo/ needs its owning doc page updated
  (doc actualization). Authors and maintains the Halo doc trees under the style
  charter; the voice skill holds the writing rules and the pattern catalog.
allowed-tools: [Read, Edit, Write, Grep, Glob, Bash]
paths:
  - agent-docs/**
  - human-docs/**
  - skills/**
---

# Docs

The `agent-docs/` tree is plain GitHub-rendered markdown, and each section's `README.md` is its overview
page. Two rules from `CLAUDE.md` govern this work. Docs first: check `agent-docs/` before searching the
code. Keep docs current: a significant `halo/` change updates its owning doc page. This skill does both.

## Workflow

1. Find the owning page. For a changed `halo/` path, look it up in
   [`docs-ownership.md`](docs-ownership.md). For a new topic, place it in the section directory it
   belongs to and add it to that section's `README.md`, so that no page is orphaned.
2. Ground every edit against the code. Read the `halo/` file before you write the claim. When the code
   and the doc disagree, the code wins and the doc gets fixed. `CLAUDE.md` and sibling docs can lag the
   code themselves, so treat them as leads and verify against the `halo/` file. Grounding is about
   correctness and does not mean citing: name the path only where the reader needs to find or set that
   thing (see `style.md`).
3. Apply the charter and the voice rules. [`style.md`](style.md) decides what goes on the page: the
   answer first, load-bearing facts only, sparse citations, a table only where it earns its space, the
   present state, and the register of the page's kind (reference, guide or lecture). The `voice` skill
   ([`../voice/SKILL.md`](../voice/SKILL.md)) decides how the sentences read. Run its scanner on the page,
   `python3.12 skills/voice/scripts/scan.py <page>`, and read the hits.
4. Keep the indexes and cross-links in sync. A new or renamed page is wired into its section
   `README.md`, and a removed page comes out of it. Link a heading by the slug GitHub generates:
   lowercased, punctuation such as `&`, `+` and `/` dropped, each space a hyphen. So `## EP+TP mode`
   becomes `#eptp-mode`, and `## Checkpointing & state dict` becomes `#checkpointing--state-dict`, with a
   double hyphen where a symbol sat between spaces. A repeated heading gets `-1`, `-2`. A `{#custom-id}`
   suffix is not an anchor on GitHub, where it renders as heading text, and a `**bold line**` is not a
   heading. Renaming a heading breaks every link to it, which the link check in step 6 catches.
5. Reconcile a changed fact across the tree. When you change a fact that appears on more than one page
   (a support-matrix cell, a default, a migration such as DeepEP V1 to V2), `grep` `agent-docs/`,
   `human-docs/` and `skills/` for the old wording and fix every copy. Fixing only the owning page leaves
   the siblings stale. `style.md` asks for one home per fact, and this step restores it.
6. Run the link check. `./scripts/docs/check_links.sh` resolves every relative markdown link and heading
   anchor in `agent-docs/`, `human-docs/`, `skills/` and the root markdown, and every repo path and
   dotted first-party name their code cites. It fails on a broken target, a missing anchor, a `{#id}`
   heading, or a cited path or name that does not exist. CI runs the same script in `docs.yml`. It runs
   on the bare host with stdlib Python 3.12 and needs no image. It must be green before the work is done.

## Citing repo paths

Cite a repo path as plain inline code (`` `halo/trainers/distributed_trainer.py` ``, `scripts/...`,
`examples/...`, `tests/...`, `Dockerfile`). Do not hand-write `github.com` URLs. A relative markdown link
is fine where the reader should click through to another doc page, and plain code is right for source
paths. Class and method names (`EPMoELayerBase._compute_experts`), config fields (`ep_size`), flags
(`--expert_parallel_size`) and env vars stay plain code. The charter still asks you to cite sparingly.
The link check resolves each cited path and dotted name, so spell a placeholder in angle brackets
(`examples/<method>/<family>/`) and give a path outside this repo its own root (`<gist>/scripts/…`).

## Reference material

- [`style.md`](style.md) is the style charter: what goes on a doc page, the three kinds of page and
  good and bad examples. Read it before editing prose.
- [`../voice/SKILL.md`](../voice/SKILL.md) and [`../voice/patterns.md`](../voice/patterns.md) are the
  writing rules, the pattern catalog with its limits and the banned words.
- [`docs-ownership.md`](docs-ownership.md) maps each `halo/` area to its owning doc pages. Read it
  before documenting a code change.

## Sources of truth

`agent-docs/` describes intended behavior, and the code under `halo/` is the final authority, so
workflow step 2 is not optional. Never document a number, flag or behavior you have not read in the
`halo/` file. Related skills own their domains: `parallelism`, `checkpoints`, `data`, `optimize`,
`rl-setup` and `add-model` each ground in their code, so keep the doc page consistent with the skill
that owns the feature.
