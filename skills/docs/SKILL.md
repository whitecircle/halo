---
name: docs
description: >-
  Use when writing, editing, or reviewing any page under agent-docs/, or
  when a code change in src/ needs its owning doc page updated (doc actualization).
  Authors and maintains the Halo agent-docs/ tree and enforces the anti-AI-slop
  style charter.
allowed-tools: [Read, Edit, Write, Grep, Glob, Bash]
paths:
  - agent-docs/**
  - human-docs/**
---

# docs — author and maintain the Halo docs tree

The `agent-docs/` tree is plain GitHub-rendered markdown (each section's `README.md` is its
overview page). Two rules from `CLAUDE.md` govern this work: **Docs first** (docs are the source
of truth for how the code behaves) and **Doc maintenance** (a significant `src/` change must
update the owning doc page). This skill does both, to the house voice.

## Workflow

1. **Find the owning page.** For a changed `src/` path, look it up in
   [`docs-ownership.md`](docs-ownership.md). For a new topic, place it in the
   section directory it belongs to and add it to that section's `README.md` — do
   not orphan a page.
2. **Ground every edit against the code.** Read the actual `src/` file before you
   write the claim; when the code and the doc disagree, the code wins — fix the
   doc. `CLAUDE.md` and sibling docs can themselves lag the code (e.g. a
   mid-migration WIP commit) — treat them as leads, not authority, and verify
   against the `src/` file. Grounding is about correctness, not citation: cite the
   path only where the reader needs to find or set that thing (see `style.md`), not
   on every sentence.
3. **Apply the style charter.** Follow [`style.md`](style.md) on every edit: lead
   with the answer, cut to load-bearing facts, cite sparingly, tables only when
   they earn their space, present-state only, banned register removed.
4. **Keep the indexes and cross-links in sync.** A new or renamed page must be
   wired into its section `README.md` **and** the Documentation index in `CLAUDE.md`;
   a removed page comes out of both. Cross-links stay valid — GitHub slugifies a
   heading by lowercasing, dropping punctuation like `&`/`+`, and collapsing
   separators, so `## EP+TP mode` → `#eptp-mode` and
   `## Checkpointing & state dict` → `#checkpointing--state-dict` on GitHub
   (note the double hyphen where a symbol sat between spaces), and a
   `**bold line**` is not a heading and has no anchor. Renaming or trimming a
   heading breaks every page that links to it — the link check (step 6) catches
   broken paths, but anchors need a by-hand check.
5. **Reconcile a changed fact across the tree.** When you change a fact that
   appears on more than one page (a support-matrix cell, a default, a migration
   like DeepEP V1→V2), `grep` the whole `agent-docs/` tree for the old wording and fix
   every copy — fixing only the owning page leaves the siblings stale. This is the
   `style.md` "one home per fact" rule enforced after the fact.
6. **Run the link check.** `./scripts/docs/check_links.sh` resolves every relative
   markdown link in `agent-docs/`, `human-docs/`, `skills/` and the root markdown
   and fails on a broken target (CI runs the same script in `docs.yml`). It runs
   on the bare host — no image needed. It must be green before the work is done.

## Citing repo paths

Cite a repo path as plain inline code (`` `src/trainers/mixins/base.py` ``,
`scripts/...`, `examples/...`, `tests/...`, `Dockerfile`, …). Do **not**
hand-write `github.com` URLs — a relative markdown link is fine where the reader
should click through to another doc page, and plain code is right for source
paths. Class/method names (`EPMoELayerBase._compute_experts`), config fields
(`ep_size`), flags (`--expert_parallel_size`), and env vars stay plain code.
This does not change the charter: still cite sparingly.

## Reference material

- [`style.md`](style.md) — the anti-AI-slop style charter (voice, banned
  register, good-vs-bad examples). Read it before editing prose.
- [`docs-ownership.md`](docs-ownership.md) — the `src/` area → owning doc
  page(s) map. Read it before documenting a code change.

## Sources of truth

This skill's whole premise: `agent-docs/` describes intended behavior, but the **code under `src/` is the
ultimate authority** — workflow step 2 (ground every edit against the code; when code and doc disagree,
the code wins) is non-negotiable. Never document a number, flag, or behavior you have not read in the
`src/` file. Related skills own their domains — `parallelism`, `checkpoints`, `data`, `optimize`,
`rl-setup`, `add-model` each ground in their code; keep the doc page consistent with the skill that
owns the feature.
