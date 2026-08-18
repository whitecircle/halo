---
name: docs
description: >-
  Use when writing, editing, or reviewing any page under agent-docs/ or mkdocs.yml, or
  when a code change in src/ needs its owning doc page updated (doc actualization).
  Authors and maintains the Halo agent-docs/ tree and enforces the anti-AI-slop
  style charter.
allowed-tools: [Read, Edit, Write, Grep, Glob, Bash]
paths:
  - agent-docs/**
  - human-docs/**
  - mkdocs.yml
---

# docs — author and maintain the Halo docs tree

The `agent-docs/` tree is MkDocs Material (`mkdocs.yml`, nav-driven). Two rules from
`CLAUDE.md` govern this work: **Docs first** (docs are the source of truth for
how the code behaves) and **Doc maintenance** (a significant `src/` change must
update the owning doc page). This skill does both, to the house voice.

## Workflow

1. **Find the owning page.** For a changed `src/` path, look it up in
   [`docs-ownership.md`](docs-ownership.md). For a new topic, place it in the
   nav section it belongs to (`mkdocs.yml`) — do not orphan a page.
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
   wired into `mkdocs.yml` `nav:` **and** the Documentation index in `CLAUDE.md`;
   a removed page comes out of both. Cross-links stay valid — MkDocs slugifies a
   heading by lowercasing, **dropping** `&`/`+`, and collapsing separators, so
   `## EP+TP mode` → `#eptp-mode` and `## Checkpointing & state dict` →
   `#checkpointing-state-dict` (not `#ep-tp-mode` / `#checkpointing--state-dict`),
   and a `**bold line**` is not a heading and has no anchor. When unsure of a slug,
   read the real id from the built HTML
   (`grep -o 'id="[^"]*"' site/<page>/index.html`). Renaming or trimming a heading
   breaks every page that links to it — `--strict` (step 5) catches these.
5. **Reconcile a changed fact across the tree.** When you change a fact that
   appears on more than one page (a support-matrix cell, a default, a migration
   like DeepEP V1→V2), `grep` the whole `agent-docs/` tree for the old wording and fix
   every copy — fixing only the owning page leaves the siblings stale. This is the
   `style.md` "one home per fact" rule enforced after the fact.
6. **Build with `--strict`.** `mkdocs build --strict` fails on broken nav and
   broken internal links. There is **no host Python** — run it inside the
   Docker image (see below). The build must be green before the work is done.

## Running the build (no host Python)

The host has no usable Python; `mkdocs` lives only inside the prebuilt images.
The build needs **no GPU** — omit `--gpus` so it also runs on a GPU-less or
fabric-down host (the image is uv-built, so `mkdocs` is on `PATH` — call it
directly, no prefix):

```bash
docker run --rm \
  -v $(pwd):/workspace -w /workspace \
  halo:blackwell \
  bash -lc "mkdocs build --strict"
```

`mkdocs --strict` catches nav and link breakage, **not** prose style or
spelling — that is the charter's job, applied by hand on every touched page. To
resolve an anchor it disputes, build once without `--strict` and grep the
generated `site/<page>/index.html` for the real heading `id`.

## Source links are automatic — never hand-write a GitHub URL

A build hook (`mkdocs_hooks/github_links.py`) turns any inline-code **repo path**
into a link to that file/dir on GitHub, displayed by its basename: write
`` `src/trainers/mixins/base.py` `` and the site renders `base.py ↗` linking to
`blob/main/src/trainers/mixins/base.py`; `` `src/callbacks/` `` → `callbacks/ ↗`
(tree); a `:NN` suffix (`` `src/foo.py:42` ``) becomes an `#L42` anchor. So:

- **Cite a path as plain inline code** (`` `src/...` ``, `scripts/...`,
  `examples/...`, `tests/...`, `Dockerfile`, …). Do **not** write a
  Markdown link or a `github.com` URL by hand — the hook owns the URL, branch
  (`main`), and basename display, in one place.
- The path **must exist on disk** to link. A path-shaped span that does not
  resolve stays plain code and is logged at INFO during the build — a free
  stale-reference check. Placeholders (`...<family>...`, `my_first_sft.yaml`,
  `model-XXXXX-of-...`) are intentionally left unlinked.
- Class/method (`EPMoELayerBase._compute_experts`), config fields (`ep_size`),
  flags (`--expert_parallel_size`), and env vars stay plain code — only real
  paths link. This does not change the charter: still cite sparingly.

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
