# Style charter

This charter holds the rules that apply to Halo's doc pages alone: what goes on a page, how long it
runs, how it cites code and the register of each kind of page. How a sentence is written
is the `voice` skill's subject, for docs and for every other text a person reads. Its working rules are
in [`../voice/SKILL.md`](../voice/SKILL.md). Its catalog, [`../voice/patterns.md`](../voice/patterns.md),
lists each writing pattern with its limit and the banned words. Read both before you edit prose, and run
`skills/voice/scripts/scan.py` on the page when you are done.

## The one rule

Cut to what is load-bearing. A fact is load-bearing if the reader acts differently without it: a
supported or unsupported combination, a default, a measured number with its shape, a command, a known
limit. The rest is padding: the intro that restates the page, the third benchmark table, the
re-derivation of a point made one section up, the path appended to a sentence that does not send the
reader to code. Delete it.

## Three kinds of page

Most of `agent-docs/` is reference pages. The reader of one, a person or an agent, came with a question,
so give the shortest version that still answers it. Most reference pages fit in 150 to 300 lines. One
past about 400 is almost always carrying redundancy or content that belongs on another page.

A guide page is a page of `human-docs/`. It is as concise as a reference page and it talks to the
reader. It says "you", gives each instruction with its reason and says what the mistake looks like when
you make it.

A lecture page teaches a subject to someone new to it. `agent-docs/reference/gpu-training-theory.md` is
Halo's lecture page, written for a newcomer to GPU training whether or not they use Halo. A lecture
explains in full sentences, defines a term in the sentence that first uses it and builds each idea on
the ones before it. It may run long, and the line target for reference pages does not apply to it. The
one rule applies to it all the same. For a newcomer the steps of an explanation are load-bearing, and a
point made twice is not.

A lecture also shows arithmetic the reader can redo and marks each number as measured, vendor
specification or computed. It keeps the sentences about Halo's own choices apart from the general ones,
so a reader who does not use Halo knows which apply to them. Detail that only a Halo user needs stays on
the page that owns it, and the lecture links there.

## Rules for every page

- Write American English: color, behavior, optimize.
- Use the active voice.
- Lead with the answer. Open with the fact the reader came for. Leave out an "Overview" section that
  restates the page, an opening that announces what the section will do and a closing summary that
  repeats the body. A lecture may close with one list of checks if it gives an order the body did not.
- On a reference or guide page, prefer the shorter form: a sentence over a paragraph, a short list over
  a table, one example over three. If a sentence can go without losing a load-bearing fact, it goes.
- Cite sparingly. Anchor a claim to a `halo/` path, config field or class name only when the reader
  needs to find or set that thing. One or two pointers to the owning code per topic is enough. Do not
  append a path to every sentence or list every method of a module, because such a list drifts out of
  date. Name the owning file once and let the reader open it.
- On a reference or guide page, quote one representative measured number with its hardware, model and
  shape, and state the trend in words. One figure and a sentence carry the point better than three
  overlapping benchmark tables. A lecture tables the measurements it teaches from, and says after each
  table what to take from it.
- Use a table for a real matrix: model by mode, GPU state by interpretation, config by result. A
  two-row relationship is a sentence. Never table what a short list says shorter.
- Write plain markdown in `agent-docs/` and `human-docs/`, with no raw HTML tag. A viewer that does not
  render HTML prints the tag as text, and the link check refuses it. For an anchor, link the heading's
  slug. For text you want folded, give it its own section and link that. Put a placeholder such as
  `<split>` in a code span, because GitHub drops a tag it does not know.
- State limits plainly. Unsupported combinations, known bugs and measured results that came out slower
  are load-bearing. Keep each in one line, with the reason.
- Describe the present state. A doc says what the code does now. Leave out dated verification
  ("verified 2026-06-10", "re-confirmed"), change framing ("previously X, now Y", "no longer", "used to",
  "newly enabled", "after the fix") and anecdote ("turned out to be", "early runs passed"), and rewrite
  to the current fact. Keep a date only when it identifies a thing: a version pin, a CVE, a checkpoint
  version. History lives in git.
- Give each fact one home. A fact lives on one authoritative page, and a second page that needs it gets
  a one-line link, never a copy that will drift. A guide or a lecture may quote a measured number from
  the page that owns the run, with its shape and a link to that page. A configuration field's type and
  default are tabled only in `agent-docs/reference/configuration-reference.md`. A feature page explains
  the behavior and links there (`tests/cpu/conventions/test_docs_restate_no_config_fields.py`).
- Make an argument once. On a reference or guide page, keep the unique technical substance and cut the
  re-derivation of a point the linked page already makes. A lecture derives what its reader needs for
  the next step, and repeats a main point only on purpose, in its closing checks.

## Good and bad

A page fails in two ways. Slop is a claim with no anchor, in marketing words. Bloat is true and padded:
an overview, a rule stated twice, a path on every clause.

Slop:

> Halo leverages a powerful, cutting-edge grouped GEMM kernel that
> seamlessly delivers blazing-fast MoE performance.

Bloat:

> ## Overview
>
> The S3 utilities provide functions for DataFrame operations, dataset operations,
> folder operations, file management, nested paths, local caching, and a CLI.
>
> ## Path formats
>
> The toolkit supports three source types… (then a second "Path format rules"
> section restating the same three formats).

Lean:

> Grouped GEMM (`torch.nn.functional.grouped_mm`) batches the per-expert matmuls
> into one kernel launch (SM90+, PyTorch 2.11+). On Qwen3-30B-A3B at `ep2` it runs
> 2.45× faster than the loop, measured on 2× B300 with 8,192-token rows at batch 4.
> The gain shrinks as the batch grows, so measure at batch 4 or more.

The lean passage names the thing, gives one number with its shape and adds the caveat that protects the
reader.
