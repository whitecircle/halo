# Style charter — lean, anti-slop

The Halo docs read like an engineer's note to another engineer: precise,
measured, no marketing — **and no padding**. The goal of every page is the
shortest version that still answers the reader's question. Match the voice of the
leanest existing pages, not the longest. This charter is the gate for every prose
edit.

## The one rule

**Cut to what is load-bearing.** A fact is load-bearing if the reader acts
differently without it: a supported/unsupported combination, a default, a measured
number with its shape, a command, a known limit. Everything else — the intro that
restates the page, the third benchmark table, the re-derivation of a point made
one section up, the path appended to a sentence that does not send the reader to
code — is padding. Delete it.

## Rules

- **American English.** color, behavior, optimize. Not colour/optimise.
- **Active voice, short sentences.** One claim per sentence.
- **Lead with the answer.** No "Overview" section that restates the page, no "In
  this section we will…", no closing summary that repeats the body. Open with the
  fact the reader came for.
- **Brevity is a rule, not a nicety.** Prefer the shorter form everywhere: a
  sentence over a paragraph, a short list over a table, one example over three. If
  a sentence can go without losing a load-bearing fact, it goes. Most reference
  pages fit in 150–300 lines; a page past ~400 is almost always carrying
  redundancy or content that belongs on another page.
- **Cite sparingly — only where it helps the reader act.** Anchor a claim to a
  `src/` path, config field, or class name **only when the reader needs to find or
  set that thing**. One or two pointers to the owning code per topic is enough. Do
  not append a path to every sentence, and do not enumerate every method of a
  module — that "Implementation" dump is noise that drifts out of date. Name the
  owning file once; let the reader open it.
- **One number, not a table of them.** Quote a single representative measured
  number with its hardware/model/shape, and state the trend in words. Do not paste
  three overlapping benchmark tables where one figure plus a sentence carries the
  point.
- **Tables only when they earn their space.** A real matrix — model × mode, GPU
  state × interpretation, config × result — is a table. A two-row relationship is a
  sentence. Never table what a short list says shorter.
- **State limits plainly.** Unsupported combinations, known bugs, and measured
  net-slower results are load-bearing — keep them, in one line, with the reason.
- **Present-state only — no changelog, no story-telling.** A doc says what the code
  does *now*. Ban dated verification ("verified 2026-06-10", "re-confirmed"),
  change framing ("previously X, now Y", "no longer", "used to", "newly enabled",
  "after the fix"), and anecdote ("turned out to be", "early runs passed"). Rewrite
  to the current fact. Keep a date only when it *identifies* a thing (an image tag
  `halo:blackwell`, a version pin, a CVE, a checkpoint version). History
  lives in git, never in `agent-docs/`.
- **One home per fact.** Each fact lives on exactly one authoritative page. A
  second page that needs it gets a one-line link, never a copy that will drift.
- **No deep-dive sprawl.** Keep the unique technical substance; cut the repetition
  and the re-derivation of a point the linked page already makes. A theory page
  makes its argument once, tightly.

## Banned register

Never use: delve, leverage, seamless, robust, powerful, cutting-edge, it's worth
noting, simply. Also banned: marketing adjectives (revolutionary, blazing-fast,
state-of-the-art-as-a-boast), filler hedges ("essentially", "basically", "in order
to" → "to"), and emoji in body prose.

| Banned | Use instead |
|---|---|
| leverage X | use X |
| robust / powerful | (delete, or state the measured property) |
| seamless / cutting-edge | (delete, or name the version) |
| it's worth noting that Y | Y |
| simply / just call Z | call Z |
| in order to | to |

## Good vs bad

Two failure modes: slop (vibe-claims, marketing, passive) and bloat (padding,
redundancy, over-citation). Cut both.

**Slop** — unanchored, banned register, marketing:

> Halo leverages a powerful, cutting-edge grouped GEMM kernel that
> seamlessly delivers blazing-fast MoE performance.

**Bloat** — true, but padded with an overview, a redundant rule restatement, and a
path on every clause:

> ## Overview
> The S3 utilities provide functions for DataFrame operations, dataset operations,
> folder operations, file management, nested paths, local caching, and a CLI.
> ## Path formats
> The toolkit supports three source types… (then a second "Path format rules"
> section restating the same three formats).

**Lean** — the bar: name the thing, give the one number with its shape, add the
caveat that protects the reader:

> Grouped GEMM (`torch.nn.functional.grouped_mm`) batches the per-expert matmuls
> into one kernel launch (SM90+, PyTorch 2.11+). On Qwen3-30B-A3B EP=2 (2× B300,
> seq 8192, batch 4): **2.3×** over the loop. The win shrinks as batch grows —
> measure at batch ≥ 4.

That is the bar: one anchor, one number with its shape, one caveat, nothing else.
