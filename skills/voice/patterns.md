# Writing patterns and their limits

The catalog behind the [`voice`](SKILL.md) skill. Each entry says what the pattern is, shows it in one of
our own texts where we have an example, gives what a person writes in its place and states the limit.

## How to read an entry

A strong tell is rare in careful writing, so fix each one you find. A density pattern is normal English
that becomes a mark only when it repeats or always sits in the same place, so it has a limit per 1,000
words of prose. Some patterns cannot be found by an expression, and you find them by reading aloud.

The limits are the ones `skills/voice/scripts/scan.py` applies, and the rates beside them are what it
counts on prose, with tables left out. A count marked "by reading" is one a person made, and it is the
larger one where the expression misses cases. A text under 500 words gets the allowance of 500.

The limits are fitted to five of our own texts: the 42 pages of `human-docs/`, the README, one draft of
release notes, 16 PR comments and the theory page. "The theory page" is
`agent-docs/reference/gpu-training-theory.md` as it read at commit `b83a5761f`, before it was rewritten as
a lecture. It is over the limit on 24 rows, and 29 of the 42 `human-docs/` pages are over on none. Treat
a limit as a prompt to read the hits.

Quoted lines are from those texts as they read when the patterns were counted, and some have been fixed
since.

## Sentence constructions

### Em dash

Density, 2 per 1,000 words, and none in a heading or between a list label and its text. The pattern is a
dash that joins two clauses where a comma, parentheses or a new sentence would do. `human-docs/`, the
release notes and the PR comments carry none in 29,400 words of prose, and the theory page had 80. From
`agent-docs/reference/why-this-framework.md`:

> Built on HuggingFace Transformers, Accelerate, and TRL — every distributed trainer subclasses a TRL,
> Transformers, or SentenceTransformers trainer.

Instead: Halo is built on HuggingFace Transformers, Accelerate and TRL. Every distributed trainer
subclasses a TRL, Transformers or SentenceTransformers trainer.

Fine as a rare aside. A colon in the dash's place is the same construction.

### The cause shown as a reveal

Density, 0.5 per 1,000 words and at most 3 on a page, for `That is why`, `This is why`, `which is why`
and `The corollary`. The sentence states a mechanism and then presents a design choice as its discovery.
The theory page had 17 and the other 32,100 words had one. From the theory page:

> Fixed shapes push the same way: the P2P buffers freeze on the first step, so every batch pads to a
> constant width. That is why packing (fixed-width rows) is the shape to run under PP, and why the
> padding-free collator, whose width is the summed document length, is refused.

Instead: The buffers between stages are sized on the first step and never resized, so every batch has to
be padded to one width. Packing produces rows of a fixed width, so use it under PP. The padding-free
collator produces rows as wide as the documents in them, so PP rejects it.

Fine once or twice on a page, where the reader has just wondered why anyone would do that. The one use in
`human-docs/` reads naturally: "ZAYA1 has no auxiliary loss, which is why the recipes set
`output_router_logits: false`."

### Contrast

Density, 2 per 1,000 words, counting `, not Y` and `rather than` together. Never two in one paragraph and
never the same rejected half twice. `human-docs/` runs at 1.6 and the theory page ran at 4.2.

There are two kinds. A corrective contrast names what the reader would otherwise assume, and it stays:
"A row longer than `max_length` is dropped, not truncated", "run one torchrun per node, not one per GPU".
A foil brings in something nobody proposed so that the sentence can knock it down. From the theory page:

> A stream is an **ordering constraint, not a hardware lane**.

Instead: A stream fixes the order of the kernels put into it. It does not reserve any SMs.

The form `not just X, but Y` is a strong tell in any text. State Y. The trailing form, `Y, not just X`,
is counted as a contrast.

### Colon that sets up a payoff

Density, 0.5 per 1,000 words for the kind the scanner finds, which is about a third of them. A short
clause names a category and the colon delivers the item. Test each colon by deleting the words before
it. From the theory page:

> Split a model across GPUs and a third speed enters the picture: inter-GPU bandwidth.

Instead: Once a model is split across GPUs, the bandwidth between GPUs limits the step as well, next to
arithmetic throughput and memory bandwidth.

Fine before a list, a formula, a definition or an example, and after an instruction that gives its
reason: "Don't skip the retag: this guide, the `make` targets and the compose files use the local name."

### Announcement sentence

Density, 0.1 per 1,000 words. A short sentence says that a fact, a measurement or a rule is coming, and
the next sentence gives it. It is the reveal colon with a full stop in its place. It appeared when a
draft of the lecture was cleared of dashes and reveal colons: a reader counted 26 in 25,172 words, and
`human-docs/` has none. From that draft:

> One measurement shows the cost. On Qwen3-8B across two 8-GPU B300 nodes, 8 replicas of a 2-way split
> ran 10,780 tokens/s per GPU where 16 plain replicas ran 18,337, 0.59 times

Instead: In the one recorded comparison the split ran at 0.59 times the rate of plain replicas. On
Qwen3-8B across two 8-GPU B300 nodes, 8 replicas of a 2-way split ran 10,780 tokens/s per GPU where 16
plain replicas ran 18,337.

> A design rule follows. A collective that runs at every layer, and whose result the next operation
> needs, stays inside the NVLink domain.

Instead: So keep a collective that runs at every layer, and whose result the next operation needs,
inside the NVLink domain.

Test each one by deleting it. If the next sentence still reads, the announcement carried nothing. One or
two on a long page read as speech, and so does one in a PR comment. Keep one that states a fact the next
sentence builds on, such as "One sum is the exception."

The scanner finds 18 of those 26. It takes a sentence that opens on `One`, `A few` or `Several` and a
noun such as measurement, caveat or point, a sentence in which a measurement or a table `shows` or
`follows`, and `Two things follow.` It misses an announcement with no stock subject: "The recorded cases
cover each outcome."

### Frames and pronoun openers

Density, 0.5 per 1,000 words each. A frame is `The cost is`, `The fix is`, `The limit is` and their
family. An opener is a sentence that starts `This is`, `That is` or `That makes`, where the pronoun
points at the whole previous sentence and a newcomer cannot tell which part is meant. From the theory
page:

> The cost is rigidity: fixed shapes and addresses fight dynamic control flow, so graphs are awkward for
> MoE, where per-expert token counts change every step.

Instead: A CUDA graph replays the kernels it recorded with the shapes and addresses it recorded. A MoE
step sends a different number of tokens to each expert every time, so the recording does not fit the
next step.

### Slogan at the end of a paragraph

A strong tell once it repeats, and one the scanner cannot count. The line restates the paragraph in a
neater form. The theory page had 25. From the theory page:

> Comm does no FLOPs.

Instead: While a GPU waits on a collective its tensor cores compute nothing.

Keep a short closing line that tells the reader what to do. `human-docs/` has several: "Sharding costs
throughput, so use the least that fits".

The split reversal is the same move in two sentences, and a strong tell:

> The geometry suggests throughput should sawtooth and collapse just past each boundary. It doesn't.

Instead, write one sentence with the measured result first and the expectation second.

### Asides in parentheses

Density, 15 asides per 1,000 words, and 2 sentences with two or more asides per 1,000 words. A link cited
in parentheses and a parenthesis that holds only code are not counted. The theory page ran at 18 asides,
`human-docs/` at 7.5, the PR comments at 4.0 and the release notes at 3.0. From the theory page:

> That makes FLCE SFT-only (the preference and distillation losses read per-token log-probs or a whole
> distribution, which the fused kernel does not return), turns off entropy logging, and forces it off
> under TP and PP (under CP the Ulysses wrapper runs it over each chunk).

Instead: FLCE therefore works for SFT only. The preference and distillation losses need per-token
log-probs or the whole distribution, and the fused kernel returns neither. Entropy logging is off with
it. FLCE is also disabled under TP and PP. Under CP the Ulysses wrapper runs it on each chunk.

### Conditions in front of the result

Density: sentences of 35 words or more are at most 10% of a text's paragraph sentences. `human-docs/`
has 2% and the release notes had 15%, because each result came after its hardware, layout and batch.
From the release notes:

> On 8× H200 at ep8 with 4,096-token rows, three runs each, a whole simulated mxfp4 step runs at 0.99×
> bf16 on gpt-oss-20b, 0.94× on Qwen3-30B-A3B and 0.93× on Qwen3.5-35B-A3B, at batch 4, 8 and 4.

Instead: A whole simulated mxfp4 step runs at 0.99× bf16 on gpt-oss-20b, 0.94× on Qwen3-30B-A3B and
0.93× on Qwen3.5-35B-A3B. Those are three runs each on 8 H200s at ep8 with 4,096-token rows, at batch
4, 8 and 4.

The scanner prints the mean length of a text's paragraph sentences on its first line. `human-docs/` has
14.8 words, the PR comments 16.1, the theory page had 18.3 and the release notes 20.2.

### Fragments and symbols as verbs

Found by reading. The scanner limits the tilde to 0.5 per 1,000 words and allows no arrow in prose. Notes
written as prose have labels in place of verbs and `→`, `=` and `~` in place of words. From the theory
page:

> - **AI below ridge → memory-bound.** Achieved FLOP/s = `AI × bandwidth`. Fix: move fewer bytes, or use
>   faster HBM. More tensor cores won't help.

Instead: If a kernel's arithmetic intensity is below the ridge, it is memory-bound and runs at no more
than AI × bandwidth FLOP/s. To speed it up, move fewer bytes or use faster memory. More tensor cores
will not help.

A verbless label in front of a table or a formula is the same shortcut. A draft of the lecture
introduced eight of them this way: "One row by hand, for `M = 128` and `K = N = 2880`:". Give the label a
verb: "Work one row by hand, with M = 128 and K = N = 2880."

Fine in a table cell and in a formula. An arrow inside the text of a link, the house form
`Page → Section`, is not counted. Verbless bullets in a Limits list, such as "No online RL.", are a list
convention.

### Smaller constructions

- A participial tail that only claims importance, such as `, underscoring the value of fusion`. Read
  each hit. Keep a tail that states a fact which could be false.
- A negation chain, `no X, no Y`. Read each hit. A list of what a thing lacks is the fine case, and all
  nine in our texts were that. The tell is the chain built for rhythm and closed with `just`.
- A cleft, `X is what Y`, at 0.5 per 1,000 words. Write subject, verb, object. A definition is fine.
- Two short clauses balanced on a semicolon, found by reading. The theory page had "Deep pipelines want
  many microbatches; the roofline wants few and fat." Say what each half means: more stages need more
  microbatches to keep the idle share down, and every extra microbatch makes each GEMM smaller. A
  semicolon that joins two plain facts is fine.
- A question the writer answers at once, such as `The result?`. A real question to the reader is fine.
- A list of three made for rhythm. Ask whether a fourth item was dropped or a third invented. Write
  "A, B and C" without the serial comma, the form `human-docs/` uses 259 times against 56.

## Word choice

### Wishes and appetites

Density, 1 per 1,000 words. Hardware and software are given a will: a tile `wants`, a host `races`, a
kernel `eats`. The theory page had 64 such verbs by reading, and the scanner finds 42. From the theory
page:

> The CPU enqueues each kernel into a **stream** that the GPU drains asynchronously, and the host races
> ahead to keep SMs fed.

Instead: The CPU puts each kernel into a stream and returns without waiting for it. The GPU works
through the stream in the background, so the CPU can issue the next kernels while earlier ones are still
running.

The plain version is longer and says what "asynchronously" means, which a newcomer needs. "Sits",
"lands" and "lives in" are settled engineer English and are not counted. Neither are a page that owns a
fact and a metric that climbs.

### Money metaphor

Density, 0.5 per 1,000 words, for `buy`, `pay`, `spend`, `price` and `bill` said of bytes, time or
memory, and for `is worth` said of a gain. The theory page had 31 by reading and the other texts 10.
From the theory page:

> Each mode buys memory headroom by spending a specific collective.

Instead: Each mode frees memory and adds one kind of collective to every step.

"Cost" in its plain sense is fine: a matmul costs `2·M·N·K` FLOPs. A draft of the lecture that had lost
`buys` wrote that one exchange in place of three "was worth 4.4%". Write that it gained 4.4%.

### Moral adjective on a thing

A strong tell: `honest`, `faithful` or `truthful` said of a metric, a shape or a number. From the theory
page:

> Power is the honest signal:

Instead, say what the metric tracks and what the other one misses. The word is fine of a person, and of
a policy that does not cheat: "an honest solution".

### Color words and importance flags

Color words are limited to 1 per 1,000 words: `pure`, `far`, `enormous`, `sharply`, `at all` and
`wins big`. Importance flags are limited to one per file between them: `the central tool`,
`the canonical`, `most common`, `important`, `fundamental` and `worth knowing`. The theory page had 12
and 2, and the other texts 2 and none. Give the number, or drop the word. From the theory page:

> …so grouped GEMM's single batched launch wins big.

Instead: …so one grouped launch is faster: 0.11 ms against 1.12 ms for the loop of 128 matmuls.

A technical or literal use is fine: "exactly zero", "pure ETP", "critical path". `the real ceiling` is
the same emphasis and is found by reading, since nearly every other "real" in our texts is literal: "a
real checkpoint".

### Hedges

Density, 2 per 1,000 words, and no stack such as `could potentially`. Hedge a measurement once, with its
spread. Several of the theory page's 19 stood where a number should be, as in "often the difference
between a long sequence and OOM". Keep "usually" where it is true: "A NaN loss on SMPO usually means…".

### Clipped words

Density, 0.5 per 1,000 words, for `comm`, `util`, `seq`, `params`, `vocab`, `arch`, `quant` and `grads`
in prose. They are fine spoken in the lab, and each is a word a newcomer has to expand. The theory page
had 43 by reading and `human-docs/` has 2. Write "about" where the theory page wrote `~` 46 times.

### Names

- Write "Halo". `the toolkit` is limited to once per file: the theory page said it 14 times, and
  `human-docs/` says "Halo" 68 times and "the toolkit" once.
- Keep one name per thing. A newcomer cannot know whether two names are two things.
- Keep one thing per name. A draft of the lecture used "row" for a token, which is a row of a GEMM, and
  for a sequence, which is a row of a batch: "More rows to process do it, and so do longer rows". Say
  once which is which, and qualify the word wherever both meanings are near: "A row of a GEMM is one
  token. A row of a batch is one sequence, and its length in tokens is the row length."
- Use the established term. The theory page met the reader with about a dozen names ending in "-bound",
  "wall", "floor" or "ramp" and defined two of them.

### Banned words

A strong tell, in any text. None of them carries a measurement.

| Banned | Use instead |
|---|---|
| `leverage X` | use X |
| `robust`, `powerful` | delete, or state the measured property |
| `seamless`, `cutting-edge`, `state-of-the-art` | delete, or name the version |
| `it's worth noting that Y` | Y |
| `simply call Z`, `just call Z` | call Z |
| `in order to` | to |
| `essentially`, `basically` | delete |
| `delve`, `revolutionary`, `blazing-fast` | delete |

The words that rose in machine-written text after 2022 are limited to one per file between them:
`additionally`, `crucial`, `pivotal`, `enhance`, `foster`, `highlight` as a verb, `landscape` as an
abstract noun, `showcase`, `underscore` as a verb, `testament`, `valuable`, `quietly`, `dependable`,
`meaningful`. Grand verbs for "is", such as `serves as a` and `stands as a`, are limited to 0.5 per 1,000
words. A literal use of any of these is fine.

## Emphasis and layout

### Bold

Bold marks a term where it is defined. One bold sentence per section may carry a warning that costs the
reader a run if missed, as in "**The server must own GPUs that no trainer rank uses.**" The scanner
limits bold sentences to 1 per 1,000 words and lists every bold span for you to read. In a PR or
issue comment and in release notes the limit for both is none. From the README:

> - **Parallelism is added to the existing model.** EP wraps MoE blocks, CP wraps attention, and TP/ETP
>   shard weights in place. There is no separate distributed implementation of the model.

Instead: Halo adds parallelism to the model you already have. EP wraps the MoE blocks, CP wraps
attention, and TP and ETP shard weights in place. No second, distributed copy of the model code exists.

A short bold label is fine where the reader scans for it: a glossary entry, an option, a recipe setting.
A list of three such labels that carries an argument reads better as a paragraph. The scanner cannot
tell a bold sentence from a label of four words or more that ends in a full stop, such as "**Pad and EOS
tokens.**", and a third of its hits are such labels.

### Italics for stress

Density, 0.5 per 1,000 words. The theory page had 14 by reading and the other texts 4. From the theory
page:

> Under PP the microbatches *are* the gradient accumulation.

Instead, the way `human-docs/` puts it: `pipeline_microbatches` replaces gradient accumulation.

Italics on a title are fine. The scanner takes a span that opens in lower case, which leaves titles out.

### Heading with a tagline

Limit none. A heading that adds a slogan after a dash or a colon: 12 of the theory page's 42 headings
and one of the 304 in `human-docs/`. The theory page had `## 6. Attention — the canonical memory-bound
fix`.

Instead: `## 6. Attention and FlashAttention`.

A plain conversational heading is welcome: "Rules that save you a wasted run", "When it goes wrong".

### Other marks

- A horizontal rule before each section. The heading already marks the break. Limit none.
- Curly quotes. Limit none; this repository uses straight quotes.
- Emoji as formatting. A strong tell. The check mark and the cross of a support matrix are not counted.
- Title case in section headings. Use sentence case.

## Pointers and structure

### Pointers into the page

A section sign has the limit none, and `(below)`, `the table above` and `this page describes` are
limited to 0.5 per 1,000 words. The theory page pointed at another part of itself every 129 words. A
reader going top to bottom cannot resolve a forward pointer. From the theory page:

> The same shortfall resurfaces as §4's small-M ramp and §9's strong-scaling collapse.

Instead: You will meet this effect twice more: in section 4, where a small M runs slowly, and in
section 9, where adding GPUs to a job of fixed size stops helping.

A link to the page a reader needs at that point is fine. Give it once.

### Counted announcement

Read each hit. The sentence gives the number of things to come. It is fine when a list of that length
follows directly, as in "Three settings decide most runs:". A count with nothing counted after it is an
announcement sentence, as on the theory page:

> One ratio drives distributed design.

Instead, state the ratio and then say that the choices in the section follow from it.

Check each count against what follows it. The theory page titled a section "The four bottlenecks a step
can hit" and named a fifth nineteen lines later.

### The reader who disappears

Found by reading. A guide, a lecture or a message speaks to "you", then stops for whole sections and
reads as a reference page. A draft of the lecture said "you" 84 times and not once in its sections on
data, tensor, expert and context parallelism, 2,635 words together. From that draft:

> A dense model therefore has to fit on one GPU, however many GPUs the job has.

Instead: So adding GPUs will not make a dense model fit. Under data parallelism it has to fit on one.

Count "you" section by section. One sentence at the place where the reader would act is enough to keep
the address. `human-docs/` says "you" 4.5 times per 1,000 words and the PR comments 9.9.

### The same fact said again

Found by reading. The theory page carried one ten-word clause twice, eleven lines apart, and stated
eight facts three times or more. Say a fact once in the body. A lecture repeats its main points on
purpose in one place, such as a closing list of checks.

A number said twice needs one value. A draft of the lecture declared HBM bandwidth as 6.6 TB/s, "with
6.65 in arithmetic", and then used 6.5 on three lines. Search for every headline number and read each
place it appears.

### One sentence skeleton down a list

Found by reading. Every bullet has the same build: a label, a fix, a thing that will not help. A
template is right for recipes and for a catalog such as this one. In an argument, turn a three-bullet
list into a paragraph.

### Paragraph rhythm

Found by reading; the scanner lists paragraphs over 120 words. Almost half of the theory page's
paragraphs had two sentences, a claim and its consequence. Let the length follow the content.

A run of paragraphs that hold only numbers is the same fault. A draft of the lecture gave three
paragraphs of percentages in a row and said what to take from them fifteen lines later. Put the finding
first and the numbers in a small table.

## Facts and sources

### Unsourced measurement

A strong tell. A gain is "published" or "measured" and no paper, run or page is named. From the theory
page:

> …so published training gains are smaller (1.1–1.2× on MoE) and MXFP8 often raises memory.

Instead, give the number with its hardware, model and shape, and the citation or the page that holds
the run. The scanner finds only the stock phrasings, so read every "measured" and "published".

### Drafting notes left in the text

A strong tell, found by reading. The text narrates an authoring decision or the repository's stock of
measurements. From the theory page:

> `P` is the parameter count (the standard `6N` rule, renamed so `N` stays the GEMM width of §2).

Instead, define the symbol once where it first appears.

A tag repeated on every number is the same note. A draft of the lecture tagged eight numbers "Halo 1.1.0
or earlier". Say it once, where the page explains how to read its numbers. The scanner lists
every HTML comment in a page so that a note to the author, such as `<!-- WAITS: ... -->`, is read before
the page lands.

### Our own docs in quotation marks

Found by reading; the scanner skips quoted text. A page quotes a sister page as if it were an outside
authority. From a draft of the lecture:

> An environment variable that is set on some nodes only "is a hang or a silent drift, not a slow run"

Instead: An environment variable that is set on some nodes only can hang the job or let the ranks drift
apart silently. It does not show up as a slow run.

Keep the link to the sister page. Quote an outside source, such as NVIDIA's guide or a paper, in its own
words.

### Words about change on a doc page

Read each hit. `previously`, `no longer`, `used to` and `after the fix` belong in release notes, commit
messages and PR text, where the change is the subject. A doc page states what the code does now, as
[`style.md`](../docs/style.md) requires.

## Chat residue

Each is a strong tell. The only ones in the texts we counted were two slots left to fill in the draft of
the release notes.

- An offer or a sign-off: `I hope this helps`, `Let me know`, `Would you like`, `Feel free to`.
- Praise and thanks as an opener: `Great question`, `Good catch`, `You're absolutely right`,
  `Thanks for the thorough review`.
- A stock hook: `The catch is`, `Here's the thing`, `Heads up`, `Honestly` at the start of a sentence,
  `Let's dive in`.
- A summary closer: `In summary`, `In short`, `The takeaway`.
- An authority nobody can check: `Studies show`, `Experts say`.
- A leftover: `as an AI`, `[Your Name]`, `TODO`, `TBD`, and a line that is one upper-case name, such as
  `VALIDATION_TO_FILL`.

## Tics of our cleaned texts

The texts that already carry no em dash have habits of their own.

- "Refuse" is the house verb, and the scanner limits it to 6 per 1,000 words. The release notes ran
  at 6.4. "Other families stay refused on vLLM adapter sync" reads better as "vLLM adapter sync is still
  off for other families".
- A trailing `, as X does`, at 1 per 1,000 words, hangs a comparison on a sentence to save a second
  one. "Tool-call `arguments` stored as JSON text render as a mapping, as a serving engine does" becomes
  two sentences.
- The colon, the semicolon and `, so` are the joints these texts use most. None is wrong. On a long page
  vary them with a full stop.
- "Load-bearing" is on one published list of machine-writing words. It is the charter's term for a fact
  the reader acts on, and it stays in that sense.

## Where the limits come from

The counts are over 61 files and 39,300 words of prose: the five texts named at the top. The patterns
were first listed from a wider reading of the same texts and three more `agent-docs/` pages, 51,171
words with their tables.

Five limits have no count of ours behind them, because our texts hold too few cases to fit one. No hedge
stack, one post-2022 word per file, grand verbs at 0.5 per 1,000 words and no curly quotes follow the
published guides. The limit on announcement sentences comes from the one draft in which a reader counted
them.

The published side is Wikipedia's field guide
[Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing) and the
[tropes.fyi directory](https://tropes.fyi/directory). The Wikipedia guide calls itself descriptive and
warns against treating its signs as a list of things to fix one by one. This catalog is for writing
well. It cannot show who wrote a text.
