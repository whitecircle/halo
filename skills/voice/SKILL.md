---
name: voice
description: >-
  Load before writing or reviewing any text a person will read: a pull-request
  description or comment, an issue reply, a commit message, release notes, the
  README, a human-docs page, or a lecture page such as
  agent-docs/reference/gpu-training-theory.md. Holds the rules that keep
  machine-writing patterns out of Halo's text, the pattern catalog with a limit
  for each pattern, and a scanner that counts them.
allowed-tools: [Read, Edit, Write, Grep, Glob, Bash]
---

# Voice

Text that a person reads should sound like one engineer telling another what they found. The fact comes
first, in full sentences, with the number and where the number came from. This skill holds the working
rules for that. It applies to a pull-request description or comment, an issue reply, a commit message,
release notes, the README, a `human-docs/` page and a lecture page.

[`patterns.md`](patterns.md) is the catalog: every pattern with an example from our own texts, what to
write in its place and the limit that applies. The scanner, `skills/voice/scripts/scan.py`, counts the
patterns that a regular expression can find. A doc page also follows the charter of the `docs` skill,
[`style.md`](../docs/style.md), which holds the rules for docs alone: what belongs on a page, how long
it runs and how it cites code.

## Procedure

1. Draft. Decide who reads the text and what they will do with it, then open with the fact they came for.
   Check each claim before you write it down: read the code, run the command, open the log.
2. Scan. Run the scanner on the draft and read its report.

   ```bash
   python3.12 skills/voice/scripts/scan.py human-docs/performance.md
   python3.12 skills/voice/scripts/scan.py --message /tmp/comment.md
   gh pr view <number> --json body -q .body | python3.12 skills/voice/scripts/scan.py --message -
   ```

3. Read aloud. Read the whole text from the top, as its reader would, and listen for what the scanner
   cannot count. The list is under [Reading aloud](#reading-aloud).
4. Fix what the scan and the reading found, then scan and read again.
   [What to leave alone](#what-to-leave-alone) says how a repair goes wrong.

### Reading the report

The scanner needs only Python 3.12 and its standard library, so it runs on the host without an image. It
is a tool for the writer, and it reads English. Nothing in CI or `make docs` runs it, and a clean report
does not mean the text is good.

It counts prose only. Fenced code, inline code, blockquotes, HTML comments and text in double quotes are
left out, so a sentence can quote a pattern without counting as a use of it. Table rows are searched for
most of the strong tells and for a few word patterns, and they do not add to the word count.

The first line gives the words of prose and the mean length of the paragraph sentences. The rows come
in three groups:

- Strong tells. One is enough to fix, and any hit makes the exit status 1.
- Density. Each row gives the count, the rate per 1,000 words of prose and the limit, and says `OVER`
  when the count is past it. A text under 500 words gets the allowance of 500, since a rate means
  little on a short comment.
- Read each hit. These have no limit because a count cannot separate the good uses from the bad ones.

Every row lists the line numbers of its hits. `--notes` adds what reading the hits of that expression
showed: how many were false, and what the expression misses. `--message` is for a PR or issue comment, a
commit message and release notes. It allows no bold and does not report words about change such as "no
longer", which are what such a text is about. On a PR body the template's checklist lines and its
AI-disclosure labels are counted too, and their bold, dashes and asides are not yours to fix. A file the
scanner cannot read ends the run with status 2 before any report is printed.

### Reading aloud

The scanner finds none of these, and they decide whether a page reads well:

- Is the last sentence of a paragraph a restatement of the paragraph?
- Does a sentence only announce the next one? Delete it and see whether anything is lost.
- Does a term appear before its definition, or does one word carry two meanings?
- Does a fact appear twice, or a number with two values? Search for every headline number.
- Does every count match what follows it? A text that announces four items and names a fifth is wrong.
- Can a newcomer resolve every pointer where it stands?
- Does "you" drop out for a whole section of a guide or a lecture?
- Do several paragraphs in a row have the same shape, such as a claim and then its consequence, or
  numbers with the finding left for later?
- Would you say this sentence to a colleague at a desk? If you would not, write what you would say.

## Rules

### Sentences

- Say what the thing is and stop. Keep a contrast only where the reader would otherwise act on the
  rejected half, and never bring the same rejected half back.
- Put the cause first and join with "so", or the effect first and join with "because". Keep
  "which is why" for the one or two places where the reader has just wondered why anyone would do
  that.
- A colon introduces a list, a formula, a definition or an example. If the words before it can be
  deleted with nothing lost, they were a drumroll: fold them into the claim. Do the same with a sentence
  whose only job is to say that the next one holds a fact.
- End a paragraph on its last fact.
- Give one claim per sentence and the result before its conditions. The model, layout, batch and GPU
  count go in the sentence after the number.
- Use at most one aside in parentheses per sentence.
- Write full sentences with a verb. A label with a fragment after it (`Fix: move fewer bytes`) belongs
  in a table cell.

### Words

- Use one name per thing and one thing per name. Write "Halo" where a sentence is about Halo's choice.
- Prefer the established term. A name coined on the page gets one defining sentence where it first
  appears.
- Say what hardware and software physically do. A kernel does not `want`, `eat` or `fight`, a mode does
  not `buy` or `pay`, and a metric cannot be `honest`.
- Give the number in place of the adjective. Where no number exists, drop the adjective.
- Hedge a measurement once, with its spread.
- Spell out clipped words (`seq`, `params`, `comm`, `util`) and write "about" in prose. The tilde, the
  arrow and the equals sign stay in tables and formulas.
- Never use the banned words listed in [`patterns.md`](patterns.md#banned-words).

### Structure

- A heading is a plain noun phrase or a short plain statement in sentence case. It has no dash, colon
  or slogan in it.
- Bold marks a term where it is defined, and at most one warning per section that costs the reader a run
  if missed. Italics are for titles.
- Use bullets for parallel things a reader scans, and prose where each point depends on the one before.
- Refer to another section by its name, in words, at the place where the reader needs it.
- Announce a count only when a list of that length follows directly.
- Let paragraph length follow the content: one line for a fact, six sentences for a derivation.
- Address the reader as "you" in guides, lectures and messages, in every section.

### Facts

- Write only what you checked by your own run or your own reading of the code, and say which it was.
- Put the hardware, model and shape beside every measured number, and the source beside every
  published one. If it was one run, say "one run".
- Mark what you did not check as not checked.
- Keep the limits and the results that came out slower, because a reader cannot guess them.
- Leave your own drafting out of the text: no note on why a symbol was renamed, no inventory of what
  the repository has recorded.

## What to leave alone

A banned construction comes back in the nearest form that is still allowed, so repair the sentence and
never the mark. Each of these happened in our own texts:

- The em dash went and the colon took its job. `human-docs/` has no dash and an estimated 108 colons
  that join a claim to its reason.
- The reveal colon went and a short sentence took its job. A draft of the lecture that the scanner
  passed had 26 announcements such as "One measurement shows the cost."
- The dash after a label went and a verbless label with a colon took its place: "One row by hand, for
  `M = 128` and `K = N = 2880`:".
- `buys` and `pays` went and "was worth 4.4%" took their place.

Split the sentence, give the fact a subject and a verb, or delete the sentence that only points at the
next one. Three more repairs make a text worse:

- Chopping sentences into fragments to vary the rhythm. Prose made only of commas and full stops is its
  own mark, so keep parentheses, semicolons and real questions in normal use.
- Replacing a coined label with another coined label.
- Deleting hits until a count fits its limit. A count over the limit is a reason to read the hits.

Some things are fine and need no repair. A contrast that stops a mistake the reader was about to make is
one, and so are the spread on a measurement and "usually" in troubleshooting advice. A technical term is
repeated as often as it is needed. "Sits", "lands" and "lives in" are ordinary engineer English, and
"refuse" is the house verb for a config Halo rejects; use a plainer verb only where one names the
mechanism better.

## Pull requests, issues, commits and release notes

The same rules apply, and a message adds its own:

- Be brief by cutting the framing and keeping the detail: the file, the line, the commit, the number
  with its shape.
- Write in the first person and say what you did. "I checked every changed sentence against the code and
  recipes at `e92c5d353`" tells the reader what the claim rests on.
- Name the command and the commit behind a claim. If something was not run, say so in plain words.
- Do not praise, and do not acknowledge every point. Answer what needs an answer, and leave out thanks
  for the review and restatements of what the other person wrote.
- Leave bold lead-in labels and headings out of a comment. A PR description keeps the headings of the
  template and nothing more.
- Use no stock phrases. `patterns.md` lists them under [Chat residue](patterns.md#chat-residue).
- Ask for what you need as a direct question or request.
- In release notes, open with a summary a user can act on, then one topic per paragraph with the result
  first and the measurement conditions after it. Titles are plain and descriptive.
- A commit subject names the change in plain words. The body says why, and what you ran.

The AI disclosure that `CONTRIBUTING.md` asks for stays in the PR.

## Texts to copy

Our PR comments are the model for voice. They are in the first person, give one fact per sentence, put
the evidence beside the claim and ask directly:

> `frozen_quantized_experts=False` in the stub should do it. Could you add it with the merge of main? I'll
> rerun the row on the new head.

> Most hold as written. Three need a small change before I approve.

`human-docs/` is the model for mechanics. Copy three of its habits anywhere:

- An instruction with its reason in the same breath: "Don't `pip install` Halo on the host; the compiled
  kernels exist only in the image."
- A sentence that says what the mistake looks like: "Without the single quotes every task gets node
  rank 0."
- Plain scope: "No run has trained to a quality metric."

A lecture adds worked arithmetic that the reader can redo, results that did not work out, and an opening
that starts from what the reader believes: "You may have read that fp8 made inference twice as fast".

## Sources of truth

The limits in `patterns.md` come from a count over Halo's own docs, release notes and PR comments, and
from the published field guides that the catalog names. `scan.py` implements those limits. If the catalog
and the scanner disagree, fix the scanner. The rules for doc pages are in the `docs` skill.
