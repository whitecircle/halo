"""Count the writing patterns of the voice skill in markdown or plain text, each against its limit.

A tool for the writer, not a gate: nothing in CI runs it. It exits 1 when a strong tell is present and 2 on a
file it cannot read; a density over its limit only shows in the report. English text, standard library only,
Python 3.12.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, NamedTuple

LONG_SENTENCE_WORDS = 35
LONG_SENTENCE_SHARE = 0.10
LONG_PARAGRAPH_WORDS = 120
CROWDED_SENTENCES_PER_1000 = 2
# A rate means little on a short text, so one is held to the allowance of this many words.
SHORT_TEXT_WORDS = 500

_FENCE = re.compile(r"^(`{3,}|~{3,})")
_HEADING = re.compile(r"^#{1,6} ")
_RULE = re.compile(r"^(?:-{3,}|\*{3,}|_{3,})$")
_LIST_ITEM = re.compile(r"^(?:[-*+]|\d+[.)])\s+(.*)$")
# A code span is matched only to be stepped over: a comment marker inside one is code.
_CODE_OR_COMMENT = re.compile(r"(?P<code>`+).*?(?P=code)|<!--.*?(?P<closed>-->)|<!--.*")
# A sentence may close on a bracket or bold mark and open on anything: a number, a tilde, a lower-case name.
_SENTENCE_END = re.compile(r"(?<!\be\.g)(?<!\bi\.e)(?<!\bvs)(?<!\bcf)[.!?][)*\]]*\s+(?=\S)")
_SENTENCE_START = r"(?:^|(?<=[.!?] )|(?<=[.!?]\) )|(?<=[.!?]\*\* ))"
_LINK = r"\[[^\]]+\]\([^)]+\)"
# An aside is a parenthesis of prose: one that holds only code spans or quotes, a section sign or a list
# marker is not one.
_MENTION = r"(?:CODE|QUOTE)"
_ASIDE = re.compile(
    rf"\((?!(?:{_MENTION}(?:(?:,\s*|\s*/\s*|\s+)(?:(?:or|and)\s+)?{_MENTION})*|§\d+|\d+|[a-z])\))[^()]+\)"
)
_CONTRAST = re.compile(r"(?:,|—|;)\s+not\s+|\brather than\b")
# Applied in order. Code and quoted text are mentions, so a page can name a pattern without using it. A
# parenthesis that holds only links is a citation, and an arrow in a link's text is the house form for
# Page → Section.
_CLEANERS = (
    (re.compile(r"!\[[^\]]*\]\([^)]*\)"), ""),
    (re.compile(r"(`+).+?\1", re.DOTALL), "CODE"),
    (re.compile(rf"\((?:see\s+)?{_LINK}(?:\s*↗)?(?:(?:,|;|\s+and|\s+or)\s+{_LINK}(?:\s*↗)?)*\)"), ""),
    (re.compile(r"→(?=[^\[\]]*\]\()"), "to"),
    (re.compile(r"\[([^\]]+)\]\([^)]+\)"), r"\1"),
    (re.compile(r"https?://\S+"), "URL"),
    (re.compile(r'"[^"]+"'), "QUOTE"),
)
# The subjects of a sentence that only says the next one holds a fact.
_ANNOUNCED = (
    r"measurements?|examples?|caveats?|cautions?|points?|facts?|lessons?|propert(?:y|ies)|things?|consequences?|"
    r"observations?|comparisons?|details?|exceptions?|notes?|questions?|distinctions?|values"
)

Kind = Literal["strong", "density", "read"]
Where = Literal["prose", "prose_or_heading", "text", "heading", "rule", "comment"]

_BLOCK_KINDS: dict[Where, tuple[str, ...]] = {
    "prose": ("paragraph", "item"),
    "prose_or_heading": ("paragraph", "item", "heading"),
    "text": ("paragraph", "item", "heading", "table"),
    "heading": ("heading",),
    "rule": ("rule",),
    "comment": ("comment",),
}
_GROUP_TITLES: dict[Kind, str] = {
    "strong": "strong tells, fix each",
    "density": "density, hits against the limit",
    "read": "read each hit",
}


@dataclass(frozen=True)
class Pattern:
    """One detectable pattern. ``note`` is what reading the hits of ``regex`` showed, which a count cannot."""

    name: str
    kind: Kind
    regex: str
    where: Where = "prose"
    per_1000: float | None = None  # density: hits allowed per 1,000 words of prose
    cap: int | None = None  # density: hits allowed in one file whatever its length
    in_message: Literal["same", "none", "skip"] = "same"  # under --message: allow no hit, or do not report
    note: str = ""

    def __post_init__(self) -> None:
        if self.kind == "density" and self.per_1000 is None and self.cap is None:
            raise ValueError(f"density pattern {self.name!r} has no limit: set per_1000 or cap, or make it 'read'")


@dataclass
class Block:
    kind: Literal["paragraph", "item", "heading", "table", "rule", "comment"]
    line: int
    lines: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Row:
    """One line of the report: where a pattern hit and how many hits its limit allows (None: no limit)."""

    name: str
    kind: Kind
    hits: list[int]
    allowed: int | None
    limit: str
    note: str

    @property
    def over(self) -> bool:
        return self.allowed is not None and len(self.hits) > self.allowed


class Scan(NamedTuple):
    words: int
    sentence_lengths: list[int]  # of paragraph sentences; list items are left out
    rows: list[Row]


# The notes give what reading the hits showed on 39,300 words of prose: human-docs, the README, one draft of
# release notes, 16 PR comments and the GPU theory page at b83a5761f, which the notes call the theory page.
PATTERNS = (
    Pattern(
        "not just X, but Y",
        "strong",
        r"(?i)(?<![,;—] )(?<!\band )\bnot (?:just|only|merely|simply)\b|\bisn'?t (?:just|only|merely)\b",
        "text",
        note="Takes the form that leads with the rejected half, in one sentence or split over two (it is not only "
        "speed.). After a comma, Y, not just X is counted as a contrast.",
    ),
    Pattern(
        "split reversal",
        "strong",
        _SENTENCE_START + r"(?:It|They|That|This) "
        r"(?:doesn't|does not|don't|do not|didn't|did not|isn't|is not|won't|never \w+)\.(?= |$)",
        note="Takes only a pronoun and a negation, so it misses a reversal that opens any other way.",
    ),
    Pattern(
        "summary closer",
        "strong",
        r"(?i)\b(?:in summary|in conclusion|in short|to sum up|the takeaway|the bottom line)\b",
        note="Overall is left out: it is nearly always the adjective, as in overall throughput.",
    ),
    Pattern(
        "moral adjective on a thing",
        "strong",
        r"(?i)\b(?:honest|faithful|truthful)\b",
        "text",
        note="Fine of a person, of a policy that does not cheat (an honest solution) and as a numerics term.",
    ),
    Pattern(
        "banned word",
        "strong",
        r"(?i)\b(?:delv\w+|leverag\w+|seamless\w*|robust|powerful|cutting-edge|it'?s worth noting|simply|"
        r"just call|essentially|basically|in order to|revolutionary|blazing-fast|state-of-the-art)\b",
        "text",
        note="The banned words of patterns.md.",
    ),
    Pattern(
        "chat residue",
        "strong",
        r"(?i)\b(?:i hope this helps|let me know|great question|certainly!|of course!|you're absolutely right|"
        r"would you like|feel free to|happy to help)|" + _SENTENCE_START + r"here(?:'s| is) (?:a|the|an) |"
        r"\blet's (?:break|dive|explore|unpack|take a look|walk)\b|\bwe(?:'ll| will) (?:explore|see|look|walk)\b|"
        r"as of my (?:last|knowledge)|\bas an ai\b",
        "text",
        note="Here is counts only at a sentence start, since mid-sentence it is ordinary (fp8 here is a QAT path). "
        "Let's counts only before a tutorial verb: a PR comment that proposes a plan with it is plain speech.",
    ),
    Pattern(
        "praise or stock phrase",
        "strong",
        r"(?i)\b(?:great|good|nice|excellent) (?:catch|point|question|work|job|find)\b(?! to\b)|"
        r"\bthanks? (?:you )?for (?:the|your) (?:thorough|detailed|careful|great|excellent)\b|\bheads[- ]up\b|"
        r"\bthe catch is\b|\bhere's the thing\b|" + _SENTENCE_START + r"Honestly\b",
        "text",
    ),
    Pattern(
        "vague attribution",
        "strong",
        r"(?i)\b(?:experts?|observers|researchers|studies|industry reports?) "
        r"(?:argue|say|believe|show|suggest|note)\b|\bit is (?:widely|generally) (?:known|accepted)\b|"
        r"\bpublished (?:training )?gains\b",
        "text",
        note="Finds the stock phrasings only, one of the three on the theory page, so read every measured and "
        "published for its source.",
    ),
    Pattern(
        "placeholder",
        "strong",
        r"\[(?:Your|Insert|Add|Enter|Describe|Specify|Choose)[^\]]+\]|\bTODO\b|\bTBD\b|20\d\d-(?:XX|xx)|"
        r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$|\b(?:INSERT|FILL|REPLACE)_[A-Z0-9_]+\b|\b[A-Z0-9_]+_(?:TO_FILL|PLACEHOLDER)\b",
        "text",
        note="A paragraph that is one upper-case name (VALIDATION_TO_FILL) counts: both hits in the release "
        "notes draft were slots left to fill.",
    ),
    Pattern(
        "emoji",
        "strong",
        r"[\U0001F300-\U0001FAFF☀-✒✕✖✙-➿](?!\ufe0f? ?=)",
        "prose_or_heading",
        note="The plain check mark and cross are left out, table rows are not searched, and a mark followed by = "
        "is a legend entry in prose (a check mark = patched) and is not counted. Headings are searched.",
    ),
    Pattern(
        "em dash",
        "density",
        "—",
        per_1000=2,
        note="A dash inside a bold label counts. A dash in a heading is counted under heading with a tagline, and "
        "one right after a bold or code label under dash after a label.",
    ),
    Pattern(
        "dash after a label",
        "density",
        r"^(?:\*\*[^*]+\*\*|CODE)\s*—",
        cap=0,
        note="Takes a bold span or code span that opens a paragraph or a list item, then a dash. A label in plain "
        "words is left to the em dash rate.",
    ),
    Pattern(
        "is why",
        "density",
        r"\b(?:is|are|was) why\b|\b[Tt]hat's why\b|\bwhich is why\b|\b[Tt]he corollary\b",
        per_1000=0.5,
        cap=3,
        note="None of 18 hits was false.",
    ),
    Pattern(
        "contrast",
        "density",
        _CONTRAST.pattern,
        per_1000=2,
        note="None of 71 hits was false, and the count cannot tell a contrast that states a limit from a foil. "
        "Human-docs runs at 1.6 per 1,000 words, nearly all of the first kind.",
    ),
    Pattern(
        "setup colon",
        "density",
        r"(?:^|(?<=[.] ))(?:The|That|This|One|Each|Every|Its)?[^.:(]{3,70}\b(?:cost|fix|corollary|rule|question|"
        r"point|answer|result|reason|catch|problem|shape|tensor|quantity|operation|argument|defense|rigidity|"
        r"picture)\b[^.:]{0,25}: (?=[a-z*]|CODE)",
        per_1000=0.5,
        note="A noun-list heuristic. It found 11 of the 30 on the theory page, and 5 of its 16 hits were false: "
        "a claim with its reason, or a label before a list. Read every colon that does not open a list, a "
        "formula, a definition or an example.",
    ),
    Pattern(
        "announcement sentence",
        "density",
        _SENTENCE_START + r"(?:(?:(?:One|A few|Several)(?: more)?(?: [a-z-]+){0,2} (?:" + _ANNOUNCED + r")\b"
        r"(?! (?:is|are)\b)(?: [^\s.!?:]+){0,14}|One(?: more)?(?: [^\s.!?:]+){1,6}? (?:shows|follows|matters|"
        r"applies|appears|belongs|drives)\b(?: [^\s.!?:\d]+){0,10}|(?:A|An|The)(?: [a-z-]+){0,2} (?:measurements?|"
        r"examples?|rules?|tables?|numbers?|figures?|comparisons?|tests?|runs?)\b(?: [^\s.!?:]+){0,6} "
        r"(?:shows?|follows?)\b(?: [^\s.!?:\d]+){0,9})[.:]|(?:A|An|One|Two|Three|Four|The)(?: [a-z-]+){0,3} "
        r"follows?\.)(?= |$)",
        per_1000=0.1,
        note="No hit in human-docs. Of the 26 a reader counted on one draft of the GPU lecture it finds 18, and it "
        "misses one with no stock subject (The recorded cases cover each outcome). One in a comment is speech.",
    ),
    Pattern(
        "frame",
        "density",
        r"\b(?<!over )(?<!under )(?<!past )(?<!at )(?<!of )(?<!to )(?<!in )(?<!on )(?<!for )(?<!with )(?<!by )"
        r"[Tt]he (?:cost|fix|limit|catch|trade|price|win|point|rule|trick|exception|difference|reason|"
        r"verdict|bar|levers?|defenses?|crossover|margin) (?:is|are|was)\b",
        per_1000=0.5,
        note="One of 12 hits was literal (the difference is one word). Result is left out: the result is exact. "
        "A frame after a preposition is not counted (a count over the limit is).",
    ),
    Pattern(
        "this-is opener",
        "density",
        r"(?:^|[.!?]\s+)(?:This|That) (?:is|means|makes|leaves|was)\b|\bThat's\b",
        per_1000=0.5,
        note="2 of 21 hits were false: the first sentence of a page, where This is the page.",
    ),
    Pattern(
        "wish or appetite",
        "density",
        r"\b(?:(?<![Yy]ou )wants?|eats?|races? ahead|fights?|hides?|rides?|leans? on|tolerates?|survives?|survivable|"
        r"straw|slides?|resurfaces?|swamps?|crush(?:es)?|outruns?|wobbl\w+|fires?|(?<!the )feeds?|fed|carves?|"
        r"hands|nudges?|absorbs?|pull on)\b",
        per_1000=1,
        note="3 of 53 hits were plain: a skill that fires, a fallback that hides an error, a person who wants. "
        "Sits, lands, lives in, owns and climbs are ordinary engineer English and are not searched.",
    ),
    Pattern(
        "money metaphor",
        "density",
        r"\b(?:buys?|bought|pays?|paid|paying|spends?|spending|pric(?:e|es|ed|ing)|bill|"
        r"(?:is|are|was|were) worth(?! it\b)(?! [a-z]+ing\b))\b",
        per_1000=0.5,
        note="One of 33 hits was literal (tokens an episode spends).",
    ),
    Pattern(
        "bold sentence",
        "density",
        r"^\*\*[^*\s]+(?:\s+[^*\s]+){2,}\s+[^*\s]*[.!?]\*\*|(?<!^)\*\*[^*\s]+(?:\s+[^*\s]+){3,}\*\*",
        per_1000=1,
        in_message="none",
        note="A bold span of four words or more that heads a block and ends in a full stop, or sits inside one. "
        "20 of 56 hits were labels with no verb (Pad and EOS tokens.), which the count cannot tell from a "
        "sentence. One warning that costs a run if missed is the fine case.",
    ),
    Pattern(
        "bold",
        "read",
        r"\*\*[^*]+\*\*",
        in_message="none",
        note="Fine on a term where it is defined, and as the label of a glossary entry, an option or a recipe "
        "setting. Table rows are not searched.",
    ),
    Pattern(
        "italic stress",
        "density",
        r"(?<![*\w])\*[a-z][^*]*\*(?![*\w])",
        per_1000=0.5,
        note="Takes a span that opens in lower case, which leaves out the titles of a reading list and misses "
        "stress on a capitalized word. None of 12 hits was false.",
    ),
    Pattern("heading with a tagline", "density", r"^#{1,6} .*(?:—|: ).+", "heading", cap=0, note="No false hit."),
    Pattern("section sign", "density", r"§\d+", "text", cap=0),
    Pattern(
        "pointer into the page",
        "density",
        r"(?i)\((?:below|above)\)|\bthe (?:figure|table|section) (?:below|above)|\bsee (?:below|above)\b|"
        r"\bthis (?:page|section) (?:describes|is|covers|explains)\b",
        per_1000=0.5,
        note="None of 9 hits was false. A bare above or below is not searched: most are comparisons such as below "
        "the ridge. A pointer in words (the section on memory explains it) is not found.",
    ),
    Pattern(
        "aside in parentheses",
        "density",
        _ASIDE.pattern,
        per_1000=15,
        note="Human-docs runs at 7.5 per 1,000 words, the release notes at 3.0, the theory page at 18. A link cited "
        "in parentheses and a parenthesis of code spans are not counted.",
    ),
    Pattern("tilde for about", "density", r"~\s?(?:\d|CODE)", per_1000=0.5, note="None of 37 hits was false."),
    Pattern(
        "arrow as a verb",
        "density",
        r"[→↔⇒←]",
        cap=0,
        note="None of 7 hits was false. Table rows and the text of a link are not searched.",
    ),
    Pattern(
        "absolute or color word",
        "density",
        r"\b(?i:purely|enormous|at all|far (?:below|above|better)|sharply|steeply|wins big|at its extreme)\b|"
        r"\b[Pp]ure\b(?! [A-Z]{2,3}\d*\b)",
        per_1000=1,
        note="None of 14 hits was false. Exactly is not searched: 7 of its 8 hits were literal (exactly zero). "
        "Pure before a mode name (pure ETP) is not counted.",
    ),
    Pattern(
        "hedge",
        "density",
        r"(?i)\b(?:usually|(?<!more )(?<!most )(?<!as )often|roughly|effectively|mostly|almost|typically|generally|largely|"
        r"arguably|somewhat|fairly|commonly|nearly|in practice|tends? to)\b",
        per_1000=2,
        note="In troubleshooting advice usually is the true word: 11 of the 15 hits in human-docs.",
    ),
    Pattern(
        "hedge stack",
        "density",
        r"(?i)\b(?:may|might|could|can) (?:potentially|possibly|perhaps|eventually)\b",
        cap=0,
    ),
    Pattern(
        "importance flag",
        "density",
        r"(?i)\b(?:central tool|most common|the canonical|important(?:ly)?|notabl[ey]|fundamental(?:ly)?|"
        r"critical(?:ly)?(?! path)|worth (?:knowing|remembering|holding on to))\b",
        cap=1,
        note="Key is not searched: here it is a dictionary, config or API key. Critical path is a term. The "
        "canonical is plain where it names a reference implementation.",
    ),
    Pattern(
        "AI vocabulary",
        "density",
        r"(?i)\b(?:additionally|align(?:s|ed|ing)? with|boasts?|bolster(?:s|ed|ing)?|crucial|deep dive|emphasizing|"
        r"enduring|enhanc(?:e|es|ed|ing)|foster(?:s|ed|ing)?|garner(?:s|ed|ing)?|highlight(?:s|ed|ing)?|interplay|"
        r"intricate|intricacies|landscape|meticulous(?:ly)?|pivotal|showcas(?:e|es|ed|ing)|tapestry|testament|"
        r"underscor(?:ed|ing)|underscores? (?:the|that|how|its|a|an)|valuable|vibrant|quietly|dependable|"
        r"multifaceted|prioritiz\w+|universally|deliberate|meaningful|matters because|matters more than|"
        r"rather than merely|rather than relying)\b",
        cap=1,
        note="Steady is not searched: in human-docs it is a steady rate or a steady drift. Underscore counts only "
        "as a verb. A deliberate change and a literal highlight are plain uses.",
    ),
    Pattern(
        "grand verb for is",
        "density",
        r"(?i)\b(?:(?:serv(?:es|ed|ing)|stands?|functions?|operates?|acts?) as (?:an?|the|its)\b|"
        r"holds the distinction|refers to)\b",
        per_1000=0.5,
        note="No hit in the texts read. A literal role (the node acts as a relay) is a plain use.",
    ),
    Pattern(
        "cleft",
        "density",
        r"(?i)\bis (?:exactly )?what\b",
        per_1000=0.5,
        note="2 of 8 hits were definitions (the last column is what passed), which are fine.",
    ),
    Pattern(
        "negation chain",
        "read",
        r"(?i)\bno [\w-]+(?: [\w-]+){0,3},? (?:and )?no [\w-]+",
        note="All 9 hits listed what a thing lacks, which is the fine case. The tell is the chain made for "
        "rhythm, closed by just.",
    ),
    Pattern("the toolkit for Halo", "density", r"\b[Tt]h(?:e|is) toolkit\b", "text", cap=1),
    Pattern(
        "refuse",
        "density",
        r"\brefus(?:e|es|ed|al|als|ing)\b",
        per_1000=6,
        note="The house verb: human-docs 2.8 per 1,000 words, the release notes 6.4.",
    ),
    Pattern(
        "trailing as-X-does",
        "density",
        r", as [^,.;:()]{2,70}? (?:does|do|did|are|is|says|said|ships|seed it|carry it|has)\b",
        per_1000=1,
        note="One of 11 hits was a citation (as the docstring says).",
    ),
    Pattern(
        "clipped word",
        "density",
        r"\b(?:comm|util|seq|params?|vocab|arch|quant|dequant|grads)\b",
        per_1000=0.5,
        note="One of 41 hits was a logged formula, ln(vocab). Fake-quant counts: the term is fake quantization.",
    ),
    Pattern("curly quote", "density", r"[“”‘’]", "text", cap=0, note="A quoted title is the fine case."),
    Pattern("horizontal rule", "density", r".+", "rule", cap=0, note="Front matter is not counted."),
    Pattern(
        "changelog word",
        "read",
        r"(?i)\b(?:previously|no longer|used to|newly|after the fix|turned out|re-confirmed|verified \d{4}|"
        r"as of \d{4}|recently)\b",
        in_message="skip",
        note="On a doc page state the present fact. Used to in the sense of employed to is a false positive.",
    ),
    Pattern(
        "HTML comment",
        "read",
        r"^<!--",
        "comment",
        in_message="skip",
        note="Read it for a note to the author left in the page. A line for maintainers is the fine case, and a "
        "PR body carries its template's comments, so a message skips this row.",
    ),
    Pattern(
        "counted announcement",
        "read",
        r"\b(?:[Tt]wo|[Tt]hree|[Ff]our|[Ff]ive|[Ss]ix) (?:[a-z-]+ ){0,2}(?:things|places|regimes|identities|"
        r"bottlenecks|ideas|defenses|techniques|capabilities|reasons|ways|parts|levers|settings|rules|kinds|"
        r"decisions|setups|limits|cases|relations|habits|consequences)\b",
        note="Check each count against what follows it. 3 of 19 hits were plain counts (conflicted in two places), "
        "and 7 stood before a list of that length, which is the fine case.",
    ),
    Pattern(
        "participial tail",
        "read",
        r", (?!including\b|depending\b|according\b|nothing\b|during\b)[a-z]+ing\b(?:[^,;:().!?]|\.(?=\d))*"
        r"(?:[.!?:](?= |$)| \()",
        note="Takes a tail that runs to the end of its sentence, so it misses one that holds a comma. 4 of 23 hits "
        "were nouns (learning rate). Keep a tail that states a fact which could be false.",
    ),
)


def _cut_comments(line: str, in_comment: bool) -> tuple[str, list[str], bool]:
    """``line`` with its HTML comments cut out, the comments that open on it, and whether one is open at its end.

    ``in_comment`` says the line starts inside a comment an earlier line opened.
    """
    if in_comment:
        end = line.find("-->")
        if end < 0:
            return "", [], True
        line = line[end + 3 :].strip()
    comments = [match for match in _CODE_OR_COMMENT.finditer(line) if match["code"] is None]
    if not comments:
        return line, [], False
    kept = line
    for match in reversed(comments):
        kept = f"{kept[: match.start()].rstrip()} {kept[match.end() :].lstrip()}"
    return kept.strip(), [match.group() for match in comments], comments[-1]["closed"] is None


def parse_blocks(text: str) -> list[Block]:
    """The paragraphs, list items, headings, table rows, rules and HTML comments of ``text``, each with its line.

    A comment is cut out of the line it sits in and listed on its own. Front matter, fenced code, blockquotes,
    image lines, raw HTML lines and the body of a comment are left out.
    """
    lines = text.removeprefix("\ufeff").split("\n")
    first = 0
    if lines[0].strip() == "---":
        closing = next((i for i, line in enumerate(lines[1:], 1) if line.strip() == "---"), None)
        first = 0 if closing is None else closing + 1
    blocks: list[Block] = []
    comments: list[Block] = []
    fence = ""
    in_comment = False
    flowing = False
    for number, source in enumerate(lines[first:], first + 1):
        line = source.strip()
        continues = flowing
        flowing = False
        if fence:
            if line.startswith(fence) and not line.strip(fence[0]):
                fence = ""
            continue
        line, opened, in_comment = _cut_comments(line, in_comment)
        comments += [Block("comment", number, [comment]) for comment in opened]
        if opening := _FENCE.match(line):
            fence = opening.group(1)
        elif not line or line.startswith((">", "![")) or (line.startswith("<") and line.endswith(">")):
            continue
        elif line.startswith("|"):
            blocks.append(Block("table", number, [line]))
        elif _HEADING.match(line):
            blocks.append(Block("heading", number, [line]))
        elif _RULE.match(line):
            blocks.append(Block("rule", number, [line]))
        elif item := _LIST_ITEM.match(line):
            blocks.append(Block("item", number, [item.group(1)]))
            flowing = True
        else:
            if continues:
                blocks[-1].lines.append(line)
            else:
                blocks.append(Block("paragraph", number, [line]))
            flowing = True
    return sorted([*blocks, *comments], key=lambda block: block.line)


def _keep_newlines(match: re.Match[str], replacement: str) -> str:
    """``replacement`` expanded for ``match`` and padded to the newlines the match held, so line numbers hold."""
    kept = match.expand(replacement)
    return kept + "\n" * (match.group().count("\n") - kept.count("\n"))


def clean(block: Block) -> str:
    """The block's text with code, links, quoted text and images reduced to placeholders, newlines kept."""
    text = "\n".join(block.lines)
    for regex, replacement in _CLEANERS:
        text = regex.sub(lambda match, kept=replacement: _keep_newlines(match, kept), text)
    return text


def resolve_limit(pattern: Pattern, words: int, message: bool) -> tuple[Kind, int | None, str]:
    """The group ``pattern`` reports under, the hits it may have in ``words`` words of prose, and its limit."""
    if pattern.kind == "strong":
        return "strong", 0, "none"
    if message and pattern.in_message == "none":
        return "density", 0, "none in a message"
    if pattern.kind == "read":
        return "read", None, "read each"
    if pattern.per_1000 is None:
        return "density", pattern.cap, f"{pattern.cap} per file" if pattern.cap else "none"
    allowed = math.ceil(pattern.per_1000 * max(words, SHORT_TEXT_WORDS) / 1000)
    limit = f"{pattern.per_1000:g} per 1,000 words"
    if pattern.cap is not None:
        return "density", min(allowed, pattern.cap), f"{limit}, {pattern.cap} per file"
    return "density", allowed, limit


def _sentences(body: str) -> list[tuple[int, str]]:
    starts = [0, *(match.end() for match in _SENTENCE_END.finditer(body))]
    return [(start, body[start:end]) for start, end in zip(starts, [*starts[1:], len(body)], strict=True)]


def scan(text: str, message: bool = False) -> Scan:
    """Every pattern's hits in ``text`` by line, with the prose word count the limits are scaled by."""
    cleaned = [(block, clean(block)) for block in parse_blocks(text)]
    prose = [(block, body) for block, body in cleaned if block.kind in _BLOCK_KINDS["prose"]]
    words = sum(len(body.split()) for _, body in prose)
    rows = []
    for pattern in PATTERNS:
        if message and pattern.in_message == "skip":
            continue
        regex = re.compile(pattern.regex)
        hits = [
            block.line + body.count("\n", 0, match.start())
            for block, body in cleaned
            if block.kind in _BLOCK_KINDS[pattern.where]
            for match in regex.finditer(body.replace("\n", " "))
        ]
        kind, allowed, limit = resolve_limit(pattern, words, message)
        rows.append(Row(pattern.name, kind, hits, allowed, limit, pattern.note))

    sentences = [
        (block, block.line + body.count("\n", 0, start), sentence)
        for block, body in prose
        for start, sentence in _sentences(body)
    ]
    lengths = [len(sentence.split()) for block, _, sentence in sentences if block.kind == "paragraph"]
    rows += [
        Row(
            f"sentence of {LONG_SENTENCE_WORDS}+ words",
            "density",
            [
                line
                for block, line, sentence in sentences
                if block.kind == "paragraph" and len(sentence.split()) >= LONG_SENTENCE_WORDS
            ],
            math.ceil(LONG_SENTENCE_SHARE * len(lengths)),
            f"{LONG_SENTENCE_SHARE:.0%} of the {len(lengths)} paragraph sentences",
            "Human-docs has 2%, the theory page 7%, the release notes 15%. List items are not counted.",
        ),
        Row(
            "sentence with two asides",
            "density",
            [line for _, line, sentence in sentences if len(_ASIDE.findall(sentence)) >= 2],
            math.ceil(CROWDED_SENTENCES_PER_1000 * max(words, SHORT_TEXT_WORDS) / 1000),
            f"{CROWDED_SENTENCES_PER_1000} per 1,000 words",
            "A list item is a block of its own, so two items are never read as one sentence.",
        ),
        Row(
            "paragraph with two contrasts",
            "density",
            [block.line for block, body in prose if len(_CONTRAST.findall(body.replace("\n", " "))) >= 2],
            0,
            "none",
            "A paragraph or list item that the contrast expression hits twice or more. Human-docs has one, with two "
            "corrective contrasts.",
        ),
        Row(
            f"paragraph over {LONG_PARAGRAPH_WORDS} words",
            "read",
            [block.line for block, body in prose if len(body.split()) > LONG_PARAGRAPH_WORDS],
            None,
            "read each",
            "A derivation may need the length; a paragraph of unrelated numbers does not.",
        ),
    ]
    return Scan(words, lengths, [row for row in rows if row.hits])


def report(name: str, result: Scan, with_notes: bool) -> list[str]:
    """The report for one file: a summary line, the rows that hit under their group, and the totals."""
    mean = sum(result.sentence_lengths) / len(result.sentence_lengths) if result.sentence_lengths else 0.0
    out = [
        f"{name}: {result.words:,} words of prose, {len(result.sentence_lengths)} paragraph sentences, "
        f"mean {mean:.1f} words"
    ]
    if result.words < SHORT_TEXT_WORDS:
        out[0] += f"; limits use the allowance of {SHORT_TEXT_WORDS} words"
    for kind, title in _GROUP_TITLES.items():
        group = [row for row in result.rows if row.kind == kind]
        if group:
            out.append(f"  {title}")
        for row in group:
            per_line = {line: row.hits.count(line) for line in dict.fromkeys(row.hits)}
            lines = " ".join(f"{line}x{count}" if count > 1 else str(line) for line, count in per_line.items())
            rate = f"{1000 * len(row.hits) / result.words:.1f}" if result.words else "-"
            limit = ""
            if kind == "density":
                limit = f"  limit {row.limit}" + (f" = {row.allowed}" if row.allowed else "")
                limit += "  OVER" if row.over else "  ok"
            out.append(f"    {row.name:<28}{len(row.hits):>4}  {rate:>5}/1k{limit}")
            out.append(f"        lines {lines}")
            if with_notes and row.note:
                out.append(f"        note  {row.note}")
    strong = sum(len(row.hits) for row in result.rows if row.kind == "strong")
    over = sum(row.over for row in result.rows if row.kind == "density")
    out.append(f"  {strong} strong tell{'s' * (strong != 1)}, {over} pattern{'s' * (over != 1)} over the limit")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Count the voice skill's writing patterns in markdown or text files. Exits 1 on a strong tell."
    )
    parser.add_argument("files", nargs="+", help="Markdown or text files to read; - reads standard input.")
    parser.add_argument(
        "--message",
        action="store_true",
        help="The text is a PR or issue comment, a commit message or release notes: no bold, and changelog "
        "words are not reported.",
    )
    parser.add_argument("--notes", action="store_true", help="Print under each row what reading its hits showed.")
    args = parser.parse_args(argv)
    texts = []
    for name in args.files:
        try:
            texts.append((name, sys.stdin.read() if name == "-" else Path(name).read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError) as error:
            parser.error(f"cannot read {name} ({error}); pass a UTF-8 markdown or text file, or - for standard input")
    strong = False
    for name, text in texts:
        result = scan(text, args.message)
        sys.stdout.write("\n".join(report(name, result, args.notes)) + "\n")
        strong |= any(row.kind == "strong" for row in result.rows)
    return int(strong)


if __name__ == "__main__":
    sys.exit(main())
