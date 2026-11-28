"""A deterministic weighted mix of DocumentTokenizer folders as one map-style dataset whose sample ``i`` is a pure
function of ``i``, the sources, ``max_length`` and the seed: weights become integer quotas over a fixed pattern, and each
source's windows are visited in a keyed permutation that changes per pass. Nothing it holds grows with the token count.
The mix and its held-out folders are built on every rank at once, the ranks agreeing that they read the same files."""

import hashlib
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

from torch.utils.data import Dataset

from halo.data.sources.token_shards import TokenShards, ensure_open_file_limit
from halo.log import KEY_PREVIEW_COUNT
from halo.runtime.collectives import combine_across_ranks
from halo.runtime.consensus import agree_across_ranks, refusal_text, reject_across_ranks
from halo.runtime.ranks import get_local_world_size, is_global_main_process

logger = logging.getLogger(__name__)

# Samples in one repeat of the pattern. Each quota is a whole number of them, so a share holds to 1/PATTERN_PERIOD.
PATTERN_PERIOD = 4096
# Raised whenever the sample-to-window function changes, so no checkpoint resumes onto another order.
ORDER_VERSION = 1
_UINT64 = (1 << 64) - 1
_FEISTEL_ROUNDS = 4

_Built = TypeVar("_Built")


@dataclass(frozen=True)
class TokenSource:
    """A named DocumentTokenizer output folder; ``weight`` is relative to the other sources', ``None`` for the share of
    the windows the folder holds."""

    name: str
    path: str
    weight: float | None = None


def _mix64(value: int) -> int:
    """SplitMix64's output function with its increment, a bijection of 64-bit values (Steele, Lea and Flood, "Fast
    splittable pseudorandom number generators", 2014)."""
    value = (value + 0x9E3779B97F4A7C15) & _UINT64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64
    return value ^ (value >> 31)


def keyed_permutation(index: int, size: int, key: int) -> int:
    """Entry ``index`` of the permutation of ``range(size)`` that ``key`` picks, with no table: a 4-round Feistel
    network on the smallest even bit width holding ``size``, applied again to a value of ``size`` or more until one
    falls below (cycle walking; Black and Rogaway, "Ciphers with arbitrary finite domains", 2002)."""
    if not 0 <= index < size:
        raise IndexError(f"index {index} is outside a permutation of {size}")
    half = max(1, ((size - 1).bit_length() + 1) // 2)
    mask = (1 << half) - 1
    round_keys = [_mix64((key + offset) & _UINT64) for offset in range(_FEISTEL_ROUNDS)]
    value = index
    while True:
        left, right = value >> half, value & mask
        for round_key in round_keys:
            left, right = right, left ^ (_mix64(right ^ round_key) & mask)
        value = (left << half) | right
        if value < size:
            return value


def _pass_key(seed: int, name: str, pass_index: int) -> int:
    """The permutation key of one pass over a source; the name comes last, so no two triples spell the same bytes."""
    payload = f"{seed}\0{pass_index}\0{name}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _quotas(weights: Sequence[float]) -> list[int]:
    """Whole samples of one pattern period per source, by largest remainder; equal remainders go to the earlier
    source."""
    total = math.fsum(weights)
    exact = [weight * PATTERN_PERIOD / total for weight in weights]
    quotas = [math.floor(share) for share in exact]
    by_remainder = sorted(range(len(weights)), key=lambda source: (quotas[source] - exact[source], source))
    for source in by_remainder[: PATTERN_PERIOD - sum(quotas)]:
        quotas[source] += 1
    return quotas


def _pattern(quotas: Sequence[int]) -> tuple[list[int], list[int]]:
    """Each pattern slot's source and its rank among that source's slots. Every slot goes to the source furthest behind
    its share (the earlier on a tie), so one period holds each quota exactly and a shorter run stays near every share."""
    credit = [0] * len(quotas)
    taken = [0] * len(quotas)
    slot_source, slot_rank = [], []
    for _ in range(PATTERN_PERIOD):
        credit = [held + quota for held, quota in zip(credit, quotas, strict=True)]
        chosen = max(range(len(quotas)), key=lambda source: (credit[source], -source))
        credit[chosen] -= PATTERN_PERIOD
        slot_source.append(chosen)
        slot_rank.append(taken[chosen])
        taken[chosen] += 1
    return slot_source, slot_rank


def _folder_fingerprint(shards: TokenShards) -> dict:
    """A folder's token width, content digest and files with their token counts and tokenizers."""
    return {
        "token_bytes": shards.token_bytes,
        "content": shards.content_digest(),
        "files": {file.name: [file.num_tokens, file.tokenizer] for file in shards.files},
    }


class TokenMix(Dataset):
    """``num_steps × rows_per_step`` packed rows drawn from ``sources`` by weight, for a run whose optimizer steps each
    read ``rows_per_step`` of them in order. Sample ``i`` takes pattern slot ``i % PATTERN_PERIOD``, whose source ``s``
    makes its draw ``j = (i // PATTERN_PERIOD) * quota_s + rank``; draw ``j`` reads window
    ``keyed_permutation(j % W_s, W_s, key(seed, name, j // W_s))`` of the source's ``W_s``, so a source drawn past its
    windows starts another pass in a new order."""

    column_names = TokenShards.column_names

    def __init__(
        self, sources: Sequence[TokenSource], *, max_length: int, num_steps: int, rows_per_step: int, seed: int
    ):
        self.sources = tuple(sources)
        self.max_length = max_length
        self.rows_per_step = rows_per_step
        self.num_samples = num_steps * rows_per_step
        self.seed = seed
        self.shards = tuple(TokenShards(source.path, max_length) for source in self.sources)
        ensure_open_file_limit(self.shards)
        weighted = self.sources[0].weight is not None
        weights = [
            source.weight if weighted else shards.num_windows
            for source, shards in zip(self.sources, self.shards, strict=True)
        ]
        self.quotas = _quotas(weights)
        undrawn = [source.name for source, quota in zip(self.sources, self.quotas, strict=True) if quota == 0]
        if undrawn:
            given = "its weight" if weighted else "its share of the windows (no weights given)"
            raise ValueError(
                f"Token sources {undrawn} would never be drawn: {given} is under one sample in {PATTERN_PERIOD}. "
                f"Raise their weights, or drop them."
            )
        self._slot_source, self._slot_rank = _pattern(self.quotas)
        if is_global_main_process():
            for source, shards, quota in zip(self.sources, self.shards, self.quotas, strict=True):
                draws = self.num_samples * quota / PATTERN_PERIOD
                covered = shards.num_windows * max_length / sum(file.num_tokens for file in shards.files)
                logger.info(
                    f"Token source {source.name!r}: {quota / PATTERN_PERIOD:.2%} of samples, {shards.num_windows:,} "
                    f"windows of {max_length} tokens ({covered:.1%} of its tokens, the rest in file tails), "
                    f"{draws / shards.num_windows:.3g} passes over this run."
                )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        source, window = self.locate(index)
        return self.shards[source][window]

    def locate(self, index: int) -> tuple[int, int]:
        """``(source, window)``: the source's position in ``sources`` and the window of it that sample ``index``
        reads."""
        if not 0 <= index < self.num_samples:
            raise IndexError(f"sample {index} is outside the {self.num_samples} samples of the token mix")
        cycle, slot = divmod(index, PATTERN_PERIOD)
        source = self._slot_source[slot]
        windows = self.shards[source].num_windows
        pass_index, position = divmod(cycle * self.quotas[source] + self._slot_rank[slot], windows)
        return source, keyed_permutation(
            position, windows, _pass_key(self.seed, self.sources[source].name, pass_index)
        )

    def fingerprint(self) -> dict:
        """What decides the rows each step reads, as JSON: the order version, ``max_length``, seed and rows per step (a
        resume skips the trained steps by that count), and per source in order its name, quota, token width, files and
        content digest. Paths and the step count stay out: a copy of the same files elsewhere, or a longer run, reads
        the same rows at step ``t``."""
        return {
            "order_version": ORDER_VERSION,
            "max_length": self.max_length,
            "seed": self.seed,
            "rows_per_step": self.rows_per_step,
            "sources": [
                {"name": source.name, "quota": quota, **_folder_fingerprint(shards)}
                for source, shards, quota in zip(self.sources, self.shards, self.quotas, strict=True)
            ],
        }


def _file_changes(name: str, saved: dict, live: dict) -> list[str]:
    changes = []
    removed, added = sorted(saved.keys() - live.keys()), sorted(live.keys() - saved.keys())
    if removed:
        changes.append(f"source {name!r} lost files {removed[:KEY_PREVIEW_COUNT]}")
    if added:
        changes.append(f"source {name!r} gained files {added[:KEY_PREVIEW_COUNT]}")
    for file in sorted(saved.keys() & live.keys()):
        (saved_tokens, saved_tokenizer), (live_tokens, live_tokenizer) = saved[file], live[file]
        if saved_tokens != live_tokens:
            changes.append(f"source {name!r} file {file} holds {live_tokens} tokens, not {saved_tokens}")
        if saved_tokenizer != live_tokenizer:
            changes.append(f"source {name!r} file {file} names tokenizer {live_tokenizer!r}, not {saved_tokenizer!r}")
    return changes


def _token_mix_changes(saved: dict, live: dict) -> list[str]:
    """Every field in which fingerprint ``live`` differs from ``saved``, as ``'<field> <saved> → <live>'`` phrases;
    an empty list when they match."""
    changes = [
        f"{field} {saved.get(field)} → {value}"
        for field, value in live.items()
        if field != "sources" and saved.get(field) != value
    ]
    saved_names = [source["name"] for source in saved.get("sources", [])]
    live_names = [source["name"] for source in live["sources"]]
    if saved_names != live_names:
        return [*changes, f"sources {saved_names} → {live_names}"]
    for saved_source, live_source in zip(saved["sources"], live["sources"], strict=True):
        name = live_source["name"]
        changes += [
            f"source {name!r} {field} {saved_source.get(field)} → {value}"
            for field, value in live_source.items()
            if field not in ("name", "files") and saved_source.get(field) != value
        ]
        changes += _file_changes(name, saved_source.get("files", {}), live_source["files"])
    return changes


def verify_resumed_token_mix(saved: dict, live: dict) -> None:
    """Raise when the mix a checkpoint was trained on (its saved fingerprint) differs from the live one, naming each
    changed field: the rows each step reads follow from exactly these, so the resumed run would read other rows."""
    changes = _token_mix_changes(saved, live)
    if changes:
        more = f"; and {len(changes) - KEY_PREVIEW_COUNT} more" if len(changes) > KEY_PREVIEW_COUNT else ""
        raise ValueError(
            f"The token mix differs from the one the checkpoint was trained on: "
            f"{'; '.join(changes[:KEY_PREVIEW_COUNT])}{more}. The resumed steps would not read the rows the run would "
            f"have read, and a mix cannot change on resume. Restore the checkpoint's token_sources, their files, "
            f"max_length and seed (and the Halo version, for order_version), and its rows_per_step, "
            f"per_device_train_batch_size × gradient_accumulation_steps × data-parallel size (the accumulation left "
            f"out under pipeline parallelism): to change the layout, halve one factor and double another. Or start a "
            f"new run."
        )


def _describe_views(views: list[dict], local_world: int) -> str:
    """Which ranks (and nodes) see a view other than rank 0's, and how it differs."""
    groups: dict[str, list[int]] = {}
    for rank, view in enumerate(views):
        groups.setdefault(repr(view), []).append(rank)
    described = []
    for ranks in groups.values():
        if ranks[0] == 0:
            continue
        nodes = sorted({rank // local_world for rank in ranks})
        changes = _token_mix_changes(views[0], views[ranks[0]])
        described.append(
            f"ranks {ranks[:KEY_PREVIEW_COUNT]} (nodes {nodes[:KEY_PREVIEW_COUNT]}) differ from rank 0 in "
            f"{'; '.join(changes[:KEY_PREVIEW_COUNT])}"
        )
    return " | ".join(described)


def _build_agreed_across_ranks(build: Callable[[], _Built], view: Callable[[_Built], dict], knob: str) -> _Built:
    """COLLECTIVE: every rank's ``build()``, after one rank's failure raises on every rank and the ranks agree on the
    ``view`` of what they built, a fingerprint shaped as :meth:`TokenMix.fingerprint`; a rank seeing other files than
    its peers under ``knob`` would read other rows. One all-reduce of the view's digest, and on a mismatch a rank-0
    combine naming who differs."""
    built, seen, failure = None, None, None
    try:
        built = build()
        seen = view(built)
    except Exception as error:  # a rank raising alone would strand its peers in the reduce below
        failure = refusal_text(error)
    reject_across_ranks(failure, f"Reading {knob}", ValueError)
    if agree_across_ranks(seen):
        return built
    local_world = get_local_world_size()
    divergence = combine_across_ranks(
        seen, lambda views: _describe_views(views, local_world), what=f"the {knob} views"
    )
    raise ValueError(
        f"Ranks see different files or tokens under {knob}: {divergence}. A folder still being written, a node "
        f"whose view of a shared filesystem is stale, or a node-local copy that differs would give each rank other "
        f"rows. Let the writer finish or re-sync the copies, then relaunch."
    )


def build_token_mix_across_ranks(
    sources: Sequence[TokenSource], *, max_length: int, num_steps: int, rows_per_step: int, seed: int
) -> TokenMix:
    """COLLECTIVE: every rank's :class:`TokenMix`, once the ranks agree on its fingerprint."""
    return _build_agreed_across_ranks(
        lambda: TokenMix(sources, max_length=max_length, num_steps=num_steps, rows_per_step=rows_per_step, seed=seed),
        TokenMix.fingerprint,
        "token_sources",
    )


def _held_out_shards(sources: Sequence[TokenSource], mix: TokenMix) -> dict[str, TokenShards]:
    held_out = {source.name: TokenShards(source.path, mix.max_length) for source in sources}
    ensure_open_file_limit([*mix.shards, *held_out.values()])
    return held_out


def _held_out_fingerprint(held_out: dict[str, TokenShards]) -> dict:
    return {"sources": [{"name": name, **_folder_fingerprint(shards)} for name, shards in held_out.items()]}


def build_held_out_shards_across_ranks(sources: Sequence[TokenSource], *, mix: TokenMix) -> dict[str, TokenShards]:
    """COLLECTIVE: each held-out folder of ``mix``'s run by name, in windows of its ``max_length``, once the ranks
    agree on the folders' fingerprint; the open-file limit counts the mix's open files beside theirs."""
    return _build_agreed_across_ranks(
        lambda: _held_out_shards(sources, mix), _held_out_fingerprint, "eval_token_sources"
    )
