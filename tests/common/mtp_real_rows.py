"""The rows a released head's real-weights fit trains on and the bars its measurements answer to. The bars read numbers
alone, so a CPU test holds each to a reading that must trip it."""

from __future__ import annotations

import glob
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass

from datasets import Dataset, load_dataset
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from halo.env import env_str
from halo.models.mtp.acceptance import MTP_ACCEPTANCE_GREEDY, MTP_ACCEPTANCE_TV, MTP_SCORED_POSITIONS
from tests.common.models import QWEN3_5_MOE_35B, QWEN3_6_MOE_35B

REAL_MODEL = env_str("HALO_TEST_MTP_REAL_MODEL", QWEN3_6_MOE_35B)
REAL_STALE_HEAD = env_str("HALO_TEST_MTP_REAL_STALE_HEAD", QWEN3_5_MOE_35B)
# The numbers below were read off these weights, and hold for no other.
REAL_BARS_APPLY = REAL_MODEL == QWEN3_6_MOE_35B and REAL_STALE_HEAD == QWEN3_5_MOE_35B
LABEL_IGNORE = -100
ULTRACHAT_DATASET = "HuggingFaceH4/ultrachat_200k"
_READING_TV = f"eval_{MTP_ACCEPTANCE_TV}"
_READING_GREEDY = f"eval_{MTP_ACCEPTANCE_GREEDY}"
_READING_SCORED = f"eval_{MTP_SCORED_POSITIONS}"


def cached_source(source: str) -> str | None:
    """Where ``source`` sits on this machine, or ``None``."""
    if os.path.isdir(source):
        return source
    try:
        return snapshot_download(source, local_files_only=True)
    except LocalEntryNotFoundError:
        return None


def ultrachat_conversations(split: str) -> Dataset:
    """A split's rows, from the local Hub cache. Raises where the dataset is not cached."""
    root = snapshot_download(ULTRACHAT_DATASET, repo_type="dataset", local_files_only=True)
    return load_dataset("parquet", data_files=sorted(glob.glob(os.path.join(root, "data", f"{split}-*.parquet"))))[
        "train"
    ]


def first_turn_rows(
    conversations: Iterable[dict],
    tokenizer,
    count: int,
    *,
    length: int,
    max_prompt: int,
    template_kwargs: dict | None = None,
) -> list[tuple[list[int], int]]:
    """``count`` rows of a first user turn and the reply it got, tokenized apart, as ``(ids, reply_start)``: long enough
    to fill ``length``, with a prompt of at most ``max_prompt``. The SFT pipeline renders a whole conversation and masks
    every turn, so a bar set on these rows does not transfer to it. ``template_kwargs`` reach the chat template."""
    rows: list[tuple[list[int], int]] = []
    for conversation in conversations:
        messages = conversation["messages"]
        if len(messages) < 2 or messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
            continue
        rendered = tokenizer.apply_chat_template(
            messages[:1], tokenize=True, add_generation_prompt=True, return_dict=False, **(template_kwargs or {})
        )
        prompt = list(rendered["input_ids"] if hasattr(rendered, "keys") else rendered)
        if len(prompt) > max_prompt:
            continue
        reply = tokenizer(messages[1]["content"], add_special_tokens=False)["input_ids"]
        if len(prompt) + len(reply) < length:
            continue
        rows.append((prompt + reply[: length - len(prompt)], len(prompt)))
        if len(rows) == count:
            return rows
    raise ValueError(f"Only {len(rows)} of the {count} first-turn rows of {length} tokens were found.")


def rows_dataset(rows: list[tuple[list[int], int]]) -> Dataset:
    """Pre-tokenized rows with each prompt masked."""
    return Dataset.from_dict(
        {
            "input_ids": [ids for ids, _start in rows],
            "labels": [[LABEL_IGNORE] * start + ids[start:] for ids, start in rows],
            "attention_mask": [[1] * len(ids) for ids, _start in rows],
        }
    )


def off_by_one_rows_dataset(rows: list[tuple[list[int], int]]) -> Dataset:
    """Labels one token early, so a head taking its target two ahead fits the token it reads as an input."""
    return Dataset.from_dict(
        {
            "input_ids": [ids for ids, _start in rows],
            "labels": [[LABEL_IGNORE] * start + ids[start - 1 : -1] for ids, start in rows],
            "attention_mask": [[1] * len(ids) for ids, _start in rows],
        }
    )


@dataclass(frozen=True)
class AcceptanceBars:
    """What a fit must read. ``None`` records the quantity without a bar. ``gate_stops`` is the acceptance gate's
    verdict; ``scored_positions`` proves the held-out rows are the ones a bar was set on."""

    first_tv: float | None = None
    first_greedy: float | None = None
    first_window: float = 0.012
    last_tv_at_least: float | None = None
    last_tv_at_most: float | None = None
    tv_gain_at_least: float | None = None
    gate_stops: bool | None = None
    scored_positions: int | None = None


def acceptance_checks(readings: list[dict], gate_stopped: bool | None, bars: AcceptanceBars) -> dict[str, bool]:
    """The verdicts on a fit's ordered eval readings, each entry as the trainer logged it."""
    checks = {"scored_before_the_first_step_and_at_the_last": len(readings) >= 2 and readings[0]["step"] == 0}
    if len(readings) < 2:
        return checks
    first, last = readings[0], readings[-1]
    if bars.scored_positions is not None:
        checks["every_reading_scored_the_stream"] = all(
            entry[_READING_SCORED] == bars.scored_positions for entry in readings
        )
    if bars.first_tv is not None:
        checks["first_acceptance_is_the_measured_start"] = abs(first[_READING_TV] - bars.first_tv) <= bars.first_window
    if bars.first_greedy is not None:
        checks["first_greedy_is_the_measured_start"] = (
            abs(first[_READING_GREEDY] - bars.first_greedy) <= bars.first_window
        )
    if bars.last_tv_at_least is not None:
        checks["last_acceptance_reached_its_bar"] = last[_READING_TV] >= bars.last_tv_at_least
    if bars.last_tv_at_most is not None:
        checks["last_acceptance_collapsed"] = last[_READING_TV] <= bars.last_tv_at_most
    if bars.tv_gain_at_least is not None:
        checks["acceptance_gained_its_bar"] = last[_READING_TV] - first[_READING_TV] >= bars.tv_gain_at_least
    if bars.gate_stops is not None:
        checks["gate_verdict"] = gate_stopped is bars.gate_stops
    return checks


@dataclass(frozen=True)
class ArmCost:
    """One arm's median seconds per step past its warmup and its world's peak allocated bytes."""

    step_seconds: float
    peak_bytes: int


@dataclass(frozen=True)
class CostBars:
    """The bars on a sweep of arms, against the head-less arm named ``off``."""

    joint_at_most: float = 1.30
    head_only_at_most: float = 0.60
    soft_within: float = 0.02
    repeat_within: float = 0.03


def worst_soft_gap(arms: dict[str, ArmCost], scopes: Iterable[str]) -> float:
    return max(abs(arms[f"{scope}_soft"].step_seconds / arms[f"{scope}_hard"].step_seconds - 1) for scope in scopes)


def cost_checks(arms: dict[str, ArmCost], memory_allowance: int, bars: CostBars) -> dict[str, bool]:
    """``arms`` holds ``off``, its repeat ``off_again``, and hard and soft of each scope; ``memory_allowance`` is what a
    head may add to the off arm's peak. A soft head-only step routes a gradient a frozen trunk's next-token loss would
    not take, so only the arms where that loss already takes one hold soft to hard."""
    off = arms["off"]
    return {
        "the_off_arm_repeats": abs(arms["off_again"].step_seconds / off.step_seconds - 1) <= bars.repeat_within,
        "joint_costs_at_most_its_bar": max(arms["joint_hard"].step_seconds, arms["joint_soft"].step_seconds)
        <= bars.joint_at_most * off.step_seconds,
        "soft_costs_what_hard_does_where_the_trunk_takes_a_gradient": (
            worst_soft_gap(arms, ("joint", "detached")) <= bars.soft_within
        ),
        "head_only_costs_at_most_its_bar": max(
            arms["head_only_hard"].step_seconds, arms["head_only_soft"].step_seconds
        )
        <= bars.head_only_at_most * off.step_seconds,
        "no_arm_peaks_past_its_allowance": all(
            arm.peak_bytes <= off.peak_bytes + memory_allowance for arm in arms.values()
        ),
    }


def prometheus_total(metrics: str, name: str, **labels: str) -> float:
    """The sum of counter ``name``'s samples carrying ``labels``, over a server's ``/metrics`` text."""
    pattern = re.compile(rf"^{re.escape(name)}_total(?:\{{(?P<labels>[^}}]*)\}})? (?P<value>\S+)$", re.MULTILINE)
    wanted = [f'{key}="{value}"' for key, value in labels.items()]
    return sum(
        float(match["value"])
        for match in pattern.finditer(metrics)
        if all(pair in (match["labels"] or "") for pair in wanted)
    )
