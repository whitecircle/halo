#!/usr/bin/env python
"""The real-weights MTP rows' stream and bars, which no GPU tier reaches without weights: each bar trips on the reading it
exists for, and the rows are the ones the bars were set on."""

import pytest

from tests.common.mtp_real_rows import (
    AcceptanceBars,
    ArmCost,
    CostBars,
    acceptance_checks,
    cost_checks,
    first_turn_rows,
    off_by_one_rows_dataset,
    prometheus_total,
    rows_dataset,
)

_TV = "eval_mtp_acceptance_tv"
_GREEDY = "eval_mtp_acceptance_greedy"
_SCORED = "eval_mtp_scored_positions"


def _reading(step: int, tv: float, greedy: float = 0.65, scored: int = 4954) -> dict:
    return {"step": step, _TV: tv, _GREEDY: greedy, _SCORED: scored}


class _Tokens:
    """Each word one id, in order of first use; a prompt renders as its words plus a generation marker."""

    def __init__(self):
        self.ids: dict[str, int] = {}

    def _encode(self, text: str) -> list[int]:
        return [self.ids.setdefault(word, len(self.ids)) for word in text.split()]

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_dict):
        assert tokenize and add_generation_prompt and not return_dict
        return self._encode(messages[0]["content"]) + self._encode("<gen>")

    def __call__(self, text, add_special_tokens):
        assert not add_special_tokens
        return {"input_ids": self._encode(text)}


def _conversation(user: str, assistant: str, first_role: str = "user") -> dict:
    return {
        "messages": [{"role": first_role, "content": user}, {"role": "assistant", "content": assistant}],
    }


def test_first_turn_rows_keep_a_long_first_reply_cut_to_length():
    tokens = _Tokens()
    conversations = [
        _conversation("too long a prompt for the row at all", "a b c d e f g h"),
        _conversation("short question", "too few"),
        _conversation("brief", "a b c d e f g h", first_role="system"),
        _conversation("short question", "a b c d e f g h"),
    ]
    rows = first_turn_rows(conversations, tokens, 1, length=6, max_prompt=4)
    assert len(rows) == 1
    ids, start = rows[0]
    assert start == 3, "the prompt's two words and its generation marker"
    assert len(ids) == 6
    assert ids[:2] == [tokens.ids["short"], tokens.ids["question"]], "the row the three before it were dropped for"


def test_first_turn_rows_raise_when_the_stream_runs_short():
    with pytest.raises(ValueError, match="Only 0 of the 1"):
        first_turn_rows([_conversation("q", "a")], _Tokens(), 1, length=6, max_prompt=4)


def test_a_row_masks_its_prompt_and_trains_its_reply():
    row = rows_dataset([([1, 2, 3, 4, 5], 2)])[0]
    assert row["labels"] == [-100, -100, 3, 4, 5]
    assert row["attention_mask"] == [1] * 5


def test_the_off_by_one_row_labels_each_position_with_the_token_before_it():
    """The head at index t targets ``labels[t + 2]``, which then holds token ``t + 1``: the one it reads as input."""
    labels = off_by_one_rows_dataset([([1, 2, 3, 4, 5], 2)])[0]["labels"]
    assert labels == [-100, -100, 2, 3, 4]
    assert labels[1 + 2] == [1, 2, 3, 4, 5][1 + 1]


_PASSING = AcceptanceBars(
    first_tv=0.776, first_greedy=0.653, last_tv_at_least=0.815, gate_stops=False, scored_positions=4954
)


def test_a_fit_reaching_every_bar_passes_each_check():
    checks = acceptance_checks([_reading(0, 0.776, 0.653), _reading(150, 0.82, 0.67)], False, _PASSING)
    assert all(checks.values()) and len(checks) == 6


@pytest.mark.parametrize(
    ("readings", "stopped", "failed"),
    [
        ([_reading(0, 0.776, 0.653)], False, "scored_before_the_first_step_and_at_the_last"),
        ([_reading(30, 0.776, 0.653), _reading(150, 0.82)], False, "scored_before_the_first_step_and_at_the_last"),
        ([_reading(0, 0.70, 0.653), _reading(150, 0.82)], False, "first_acceptance_is_the_measured_start"),
        ([_reading(0, 0.776, 0.60), _reading(150, 0.82)], False, "first_greedy_is_the_measured_start"),
        ([_reading(0, 0.776, 0.653), _reading(150, 0.80)], False, "last_acceptance_reached_its_bar"),
        ([_reading(0, 0.776, 0.653), _reading(150, 0.82)], True, "gate_verdict"),
        ([_reading(0, 0.776, 0.653, scored=4000), _reading(150, 0.82)], False, "every_reading_scored_the_stream"),
    ],
)
def test_each_acceptance_bar_trips_on_the_reading_it_exists_for(readings, stopped, failed):
    checks = acceptance_checks(readings, stopped, _PASSING)
    assert checks[failed] is False
    assert [name for name, ok in checks.items() if not ok] == [failed]


def test_a_collapsed_head_is_held_under_its_ceiling_and_a_gain_over_its_start():
    collapsed = AcceptanceBars(last_tv_at_most=0.05, gate_stops=True)
    assert all(acceptance_checks([_reading(0, 0.776), _reading(150, 0.04)], True, collapsed).values())
    assert not acceptance_checks([_reading(0, 0.776), _reading(150, 0.10)], True, collapsed)[
        "last_acceptance_collapsed"
    ]
    gained = AcceptanceBars(tv_gain_at_least=0.013)
    assert acceptance_checks([_reading(0, 0.869), _reading(150, 0.89)], None, gained)["acceptance_gained_its_bar"]
    assert not acceptance_checks([_reading(0, 0.869), _reading(150, 0.875)], None, gained)["acceptance_gained_its_bar"]


def test_a_recorded_row_holds_only_that_it_scored():
    checks = acceptance_checks([_reading(0, 0.2), _reading(150, 0.1)], True, AcceptanceBars())
    assert checks == {"scored_before_the_first_step_and_at_the_last": True}


_GIB = 1024**3


def _arms(**overrides) -> dict[str, ArmCost]:
    seconds = {
        "off": 10.0,
        "off_again": 10.1,
        "joint_hard": 12.0,
        "joint_soft": 12.1,
        "detached_hard": 11.0,
        "detached_soft": 11.1,
        "head_only_hard": 5.0,
        "head_only_soft": 5.8,
        **overrides,
    }
    return {name: ArmCost(value, 100 * _GIB) for name, value in seconds.items()}


def test_measured_costs_inside_every_bar_pass():
    assert all(cost_checks(_arms(), _GIB, CostBars()).values())


@pytest.mark.parametrize(
    ("overrides", "failed"),
    [
        ({"off_again": 11.0}, "the_off_arm_repeats"),
        ({"joint_hard": 13.5, "joint_soft": 13.5}, "joint_costs_at_most_its_bar"),
        ({"joint_soft": 12.5}, "soft_costs_what_hard_does_where_the_trunk_takes_a_gradient"),
        ({"detached_soft": 11.5}, "soft_costs_what_hard_does_where_the_trunk_takes_a_gradient"),
        ({"head_only_soft": 6.5}, "head_only_costs_at_most_its_bar"),
    ],
)
def test_each_cost_bar_trips_on_the_arm_it_exists_for(overrides, failed):
    checks = cost_checks(_arms(**overrides), _GIB, CostBars())
    assert [name for name, ok in checks.items() if not ok] == [failed]


def test_a_soft_head_only_step_may_cost_more_than_a_hard_one():
    """A frozen trunk's next-token loss takes no gradient until the route gives it one."""
    assert cost_checks(_arms(head_only_hard=4.0, head_only_soft=5.8), _GIB, CostBars())[
        "soft_costs_what_hard_does_where_the_trunk_takes_a_gradient"
    ]


def test_an_arm_past_the_memory_allowance_trips_the_bar():
    arms = _arms()
    arms["joint_soft"] = ArmCost(12.1, 102 * _GIB)
    assert not cost_checks(arms, _GIB, CostBars())["no_arm_peaks_past_its_allowance"]
    assert cost_checks(arms, 2 * _GIB, CostBars())["no_arm_peaks_past_its_allowance"]


_METRICS = """# HELP vllm:spec_decode_num_drafts_total Number of spec decoding drafts.
# TYPE vllm:spec_decode_num_drafts_total counter
vllm:spec_decode_num_drafts_total{engine="0",model_name="m"} 200.0
vllm:spec_decode_num_drafts_total{engine="1",model_name="m"} 100.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="m",position="0"} 150.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="m",position="1"} 7.0
vllm:spec_decode_num_accepted_tokens_per_pos_created{engine="0",model_name="m",position="0"} 1.7e9
"""


def test_a_counter_sums_its_engines_and_takes_only_the_position_asked_for():
    assert prometheus_total(_METRICS, "vllm:spec_decode_num_drafts") == 300.0
    assert prometheus_total(_METRICS, "vllm:spec_decode_num_accepted_tokens_per_pos", position="0") == 150.0
    assert prometheus_total(_METRICS, "vllm:absent") == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
