"""The rollout argument bundles ``AsyncRolloutConfig`` mixes in: sampling settings and token budgets."""

from dataclasses import dataclass, field

from halo.config_validation import ValidatedConfig, present, require_finite, require_int
from halo.environments.rollout_config import REASONING_END_TOKEN_EXAMPLES, RolloutConfig


@dataclass
class RolloutSamplingArguments(ValidatedConfig):
    rollout_temperature: float = field(
        default=RolloutConfig.temperature, metadata={"help": "Temperature for rollout generation."}
    )
    rollout_top_p: float = field(
        default=RolloutConfig.top_p, metadata={"help": "Top-p (nucleus) sampling for rollout generation."}
    )
    rollout_top_k: int = field(
        default=RolloutConfig.top_k,
        metadata={
            "help": "Top-k sampling for rollouts: -1 (off) or an int >= 1, since SGLang rejects 0. Sent on every "
            "request, like rollout_min_p and rollout_repetition_penalty: both engines fill an omitted one from "
            "generation_config.json."
        },
    )
    rollout_min_p: float = field(
        default=RolloutConfig.min_p,
        metadata={
            "help": "Min-p sampling for rollout generation, in [0, 1]; 0 = off. vLLM rejects every request "
            "with min_p > 0 on a server running speculative decoding (MTP)."
        },
    )
    rollout_repetition_penalty: float = field(
        default=RolloutConfig.repetition_penalty,
        metadata={
            "help": "Repetition penalty for rollouts, in (0, 2]; 1 = off. Refused other than 1 under a "
            "sequence-level ratio (isr_geo_band_*, isr_opsm_delta), since the trainer's recomputed log-probs do not "
            "carry it."
        },
    )

    def _validate_fields(self) -> None:
        super()._validate_fields()
        owner = type(self).__name__
        # The trainer scores at the sampling temperature, so a 0 is a ZeroDivisionError mid-step.
        require_finite(owner, gt=0, rollout_temperature=self.rollout_temperature)
        # Sent verbatim; outside these ranges the server rejects every rollout request (SGLang's, the narrower of the
        # two engines', refuses top_k 0 and a penalty above 2).
        require_finite(owner, gt=0, le=1, rollout_top_p=self.rollout_top_p)
        # True would pass as 1 and sample every rollout greedily; a fractional top-k fails every request.
        require_int(owner, rollout_top_k=self.rollout_top_k)
        if self.rollout_top_k != -1:
            require_int(owner, ge=1, why=" or -1 (off)", rollout_top_k=self.rollout_top_k)
        require_finite(owner, ge=0, le=1, rollout_min_p=self.rollout_min_p)
        require_finite(owner, gt=0, le=2, rollout_repetition_penalty=self.rollout_repetition_penalty)


@dataclass
class RolloutBudgetArguments(ValidatedConfig):
    """The token budgets of one turn, one episode and one training row, and the reasoning close a turn cap cuts at."""

    rollout_max_tokens: int = field(
        default=RolloutConfig.max_tokens,
        metadata={
            "help": "Most tokens one turn generates (one /chat/completions call), checked against the server at "
            "startup. The trajectory accumulates across turns, bounded by rollout_max_episode_tokens where set and "
            "by the model's context window, which it fails past. "
            "See agent-docs/training-methods/async-grpo/episodes.md#trajectory-length."
        },
    )
    rollout_max_episode_tokens: int | None = field(
        default=RolloutConfig.max_episode_tokens,
        metadata={
            "help": "Most tokens one episode may sample over all its turns, reasoning included; unset = max_turns × "
            "rollout_max_tokens. Enforced by the engine per turn, never stated to the model, and at least "
            "rollout_max_tokens. See agent-docs/training-methods/async-grpo/episodes.md#episode-output-budget."
        },
    )
    rollout_max_thinking_tokens: int | None = field(
        default=RolloutConfig.max_thinking_tokens,
        metadata={
            "help": "Per-turn reasoning-token cap (vLLM thinking_token_budget) that then forces the answer out of "
            "the rest of the turn; a level's smaller enforced thinking_tokens wins, and under a soft level it is "
            "the ceiling. Needs a reasoning parser on the vLLM server; unset leaves reasoning to the levels' "
            "budgets. See agent-docs/training-methods/async-grpo/reasoning.md#reasoning-budget."
        },
    )
    rollout_max_answer_tokens: int | None = field(
        default=RolloutConfig.max_answer_tokens,
        metadata={
            "help": "Most a turn may generate past its reasoning cap, never stated to the model: an int in "
            "[1, rollout_max_tokens), unset = rollout_max_tokens alone bounds the turn. "
            "See agent-docs/training-methods/async-grpo/reasoning.md#answer-room."
        },
    )
    rollout_reasoning_end_token: str = field(
        default="</think>",
        metadata={
            "help": "The string the server's reasoning parser ends reasoning with, encoded as vLLM encodes it "
            f"({REASONING_END_TOKEN_EXAMPLES}). Wherever a vLLM thinking budget can bind, the trainer gives a "
            "forced run of its ids ratio 0 in the loss. "
            "See agent-docs/training-methods/async-grpo/objective.md#forced-reasoning-closes."
        },
    )
    max_train_row_tokens: int | None = field(
        default=None,
        metadata={
            "help": "Longest training row (prompt + completion tokens) a training rank takes, above "
            "rollout_max_tokens; unset = no cap beyond the context check. A per-turn row over it leaves the batch, "
            "a whole-trajectory row trains at zero weight. "
            "See agent-docs/training-methods/async-grpo/episodes.md#training-row-cap."
        },
    )

    def _validate_fields(self) -> None:
        super()._validate_fields()
        owner = type(self).__name__
        # Doubles as the dr_grpo loss normalizer, and is the request's max_tokens, which no engine reads as a fraction.
        require_int(owner, ge=1, rollout_max_tokens=self.rollout_max_tokens)
        max_tokens = self.rollout_max_tokens
        # A thinking cap at rollout_max_tokens leaves the turn no room for its answer or tool call, and 0 would be sent
        # as 1 while counting as no cap; an answer bound at or above it shrinks no turn, and below 1 leaves no room.
        require_int(
            owner,
            ge=1,
            lt=max_tokens,
            why=" (below rollout_max_tokens)",
            **present(
                rollout_max_thinking_tokens=self.rollout_max_thinking_tokens,
                rollout_max_answer_tokens=self.rollout_max_answer_tokens,
            ),
        )
        # Below one turn's cap the first turn could never use the per-turn budget the run states. A turn's
        # total is at most rollout_max_tokens, so this also holds one reasoning cap plus its answer room.
        require_int(
            owner,
            ge=max_tokens,
            why=" (at least rollout_max_tokens)",
            **present(rollout_max_episode_tokens=self.rollout_max_episode_tokens),
        )
        # A row is a turn's prompt plus its completion, which may run to rollout_max_tokens: a cap at or below that
        # leaves out every turn that used its budget, a length bias rather than a memory bound.
        require_int(
            owner,
            gt=max_tokens,
            why=" (above rollout_max_tokens)",
            **present(max_train_row_tokens=self.max_train_row_tokens),
        )
