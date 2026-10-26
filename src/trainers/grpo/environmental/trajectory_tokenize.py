"""Environmental GRPO rollouts as training rows: one whole-trajectory render with the assistant spans located in
it, or under ``train_on_sampled_tokens`` one row per assistant turn over the engine's own ids."""

import logging
from collections.abc import Sequence
from functools import cached_property
from typing import NamedTuple

import torch

from src.environments.episode import RolloutResult
from src.environments.episode_keys import EPISODE_INVALID_KEY, EPISODE_INVALID_REASON_KEY
from src.environments.messages import engine_view
from src.environments.registry import build_environment
from src.environments.rollout_config import (
    REASONING_BUDGET_SOFT_TEMPLATE_VAR,
    REASONING_BUDGET_TEMPLATE_VAR,
    VLLM_BACKEND,
)
from src.log import warn_once
from src.models.context_window import UNSET_MODEL_MAX_LENGTH, get_model_context_window, is_bounded_length
from src.trainers.grpo.environmental.trajectory_spans import TemplateSpanError, locate_assistant_spans

logger = logging.getLogger(__name__)

# Untrainable turns (``Message.untrainable``) that became no negative-only row, over all of them.
UNTRAINABLE_TURNS_ROWLESS_FRAC_KEY = "sampling/untrainable_turns_rowless_frac"
# ``warn_once`` key of the re-render fallback for a rollout that returned no sampled ids.
SAMPLED_IDS_MISSING_WARNING = "sampled_ids_missing"


# Engine MoE routing for one turn: decoded mask + the prompt-token count it is aligned on.
TurnRouting = tuple[torch.Tensor, int | None]


class TurnRow(NamedTuple):
    """One training row; ``negative_only`` tags an untrainable turn's row, kept in the loss only under a negative
    advantage so the turn takes the failure signal and is never rewarded."""

    prompt_ids: torch.Tensor
    completion_ids: torch.Tensor
    completion_mask: torch.Tensor
    sampling_logps: torch.Tensor | None
    turn_routing: TurnRouting | None
    negative_only: bool = False


def single_trajectory_row(tokenized: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> list[TurnRow]:
    """A whole re-tokenized trajectory as one training row, also the fallback of the per-turn split."""
    return [TurnRow(*tokenized, None, None)]


def rollout_template_kwargs(
    rollout_kwargs: dict, reasoning_effort: str | None, reasoning_budget: int | None = None, budget_soft: bool = False
) -> dict:
    """The chat-template kwargs a rollout's requests carry: the run's template variables plus the episode's
    ``reasoning_effort`` and per-turn thinking budget (flagged soft where the engine did not enforce it), so a
    trainer-side render reproduces the engine's."""
    kwargs = dict(rollout_kwargs)
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    if reasoning_budget is not None:
        kwargs[REASONING_BUDGET_TEMPLATE_VAR] = reasoning_budget
        if budget_soft:
            kwargs[REASONING_BUDGET_SOFT_TEMPLATE_VAR] = True
    return kwargs


class TrajectoryTokenizeMixin:
    """Chat-template rendering and trajectory tokenization for the environmental GRPO trainer; a fatal row
    failure is recorded on the ``_batch_errors`` fence, never raised on one rank."""

    @cached_property
    def _rollout_env(self):
        """A throwaway instance of the run's environment, which the trainer's renders read the engine's
        context off (tool schema, carried reasoning)."""
        return build_environment(self._environment_spec, self._env_config_dict)

    def _context_limit(self) -> int:
        """The context window bounding every training row: the tokenizer's, where the training script pins the
        served window, else the model config's (:func:`get_model_context_window`, which raises rather than
        fall back to a size that would disable the overflow check)."""
        limit = self._tokenizer.model_max_length
        if is_bounded_length(limit) and limit < UNSET_MODEL_MAX_LENGTH:
            return int(limit)
        return get_model_context_window(self.model, self._tokenizer)

    def _warn_if_no_reasoning_captured(
        self, rollout_results: list[RolloutResult], *, price_on: bool, floor_on: bool
    ) -> None:
        """Warn once per run on a step whose assistant turns carry no reasoning while a knob reads it, the sign of
        a server without a reasoning parser; a step with no assistant turn is no evidence."""
        turns = [m for r in rollout_results if r.trajectory for m in r.trajectory.messages if m.role == "assistant"]
        if not turns or any(m.thinking for m in turns):
            return
        consumers = [
            name
            for name, on in (
                ("reasoning_price", price_on),
                ("reasoning_floor", floor_on),
                ("carry_reasoning", self._carry_reasoning),
            )
            if on
        ]
        warn_once(
            logger,
            self._warned_once,
            "no_reasoning_captured",
            "No assistant turn in this step carried reasoning, and the run reads it (%s): the rollout server is most "
            "likely running without a reasoning parser (or the model emits none). The reasoning floor then "
            "scores every episode as maximal under-use, the reasoning price charges nothing, and a carried "
            "thought is never sent. Serve with the family's reasoning parser, or turn off what reads it.",
            " and ".join(consumers),
        )

    @cached_property
    def _carry_reasoning(self) -> bool:
        """Whether the rollout sent prior-turn reasoning to the engine, so a context re-render includes it."""
        return self._rollout_env.carry_reasoning

    def _tokenize_step_rows(self, rollout_results: list[RolloutResult]) -> list[list[TurnRow]]:
        """Each rollout's training rows for this step, split per assistant turn under ``train_on_sampled_tokens``,
        and the step's ``sampling/rows_over_cap_frac`` and :data:`UNTRAINABLE_TURNS_ROWLESS_FRAC_KEY`."""
        self._rows_over_cap = 0
        per_trajectory = [
            self._tokenize_trajectory_turns(result)
            if self.async_config.train_on_sampled_tokens
            else single_trajectory_row(self._tokenize_trajectory(result))
            for result in rollout_results
        ]
        untrainable_turns = sum(
            m.role == "assistant" and m.untrainable
            for r in rollout_results
            if r.trajectory
            for m in r.trajectory.messages
        )
        tagged_rows = sum(row.negative_only for rows in per_trajectory for row in rows)
        self._world_metrics.fraction(
            UNTRAINABLE_TURNS_ROWLESS_FRAC_KEY, untrainable_turns - tagged_rows, untrainable_turns
        )
        if self.async_config.max_train_row_tokens is not None:
            trainable_rows = sum(bool(row.completion_mask.any()) for rows in per_trajectory for row in rows)
            self._world_metrics.fraction(
                "sampling/rows_over_cap_frac", self._rows_over_cap, self._rows_over_cap + trainable_rows
            )
        return per_trajectory

    def _tokenize_trajectory_turns(self, result: RolloutResult) -> list[TurnRow]:
        """One :class:`TurnRow` per assistant turn over the engine's prompt and sampled ids, falling back to the
        re-tokenized row when sampled ids are missing and to one masked row when no turn yields a row. Per turn,
        not concatenated: a template may drop prior-turn CoT, so a spliced later turn would be scored under a
        context it never saw.
        """
        traj = result.trajectory
        if not traj or not any(m.role == "assistant" for m in traj.messages):
            return single_trajectory_row(self._tokenize_trajectory(result))
        messages = traj.messages
        # One always-trained turn without captured ids falls the whole trajectory back rather than silently
        # dropping it (an empty capture is a zero-token turn, not a missing one).
        if any(m.token_ids is None for m in messages if m.role == "assistant" and not m.untrainable):
            remedy = (
                "Run the vLLM server with --return-tokens-as-token-ids (docker-compose.vllm.yml passes it)."
                if self._rollout_backend == VLLM_BACKEND
                else "SGLang needs no server flag for this — the ids are requested per call, so a "
                "rollout that returned none usually means the engine errored or was killed mid-turn."
            )
            warn_once(
                logger,
                self._warned_once,
                SAMPLED_IDS_MISSING_WARNING,
                "train_on_sampled_tokens is on but a rollout returned no sampled token ids — falling back to "
                "re-tokenization for that trajectory. %s",
                remedy,
            )
            return single_trajectory_row(self._tokenize_trajectory(result))

        template_kwargs = self._template_kwargs(traj)
        rows: list[TurnRow] = []
        for idx, m in enumerate(messages):
            # A zero-token turn has nothing to train: its render would weight scaffolding the engine never emitted.
            if m.role != "assistant" or not m.token_ids:
                continue
            # Engine prompt ids first: a client re-render drifts on effort steering, tool schemas and channels.
            if m.prompt_token_ids:
                prompt_ids = list(m.prompt_token_ids)
            else:
                # A template that rejects this prefix fails on one rank only: the episode is dropped, never raised.
                try:
                    prompt_ids = self._render_messages_to_ids(
                        engine_view(messages[:idx], self._carry_reasoning), True, template_kwargs
                    )
                except Exception as e:  # never a per-rank raise; the episode is dropped, the run goes on
                    if m.untrainable:  # its row only ever adds a penalty; losing it costs no other turn
                        continue
                    self._invalidate_untrainable_episode(
                        result,
                        f"per-turn re-render of the prompt prefix failed ({type(e).__name__}: {e}); the engine "
                        f"returned no prompt_token_ids for this turn and the template rejects this prefix",
                    )
                    return single_trajectory_row(self._masked_trajectory_tensors())
            # Over the row cap this turn leaves the batch; the episode's other turns stay.
            if self._admit_row(
                "Per-turn training row",
                len(prompt_ids),
                "sampled completion",
                len(m.token_ids),
                "a template re-render drift",
            ):
                rows.append(self._sampled_turn_row(m, prompt_ids))
        return rows or single_trajectory_row(self._masked_trajectory_tensors())

    def _sampled_turn_row(self, message, prompt_ids: list[int]) -> TurnRow:
        """An assistant turn's row over the engine's sampled ids: all of them trained, their sampling log-probs
        where the engine returned one per id, and the turn's engine routing under ``routing_replay='rollout'``
        (a malformed payload recorded rather than raised, so the raise stays rank-uniform)."""
        completion = list(message.token_ids)
        logprobs = message.token_logprobs
        turn_routing = None
        if self._rollout_routing_replay and message.routing_mask:
            try:
                turn_routing = (
                    self._routing_injector.decode_engine_mask(message.routing_mask),
                    message.routing_prompt_tokens,
                )
            except ValueError as e:
                self._batch_errors.record(f"routing_replay='rollout': malformed routed_experts payload: {e}")
        return TurnRow(
            prompt_ids=torch.tensor(prompt_ids, dtype=torch.long),
            completion_ids=torch.tensor(completion, dtype=torch.long),
            completion_mask=torch.ones(len(completion), dtype=torch.long),
            sampling_logps=(
                torch.tensor(logprobs, dtype=torch.float32) if logprobs and len(logprobs) == len(completion) else None
            ),
            turn_routing=turn_routing,
            negative_only=message.untrainable,
        )

    def _tokenize_trajectory(self, result: RolloutResult) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(prompt, completion, loss mask)`` of one render of the whole trajectory, its assistant spans located
        inside it (:func:`locate_assistant_spans`). The mask is 0 on env-injected tokens and on untrainable
        turns, whose re-render is not what the engine emitted (a cut turn comes back closed by template tokens).
        """
        messages = result.trajectory.messages if result.trajectory else []
        first_assistant_idx = next((i for i, m in enumerate(messages) if m.role == "assistant"), None)
        if first_assistant_idx is None:
            return self._masked_trajectory_tensors()
        template_kwargs = self._template_kwargs(result.trajectory)

        def _render(msgs: Sequence, add_generation_prompt: bool, include_thinking: bool) -> list[int]:
            return self._render_messages_to_ids(msgs, add_generation_prompt, template_kwargs, include_thinking)

        try:
            spans = locate_assistant_spans(_render, messages, first_assistant_idx)
            # One span per assistant message, in message order (:meth:`_SpanLocator.resolve`).
            turn_spans = list(zip((m for m in messages if m.role == "assistant"), spans.turn_spans, strict=True))
        except (TemplateSpanError, ValueError) as e:
            # Never raised per rank (a raise ahead of the batch collectives would break their order),
            # and not fatal either: independently rendered per-turn prefixes would splice in tokens the
            # serving template never emits, so the episode is dropped rather than trained on a guessed
            # render.
            self._invalidate_untrainable_episode(
                result,
                f"cannot locate the trained turn spans of {self._tokenizer.name_or_path} inside its own "
                f"chat-template render ({e}); serve with --return-tokens-as-token-ids and "
                f"train_on_sampled_tokens: true to train the engine's sampled ids directly",
            )
            return self._masked_trajectory_tensors()

        prompt_token_ids = spans.full_ids[: spans.prompt_len]
        completion_ids = spans.full_ids[spans.prompt_len :]
        completion_mask = [0] * len(completion_ids)
        for turn, (start, end) in turn_spans:
            if not turn.untrainable:
                completion_mask[start - spans.prompt_len : end - spans.prompt_len] = [1] * (end - start)
        admitted = self._admit_row(
            "Trajectory", len(prompt_token_ids), "completion", len(completion_ids), "the prompt length"
        )
        # Nothing trainable left (an empty completion, or every assistant turn excluded): the one-token masked
        # row keeps the rank-uniform row count without padding the round's batch to this trajectory's width.
        if not admitted or not any(completion_mask):
            return self._masked_trajectory_tensors()
        return (
            torch.tensor(prompt_token_ids, dtype=torch.long),
            torch.tensor(completion_ids, dtype=torch.long),
            torch.tensor(completion_mask, dtype=torch.long),
        )

    def _template_kwargs(self, trajectory) -> dict:
        """The chat-template kwargs ``trajectory``'s requests carried (:func:`rollout_template_kwargs`)."""
        return rollout_template_kwargs(
            self.async_config.rollout_chat_template_kwargs,
            trajectory.reasoning_effort,
            trajectory.reasoning_budget,
            trajectory.reasoning_budget_soft,
        )

    def _render_messages_to_ids(
        self,
        msgs: Sequence,
        add_generation_prompt: bool,
        template_kwargs: dict,
        include_thinking: bool = True,
    ) -> list[int]:
        """``msgs`` rendered through the serving chat template and tokenized; ``include_thinking=False`` emits no
        reasoning, the probe for a template that hides prior-turn CoT."""
        text = self.processing_class.apply_chat_template(
            [m.to_dict(include_thinking=include_thinking) for m in msgs],
            tools=self._tools_schema,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **template_kwargs,
        )
        # _tokenizer, not processing_class: a VLM processor's __call__ takes images positionally.
        return self._tokenizer(text, add_special_tokens=False)["input_ids"]

    @cached_property
    def _tools_schema(self) -> list[dict] | None:
        """The OpenAI tool schema the rollout passes the engine, rendered into the trainer's prompts too."""
        return self._rollout_env.get_tools_schema()

    def _admit_row(self, row: str, prompt_len: int, completion_label: str, completion_len: int, suspect: str) -> bool:
        """Whether a training row fits the row cap (``max_train_row_tokens``), counting one it leaves out. A row
        past the context window is first recorded on the fence, never truncated: the overflow is a config error."""
        limit = self._context_limit()
        row_len = prompt_len + completion_len
        if row_len > limit:
            self._batch_errors.record(
                f"{row} of {prompt_len + completion_len} tokens (prompt {prompt_len} + {completion_label} "
                f"{completion_len}) exceeds the model context window {limit}. Rollouts are context-bounded by the "
                f"rollout engine, so this points to a mismatch between the served context length (vLLM "
                f"max_model_len / SGLang context-length), the trainer context, or {suspect} — trajectories are "
                f"trained in full, never truncated."
            )
        cap = self.async_config.max_train_row_tokens
        if cap is not None and row_len > cap:
            self._rows_over_cap += 1
            return False
        return True

    def _invalidate_untrainable_episode(self, result: RolloutResult, reason: str) -> None:
        """Mark an episode the chat template cannot re-render invalid, so it leaves the group baseline instead of
        failing the run; a systematic cause still surfaces through the all-invalid step halt."""
        logger.warning("Dropping an episode the chat template cannot re-render for training: %s", reason)
        if result.trajectory is not None:
            result.trajectory.info[EPISODE_INVALID_KEY] = True
            result.trajectory.info[EPISODE_INVALID_REASON_KEY] = reason

    def _masked_trajectory_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One fully masked row (one attended prompt token, zero loss weight) for a trajectory with nothing to
        train, which a weighted row would reinforce at its advantage."""
        completion_token = self.eos_token_id if self.eos_token_id is not None else self.pad_token_id
        return (
            torch.tensor([self.pad_token_id], dtype=torch.long),
            torch.tensor([completion_token], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
        )
