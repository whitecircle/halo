"""Trajectory → training-token conversion for environmental GRPO.

A returned rollout becomes trainable rows here: one whole-trajectory render with the assistant spans
located inside it, or, under ``train_on_sampled_tokens``, one row per assistant turn carrying the
engine's own prompt/sampled ids, sampling log-probs and routing capture.
"""

import logging
from collections.abc import Sequence
from functools import cached_property
from typing import NamedTuple

import torch

from src.environments.base import Trajectory
from src.environments.episode import RolloutResult
from src.environments.registry import create_environment
from src.models.loading.tokenizer_setup import UNSET_MODEL_MAX_LENGTH, get_model_context_window, is_bounded_length
from src.trainers.grpo.rollout.routing_replay import decode_rollout_routing
from src.trainers.grpo.rollout.trajectory_spans import TemplateSpanError, locate_assistant_spans

logger = logging.getLogger(__name__)


# Engine MoE routing for one turn: decoded mask + the prompt-token count it is aligned on.
TurnRouting = tuple[torch.Tensor, int | None]


class TurnRow(NamedTuple):
    """One per-turn training row (tuple-compatible; the trajectory fallback builds it by splat)."""

    prompt_ids: torch.Tensor
    completion_ids: torch.Tensor
    completion_mask: torch.Tensor
    sampling_logps: torch.Tensor | None
    turn_routing: TurnRouting | None


def single_trajectory_row(tokenized: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> list[TurnRow]:
    """A whole re-tokenized trajectory as one training row: the shape used when the per-turn split is
    off, and the fallback whenever it cannot be taken (no sampled ids, no assistant turn)."""
    return [TurnRow(*tokenized, None, None)]


def _effort_template_kwargs(trajectory: Trajectory) -> dict[str, str]:
    """Chat-template kwargs pinning the ``reasoning_effort`` the rollout used, so the template's
    effort-dependent preamble matches. Empty when the trajectory carries no effort."""
    effort = trajectory.reasoning_effort
    return {"reasoning_effort": effort} if effort is not None else {}


class TrajectoryTokenizeMixin:
    """Chat-template rendering and trajectory tokenization for the environmental GRPO trainer.

    Reads the trainer's tokenizer/processor, environment spec, context window and routing-replay
    state. Fatal per-row failures are recorded in ``self._batch_build_error`` rather than raised, so
    the trainer's rank-uniform fence raises them together.
    """

    def _render_messages_to_ids(
        self,
        msgs: Sequence,
        add_generation_prompt: bool,
        template_kwargs: dict,
        include_thinking: bool = True,
    ) -> list[int]:
        """Render messages through the serving chat template and tokenize to ids.

        ``include_thinking`` emits assistant reasoning: ``True`` when the render holds the trained
        completion; ``False`` for a pure context/prompt render (the rollout sent no reasoning to the
        server, so rendering prior-turn CoT would fabricate context the model never conditioned on).
        """
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
        """OpenAI tool schema the rollout passes to vLLM (``tools=``), or ``None``. Rendered into the
        trainer's prompt so recompute conditions on vLLM's exact context; built from a throwaway env."""
        env = create_environment(self._environment_spec, self._env_config_dict)
        return env.get_tools_schema()

    def _context_limit(self) -> int:
        """Model context window bounding every training row.

        The tokenizer is consulted first: the training script pins the *served* window there, which is
        the limit vLLM enforces during rollout. An unset value (``None``, non-positive, or at/above
        :data:`UNSET_MODEL_MAX_LENGTH`, the threshold HF's oversized "no limit" sentinels sit above)
        falls through to :func:`get_model_context_window`, which reads the model config (composite/VLM
        safe) and raises when no window is derivable; a large fallback would instead disable the
        trajectory-overflow check. The tokenizer predicate is the shared resolver's own.
        """
        limit = self._tokenizer.model_max_length
        if is_bounded_length(limit) and limit < UNSET_MODEL_MAX_LENGTH:
            return int(limit)
        return get_model_context_window(self.model, self._tokenizer)

    def _masked_trajectory_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One fully masked row: trainable shape, zero loss weight.

        Every trajectory that yields nothing trainable comes through here. A weighted row would
        reinforce, at that trajectory's advantage, either P(EOS | non-completion) or — when every
        assistant turn was excluded as unusable — the fragment the exclusion suppresses.
        """
        completion_token = self.eos_token_id if self.eos_token_id is not None else self.pad_token_id
        return (
            torch.tensor([self.pad_token_id], dtype=torch.long),
            torch.tensor([completion_token], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
        )

    def _tokenize_trajectory(self, result: RolloutResult) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize a multi-turn trajectory into prompt, completion, and completion-mask tensors.

        Prompt plus completion is one render of the whole trajectory, so the trained sequence is what
        the serving template emits; per-turn spans are located inside it
        (:func:`locate_assistant_spans`) rather than accumulated from independently rendered prefixes,
        which no non-monotone template reproduces. The returned mask is the loss mask (1 on assistant
        spans, 0 on env-injected tokens), consumed as TRL's ``tool_mask``; ``_build_training_tensors``
        derives the attention-valid ``completion_mask`` (all real tokens) from it, so tool outputs stay
        visible to attention while contributing no loss.
        """
        fallback_completion_token = self.eos_token_id if self.eos_token_id is not None else self.pad_token_id

        if not result.trajectory or not result.trajectory.messages:
            return self._masked_trajectory_tensors()

        messages = result.trajectory.messages

        first_assistant_idx = None
        for i, m in enumerate(messages):
            if m.role == "assistant":
                first_assistant_idx = i
                break

        if first_assistant_idx is None:
            return self._masked_trajectory_tensors()

        template_kwargs = _effort_template_kwargs(result.trajectory)

        def _render(msgs: Sequence, add_generation_prompt: bool, include_thinking: bool) -> list[int]:
            return self._render_messages_to_ids(msgs, add_generation_prompt, template_kwargs, include_thinking)

        try:
            spans = locate_assistant_spans(_render, messages, first_assistant_idx)
        except TemplateSpanError as e:
            # Recorded rather than raised: a per-rank raise would break the batch collectives' order.
            if self._batch_build_error is None:
                self._batch_build_error = (
                    f"Cannot locate the trained turn spans of {self._tokenizer.name_or_path} inside its own "
                    f"chat-template render ({e}). Training on independently rendered per-turn prefixes would "
                    f"splice in tokens the serving template never emits (a mid-episode stop token, dropped "
                    f"context). Serve with --return-tokens-as-token-ids and set train_on_sampled_tokens: true "
                    f"to train the engine's sampled ids directly."
                )
            return self._masked_trajectory_tensors()

        prompt_token_ids = spans.full_ids[: spans.prompt_len]
        completion_ids = spans.full_ids[spans.prompt_len :]
        completion_mask = [0] * len(completion_ids)
        for start, end in spans.turn_spans:
            completion_mask[start - spans.prompt_len : end - spans.prompt_len] = [1] * (end - start)

        # No truncation (reward would decouple from trained tokens); recorded rather than raised.
        context_limit = self._context_limit()
        total_len = len(prompt_token_ids) + len(completion_ids)
        if total_len > context_limit and self._batch_build_error is None:
            self._batch_build_error = (
                f"Trajectory of {total_len} tokens (prompt {len(prompt_token_ids)} + completion "
                f"{len(completion_ids)}) exceeds the model context window {context_limit}. Rollouts are "
                f"context-bounded by vLLM, so this points to a mismatch between the served max_model_len, "
                f"the trainer context, or the prompt length — trajectories are trained in full, never truncated."
            )

        if len(completion_ids) == 0:
            # Masked for the same reason _masked_trajectory_tensors() masks: a trainable row here
            # would reinforce P(EOS|prompt) at this advantage on a completion the policy never emitted.
            completion_ids = [fallback_completion_token]
            completion_mask = [0]

        return (
            torch.tensor(prompt_token_ids, dtype=torch.long),
            torch.tensor(completion_ids, dtype=torch.long),
            torch.tensor(completion_mask, dtype=torch.long),
        )

    def _tokenize_trajectory_turns(self, result: RolloutResult) -> list[TurnRow]:
        """One :class:`TurnRow` per assistant turn, for ``train_on_sampled_tokens``.

        * ``prompt_ids`` — the engine's rendered prompt ids for this turn's request
          (``Message.prompt_token_ids``): byte-exact conditioning, no re-render. Falls back to a
          serving-template render of the history without CoT (what the rollout sent).
        * ``completion_ids`` — the turn's sampled ids, so the loss reinforces what was emitted.
        * ``completion_mask`` — all ones (the whole turn is model-generated).
        * ``sampling_logps`` — the engine's sampling log-probs, 1:1 with ``completion_ids``: the
          behavior-policy reference for the IS trust region. ``None`` when the turn has no aligned
          logprobs.
        * ``turn_routing`` — decoded engine routing plus its prompt-token count, else ``None``.

        Per turn, not concatenated: a template may drop prior-turn CoT, so splicing would score a later
        turn under a context it never saw. Rows share the trajectory's advantage; falls back to the single
        re-tokenized row when sampled ids are missing.
        """
        traj = result.trajectory
        if not traj or not traj.messages:
            return single_trajectory_row(self._tokenize_trajectory(result))

        messages = traj.messages
        assistant_turns = [m for m in messages if m.role == "assistant"]
        # All-or-nothing: if any turn lacks captured ids, fall back rather than drop turns.
        if not assistant_turns or any(not m.token_ids for m in assistant_turns):
            if assistant_turns and not self._warned_capture_missing:
                self._warned_capture_missing = True
                # The remedy is engine-specific; naming the wrong one points at a flag the configured
                # server does not accept.
                remedy = (
                    "Run the vLLM server with --return-tokens-as-token-ids (docker-compose.vllm.yml passes it)."
                    if self._rollout_backend == "vllm"
                    else "SGLang needs no server flag for this — the ids are requested per call, so a "
                    "rollout that returned none usually means the engine errored or was killed mid-turn."
                )
                logger.warning(
                    f"train_on_sampled_tokens is on but a rollout returned no sampled token ids — "
                    f"falling back to re-tokenization for that trajectory. {remedy}"
                )
            return single_trajectory_row(self._tokenize_trajectory(result))

        template_kwargs = _effort_template_kwargs(traj)

        context_limit = self._context_limit()
        rows: list[TurnRow] = []
        excluded_unusable = False
        for idx, m in enumerate(messages):
            if m.role != "assistant":
                continue
            if m.truncated or m.calls_rejected:
                # A turn that produced nothing usable: an engine-cut fragment, or one whose every tool
                # call named a nonexistent tool. It stays in the next turn's prompt (the model must
                # condition on what it emitted), but training on it would reinforce the runaway or the
                # invented call whenever the episode recovers and earns a positive advantage.
                excluded_unusable = True
                continue
            # Engine prompt ids take priority: a client re-render drifts on effort steering, tool
            # schemas and channel placement.
            if m.prompt_token_ids:
                prompt_ids = list(m.prompt_token_ids)
            else:
                # Recorded rather than raised, like every other render site: a template that rejects
                # this prefix (a turn ending on a `tool` message) fails on one rank only, and the
                # raise must stay rank-uniform.
                try:
                    prompt_ids = self._render_messages_to_ids(
                        messages[:idx], True, template_kwargs, include_thinking=False
                    )
                except Exception as e:  # any template failure must stay rank-uniform
                    if self._batch_build_error is None:
                        self._batch_build_error = (
                            f"Per-turn re-render of the prompt prefix failed ({type(e).__name__}: {e}). The "
                            f"engine returned no prompt_token_ids for this turn, so the chat template had to "
                            f"rebuild it — and it rejects this message prefix."
                        )
                    continue
            comp = list(m.token_ids)
            if len(prompt_ids) + len(comp) > context_limit and self._batch_build_error is None:
                # First error wins, like every sibling write: a later overflow would otherwise
                # replace the root cause.
                self._batch_build_error = (
                    f"Per-turn training row of {len(prompt_ids) + len(comp)} tokens (prompt "
                    f"{len(prompt_ids)} + sampled completion {len(comp)}) exceeds the model context "
                    f"window {context_limit}. Rollouts are context-bounded by vLLM, so this points to a "
                    f"mismatch between the served max_model_len, the trainer context, or a template "
                    f"re-render drift — trajectories are trained in full, never truncated."
                )
            lp = m.token_logprobs
            sampling_logps = torch.tensor(lp, dtype=torch.float32) if lp and len(lp) == len(comp) else None
            # Recorded rather than raised: the raise must stay rank-uniform.
            turn_routing = None
            if self._rollout_routing_replay and m.routing_mask:
                try:
                    turn_routing = (
                        decode_rollout_routing(
                            m.routing_mask, self._routing_injector.num_layers, self._routing_injector.top_k
                        ),
                        m.routing_prompt_tokens,
                    )
                except ValueError as e:
                    if self._batch_build_error is None:
                        self._batch_build_error = f"routing_replay='rollout': malformed routed_experts payload: {e}"
            rows.append(
                TurnRow(
                    prompt_ids=torch.tensor(prompt_ids, dtype=torch.long),
                    completion_ids=torch.tensor(comp, dtype=torch.long),
                    completion_mask=torch.ones(len(comp), dtype=torch.long),
                    sampling_logps=sampling_logps,
                    turn_routing=turn_routing,
                )
            )

        if not rows:
            if excluded_unusable:
                # Re-tokenizing would weight every assistant span, truncated ones included, inverting
                # the exclusion above into full-weight training on what it suppresses.
                return single_trajectory_row(self._masked_trajectory_tensors())
            return single_trajectory_row(self._tokenize_trajectory(result))
        return rows
