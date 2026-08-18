# SPDX-License-Identifier: Apache-2.0
"""Reasoning-parser plugin enabling ``thinking_token_budget`` for harmony-disabled gpt-oss.

The toolkit serves gpt-oss with vLLM's harmony pipeline disabled, which drops gpt-oss from
the reasoning-parser registry (and thus ``thinking_token_budget``) and makes the base
non-streaming ``extract_reasoning`` raise. This parser restores both: it arms and ends the budget on
markers the harmony-disabled render actually emits, and splits reasoning vs. answer on the plain-text
channel markers the model still emits (``…analysis…assistantfinal…``).

Serve alongside the ``gpt_oss_text`` tool parser:
  --reasoning-parser-plugin /opt/gpt_oss_reasoning_parser.py --reasoning-parser openai_gptoss
"""

import threading

import regex as re
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.gptoss_reasoning_parser import GptOssReasoningParser

# vLLM builds a parser per request; the base __init__ calls the shared HF fast tokenizer's ``encode``
# (mutates Rust truncation state), which under concurrent requests races the per-token ``logprobs``
# detokenization and raises ``RuntimeError: Already borrowed`` (500). Markers are constant, so snapshot
# the constructed state once per tokenizer and replay it with zero tokenizer access.
_STATE_LOCK = threading.Lock()
_STATE_CACHE: dict[int, dict] = {}

# ``final`` channel marker in harmony-disabled text: channels after the first are prefixed by the
# role word, so it decodes as ``assistantfinal`` (or a start-anchored ``final``). Anchoring to
# start-or-``assistant`` avoids matching ``final`` inside words like "finally" in the CoT.
_FINAL_RE = re.compile(r"(?:^|assistant)final", re.DOTALL)


@ReasoningParserManager.register_module(["openai_gptoss"], force=True)
class GptOssBudgetReasoningParser(GptOssReasoningParser):
    """GptOss reasoning parser that works under harmony-disabled serving and concurrent load."""

    def __init__(self, tokenizer, *args, **kwargs):
        key = id(tokenizer)
        state = _STATE_CACHE.get(key)
        if state is None:
            with _STATE_LOCK:
                if key not in _STATE_CACHE:
                    super().__init__(tokenizer, *args, **kwargs)
                    _STATE_CACHE[key] = self.__dict__.copy()
                    return
                state = _STATE_CACHE[key]
        self.__dict__.update(state)

    # Harmony markers vLLM tokenizes into the budget's arm/force ids. These override read-only base
    # properties, so they must be properties, not __init__ assignments.
    @property
    def reasoning_start_str(self) -> str:
        """Where the budget arms: the generation prompt's own trailing ``<|start|>assistant``.

        vLLM counts from the last occurrence of ONE fixed id sequence, so no channel opening serves.
        This parser calls every channel before ``final`` reasoning while the harmony-disabled render
        opens ``analysis`` or ``commentary``, and bare ``<|channel|>`` also matches the tool-result
        turn already in the prompt, burning the budget before the first sampled token. Matching the
        prompt tail arms vLLM's ``continue_thinking`` state at a zero prompt-side count, so the
        budget bounds exactly the tokens generated before the ``final`` channel.
        """
        return "<|start|>assistant"

    @property
    def reasoning_end_str(self) -> str:
        """What the budget forces — the model's OWN final-channel opening, ids included.

        The engine appends these ids verbatim, so the marker must decode to the same text the model
        writes when it ends reasoning on its own. Dropping ``<|start|>assistant`` (the channel marker
        alone) decodes to a bare ``final`` glued to the truncated CoT, which no anchored pattern can
        tell from prose ``final`` — the split then strands the whole answer in ``reasoning``. Carrying
        the role tokens makes the forced cut byte-identical to the natural ``assistantfinal``.
        """
        return "<|start|>assistant<|channel|>final<|message|>"

    def extract_reasoning(self, model_output: str, request) -> tuple[str | None, str | None]:
        return self._split(model_output)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
    ) -> "DeltaMessage | None":
        """Route each delta to the reasoning or content channel.

        The base implementation raises (it assumes harmony does the parsing), so without this
        override every ``stream=true`` request 501s. Splitting both the previous and the current text
        and emitting the difference keeps streaming consistent with the non-streaming result.
        """
        prev_reasoning, prev_content = self._split(previous_text)
        cur_reasoning, cur_content = self._split(current_text)

        def _delta(previous: str | None, current: str | None) -> str | None:
            previous, current = previous or "", current or ""
            if current == previous:
                return None
            return current[len(previous) :] if current.startswith(previous) else current

        reasoning_delta = _delta(prev_reasoning, cur_reasoning)
        content_delta = _delta(prev_content, cur_content)
        if reasoning_delta is None and content_delta is None:
            return None
        return DeltaMessage(reasoning=reasoning_delta, content=content_delta)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """Token ids after the reasoning-end marker (the base raises instead of answering)."""
        prefix = list(self.reasoning_end_token_ids_prefix)
        suffix = list(self.reasoning_end_token_ids_suffix)
        ids = list(input_ids)
        for i in range(len(ids) - len(prefix), -1, -1):
            if ids[i : i + len(prefix)] != prefix:
                continue
            for j in range(i + len(prefix), len(ids) - len(suffix) + 1):
                if ids[j : j + len(suffix)] == suffix:
                    return ids[j + len(suffix) :]
        return []

    def _split(self, model_output: str) -> tuple[str | None, str | None]:
        # Reasoning is only the ``analysis`` channel. A commentary tool call and the ``final`` answer
        # are both content — vLLM extracts tool calls exclusively from content — so split at the
        # earliest channel boundary or the call left in reasoning is dropped and mis-scored.
        call_idx = model_output.find("to=functions.")
        call_split = None
        if call_idx != -1:
            commentaries = list(re.finditer(r"commentary", model_output[:call_idx]))
            call_split = commentaries[-1].start() if commentaries else call_idx
        final_m = _FINAL_RE.search(model_output)
        final_start = final_m.start() if final_m is not None else None

        if call_split is not None and (final_start is None or call_split <= final_start):
            # Commentary call comes first: content carries the call and any later ``final`` answer.
            reasoning = model_output[:call_split].strip() or None
            content = model_output[call_split:].strip() or None
            return (reasoning, content)
        if final_m is not None:
            # ``final`` channel is the boundary: content is the answer after the marker.
            reasoning = model_output[: final_m.start()].strip() or None
            content = model_output[final_m.end() :].strip() or None
            return (reasoning, content)
        return (model_output or None, None)
