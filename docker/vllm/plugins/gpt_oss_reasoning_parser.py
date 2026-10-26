# SPDX-License-Identifier: Apache-2.0
"""Reasoning-parser plugin enabling ``thinking_token_budget`` for harmony-disabled gpt-oss.

The toolkit serves gpt-oss with vLLM's harmony pipeline disabled, which drops gpt-oss from the
reasoning-parser registry (and so from ``thinking_token_budget``) and makes the base non-streaming
``extract_reasoning`` raise. This parser restores both: its markers bound the budget to the analysis
channel, and it splits reasoning from answer on the plain-text channel markers the model still emits
(``…analysis…assistantfinal…``).

Serve alongside the ``gpt_oss_text`` tool parser:
  --reasoning-parser-plugin /opt/gpt_oss_reasoning_parser.py --reasoning-parser openai_gptoss
The forced cut needs the image's ``vllm_reasoning_forced_end_patch`` and
``vllm_thinking_budget_completion_patch``.
"""

import threading

import regex as re
from vllm.entrypoints.generate.base.protocol import DeltaMessage
from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.gptoss_reasoning_parser import GptOssReasoningParser

# vLLM builds a parser per request, and the HF fast tokenizer's ``encode`` mutates shared Rust
# truncation state; under concurrent requests that races the per-token ``logprobs`` detokenization and
# raises ``RuntimeError: Already borrowed`` (500). Markers are constant, so the constructed state is
# snapshotted once per tokenizer and replayed without touching the tokenizer.
_STATE_LOCK = threading.Lock()
_STATE_CACHE: dict[int, dict] = {}

# ``final`` channel marker in harmony-disabled text: channels after the first are prefixed by the
# role word, so it decodes as ``assistantfinal`` (or a start-anchored ``final``). Anchoring to
# start-or-``assistant`` avoids matching ``final`` inside words such as "finally" in the CoT.
_FINAL_RE = re.compile(r"(?:^|assistant)final", re.DOTALL)

# Token-level end of reasoning: the ``final`` channel opening, allowing a constrain hint of up to this
# many tokens before ``<|message|>``, within the current message (the scan stops at the previous
# message's ``<|end|>``).
_MAX_TOKENS_BETWEEN = 20


@ReasoningParserManager.register_module(["openai_gptoss"], force=True)
class GptOssBudgetReasoningParser(GptOssReasoningParser):
    """GptOss reasoning parser that works under harmony-disabled serving and concurrent load."""

    # What the budget forces once an analysis channel overruns: the model's own move into the final
    # channel, ids included, so the cut decodes to the natural ``assistantfinal`` and the split below
    # finds the answer behind it. vLLM stops forcing at the first natural end marker, which this
    # sequence reaches at its ``<|channel|>``; the completion patch forces the rest.
    reasoning_forced_end_str = "<|end|><|start|>assistant<|channel|>final<|message|>"

    def __init__(self, tokenizer, *args, **kwargs):
        key = id(tokenizer)
        state = _STATE_CACHE.get(key)
        if state is None:
            with _STATE_LOCK:
                if key not in _STATE_CACHE:
                    super().__init__(tokenizer, *args, **kwargs)
                    self._final_prefix_ids = tokenizer.encode("<|channel|>final", add_special_tokens=False)
                    self._message_ids = tokenizer.encode("<|message|>", add_special_tokens=False)
                    (self._end_of_message_id,) = tokenizer.encode("<|end|>", add_special_tokens=False)
                    _STATE_CACHE[key] = self.__dict__.copy()
                    return
                state = _STATE_CACHE[key]
        self.__dict__.update(state)

    # vLLM counts while the last start marker sits after the last end marker, from the tokens after
    # the start marker. These override read-only base properties, so they are properties.
    @property
    def reasoning_start_str(self) -> str:
        """Where the count starts: an analysis channel's opening, without its ``<|channel|>``.

        Every channel header carries ``<|channel|>``, the end marker, so the start marker has to begin
        after it for an analysis opening to land on the counting side.
        """
        return "analysis<|message|>"

    @property
    def reasoning_end_str(self) -> str:
        """Where the count stops: any channel header.

        A tool call, a commentary preamble and the final answer each open with ``<|channel|>``, so
        none of them is counted or cut. The first header of a turn follows the prompt's own
        ``<|start|>assistant``, and Model Runner V2 matches markers across that boundary.
        """
        return "<|channel|>"

    def is_reasoning_end(self, input_ids) -> bool:
        """Whether the current message has opened its ``final`` channel.

        The base answers True unconditionally, since harmony parses the output there. Under
        harmony-disabled serving that ends reasoning at the prompt: streaming routes every delta to
        content and structured outputs constrain the CoT.
        """
        prefix, suffix = self._final_prefix_ids, self._message_ids
        ids = list(input_ids)
        for i in range(len(ids) - len(prefix), -1, -1):
            if ids[i] == self._end_of_message_id:
                return False
            if ids[i : i + len(prefix)] == prefix:
                start = i + len(prefix)
                for j in range(start, min(start + _MAX_TOKENS_BETWEEN, len(ids) - len(suffix) + 1)):
                    if ids[j : j + len(suffix)] == suffix:
                        return True
        return False

    def is_reasoning_end_streaming(self, input_ids, delta_ids) -> bool:
        # The window spans the marker plus the whole delta, which speculative decoding can make long.
        window = len(self._final_prefix_ids) + _MAX_TOKENS_BETWEEN + len(self._message_ids) + len(tuple(delta_ids))
        return self.is_reasoning_end(list(input_ids)[-window:])

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

        The base implementation raises, assuming harmony does the parsing, so without this override
        every ``stream=true`` request 501s. Splitting both the previous and the current text and
        emitting the difference keeps streaming consistent with the non-streaming result.
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
        """Token ids after the last ``final`` channel opening (the base raises instead of answering)."""
        prefix = self._final_prefix_ids
        suffix = self._message_ids
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
        # are both content, since vLLM extracts tool calls from content only, so the split is at the
        # earliest channel boundary; otherwise a call left in reasoning is dropped and mis-scored.
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
