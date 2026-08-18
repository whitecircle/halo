# SPDX-License-Identifier: Apache-2.0
"""Text-based tool-call parser for gpt-oss served with harmony DISABLED.

The toolkit serves gpt-oss with vLLM's harmony pipeline disabled
(``patch_vllm_disable_gptoss.sh`` sets ``use_harmony = False``) because the
native harmony serving path produces garbage / parser errors on this stack.
With harmony off, the model still follows its harmony *chat template*, so it
emits the channel structure — but vLLM detokenizes the output with the harmony
control tokens (``<|channel|>`` / ``<|message|>`` / ``<|constrain|>`` /
``<|call|>``) **stripped**, leaving only the bare channel/role words and the
payload as **plain text**. A reasoning answer therefore arrives as::

    analysis<reasoning>assistantfinal<answer>

and a tool call as::

    analysis<reasoning>assistantcommentary to=functions.calculator json{"expression":"17*23"}

(the role word ``assistant`` prefixes every channel after the first, and
``<|constrain|>json`` collapses to a bare `` json`` before the argument object).

The stock ``openai`` parser needs the harmony *token IDs* (stripped here) and the
``seed_oss`` parser expects ``<seed:tool_call>`` tags, so neither works. This
parser extracts the calls straight from the text: every

    to=functions.<NAME> ... json<JSON-OBJECT>

becomes a tool call, with ``<JSON-OBJECT>`` recovered by brace balancing so it is
robust to the stripped harmony delimiters. The ``final`` channel text (everything
after the ``assistant``-prefixed ``final`` marker) is returned as the assistant
``content``, which drops the leading ``analysis`` chain-of-thought.

Register at runtime with ``--tool-parser-plugin`` (no image rebuild) and select
with ``--tool-call-parser gpt_oss_text``.
"""

import ast
import json
import uuid
from collections.abc import Sequence
from typing import Any

import regex as re
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tool_parsers.abstract_tool_parser import ToolParser, ToolParserManager

logger = init_logger(__name__)

# `to=functions.<name>` optionally followed by a constrain hint, then `json`, then
# the JSON argument object (located by name; the object itself is brace-balanced).
_CALL_RE = re.compile(r"to=functions\.(?P<name>[A-Za-z0-9_\-]+)")
# ``final`` channel marker → everything after it is the answer. With harmony control tokens
# stripped it decodes as ``assistantfinal`` (or a start-anchored ``final``). Anchor to
# start-or-``assistant`` only, so ``final`` inside words like "finally" in the CoT can't split content.
_FINAL_RE = re.compile(r"(?:^|assistant)final(?P<text>.*)$", re.DOTALL)


def _balanced_json(text: str, start: int) -> tuple[str | None, int]:
    """Return the brace-balanced JSON object starting at/after ``start``.

    Scans for the first ``{`` at or after ``start`` and matches braces while
    respecting string literals and escapes. Returns (json_str, end_index) or
    (None, start) if no balanced object is found.
    """
    i = text.find("{", start)
    if i == -1:
        return None, start
    depth = 0
    in_str = False
    esc = False
    for j in range(i, len(text)):
        ch = text[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[i : j + 1], j + 1
    return None, start


def _json_canonical_keys(obj: Any) -> Any:
    """Recursively stringify dict keys JSON cannot represent.

    ``json.dumps`` already coerces int/float/bool/None keys to strings but raises ``TypeError`` on any
    other key type, so a Python-literal dict keyed by a tuple (``{(1, 2): 3}``, common in generated
    memo/test tables) would kill the call. Extend the same coercion to those keys.
    """
    if isinstance(obj, dict):
        return {
            (k if isinstance(k, (str, int, float, bool)) or k is None else str(k)): _json_canonical_keys(v)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [_json_canonical_keys(v) for v in obj]
    return obj


def _coerce_json_args(args_str: str, name: str) -> str:
    """Return a guaranteed-valid JSON object string for a tool call's arguments.

    The model emits its arguments under the harmony ``<|constrain|>json`` hint, but a
    fraction of generations slip into Python-dict syntax (single quotes, ``True``/
    ``None``) or drop key quoting entirely. Returning such a string verbatim is fatal:
    the NEXT turn replays the assistant ``tool_calls`` to the server, and vLLM's
    ``_postprocess_messages`` does ``json.loads(arguments)`` on it — an invalid string
    raises ``400 Bad Request`` and kills the whole rollout. So a Python-literal slip is
    recovered through ``ast.literal_eval`` with non-JSON keys canonicalized to strings,
    and text that stays unparsable is wrapped under ``_raw``: the request stays valid and
    the tool layer surfaces a normal argument error the model can recover from on a later
    turn.

    This function must NEVER raise: any escape becomes a 400 that kills the rollout.
    """
    try:
        return json.dumps(json.loads(args_str))
    except json.JSONDecodeError:
        pass
    try:
        # literal_eval raises ValueError/SyntaxError/MemoryError/RecursionError on hostile input; dumps
        # still raises TypeError on values JSON cannot encode (sets, bytes, complex). Both → _raw.
        parsed = ast.literal_eval(args_str)
        if isinstance(parsed, dict):
            return json.dumps(_json_canonical_keys(parsed))
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        pass
    logger.warning("gpt_oss_text: unparsable tool args for %s, wrapping raw: %s", name, args_str[:120])
    return json.dumps({"_raw": args_str})


def _declared_tool_names(request: ChatCompletionRequest) -> list[str]:
    """Names of the tools the request declares (for name resolution / validation)."""
    names: list[str] = []
    for tool in getattr(request, "tools", None) or []:
        fn = getattr(tool, "function", None) or (tool.get("function") if isinstance(tool, dict) else None)
        fn_name = getattr(fn, "name", None) or (fn.get("name") if isinstance(fn, dict) else None) if fn else None
        if fn_name:
            names.append(fn_name)
    return names


def _resolve_tool_name(raw: str, request: ChatCompletionRequest) -> str:
    """Map the regex-captured function name back to a declared tool name.

    With harmony disabled the ``<|constrain|>`` / ``<|channel|>`` control tokens are stripped, so the
    recipient header decodes with the channel/constraint word glued onto the name —
    ``to=functions.submit_solution`` arrives as ``submit_solutionjson`` / ``submit_solutioncommentary``
    (and the greedy name regex swallows the suffix). Recover the real tool by taking the LONGEST
    declared tool name that the captured token starts with. This is strict: it only rewrites to a tool
    the request actually declares, so a genuinely hallucinated name is left untouched (and the env
    rejects it) rather than invented."""
    if not raw:
        return raw
    known = _declared_tool_names(request)
    if raw in known:
        return raw
    prefixes = [t for t in known if raw.startswith(t)]
    return max(prefixes, key=len) if prefixes else raw


class GptOssTextToolParser(ToolParser):
    """Extract gpt-oss tool calls from the plain-text harmony channel stream."""

    # The base defaults this True, which routes ``tool_choice="required"`` and named tool_choice down
    # vLLM's standard-JSON branch: it validates the harmony channel text as a JSON tool list, the
    # ValidationError is swallowed, and the caller gets an empty message with no tool call and no
    # error. False keeps both forms coming through extract_tool_calls, the only parser here that can
    # read the plain-text format.
    supports_required_and_named = False

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
        **kwargs: Any,
    ) -> ExtractedToolCallInformation:
        if "to=functions." not in model_output:
            # No tool call — surface any final-channel answer as content.
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=self._final_content(model_output) or model_output,
            )

        # With harmony disabled the model omits the `<|call|>` stop token, so it often keeps generating
        # after the first genuine call (hallucinating the tool result as more `to=functions.*` text). For
        # env-GRPO (one call per turn, executed by the environment) only the FIRST call is meaningful, so
        # we stop after it. Parallel-call models would need the harmony stop token; not relevant here.
        tool_calls: list[ToolCall] = []
        m = _CALL_RE.search(model_output)
        if m is not None:
            name = _resolve_tool_name(m.group("name"), request)
            args_str, _ = _balanced_json(model_output, m.end())
            if args_str is not None:
                # Always normalize to valid JSON — a raw invalid string would 400 the
                # next turn's replay and kill the rollout (see _coerce_json_args).
                arguments = _coerce_json_args(args_str, name)
                tool_calls.append(
                    ToolCall(
                        id=f"call_{uuid.uuid4().hex[:24]}",
                        function=FunctionCall(name=name, arguments=arguments),
                    )
                )
            # No balanced JSON after the header → not a well-formed call, and we deliberately do NOT
            # rescue it: executing malformed calls would train the policy off the ``json{…}<|call|>``
            # format the template asks for. A dropped call becomes a no-tool turn — the right RL signal
            # to emit valid JSON next time.

        if not tool_calls:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=tool_calls,
            content=self._final_content(model_output),
        )

    @staticmethod
    def _final_content(model_output: str) -> str | None:
        m = _FINAL_RE.search(model_output)
        if not m:
            return None
        text = m.group("text").strip()
        return text or None

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        # Environmental GRPO rollouts use non-streaming completions, so streaming
        # tool extraction is not required here. Stream deltas through as content
        # until a tool call is detected, then emit nothing further (the full call
        # is recovered by extract_tool_calls at the end).
        if "to=functions." in current_text:
            return None
        return DeltaMessage(content=delta_text)


ToolParserManager.register_module(["gpt_oss_text"])(GptOssTextToolParser)
