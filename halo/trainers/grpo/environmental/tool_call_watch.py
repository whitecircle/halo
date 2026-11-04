"""The tool-parser watch of environmental GRPO. A server whose parser cannot read this model's calls leaves them as text,
so each turn reads as a final answer and every episode ties with no error anywhere. A server reports no parser identity
to compare, so the watch reads the replies: the literal a rendered call opens with, in content, with no parsed call."""

import logging
import os.path

from halo.environments.episode import RolloutResult
from halo.runtime.ranks import is_global_main_process
from halo.trainers.world_metrics import world_sums

logger = logging.getLogger(__name__)

_PROBE_TOOL = "halo_probe_tool"
_PROBE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": _PROBE_TOOL,
            "description": "Probe.",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string", "description": "Probe."}},
                "required": ["value"],
            },
        },
    }
]


def rendered_tool_call_marker(tokenizer, template_kwargs: dict) -> str | None:
    """The text a call opens with, past the generation prompt: from the last added token before the tool's name, or all
    of it where it holds none (gpt-oss). ``None`` when the template cannot render a call or opens it with nothing."""
    messages = [{"role": "user", "content": "Probe."}]
    call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"type": "function", "id": "call_0", "function": {"name": _PROBE_TOOL, "arguments": {"value": "1"}}}
        ],
    }
    render = {"tools": _PROBE_TOOLS, "tokenize": False, **template_kwargs}
    try:
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, **render)
        full = tokenizer.apply_chat_template([*messages, call], add_generation_prompt=False, **render)
    except Exception as exc:  # a template's own raise: the watch stands down, as the rollouts decide
        logger.debug(f"Tool-call probe render failed: {exc}")
        return None
    shared, end = len(os.path.commonprefix([prompt, full])), full.rfind(_PROBE_TOOL)
    if end < shared:
        # The template drops the call, so what it writes past the prompt is only its turn's close.
        return None
    head = full[:end]
    position, token = max(((head.rfind(token), token) for token in tokenizer.get_added_vocab()), default=(-1, ""))
    # A token straddling the shared prefix is the opening's own (Gemma 4's prompt and call both start "<|"); one wholly
    # inside it is the prompt's.
    marker = head[position:] if position + len(token) > shared else head[shared:]
    return marker if marker.strip() else None


class ToolCallParserWatch:
    """Armed until the run sees its first parsed call, so a later malformed call never trips it. ``rounds_before_verdict``
    counts rounds holding an unparsed call: one may be one malformed call, which any parser leaves as text, and a wrong
    parser leaves every round that tries a call."""

    def __init__(self, marker: str, rounds_before_verdict: int = 2):
        self._marker = marker
        self._rounds_before_verdict = rounds_before_verdict
        self._rounds = 0
        self._rounds_unparsed = 0
        self._parsed = 0
        self._unparsed = 0
        self.armed = True

    def observe(self, results: list[RolloutResult]) -> None:
        """COLLECTIVE while armed, on every rank alike, since the verdict comes from world sums."""
        parsed = unparsed = 0
        for result in results:
            for message in result.trajectory.messages if result.trajectory else ():
                if message.role != "assistant":
                    continue
                if message.tool_calls or message.cut_tool_calls:
                    parsed += 1
                # A call the engine cut mid-way is text under every parser, the right one included.
                elif not message.truncated and self._marker in (message.content or ""):
                    unparsed += 1
        parsed, unparsed = world_sums([parsed, unparsed])
        self._rounds += 1
        self._rounds_unparsed += bool(unparsed)
        self._parsed += int(parsed)
        self._unparsed += int(unparsed)
        if self._parsed:
            self.armed = False
            return
        if self._rounds_unparsed >= self._rounds_before_verdict:
            raise RuntimeError(
                f"{self._unparsed} assistant turns over {self._rounds} rounds hold {self._marker!r}, how this model "
                f"writes a tool call, in their text, and the server parsed no call: its tool parser does not read "
                f"this model's format, so each turn reads as a final answer and every episode scores alike. Serve "
                f"with the family's parser (VLLM_TOOL_PARSER with --enable-auto-tool-choice, SGLANG_TOOL_PARSER; "
                f"agent-docs/infrastructure/rollout-servers.md#running-it). If the parser is right and these calls are "
                f"malformed or prose, raise tool_parser_watch_rounds, or set it to null."
            )


def build_tool_call_watch(
    environment, tokenizer, template_kwargs: dict, rounds_before_verdict: int | None = 2
) -> ToolCallParserWatch | None:
    """``None`` with the watch configured off, for an environment that sends no tools, and where the template renders no
    opening to look for."""
    if rounds_before_verdict is None:
        return None
    if not environment.get_tools_schema() and not environment.TOOLS_ARRIVE_ON_RESET:
        return None
    marker = rendered_tool_call_marker(tokenizer, template_kwargs)
    if marker is None:
        if is_global_main_process():
            logger.warning(
                "The chat template renders no tool-call opening to watch for, so a server whose parser cannot read "
                "this model's calls is only visible as a flat zero gradient."
            )
        return None
    return ToolCallParserWatch(marker, rounds_before_verdict)
