"""Build-time check that a tool-bearing request at tool_choice auto samples with no grammar.

A grammar masks the logits before the sampler, and the reported log-probs carry the mask while the trainer scores the
unmasked distribution, so RL would be off-policy with no error. This runs the installed parser over every name a family
resolves to, then checks that glm47, which needed the source edit, still reads a rendered call and leaves a malformed
one as text. It imports sglang, which only this image has, so the image build is its only gate.

    python3 verify_glm47_unconstrained.py
"""

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser

_TOOLS = [
    Tool(
        type="function",
        function={
            "name": "python_repl",
            "description": "run",
            "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
        },
    )
]
_PARSERS = ("glm47", "glm45", "poolside_v1", "qwen", "qwen25", "hermes", "qwen3_coder", "gemma4", "gpt-oss", "lfm2")
_GLM47_CALL = "<tool_call>python_repl<arg_key>code</arg_key><arg_value>print(1)</arg_value></tool_call>"


def main() -> None:
    for name in _PARSERS:
        for thinking in (True, False):
            constraint = FunctionCallParser(_TOOLS, name).get_structure_constraint("auto", thinking_mode=thinking)
            assert constraint is None, f"{name} constrains tool_choice auto (thinking={thinking}): {constraint[0]}"
    print(f"tool_choice auto is unconstrained on {len(_PARSERS)} parsers: OK")

    parser = FunctionCallParser(_TOOLS, "glm47")
    assert parser.has_tool_call(_GLM47_CALL), "the glm47 parser no longer detects a rendered GLM-4.7 call"
    _, calls = parser.parse_non_stream(_GLM47_CALL)
    assert [call.name for call in calls] == ["python_repl"], f"the glm47 parser read {calls}"
    _, malformed = FunctionCallParser(_TOOLS, "glm47").parse_non_stream(
        "<tool_call>python_repl<arg_key>code</arg_key>"
    )
    assert not malformed, f"the glm47 parser read a malformed call as {malformed}"
    print("glm47 reads a rendered call and leaves a malformed one as text: OK")


if __name__ == "__main__":
    main()
