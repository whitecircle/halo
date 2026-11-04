"""Source edits to the installed vLLM, applied at image build through ``engine_source_patch``.

python3 patch_vllm_source.py            # patch the installed vllm in place
python3 patch_vllm_source.py --verify   # assert the post-images only
"""

import pathlib
import re

from engine_source_patch import Edit, main

_GPT_OSS_GATE = 'model_config.hf_config.model_type == "gpt_oss"'
# Any harmony gate keyed on gpt_oss, in whatever comparison form, outside a comment.
_HARMONY_GATE = re.compile(rb"(use|is)_harmony[ \t]*=[^#\n]*gpt_oss")
_MOE_BACKENDS = """    elif current_platform.is_cuda():
        _AVAILABLE_BACKENDS = [
            UnquantizedMoeBackend.FLASHINFER_TRTLLM,
            UnquantizedMoeBackend.FLASHINFER_CUTLASS,
            UnquantizedMoeBackend.TRITON,
            UnquantizedMoeBackend.BATCHED_TRITON,
        ]"""
_CUTLASS_DEMOTED = """

        # WORKAROUND (halo): FlashInfer CUTLASS BF16 has
        # correctness issues for unquantized MoE (vLLM authors already
        # demote it for Qwen3.5 + DP>1 below — same kernel produces pure
        # garbage for gpt-oss-20b bf16 unquantized MoE at any DP/TP).
        # Demote *only* CUTLASS: TRTLLM stays in priority slot 1 (correct
        # for Qwen3.5/3.6, faster than TRITON), and models whose layout
        # TRTLLM rejects (e.g. gpt-oss-20b) fall through to TRITON instead
        # of the broken CUTLASS kernel.
        _move_to_back(_AVAILABLE_BACKENDS, UnquantizedMoeBackend.FLASHINFER_CUTLASS)"""

EDITS = (
    # Harmony on, gpt-oss renders through openai_harmony and ignores the jinja chat template the
    # trainer rendered with; the toolkit's parser plugins serve it harmony-off.
    *(
        Edit(
            path,
            f"is_harmony=self.{_GPT_OSS_GATE},",
            'is_harmony=False,  # patched: was model_type == "gpt_oss"',
            f"{path} is_harmony",
        )
        for path in ("entrypoints/openai/chat_completion/serving.py", "entrypoints/openai/responses/serving.py")
    ),
    Edit(
        "entrypoints/openai/responses/serving.py",
        f"self.use_harmony = self.{_GPT_OSS_GATE}",
        f"self.use_harmony = False  # self.{_GPT_OSS_GATE}",
        "responses use_harmony",
    ),
    *(
        Edit(
            path,
            f"self.use_harmony = {_GPT_OSS_GATE}",
            f"self.use_harmony = False  # {_GPT_OSS_GATE}",
            f"{path} use_harmony",
        )
        for path in ("renderers/online_renderer.py", "renderers/online_derenderer.py")
    ),
    # No reasoning parser is forced on gpt-oss and the harmony-only stock one is unregistered: the
    # server names the toolkit's (--reasoning-parser-plugin).
    Edit(
        "reasoning/__init__.py",
        '    "openai_gptoss": (\n        "gptoss_reasoning_parser",\n        "GptOssReasoningParser",\n    ),\n',
        '    # "openai_gptoss": (\n    #     "gptoss_reasoning_parser",\n    #     "GptOssReasoningParser",\n    # ),\n',
        "openai_gptoss reasoning parser registration",
    ),
    Edit(
        "model_executor/models/config.py",
        '        if structured_outputs_config.reasoning_parser == "":\n'
        '            structured_outputs_config.reasoning_parser = "openai_gptoss"\n',
        "        # Disabled: do not force openai_gptoss reasoning parser\n"
        '        # if structured_outputs_config.reasoning_parser == "":\n'
        '        #     structured_outputs_config.reasoning_parser = "openai_gptoss"\n',
        "openai_gptoss reasoning parser auto-set",
    ),
    # stderr, where uvicorn logs, so --enable-log-requests / --enable-log-outputs are visible.
    Edit(
        "envs.py",
        'VLLM_LOGGING_STREAM: str = "ext://sys.stdout"',
        'VLLM_LOGGING_STREAM: str = "ext://sys.stderr"  # patched: was stdout',
        "logging stream annotation",
    ),
    Edit(
        "envs.py",
        '"VLLM_LOGGING_STREAM": lambda: os.getenv("VLLM_LOGGING_STREAM", "ext://sys.stdout")',
        '"VLLM_LOGGING_STREAM": lambda: os.getenv("VLLM_LOGGING_STREAM", "ext://sys.stderr")',
        "logging stream default",
    ),
    Edit(
        "model_executor/layers/fused_moe/oracle/unquantized.py",
        _MOE_BACKENDS,
        _MOE_BACKENDS + _CUTLASS_DEMOTED,
        "unquantized MoE FlashInfer CUTLASS demotion",
    ),
)


def reject_harmony_gates(root: pathlib.Path) -> None:
    """No gpt_oss harmony gate is left anywhere in vLLM, so a gate in a file the table misses fails."""
    gated = sorted(
        str(path.relative_to(root)) for path in root.rglob("*.py") if _HARMONY_GATE.search(path.read_bytes())
    )
    if gated:
        raise SystemExit(f"unpatched use_harmony/is_harmony gpt_oss gate in {gated} — upstream changed")


if __name__ == "__main__":
    main("vllm", EDITS, reject_harmony_gates)
