#!/usr/bin/env python
"""environmental_grpo.py script contracts: honest prompt conditioning + context accounting.

- ``system_prompt`` must be REJECTED at startup: the environment builds the rollout conversation
  (its own system prompt + tool schema) from only the dataset's user prompt, so a YAML
  ``system_prompt`` is a silent no-op for conditioning (it only distorts the prompt-length filter).
- ``measure_env_prompt_overhead`` measures the environment's real preamble (system prompt + tool
  schema) so the startup context-window check validates the prompt the model actually sees, with a
  conservative fallback when the template cannot render it.
- The retired ``use_parallel_weight_sync`` knob is refused by the same unknown-key check.

    python tests/cpu/grpo/test_env_grpo_script_contracts.py
"""

import dataclasses
import re
import sys
import types
from pathlib import Path
from unittest import mock

import pytest
import yaml

from src.configs.async_training_config import AsyncTrainingConfig
from tests.common.utils import load_script_module

_REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def env_grpo_module():
    return load_script_module("scripts/training/environmental_grpo.py", "halo_test_environmental_grpo")


# system_prompt is rejected loudly (it cannot condition env rollouts)


def test_system_prompt_rejected_at_startup(env_grpo_module, tmp_path):
    """No env-GRPO config declares ``system_prompt`` — the environment owns the rollout's system
    turn — so it must reach the parser's strict unknown-key check rather than being absorbed and
    silently skewing only the prompt-length filter."""
    config = tmp_path / "config.yaml"
    config.write_text(
        f"model_name_or_path: dummy/model\noutput_dir: {tmp_path / 'out'}\n"
        "bf16: false\nuse_cpu: true\nenvironment_type: code_contests\nsystem_prompt: 'be helpful'\n"
    )
    with (
        mock.patch("src.training.parser.install_log_tee"),
        mock.patch.object(sys, "argv", ["prog", str(config)]),
        pytest.raises(ValueError, match="system_prompt"),
    ):
        env_grpo_module.main()


def test_retired_parallel_weight_sync_key_rejected_at_startup(env_grpo_module, tmp_path):
    """``use_parallel_weight_sync`` is retired — multi-server weight sync is always rolling. A YAML
    still carrying it must reach the parser's strict unknown-key check, not be absorbed while the
    run silently syncs in a shape the key no longer selects."""
    declared = {field.name for field in dataclasses.fields(AsyncTrainingConfig)}
    assert "use_parallel_weight_sync" not in declared, "the retired knob is still a declared field"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"model_name_or_path: dummy/model\noutput_dir: {tmp_path / 'out'}\n"
        "bf16: false\nuse_cpu: true\nenvironment_type: code_contests\nuse_parallel_weight_sync: true\n"
    )
    with (
        mock.patch("src.training.parser.install_log_tee"),
        mock.patch.object(sys, "argv", ["prog", str(config)]),
        pytest.raises(ValueError, match="use_parallel_weight_sync"),
    ):
        env_grpo_module.main()


# Environment prompt overhead measurement


class _Tokenizer:
    """Chat template whose render length is exactly the char count of what it was given."""

    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=False):
        return "".join(m["content"] for m in messages) + str(tools or "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text)))}


def _env(system_prompt="SYSTEM PROMPT", tools=None):
    return types.SimpleNamespace(system_prompt=system_prompt, get_tools_schema=lambda: tools)


def test_overhead_counts_system_prompt_and_tool_schema(env_grpo_module):
    tok = _Tokenizer()
    bare = env_grpo_module.measure_env_prompt_overhead(_env(), tok)
    assert bare >= len("SYSTEM PROMPT")
    with_tools = env_grpo_module.measure_env_prompt_overhead(_env(tools=[{"name": "python_repl"}]), tok)
    assert with_tools > bare  # the rendered tool schema is part of the real rollout prompt


def test_overhead_without_env_system_prompt_still_measures(env_grpo_module):
    overhead = env_grpo_module.measure_env_prompt_overhead(_env(system_prompt=None), _Tokenizer())
    assert overhead >= 0
    assert overhead != env_grpo_module.FALLBACK_ENV_PROMPT_OVERHEAD


def test_overhead_falls_back_conservatively_when_render_fails(env_grpo_module):
    broken = types.SimpleNamespace(
        apply_chat_template=mock.Mock(side_effect=TypeError("no tools support")),
    )
    overhead = env_grpo_module.measure_env_prompt_overhead(_env(), broken)
    assert overhead == env_grpo_module.FALLBACK_ENV_PROMPT_OVERHEAD


# the docstring's launch examples must be runnable


def _docstring_launches(docstring: str) -> list[tuple[int, str]]:
    """``(nproc_per_node, example config path)`` for every launch block in the module docstring."""
    blocks = re.findall(r"--nproc_per_node=(\d+)(.*?)(examples/\S+\.yaml)", docstring, flags=re.S)
    return [(int(nproc), path) for nproc, _, path in blocks]


def test_docstring_launch_examples_are_runnable(env_grpo_module):
    """A launch line whose world size cannot host the config's own ``expert_parallel_size`` is
    rejected by ``ParallelismConfig`` before the first step, so a docstring carrying one teaches a
    command that cannot run. Every cited config must also still exist."""
    launches = _docstring_launches(env_grpo_module.__doc__)
    assert launches, "premise: the docstring shows at least one launch command"
    for nproc, relative in launches:
        config = _REPO_ROOT / relative
        assert config.is_file(), f"the docstring cites {relative}, which does not exist"
        ep_size = int(yaml.safe_load(config.read_text()).get("expert_parallel_size", 1))
        assert ep_size <= nproc and nproc % ep_size == 0, (
            f"{relative} declares expert_parallel_size={ep_size}, which --nproc_per_node={nproc} cannot host"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
