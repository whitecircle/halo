"""Every hub dataset the env-GRPO benchmark suite loads passes ``HALO_TEST_BENCHMARK_DATASETS_CACHE`` as
``cache_dir``. Each run's ``HF_DATASETS_CACHE`` starts empty, so a load without it fails offline."""

import ast

import pytest

from tests.common.repo import REPO_ROOT

SCRIPT = REPO_ROOT / "tests/gpu/trainers/grpo/test_environmental_grpo_benchmarks.py"
KNOB = "HALO_TEST_BENCHMARK_DATASETS_CACHE"


def _cache_name(tree: ast.Module) -> str | None:
    """The module-level name the knob is read into, or None."""
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "env_str"
            and node.value.args
            and isinstance(node.value.args[0], ast.Constant)
            and node.value.args[0].value == KNOB
        ):
            return node.targets[0].id
    return None


def _uncached_loads(source: str) -> tuple[int, list[str]]:
    """``(load_dataset calls, those not passing the knob's name as cache_dir)``."""
    tree = ast.parse(source)
    name = _cache_name(tree)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "load_dataset"
    ]
    missing = [
        f"line {call.lineno}: {ast.unparse(call)}"
        for call in calls
        if not any(
            kw.arg == "cache_dir" and isinstance(kw.value, ast.Name) and name is not None and kw.value.id == name
            for kw in call.keywords
        )
    ]
    return len(calls), missing


def test_every_benchmark_load_reads_the_prepared_cache():
    count, missing = _uncached_loads(SCRIPT.read_text())
    assert count >= 4, f"expected the four benchmark loads (gsm8k, SimpleQA, MMLU-Pro, code_contests), found {count}"
    assert not missing, f"pass cache_dir=<{KNOB}> to:\n" + "\n".join(missing)


def test_a_load_without_the_knob_is_caught():
    source = (
        f'DATASETS_CACHE = env_str("{KNOB}")\n'
        'a = load_dataset("openai/gsm8k", "main", split="train[:8]", cache_dir=DATASETS_CACHE)\n'
        'b = load_dataset("deepmind/code_contests", split="test")\n'
    )
    count, missing = _uncached_loads(source)
    assert count == 2 and len(missing) == 1 and "code_contests" in missing[0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
