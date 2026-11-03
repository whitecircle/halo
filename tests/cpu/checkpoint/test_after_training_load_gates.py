"""How after-training tools load the checkpoints they rewrite: through the coverage gate, with the remote-code shims
first. A raw ``from_pretrained`` random-initializes absent tensors with a log line nobody sees."""

from __future__ import annotations

import ast
import pathlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file
from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen3ForSequenceClassification

from halo.checkpoint.tools import adapter_merge, bf16_conversion
from halo.model_loading import model_preparation
from halo.model_loading.parallel.whole_model import resolve_trainer_model
from tests.common.models import TINY_QWEN3_CONFIG
from tests.common.repo import REPO_ROOT
from tests.common.source_sweep import imports_name

# Derived rather than listed, so a new tool is swept in.
SHARED_MODEL_LOADER = "auto_load_model"
REMOTE_CODE_SHIMS = "apply_remote_code_compat_shims"
WEIGHTS_FILE = "model.safetensors"


@pytest.fixture(scope="module")
def causal_lm_checkpoint(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("causal_base")
    Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG)).save_pretrained(path)
    return str(path)


@pytest.fixture(scope="module")
def reward_checkpoint(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("reward_base")
    Qwen3ForSequenceClassification(Qwen3Config(**TINY_QWEN3_CONFIG, num_labels=1)).save_pretrained(path)
    return str(path)


def _truncate(checkpoint_dir: str, tmp_path) -> str:
    """The shape of a partially uploaded or interrupted checkpoint."""
    truncated = tmp_path / "truncated"
    truncated.mkdir()
    for entry in pathlib.Path(checkpoint_dir).iterdir():
        if entry.name != WEIGHTS_FILE:
            (truncated / entry.name).write_bytes(entry.read_bytes())
    tensors = load_file(str(pathlib.Path(checkpoint_dir) / WEIGHTS_FILE))
    dropped = next(key for key in tensors if key.endswith("mlp.down_proj.weight"))
    del tensors[dropped]
    save_file(tensors, str(truncated / WEIGHTS_FILE), metadata={"format": "pt"})
    return str(truncated)


def _adapter_for(base_path: str, tmp_path) -> str:
    adapter_dir = tmp_path / "adapter"
    model = get_peft_model(
        Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG)),
        LoraConfig(r=2, target_modules=["q_proj"], task_type="CAUSAL_LM"),
    )
    model.peft_config["default"].base_model_name_or_path = base_path
    model.save_pretrained(str(adapter_dir))
    return str(adapter_dir)


def _load(path: str, *, excuse_task_head: bool):
    return adapter_merge.load_task_base_model(
        path,
        "classification",
        torch.float32,
        None,
        1,
        None,
        excuse_task_head=excuse_task_head,
    )


def test_a_missing_score_head_the_adapter_cannot_supply_is_refused(causal_lm_checkpoint):
    """Ungated, the merge saves a random ``score``: every reward from the checkpoint is noise that looks like a
    score."""
    with pytest.raises(RuntimeError, match="randomly initialized"):
        _load(causal_lm_checkpoint, excuse_task_head=False)


def test_the_adapters_own_head_is_a_legitimate_absence(causal_lm_checkpoint):
    """``modules_to_save`` is how a classification adapter carries the head its base lacks."""
    model = _load(causal_lm_checkpoint, excuse_task_head=True)
    assert isinstance(model, Qwen3ForSequenceClassification)


def test_a_base_that_carries_the_head_loads_the_trained_one(reward_checkpoint):
    """A re-initialized ``score`` of the right shape is the failure guarded here."""
    model = _load(reward_checkpoint, excuse_task_head=False)
    saved = load_file(str(pathlib.Path(reward_checkpoint) / WEIGHTS_FILE))
    assert torch.equal(model.score.weight, saved["score.weight"].to(model.score.weight.dtype))


def _calls(source_path: pathlib.Path, call_name: str) -> list[ast.Call]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == call_name
    ]


def _keyword_values(source_path: pathlib.Path, call_name: str, keyword: str) -> list[ast.expr]:
    calls = _calls(source_path, call_name)
    assert calls, f"no {call_name} call in {source_path.name}"
    values = [kw.value for call in calls for kw in call.keywords if kw.arg == keyword]
    assert len(values) == len(calls), f"a {call_name} call in {source_path.name} does not pass {keyword}"
    return values


def test_the_excuse_is_derived_from_the_adapter_config():
    """Excusing the head unconditionally, as a literal ``True`` does, is the ungated behaviour under another name."""
    source = pathlib.Path(adapter_merge.__file__)
    (derived,) = _keyword_values(source, "load_base_for_adapter", "excuse_task_head")
    assert not isinstance(derived, ast.Constant), (
        "excuse_task_head is passed as a literal — the excuse must be read off peft_config.modules_to_save"
    )
    assert "modules_to_save" in ast.dump(derived), "the excuse must be read off peft_config.modules_to_save"
    for forwarded in _keyword_values(source, "load_base_model", "excuse_task_head"):
        assert isinstance(forwarded, ast.Name) and forwarded.id == "excuse_task_head", (
            "a base load passes its own excuse instead of the one derived from the adapter"
        )


@pytest.mark.parametrize("task", ["causal_lm", "classification", "base"])
def test_a_truncated_source_is_refused_on_every_conversion_path(task, causal_lm_checkpoint, tmp_path):
    """A direct ``from_pretrained`` under ``--task classification`` / ``base`` writes random tensors as a
    complete-looking bf16 model."""
    with pytest.raises(RuntimeError, match="randomly initialized"):
        bf16_conversion.load_model(_truncate(causal_lm_checkpoint, tmp_path), task, dtype=torch.float32)


def test_a_truncated_base_is_refused_under_peft(causal_lm_checkpoint, tmp_path):
    """PEFT's auto-class loads the base through a raw ``from_pretrained``."""
    adapter_dir = _adapter_for(_truncate(causal_lm_checkpoint, tmp_path), tmp_path)
    with pytest.raises(RuntimeError, match="randomly initialized"):
        bf16_conversion.load_model(adapter_dir, "causal_lm", is_peft=True, dtype=torch.float32)


def test_an_intact_base_still_loads_under_peft(causal_lm_checkpoint, tmp_path):
    adapter_dir = _adapter_for(causal_lm_checkpoint, tmp_path)
    model = bf16_conversion.load_model(adapter_dir, "causal_lm", is_peft=True, dtype=torch.float32)
    assert model.merge_and_unload() is not None


def _direct_auto_model_loads(path: pathlib.Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node.func.value.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_pretrained"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id.startswith("AutoModel")
    ]


@pytest.mark.parametrize(
    "script", sorted((REPO_ROOT / "scripts").rglob("*.py")), ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_no_tool_loads_a_model_around_the_coverage_gate(script):
    """``from_pretrained_verified`` raises on a randomly initialized tensor instead of logging it."""
    assert not _direct_auto_model_loads(script), (
        f"{script.name} calls AutoModel*.from_pretrained directly — a truncated or "
        f"wrong-architecture checkpoint would load with random weights and only a log line"
    )


def test_the_shared_loader_applies_the_remote_code_shims_before_the_config_fetch(causal_lm_checkpoint):
    """The config fetch inside the loader is already enough to import a remote modeling file, so order matters as
    much as the call."""
    order: list[str] = []
    real_from_pretrained = model_preparation.AutoConfig.from_pretrained

    def recording_from_pretrained(*args, **kwargs):
        order.append("config")
        return real_from_pretrained(*args, **kwargs)

    with (
        patch.object(model_preparation, REMOTE_CODE_SHIMS, lambda: order.append("shims")),
        patch.object(model_preparation.AutoConfig, "from_pretrained", recording_from_pretrained),
    ):
        model_preparation.auto_load_model(causal_lm_checkpoint, dtype=torch.float32)

    assert order[:2] == ["shims", "config"], order


def test_the_string_path_loader_applies_the_shims_before_it_materializes_weights(causal_lm_checkpoint):
    """Without the shims, a Bailing/Ling file keeps its v4-era ``_tied_weights_keys`` (``save_pretrained`` crashes)
    and lacks the SDPA dispatch shim; a pinned ``model_cls`` skips the fetch."""
    order: list[str] = []
    real_verified = model_preparation.from_pretrained_verified

    def recording_verified(*args, **kwargs):
        order.append("load")
        return real_verified(*args, **kwargs)

    args = SimpleNamespace(model_init_kwargs={"torch_dtype": torch.float32})
    with (
        patch.object(model_preparation, REMOTE_CODE_SHIMS, lambda: order.append("shims")),
        patch.object(model_preparation, "from_pretrained_verified", recording_verified),
    ):
        resolve_trainer_model(causal_lm_checkpoint, args, parallelism_config=None, model_class=Qwen3ForCausalLM)

    assert order[0] == "shims" and order[-1] == "load" and order.count("load") == 1, order


def test_a_pinned_backend_reads_its_config_after_the_shims(causal_lm_checkpoint):
    """That fetch can already import a remote modeling file."""
    order: list[str] = []
    real_from_pretrained = model_preparation.AutoConfig.from_pretrained

    def recording_from_pretrained(*args, **kwargs):
        order.append("config")
        return real_from_pretrained(*args, **kwargs)

    args = SimpleNamespace(model_init_kwargs={"dtype": torch.float32, "attn_implementation": "eager"})
    with (
        patch.object(model_preparation, REMOTE_CODE_SHIMS, lambda: order.append("shims")),
        patch.object(model_preparation.AutoConfig, "from_pretrained", recording_from_pretrained),
    ):
        resolve_trainer_model(causal_lm_checkpoint, args, parallelism_config=None)

    assert order[:2] == ["shims", "config"], order


def _calls_name(func: ast.FunctionDef, name: str) -> bool:
    return any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
        for node in ast.walk(func)
    )


# The Auto*-class seam, the frozen teacher/reference loader and the reward-model scorer materialize weights themselves.
LOADER_MODULES = [
    REPO_ROOT / "halo/model_loading/model_preparation.py",
    REPO_ROOT / "halo/model_loading/parallel/loader.py",
    REPO_ROOT / "halo/model_loading/parallel/frozen_models.py",
    REPO_ROOT / "halo/entrypoints/reward_scoring.py",
]
# The dispatcher applies the shims once for its private per-mode helpers.
_LOADER_FUNCS = {
    node.name: node
    for path in LOADER_MODULES
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
    if isinstance(node, ast.FunctionDef)
    and not node.name.startswith("_")
    and _calls_name(node, "from_pretrained_verified")
}
WEIGHT_MATERIALIZING_LOADERS = sorted(_LOADER_FUNCS)
_SHIM_APPLYING_CALLS = {REMOTE_CODE_SHIMS} | {
    node.name
    for node in ast.walk(ast.parse(LOADER_MODULES[0].read_text(encoding="utf-8")))
    if isinstance(node, ast.FunctionDef) and _calls_name(node, REMOTE_CODE_SHIMS)
}


def test_the_sweep_found_the_string_path_loaders():
    """An empty sweep would pass the assertion below vacuously."""
    assert SHARED_MODEL_LOADER in WEIGHT_MATERIALIZING_LOADERS, WEIGHT_MATERIALIZING_LOADERS
    assert len(WEIGHT_MATERIALIZING_LOADERS) >= 2, WEIGHT_MATERIALIZING_LOADERS


@pytest.mark.parametrize("loader", WEIGHT_MATERIALIZING_LOADERS)
def test_every_loader_that_materializes_weights_applies_the_shims(loader):
    """A new loader that forgets the shims fails here instead of in a Bailing/Ling run whose export crashes hours
    later."""
    assert any(_calls_name(_LOADER_FUNCS[loader], name) for name in _SHIM_APPLYING_CALLS - {loader}), (
        f"{loader}() loads weights from a path but never calls {REMOTE_CODE_SHIMS}() or a helper that does "
        f"({sorted(_SHIM_APPLYING_CALLS - {REMOTE_CODE_SHIMS})}) — a remote-code modeling file imported through it "
        f"reaches none of the v5 compat shims"
    )


MODEL_LOADING_TOOLS = sorted(
    path
    for root in ("scripts", "halo/checkpoint")
    for path in (REPO_ROOT / root).rglob("*.py")
    if imports_name(path, SHARED_MODEL_LOADER)
)


def test_the_sweep_found_the_model_loading_tools():
    """An empty sweep would pass the assertion below vacuously."""
    assert len(MODEL_LOADING_TOOLS) >= 2, MODEL_LOADING_TOOLS


@pytest.mark.parametrize("script", MODEL_LOADING_TOOLS, ids=lambda p: p.stem)
def test_no_tool_re_applies_the_shims_around_the_loader(script):
    """The shims live inside the loader; a per-tool copy makes "did this CLI remember?" a live question."""
    assert not imports_name(script, REMOTE_CODE_SHIMS), (
        f"{script.name} applies the remote-code shims itself; {SHARED_MODEL_LOADER} owns them"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
