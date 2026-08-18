"""How the after-training tools LOAD the checkpoints they rewrite.

Three ways a merge or conversion produces a plausible, wrong artifact:

  * the classification branch of ``merge_peft_adapters``, and every ``convert_to_bf16`` path except
    the causal-LM one, must not reach ``AutoModel*.from_pretrained`` directly and bypass the coverage
    gate every other loader in the toolkit goes through. A truncated or wrong-architecture source
    random-initializes the absent tensors with a log line nobody sees, and the tool writes them out
    as a finished reward model / bf16 checkpoint. The seq-cls head is the one absence that can be
    legitimate — a classification adapter trained on a plain causal-LM base carries it in
    ``modules_to_save`` — and it is legitimate only then, so the excuse is driven by what the
    adapter declares.
  * PEFT's own auto-classes load the base through that same raw ``from_pretrained``, so a tool that
    hands them an adapter directory inherits the hole for every ``--peft`` run.
  * a remote-code base (every Bailing/Ling checkpoint) needs the compat shims applied BEFORE its
    modeling file is imported. They belong to the shared loader itself, not to a convention each
    standalone CLI has to remember — the config fetch inside it is already enough to import a
    remote modeling file.

    python tests/cpu/checkpoint/test_after_training_load_gates.py
"""

from __future__ import annotations

import ast
import pathlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from accelerate import PartialState
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file
from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen3ForSequenceClassification

import scripts.after_training.convert_to_bf16 as convert_to_bf16
import scripts.after_training.merge_peft_adapters as merge_peft_adapters
from src.checkpoint import adapters
from src.distributed.loading import model_loading
from src.models.loading import model_preparation
from tests.common.models import TINY_QWEN3_CONFIG
from tests.common.utils import REPO_ROOT, imports_name

PartialState()  # the loaders log through accelerate's logger

# The shared model loader: a script that binds it builds a transformers model from a checkpoint
# directory, remote-code families included. Derived rather than listed, so a new tool is swept in.
SHARED_MODEL_LOADER = "auto_load_model"
REMOTE_CODE_SHIMS = "apply_remote_code_compat_shims"
WEIGHTS_FILE = "model.safetensors"


@pytest.fixture(scope="module")
def causal_lm_checkpoint(tmp_path_factory) -> str:
    """A plain causal-LM base — the case a classification adapter is normally trained on top of."""
    path = tmp_path_factory.mktemp("causal_base")
    Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG)).save_pretrained(path)
    return str(path)


@pytest.fixture(scope="module")
def reward_checkpoint(tmp_path_factory) -> str:
    """A base that already carries a trained score head."""
    path = tmp_path_factory.mktemp("reward_base")
    Qwen3ForSequenceClassification(Qwen3Config(**TINY_QWEN3_CONFIG, num_labels=1)).save_pretrained(path)
    return str(path)


def _truncate(checkpoint_dir: str, tmp_path) -> str:
    """A copy of ``checkpoint_dir`` with one backbone tensor missing.

    The shape of a partially uploaded or interrupted checkpoint: the file loads, the config is
    intact, and ``from_pretrained`` fills the hole with random weights and a log line.
    """
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
    """A LoRA adapter directory naming ``base_path`` as its base model."""
    adapter_dir = tmp_path / "adapter"
    model = get_peft_model(
        Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG)),
        LoraConfig(r=2, target_modules=["q_proj"], task_type="CAUSAL_LM"),
    )
    model.peft_config["default"].base_model_name_or_path = base_path
    model.save_pretrained(str(adapter_dir))
    return str(adapter_dir)


def _load(path: str, *, excuse_task_head: bool):
    return merge_peft_adapters._load_base_model(
        path,
        "classification",
        torch.float32,
        None,
        1,
        None,
        excuse_task_head=excuse_task_head,
    )


def test_a_missing_score_head_the_adapter_cannot_supply_is_refused(causal_lm_checkpoint):
    """Pre-gate this returned a model with a random ``score`` and the merge saved it: every reward
    the checkpoint went on to produce was noise that looks like a score."""
    with pytest.raises(RuntimeError, match="randomly initialized"):
        _load(causal_lm_checkpoint, excuse_task_head=False)


def test_the_adapters_own_head_is_a_legitimate_absence(causal_lm_checkpoint):
    """``modules_to_save`` is how a classification adapter carries the head its base lacks — the
    merge supplies it, so the base may load without one."""
    model = _load(causal_lm_checkpoint, excuse_task_head=True)
    assert isinstance(model, Qwen3ForSequenceClassification)


def test_a_base_that_carries_the_head_loads_the_trained_one(reward_checkpoint):
    """The gate must not refuse the ordinary case — and the head that comes back has to be the one
    on disk: a re-initialized ``score`` of the right shape is exactly the failure being guarded."""
    model = _load(reward_checkpoint, excuse_task_head=False)
    saved = load_file(str(pathlib.Path(reward_checkpoint) / WEIGHTS_FILE))
    assert torch.equal(model.score.weight, saved["score.weight"].to(model.score.weight.dtype))


def _keyword_value(source_path: pathlib.Path, call_name: str, keyword: str) -> ast.expr:
    """The value node passed as ``keyword`` to the (single) call of ``call_name`` in a source file."""
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == call_name
    ]
    assert len(calls) == 1, f"expected one {call_name} call in {source_path.name}, found {len(calls)}"
    values = [kw.value for kw in calls[0].keywords if kw.arg == keyword]
    assert values, f"{call_name} in {source_path.name} does not pass {keyword}"
    return values[0]


def test_the_excuse_is_derived_from_the_adapter_config():
    """The flag must come from what the adapter declares, not from a caller's default: excusing the
    head unconditionally is the pre-gate behaviour under a new name — and a literal ``True`` here
    is exactly that. Read at the shared merge, which is where every merge tool's base load goes."""
    value = _keyword_value(pathlib.Path(adapters.__file__), "load_base_model", "excuse_task_head")
    assert not isinstance(value, ast.Constant), (
        "adapter_supplies_task_head is passed as a literal — the excuse must be read off peft_config.modules_to_save"
    )
    assert "modules_to_save" in ast.dump(value), "the excuse must be read off peft_config.modules_to_save"


# --- convert_to_bf16: every load path is gated, not just the causal-LM one --------------------


@pytest.mark.parametrize("model_type", ["causal_lm", "classifier", "base"])
def test_a_truncated_source_is_refused_on_every_conversion_path(model_type, causal_lm_checkpoint, tmp_path):
    """``--model_type classifier`` / ``base`` reached ``from_pretrained`` directly, so a checkpoint
    missing tensors was random-initialized and written back out as a complete-looking bf16 model."""
    with pytest.raises(RuntimeError, match="randomly initialized"):
        convert_to_bf16.load_model(_truncate(causal_lm_checkpoint, tmp_path), model_type, dtype=torch.float32)


def test_a_truncated_base_is_refused_under_peft(causal_lm_checkpoint, tmp_path):
    """PEFT's auto-class loads the base through a raw ``from_pretrained``, so ``--peft`` inherited
    the hole for the merge that follows."""
    adapter_dir = _adapter_for(_truncate(causal_lm_checkpoint, tmp_path), tmp_path)
    with pytest.raises(RuntimeError, match="randomly initialized"):
        convert_to_bf16.load_model(adapter_dir, "causal_lm", is_peft=True, dtype=torch.float32)


def test_an_intact_base_still_loads_under_peft(causal_lm_checkpoint, tmp_path):
    """The gate must not refuse the ordinary adapter: the merge has to keep working."""
    adapter_dir = _adapter_for(causal_lm_checkpoint, tmp_path)
    model = convert_to_bf16.load_model(adapter_dir, "causal_lm", is_peft=True, dtype=torch.float32)
    assert model.merge_and_unload() is not None


def _direct_auto_model_loads(path: pathlib.Path) -> list[str]:
    """``AutoModel*.from_pretrained(...)`` calls in a source file — the ungated way in."""
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
    """Derived rather than listed, so a new tool cannot reintroduce the hole: every model load in
    ``scripts/`` goes through ``from_pretrained_verified`` (directly, or via ``auto_load_model`` /
    ``load_distributed_model``), which raises on a randomly initialized tensor instead of logging
    it."""
    assert not _direct_auto_model_loads(script), (
        f"{script.name} calls AutoModel*.from_pretrained directly — a truncated or "
        f"wrong-architecture checkpoint would load with random weights and only a log line"
    )


# --- the remote-code shims belong to the shared loader ----------------------------------------


def test_the_shared_loader_applies_the_remote_code_shims_before_the_config_fetch(causal_lm_checkpoint):
    """A standalone CLI is by definition the first remote-code load of its process, so the shims
    cannot depend on the caller having applied them — and the config fetch inside the loader is
    already enough to import a remote modeling file, so the order matters as much as the call."""
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
    """``load_model_from_pretrained`` is the seam SMPO / offline-GRPO / teacher-distillation reach
    when their ``model:`` is a path string, and it was the one terminal loader in this module that
    never applied the shims — so a Bailing/Ling base loaded through it imported its modeling file
    with the v4-era ``_tied_weights_keys`` list intact (``save_pretrained`` crashes) and without the
    SDPA dispatch shim (the ~190 GiB eager score plane). The shims must also precede the WEIGHT load,
    not just the config fetch: a caller that pins ``model_cls`` skips the fetch entirely.
    """
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
        model_loading.load_model_from_pretrained(causal_lm_checkpoint, args, model_cls=Qwen3ForCausalLM)

    assert order == ["shims", "load"], order


def _calls_name(func: ast.FunctionDef, name: str) -> bool:
    return any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
        for node in ast.walk(func)
    )


# Every module that owns a from-a-path loader: the Auto*-class seam on the utils side and the frozen
# teacher/reference loader are the two that materialize weights themselves; the dispatcher and
# ``load_model_from_pretrained`` reach the shims through the first.
LOADER_MODULES = [
    REPO_ROOT / "src/models/loading/model_preparation.py",
    REPO_ROOT / "src/distributed/loading/model_loading.py",
    REPO_ROOT / "src/distributed/loading/frozen_models.py",
]
# Public entry points only: the private per-mode helpers inside the dispatcher are reached through it,
# and it applies the shims once for all of them.
_LOADER_FUNCS = {
    node.name: node
    for path in LOADER_MODULES
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
    if isinstance(node, ast.FunctionDef)
    and not node.name.startswith("_")
    and _calls_name(node, "from_pretrained_verified")
}
WEIGHT_MATERIALIZING_LOADERS = sorted(_LOADER_FUNCS)


def test_the_sweep_found_the_string_path_loaders():
    """Guard the derivation: an empty sweep would pass the assertion below vacuously."""
    assert SHARED_MODEL_LOADER in WEIGHT_MATERIALIZING_LOADERS, WEIGHT_MATERIALIZING_LOADERS
    assert len(WEIGHT_MATERIALIZING_LOADERS) >= 2, WEIGHT_MATERIALIZING_LOADERS


@pytest.mark.parametrize("loader", WEIGHT_MATERIALIZING_LOADERS)
def test_every_loader_that_materializes_weights_applies_the_shims(loader):
    """Derived from the module rather than listed, so a NEW loader that forgets the shims fails here
    instead of in a Bailing/Ling run whose export crashes hours later."""
    assert _calls_name(_LOADER_FUNCS[loader], REMOTE_CODE_SHIMS), (
        f"{loader}() loads weights from a path but never calls {REMOTE_CODE_SHIMS}() — a remote-code "
        f"modeling file imported through it reaches none of the v5 compat shims"
    )


MODEL_LOADING_SCRIPTS = sorted(
    path for path in (REPO_ROOT / "scripts").rglob("*.py") if imports_name(path, SHARED_MODEL_LOADER)
)


def test_the_sweep_found_the_model_loading_tools():
    """Guard the derivation itself: an empty sweep would pass the assertion below vacuously."""
    assert len(MODEL_LOADING_SCRIPTS) >= 4, MODEL_LOADING_SCRIPTS


@pytest.mark.parametrize("script", MODEL_LOADING_SCRIPTS, ids=lambda p: p.stem)
def test_no_tool_re_applies_the_shims_around_the_loader(script):
    """The shims moved INTO the loader, so a tool applying them again is a copy of a contract that
    now has one owner — and the copy is what made "did this CLI remember?" a live question."""
    assert not imports_name(script, REMOTE_CODE_SHIMS), (
        f"{script.name} applies the remote-code shims itself; {SHARED_MODEL_LOADER} owns them"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
