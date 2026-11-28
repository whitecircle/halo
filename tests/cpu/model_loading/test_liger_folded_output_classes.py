"""After ``ensure_folded_output_classes`` every fused forward liger-kernel ships imports, constructs its output with
keywords the bound class takes, and runs on a family whose decoder output transformers folded into the generic one."""

import ast
import importlib
import inspect
import pkgutil
import sys
from dataclasses import fields

import pytest
import torch
from liger_kernel.transformers import model as upstream_models
from liger_kernel.transformers.model import output_classes
from transformers import Qwen3Config
from transformers.utils import ModelOutput

from halo.model_loading.liger import dispatch
from halo.model_loading.liger.upstream_outputs import ensure_folded_output_classes
from tests.common.models import TINY_QWEN3_CONFIG
from tests.common.tiny_models import TINY_MOE_VLM_FAMILIES, tiny_family_model

FUSED_FORWARD_MODULES = sorted(
    f"{upstream_models.__name__}.{info.name}" for info in pkgutil.iter_modules(upstream_models.__path__)
)
# The fused forward upstream binds to a folded family's composite class, by the tiny model it is called on.
FOLDED_FAMILIES = {
    "qwen3_5_moe": ("liger_kernel.transformers.model.qwen3_5_moe", "lce_forward_conditional_generation")
}


def _output_constructions(module) -> list[tuple[type, set[str]]]:
    """``(output class, keywords)`` for each call in ``module``'s source that constructs a ``ModelOutput``."""
    constructions = []
    for node in ast.walk(ast.parse(inspect.getsource(module))):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        bound = getattr(module, node.func.id, None)
        if isinstance(bound, type) and issubclass(bound, ModelOutput):
            constructions.append((bound, {keyword.arg for keyword in node.keywords if keyword.arg}))
    return constructions


def test_the_roster_reads_modules_that_construct_outputs():
    ensure_folded_output_classes()
    constructing = [name for name in FUSED_FORWARD_MODULES if _output_constructions(importlib.import_module(name))]
    assert len(constructing) > 30, constructing


def test_a_fused_forward_imports_and_its_output_takes_what_it_passes(subtests):
    assert FUSED_FORWARD_MODULES, "the sweep found no liger fused-forward module"
    ensure_folded_output_classes()
    for module_name in FUSED_FORWARD_MODULES:
        with subtests.test(module_name):
            module = importlib.import_module(module_name)
            for output_class, keywords in _output_constructions(module):
                unknown = keywords - {field.name for field in fields(output_class)}
                assert not unknown, f"{module_name} passes {sorted(unknown)} to {output_class.__name__}"


@pytest.mark.parametrize("family", FOLDED_FAMILIES)
def test_a_folded_family_fused_forward_returns_the_stock_loss(family):
    """``skip_logits=False`` keeps the loss on the model's own function, so the forward runs without Triton."""
    ensure_folded_output_classes()
    torch.manual_seed(0)
    model = tiny_family_model(TINY_MOE_VLM_FAMILIES[family]).eval()
    input_ids = torch.randint(3, model.config.text_config.vocab_size, (2, 8))
    stock = model(input_ids=input_ids, labels=input_ids)
    module_name, forward = FOLDED_FAMILIES[family]
    fused = getattr(importlib.import_module(module_name), forward)(
        model, input_ids=input_ids, labels=input_ids, skip_logits=False
    )
    assert fused.rope_deltas is None
    torch.testing.assert_close(fused.loss, stock.loss)
    torch.testing.assert_close(fused.logits, stock.logits)


def test_sanitizing_a_liger_config_supplies_the_classes_upstream_appliers_import(monkeypatch):
    """HF Trainer calls upstream's applier itself at ``train()``, after the trainer sanitized the config."""
    name = "LigerQwen3VLCausalLMOutputWithPast"
    module_name = "liger_kernel.transformers.model.qwen3_vl"
    monkeypatch.delattr(output_classes, name, raising=False)
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    with pytest.raises(ImportError, match=name):
        importlib.import_module(module_name)
    dispatch.sanitize_liger_config({}, needs_ep_layers=False, model_config=Qwen3Config(**TINY_QWEN3_CONFIG))
    importlib.import_module(module_name)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
