#!/usr/bin/env python
"""The embedding script loads a MoE through the parallel loader wherever the run asks for its EP layers.

The grouped-GEMM EP layers (``use_grouped_gemm``, the default) exist only on the parallel loader's path, which
every other training script always takes: sentence-transformers' own load builds the stock experts, and with them
``use_grouped_gemm`` and ``fsdp_shard_ep1_experts`` would be inert while the resume resolver, the best-model gate and
the balancing resolution read the run as wrapped. So at ``expert_parallel_size`` 1 an EP family's backbone (its
multimodal wrapper too) takes the parallel loader, while a dense backbone, a MoE no EP layer class wraps (Mixtral)
and any backbone at ``use_grouped_gemm: false`` keep sentence-transformers' load and the checkpoint's pipeline.

Injected LoRA, which those EP layers refuse, is refused by the script before the model loads, on the same verdict.

    python tests/cpu/models/test_embedding_loader_route.py
"""

import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest

import scripts.training.embedding as embedding_script
from halo.configs.embedding import EmbeddingConfig, EmbeddingScriptArguments
from halo.configs.script_arguments import ModelArguments
from halo.entrypoints.script import ScriptContext
from halo.trainers.embedding.trainer import EmbeddingTrainer
from tests.common.parallelism import make_parallelism_config
from tests.common.tiny_models import (
    TINY_DENSE_FAMILY,
    TINY_MOE_FAMILIES,
    TINY_MOE_VLM_FAMILIES,
    TINY_UNWRAPPED_MOE_FAMILY,
    tiny_family_model,
)

# Remote-code families read their config through code the checkpoint ships, which a config-only directory lacks.
EP_FAMILIES = {
    **{name: family for name, family in TINY_MOE_FAMILIES.items() if not family.trust_remote_code},
    "gemma4_vlm": TINY_MOE_VLM_FAMILIES["gemma4"],
}
STOCK_FAMILIES = {"qwen3": TINY_DENSE_FAMILY, "mixtral": TINY_UNWRAPPED_MOE_FAMILY}
PARALLEL = "parallel loader"
SENTENCE_TRANSFORMERS = "sentence-transformers"


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory) -> dict[str, str]:
    """Each family's tiny config saved alone: the route reads the checkpoint's config, never its weights."""
    root = tmp_path_factory.mktemp("embedding_loader_route")
    paths = {}
    for name, family in {**EP_FAMILIES, **STOCK_FAMILIES}.items():
        paths[name] = str(root / name)
        tiny_family_model(family).config.save_pretrained(paths[name])
    return paths


def _context(checkpoint: str, **parallelism) -> ScriptContext:
    return ScriptContext(
        make_parallelism_config(world_size=2, gpus_per_node=2, **parallelism), 0, None, checkpoint, EmbeddingTrainer
    )


def _route(monkeypatch, tmp_path, checkpoint: str, **parallelism) -> list[str]:
    """The loaders ``build_sentence_transformer`` takes for ``checkpoint`` under ``parallelism``."""
    taken = []
    pipeline = SimpleNamespace(model_card_data=SimpleNamespace(add_tags=lambda tags: None))
    monkeypatch.setattr(
        embedding_script, "_assemble_preloaded_pipeline", lambda *args: taken.append(PARALLEL) or pipeline
    )
    monkeypatch.setattr(
        embedding_script, "_load_sentence_transformer", lambda *args: taken.append(SENTENCE_TRANSFORMERS) or pipeline
    )
    embedding_script.build_sentence_transformer(
        _context(checkpoint, **parallelism),
        EmbeddingConfig(output_dir=str(tmp_path), bf16=False, use_cpu=True),
        ModelArguments(model_name_or_path=checkpoint),
        EmbeddingScriptArguments(),
    )
    return taken


@pytest.mark.parametrize("family", sorted(EP_FAMILIES))
def test_an_ep_family_takes_the_parallel_loader_by_default(monkeypatch, tmp_path, checkpoints, family):
    assert _route(monkeypatch, tmp_path, checkpoints[family]) == [PARALLEL]


@pytest.mark.parametrize("family", sorted(EP_FAMILIES))
def test_an_ep_family_keeps_sentence_transformers_without_grouped_gemm(monkeypatch, tmp_path, checkpoints, family):
    assert _route(monkeypatch, tmp_path, checkpoints[family], use_grouped_gemm=False) == [SENTENCE_TRANSFORMERS]


@pytest.mark.parametrize("family", sorted(STOCK_FAMILIES))
def test_a_backbone_without_an_ep_layer_class_keeps_sentence_transformers(monkeypatch, tmp_path, checkpoints, family):
    assert _route(monkeypatch, tmp_path, checkpoints[family]) == [SENTENCE_TRANSFORMERS]


@pytest.mark.parametrize("parallelism", [{"tp_size": 2}, {"ep_size": 2}], ids=["tp", "ep"])
def test_tp_and_ep_take_the_parallel_loader_on_any_backbone(monkeypatch, tmp_path, checkpoints, parallelism):
    assert _route(monkeypatch, tmp_path, checkpoints["qwen3"], **parallelism) == [PARALLEL]


@pytest.mark.parametrize("family", sorted(EP_FAMILIES))
def test_lora_on_the_ep1_layers_is_refused_before_the_load(checkpoints, family):
    with pytest.raises(ValueError, match="or on the grouped-GEMM EP layers"):
        embedding_script.reject_lora_before_the_load(
            _context(checkpoints[family]), ModelArguments(model_name_or_path=checkpoints[family], use_peft=True)
        )


def test_lora_under_ep_is_refused_before_the_load(checkpoints):
    with pytest.raises(ValueError, match="not supported with Expert Parallelism"):
        embedding_script.reject_lora_before_the_load(
            _context(checkpoints["gpt_oss"], ep_size=2),
            ModelArguments(model_name_or_path=checkpoints["gpt_oss"], use_peft=True),
        )


@pytest.mark.parametrize(
    ("family", "parallelism"),
    [("gpt_oss", {"use_grouped_gemm": False}), ("mixtral", {}), ("qwen3", {}), ("gpt_oss", {"tp_size": 2})],
    ids=["stock_experts", "no_ep_layer_class", "dense", "tp_left_to_its_own_gate"],
)
def test_lora_without_the_ep1_layers_passes_before_the_load(checkpoints, family, parallelism):
    embedding_script.reject_lora_before_the_load(
        _context(checkpoints[family], **parallelism),
        ModelArguments(model_name_or_path=checkpoints[family], use_peft=True),
    )


@pytest.mark.parametrize("parallelism", [{}, {"ep_size": 2}], ids=["ep1_layers", "ep"])
def test_a_full_fine_tune_passes_before_the_load(checkpoints, parallelism):
    embedding_script.reject_lora_before_the_load(
        _context(checkpoints["gpt_oss"], **parallelism), ModelArguments(model_name_or_path=checkpoints["gpt_oss"])
    )


def test_main_refuses_lora_before_it_builds_the_model():
    calls = {
        node.func.id: node.lineno
        for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(embedding_script.main))))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert calls["reject_lora_before_the_load"] < calls["build_sentence_transformer"], calls


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
