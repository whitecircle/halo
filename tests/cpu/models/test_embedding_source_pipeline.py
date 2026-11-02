#!/usr/bin/env python
"""The parallel loader's embedding pipeline keeps the checkpoint's own pipeline or refuses it, never drops it.

sentence-transformers' own load keeps a checkpoint's whole pipeline and aligns its pooling and normalization to the
config. The parallel loader (EP, TP, the grouped-GEMM EP layers at ``expert_parallel_size`` 1) builds its own
backbone, so it rebuilds the rest from the checkpoint's files: each Pooling from its saved config, a Normalize, the
backbone module's settings and the model-level prompts and similarity, aligned to the config the same way. A module
it does not rebuild (a ``Dense`` head) and dropping the checkpoint's Normalize are refused before any weight loads.
A bare backbone takes the pooling and normalization the config describes.

    python tests/cpu/models/test_embedding_source_pipeline.py
"""

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Dense, Normalize, Pooling
from transformers import AutoModel

import scripts.training.embedding as embedding_script
from halo.configs.embedding import EmbeddingConfig, EmbeddingScriptArguments
from halo.configs.script_arguments import ModelArguments
from halo.entrypoints.script import ScriptContext
from halo.trainers.embedding.preloaded_transformer import PreloadedTransformer
from halo.trainers.embedding.trainer import EmbeddingTrainer
from tests.common.parallelism import make_parallelism_config
from tests.common.tiny_models import TINY_DENSE_FAMILY, tiny_family_model, word_level_tokenizer

PROMPTS = {"query": "query: ", "document": ""}
SETTINGS = {"prompts": PROMPTS, "default_prompt_name": "query", "similarity_fn_name": "dot"}


@pytest.fixture(scope="module")
def backbone_config():
    return tiny_family_model(TINY_DENSE_FAMILY).config


def _checkpoint(path: Path, config, modules: list | None, *, settings: dict | None = None) -> str:
    """A checkpoint of ``config``'s backbone and, unless ``modules`` is None, a sentence-transformers pipeline of
    ``modules`` after it, spelled as hub checkpoints spell their ``modules.json`` types."""
    config.save_pretrained(path)
    if modules is not None:
        entries = [{"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"}]
        for idx, module in enumerate(modules, start=1):
            folder = f"{idx}_{type(module).__name__}"
            (path / folder).mkdir()
            module.save(str(path / folder))
            entries.append(
                {"idx": idx, "name": str(idx), "path": folder, "type": f"sentence_transformers.models.{folder[2:]}"}
            )
        (path / "modules.json").write_text(json.dumps(entries))
        (path / PreloadedTransformer.config_file_name).write_text(
            json.dumps({"max_seq_length": 256, "do_lower_case": True})
        )
    if settings is not None:
        (path / "config_sentence_transformers.json").write_text(json.dumps(settings))
    return str(path)


def _assemble(monkeypatch, tmp_path, checkpoint: str, config, loads: list, **embedding) -> SentenceTransformer:
    """``_assemble_preloaded_pipeline`` over ``checkpoint``, its backbone load stubbed with ``config``'s and recorded in
    ``loads``."""

    def load(*_args, **_kwargs):
        loads.append(checkpoint)
        return AutoModel.from_config(config), word_level_tokenizer()

    monkeypatch.setattr(embedding_script, "load_model_and_tokenizer", load)
    monkeypatch.setattr(
        embedding_script, "plan_training_rows", lambda *args, **kwargs: SimpleNamespace(attn_default=None)
    )
    context = ScriptContext(make_parallelism_config(), 0, None, checkpoint, EmbeddingTrainer)
    return embedding_script._assemble_preloaded_pipeline(
        context,
        EmbeddingConfig(output_dir=str(tmp_path / "out"), bf16=False, use_cpu=True, max_length=64, **embedding),
        ModelArguments(model_name_or_path=checkpoint),
        EmbeddingScriptArguments(),
    )


def _script_warnings(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name == embedding_script.logger.name]


def _modules(pipeline: SentenceTransformer) -> list[tuple[str, str | None]]:
    return [(type(module).__name__, getattr(module, "pooling_mode", None)) for module in pipeline]


def test_a_module_the_pipeline_does_not_rebuild_is_refused_before_the_load(monkeypatch, tmp_path, backbone_config):
    hidden = backbone_config.hidden_size
    checkpoint = _checkpoint(
        tmp_path / "ckpt", backbone_config, [Pooling(hidden, "lasttoken"), Dense(hidden, hidden), Normalize()]
    )
    loads = []

    with pytest.raises(ValueError, match=r"\['sentence_transformers.models.Dense'\] past its backbone"):
        _assemble(monkeypatch, tmp_path, checkpoint, backbone_config, loads, pooling_mode="lasttoken")
    assert not loads


def test_the_checkpoints_pipeline_and_settings_are_kept(monkeypatch, tmp_path, backbone_config):
    hidden = backbone_config.hidden_size
    checkpoint = _checkpoint(
        tmp_path / "ckpt",
        backbone_config,
        [Pooling(hidden, "lasttoken", include_prompt=False), Normalize()],
        settings=SETTINGS,
    )

    loads = []

    pipeline = _assemble(monkeypatch, tmp_path, checkpoint, backbone_config, loads, pooling_mode="lasttoken")

    assert loads == [checkpoint]
    assert _modules(pipeline) == [("PreloadedTransformer", None), ("Pooling", "lasttoken"), ("Normalize", None)]
    assert pipeline[1].include_prompt is False
    assert pipeline[0].do_lower_case is True
    assert pipeline.max_seq_length == 64
    assert (pipeline.prompts, pipeline.default_prompt_name, pipeline.similarity_fn_name) == (PROMPTS, "query", "dot")


def test_dropping_the_checkpoints_normalize_is_refused_before_the_load(monkeypatch, tmp_path, backbone_config):
    checkpoint = _checkpoint(tmp_path / "ckpt", backbone_config, [Pooling(backbone_config.hidden_size), Normalize()])
    loads = []

    with pytest.raises(ValueError, match="normalize_embeddings=False"):
        _assemble(monkeypatch, tmp_path, checkpoint, backbone_config, loads, normalize_embeddings=False)
    assert not loads


def test_the_config_overrides_the_checkpoints_pooling_and_adds_normalize_with_warnings(
    monkeypatch, tmp_path, backbone_config, caplog
):
    checkpoint = _checkpoint(tmp_path / "ckpt", backbone_config, [Pooling(backbone_config.hidden_size, "mean")])

    with caplog.at_level(logging.WARNING):
        pipeline = _assemble(monkeypatch, tmp_path, checkpoint, backbone_config, [], pooling_mode="lasttoken")

    assert _modules(pipeline)[1:] == [("Pooling", "lasttoken"), ("Normalize", None)]
    warnings = _script_warnings(caplog)
    assert any(
        "Overriding the checkpoint's pooling_mode 'mean' with the configured 'lasttoken'" in w for w in warnings
    )
    assert any("Appending a Normalize module" in warning for warning in warnings)


def test_a_bare_backbone_takes_the_configured_pipeline_without_warnings(
    monkeypatch, tmp_path, backbone_config, caplog
):
    checkpoint = _checkpoint(tmp_path / "ckpt", backbone_config, None)

    loads = []

    with caplog.at_level(logging.WARNING):
        pipeline = _assemble(monkeypatch, tmp_path, checkpoint, backbone_config, loads, pooling_mode="lasttoken")

    assert loads == [checkpoint]
    assert _modules(pipeline) == [("PreloadedTransformer", None), ("Pooling", "lasttoken"), ("Normalize", None)]
    assert pipeline.default_prompt_name is None and not any(pipeline.prompts.values())
    assert not _script_warnings(caplog)


def test_the_trainers_pipeline_config_rebuilds_unchanged(monkeypatch, tmp_path, backbone_config):
    """A checkpoint the embedding trainer saved, as a resume reloads it: the module types its writer spells."""
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    backbone_config.save_pretrained(checkpoint)
    saved = SentenceTransformer(
        modules=[
            PreloadedTransformer(AutoModel.from_config(backbone_config), word_level_tokenizer(), do_lower_case=True),
            Pooling(backbone_config.hidden_size, "lasttoken", include_prompt=False),
            Normalize(),
        ],
        **SETTINGS,
    )
    EmbeddingTrainer._save_st_pipeline_config(None, saved, str(checkpoint))

    pipeline = _assemble(monkeypatch, tmp_path, str(checkpoint), backbone_config, [], pooling_mode="lasttoken")

    assert _modules(pipeline) == _modules(saved)
    assert pipeline[1].include_prompt is False and pipeline[0].do_lower_case is True
    assert (pipeline.prompts, pipeline.default_prompt_name, pipeline.similarity_fn_name) == (PROMPTS, "query", "dot")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
