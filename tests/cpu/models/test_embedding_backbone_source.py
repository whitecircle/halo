#!/usr/bin/env python
"""The embedding backbone loads outside ``load_distributed_model``, so it must use the commit the ranks agreed on.
The fetch takes the whole repo, since the pipeline reads its pooling and dense modules from subfolders."""

from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

import scripts.training.embedding as embedding_script
from halo.configs.embedding import EmbeddingConfig
from halo.configs.script_arguments import ModelArguments
from halo.model_loading.parallel.model_source import ModelSource
from tests.common.models import TINY_QWEN3_CONFIG
from tests.common.tiny_models import word_level_tokenizer

REPO = "org/model"
COMMIT_A = "a" * 40


class _BackboneBuilt(Exception):
    """Stops the embedding build once sentence-transformers was asked for the backbone."""


def test_the_embedding_backbone_loads_at_the_agreed_revision(monkeypatch, tmp_path):
    seen = {}

    def resolve(model_name_or_path, revision, *, tag, whole_repo=False):
        seen["resolve"] = (model_name_or_path, revision, tag, whole_repo)
        return ModelSource(model_name_or_path, COMMIT_A, model_name_or_path)

    def build(model_name_or_path, *, revision, **kwargs):
        seen["load"] = (model_name_or_path, revision)
        raise _BackboneBuilt

    def config(model_name_or_path, *, revision, trust_remote_code):
        seen["config"] = (model_name_or_path, revision)
        return Qwen3Config(**TINY_QWEN3_CONFIG)

    monkeypatch.setattr(embedding_script, "resolve_model_source", resolve)
    monkeypatch.setattr(embedding_script, "resolve_build_config", config)
    monkeypatch.setattr(embedding_script, "SentenceTransformer", build)
    runtime = SimpleNamespace(
        parallelism_config=SimpleNamespace(
            has_ep_group=False, is_tp_mode=False, needs_ep_layers=False, max_concurrent_loading=None
        ),
        model_source=REPO,
    )
    with pytest.raises(_BackboneBuilt):
        embedding_script.build_sentence_transformer(
            runtime,
            EmbeddingConfig(output_dir=str(tmp_path), bf16=False, use_cpu=True),
            ModelArguments(model_name_or_path=REPO, model_revision="v1"),
            SimpleNamespace(reset_sinks=True, train_sinks=False),
        )
    assert seen == {
        "resolve": (REPO, "v1", "embedding_model", True),
        "config": (REPO, COMMIT_A),
        "load": (REPO, COMMIT_A),
    }


@pytest.mark.parametrize("attn_implementation", ["sdpa", None], ids=["pinned", "unset"])
def test_the_embedding_backbone_is_warmed_after_its_run_policies(monkeypatch, tmp_path, attn_implementation):
    """Unwarmed, a pinned FA4 or Gemma 4 flex-sliding backbone compiles at one rank's first forward as peers wait."""
    checkpoint = tmp_path / "base"
    Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG)).save_pretrained(checkpoint)
    word_level_tokenizer().save_pretrained(checkpoint)
    order = []
    run_policies = embedding_script.apply_run_model_policies
    monkeypatch.setattr(
        embedding_script,
        "apply_run_model_policies",
        lambda model, *args, **kwargs: order.append("policies") or run_policies(model, *args, **kwargs),
    )
    monkeypatch.setattr(
        embedding_script, "warm_attention_kernels", lambda model, *, dtype: order.append(("warmup", model, dtype))
    )
    runtime = SimpleNamespace(
        parallelism_config=SimpleNamespace(
            has_ep_group=False,
            is_tp_mode=False,
            needs_ep_layers=False,
            fp32_non_ep_params=False,
            max_concurrent_loading=None,
        ),
        model_source=str(checkpoint),
    )
    st_model = embedding_script.build_sentence_transformer(
        runtime,
        EmbeddingConfig(output_dir=str(tmp_path / "run"), bf16=True, use_cpu=True, max_length=32),
        ModelArguments(model_name_or_path=str(checkpoint), attn_implementation=attn_implementation),
        SimpleNamespace(reset_sinks=True, train_sinks=False),
    )
    assert order == ["policies", ("warmup", st_model[0].auto_model, torch.bfloat16)]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
