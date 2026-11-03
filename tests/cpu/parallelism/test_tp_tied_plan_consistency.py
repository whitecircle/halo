#!/usr/bin/env python
"""Tied embedding/head TP contract: both ends shard, or neither. A tied pair ends as one parameter on ``Shard(0)``; the
GPU proof is ``tests/gpu/parallelism/tp/test_tp_correctness.py``."""

import pytest
import torch.nn as nn
from torch.distributed.tensor import Shard, distribute_tensor
from transformers import CONFIG_MAPPING, AutoConfig, AutoModelForCausalLM, PretrainedConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from halo.checkpoint.parallel.writing import chunked_saveable_tensors
from halo.distributed.tensor_parallel.hf_plan import consistent_tied_tp_plan, validate_tied_pair_consistent
from tests.common.distributed import fake_process_group_mesh
from tests.common.models import TINY_QWEN3_CONFIG

# The flag has to reach the backbone config: the outer one for a plain model, the text sub-config for a composite.
TIED_CONFIG_KWARGS = {
    "qwen3": {"tie_word_embeddings": True},
    "qwen3_5": {"text_config": {"tie_word_embeddings": True}},
}


def _tiny_tied_model(tie: bool = True):
    config = AutoConfig.for_model("qwen3", **{**TINY_QWEN3_CONFIG, "tie_word_embeddings": tie})
    return AutoModelForCausalLM.from_config(config)


def _saved_keys(model) -> set[str]:
    return {key for chunk in chunked_saveable_tensors(model, retain=True) for key in chunk}


@pytest.mark.parametrize("model_type", sorted(TIED_CONFIG_KWARGS), ids=["plain", "composite"])
def test_a_symmetric_tied_plan_is_left_alone(model_type):
    """Dropping either costs a full replica per rank on a 150k+ vocabulary; dropping one is the mixed plain/DTensor
    crash."""
    config = CONFIG_MAPPING[model_type](**TIED_CONFIG_KWARGS[model_type])
    backbone_config = config.get_text_config()
    concrete = AutoModelForCausalLM._model_mapping[type(config)]
    assert "embed_tokens" in (backbone_config.base_model_tp_plan or {}), (
        "premise: a tied config's backbone plan shards the embedding"
    )
    assert "lm_head" in (concrete._tp_plan or {}), "premise: the ForCausalLM plan shards the head"

    with consistent_tied_tp_plan(AutoModelForCausalLM, config):
        assert backbone_config.base_model_tp_plan["embed_tokens"] == "embedding_rowwise"
        assert concrete._tp_plan["lm_head"] == "colwise_gather_output"


def test_an_embedding_only_plan_drops_the_embedding():
    """Sharding only the embedding hands the tied head a DTensor its un-transformed forward cannot multiply."""

    class _EmbeddingOnly:
        _tp_plan = {}
        _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    config = CONFIG_MAPPING["qwen3"](tie_word_embeddings=True)
    original = dict(config.base_model_tp_plan)
    assert "embed_tokens" in original, "premise: the tied text config injects the embedding entry"

    with consistent_tied_tp_plan(_EmbeddingOnly, config):
        assert "embed_tokens" not in config.base_model_tp_plan
        assert set(config.base_model_tp_plan) == set(original) - {"embed_tokens"}
    assert config.base_model_tp_plan == original


def test_a_head_only_plan_drops_the_head():
    """The tie overwrites the sharded head with the replicated embedding while the head keeps TP's transforms, the
    same crash."""

    class _HeadOnly:
        _tp_plan = {"lm_head": "colwise_gather_output"}
        _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    config = CONFIG_MAPPING["qwen3"](tie_word_embeddings=True)
    config.base_model_tp_plan = None

    with consistent_tied_tp_plan(_HeadOnly, config):
        assert _HeadOnly._tp_plan == {}
    assert _HeadOnly._tp_plan == {"lm_head": "colwise_gather_output"}


def test_restore_runs_on_exception():
    config = CONFIG_MAPPING["gemma3"]()
    config.get_text_config().tie_word_embeddings = True
    original = dict(config.get_text_config().base_model_tp_plan)
    with pytest.raises(RuntimeError, match="boom"):
        with consistent_tied_tp_plan(AutoModelForCausalLM, config):
            raise RuntimeError("boom")
    assert config.get_text_config().base_model_tp_plan == original


def test_untied_config_is_a_no_op():
    config = CONFIG_MAPPING["qwen3"](tie_word_embeddings=False)
    plan = Qwen3ForCausalLM._tp_plan
    base_plan = config.base_model_tp_plan
    with consistent_tied_tp_plan(AutoModelForCausalLM, config):
        assert Qwen3ForCausalLM._tp_plan is plan
        assert config.base_model_tp_plan is base_plan


def test_remote_code_config_class_does_not_crash():
    """A class outside the Auto mapping declares no tie keys, so the context manager yields; the post-load check is
    the guard left."""

    class _RemoteConfig(PretrainedConfig):
        model_type = "halo-test-remote-tied"

    config = _RemoteConfig()
    config.tie_word_embeddings = True
    with consistent_tied_tp_plan(AutoModelForCausalLM, config):
        pass


def test_validate_rejects_a_pair_the_load_left_untied():
    """transformers ties after sharding, so two objects mean it refused the tie, and each end trains on half the
    gradient."""
    model = _tiny_tied_model()
    model.lm_head.weight = nn.Parameter(model.lm_head.weight.detach().clone())
    with pytest.raises(RuntimeError, match="two independent parameters"):
        validate_tied_pair_consistent(model, {})


def test_validate_rejects_a_replicated_pair_the_plan_shards():
    """The forward would die on mixed plain/DTensor operands."""
    model = _tiny_tied_model()
    assert model.lm_head.weight is model.get_input_embeddings().weight, "premise: the pair is tied"
    validate_tied_pair_consistent(model, {})  # No plan entry, replicated weight: consistent.
    with pytest.raises(RuntimeError, match="disagree"):
        validate_tied_pair_consistent(model, {"lm_head": "colwise_gather_output"})


def test_validate_accepts_a_sharded_pair_the_plan_shards():
    """The same pair under an empty plan must still raise."""
    with fake_process_group_mesh(rank=0, world_size=2) as mesh:
        model = _tiny_tied_model()
        embedding = model.get_input_embeddings()
        embedding.weight = nn.Parameter(distribute_tensor(embedding.weight.detach(), mesh, [Shard(0)]))
        model.lm_head.weight = embedding.weight
        validate_tied_pair_consistent(model, {"lm_head": "colwise_gather_output"})
        with pytest.raises(RuntimeError, match="disagree"):
            validate_tied_pair_consistent(model, {})


def test_a_tied_pair_reaches_the_export_as_one_key():
    """transformers 5.20 gathers per key, then de-duplicates, writing ``lm_head.weight`` again, which sends the next
    load into the tie collective."""
    tied_keys = _saved_keys(_tiny_tied_model())
    assert "model.embed_tokens.weight" in tied_keys
    assert "lm_head.weight" not in tied_keys
    # Anti-vacuity: an untied head must still reach the export.
    assert "lm_head.weight" in _saved_keys(_tiny_tied_model(tie=False))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
