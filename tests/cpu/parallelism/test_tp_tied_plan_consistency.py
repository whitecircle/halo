#!/usr/bin/env python
"""Tied embedding/head TP contract: both ends shard, or neither does.

transformers shards module by module (``apply_tensor_parallelism``) and ties afterwards, so a tied
pair ends as ONE parameter — a tied config's injected ``embed_tokens: embedding_rowwise`` and a
ForCausalLM's ``lm_head: colwise_gather_output`` agree on ``Shard(0)`` of the vocab dim, which is
what makes the pair vocab-parallel instead of a full replica per rank. ``consistent_tied_tp_plan``
therefore leaves a symmetric plan alone and only drops the lone entry of an asymmetric one;
``validate_tied_pair_consistent`` fails the load loud when the loaded pair and the applied plan
disagree. The GPU behavioural proof lives in ``tests/gpu/parallelism/tp/test_tp_correctness.py``.

Run: python tests/cpu/parallelism/test_tp_tied_plan_consistency.py
"""

import pytest
import torch.nn as nn
from torch.distributed.tensor import Shard, distribute_tensor
from transformers import CONFIG_MAPPING, AutoConfig, AutoModelForCausalLM, PretrainedConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from src.distributed.checkpoint.write import gather_saveable_tensors
from src.distributed.tensor_parallel.tie_plan import (
    consistent_tied_tp_plan,
    validate_tied_pair_consistent,
)
from tests.common.distributed import fake_process_group_mesh
from tests.common.models import TINY_QWEN3_CONFIG

# A tied config, per config shape: the flag has to reach the BACKBONE config, which is the outer one
# for a plain model and the text sub-config for a composite.
TIED_CONFIG_KWARGS = {
    "qwen3": {"tie_word_embeddings": True},
    "qwen3_5": {"text_config": {"tie_word_embeddings": True}},
}


def _tiny_tied_model(tie: bool = True):
    config = AutoConfig.for_model("qwen3", **{**TINY_QWEN3_CONFIG, "tie_word_embeddings": tie})
    return AutoModelForCausalLM.from_config(config)


@pytest.mark.parametrize("model_type", sorted(TIED_CONFIG_KWARGS), ids=["plain", "composite"])
def test_a_symmetric_tied_plan_is_left_alone(model_type):
    """Both ends of the tie carry a plan entry, so both must survive the load: dropping either one
    costs the vocab-dim shard of the embedding AND the head (a full replica per rank on a 150k+
    vocabulary), and dropping only one is the mixed plain/DTensor forward crash."""
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
    """A multimodal wrapper class declares no ``lm_head`` entry while its text config still injects
    ``embed_tokens``. Sharding only the embedding hands the tied head a DTensor its un-transformed
    forward cannot multiply, so the entry comes out and the pair stays replicated."""
    config = CONFIG_MAPPING["gemma3"]()
    config.get_text_config().tie_word_embeddings = True
    backbone_config = config.get_text_config()
    concrete = AutoModelForCausalLM._model_mapping[type(config)]
    original = dict(backbone_config.base_model_tp_plan)
    assert "embed_tokens" in original, "premise: the tied text config injects the embedding entry"
    assert "lm_head" not in (concrete._tp_plan or {}), "premise: this wrapper plans no head"

    with consistent_tied_tp_plan(AutoModelForCausalLM, config):
        assert "embed_tokens" not in backbone_config.base_model_tp_plan
        assert set(backbone_config.base_model_tp_plan) == set(original) - {"embed_tokens"}
    assert backbone_config.base_model_tp_plan == original


def test_a_head_only_plan_drops_the_head():
    """The mirror asymmetry — a class planning ``lm_head`` over a backbone shipping no
    ``base_model_tp_plan``. The tie then overwrites the sharded head with the replicated embedding
    while the head keeps TP's input/output transforms, which is the same crash from the other side."""

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
    """A config class outside the Auto mapping (remote code) declares no tie keys to derive from —
    the context manager must yield instead of raising KeyError; the post-load check is the guard
    that remains."""

    class _RemoteConfig(PretrainedConfig):
        model_type = "halo-test-remote-tied"

    config = _RemoteConfig()
    config.tie_word_embeddings = True
    with consistent_tied_tp_plan(AutoModelForCausalLM, config):
        pass


def test_validate_rejects_a_pair_the_load_left_untied():
    """transformers ties AFTER sharding, so two objects here mean it refused the tie — each end
    would then train on half the tied gradient."""
    model = _tiny_tied_model()
    model.lm_head.weight = nn.Parameter(model.lm_head.weight.detach().clone())
    with pytest.raises(RuntimeError, match="two independent parameters"):
        validate_tied_pair_consistent(model, {})


def test_validate_rejects_a_replicated_pair_the_plan_shards():
    """The applied plan gave the head TP's input/output transforms while the tie handed it a plain
    replica: the forward dies on mixed plain/DTensor operands. Loud here, not there."""
    model = _tiny_tied_model()
    assert model.lm_head.weight is model.get_input_embeddings().weight, "premise: the pair is tied"
    validate_tied_pair_consistent(model, {})  # no plan entry, replicated weight — consistent
    with pytest.raises(RuntimeError, match="disagree"):
        validate_tied_pair_consistent(model, {"lm_head": "colwise_gather_output"})


def test_validate_accepts_a_sharded_pair_the_plan_shards():
    """Anti-vacuity for the branch above: a DTensor weight under a sharding entry is the healthy
    vocab-parallel pair and must pass — and the same pair under an EMPTY plan must still raise."""
    with fake_process_group_mesh(rank=0, world_size=2) as mesh:
        model = _tiny_tied_model()
        embedding = model.get_input_embeddings()
        embedding.weight = nn.Parameter(distribute_tensor(embedding.weight.detach(), mesh, [Shard(0)]))
        model.lm_head.weight = embedding.weight
        validate_tied_pair_consistent(model, {"lm_head": "colwise_gather_output"})
        with pytest.raises(RuntimeError, match="disagree"):
            validate_tied_pair_consistent(model, {})


def test_a_tied_pair_reaches_the_export_as_one_key():
    """One key, one gather. transformers own 5.16 ``save_pretrained`` gathers per key and only then
    de-duplicates by storage, so at ``tp_size > 1`` it writes ``lm_head.weight`` a second time — and
    that duplicate is exactly what sends the NEXT load into the tie equality collective. The
    toolkit gathered save de-duplicates FIRST (``named_parameters()`` visits a tied pair once), so
    the export carries the embedding key alone and pays one ``full_tensor()`` instead of two.
    """
    tied_keys = set(gather_saveable_tensors(_tiny_tied_model(), retain=True))
    assert "model.embed_tokens.weight" in tied_keys
    assert "lm_head.weight" not in tied_keys
    # Anti-vacuity: an UNTIED head is its own parameter and must still reach the export.
    assert "lm_head.weight" in set(gather_saveable_tensors(_tiny_tied_model(tie=False), retain=True))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
