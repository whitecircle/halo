"""The stage-aware loader on a tied checkpoint shipping both keys. ``torch.equal`` has no meta kernel, so the shell is
built with the tie suppressed; the gate still permits a declared tie with no output embedding."""

import json
import os

import pytest
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, Qwen3Config, Qwen3ForCausalLM

from halo.distributed.pipeline_parallel.model_rules import reject_unsupported_pp_structure
from halo.model_loading.parallel.pp import _load_stage_lazy
from halo.models.config_access import tie_flag_configs
from tests.common.models import TINY_QWEN3_CONFIG

PP_SIZE = 2


@pytest.fixture(scope="module")
def tied_checkpoint(tmp_path_factory) -> str:
    """Written untied, then re-declared: HF compares the two only when neither key is missing, which triggers the
    ``torch.equal`` branch."""
    torch.manual_seed(0)
    config = Qwen3Config(**TINY_QWEN3_CONFIG)
    path = tmp_path_factory.mktemp("pp_tied_ckpt")
    Qwen3ForCausalLM(config).to(torch.bfloat16).save_pretrained(path, safe_serialization=True)

    config_path = os.path.join(path, "config.json")
    with open(config_path) as fh:
        raw = json.load(fh)
    raw["tie_word_embeddings"] = True
    with open(config_path, "w") as fh:
        json.dump(raw, fh)
    return str(path)


def _load(path: str, pp_rank: int = 0, **kwargs):
    return _load_stage_lazy(
        path,
        pp_rank,
        PP_SIZE,
        config=AutoConfig.from_pretrained(path, trust_remote_code=False),
        trust_remote_code=False,
        dtype=torch.bfloat16,
        **kwargs,
    )


def test_tied_causal_lm_is_rejected_with_the_pipeline_diagnosis(tied_checkpoint):
    """Must name the tie and PP, not an aten op with no meta kernel."""
    with pytest.raises(ValueError) as excinfo:
        _load(tied_checkpoint)

    message = str(excinfo.value)
    assert "tie_word_embeddings" in message
    assert "pipeline parallelism" in message
    assert "aten::equal" not in message


def test_declared_tie_without_an_output_head_still_loads(tied_checkpoint):
    """Rejecting on the config flag alone would take sequence classification and reward modelling off PP."""
    stage = _load(tied_checkpoint, model_class=AutoModelForSequenceClassification)

    assert stage.get_output_embeddings() is None
    assert not stage.state_dict()["score.weight"].is_meta


def test_the_declared_tie_flag_survives_the_untied_meta_build(tied_checkpoint):
    """That config is what every stage saves."""
    stage = _load(tied_checkpoint, model_class=AutoModelForSequenceClassification)

    assert stage.config.tie_word_embeddings is True


class _CompositeConfig(Qwen3Config):
    """A config whose decoder lives on a ``text_config`` sub-config."""

    sub_configs = {"text_config": Qwen3Config}


def _composite_declaring_the_tie_on_text_only():
    text = Qwen3Config(**{**TINY_QWEN3_CONFIG, "tie_word_embeddings": True})
    composite = _CompositeConfig(**{**TINY_QWEN3_CONFIG, "tie_word_embeddings": False})
    composite.text_config = text
    return composite


def test_tie_flag_is_collected_from_every_config_level():
    """transformers ties across every submodel, so suppressing only the top level leaves the tie live."""
    text = Qwen3Config(**{**TINY_QWEN3_CONFIG, "tie_word_embeddings": True})
    assert {path: id(h) for path, h in tie_flag_configs(text).items()} == {"": id(text)}

    composite = _composite_declaring_the_tie_on_text_only()
    assert {path: id(h) for path, h in tie_flag_configs(composite).items()} == {
        "": id(composite),
        "text_config.": id(composite.text_config),
    }


def test_a_tie_declared_only_on_a_sub_config_is_rejected():
    """Top-level reading lets stage 0's embedding and the last stage's head be one parameter on different ranks."""
    model = Qwen3ForCausalLM(Qwen3Config(**{**TINY_QWEN3_CONFIG, "tie_word_embeddings": False}))
    assert model.lm_head.weight is not model.model.embed_tokens.weight, "untied storage is the premise"
    model.config = _composite_declaring_the_tie_on_text_only()

    with pytest.raises(ValueError, match="tie_word_embeddings"):
        reject_unsupported_pp_structure(model)


def test_a_sub_config_tie_without_an_output_head_still_loads():
    model = AutoModelForSequenceClassification.from_config(
        Qwen3Config(**{**TINY_QWEN3_CONFIG, "tie_word_embeddings": False, "num_labels": 2})
    )
    model.config = _composite_declaring_the_tie_on_text_only()

    assert model.get_output_embeddings() is None
    reject_unsupported_pp_structure(model)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
