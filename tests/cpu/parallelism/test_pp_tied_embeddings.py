"""The stage-aware pipeline loader on a checkpoint whose embedding and head are TIED.

A tied checkpoint that ships both keys (``Qwen3-0.6B``, ``Qwen3-1.7B``, and most tied hub exports)
cannot go through a plain ``from_pretrained(device_map="meta")``: transformers decides whether to
honour a declared tie by comparing the two candidate weights with ``torch.equal``, which has no meta
kernel, so the shell dies with ``NotImplementedError: aten::equal`` one line before
``_reject_tied_embeddings`` gets to say what is actually wrong. The loader therefore builds the shell
with the tie suppressed and writes the declared flag back, and this pins that the user sees the
repo's own diagnosis.

The other half matters as much: the gate deliberately PERMITS a declared tie on a model with no
output embedding, which is what lets reward / sequence-classification heads run under PP. A
config-level pre-check would pass the test above and silently break that, so it is pinned here too.

    python tests/cpu/parallelism/test_pp_tied_embeddings.py
"""

import json
import os

import pytest
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, Qwen3Config, Qwen3ForCausalLM

from src.distributed.pipeline_parallel.lazy_loader import load_pp_stage_model
from src.distributed.pipeline_parallel.split import tie_flag_configs, validate_model_structure_supports_pp
from tests.common.models import TINY_QWEN3_CONFIG

PP_SIZE = 2


@pytest.fixture(scope="module")
def tied_checkpoint(tmp_path_factory) -> str:
    """A checkpoint declaring ``tie_word_embeddings`` while carrying BOTH tied keys on disk.

    Written untied (``save_pretrained`` drops a tied target) and then re-declared, which is exactly
    the on-disk shape of the tied Qwen3 exports — and the shape that triggers the ``torch.equal``
    branch, since HF compares the two only when neither key is missing.
    """
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
    return load_pp_stage_model(
        path,
        pp_rank,
        PP_SIZE,
        config=AutoConfig.from_pretrained(path, trust_remote_code=False),
        trust_remote_code=False,
        dtype=torch.bfloat16,
        **kwargs,
    )


def test_tied_causal_lm_is_rejected_with_the_pipeline_diagnosis(tied_checkpoint):
    """The failure a user sees must name the tie and PP, not an aten op with no meta kernel."""
    with pytest.raises(ValueError) as excinfo:
        _load(tied_checkpoint)

    message = str(excinfo.value)
    assert "tie_word_embeddings" in message
    assert "pipeline parallelism" in message
    assert "aten::equal" not in message


def test_declared_tie_without_an_output_head_still_loads(tied_checkpoint):
    """The carve-out PP depends on: a reward / classification head declares the tie it inherited from
    its base config but has no output embedding to tie, so it must load. Any fix that rejects on the
    config flag alone would take sequence classification and reward modelling off PP entirely."""
    stage = _load(tied_checkpoint, model_class=AutoModelForSequenceClassification)

    assert stage.get_output_embeddings() is None
    assert not stage.state_dict()["score.weight"].is_meta


def test_the_declared_tie_flag_survives_the_untied_meta_build(tied_checkpoint):
    """Suppression is a build-time device, not a config edit: the flag reaches the stage's own config
    unchanged, because that config is what every stage saves."""
    stage = _load(tied_checkpoint, model_class=AutoModelForSequenceClassification)

    assert stage.config.tie_word_embeddings is True


class _CompositeConfig(Qwen3Config):
    """Stand-in for a VLM wrapper: a config whose decoder lives on a ``text_config`` sub-config."""

    sub_configs = {"text_config": Qwen3Config}


def _composite_declaring_the_tie_on_text_only():
    """A wrapper config with ``tie_word_embeddings`` False at the top and True on ``text_config``."""
    text = Qwen3Config(**{**TINY_QWEN3_CONFIG, "tie_word_embeddings": True})
    composite = _CompositeConfig(**{**TINY_QWEN3_CONFIG, "tie_word_embeddings": False})
    composite.text_config = text
    return composite


def test_tie_flag_is_collected_from_every_config_level():
    """A composite (VLM) config declares the tie on its text sub-config too, and transformers ties
    across every submodel — suppressing only the top level would leave the tie live and the meta
    build would still die on it. The paths are what lets the flags be put back on the deep copy
    ``from_pretrained`` hands the model."""
    text = Qwen3Config(**{**TINY_QWEN3_CONFIG, "tie_word_embeddings": True})
    assert {path: id(h) for path, h in tie_flag_configs(text).items()} == {"": id(text)}

    composite = _composite_declaring_the_tie_on_text_only()
    assert {path: id(h) for path, h in tie_flag_configs(composite).items()} == {
        "": id(composite),
        "text_config.": id(composite.text_config),
    }


def test_a_tie_declared_only_on_a_sub_config_is_rejected():
    """The gate must walk the same config levels the suppression does.

    The stage-aware loader builds its meta shell with every declared flag suppressed, so shared
    storage is False by construction and the DECLARATION is the only surviving signal. Reading it at
    the top level alone lets a composite that declares the tie on ``text_config`` — where
    transformers resolves it — through the gate: stage 0's embedding and the last stage's head are
    then one parameter on different ranks, each taking only its own path's gradient.
    """
    model = Qwen3ForCausalLM(Qwen3Config(**{**TINY_QWEN3_CONFIG, "tie_word_embeddings": False}))
    assert model.lm_head.weight is not model.model.embed_tokens.weight, "untied storage is the premise"
    model.config = _composite_declaring_the_tie_on_text_only()

    with pytest.raises(ValueError, match="tie_word_embeddings"):
        validate_model_structure_supports_pp(model)


def test_a_sub_config_tie_without_an_output_head_still_loads():
    """The sub-config walk must not cost the reward / classification carve-out: a declared tie with
    no output embedding ties nothing, at any config level."""
    model = AutoModelForSequenceClassification.from_config(
        Qwen3Config(**{**TINY_QWEN3_CONFIG, "tie_word_embeddings": False, "num_labels": 2})
    )
    model.config = _composite_declaring_the_tie_on_text_only()

    assert model.get_output_embeddings() is None
    validate_model_structure_supports_pp(model)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
