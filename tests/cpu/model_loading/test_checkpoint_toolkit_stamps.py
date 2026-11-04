"""Toolkit stamps another writer serialized into ``config.json`` are dropped at each load, so none reads as this run's
state. A config the caller hands the load is the run's and keeps its stamps."""

import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import AutoConfig

from halo.configs.embedding import EmbeddingConfig
from halo.configs.script_arguments import ModelArguments
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.checkpoint_coverage import from_pretrained_verified
from halo.model_loading.liger.dispatch import LIGER_APPLIED_CONFIG_ATTR, LIGER_CP_FUSED_LOSS_ATTR
from halo.model_loading.model_preparation import resolve_auto_model_class
from halo.model_loading.parallel import frozen_models, loader
from halo.model_loading.parallel.whole_model import resolve_trainer_model
from halo.models.attention.gpt_oss_sinks import RESOLVED_SINKS_RESET_ATTR, SinksPolicy
from halo.models.config_access import TOOLKIT_STAMP_PREFIX, config_export_ready, config_sources
from halo.models.moe.router_logits import (
    ROUTER_LOGITS_FORCED_OFF_ATTR,
    mark_router_logits_forced_off,
    router_logits_forced_off,
)
from scripts.training.embedding import build_sentence_transformer
from tests.common.tiny_models import (
    TINY_DENSE_FAMILY,
    TINY_MOE_FAMILIES,
    TINY_QWEN3_VL_FAMILY,
    build_tiny_family_checkpoint,
    word_level_tokenizer,
)

# Each at the value that would steer a run: fused loss armed, FLCE warning raised, SDPA allowed on live sinks, the MoE
# metrics callback unwired.
CHECKPOINT_STAMPS = {
    LIGER_APPLIED_CONFIG_ATTR: {"fused_linear_cross_entropy": True},
    LIGER_CP_FUSED_LOSS_ATTR: True,
    RESOLVED_SINKS_RESET_ATTR: True,
    ROUTER_LOGITS_FORCED_OFF_ATTR: True,
}
FAMILIES = {"qwen3": TINY_DENSE_FAMILY, "qwen3_vl": TINY_QWEN3_VL_FAMILY, "gpt_oss": TINY_MOE_FAMILIES["gpt_oss"]}
# sentence-transformers also loads a multimodal checkpoint's image processor, which the tiny composite does not ship.
EMBEDDING_FAMILIES = ("gpt_oss", "qwen3")


def _stamps(config) -> list[str]:
    return sorted(
        f"{type(level).__name__}.{key}"
        for level in config_sources(config)
        for key in vars(level)
        if key.startswith(TOOLKIT_STAMP_PREFIX) or key in CHECKPOINT_STAMPS
    )


def _stamped_checkpoint(tmp_path, family: str) -> str:
    build_tiny_family_checkpoint(FAMILIES[family], str(tmp_path))
    word_level_tokenizer(("hello",)).save_pretrained(tmp_path)
    config_path = tmp_path / "config.json"
    payload = json.loads(config_path.read_text())
    payload |= CHECKPOINT_STAMPS
    if isinstance(payload.get("text_config"), dict):
        payload["text_config"] |= CHECKPOINT_STAMPS
    config_path.write_text(json.dumps(payload))
    # transformers reads every stamp back as a live attribute on every level.
    read_back = AutoConfig.from_pretrained(tmp_path)
    assert all(
        vars(level).get(key) == value
        for level in config_sources(read_back)
        for key, value in CHECKPOINT_STAMPS.items()
    )
    return str(tmp_path)


@pytest.fixture(params=sorted(FAMILIES))
def stamped(request, tmp_path):
    return request.param, _stamped_checkpoint(tmp_path, request.param)


def test_the_policy_config_drops_every_checkpoint_stamp(stamped):
    _, path = stamped
    _, model_config, _, _ = loader._resolve_run_config(
        path,
        None,
        ParallelismConfig(),
        torch.bfloat16,
        trust_remote_code=False,
        model_config_overrides=None,
        sinks_policy=SinksPolicy.NEUTRALIZED,
        quantization_config=None,
        init_from_scratch=False,
        model_class=None,
        text_only_model=False,
    )
    assert _stamps(model_config) == []
    assert not router_logits_forced_off(model_config)


def test_a_frozen_model_drops_every_checkpoint_stamp(stamped):
    """The model carries only the sinks verdict this load resolved (live, against the checkpoint's reset)."""
    family, path = stamped
    resolved_against = []
    resolve = frozen_models.resolve_attn_implementation

    def recording_resolve(model_config, *args, **kwargs):
        resolved_against.append(_stamps(model_config))
        return resolve(model_config, *args, **kwargs)

    with patch.object(frozen_models, "resolve_attn_implementation", recording_resolve):
        model = frozen_models.load_frozen_auxiliary_model(
            path,
            dtype=torch.float32,
            trust_remote_code=False,
            attn_implementation="eager",
            reset_sinks=False,
            is_vlm=family == "qwen3_vl",
            download_tag=None,
        )
    assert resolved_against == [[]]
    levels = config_sources(model.config)
    assert _stamps(model.config) == sorted(f"{type(level).__name__}.{RESOLVED_SINKS_RESET_ATTR}" for level in levels)
    assert all(vars(level)[RESOLVED_SINKS_RESET_ATTR] is False for level in levels)


def test_a_trainer_model_path_drops_every_checkpoint_stamp(stamped):
    """A trainer constructor handed a path loads it itself."""
    _, path = stamped
    model = resolve_trainer_model(path, None, parallelism_config=None)
    assert _stamps(model.config) == []
    assert not router_logits_forced_off(model.config)


def test_a_config_the_caller_passes_keeps_its_stamps(stamped):
    """The policy loader stamps the config before the load."""
    _, path = stamped
    config = AutoConfig.from_pretrained(path)
    passed = _stamps(config)
    model = from_pretrained_verified(resolve_auto_model_class(config), path, config=config)
    assert passed and _stamps(model.config) == passed


@pytest.mark.parametrize("family", EMBEDDING_FAMILIES)
def test_the_embedding_backbone_drops_every_checkpoint_stamp(family, tmp_path):
    """sentence-transformers loads the backbone itself."""
    path = _stamped_checkpoint(tmp_path / "checkpoint", family)
    runtime = SimpleNamespace(
        parallelism_config=SimpleNamespace(
            has_ep_group=False,
            is_tp_mode=False,
            needs_ep_layers=False,
            fp32_non_ep_params=False,
            max_concurrent_loading=None,
        ),
        model_source=path,
    )
    embedding_config = EmbeddingConfig(
        output_dir=str(tmp_path / "run"), bf16=False, pooling_mode="mean", normalize_embeddings=False, max_length=32
    )
    st_model = build_sentence_transformer(
        runtime,
        embedding_config,
        ModelArguments(model_name_or_path=path),
        SimpleNamespace(reset_sinks=True, train_sinks=False),
    )
    backbone = st_model[0].auto_model
    assert _stamps(backbone.config) == []
    assert not router_logits_forced_off(backbone.config)


def test_the_forced_off_mark_never_exports_from_a_copied_config():
    """The deep copy holds no recorded original for a mark written before it, so the export lifts it by prefix."""
    config = AutoConfig.for_model("qwen3")
    mark_router_logits_forced_off(config)
    copied = copy.deepcopy(config)
    with config_export_ready(copied):
        exported = copied.to_dict()
    assert [key for key in exported if "router_logits_forced_off" in key] == []
    assert router_logits_forced_off(copied)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
