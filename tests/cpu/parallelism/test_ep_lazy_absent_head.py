"""The EP lazy loader on a checkpoint without the task head. Left on meta, the device sweep replaces it with
``torch.empty``, different on every rank; EP patching is stubbed."""

import pytest
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    GptOssConfig,
    GptOssForCausalLM,
    Qwen3Config,
)

import halo.model_loading.parallel.ep as ep_lazy_loader
from halo.distributed.expert_parallel.config import EPConfig
from halo.model_loading.lazy_safetensors.materialization import init_checkpoint_absent_modules
from halo.model_loading.parallel.ep import load_ep_model_lazy
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from tests.common.models import TINY_GPTOSS_CONFIG

SEED = 1234
INITIALIZER_RANGE = 0.02


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> str:
    """Fused 3D experts, no ``score`` tensor."""
    torch.manual_seed(0)
    config = GptOssConfig(**TINY_GPTOSS_CONFIG, initializer_range=INITIALIZER_RANGE)
    model = GptOssForCausalLM(config).to(torch.bfloat16)
    path = tmp_path_factory.mktemp("ep_ckpt")
    model.save_pretrained(path, safe_serialization=True)
    return str(path)


@pytest.fixture(autouse=True)
def _stub_ep_patching(monkeypatch):
    """EP patching needs DeepEP + process groups; the load itself does not."""
    monkeypatch.setattr(ep_lazy_loader, "apply_ep_layers", lambda model, *a, **k: model)


def _load_seq_cls(path: str, run_dtype: torch.dtype, seed: int = SEED):
    torch.manual_seed(seed)  # The run seeds identically on every rank before the load.
    return load_ep_model_lazy(
        path,
        EPConfig(ep_size=1, stage_world_size=1, nvlink_domain_size=1, experts_fsdp_managed=True),
        AutoConfig.from_pretrained(path, trust_remote_code=False),
        dtype=run_dtype,
        trust_remote_code=False,
        model_class=AutoModelForSequenceClassification,
    )


@pytest.mark.parametrize("run_dtype", [torch.bfloat16, torch.float32])
def test_checkpoint_absent_head_is_init_weights_initialized(checkpoint, run_dtype):
    """The std gate separates ``N(0, initializer_range)`` from uninitialized memory, a zero page or values orders of
    magnitude away."""
    state = _load_seq_cls(checkpoint, run_dtype).state_dict()
    score = state["score.weight"]

    assert not score.is_meta
    assert score.dtype == run_dtype
    assert score.dtype == state["model.layers.0.self_attn.q_proj.weight"].dtype
    assert torch.isfinite(score.float()).all()
    std = score.float().std().item()
    assert 0.5 * INITIALIZER_RANGE < std < 2.0 * INITIALIZER_RANGE, f"score std {std} is not an N(0, 0.02) draw"
    assert abs(score.float().mean().item()) < INITIALIZER_RANGE


def test_absent_head_comes_from_the_rng_so_every_rank_draws_the_same(checkpoint):
    """A different seed proves a draw rather than what the allocator handed back."""
    same = [_load_seq_cls(checkpoint, torch.float32).state_dict()["score.weight"].clone() for _ in range(2)]
    other = _load_seq_cls(checkpoint, torch.float32, seed=SEED + 1).state_dict()["score.weight"].clone()

    assert torch.equal(same[0], same[1])
    assert not torch.equal(same[0], other)


def test_no_parameter_is_left_on_meta(checkpoint):
    model = _load_seq_cls(checkpoint, torch.bfloat16)
    assert [name for name, param in model.named_parameters() if param.is_meta] == []


def test_device_placement_rejects_a_meta_parameter():
    """It must not be silently replaced with memory that differs across ranks."""
    model = nn.Linear(4, 4)
    model.weight = nn.Parameter(torch.empty(4, 4, device="meta"))

    with pytest.raises(RuntimeError, match="still on the meta device"):
        DistributedTrainerMixin._move_model_to_device(None, model, "cpu")


@pytest.mark.parametrize("tied,expected", [(True, []), (False, ["lm_head.weight"])])
def test_a_tied_shadow_head_is_not_drawn_again(tied, expected):
    """A second vocab-sized draw would cost gigabytes only to be thrown away by ``tie_weights()``; untied, the key
    is absent."""
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        tie_word_embeddings=tied,
    )
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    planned = set(model.state_dict()) - {"lm_head.weight"}

    assert init_checkpoint_absent_modules(model, planned, "cpu", "unit", dtype=torch.float32) == expected


def test_partially_absent_module_raises():
    """``_init_weights`` would overwrite the loaded tensors; skipping strands the rest on meta."""
    model = nn.Sequential(nn.Linear(4, 4))
    model._init_weights = lambda module: None
    planned = {"0.weight"}

    with pytest.raises(RuntimeError, match="PARTIALLY absent"):
        init_checkpoint_absent_modules(model, planned, "cpu", "unit")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
