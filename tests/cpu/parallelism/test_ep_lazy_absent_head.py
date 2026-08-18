"""CPU tests for the EP lazy loader against a checkpoint that does not carry the task head.

``AutoModelForSequenceClassification`` (reward / classification) over a base CausalLM checkpoint has
no ``score`` tensor on disk. The lazy loader bypasses ``from_pretrained``, so nothing else initializes
it: left uninitialized it reaches the trainer still on the meta device, where the mixin's
device-placement sweep replaces it with ``torch.empty`` — uninitialized memory, DIFFERENT on every
rank, behind one warning.
``examples/classification/gptoss/clf-gptoss-20b-mage-ep.yaml`` is exactly this shape.

These run the real :func:`~src.distributed.expert_parallel.lazy_loader.load_ep_model_lazy` against a
tiny fused-expert checkpoint written to disk (EP patching stubbed — it needs DeepEP and process
groups), which is the only way to catch that the head is initialized AT ALL, at the RUN's dtype, and
from the RNG rather than from the allocator.

    python tests/cpu/parallelism/test_ep_lazy_absent_head.py
"""

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

import src.distributed.expert_parallel.lazy_loader as ep_lazy_loader
from src.distributed.expert_parallel.config import EPConfig
from src.distributed.expert_parallel.lazy_loader import init_checkpoint_absent_modules, load_ep_model_lazy
from src.trainers.mixins.base import DistributedTrainerMixin
from tests.common.models import TINY_GPTOSS_CONFIG

SEED = 1234
INITIALIZER_RANGE = 0.02


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> str:
    """A tiny bf16 gpt-oss checkpoint: fused 3D experts, no ``score`` tensor."""
    torch.manual_seed(0)
    config = GptOssConfig(**TINY_GPTOSS_CONFIG, initializer_range=INITIALIZER_RANGE)
    model = GptOssForCausalLM(config).to(torch.bfloat16)
    path = tmp_path_factory.mktemp("ep_ckpt")
    model.save_pretrained(path, safe_serialization=True)
    return str(path)


@pytest.fixture(autouse=True)
def _stub_ep_patching(monkeypatch):
    """EP patching and buffer creation need DeepEP + process groups; the load itself does not."""
    monkeypatch.setattr(ep_lazy_loader, "patch_moe_model_for_ep", lambda model, *a, **k: model)
    monkeypatch.setattr(ep_lazy_loader, "create_ep_buffers", lambda *a, **k: None)


def _load_seq_cls(path: str, run_dtype: torch.dtype, seed: int = SEED):
    torch.manual_seed(seed)  # the run seeds identically on every rank before the load
    return load_ep_model_lazy(
        path,
        EPConfig(ep_size=1, world_size=1, gpus_per_node=1),
        AutoConfig.from_pretrained(path, trust_remote_code=False),
        dtype=run_dtype,
        trust_remote_code=False,
        model_class=AutoModelForSequenceClassification,
    )


@pytest.mark.parametrize("run_dtype", [torch.bfloat16, torch.float32])
def test_checkpoint_absent_head_is_init_weights_initialized(checkpoint, run_dtype):
    """``score`` must hold a real ``_init_weights`` draw at the RUN's dtype.

    The std gate is what separates a real draw from the ``torch.empty`` backstop: gpt-oss
    initializes an ``nn.Linear`` from ``N(0, initializer_range)``, while uninitialized memory is
    either an all-zero fresh page or arbitrary values orders of magnitude away.
    """
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
    """Ranks agree only because the values come from the identically seeded RNG.

    Same seed → bit-identical (what ranks do); different seed → different (what proves the values are
    a draw and not whatever the allocator handed back, which no seed would change).
    """
    same = [_load_seq_cls(checkpoint, torch.float32).state_dict()["score.weight"].clone() for _ in range(2)]
    other = _load_seq_cls(checkpoint, torch.float32, seed=SEED + 1).state_dict()["score.weight"].clone()

    assert torch.equal(same[0], same[1])
    assert not torch.equal(same[0], other)


def test_no_parameter_is_left_on_meta(checkpoint):
    """The trainer's meta sweep must have nothing left to do for this shape."""
    model = _load_seq_cls(checkpoint, torch.bfloat16)
    assert [name for name, param in model.named_parameters() if param.is_meta] == []


def test_device_placement_rejects_a_meta_parameter():
    """The mixin backstop is a hard stop: a parameter that never loaded must not be silently
    replaced with uninitialized memory that differs across ranks."""
    model = nn.Linear(4, 4)
    model.weight = nn.Parameter(torch.empty(4, 4, device="meta"))

    with pytest.raises(RuntimeError, match="still on the meta device"):
        DistributedTrainerMixin._move_model_to_device(None, model, "cpu")


@pytest.mark.parametrize("tied,expected", [(True, []), (False, ["lm_head.weight"])])
def test_a_tied_shadow_head_is_not_drawn_again(tied, expected):
    """``lm_head.weight`` is missing from a tied checkpoint but shares the embedding tensor on the
    shell and is restored by the post-load ``tie_weights()``. Treating it as absent would draw a
    second vocab-sized tensor — gigabytes on a production vocab — only to throw it away. Untied, the
    very same key IS absent and must be initialized."""
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
    """A module with some tensors on disk and some missing is ambiguous: ``_init_weights`` would
    overwrite the loaded ones, and skipping it would strand the rest on meta."""
    model = nn.Sequential(nn.Linear(4, 4))
    model._init_weights = lambda module: None
    planned = {"0.weight"}

    with pytest.raises(RuntimeError, match="PARTIALLY absent"):
        init_checkpoint_absent_modules(model, planned, "cpu", "unit")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
