"""CPU tests for how the EP/PP lazy loaders build their meta-device model shell.

``instantiate_on_meta`` is the zero-byte shell both lazy loaders plan against. Two properties are
load-bearing and neither is visible from a passing training run:

  * the shell carries the RUN's dtype, not the checkpoint config's — every tensor the loader
    materializes is cast to the run's, so anything the checkpoint does NOT carry (a task head) would
    otherwise be left alone in the wrong dtype;
  * the fallback for architectures ``device_map="meta"`` cannot place builds from the CONFIG alone.
    ``from_pretrained`` under a ``torch.device("meta")`` context is not that: the context only
    redirects tensor factories, so it still streams the whole checkpoint into HOST RAM — an
    unconditional host OOM on a production MoE, reached silently.

    python tests/cpu/parallelism/test_ep_lazy_meta_shell.py
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, GptOssConfig, GptOssForCausalLM

from src.distributed.expert_parallel.lazy_loader import instantiate_on_meta
from src.models.loading.lazy_safetensors.meta_shell import _instantiate_from_config_on_meta
from tests.common.models import TINY_GPTOSS_CONFIG

NO_SUCH_CHECKPOINT = "/nonexistent/checkpoint-the-fallback-must-never-read"


@pytest.fixture
def config() -> GptOssConfig:
    """A checkpoint config declaring bf16 — the dtype the shell must NOT inherit blindly."""
    return GptOssConfig(**TINY_GPTOSS_CONFIG, dtype=torch.bfloat16)


@pytest.fixture
def checkpoint(tmp_path, config) -> str:
    torch.manual_seed(0)
    GptOssForCausalLM(config).to(torch.bfloat16).save_pretrained(tmp_path, safe_serialization=True)
    return str(tmp_path)


@pytest.mark.parametrize("run_dtype", [torch.bfloat16, torch.float32])
def test_shell_carries_the_runs_dtype(checkpoint, config, run_dtype):
    model = instantiate_on_meta(checkpoint, GptOssForCausalLM, config, dtype=run_dtype, trust_remote_code=False)
    params = dict(model.named_parameters())
    assert all(param.is_meta for param in params.values())
    assert params["model.embed_tokens.weight"].dtype == run_dtype


@pytest.mark.parametrize("model_class", [GptOssForCausalLM, AutoModelForCausalLM])
def test_config_fallback_builds_the_shell_without_touching_the_checkpoint(monkeypatch, config, model_class):
    """The path is proven by construction: ``model_name_or_path`` does not exist, so a fallback that
    read weights (a ``from_pretrained`` under a meta context) could not return at all."""

    def reject(*args, **kwargs):
        raise AttributeError("simulated device_map='meta' placement failure")

    monkeypatch.setattr(model_class, "from_pretrained", reject)
    model = instantiate_on_meta(NO_SUCH_CHECKPOINT, model_class, config, dtype=torch.float32, trust_remote_code=False)

    params = dict(model.named_parameters())
    assert all(param.is_meta for param in params.values())
    assert params["model.embed_tokens.weight"].dtype == torch.float32


def test_both_meta_paths_failing_raises_instead_of_loading_into_host_ram(monkeypatch, config):
    def reject(*args, **kwargs):
        raise AttributeError("simulated device_map='meta' placement failure")

    def reject_from_config(*args, **kwargs):
        raise TypeError("simulated config-only build failure")

    monkeypatch.setattr(GptOssForCausalLM, "from_pretrained", reject)
    monkeypatch.setattr(GptOssForCausalLM, "_from_config", reject_from_config)

    with pytest.raises(RuntimeError, match="ep_lazy_loading=False"):
        instantiate_on_meta(
            NO_SUCH_CHECKPOINT, GptOssForCausalLM, config, dtype=torch.float32, trust_remote_code=False
        )


class _CtorArgRotary(torch.nn.Module):
    """Mimics Qwen VL's vision rotary: a non-persistent buffer derived from ctor args it never
    stores, so nothing can recompute it after a meta build."""

    def __init__(self, dim: int):
        super().__init__()
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)


class _CtorBufferModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = torch.nn.Linear(4, 4, bias=False)
        self.rotary = _CtorArgRotary(8)

    @classmethod
    def _from_config(cls, config, **kwargs):
        return cls(config)


def test_config_only_shell_keeps_ctor_computed_buffers_real(config):
    """The config-only shell must build under ``init_empty_weights(include_buffers=False)``:
    parameters on meta (zero bytes for weights), buffers computed for REAL. A wholesale meta build
    strands ctor-derived non-persistent buffers (Qwen VL vision ``inv_freq``) with no recompute
    path, and the trainer's meta-buffer gate then rejects every shipped Qwen3.5/VL EP config."""
    shell = _instantiate_from_config_on_meta(_CtorBufferModel, config, torch.float32, False)

    assert shell.embed.weight.is_meta, "parameters must stay on meta (that is the zero-byte point)"
    assert not shell.rotary.inv_freq.is_meta, "ctor-computed buffer stranded on meta"
    expected = 1.0 / (10000.0 ** (torch.arange(0, 8, 2, dtype=torch.float) / 8))
    assert torch.allclose(shell.rotary.inv_freq, expected), "buffer materialized with wrong values"


class _MetaFromPretrained(_CtorBufferModel):
    """Mimics ``from_pretrained(device_map="meta")``: the whole tree — buffers included — on meta."""

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        with torch.device("meta"):
            return cls(kwargs.get("config"))


class _MetaWithPersistentBuffer(_MetaFromPretrained):
    """Adds a PERSISTENT buffer: the checkpoint carries it, so the graft must leave it alone."""

    def __init__(self, config):
        super().__init__(config)
        self.register_buffer("expert_bias", torch.zeros(4))


def test_primary_branch_repairs_buffers_the_meta_dispatch_stranded(config):
    """The PRIMARY ``from_pretrained(device_map="meta")`` branch — the one every ordinary EP lazy
    load takes — moves buffers to meta too. The shell must come back with its non-persistent
    buffers grafted from the config-only twin; grafting only in the fallback branch leaves every
    shipped Qwen3.5/VL EP config aborting at the trainer's meta-buffer gate."""
    shell = instantiate_on_meta(
        NO_SUCH_CHECKPOINT, _MetaFromPretrained, config, dtype=torch.float32, trust_remote_code=False
    )

    assert shell.embed.weight.is_meta, "parameters must stay on meta"
    assert not shell.rotary.inv_freq.is_meta, "primary-branch shell kept the stranded meta buffer"
    expected = 1.0 / (10000.0 ** (torch.arange(0, 8, 2, dtype=torch.float) / 8))
    assert torch.allclose(shell.rotary.inv_freq, expected), "grafted buffer has wrong values"


def test_the_graft_leaves_persistent_meta_buffers_for_the_loader(config):
    """A PERSISTENT meta buffer (Bailing ``expert_bias``, Zaya ``balancing_biases``) is carried by
    the checkpoint and materialized by the lazy loader like a parameter — grafting it from the
    config-only twin would overwrite the loaded value with ``__init__``'s default."""
    shell = instantiate_on_meta(
        NO_SUCH_CHECKPOINT, _MetaWithPersistentBuffer, config, dtype=torch.float32, trust_remote_code=False
    )

    assert shell.expert_bias.is_meta, "a checkpoint-carried buffer must stay meta for the loader"
    assert not shell.rotary.inv_freq.is_meta, "the non-persistent graft must still run alongside"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
