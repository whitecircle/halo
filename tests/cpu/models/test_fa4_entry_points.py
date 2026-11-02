#!/usr/bin/env python
"""The wrappers on FA4's ``flash_attn.cute`` entry points and the family patches every FA4 load takes.

The trainable-sink rescale and the varlen softcap refusal share one install seam (``install_fa4_entry_point_wrapper``) and sit
in front of FA4 calls process-wide, and transformers decides which optional kwargs (sliding window, softcap, the
gpt-oss sink) it forwards at all from the varlen entry point's signature: a wrapper that hid the signature would drop
them on every FA4 model, silently. So the signature through both wrappers, the install seam and the rescale's lse
layouts are held here, and so are the loads outside the policy and frozen loaders reaching the family patches.

Run: python tests/cpu/models/test_fa4_entry_points.py  (or pytest)
"""

import collections
import inspect
import sys
import types

import pytest
import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig, Qwen3Config
from transformers import modeling_flash_attention_utils as flash_utils

import scripts.training.embedding as embedding_script
from halo.configs.embedding import EmbeddingConfig
from halo.configs.script_arguments import ModelArguments
from halo.entrypoints import reward_scoring
from halo.model_loading import model_preparation
from halo.model_loading.parallel import whole_model
from halo.models.attention import fa4_entry_points
from halo.models.attention.backend_setup import install_fa4_varlen_softcap_refusal
from halo.models.attention.gpt_oss_sinks import install_fa4_trainable_sink_rescale
from tests.common.models import TINY_QWEN35_CONFIG
from tests.common.tiny_models import word_level_tokenizer

FA4 = "flash_attention_4"

# The optional kwargs transformers forwards only when the kernel's signature names them, by flash-API name.
_SIGNATURE_GATED_KWARGS = ("window_size", "softcap", "learnable_sink")

# The process-wide fix every FA4 build needs, by the name ``apply_family_patches`` calls it under.
_FA4_LOAD_PATCH = "patch_transformers_flash_varlen_int_seqlen"


def _supports_mapping(flash_fn) -> dict[str, bool]:
    """The per-kwarg capability map transformers builds off a varlen entry point."""
    return flash_utils._lazy_define_process_function(flash_fn).keywords["supports_mapping"]


class _RecordingKernel:
    """An FA4 entry point stand-in carrying the real kernel's signature, recording what reaches it."""

    def __init__(self, signature: inspect.Signature):
        self.__signature__ = signature
        self.__name__ = "flash_attn_varlen_func"
        self.calls: list[tuple[tuple, dict]] = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return args[0]


class _FakeCute(types.ModuleType):
    """``flash_attn.cute`` with recording entry points carrying the installed build's signatures."""

    def __init__(self, real):
        super().__init__("flash_attn.cute")
        self.flash_attn_func = _RecordingKernel(inspect.signature(real.flash_attn_func))
        self.flash_attn_varlen_func = _RecordingKernel(inspect.signature(real.flash_attn_varlen_func))


@pytest.fixture
def fake_cute(monkeypatch):
    cute = _FakeCute(pytest.importorskip("flash_attn.cute"))
    monkeypatch.setitem(sys.modules, "flash_attn.cute", cute)
    monkeypatch.setattr(sys.modules["flash_attn"], "cute", cute, raising=False)
    monkeypatch.setattr(fa4_entry_points, "ensure_fa4_kernel_cache_env", lambda: None)
    # transformers' cached references, as a model load that ran before the install leaves them.
    monkeypatch.setattr(flash_utils, "_flash_fn", cute.flash_attn_func)
    monkeypatch.setattr(flash_utils, "_flash_varlen_fn", cute.flash_attn_varlen_func)
    return cute


def test_the_install_wraps_both_entry_points_and_transformers_cached_references(fake_cute):
    """Both entry points and the references transformers already cached take the wrapper, once."""
    originals = fake_cute.flash_attn_func, fake_cute.flash_attn_varlen_func
    assert install_fa4_trainable_sink_rescale()
    assert install_fa4_trainable_sink_rescale()
    wrapped = fake_cute.flash_attn_func, fake_cute.flash_attn_varlen_func
    assert tuple(fn.__wrapped__ for fn in wrapped) == originals, "a second install must not wrap the wrapper"
    assert (flash_utils._flash_fn, flash_utils._flash_varlen_fn) == wrapped


@pytest.mark.parametrize("rescale_first", [True, False], ids=["sink_rescale_first", "softcap_refusal_first"])
def test_the_two_fa4_installs_stack_once_each_in_either_order(fake_cute, rescale_first):
    """The sink rescale and the softcap refusal share the install seam: in either order each wraps the entry points
    it covers exactly once (the refusal leaves the dense one alone), repeated installs add nothing, transformers'
    references follow the outermost wrapper, and the stack keeps the kernel's signature, so transformers still
    forwards the window, the softcap and the sink."""
    dense, varlen = fake_cute.flash_attn_func, fake_cute.flash_attn_varlen_func
    installs = [install_fa4_trainable_sink_rescale, install_fa4_varlen_softcap_refusal]
    if not rescale_first:
        installs.reverse()
    for install in installs + installs:
        assert install()
    assert fake_cute.flash_attn_func.__wrapped__ is dense
    assert fake_cute.flash_attn_varlen_func.__wrapped__.__wrapped__ is varlen
    assert (flash_utils._flash_fn, flash_utils._flash_varlen_fn) == (
        fake_cute.flash_attn_func,
        fake_cute.flash_attn_varlen_func,
    )
    stacked_map = _supports_mapping(fake_cute.flash_attn_varlen_func)
    assert stacked_map == _supports_mapping(varlen)
    assert all(stacked_map[name] for name in _SIGNATURE_GATED_KWARGS), stacked_map


class _LseKernel(_RecordingKernel):
    """An entry point stand-in returning a fixed ``(out, lse)`` pair."""

    def __init__(self, signature: inspect.Signature, out: torch.Tensor, lse: torch.Tensor):
        super().__init__(signature)
        self.result = out, lse

    def __call__(self, *args, **kwargs):
        super().__call__(*args, **kwargs)
        return self.result


@pytest.mark.parametrize(
    ("name", "out_shape", "lse_shape", "head_axis"),
    [
        ("flash_attn_varlen_func", (5, 3, 4), (3, 5), 0),
        ("flash_attn_func", (1, 5, 3, 4), (1, 3, 5), 1),
    ],
    ids=["varlen", "dense"],
)
def test_the_sink_rescale_reads_each_entry_points_lse_layout(fake_cute, name, out_shape, lse_shape, head_axis):
    """Through the shared seam the rescale must still gate by the right head axis per entry point (lse is
    ``[heads, tokens]`` beside a varlen out, ``[batch, heads, seq]`` beside a dense one) and call the kernel
    sink-less with ``return_lse``; a swapped axis gates every head by another head's sink."""
    real = getattr(fake_cute, name)
    out, lse = torch.randn(out_shape), torch.randn(lse_shape)
    kernel = _LseKernel(real.__signature__, out, lse)
    setattr(fake_cute, name, kernel)
    assert install_fa4_trainable_sink_rescale()
    sink = torch.randn(3, requires_grad=True)
    q = k = v = torch.randn(out_shape)
    gated, sinked_lse = getattr(fake_cute, name)(q, k, v, learnable_sink=sink)
    ((_, kwargs),) = kernel.calls
    assert kwargs == {"learnable_sink": None, "return_lse": True}
    per_head = sink.detach().reshape([-1 if d == head_axis else 1 for d in range(lse.dim())])
    gate = torch.sigmoid(lse - per_head).movedim(head_axis, -1).unsqueeze(-1)
    torch.testing.assert_close(gated.detach(), out * gate, rtol=0, atol=0)
    torch.testing.assert_close(sinked_lse.detach(), torch.logaddexp(lse, per_head), rtol=0, atol=0)


@pytest.mark.parametrize(
    ("attn_implementation", "installs"),
    [("flash_attention_4", True), ("flash_attention_2", False), ("sdpa", False)],
)
def test_the_family_patches_install_it_for_every_fa4_load(monkeypatch, attn_implementation, installs):
    """``apply_family_patches`` runs before every policy and frozen-model build, so an FA4 load always takes the
    int-seqlen fix before its first varlen backward; other backends never do."""
    calls = []
    monkeypatch.setattr(model_preparation, _FA4_LOAD_PATCH, lambda: calls.append(1))
    model_preparation.apply_family_patches(Qwen3Config(), attn_implementation)
    assert bool(calls) is installs


class _Built(Exception):
    """Stops a load at its weight build: transformers refuses an FA4 build on a CPU device."""


@pytest.fixture(scope="module")
def qwen3_5_text_checkpoint(tmp_path_factory) -> str:
    """A tiny ``qwen3_5_text`` checkpoint (head_dim 256, partial rotary)."""
    path = tmp_path_factory.mktemp("qwen3_5_text")
    Qwen3_5ForCausalLM(Qwen3_5TextConfig(**TINY_QWEN35_CONFIG)).save_pretrained(path)
    word_level_tokenizer().save_pretrained(path)
    return str(path)


def _load_reward_model(checkpoint, tmp_path, monkeypatch, build) -> None:
    monkeypatch.setattr(reward_scoring, "from_pretrained_verified", lambda cls, path, **kwargs: build(kwargs))
    reward_scoring.load_reward_model(checkpoint, FA4, 64, "cuda:0", "bfloat16", trust_remote_code=False)


def _load_sentence_transformer(checkpoint, tmp_path, monkeypatch, build) -> None:
    monkeypatch.setattr(embedding_script, "SentenceTransformer", lambda path, **kwargs: build(kwargs["model_kwargs"]))
    runtime = types.SimpleNamespace(
        parallelism_config=types.SimpleNamespace(has_ep_group=False, is_tp_mode=False, max_concurrent_loading=None),
        model_source=checkpoint,
    )
    embedding_script.build_sentence_transformer(
        runtime,
        EmbeddingConfig(output_dir=str(tmp_path), bf16=False, use_cpu=True),
        ModelArguments(model_name_or_path=checkpoint, attn_implementation=FA4),
        types.SimpleNamespace(reset_sinks=True, train_sinks=False),
    )


def _load_trainer_model(checkpoint, tmp_path, monkeypatch, build) -> None:
    monkeypatch.setattr(model_preparation, "from_pretrained_verified", lambda cls, path, **kwargs: build(kwargs))
    args = types.SimpleNamespace(model_init_kwargs={"attn_implementation": FA4}, gradient_checkpointing=False)
    whole_model.resolve_trainer_model(checkpoint, args, parallelism_config=None)


@pytest.mark.parametrize(
    "load",
    [_load_reward_model, _load_sentence_transformer, _load_trainer_model],
    ids=["reward_scoring", "embedding", "trainer_model_id"],
)
def test_every_fa4_load_takes_the_family_patches_before_its_build(
    monkeypatch, tmp_path, qwen3_5_text_checkpoint, load
):
    """The loads outside the policy and frozen loaders build their model themselves, so each must pass the family
    patches before its build, as those two do."""
    applied = collections.Counter()
    monkeypatch.setattr(model_preparation, _FA4_LOAD_PATCH, lambda: applied.update([_FA4_LOAD_PATCH]))
    builds = []

    def build(kwargs):
        builds.append((kwargs["attn_implementation"], dict(applied)))
        raise _Built

    with pytest.raises(_Built):
        load(qwen3_5_text_checkpoint, tmp_path, monkeypatch, build)
    assert builds == [(FA4, {_FA4_LOAD_PATCH: 1})]


def test_an_unpinned_embedding_load_takes_the_backend_independent_family_patches(
    monkeypatch, tmp_path, qwen3_5_text_checkpoint
):
    """Unpinned, sentence-transformers picks the backend at its own build, yet the patches no backend keys (Zaya's
    checkpointing refusal and document isolation) still apply, as on the policy, frozen and trainer-path loads."""
    patched = []
    monkeypatch.setattr(
        embedding_script, "apply_family_patches", lambda config, attn: patched.append((config.model_type, attn))
    )
    builds = []

    def build(kwargs):
        builds.append(kwargs)
        raise _Built

    monkeypatch.setattr(embedding_script, "SentenceTransformer", lambda path, **kwargs: build(kwargs["model_kwargs"]))
    runtime = types.SimpleNamespace(
        parallelism_config=types.SimpleNamespace(has_ep_group=False, is_tp_mode=False, max_concurrent_loading=None),
        model_source=qwen3_5_text_checkpoint,
    )
    with pytest.raises(_Built):
        embedding_script.build_sentence_transformer(
            runtime,
            EmbeddingConfig(output_dir=str(tmp_path), bf16=False, use_cpu=True),
            ModelArguments(model_name_or_path=qwen3_5_text_checkpoint),
            types.SimpleNamespace(reset_sinks=True, train_sinks=False),
        )
    assert patched == [("qwen3_5_text", None)]
    assert "attn_implementation" not in builds[0], "an unpinned load left the backend to sentence-transformers"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
