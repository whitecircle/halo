#!/usr/bin/env python
"""The FA4 head_dim-256 contiguity wrapper: it must change nothing but the strides of the calls it targets.

flash-attn-4 4.0.0b16's dedicated head_dim-256 forward misreads a head-major q/k, so
``install_fa4_contiguous_dedicated_head_inputs`` wraps both ``flash_attn.cute`` entry points to hand that kernel
contiguous inputs. The wrapper sits in front of every FA4 call in the process, and transformers decides which
optional kwargs (sliding window, softcap, the gpt-oss sink) it forwards at all from the varlen entry point's
signature: a wrapper that hid the signature would drop them on every FA4 model, silently. So the signature, the
kwargs, the strides and the install seam are each held here, the seam together with the trainable-sink rescale that
shares it (``wrap_fa4_entry_points``), and so are the loads outside the policy and frozen loaders reaching it.

Run: python tests/cpu/models/test_fa4_dedicated_head_contiguous.py  (or pytest)
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
from halo.models.attention.backend_setup import (
    fa4_contiguous_dedicated_head_inputs,
    install_fa4_contiguous_dedicated_head_inputs,
)
from halo.models.attention.gpt_oss_sinks import install_fa4_trainable_sink_rescale
from halo.models.attention.selection import FA4_DEDICATED_HEAD_DIM
from tests.common.models import TINY_QWEN35_CONFIG
from tests.common.tiny_models import word_level_tokenizer

FA4 = "flash_attention_4"

# The optional kwargs transformers forwards only when the kernel's signature names them, by flash-API name.
_SIGNATURE_GATED_KWARGS = ("window_size", "softcap", "learnable_sink")

# The process-wide fixes every FA4 build needs, by the name ``apply_family_patches`` calls them under.
_FA4_LOAD_PATCHES = ("install_fa4_contiguous_dedicated_head_inputs", "patch_transformers_flash_varlen_int_seqlen")


def _supports_mapping(flash_fn) -> dict[str, bool]:
    """The per-kwarg capability map transformers builds off a varlen entry point."""
    return flash_utils._lazy_define_process_function(flash_fn).keywords["supports_mapping"]


def _head_major(tokens: int, heads: int, head_dim: int) -> torch.Tensor:
    """A ``[tokens, heads, head_dim]`` view over head-major memory, as a partial-rotary ``torch.cat`` leaves it."""
    return torch.randn(heads, tokens, head_dim).transpose(0, 1)


class _RecordingKernel:
    """An FA4 entry point stand-in carrying the real kernel's signature, recording what reaches it."""

    def __init__(self, signature: inspect.Signature):
        self.__signature__ = signature
        self.__name__ = "flash_attn_varlen_func"
        self.calls: list[tuple[tuple, dict]] = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return args[0]


@pytest.fixture
def fa4_varlen():
    return pytest.importorskip("flash_attn.cute").flash_attn_varlen_func


def test_the_wrapper_keeps_the_kernels_signature(fa4_varlen):
    """Transformers reads ``inspect.signature`` of the varlen entry point; the wrapper must report the kernel's."""
    assert inspect.signature(fa4_contiguous_dedicated_head_inputs(fa4_varlen)) == inspect.signature(fa4_varlen)


def test_transformers_still_forwards_every_signature_gated_kwarg_through_the_wrapper(fa4_varlen):
    """The capability map built off the wrapper must equal the kernel's own, with the window, softcap and sink on."""
    wrapped_map = _supports_mapping(fa4_contiguous_dedicated_head_inputs(fa4_varlen))
    assert wrapped_map == _supports_mapping(fa4_varlen)
    assert all(wrapped_map[name] for name in _SIGNATURE_GATED_KWARGS), wrapped_map


def test_every_kwarg_transformers_builds_reaches_the_kernel(fa4_varlen):
    """End to end on the transformers side: the kwargs its processor builds for a windowed, soft-capped, sinked
    varlen call arrive at the kernel unchanged, object for object."""
    kernel = _RecordingKernel(inspect.signature(fa4_varlen))
    wrapped = fa4_contiguous_dedicated_head_inputs(kernel)
    q = k = v = torch.randn(8, 2, 64)
    sink = torch.randn(2)
    flash_kwargs = flash_utils._lazy_define_process_function(wrapped)(
        query_length=8,
        key_length=8,
        is_causal=True,
        sliding_window=4,
        softcap=30.0,
        s_aux=sink,
    )
    assert {"window_size", "softcap", "learnable_sink"} <= set(flash_kwargs), flash_kwargs
    extra = {"cu_seqlens_q": torch.tensor([0, 8], dtype=torch.int32), "max_seqlen_q": 8}
    wrapped(q, k, v, **flash_kwargs, **extra)
    ((args, kwargs),) = kernel.calls
    assert args == (q, k, v)
    assert kwargs.keys() == {**flash_kwargs, **extra}.keys()
    assert all(kwargs[name] is value for name, value in {**flash_kwargs, **extra}.items())


def test_a_head_major_dedicated_head_reaches_the_kernel_contiguous():
    """At head_dim 256 the kernel must see contiguous q/k/v holding the same values."""
    kernel = _RecordingKernel(inspect.Signature())
    q, k = _head_major(16, 4, FA4_DEDICATED_HEAD_DIM), _head_major(16, 2, FA4_DEDICATED_HEAD_DIM)
    v = torch.randn(16, 2, FA4_DEDICATED_HEAD_DIM)
    assert not q.is_contiguous() and not k.is_contiguous()
    fa4_contiguous_dedicated_head_inputs(kernel)(q, k, v, causal=True)
    ((args, kwargs),) = kernel.calls
    assert all(t.is_contiguous() for t in args)
    assert all(torch.equal(seen, sent) for seen, sent in zip(args, (q, k, v), strict=True))
    assert kwargs == {"causal": True}


@pytest.mark.parametrize("head_dim", [64, 128, 192])
def test_other_head_widths_pass_through_untouched(head_dim):
    """Anti-vacuity: every other width runs a kernel that reads strides, so its tensors pass by identity."""
    kernel = _RecordingKernel(inspect.Signature())
    q, k, v = _head_major(16, 4, head_dim), _head_major(16, 2, head_dim), _head_major(16, 2, head_dim)
    fa4_contiguous_dedicated_head_inputs(kernel)(q, k, v)
    ((args, _),) = kernel.calls
    assert all(seen is sent for seen, sent in zip(args, (q, k, v), strict=True))


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
    assert install_fa4_contiguous_dedicated_head_inputs()
    assert install_fa4_contiguous_dedicated_head_inputs()
    wrapped = fake_cute.flash_attn_func, fake_cute.flash_attn_varlen_func
    assert tuple(fn.__wrapped__ for fn in wrapped) == originals, "a second install must not wrap the wrapper"
    assert (flash_utils._flash_fn, flash_utils._flash_varlen_fn) == wrapped


@pytest.mark.parametrize("contiguity_first", [True, False], ids=["contiguity_first", "sink_rescale_first"])
def test_the_two_fa4_installs_stack_once_each_in_either_order(fake_cute, contiguity_first):
    """The sink rescale and the contiguity wrapper share the install seam: in either order each wraps both entry
    points exactly once, repeated installs add nothing, and transformers' references follow the outermost one."""
    originals = fake_cute.flash_attn_func, fake_cute.flash_attn_varlen_func
    installs = [install_fa4_contiguous_dedicated_head_inputs, install_fa4_trainable_sink_rescale]
    if not contiguity_first:
        installs.reverse()
    for install in installs + installs:
        assert install()
    stacked = fake_cute.flash_attn_func, fake_cute.flash_attn_varlen_func
    assert tuple(fn.__wrapped__.__wrapped__ for fn in stacked) == originals
    assert (flash_utils._flash_fn, flash_utils._flash_varlen_fn) == stacked


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
    wrapper before transformers caches the entry points; other backends never do."""
    calls = []
    monkeypatch.setattr(model_preparation, "install_fa4_contiguous_dedicated_head_inputs", lambda: calls.append(1))
    monkeypatch.setattr(model_preparation, "patch_transformers_flash_varlen_int_seqlen", lambda: None)
    model_preparation.apply_family_patches(Qwen3Config(), attn_implementation)
    assert bool(calls) is installs


class _Built(Exception):
    """Stops a load at its weight build: transformers refuses an FA4 build on a CPU device."""


@pytest.fixture(scope="module")
def qwen3_5_text_checkpoint(tmp_path_factory) -> str:
    """A tiny ``qwen3_5_text`` checkpoint, of the partial-rotary family whose head_dim-256 q/k the shim exists for."""
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
def test_every_fa4_load_installs_the_contiguity_shim(monkeypatch, tmp_path, qwen3_5_text_checkpoint, load):
    """The loads outside the policy and frozen loaders build their model themselves, so each must pass the family
    patches before its build, as those two do: an unshimmed Qwen3.5 reward model, embedder or trainer-loaded policy
    reads scrambled q/k and returns finite, wrong scores or gradients."""
    applied = collections.Counter()
    for name in _FA4_LOAD_PATCHES:
        monkeypatch.setattr(model_preparation, name, lambda name=name: applied.update([name]))
    builds = []

    def build(kwargs):
        builds.append((kwargs["attn_implementation"], dict(applied)))
        raise _Built

    with pytest.raises(_Built):
        load(qwen3_5_text_checkpoint, tmp_path, monkeypatch, build)
    assert builds == [(FA4, dict.fromkeys(_FA4_LOAD_PATCHES, 1))]


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
