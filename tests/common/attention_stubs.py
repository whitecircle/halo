"""Stand-ins for attention kernel tests: a decoder attention module for a test calling an attention function directly,
as ``sdpa_flex_sliding`` hands FlexAttention only the call a module declares, so a bare ``nn.Module`` would send every
call to SDPA; and FA4's module, recording what its JIT cache would compile."""

from __future__ import annotations

import sys
import types
from functools import partial
from typing import NamedTuple

import torch
from transformers import modeling_flash_attention_utils as flash_utils


def causal_attention(
    heads: int = 4,
    kv_heads: int = 2,
    head_dim: int = 64,
    scaling: float | None = 1.0,
    sliding_window: int | None = None,
) -> torch.nn.Module:
    """A causal attention module declaring heads, width, ``scaling`` and ``sliding_window`` (None on a full layer)."""
    module = torch.nn.Module()
    module.is_causal = True
    module.num_heads = heads
    module.num_key_value_groups = heads // kv_heads
    module.head_dim = head_dim
    module.scaling = scaling
    module.sliding_window = sliding_window
    return module


# The transformers module globals the flash path caches the bound kernel and its kwarg map in.
_FLASH_CACHE_GLOBALS = (
    "_loaded_implementation",
    "_flash_fn",
    "_flash_varlen_fn",
    "_flash_with_kvcache_fn",
    "_pad_fn",
    "_unpad_fn",
    "_process_flash_kwargs_fn",
    "_process_with_kvcache_kwargs_fn",
)


class KernelKey(NamedTuple):
    """The call arguments FA4's compile key reads, for a forward kernel or the backward one."""

    dtype: torch.dtype
    qk_dim: int
    v_dim: int
    head_ratio: int
    causal: bool
    window: bool
    varlen: bool
    sink: bool
    softcap: bool
    lse: bool  # FA4's forward allocates the log-sum-exp only when its inputs require grad
    backward: bool
    deterministic: bool  # read by the backward's key alone


class _RecordedBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, record):
        ctx.record = record
        return q * 1.0

    @staticmethod
    def backward(ctx, grad):
        ctx.record()
        return grad, None, None, None


class CompileCountingCute(types.ModuleType):
    """Compiles once per key, as FA4's JIT cache does, and raises on a varlen call with a softcap once recorded."""

    def __init__(self):
        super().__init__("flash_attn.cute")
        self.compiled: set[tuple] = set()
        self.calls: list[tuple] = []

        def flash_attn_func(
            q,
            k,
            v,
            softmax_scale=None,
            causal=False,
            window_size=(None, None),
            learnable_sink=None,
            softcap=0.0,
            deterministic=False,
        ):
            return self._run(
                q, k, v, causal, window_size, learnable_sink, softcap, varlen=False, deterministic=deterministic
            )

        def flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=None,
            max_seqlen_k=None,
            softmax_scale=None,
            causal=False,
            window_size=(None, None),
            learnable_sink=None,
            softcap=0.0,
            deterministic=False,
        ):
            varlen = cu_seqlens_q is not None
            return self._run(
                q, k, v, causal, window_size, learnable_sink, softcap, varlen=varlen, deterministic=deterministic
            )

        self.flash_attn_func = flash_attn_func
        self.flash_attn_varlen_func = flash_attn_varlen_func

    def _run(self, q, k, v, causal, window_size, learnable_sink, softcap, *, varlen, deterministic):
        grad = any(t.requires_grad for t in (q, k, v))
        key = KernelKey(
            q.dtype,
            q.shape[-1],
            v.shape[-1],
            q.shape[-2] // k.shape[-2],
            causal,
            window_size[0] is not None,
            varlen,
            learnable_sink is not None,
            bool(softcap),
            lse=grad,
            backward=False,
            deterministic=False,
        )
        self.calls.append(key)
        if varlen and softcap:
            raise AssertionError("an FA4 varlen call with a softcap, which stays off those kernels")
        self.compiled.add(key)
        if not grad:
            return q * 1.0
        backward_key = key._replace(backward=True, deterministic=deterministic)
        return _RecordedBackward.apply(q, k, v, partial(self._dispatch_backward, backward_key))

    def _dispatch_backward(self, key: KernelKey) -> None:
        self.calls.append(key)
        self.compiled.add(key)


def install_compile_counting_cute(monkeypatch) -> CompileCountingCute:
    """In place of ``flash_attn.cute``, with transformers' cached kernel dropped for the test."""
    fake = CompileCountingCute()
    monkeypatch.setitem(sys.modules, "flash_attn.cute", fake)
    for name in _FLASH_CACHE_GLOBALS:
        monkeypatch.setattr(flash_utils, name, getattr(flash_utils, name))
    flash_utils._loaded_implementation = None
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    return fake
