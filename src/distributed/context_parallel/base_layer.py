"""Abstract base class for Ulysses-wrapped attention modules.

Family wrappers in :mod:`.layers` implement the Q/K/V projection and rotary hooks; this base runs
the all-to-alls, GQA flash attention and the legacy fallback. Two forward paths:

- Optimized (the default): RoPE on the local chunk before the all-to-all, then ``flash_attn_func``
  with native GQA in ``[B, S, H, D]`` — no ``repeat_kv``, no transposes.
- Legacy (the MLA families, whose Q/K and V head dims differ): all-to-all first, gather the position
  embeddings, then attention in ``[B, H, S, D]``.
"""

from __future__ import annotations

import functools
import inspect
from abc import ABC, abstractmethod

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from src.distributed.context_parallel.autograd import (
    gather_pos_embeddings,
    ulysses_all_to_all,
    ulysses_all_to_all_fused_kv,
)
from src.hardware import is_blackwell_gpu, is_hopper_gpu
from src.models.patches.attention import (
    ensure_fa4_kernel_cache_env,
    model_fa4_backward_nan_prone,
)

# Hub kernel id of the flash-attn2 build the CP fallback loads; ``validation.py`` accepts the same
# spelling as an attn_implementation, so the two must stay in sync.
HUB_FLASH_ATTN2_KERNEL = "kernels-community/flash-attn2"


def _adapt_to_fa2_contract(func):
    """Wrap a flash-attn variant so CP's FA2-style call works against its actual interface.

    Two differences are absorbed here:

    * ``flash_attn.cute.flash_attn_func`` (FA4) takes no ``dropout_p``. Kwargs the kernel lacks are
      dropped when neutral (dropout 0, full window) and raise otherwise.
    * FA4 returns ``(out, lse)`` even at its default ``return_lse=False``, where FA2 returns the
      output tensor alone; the tuple is unwrapped so callers see one contract.
    """
    params = set(inspect.signature(func).parameters)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for name, neutral in (("dropout_p", 0.0), ("window_size", (-1, -1))):
            if name in kwargs and name not in params:
                if kwargs[name] != neutral:
                    raise ValueError(
                        f"CP flash-attn fallback {func.__module__}.{func.__name__} does not support "
                        f"{name}={kwargs[name]!r} (only the neutral {neutral!r} can be dropped)."
                    )
                kwargs.pop(name)
        out = func(*args, **kwargs)
        return out[0] if isinstance(out, tuple) else out

    return wrapper


@functools.lru_cache(maxsize=2)
def get_flash_attn_func(allow_fa4: bool = True):
    """Return ``flash_attn_func``: FA3 on Hopper, FA4 on Blackwell, else FA2, then the HF kernel.

    CP calls ``flash_attn_func`` directly rather than through the model's attention module, so this
    probe, not ``_attn_implementation``, decides which kernel a CP run executes. The arch-specific
    probes must stay ahead of the FA2 one: both images ship flash-attn 2, so an FA2 probe placed
    first would win everywhere and leave the arch kernels below it unreachable.

    ``allow_fa4=False`` skips the Blackwell probe for families whose FA4 backward emits NaN
    gradients; since the probe overrides the configured label, they cannot opt out by requesting FA2.
    """
    # Hopper (sm_90) cannot take the FA2 path: the image stubs out the split-K kernels that FA2's
    # non-varlen forward (the one CP calls) selects by occupancy heuristic, so it would raise.
    if is_hopper_gpu():
        try:
            from flash_attn_interface import (  # noqa: PLC0415 — optional arch-specific attention backend
                flash_attn_func as fa3_func,
            )

            return _adapt_to_fa2_contract(fa3_func)
        except ImportError:
            pass

    # FA4 only on data-center Blackwell (SM100+): its CuTe kernels do not run elsewhere. It is also
    # the backend the model was labelled with and whose kernels warmup_fa4_kernels pre-compiled.
    if allow_fa4 and is_blackwell_gpu():
        try:
            ensure_fa4_kernel_cache_env()  # persist the CuTe DSL compile cache before first import
            from flash_attn.cute import (  # noqa: PLC0415 — optional arch-specific attention backend
                flash_attn_func as fa4_func,
            )

            return _adapt_to_fa2_contract(fa4_func)
        except ImportError:
            pass

    try:
        from flash_attn import flash_attn_func  # noqa: PLC0415 — optional arch-specific attention backend

        return flash_attn_func
    except ImportError:
        pass

    try:
        import kernels  # noqa: PLC0415 — optional arch-specific attention backend

        kernel = kernels.get_kernel(HUB_FLASH_ATTN2_KERNEL)
        return kernel.flash_attn_func
    except Exception as last_resort_error:
        # Chained: this last probe also hits the hub, so a cache/auth/kernel failure here is not the
        # same condition as flash-attn being absent.
        raise ImportError(
            f"flash-attn is required for Context Parallelism but neither flash_attn nor "
            f"{HUB_FLASH_ATTN2_KERNEL} could be loaded. Install flash-attn "
            f"(see agent-docs/optimization/flash-attention.md) or the kernels package."
        ) from last_resort_error


class UlyssesAttentionBase(nn.Module, ABC):
    """Abstract base for Ulysses-wrapped attention modules.

    Subclasses implement :meth:`_project_qkv`; the rotary defaults to rotate-half in the optimized
    ``[B, S, H, D]`` layout (:meth:`_apply_rotary_core`), and families off that path
    (``_optimize_attention = False``, the MLA base) implement :meth:`_apply_rotary_pos_emb` for the
    legacy ``[B, H, S, D]`` one instead.
    """

    # False for the families off the optimized path: :class:`MLAUlyssesAttentionBase`, whose Q/K and
    # V head dims differ.
    _optimize_attention: bool = True

    HF_MODULE_NAMES: tuple[str, ...] = ()

    # The config's attention label describes the family's own attention forward, which this wrapper
    # replaces (the running kernel comes from get_flash_attn_func). False declares the wrapper's
    # geometry flash-compatible for a family whose modeling code cannot carry a flash label at all
    # (v4-era remote code declaring `_supports_flash_attn_2`, which transformers v5 ignores in favour
    # of `_supports_flash_attn`), so validation reads that instead of a label about dead code.
    REQUIRES_FLASH_ATTN_LABEL: bool = True

    # Full-sequence ``position_ids``, published once per forward by
    # :class:`~src.distributed.context_parallel.wrapper.UlyssesCPModelWrapper`, which holds them
    # before the split. The legacy path applies RoPE after the all-to-all, where Q/K span the whole
    # sequence, so family hooks there need these rather than this rank's chunk.
    global_position_ids: torch.Tensor | None = None

    def __init__(
        self,
        original_attention: nn.Module,
        cp_group: dist.ProcessGroup,
        cp_size: int,
    ):
        super().__init__()
        self.original_attention = original_attention
        self.cp_group = cp_group
        self.cp_size = cp_size

        self.config = original_attention.config
        self.layer_idx = original_attention.layer_idx
        self.head_dim = original_attention.head_dim
        self.scaling = self._resolve_scaling()
        self.attention_dropout = original_attention.attention_dropout
        self.is_causal = original_attention.is_causal
        self.sliding_window = getattr(original_attention, "sliding_window", None)

        # Projections are read through self.original_attention.*: PEFT wraps them after this wrapper.

        self.num_q_heads = self.config.num_attention_heads
        self.num_kv_heads = self._resolve_num_kv_heads(original_attention)

        if self.num_q_heads % cp_size != 0:
            raise ValueError(f"Q heads ({self.num_q_heads}) must be divisible by CP size ({cp_size})")
        if self.num_kv_heads % cp_size != 0:
            raise ValueError(f"KV heads ({self.num_kv_heads}) must be divisible by CP size ({cp_size})")

        self.local_q_heads = self.num_q_heads // cp_size
        self.local_kv_heads = self.num_kv_heads // cp_size
        self.local_num_key_value_groups = self.local_q_heads // self.local_kv_heads

        # The kernel probe overrides the configured label, so a family whose FA4 backward emits NaN
        # gradients cannot opt out by requesting FA2. Resolved once, not per forward.
        self._allow_fa4 = not model_fa4_backward_nan_prone(self.config)

        # Last, so family-derived state sees every field above.
        self._configure(original_attention)

    def _configure(self, original_attention: nn.Module) -> None:
        """Family hook for state derived from the config or the unpatched module. Default: none."""

    def _resolve_scaling(self) -> float:
        """The softmax scale. Families computing it inline in their own forward (Bailing divides the
        scores by ``sqrt(head_dim)``) override; ``self.head_dim`` is already set here."""
        return self.original_attention.scaling

    @staticmethod
    def _resolve_num_kv_heads(attention: nn.Module) -> int:
        """Head count of the K/V tensors :meth:`_project_qkv` emits, read off the unpatched HF module
        so this wrapper and :func:`~src.distributed.context_parallel.validation.
        validate_model_for_ulysses` resolve it from the same place.

        Families whose K/V are not the config's ``num_key_value_heads`` override: MLA expands
        compressed KV through ``kv_b_proj``, which is built with one head per query head.
        """
        return attention.config.num_key_value_heads

    def debug_fields(self) -> dict[str, object]:
        """``key=value`` pairs for the patcher's per-layer debug line; families extend this."""
        return {
            "Q": f"{self.num_q_heads}->{self.local_q_heads}",
            "KV": f"{self.num_kv_heads}->{self.local_kv_heads}",
        }

    @property
    def _output_projection(self) -> nn.Module:
        """The attention output projection, resolved per call because PEFT wraps the projection after
        this wrapper is built. Families naming it something other than ``o_proj`` (Bailing's
        ``dense``) override."""
        return self.original_attention.o_proj

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate-half on the last dim: ``[-x2, x1]`` (layout-agnostic)."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_partial_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Rotate-half RoPE over the first ``rotary_dim = cos.shape[-1]`` channels, the rest passed
        through (``partial_rotary_factor`` families). Layout-agnostic: only the last dim is touched."""
        rotary_dim = cos.shape[-1]
        if rotary_dim == x.shape[-1]:
            return (x * cos) + (self._rotate_half(x) * sin)
        x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
        x_rot = (x_rot * cos) + (self._rotate_half(x_rot) * sin)
        return torch.cat([x_rot, x_pass], dim=-1)

    @abstractmethod
    def _project_qkv(
        self, hidden_states: torch.Tensor, batch_size: int, local_seq_len: int
    ) -> tuple[torch.Tensor, ...]:
        """Project hidden states to Q, K, V tensors. Model-specific.

        Returns ``(query, key, value)`` plus any extra tensors the family's :meth:`_post_attention`
        consumes (Qwen3.5's sigmoid gate). :meth:`forward` threads the extras through as locals, so
        nothing is stashed on the module and gradient-checkpoint recompute stays re-entrant.
        """

    def _project_qkv_plain(
        self, hidden_states: torch.Tensor, batch_size: int, local_seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Separate ``q_proj``/``k_proj``/``v_proj`` straight into ``[B, S, H, D]``.

        The whole projection for families that add nothing before RoPE (GptOss, Cohere2 MoE). Head
        counts are given explicitly rather than inferred, so a projection whose width disagrees with
        the config raises here instead of reshaping.
        """
        attn = self.original_attention
        return (
            attn.q_proj(hidden_states).view(batch_size, local_seq_len, self.num_q_heads, self.head_dim),
            attn.k_proj(hidden_states).view(batch_size, local_seq_len, self.num_kv_heads, self.head_dim),
            attn.v_proj(hidden_states).view(batch_size, local_seq_len, self.num_kv_heads, self.head_dim),
        )

    def _apply_rotary_pos_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embeddings in the legacy ``[B, H, S, D]`` layout.

        Reached only from :meth:`_forward_legacy`, so only families off the optimized path
        (``_optimize_attention = False``) implement it; not abstract, which would force every
        optimized family to write an unreachable implementation.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _apply_rotary_pos_emb. Implement it for the "
            f"legacy [B, H, S, D] path (_optimize_attention = False), or override _apply_rotary_core "
            f"for the optimized [B, S, H, D] path."
        )

    def _apply_rotary_core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings in ``[B, S, H, D]`` layout (optimized path), zero-copy.

        ``cos`` / ``sin`` arrive as ``[B, S, 1, D]``, broadcasting over the head axis. The default is
        rotate-half over the leading ``cos.shape[-1]`` channels, the convention Qwen3, Qwen3.5/3.6
        and Bailing share; families with a different rotary override (GptOss's split-concat,
        Cohere2's interleaved fp32). A new optimized family inherits rotate-half by default, so check
        its rotary against this one (``tests/cpu/parallelism/test_cp_qwen3_rotary.py``).
        """
        return self._apply_partial_rotary(q, cos, sin), self._apply_partial_rotary(k, cos, sin)

    def _post_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.LongTensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Family hook applied to Q/K straight after RoPE (default: identity).

        ``position_ids`` describe the sequence axis Q/K carry at this point: this rank's local chunk
        on the optimized path (RoPE before the all-to-all), the wrapper-published
        :attr:`global_position_ids` on the legacy one (RoPE after it). Called on both paths, so an
        implementation must be layout-aware: ``[B, S, H, D]`` optimized, ``[B, H, S, D]`` legacy.
        """
        return q, k

    def _post_attention(self, attn_output: torch.Tensor, *extras: torch.Tensor) -> torch.Tensor:
        """Family hook applied to the attention output just before ``o_proj`` (default: identity).

        ``extras`` are whatever :meth:`_project_qkv` returned beyond Q/K/V, passed through as locals.
        """
        return attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward with Ulysses sequence parallelism.

        The decoder layer's remaining attention kwargs (``attention_mask``, ``past_key_values``,
        ``cache_position``) land in ``**kwargs`` and are dropped: CP is training-only (no cache) and
        the flash backends take causal masking from the varlen metadata, not a mask tensor.

        Args:
            hidden_states: ``[batch, seq/CP, hidden]``
            position_embeddings: ``(cos, sin)`` for RoPE on the local chunk only.
            position_ids: local-chunk positions, threaded to :meth:`_post_rope` on the optimized
                path (the legacy path hands it :attr:`global_position_ids` instead).
        """
        batch_size, local_seq_len, _ = hidden_states.shape

        query_states, key_states, value_states, *extras = self._project_qkv(hidden_states, batch_size, local_seq_len)

        cos, sin = position_embeddings

        # Per-family choice; validate_model_for_ulysses has already checked the flash backend.
        forward_fn = self._forward_optimized if self._optimize_attention else self._forward_legacy
        # The legacy path RoPEs after the all-to-all, where Q/K span the whole sequence, so its
        # _post_rope hook takes the wrapper-published global positions rather than the local chunk.
        path_position_ids = position_ids if self._optimize_attention else self.global_position_ids
        attn_output = forward_fn(
            query_states,
            key_states,
            value_states,
            cos,
            sin,
            batch_size,
            local_seq_len,
            path_position_ids,
        )

        attn_output = self._post_attention(attn_output, *extras)
        attn_output = self._output_projection(attn_output)

        return attn_output, None

    def _forward_optimized(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        batch_size: int,
        local_seq_len: int,
        position_ids: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        """Optimized forward: RoPE before the all-to-all, then native GQA flash attention.

        Unlike the legacy path: no position-embedding all-gather, no ``repeat_kv`` (flash_attn
        handles GQA natively) and no transposes.
        """
        cos_expanded = cos.unsqueeze(2)
        sin_expanded = sin.unsqueeze(2)
        query_states, key_states = self._apply_rotary_core(query_states, key_states, cos_expanded, sin_expanded)
        query_states, key_states = self._post_rope(query_states, key_states, position_ids)

        query_states, key_states, value_states = ulysses_all_to_all_fused_kv(
            query_states,
            key_states,
            value_states,
            self.cp_group,
            scatter_dim=2,
            gather_dim=1,
        )

        flash_attn_func = get_flash_attn_func(self._allow_fa4)

        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            **self._flash_call_kwargs(key_states.shape[1]),
        )

        attn_output = ulysses_all_to_all(
            attn_output,
            self.cp_group,
            scatter_dim=1,
            gather_dim=2,
        )

        return attn_output.reshape(batch_size, local_seq_len, -1)

    def _forward_legacy(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        batch_size: int,
        local_seq_len: int,
        global_position_ids: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        """Legacy forward: all-to-all → gather position embeddings → RoPE → attention.

        For families off the optimized path (the MLA base). Q/K span the whole sequence from the
        all-to-all on, so :meth:`_post_rope` gets the global positions, not this rank's chunk.
        """
        query_states, key_states, value_states = ulysses_all_to_all_fused_kv(
            query_states,
            key_states,
            value_states,
            self.cp_group,
            scatter_dim=2,
            gather_dim=1,
        )

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = gather_pos_embeddings(cos, sin, self.cp_group, self.cp_size)

        query_states, key_states = self._apply_rotary_pos_emb(query_states, key_states, cos, sin)
        query_states, key_states = self._post_rope(query_states, key_states, global_position_ids)

        attn_output = self._compute_attention(query_states, key_states, value_states)

        attn_output = attn_output.transpose(1, 2)

        attn_output = ulysses_all_to_all(
            attn_output,
            self.cp_group,
            scatter_dim=1,
            gather_dim=2,
        )

        return attn_output.reshape(batch_size, local_seq_len, -1)

    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat KV heads for GQA."""
        if n_rep == 1:
            return hidden_states
        batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
        return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)

    def _compute_attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Compute attention with local heads but the full sequence.

        Always flash: every CP entry point runs ``validate_model_for_ulysses`` first, which raises
        unless the attention implementation is in ``SUPPORTED_ATTN_IMPLEMENTATIONS``.
        """
        key = self._repeat_kv(key, self.local_num_key_value_groups)
        value = self._repeat_kv(value, self.local_num_key_value_groups)
        return self._flash_attention(query, key, value)

    def _flash_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Flash attention implementation."""
        flash_attn_func = get_flash_attn_func(self._allow_fa4)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        attn_output = flash_attn_func(query, key, value, **self._flash_call_kwargs(key.shape[1]))

        return attn_output.transpose(1, 2)

    def _flash_call_kwargs(self, kv_len: int) -> dict:
        """Flash-attn call kwargs shared by every CP attention path.

        The sliding window follows the HF convention: applied only past its own length, minus 1 for
        the diagonal. MLA families carry no sliding window, so the window is ``(-1, -1)`` there.
        """
        return {
            "dropout_p": 0.0 if not self.training else self.attention_dropout,
            "softmax_scale": self.scaling,
            "causal": self.is_causal,
            "window_size": (
                (self.sliding_window - 1, self.sliding_window - 1)
                if self.sliding_window and kv_len > self.sliding_window
                else (-1, -1)
            ),
        }


class MLAUlyssesAttentionBase(UlyssesAttentionBase):
    """Ulysses CP attention for DeepSeek-V3-style MLA models: GLM4-MoE-Lite, Mistral4.

    Implements the shared MLA pieces: geometry read off the wrapped module, the two rotary helpers,
    the compressed-KV :meth:`_project_qkv`, the rope/nope-split :meth:`_apply_rotary_pos_emb` and the
    V-padding attention fn. Subclasses add only their HF class names and any family-specific hook.
    """

    # MLA needs the [B, H, S, D] legacy path: its compressed projections, its nope/rope split and a
    # ``v_head_dim`` that may differ from ``qk_head_dim`` (V-padding below) do not fit flash-attn's
    # optimized GQA path, which assumes plain per-head QKV of a single head dim.
    _optimize_attention = False

    def __init__(
        self,
        original_attention: nn.Module,
        cp_group: dist.ProcessGroup,
        cp_size: int,
    ):
        # MLA families spell the query/key head dim ``qk_head_dim``, while the parent reads
        # ``head_dim`` off the wrapped module. Back-fill before delegating, only where the family
        # carries no ``head_dim`` of its own.
        if not hasattr(original_attention, "head_dim"):
            original_attention.head_dim = original_attention.qk_head_dim
        super().__init__(original_attention, cp_group, cp_size)

        self.qk_head_dim = original_attention.qk_head_dim
        self.qk_nope_head_dim = original_attention.qk_nope_head_dim
        self.qk_rope_head_dim = original_attention.qk_rope_head_dim
        self.v_head_dim = original_attention.v_head_dim
        self.kv_lora_rank = original_attention.kv_lora_rank

        # Whether Q uses the LoRA-style compression (KV always does), read off the modules
        # :meth:`_project_qkv` calls: transformers keeps the unused branch's attribute present but
        # None, so presence alone does not answer it.
        self.use_qkv_lora = getattr(original_attention, "q_a_proj", None) is not None

        # Read unconditionally, as both families' own forwards do: their configs default it to True,
        # so a ``getattr(..., False)`` fallback would invert the family's rotary.
        self.rope_interleave = bool(self.config.rope_interleave)

    @staticmethod
    def _resolve_num_kv_heads(attention: nn.Module) -> int:
        """MLA's K/V come out of ``kv_b_proj``, which both families build with one head per query
        head, so ``config.num_key_value_heads`` does not describe that tensor. Derived from the
        projection itself, so the head math matches the weights being reshaped."""
        return attention.kv_b_proj.out_features // (attention.qk_nope_head_dim + attention.v_head_dim)

    def debug_fields(self) -> dict[str, object]:
        return super().debug_fields() | {
            "qk_head_dim": f"{self.qk_head_dim} (nope={self.qk_nope_head_dim}, rope={self.qk_rope_head_dim})",
            "v_head_dim": self.v_head_dim,
            "kv_lora_rank": self.kv_lora_rank,
            "use_qkv_lora": self.use_qkv_lora,
            "rope_interleave": self.rope_interleave,
        }

    def _flash_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Flash attention with V padded to ``qk_head_dim`` where ``v_head_dim`` differs.

        flash-attn needs Q/K/V to share a head dim; the pad and crop are no-ops for a family whose
        dims already match (GLM4-MoE-Lite and Mistral4 ship them equal). MLA families have no sliding
        window, so the shared kernel runs unwindowed.
        """
        pad = self.qk_head_dim - self.v_head_dim
        if pad > 0:
            value = F.pad(value, (0, pad))

        attn_output = super()._flash_attention(query, key, value)

        if pad > 0:
            attn_output = attn_output[..., : self.v_head_dim].contiguous()
        return attn_output

    def _apply_rotary_emb(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Standard rotate-half rotary embedding."""
        return (x * cos) + (self._rotate_half(x) * sin)

    def _apply_rotary_emb_interleaved(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Interleaved rotary (mirrors transformers' ``apply_rotary_pos_emb_interleave``):
        reshape to pairs, swap, flatten, then rotate-half."""
        b, h, s, d = x.shape
        x = x.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        return (x * cos) + (self._rotate_half(x) * sin)

    def _project_qkv(
        self, hidden_states: torch.Tensor, batch_size: int, local_seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project hidden states to Q, K, V for a DeepSeek-V3-style MLA attention.

        Q optionally uses the LoRA-style compression (``use_qkv_lora``); KV always does. The
        ``kv_lora_rank`` slots expand (via norm + ``kv_b_proj``) into ``k_nope`` + V, and the trailing
        ``qk_rope_head_dim`` slots are the shared rotary K, broadcast to every KV head before the
        all-to-all so the head-dim scatter sees a contiguous tensor.

        Returns Q/K shaped ``[B, S_local, heads, qk_head_dim]`` and V ``[B, S_local, kv_heads,
        v_head_dim]``.
        """
        attn = self.original_attention

        if self.use_qkv_lora:
            q = attn.q_b_proj(attn.q_a_layernorm(attn.q_a_proj(hidden_states)))
        else:
            q = attn.q_proj(hidden_states)
        q = q.view(batch_size, local_seq_len, self.num_q_heads, self.qk_head_dim)

        compressed_kv = attn.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot_shared = torch.split(
            compressed_kv,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )

        kv = attn.kv_b_proj(attn.kv_a_layernorm(k_pass))
        kv = kv.view(
            batch_size,
            local_seq_len,
            self.num_kv_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        k_nope = kv[..., : self.qk_nope_head_dim]
        v = kv[..., self.qk_nope_head_dim :]

        k_rot = k_rot_shared.view(batch_size, local_seq_len, 1, self.qk_rope_head_dim)
        k_rot = k_rot.expand(batch_size, local_seq_len, self.num_kv_heads, self.qk_rope_head_dim)

        k = torch.cat([k_nope, k_rot.contiguous()], dim=-1)
        return q, k, v

    def _apply_rotary_pos_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotary on the rope half of the head dims only (rope/nope split), legacy ``[B, H, S, D]``."""
        q_nope = q[..., : self.qk_nope_head_dim]
        q_rope = q[..., self.qk_nope_head_dim :]
        k_nope = k[..., : self.qk_nope_head_dim]
        k_rope = k[..., self.qk_nope_head_dim :]

        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        rotary = self._apply_rotary_emb_interleaved if self.rope_interleave else self._apply_rotary_emb
        q_rope = rotary(q_rope, cos, sin)
        k_rope = rotary(k_rope, cos, sin)

        return torch.cat([q_nope, q_rope], dim=-1), torch.cat([k_nope, k_rope], dim=-1)
