"""Attention-implementation dispatch and model-specific attention quirks.

Picks the best ``attn_implementation`` (FA4/FA3 → FA2 → SDPA → flex → eager) by GPU capability and
model support flags — including which implementations may run against GptOss's live sinks — then
applies the per-backend shims below (Gemma4 SDPA, flex compile, FA4 warm-up, packed position ids).
The GptOss sink policy itself (neutralize / freeze / train, stamps, save-time inverse) lives in
``gpt_oss_sinks.py``.
"""

from __future__ import annotations

import functools
import inspect
import os
import tempfile
import warnings

import torch
from accelerate.logging import get_logger
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, PreTrainedModel
from transformers import modeling_flash_attention_utils as flash_utils
from transformers.integrations import sdpa_attention as _sdpa_mod
from transformers.utils import is_flash_attn_2_available

from src.hardware import is_blackwell_gpu, is_hopper_gpu
from src.models.attention_geometry import (
    resolve_head_dim,
    resolve_num_key_value_heads,
)
from src.models.loading.config_levels import set_config_field_run_scoped, text_config

try:
    # Version-compat, not an optional dep: patch targets upstream may rename, and an absent one
    # makes the owning patch log loudly and no-op rather than crash the run.
    from transformers.integrations.flex_attention import WrappedFlexAttention, flex_attention
except ImportError:
    WrappedFlexAttention = flex_attention = None

try:
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssPreTrainedModel
except ImportError:
    GptOssPreTrainedModel = None

try:
    # Same version-compat guard as the two above: an unguarded family import makes this module —
    # which every loader imports for attention dispatch — unimportable on a transformers build
    # without that family, long before the per-model guard inside the patch could no-op.
    from transformers.models.mistral4 import modeling_mistral4 as m4
except ImportError:
    m4 = None

logger = get_logger(__name__)

# Varlen backends consume cu_seqlens from packed position_ids. Dense backends isolate packed
# documents only where the model plumbs position_ids into its mask — see agent-docs/data/collators.md.
VARLEN_ATTN_IMPLEMENTATIONS = ("flash_attention_2", "flash_attention_3", "flash_attention_4")

# Families interleaving GatedDeltaNet (linear-attention) layers with softmax attention. Prefixes,
# not exact spellings, so text-tower variants (``qwen3_5_moe_text``) match; one home for the
# collator factory's seq_idx emission and the packing shim below.
GDN_MODEL_TYPE_PREFIXES = ("qwen3_5", "qwen3_next")

# The two spellings a flash kernel gives the learnable-sink argument (FA3's ``s_aux``, FA4's
# ``learnable_sink``). transformers keys sink support off exactly these two names — the second is
# its ``_flash_api_alternative_names`` entry for the first.
FLASH_SINK_PARAM_NAMES = ("s_aux", "learnable_sink")

# The sink-reset verdict :func:`validate_attn_implementation` records on the config INSTANCE so its
# nested re-validations decide on the same state. Written run-scoped, so ``config_export_ready``
# strips it from every exported ``config.json``.
_SINKS_RESET_ATTR = "_halo_sinks_reset"

# Shapes :func:`warmup_fa4_kernels` compiles against. The floor keeps the dense kernel on a realistic
# tile count; the margin is what pushes the row PAST a sliding window, so the windowed variant is the
# one that compiles rather than falling back to the full-attention path.
_WARMUP_MIN_SEQ_LEN = 512
_WARMUP_WINDOW_MARGIN = 8


def silence_cute_dsl_deprecations() -> None:
    """Mute the CuTe DSL's per-compile ``DeprecationWarning`` flood from FA4 kernel JIT.

    Each compile carries a fresh warning registry, so Python's once-per-location dedup never engages
    and the log drowns in one third-party message about an API only the DSL calls. Scoped to that
    package, so every other deprecation still surfaces.
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"cutlass(\.|$)")


def anchor_jit_cache_dir(var: str, subdir: str) -> None:
    """Point a JIT cache directory at ``HF_HOME`` (else the temp dir) unless the caller already set it.

    The default homes are ephemeral inside a ``--rm`` container, so every rank recompiles what a
    previous run already built. One helper so the FA4 and Triton caches cannot drift onto different
    anchors — one volume carries every kernel cache.
    """
    if var not in os.environ:
        os.environ[var] = os.path.join(os.environ.get("HF_HOME") or tempfile.gettempdir(), subdir)


def ensure_fa4_kernel_cache_env() -> None:
    """Enable the FA4 (``flash_attn.cute``) persistent kernel cache + TVM-FFI.

    Anchors the on-disk AOT cache on ``HF_HOME`` (else every rank recompiles the JIT kernels) and
    mutes the DSL's deprecation flood — both preconditions of the same import. Must run BEFORE the
    first ``import flash_attn.cute`` (env read at import). Idempotent.
    """
    silence_cute_dsl_deprecations()
    os.environ.setdefault("FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED", "1")
    os.environ.setdefault("CUTE_DSL_ENABLE_TVM_FFI", "1")
    anchor_jit_cache_dir("FLASH_ATTENTION_CUTE_DSL_CACHE_DIR", "flash_attn_cute_dsl_cache")


def patch_transformers_flash_varlen_int_seqlen() -> None:
    """Make transformers pass varlen ``max_seqlen_{q,k}`` to the flash kernel as python ints, not tensors.

    In eager, ``_process_flash_attention_kwargs`` forwards a fresh CUDA tensor each step, and FA4's
    varlen-backward JIT compile key hashes it by tensor *identity* — so the backward kernel
    recompiles every call (~190 s/step vs ~10 s; the forward already coerces it). Patching the kwargs
    builder covers whichever flash fn transformers dispatches, at one ``.item()`` per call.
    Idempotent; no-op if the private helper is renamed by a transformers upgrade.
    """
    fn = getattr(flash_utils, "_process_flash_attention_kwargs", None)
    if fn is None or getattr(fn, "_halo_int_seqlen", False):
        return

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        flash_kwargs = fn(*args, **kwargs)
        mq = flash_kwargs.get("max_seqlen_q")
        mk = flash_kwargs.get("max_seqlen_k")
        if torch.is_tensor(mq):
            flash_kwargs["max_seqlen_q"] = int(mq)
            # transformers aliases max_seqlen_k to the same tensor to avoid a 2nd sync; preserve that.
            if mk is mq:
                flash_kwargs["max_seqlen_k"] = flash_kwargs["max_seqlen_q"]
                return flash_kwargs
        if torch.is_tensor(mk):
            flash_kwargs["max_seqlen_k"] = int(mk)
        return flash_kwargs

    wrapped._halo_int_seqlen = True
    flash_utils._process_flash_attention_kwargs = wrapped


class PositionIdsInjectingRegistry:
    """A view of ``ALL_ATTENTION_FUNCTIONS`` whose flash interfaces re-inject the module's stashed
    ``position_ids``. Installed into a single modeling module's namespace, so every other family
    keeps the untouched registry object."""

    def __init__(self, inner):
        self._inner = inner

    def get_interface(self, impl, *args, **kwargs):
        fn = self._inner.get_interface(impl, *args, **kwargs)
        if not isinstance(impl, str) or "flash" not in impl:
            return fn

        @functools.wraps(fn)
        def with_position_ids(module, *fargs, **fkwargs):
            position_ids = getattr(module, "_halo_packed_position_ids", None)
            if position_ids is not None and "position_ids" not in fkwargs:
                fkwargs["position_ids"] = position_ids
            return fn(module, *fargs, **fkwargs)

        return with_position_ids

    def __getattr__(self, name):
        return getattr(self._inner, name)

    # Dunders resolve on the type, bypassing __getattr__ — forward the Mapping protocol explicitly.
    def __contains__(self, key):
        return key in self._inner

    def __getitem__(self, key):
        return self._inner[key]

    def __iter__(self):
        return iter(self._inner)

    def __len__(self):
        return len(self._inner)


def install_packed_position_ids_injection(modeling_module, owner_cls, stash_targets) -> bool:
    """Stash a forward's ``position_ids`` on the attention modules ``stash_targets`` names, and give
    that modeling module a registry view whose flash interfaces re-inject the stash.

    A family whose forward declares ``position_ids`` but never hands it to the attention interface
    loses varlen packing entirely: ``_is_packed_sequence`` never fires, so a packed row runs as ONE
    dense causal sequence and every document attends across its neighbours, silently. Dense backends
    are unaffected (the packed mask forms at the model level).

    ``owner_cls`` is the class whose forward carries the tensor — the attention module itself
    (Mistral4) or the model above it (Zaya); ``stash_targets`` maps that instance to the attention
    modules the flash interface is called with. Idempotent: ``False`` when already installed.
    """
    if getattr(owner_cls.forward, "_halo_packed_position_ids", False):
        return False

    original_forward = owner_cls.forward
    signature = inspect.signature(original_forward)

    @functools.wraps(original_forward)
    def forward(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        position_ids = bound.arguments.get("position_ids")
        targets = stash_targets(self)
        for target in targets:
            target._halo_packed_position_ids = position_ids
        try:
            return original_forward(self, *args, **kwargs)
        finally:
            for target in targets:
                target._halo_packed_position_ids = None

    forward._halo_packed_position_ids = True
    owner_cls.forward = forward
    registry = getattr(modeling_module, "ALL_ATTENTION_FUNCTIONS", None)
    if registry is not None and not isinstance(registry, PositionIdsInjectingRegistry):
        modeling_module.ALL_ATTENTION_FUNCTIONS = PositionIdsInjectingRegistry(registry)
    return True


def patch_mistral4_flash_packed_position_ids() -> None:
    """Make Mistral4's flash path see ``position_ids``, so packed documents stay isolated.

    ``Mistral4Attention.forward`` declares ``position_ids`` as an explicit parameter (it feeds the
    llama-4 attention scale) and hands only ``**kwargs`` to the attention interface, so the tensor
    never reaches ``_flash_attention_forward``. The stash therefore rides the attention forward
    itself. No-ops if the internals it hooks are renamed.
    """
    if m4 is None:
        logger.warning(
            "transformers ships no mistral4 modeling module on this build, so the packed-document "
            "isolation patch cannot be applied — a packed Mistral4 row would attend across document "
            "boundaries. Train unpacked, or use a transformers build that carries the family."
        )
        return
    attn_cls = getattr(m4, "Mistral4Attention", None)
    if attn_cls is None or getattr(m4, "ALL_ATTENTION_FUNCTIONS", None) is None:
        return
    if install_packed_position_ids_injection(m4, attn_cls, lambda attention: (attention,)):
        logger.info("Patched Mistral4 flash attention to receive position_ids (packed-document isolation)")


def warmup_fa4_kernels(model: PreTrainedModel, *, dtype: torch.dtype, device: torch.device | None = None) -> None:
    """Pre-compile this model's FA4 fwd+bwd attention kernels. Rank-local, best-effort, FA4 only.

    A first-use JIT compile (~10s+) on one rank lets peers race to the next collective and deadlock,
    so the caller runs this on every rank behind one fence
    (:func:`~src.distributed.loading.warmup.warm_attention_kernels`) — every verdict taken here is
    per rank and must not gate a collective.

    The dense and varlen entry points compile SEPARATELY (padded vs packed batches), so both are
    warmed, and ``dtype`` is the run's own — warming bf16 for an fp16 run leaves the called kernel cold.
    """
    try:
        config = getattr(model, "config", None)
        if config is None:
            return
        text_cfg = text_config(config)
        if effective_attn_implementation(config) != "flash_attention_4":
            return

        from flash_attn.cute import flash_attn_func, flash_attn_varlen_func  # noqa: PLC0415 — optional dep

        head_dim = resolve_head_dim(config)
        n_heads = text_cfg.num_attention_heads
        n_kv_heads = resolve_num_key_value_heads(config)
        sliding_window = getattr(text_cfg, "sliding_window", None)
        if device is None:
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        seq_len = max(_WARMUP_MIN_SEQ_LEN, (sliding_window or 0) + _WARMUP_WINDOW_MARGIN)
        scale = head_dim**-0.5
        sink = torch.randn(n_heads, device=device, dtype=dtype) if model_has_sinks(config) else None

        def _warm(flash_fn, q_shape, kv_shape, **kw) -> None:
            q = torch.randn(*q_shape, device=device, dtype=dtype, requires_grad=True)
            k = torch.randn(*kv_shape, device=device, dtype=dtype, requires_grad=True)
            v = torch.randn(*kv_shape, device=device, dtype=dtype, requires_grad=True)
            out = flash_fn(q, k, v, softmax_scale=scale, causal=True, **kw)
            (out[0] if isinstance(out, tuple) else out).sum().backward()  # fwd+bwd both compile

        # Two documents, so the varlen kernel compiles against a real cu_seqlens rather than a
        # single-segment degenerate one.
        cu_seqlens = torch.arange(0, 3 * seq_len, seq_len, device=device, dtype=torch.int32)
        entry_points = (
            (flash_attn_func, (1, seq_len, n_heads, head_dim), (1, seq_len, n_kv_heads, head_dim), {}),
            (
                flash_attn_varlen_func,
                (2 * seq_len, n_heads, head_dim),
                (2 * seq_len, n_kv_heads, head_dim),
                {
                    "cu_seqlens_q": cu_seqlens,
                    "cu_seqlens_k": cu_seqlens,
                    "max_seqlen_q": seq_len,
                    "max_seqlen_k": seq_len,
                },
            ),
        )
        # Warm every dispatched variant (dense/varlen × full/sliding-window × sink): a missed one
        # JIT-compiles later and desyncs.
        window_variants = [{}] + (
            [{"window_size": (sliding_window - 1, sliding_window - 1)}] if sliding_window else []
        )
        sink_kwargs = {} if sink is None else {"learnable_sink": sink}
        for flash_fn, q_shape, kv_shape, shape_kwargs in entry_points:
            for window_kwargs in window_variants:
                _warm(flash_fn, q_shape, kv_shape, **shape_kwargs, **window_kwargs, **sink_kwargs)
        torch.cuda.synchronize()
        logger.info(
            f"FA4 kernel warmup done (head_dim={head_dim}, dtype={dtype}, sliding_window={sliding_window}, "
            f"sinks={sink is not None}, variants={len(entry_points) * len(window_variants)})"
        )
    except Exception as e:
        # main_process_only=False: a rank-local failure precedes the FA4-JIT/DeepEP desync hang.
        logger.warning(f"FA4 kernel warmup skipped (non-fatal): {type(e).__name__}: {e}", main_process_only=False)


def _detect_attention_impl() -> str:
    """Auto-detect best attention implementation from GPU capability.

    Blackwell (SM100+): FA4 when installed, else FA2. Hopper (SM90): FA3 when installed, else FA2. A
    BROKEN (as opposed to absent) accelerated build warns before degrading: FA2 is correct but costs
    2-3x on the attention kernel, and a silent degrade hides that for a whole campaign.
    """
    if not torch.cuda.is_available():
        return "flash_attention_2"  # CPU host; the choice is re-validated against the model downstream
    if is_blackwell_gpu():
        try:
            ensure_fa4_kernel_cache_env()
            from flash_attn.cute import flash_attn_func  # noqa: PLC0415, F401

            logger.info("Using FlashAttention-4 (Blackwell GPU detected, flash_attn.cute installed)")
            return "flash_attention_4"
        except ImportError:
            logger.info("Blackwell GPU detected but FA4 (flash_attn.cute) not installed, using FlashAttention-2")
        except Exception as e:
            logger.warning(
                f"Blackwell GPU detected and FA4 (flash_attn.cute) IS present but unusable "
                f"({type(e).__name__}: {e}); falling back to FlashAttention-2, which is correct but "
                f"2-3x slower on attention. Repair the flash_attn.cute install to regain FA4."
            )
    elif is_hopper_gpu():
        try:
            import flash_attn_3  # noqa: PLC0415, F401

            logger.info("Using FlashAttention-3 (Hopper GPU detected, flash_attn_3 installed)")
            return "flash_attention_3"
        except ImportError:
            logger.info("Hopper GPU detected but flash_attn_3 not installed, using FlashAttention-2")
        except Exception as e:
            logger.warning(
                f"Hopper GPU detected and FA3 (flash_attn_3) IS present but unusable "
                f"({type(e).__name__}: {e}); falling back to FlashAttention-2."
            )
    logger.info("Using FlashAttention-2")
    return "flash_attention_2"


def effective_attn_implementation(model_config) -> str | None:
    """The attention backend this config's DECODER actually runs, or ``None`` when none is recorded.

    Composite wrappers record the resolved backend on the text sub-config and only sometimes mirror
    it to the top level, so a top-level-only read reports ``None`` for a wrapper that IS running
    flash attention — and every gate keyed off it then decides on an absence.
    """
    if model_config is None:
        return None
    return getattr(text_config(model_config), "_attn_implementation", None) or getattr(
        model_config, "_attn_implementation", None
    )


def model_type_matches(model_config, *prefixes: str) -> bool:
    """Whether ``model_type`` (top-level or decoder-level) starts with any of ``prefixes``."""
    candidates = (
        getattr(model_config, "model_type", "") or "",
        getattr(text_config(model_config), "model_type", "") or "",
    )
    return any(mt.startswith(prefixes) for mt in candidates if mt)


def model_has_sinks(model_config) -> bool:
    """Whether this architecture's sinks constrain the attention implementation.

    Narrower than "has attention sinks": DeepSeek-V4 carries them too but is forced to eager below,
    which applies the sink column itself — so none of the kernel selection, warm-up or reset applies.
    Widening this would start resetting its pretrained sinks, a training change rather than a fix.
    """
    return model_type_matches(model_config, "gpt_oss")


def model_is_gemma4(model_config) -> bool:
    """Check whether this is a Gemma4 model (text-only or multimodal wrapper)."""
    return model_type_matches(model_config, "gemma4")


def _model_is_deepseek_v4(model_config) -> bool:
    """Whether this is DeepSeek-V4 (eager-only: head_dim=512 exceeds every FA kernel, SDPA drops the
    learnable sinks, and the compressor concatenates KV entries after the mask is built)."""
    return model_type_matches(model_config, "deepseek_v4")


def model_is_mistral4(model_config) -> bool:
    """Whether this is Mistral4 (MLA + llama-4 attention scaling)."""
    return model_type_matches(model_config, "mistral4")


def model_is_zaya(model_config) -> bool:
    """Whether this is Zaya (CCA Conv1d attention, EDA router, native balancing biases)."""
    return model_type_matches(model_config, "zaya")


def model_fa4_backward_nan_prone(model_config) -> bool:
    """Whether the FA4 backward emits NaN gradients for this model on Blackwell.

    The physical trigger is 256-wide qk attention with only part of the head rotated; SDPA is clean.
    A per-family verdict rather than a numeric one, because the families spell those numbers in
    non-overlapping vocabularies: Qwen3-Next declares ``head_dim: 256`` with
    ``partial_rotary_factor: 0.25``, while MLA GLM-4 MoE Lite declares ``head_dim: 64`` and reaches
    256 as ``qk_nope_head_dim + qk_rope_head_dim``, with no partial-rotary field — so a derivation
    would read it as clean and hand it back to FA4, which is NaN gradients with no error.
    """
    return model_type_matches(model_config, *GDN_MODEL_TYPE_PREFIXES, "glm4_moe_lite")


def patch_sdpa_for_gemma4_long_seq() -> None:
    """Force PyTorch SDPA to the mem-efficient kernel for Gemma4 (idempotent).

    Gemma4's head_dim=512 is rejected by FA2 and cuDNN SDPA, and the math kernel OOMs on the full score
    matrix; only mem-efficient handles arbitrary head_dim. Forces ``use_gqa_in_sdpa``→False (it rejects
    ``enable_gqa=True`` at head_dim=512 but accepts it after a manual KV repeat).
    """
    logger.warning(
        "Gemma4 SDPA patch: forcing mem-efficient SDPA and use_gqa_in_sdpa=False PROCESS-GLOBALLY — "
        "every other sdpa model in this process (e.g. a frozen teacher beside a non-Gemma4 policy) "
        "takes the manual-KV-repeat path too (numerics identical; perf/memory only)."
    )
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    if not getattr(_sdpa_mod.use_gqa_in_sdpa, "_gemma4_patched", False):

        def _no_gqa_in_sdpa(attention_mask, key, value):
            return False

        _no_gqa_in_sdpa._gemma4_patched = True
        _sdpa_mod.use_gqa_in_sdpa = _no_gqa_in_sdpa
    logger.info(
        "Forced mem-efficient SDPA + manual KV repeat for Gemma4 "
        "(unlocks seq>20k by avoiding the math kernel's score matrix)"
    )


def patch_flex_attention_compile(reason: str):
    """Disable transformers' WrappedFlexAttention torch.compile wrapper, using raw flex_attention.

    Must run BEFORE the first flex_attention forward. The compile wrapper deadlocks with EP's NCCL
    all-to-all on seq-length recompiles, and NaNs the FSDP2 + GptOss-sinks backward.
    """
    if WrappedFlexAttention is None:
        # Loud, not fatal: an upstream rename leaves the compile wrapper active, not gone.
        logger.warning(
            f"Could not patch transformers' WrappedFlexAttention (transformers.integrations."
            f"flex_attention did not expose it). The flex-attention torch.compile wrapper stays "
            f"ACTIVE if it still exists — known to deadlock with EP's NCCL all-to-all on "
            f"seq-length recompiles and to NaN the FSDP2 + GptOss-sinks backward. Verify the "
            f"installed transformers still needs this patch ({reason})."
        )
        return

    WrappedFlexAttention._is_flex_compiled = False
    WrappedFlexAttention._compiled_flex_attention = None

    @torch.compiler.disable(recursive=False)
    def _safe_init(self, training):
        if not self._is_flex_compiled or training != self.training:
            self.training = training
            self._compiled_flex_attention = flex_attention
            self._is_flex_compiled = True

    WrappedFlexAttention.__init__ = _safe_init
    logger.info(f"Patched FlexAttention: disabled torch.compile wrapper ({reason})")


def flash_varlen_fn(attn_impl: str):
    """The VARLEN flash kernel transformers dispatches for ``attn_impl``, or ``None`` if not installed.

    Varlen and not the dense twin because that is the signature transformers reads: it builds its
    per-argument capability map from ``_flash_varlen_fn`` alone and applies it to the dense call too,
    so probing the dense one would report a capability transformers will not use.
    """
    try:
        if attn_impl == "flash_attention_4":
            from flash_attn.cute import flash_attn_varlen_func  # noqa: PLC0415 — optional Blackwell-only dep

            return flash_attn_varlen_func
        if attn_impl == "flash_attention_3":
            # flash_attn_interface first (transformers' FA3 dispatch); a coexisting flash_attn_3
            # build would misreport the capability of the kernel actually used.
            try:
                from flash_attn_interface import flash_attn_varlen_func  # noqa: PLC0415 — optional Hopper dep
            except ImportError:
                from flash_attn_3 import flash_attn_varlen_func  # noqa: PLC0415
            return flash_attn_varlen_func
        if attn_impl == "flash_attention_2":
            from flash_attn import flash_attn_varlen_func  # noqa: PLC0415 — lazy, mirrors transformers dispatch

            return flash_attn_varlen_func
    except ImportError:
        return None
    return None


def _attn_impl_handles_sinks(attn_impl: str) -> bool:
    """Whether an attention implementation carries the learnable sink column into the softmax.

    eager and flex_attention add the sink logit in the modeling code; sdpa always drops it. A flash
    path does so only when the installed kernel accepts a sink argument transformers can forward, so
    the verdict is read off the kernel signature — a flash-attn upgrade that adds sink support is
    picked up without a code change.
    """
    if attn_impl in ("eager", "flex_attention"):
        return True
    if attn_impl == "sdpa":
        return False
    fn = flash_varlen_fn(attn_impl)
    if fn is None:
        return False
    params = inspect.signature(fn).parameters
    return any(name in params for name in FLASH_SINK_PARAM_NAMES)


def _enable_sink_model_sdpa(model_config) -> bool:
    """Allow SDPA for a sinks model (GptOss) — valid ONLY when sinks are reset for fine-tuning.

    transformers sets ``_supports_sdpa = False`` on GptOss (SDPA can't add the sink logit); with reset
    sinks the dropped column contributes 0, so SDPA matches eager/flex. Flips the class flag. Returns
    True if flipped.
    """
    if not model_has_sinks(model_config):
        return False
    try:
        model_cls = AutoModelForCausalLM._model_mapping[type(model_config)]
    except KeyError:
        return False
    flipped = False
    for cls in model_cls.__mro__:
        if cls is PreTrainedModel:
            # The BASE default ``_supports_sdpa = False`` lives on PreTrainedModel itself; flipping
            # it would pass the sdpa refusal for every architecture loaded later.
            break
        if cls.__dict__.get("_supports_sdpa") is False:
            cls._supports_sdpa = True
            flipped = True
    if flipped:
        logger.info(f"Enabled SDPA for {model_cls.__name__} (sinks reset for fine-tuning → sink column is a no-op)")
    return flipped


def validate_attn_implementation(model_config, attn_impl: str, sinks_reset: bool | None = None) -> str:
    """Validate attn_implementation against the model class's ``_supports_sdpa`` /
    ``_supports_flash_attn`` flags and the sink-capability matrix, returning a supported choice.

    ``sinks_reset=True`` additionally allows SDPA on a sinks model (the dropped sink column
    contributes 0); explicit ``False`` means live sinks (never sdpa); ``None`` is a re-validation
    context that honors the state already recorded on this config instance.
    """
    has_sinks = model_has_sinks(model_config)
    if attn_impl == "sdpa" and has_sinks and sinks_reset:
        _enable_sink_model_sdpa(model_config)

    # The sink state lives on the config INSTANCE, so it flows into the nested re-validations while a
    # FRESH config cannot inherit the sdpa approval an earlier model's reset granted. RUN-scoped: it
    # describes this run, not the artifact, and transformers serializes every instance attribute.
    if sinks_reset is not None:
        set_config_field_run_scoped(model_config, _SINKS_RESET_ATTR, sinks_reset)
    recorded_reset = getattr(model_config, _SINKS_RESET_ATTR, None)

    # An unmapped config keeps the conservative defaults: sdpa stays *allowed* (an architecture that
    # truly rejects it raises at model build) while the sinks escape hatch stays CLOSED. A composite
    # VLM with no CausalLM sibling (glm5_next, mistral4) declares its flags on the
    # conditional-generation class, so the image-text mapping is the fallback.
    resolved_class = AutoModelForCausalLM._model_mapping.get(type(model_config), None)
    if resolved_class is None:
        resolved_class = AutoModelForImageTextToText._model_mapping.get(type(model_config), None)
    supports_sdpa = getattr(resolved_class, "_supports_sdpa", True)
    supports_flash = getattr(resolved_class, "_supports_flash_attn", True)

    # sdpa silently drops sinks — OK only without sinks, or when they were reset (here or originally).
    sdpa_ok = supports_sdpa and (not has_sinks or recorded_reset is True)

    # LIVE (unreset) sinks + a sink-dropping impl = silent RL corruption: the softmax loses real mass
    # (~-3 nats on gpt-oss-20b) vs the served policy.
    live_sinks = has_sinks and recorded_reset is False
    if live_sinks and not _attn_impl_handles_sinks(attn_impl):
        raise ValueError(
            f"attn_implementation='{attn_impl}' silently drops the learnable attention sinks that "
            f"reset_sinks=false keeps live — every training logprob would shift by nats vs the served "
            f"policy. Use a sink-carrying implementation (flash_attention_4 on Blackwell; flex_attention "
            f"or eager on Hopper — the stock Hopper flash_attention_3 build lacks s_aux, so FA3 works "
            f"only with a build exposing it), or set reset_sinks: true for off-policy fine-tuning."
        )

    # Fallback chain: sinks models prefer FA2 over flex, whose compile path hangs EP past the DeepEP combine barrier.
    candidates = [attn_impl]
    if has_sinks:
        if live_sinks:
            candidates.append("flash_attention_4")
        if is_flash_attn_2_available():
            candidates.append("flash_attention_2")
        candidates.append("flex_attention")
    if sdpa_ok:
        candidates.append("sdpa")
    candidates.append("eager")

    for candidate in candidates:
        if candidate == "sdpa" and not sdpa_ok:
            continue
        # The class flag is the family's own verdict (glm5_next: KDA linear attention + the DSA
        # indexer have no flash path) — transformers raises on the same flag at model build, after
        # the whole checkpoint has been fetched and placed.
        if candidate.startswith("flash_attention") and not supports_flash:
            continue
        if live_sinks and not _attn_impl_handles_sinks(candidate):
            continue
        try:
            model_config._attn_implementation = candidate
        except ValueError:
            # Only a config that validates the assignment can reject a candidate here; stock
            # transformers configs accept any string, so the explicit guards above do the filtering.
            continue
        if candidate != attn_impl:
            logger.warning(
                f"Model {type(model_config).__name__} does not support "
                f"attn_implementation='{attn_impl}', using '{candidate}'"
            )
        if candidate == "flash_attention_2":
            # Stop transformers swapping this FA2 for the SM90-only kernel-hub package.
            _disable_gpt_oss_fa_fallback(model_config)
        return candidate
    logger.warning("Could not validate any attn_implementation, defaulting to 'eager'")
    return "eager"


def revalidate_attn_kwarg(model_kwargs: dict, model_config) -> None:
    """Re-validate a caller's ``attn_implementation`` kwarg against ``model_config``, in place.

    The parallel loaders receive an implementation the POLICY's config resolved and hand it to a
    second ``from_pretrained`` — a family whose ``_supports_*`` flags refuse it would otherwise fail
    deep in the load. Only the VALIDATED value is ever put back: popping first means a second branch
    cannot leak the unvalidated one through.
    """
    attn_impl = model_kwargs.pop("attn_implementation", None)
    if attn_impl is not None:
        model_kwargs["attn_implementation"] = validate_attn_implementation(model_config, attn_impl)


def resolve_attn_implementation(
    model_config,
    attn_implementation: str | None,
    dtype: torch.dtype,
    *,
    sinks_reset: bool = True,
) -> str:
    """Resolve the attention backend for ``model_config`` from the model's own capabilities.

    Auto-detects from GPU capability when ``attn_implementation`` is None, applies the per-family
    kernel limits (fp32, FA4 backward NaN, DeepSeek-V4 eager-only, Gemma4 head_dim=512), then runs
    the sinks/`_supports_sdpa` validator. Purely model-level: parallelism-specific overrides (CP's
    flex→FA switch, EP's flex compile disable) stay with the caller that knows the topology.
    """
    attn_implementation = attn_implementation or _detect_attention_impl()

    if dtype == torch.float32 and attn_implementation.startswith("flash_attention"):
        fp32_fallback = "flex_attention" if model_has_sinks(model_config) else "sdpa"
        logger.warning(
            f"dtype=float32 (fp32 training) is incompatible with {attn_implementation}; "
            f"falling back to attn_implementation='{fp32_fallback}' (FlashAttention is half-precision only)."
        )
        attn_implementation = fp32_fallback

    if attn_implementation == "flash_attention_4" and model_fa4_backward_nan_prone(model_config):
        logger.warning(
            "FlashAttention-4 produces NaN gradients for this model's head_dim-256 partial-rotary "
            "attention; falling back to attn_implementation='sdpa'."
        )
        attn_implementation = "sdpa"

    # Force eager before from_pretrained rejects the auto-detected FA default.
    if attn_implementation != "eager" and _model_is_deepseek_v4(model_config):
        logger.warning(
            f"DeepSeek-V4 supports only eager attention (head_dim=512 exceeds FA's 256 cap; SDPA/flex "
            f"cannot carry the sink column + compressor KV concat); overriding "
            f"attn_implementation='{attn_implementation}' to 'eager'."
        )
        attn_implementation = "eager"

    if attn_implementation.startswith("flash_attention") and model_is_gemma4(model_config):
        logger.warning(
            f"Gemma4's head_dim=512 attention is unsupported by {attn_implementation} "
            "(FA2 caps at 256; FA4 overflows tensor memory); falling back to attn_implementation='sdpa'."
        )
        attn_implementation = "sdpa"

    return validate_attn_implementation(model_config, attn_implementation, sinks_reset=sinks_reset)


def _disable_gpt_oss_fa_fallback(model_config: AutoConfig) -> None:
    """Stop transformers auto-swapping GptOss FA2 for the kernel-hub vllm-flash-attn3 package.

    That package ships only SM 9.0 binaries and crashes on B200/B300. Setting the attribute to ``None``
    runs local flash_attn FA2. Must be called BEFORE ``from_pretrained``.
    """
    if not model_has_sinks(model_config):
        return
    if GptOssPreTrainedModel is None:
        # Loud, not fatal: a moved class leaves the kernel-hub auto-fallback active, not gone.
        logger.warning(
            "Could not disable the GptOss flash-attn auto-fallback (GptOssPreTrainedModel is not "
            "importable from transformers). If transformers still auto-swaps FA2 for the kernel-hub "
            "vllm-flash-attn3 package, it will crash on Blackwell (SM 9.0-only binaries). Verify "
            "the installed transformers still needs this patch."
        )
        return
    if getattr(GptOssPreTrainedModel, "_compatible_flash_implementations", None) is not None:
        GptOssPreTrainedModel._compatible_flash_implementations = None
        logger.info(
            "Disabled GptOss flash-attn auto-fallback "
            "(use locally-installed flash_attention_2 directly, no kernel-hub swap)"
        )
