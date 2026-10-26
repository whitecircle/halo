"""GptOss attention-sink policy: neutralized, live-frozen, or trainable.

GptOss appends a per-head learnable logit (``self_attn.sinks``) to every softmax. The run's policy is
applied once at load by :func:`apply_sinks_policy`, stamped on the model for the trainer gates and the
adapter's provenance, and mirrored by the checkpoint writers (:func:`neutralized_gpt_oss_sinks`).
"""

from __future__ import annotations

import contextlib
import enum
import functools

import torch
from accelerate.logging import get_logger
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers import modeling_flash_attention_utils as flash_utils
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssPreTrainedModel

from src.models.config_access import model_type_matches, text_config
from src.models.model_classes import mapped_model_class
from src.models.module_tree import backbone_with_layers

logger = get_logger(__name__)

# Where :func:`apply_sinks_policy` stamps the run's :class:`SinksPolicy`: a module attribute, so it never
# serializes into a ``config.json`` and attribute lookup falls through PEFT/FSDP2 wrappers.
SINKS_POLICY_ATTR = "_halo_live_attention_sinks"

# The sink-reset verdict the attention validator records, run-scoped, on the config instance.
RESOLVED_SINKS_RESET_ATTR = "_halo_sinks_reset"

# Implementations under which a trainable sink receives a gradient: FA4 through the rescale below,
# eager natively. FA3's fused backward returns none, and flex_attention's compiled backward NaNs on it.
_TRAINABLE_SINK_ATTN_IMPLEMENTATIONS = ("flash_attention_4", "eager")

# Marker on the wrapped ``flash_attn.cute`` entry points.
_FA4_RESCALE_ATTR = "_halo_trainable_sink_rescale"


class SinksPolicy(enum.StrEnum):
    """The sink state a GptOss run trains under, shared by the trainer gates, the adapter's
    ``training_provenance.json`` and the merge tools; ``str``-valued so it serializes as plain JSON."""

    NEUTRALIZED = "neutralized"
    LIVE = "live"
    TRAINABLE = "trainable"

    @classmethod
    def from_flags(cls, *, reset_sinks: bool, train_sinks: bool = False) -> SinksPolicy:
        """The policy the ``reset_sinks`` and ``train_sinks`` knobs select."""
        if reset_sinks and train_sinks:
            raise ValueError(
                "train_sinks: true contradicts reset_sinks: true — a dtype-min sink has zero softmax mass "
                "and zero gradient, so nothing could train. Set reset_sinks: false alongside train_sinks."
            )
        if reset_sinks:
            return cls.NEUTRALIZED
        return cls.TRAINABLE if train_sinks else cls.LIVE

    @property
    def live(self) -> bool:
        """Whether the sink column participates in the softmax (LIVE and TRAINABLE alike)."""
        return self is not SinksPolicy.NEUTRALIZED


def is_sink_key(key: str) -> bool:
    """Whether a parameter / state-dict name refers to an attention-sink tensor."""
    return ".sinks" in key


def neutralized_sink_value(dtype: torch.dtype) -> float:
    """The value a neutralized sink of ``dtype`` holds: the most negative finite one, which zeroes its
    softmax term."""
    return torch.finfo(dtype).min


def uses_gpt_oss_sinks(model_config) -> bool:
    """Whether this architecture's sinks constrain the attention implementation (GptOss).

    Narrower than "has attention sinks": DeepSeek-V4 carries them too but runs eager, which applies the
    sink column itself; widening this would reset its pretrained sinks.
    """
    return model_type_matches(model_config, "gpt_oss")


def install_sdpa_for_reset_sinks(model_config) -> bool:
    """Flip GptOss's class-level ``_supports_sdpa = False``, valid only once its sinks are reset (the
    column SDPA drops then contributes 0). Returns True if flipped."""
    if not uses_gpt_oss_sinks(model_config):
        return False
    model_cls = mapped_model_class(model_config, AutoModelForCausalLM)
    if model_cls is None:
        return False
    flipped = False
    for cls in model_cls.__mro__:
        if cls is PreTrainedModel:
            # PreTrainedModel's own default would admit sdpa for every architecture loaded later.
            break
        if cls.__dict__.get("_supports_sdpa") is False:
            cls._supports_sdpa = True
            flipped = True
    if flipped:
        logger.info(f"Enabled SDPA for {model_cls.__name__} (sinks reset for fine-tuning → sink column is a no-op)")
    return flipped


def install_local_flash_for_sinks(model_config: AutoConfig) -> None:
    """Keep the local flash build the validator chose for a GptOss model; call before ``from_pretrained``.

    transformers would swap it for the class's compatible-list entry, the kernel-hub vllm-flash-attn3:
    SM 9.0 binaries that crash on Blackwell, a download an offline run cannot make, and not the build
    the sink checks validated.
    """
    if not uses_gpt_oss_sinks(model_config):
        return
    if GptOssPreTrainedModel._compatible_flash_implementations is not None:
        GptOssPreTrainedModel._compatible_flash_implementations = None
        logger.info(
            "Disabled GptOss flash-attn auto-fallback "
            "(use locally-installed flash_attention_2 directly, no kernel-hub swap)"
        )


def _gpt_oss_sink_attentions(model, model_config=None):
    """Yield ``(layer_index, attention)`` for each decoder layer of a GptOss model; nothing otherwise.

    Yields the module that owns ``sinks``, under the CP wrapper too, so the FA2 rebind lands where the
    save paths read. ``model_config`` overrides ``model.config``.
    """
    config = model_config if model_config is not None else getattr(model, "config", None)
    if config is None or not uses_gpt_oss_sinks(config):
        return
    backbone = backbone_with_layers(model)
    if backbone is None:
        logger.warning("Could not find model.layers for GptOss attention sinks — skipping")
        return
    for index, layer in enumerate(backbone.layers):
        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            yield index, getattr(attn, "original_attention", attn)


def _reset_gpt_oss_sinks(
    model: PreTrainedModel,
    model_config: AutoConfig,
    attn_implementation: str | None = None,
) -> int:
    """Neutralize GptOss sinks; returns the number of layers reset.

    ``flash_attention_2`` gets ``sinks = None`` (its s_aux path rejects any tensor); every other backend
    reads the parameter's shape, so it is filled with ``dtype.min`` and frozen (it has no gradient left).
    """
    use_none = attn_implementation == "flash_attention_2"
    reset_count = 0
    with torch.no_grad():
        for _, attn in _gpt_oss_sink_attentions(model, model_config):
            if not hasattr(attn, "sinks"):
                # Assigning would create the attribute and report a neutralization that never happened.
                raise AttributeError(
                    f"{type(attn).__name__} carries no 'sinks' attribute, so no sink policy applies "
                    f"to it — this model's attention does not implement GptOss attention sinks."
                )
            if use_none:
                attn.sinks = None
            else:
                attn.sinks.fill_(neutralized_sink_value(attn.sinks.dtype))
                attn.sinks.requires_grad_(False)
            reset_count += 1

    if reset_count > 0:
        mode = "set to None (FA2)" if use_none else "filled with dtype min, frozen"
        logger.info(f"Disabled attention sinks in {reset_count} GptOss layers ({mode})")
    return reset_count


def _set_gpt_oss_sinks_trainable(model: PreTrainedModel, model_config: AutoConfig, *, trainable: bool) -> int:
    """Keep GptOss sinks at their pretrained values, frozen or trainable; returns the layers touched."""
    touched = 0
    for _, attn in _gpt_oss_sink_attentions(model, model_config):
        sinks = getattr(attn, "sinks", None)
        if isinstance(sinks, torch.nn.Parameter):
            sinks.requires_grad_(trainable)
            touched += 1
    if touched:
        state = "TRAINABLE" if trainable else "frozen"
        logger.info(f"Kept pretrained attention sinks live and {state} in {touched} GptOss layers")
    return touched


def install_fa4_trainable_sink_rescale() -> bool:
    """Route FA4 calls with a grad-requiring sink through a rescale that carries the sink gradient.

    The fused backward returns no sink gradient but consumes ``dlse``, so the sink-less kernel's output
    gated by ``sigmoid(lse - sink)`` in autograd-visible ops gives the identical forward and complete
    gradients. Wraps both entry points and transformers' cached references. Idempotent; False when
    ``flash_attn.cute`` is not importable.
    """
    try:
        import flash_attn.cute as cute  # noqa: PLC0415 — optional, Blackwell-only dependency
    except ImportError:
        return False
    if getattr(cute.flash_attn_varlen_func, _FA4_RESCALE_ATTR, False):
        return True
    orig_varlen, orig_dense = cute.flash_attn_varlen_func, cute.flash_attn_func

    def sink_trains(sink) -> bool:
        return sink is not None and torch.is_grad_enabled() and sink.requires_grad

    def rescale(out, lse, sink, head_dim: int):
        # The fp32 gate multiplies the bf16 ``out`` under promotion, so autograd saves the kernel's own
        # output plus a [heads, tokens] gate rather than an fp32 copy. The lse returned is the sinked one.
        sink = sink.float().reshape([-1 if d == head_dim else 1 for d in range(lse.dim())])
        gate = torch.sigmoid(lse - sink)
        out = (out * gate.movedim(head_dim, -1).unsqueeze(-1)).to(out.dtype)
        return out, torch.logaddexp(lse, sink)

    @functools.wraps(orig_varlen)
    def varlen_with_sink_grad(q, k, v, *args, learnable_sink=None, return_lse=False, **kwargs):
        if not sink_trains(learnable_sink):
            return orig_varlen(q, k, v, *args, learnable_sink=learnable_sink, return_lse=return_lse, **kwargs)
        # return_lse=True makes the backward consume dlse. lse is [heads, tokens], out [tokens, heads, dim].
        out, lse = orig_varlen(q, k, v, *args, learnable_sink=None, return_lse=True, **kwargs)
        return rescale(out, lse, learnable_sink, head_dim=0)

    @functools.wraps(orig_dense)
    def dense_with_sink_grad(q, k, v, *args, learnable_sink=None, return_lse=False, **kwargs):
        if not sink_trains(learnable_sink):
            return orig_dense(q, k, v, *args, learnable_sink=learnable_sink, return_lse=return_lse, **kwargs)
        # lse is [batch, heads, seq], out [batch, seq, heads, dim].
        out, lse = orig_dense(q, k, v, *args, learnable_sink=None, return_lse=True, **kwargs)
        return rescale(out, lse, learnable_sink, head_dim=1)

    for fn in (varlen_with_sink_grad, dense_with_sink_grad):
        setattr(fn, _FA4_RESCALE_ATTR, True)
    cute.flash_attn_varlen_func = varlen_with_sink_grad
    cute.flash_attn_func = dense_with_sink_grad
    if flash_utils._flash_varlen_fn is orig_varlen:
        flash_utils._flash_varlen_fn = varlen_with_sink_grad
    if flash_utils._flash_fn is orig_dense:
        flash_utils._flash_fn = dense_with_sink_grad
    logger.info("Installed the FA4 trainable-sink rescale (sink-less kernel + sigmoid(lse - sink) gate)")
    return True


def apply_sinks_policy(
    model: PreTrainedModel,
    model_config: AutoConfig,
    *,
    policy: SinksPolicy,
    attn_implementation: str | None = None,
) -> None:
    """Apply the run's sink policy to a freshly loaded model and stamp it there.

    Every loader routes through here: a neutralized policy is what approved a sink-dropping backend,
    so skipping the reset would run sdpa over live sinks. A model without sinks stays unstamped.
    """
    has_sinks = uses_gpt_oss_sinks(model_config)
    if (
        has_sinks
        and policy is SinksPolicy.TRAINABLE
        and attn_implementation not in _TRAINABLE_SINK_ATTN_IMPLEMENTATIONS
    ):
        raise ValueError(
            f"train_sinks: true needs an attention implementation that yields a sink gradient — "
            f"{_TRAINABLE_SINK_ATTN_IMPLEMENTATIONS} — but this run resolved to "
            f"{attn_implementation!r}. flex_attention NaNs on the sink gradient in its compiled backward "
            f"and the FA3 s_aux build returns none, so the sinks would silently not train."
        )
    if policy is SinksPolicy.NEUTRALIZED:
        touched = _reset_gpt_oss_sinks(model, model_config, attn_implementation=attn_implementation)
    else:
        touched = _set_gpt_oss_sinks_trainable(model, model_config, trainable=policy is SinksPolicy.TRAINABLE)
    if not has_sinks:
        return
    # Every FA4 sinks run: inert on frozen sinks, and a sink unfrozen by another route still trains.
    fa4 = attn_implementation == "flash_attention_4"
    if fa4 and not install_fa4_trainable_sink_rescale() and policy is SinksPolicy.TRAINABLE:
        raise RuntimeError(
            "train_sinks: true under flash_attention_4 needs the sink-gradient rescale, but "
            "flash_attn.cute is not importable — the fused kernel alone would train nothing into the sinks."
        )
    if touched == 0:
        # Stamping an unapplied policy would silence the RL sink gate and record a false provenance.
        raise RuntimeError(
            f"Sink policy '{policy}' touched no attention layers on a sinks-carrying model "
            f"(model_type={getattr(model_config, 'model_type', '?')!r}). The GptOss layer walk "
            "failed — fix the layout resolution instead of training with an unapplied sink policy."
        )
    setattr(model, SINKS_POLICY_ATTR, policy)


def stamped_sinks_policy(model) -> SinksPolicy | None:
    """The policy :func:`apply_sinks_policy` stamped on this model, or ``None`` when none ran. Walks
    ``.module`` wrapper chains, since a wrapper that does not delegate attributes would hide the stamp."""
    seen: set[int] = set()
    node = model
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        value = getattr(node, SINKS_POLICY_ATTR, None)
        if isinstance(value, SinksPolicy):
            return value
        node = getattr(node, "module", None)
    return None


def has_live_attention_sinks(model) -> bool:
    """Whether this model runs its pretrained attention sinks live (frozen or trainable).

    Read off the stamp: a neutralized sink stays a tensor off FA2, and by validation time it is a
    sharded DTensor. An unstamped sinks model (one TRL built) kept its pretrained sinks live.
    """
    policy = stamped_sinks_policy(model)
    if policy is not None:
        return policy.live
    config = getattr(model, "config", None)
    return config is not None and uses_gpt_oss_sinks(config)


def neutralized_gpt_oss_sinks(model) -> dict:
    """``{checkpoint_key: neutralized sink}`` for each GptOss sink the flash_attention_2 reset set to
    ``None``, which would otherwise leave every save; empty when the sinks are present."""
    config = getattr(model, "config", None)
    # Family gate first: a per-layer-heterogeneous config (step3p7) raises on the bare head count.
    if config is None or not uses_gpt_oss_sinks(config):
        return {}
    num_heads = int(getattr(text_config(config), "num_attention_heads", 0) or 0)
    if num_heads <= 0:
        return {}

    sinks = {}
    for index, attn in _gpt_oss_sink_attentions(model, config):
        if attn.sinks is not None:
            continue  # only sinks absent-by-None; present tensors saved by the param loop
        ref = getattr(getattr(attn, "q_proj", None), "weight", None)
        dtype = ref.dtype if ref is not None else torch.bfloat16
        # Checkpoint key space, not the caller's wrapped tree.
        sinks[f"model.layers.{index}.self_attn.sinks"] = torch.full(
            (num_heads,), neutralized_sink_value(dtype), dtype=dtype, device="cpu"
        )
    return sinks


@contextlib.contextmanager
def gpt_oss_sinks_restored(model):
    """Temporarily re-attach neutralized sinks so a plain ``save_pretrained`` writes them, restoring
    ``None`` on exit (FA2 rejects a tensor). The parallel savers re-emit them themselves."""
    restored = neutralized_gpt_oss_sinks(model)
    touched = []
    if restored:
        attentions = dict(_gpt_oss_sink_attentions(model))
        for name, tensor in restored.items():
            attn = attentions[int(name.split(".")[2])]
            attn.sinks = torch.nn.Parameter(tensor.clone(), requires_grad=False)
            touched.append(attn)
    try:
        yield
    finally:
        for attn in touched:
            attn.sinks = None
