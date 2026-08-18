"""GptOss attention-sink policy: neutralized, live-frozen, or trainable.

GptOss appends a per-head learnable logit (``self_attn.sinks``) to every softmax. The state a run
trains under is decided once at load by :func:`apply_sinks_policy`, stamped on the model for the
trainer gates and the adapter's ``training_provenance.json``, and mirrored by the checkpoint writers
(:func:`neutralized_gpt_oss_sinks` re-emits what the flash_attention_2 reset removed). Which
implementations may run against sinks at all is decided in ``attention.py``.
"""

from __future__ import annotations

import contextlib
import enum
import functools

import torch
from accelerate.logging import get_logger
from transformers import AutoConfig, PreTrainedModel
from transformers import modeling_flash_attention_utils as flash_utils

from src.models.loading.config_levels import text_config
from src.models.patches.attention import model_has_sinks
from src.models.structure import backbone_with_layers

logger = get_logger(__name__)

# Where :func:`apply_sinks_policy` stamps the run's :class:`SinksPolicy`. A module attribute, not a
# config field: it must not serialize into a saved ``config.json``, and attribute lookup falls
# through PEFT/FSDP2 wrappers where a registry keyed on one module object would not.
LIVE_SINKS_ATTR = "_halo_live_attention_sinks"

# Implementations under which a trainable sink receives a gradient: FA4 through the rescale below,
# eager natively. FA2/SDPA drop the column, FA3's fused backward returns no sink gradient, and
# flex_attention's compiled backward NaNs on exactly that gradient.
_TRAINABLE_SINK_ATTN_IMPLEMENTATIONS = ("flash_attention_4", "eager")

# Marker on the wrapped ``flash_attn.cute`` entry points (idempotent install, test introspection).
_FA4_RESCALE_ATTR = "_halo_trainable_sink_rescale"


class SinksPolicy(enum.StrEnum):
    """The sink state a GptOss run trains under — one vocabulary for the trainer gates, the
    adapter's ``training_provenance.json`` record, and the merge tools that re-apply it.
    ``str``-valued so it serializes as plain JSON and compares against a loaded record directly."""

    NEUTRALIZED = "neutralized"
    LIVE = "live"
    TRAINABLE = "trainable"

    @classmethod
    def from_flags(cls, *, reset_sinks: bool, train_sinks: bool = False) -> SinksPolicy:
        """Resolve the two user knobs (``reset_sinks``, ``train_sinks``) into one policy."""
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
        """Whether the sink column participates in the softmax — every gate keyed on live sinks
        (kernel capability, implicit-reference KL, RL weight sync) applies to LIVE and TRAINABLE alike."""
        return self is not SinksPolicy.NEUTRALIZED


def is_sink_key(key: str) -> bool:
    """Whether a parameter / state-dict name refers to an attention-sink tensor.

    One spelling behind every scan of a checkpoint's keys — a drifted copy verifies tensors it never
    touched."""
    return ".sinks" in key


def neutralized_sink_value(dtype: torch.dtype) -> float:
    """The value a neutralized sink of ``dtype`` holds: the most negative finite one, which zeroes
    its softmax term. One spelling behind the reset, the save-time re-emission and every check that
    a written checkpoint's sinks really are off."""
    return torch.finfo(dtype).min


def _gpt_oss_sink_attentions(model, model_config=None):
    """Yield ``(layer_index, self_attn)`` for each decoder layer of a sinks (GptOss) model.

    The shared preamble of every sinks patch. ``model_config`` defaults to ``model.config`` (the
    reference-model path holds a config loaded separately from the weights); a non-sinks model or an
    unrecognized layout yields nothing, warned. Always yields the module that OWNS the ``sinks``
    parameter — after CP patching ``self_attn`` is the Ulysses wrapper, and the FA2 rebind
    (``attn.sinks = None``) must land where the save paths read.
    """
    config = model_config if model_config is not None else getattr(model, "config", None)
    if config is None or not model_has_sinks(config):
        return
    backbone = backbone_with_layers(model)
    if backbone is None:
        logger.warning("Could not find model.layers for GptOss attention sinks — skipping")
        return
    for index, layer in enumerate(backbone.layers):
        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            yield index, getattr(attn, "original_attention", attn)


def neutralized_gpt_oss_sinks(model) -> dict:
    """Reconstruct GptOss attention sinks dropped during flash_attention_2 fine-tuning.

    Under FA2 ``sinks`` is set to None (leaves ``named_parameters()``), so an EP/TP save would omit it.
    Returns ``{param_name: tensor}`` of sinks filled with dtype min; empty when sinks are still present.
    """
    config = getattr(model, "config", None)
    # Family gate BEFORE the head-count read: on a per-layer-heterogeneous config (step3p7's 64
    # full / 96 sliding heads) the bare attribute raises transformers'
    # AmbiguousGlobalPerLayerAttributeError, and this runs on every gathered save and weight sync.
    if config is None or not model_has_sinks(config):
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
        # Named in the CHECKPOINT's key space (a *ForCausalLM's backbone is ``model``), not the
        # caller's tree: these names are merged into a state dict that the wrappers do not appear in.
        sinks[f"model.layers.{index}.self_attn.sinks"] = torch.full(
            (num_heads,), neutralized_sink_value(dtype), dtype=dtype, device="cpu"
        )
    return sinks


@contextlib.contextmanager
def gpt_oss_sinks_restored(model):
    """Temporarily re-attach neutralized sinks so a plain ``save_pretrained`` writes them.

    The parallel save paths re-emit them into the gathered state dict themselves; the base Trainer
    save has no such seam, so an unparallelized FA2 run would write a checkpoint with no
    ``self_attn.sinks`` and reload it as an uninitialized Parameter. Restores ``None`` on exit — FA2's
    kernel rejects a non-None ``s_aux``.
    """
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


def _reset_gpt_oss_sinks(
    model: PreTrainedModel,
    model_config: AutoConfig,
    attn_implementation: str | None = None,
) -> int:
    """Neutralize GptOss per-head learnable attention sinks for fine-tuning (both paths contribute ~0):

    * ``flash_attention_2``: set ``sinks = None`` (FA2's s_aux path rejects any non-None tensor on Blackwell).
    * else: fill with ``dtype.min`` and freeze — eager/flex read the param's shape and would crash on
      ``None``; a dtype-min sink has zero softmax mass and zero gradient, so leaving it trainable only
      buys optimizer state and, under FA4, a rescale pass that gates by exactly 1.

    Returns the number of layers reset. Save-time inverse: :func:`neutralized_gpt_oss_sinks`.
    """
    use_none = attn_implementation == "flash_attention_2"
    reset_count = 0
    with torch.no_grad():
        for _, attn in _gpt_oss_sink_attentions(model, model_config):
            if not hasattr(attn, "sinks"):
                # Assignment would CREATE the attribute on a variant that has no sinks and report a
                # neutralization that never happened (the run then stamps NEUTRALIZED having
                # disabled nothing). The fill branch already fails here; fail identically.
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
    """Keep GptOss attention sinks at their pretrained values, frozen or trainable.

    Frozen is the RL default: the trainer then scores with the same sinks the rollout engine serves.
    Returns the number of layers touched, which is what makes the run "live-sinks" (zero on any model
    without sinks, whatever the requested policy).
    """
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
    """Route FA4 calls with a grad-requiring sink through a sink-gradient-carrying rescale.

    The FA4 backward returns only ``dq, dk, dv``, so a trainable sink under the fused kernel silently
    does not train. It does consume ``dlse``, which permits an exact reformulation outside it: with
    ``Z = e^lse`` (sink-less normalization) the sinked output is ``out * sigmoid(lse - sink)``, so
    calling sink-less with ``return_lse=True`` and applying that gate in autograd-visible ops yields
    the identical forward and complete gradients. Frozen or absent sinks and no-grad calls pass
    through untouched.

    Both entry points are wrapped, plus transformers' cached references to them (its lazy dispatch
    reads the symbols once and keeps them in its own globals). The trainable branch returns
    ``(out, lse)`` with the sinked lse — the arity the pinned build returns at every ``return_lse``.
    Idempotent. False when ``flash_attn.cute`` is not importable.
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
        # ``lse`` is fp32; the gate stays fp32 and multiplies the bf16 ``out`` under type promotion,
        # so autograd keeps the kernel's own bf16 output plus a [heads, tokens] gate — not an fp32
        # copy of the attention output. Returned lse is the sinked one, as the fused kernel reports.
        sink = sink.float().reshape([-1 if d == head_dim else 1 for d in range(lse.dim())])
        gate = torch.sigmoid(lse - sink)
        out = (out * gate.movedim(head_dim, -1).unsqueeze(-1)).to(out.dtype)
        return out, torch.logaddexp(lse, sink)

    @functools.wraps(orig_varlen)
    def varlen_with_sink_grad(q, k, v, *args, learnable_sink=None, return_lse=False, **kwargs):
        if not sink_trains(learnable_sink):
            return orig_varlen(q, k, v, *args, learnable_sink=learnable_sink, return_lse=return_lse, **kwargs)
        # return_lse=True is what makes the kernel's backward consume dlse; the public API returns
        # (out, lse) either way. Varlen lse is [heads, total_tokens], out is [total_tokens, heads, dim].
        out, lse = orig_varlen(q, k, v, *args, learnable_sink=None, return_lse=True, **kwargs)
        return rescale(out, lse, learnable_sink, head_dim=0)

    @functools.wraps(orig_dense)
    def dense_with_sink_grad(q, k, v, *args, learnable_sink=None, return_lse=False, **kwargs):
        if not sink_trains(learnable_sink):
            return orig_dense(q, k, v, *args, learnable_sink=learnable_sink, return_lse=return_lse, **kwargs)
        # Dense lse is [batch, heads, seqlen], out is [batch, seqlen, heads, dim].
        out, lse = orig_dense(q, k, v, *args, learnable_sink=None, return_lse=True, **kwargs)
        return rescale(out, lse, learnable_sink, head_dim=1)

    for fn in (varlen_with_sink_grad, dense_with_sink_grad):
        setattr(fn, _FA4_RESCALE_ATTR, True)
    cute.flash_attn_varlen_func = varlen_with_sink_grad
    cute.flash_attn_func = dense_with_sink_grad
    if getattr(flash_utils, "_flash_varlen_fn", None) is orig_varlen:
        flash_utils._flash_varlen_fn = varlen_with_sink_grad
    if getattr(flash_utils, "_flash_fn", None) is orig_dense:
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
    """Apply the run's attention-sink policy to a model that has just been loaded.

    Every loader routes through here rather than reimplementing the branches, because
    ``sinks_reset=True`` is what APPROVES a sink-dropping backend: resolving the implementation under
    that premise and then skipping the reset leaves a GptOss model running sdpa over live sinks,
    shifting every logprob by nats with no other symptom. A model without sinks is left unstamped —
    the stamp doubles as the family signal for the provenance record.
    """
    has_sinks = model_has_sinks(model_config)
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
    # Installed for every FA4 sinks run, not only TRAINABLE: it is inert for frozen sinks, and a sink
    # unfrozen by any other route (``unfreeze_layers_patterns``) then trains instead of silently
    # receiving no gradient from the fused kernel.
    fa4 = attn_implementation == "flash_attention_4"
    if fa4 and not install_fa4_trainable_sink_rescale() and policy is SinksPolicy.TRAINABLE:
        raise RuntimeError(
            "train_sinks: true under flash_attention_4 needs the sink-gradient rescale, but "
            "flash_attn.cute is not importable — the fused kernel alone would train nothing into the sinks."
        )
    if touched == 0:
        # The walk found no sink layers on a sinks-carrying model (unrecognized layout) — the
        # requested policy was NOT applied. Stamping anyway would silence the RL sink gate and
        # record a false provenance that a later merge re-applies onto a live-sink artifact.
        raise RuntimeError(
            f"Sink policy '{policy}' touched no attention layers on a sinks-carrying model "
            f"(model_type={getattr(model_config, 'model_type', '?')!r}). The GptOss layer walk "
            "failed — fix the layout resolution instead of training with an unapplied sink policy."
        )
    setattr(model, LIVE_SINKS_ATTR, policy)


def stamped_sinks_policy(model) -> SinksPolicy | None:
    """The policy :func:`apply_sinks_policy` stamped on this model, or ``None`` when none ran or the
    model carries no sinks — the distinction the provenance writer needs, since guessing
    "neutralized" would neutralize a live-sink artifact.

    Walks ``.module`` wrapper chains explicitly: the stamp is a plain instance attribute, and a
    wrapper that does not delegate those would read as "no sinks" and silence the RL sink gate.
    """
    seen: set[int] = set()
    node = model
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        value = getattr(node, LIVE_SINKS_ATTR, None)
        if isinstance(value, SinksPolicy):
            return value
        node = getattr(node, "module", None)
    return None


def has_live_attention_sinks(model) -> bool:
    """Whether this model runs with its pretrained attention sinks live (``reset_sinks: false``,
    frozen or trainable).

    Read off the stamp, not the tensors: the reset yields ``sinks is None`` only under
    flash_attention_2 and fills with ``dtype.min`` everywhere else, so a presence test reports reset
    sinks as live on the production Blackwell path — and by the time a trainer validates, ``sinks``
    is a sharded DTensor whose per-rank values would diverge.
    """
    policy = stamped_sinks_policy(model)
    return policy is not None and policy.live
