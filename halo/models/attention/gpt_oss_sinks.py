"""GptOss attention-sink policy: neutralized, live-frozen, or trainable. GptOss appends a per-head learnable logit
(``self_attn.sinks``) to every softmax; :func:`apply_sinks_policy` applies the run's policy once at load."""

from __future__ import annotations

import contextlib
import enum
import functools
import logging

import torch
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssPreTrainedModel

from halo.models.attention.fa4_entry_points import install_fa4_entry_point_wrapper
from halo.models.config_access import text_config
from halo.models.model_classes import mapped_model_class
from halo.models.module_tree import backbone_with_layers
from halo.models.moe.family_specs import GPT_OSS, find_moe_family_spec_for_config
from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)

# A module attribute, so it never serializes into a ``config.json`` and lookup falls through PEFT/FSDP2 wrappers.
SINKS_POLICY_ATTR = "_halo_live_attention_sinks"

# The sink-reset verdict the attention validator records, run-scoped, on the config instance.
RESOLVED_SINKS_RESET_ATTR = "_halo_sinks_reset"

# FA4 through the rescale below, eager natively. FA3's fused backward returns no sink gradient, and flex_attention's
# compiled backward NaNs on it.
_TRAINABLE_SINK_ATTN_IMPLEMENTATIONS = ("flash_attention_4", "eager")

# ``_attn_impl_handles_sinks`` decides the set.
SINK_CARRYING_ATTN_HINT = "flash_attention_4 on Blackwell; flex_attention or eager"
# Why a GptOss teacher must keep its sinks: unlike a student, a frozen teacher never trains back from neutralized ones.
NEUTRALIZED_TEACHER_SINKS_COST = (
    "a frozen pretrained teacher without its sinks predicts far from itself (~6 nats/token KL on gpt-oss-20b)"
)

_FA4_RESCALE_ATTR = "_halo_trainable_sink_rescale"

# A fraction of the dtype's minimum, as a bf16 dtype-min sink loaded at fp32 sits just above fp32's own minimum.
_NEUTRALIZED_SINK_FLOOR = 0.99


class SinksPolicy(enum.StrEnum):
    """Shared by the trainer gates, the adapter's ``training_provenance.json`` and the merge tools."""

    NEUTRALIZED = "neutralized"
    LIVE = "live"
    TRAINABLE = "trainable"

    @classmethod
    def from_flags(cls, *, reset_sinks: bool, train_sinks: bool = False) -> SinksPolicy:
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
        return self is not SinksPolicy.NEUTRALIZED


def is_sink_key(key: str) -> bool:
    return ".sinks" in key


def neutralized_sink_value(dtype: torch.dtype) -> float:
    """The most negative finite value, which zeroes the sink's softmax term."""
    return torch.finfo(dtype).min


def uses_gpt_oss_sinks(model_config) -> bool:
    """DeepSeek-V4 carries sinks too but runs eager, which applies them; widening this would reset its sinks."""
    return find_moe_family_spec_for_config(model_config) is GPT_OSS


def install_sdpa_for_reset_sinks(model_config) -> bool:
    """Valid only once the sinks are reset, as the column SDPA drops then contributes 0."""
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
    if flipped and is_global_main_process():
        logger.info(f"Enabled SDPA for {model_cls.__name__} (sinks reset for fine-tuning → sink column is a no-op)")
    return flipped


def ensure_gpt_oss_local_flash() -> None:
    """transformers swaps a label its class's compatible list does not name (FA2, FA3) for the list's first entry, the
    kernel-hub vllm-flash-attn3: SM 9.0 binaries that crash on Blackwell, through a package the image lacks."""
    GptOssPreTrainedModel._compatible_flash_implementations = None


# At import, as transformers decides the swap in each model's __init__, so a build no validator ran for would take it.
ensure_gpt_oss_local_flash()


def _gpt_oss_sink_attentions(model, model_config=None):
    """Yields the module owning ``sinks``, under the CP wrapper too, so the FA2 rebind lands where saves read."""
    config = model_config if model_config is not None else getattr(model, "config", None)
    if config is None or not uses_gpt_oss_sinks(config):
        return
    backbone = backbone_with_layers(model)
    if backbone is None:
        if is_global_main_process():
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
    """``flash_attention_2`` gets ``sinks = None``, as its s_aux path rejects any tensor; every other backend reads the
    parameter's shape, so it is filled with ``dtype.min`` and frozen."""
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
        if is_global_main_process():
            logger.info(f"Disabled attention sinks in {reset_count} GptOss layers ({mode})")
    return reset_count


def _set_gpt_oss_sinks_trainable(model: PreTrainedModel, model_config: AutoConfig, *, trainable: bool) -> int:
    touched = 0
    for _, attn in _gpt_oss_sink_attentions(model, model_config):
        sinks = getattr(attn, "sinks", None)
        if isinstance(sinks, torch.nn.Parameter):
            sinks.requires_grad_(trainable)
            touched += 1
    if touched:
        state = "TRAINABLE" if trainable else "frozen"
        if is_global_main_process():
            logger.info(f"Kept pretrained attention sinks live and {state} in {touched} GptOss layers")
    return touched


def _sink_trains(sink) -> bool:
    return sink is not None and torch.is_grad_enabled() and sink.requires_grad


def _rescale(out, lse, sink, head_dim: int):
    # The fp32 gate multiplies the bf16 ``out`` under promotion, so autograd saves the kernel's own
    # output plus a [heads, tokens] gate rather than an fp32 copy. The lse returned is the sinked one.
    sink = sink.float().reshape([-1 if d == head_dim else 1 for d in range(lse.dim())])
    gate = torch.sigmoid(lse - sink)
    out = (out * gate.movedim(head_dim, -1).unsqueeze(-1)).to(out.dtype)
    return out, torch.logaddexp(lse, sink)


def _with_sink_grad(fn, varlen: bool):
    # lse is [heads, tokens] beside a varlen out [tokens, heads, dim], [batch, heads, seq] beside a dense one.
    lse_head_dim = 0 if varlen else 1

    @functools.wraps(fn)
    def wrapped(q, k, v, *args, learnable_sink=None, return_lse=False, **kwargs):
        if not _sink_trains(learnable_sink):
            return fn(q, k, v, *args, learnable_sink=learnable_sink, return_lse=return_lse, **kwargs)
        # return_lse=True makes the backward consume dlse.
        out, lse = fn(q, k, v, *args, learnable_sink=None, return_lse=True, **kwargs)
        return _rescale(out, lse, learnable_sink, head_dim=lse_head_dim)

    return wrapped


def install_fa4_trainable_sink_rescale() -> bool:
    """The sink-less kernel's output gated by ``sigmoid(lse - sink)`` gives the fused forward and complete gradients:
    the sink's from the gate, dq/dk/dv through ``dout`` and ``dlse``. The kernel's sink backward is unvalidated."""
    return install_fa4_entry_point_wrapper(
        _FA4_RESCALE_ATTR,
        _with_sink_grad,
        "the trainable-sink rescale (sink-less kernel + sigmoid(lse - sink) gate)",
    )


def reject_trainable_sinks_without_sinks(model_config: AutoConfig, policy: SinksPolicy) -> None:
    if policy is SinksPolicy.TRAINABLE and not uses_gpt_oss_sinks(model_config):
        raise ValueError(
            f"train_sinks: true trains GptOss attention sinks, but this model "
            f"(model_type={getattr(model_config, 'model_type', '?')!r}) has none for it to train. Remove train_sinks."
        )


def apply_sinks_policy(
    model: PreTrainedModel,
    model_config: AutoConfig,
    *,
    policy: SinksPolicy,
    attn_implementation: str | None = None,
) -> None:
    """Every loader routes through here, as a neutralized policy is what approved a sink-dropping backend and skipping
    the reset would run sdpa over live sinks. A model without sinks stays unstamped."""
    reject_trainable_sinks_without_sinks(model_config, policy)
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
    # On every FA4 sinks run, as it is inert on frozen sinks and a sink unfrozen by another route still trains.
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
    """Walks ``.module`` wrapper chains, since a wrapper that does not delegate attributes would hide the stamp."""
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
    """Read off the stamp, as a neutralized sink stays a tensor off FA2, by validation time a sharded DTensor. An
    unstamped sinks model (one TRL built) kept its pretrained sinks live."""
    policy = stamped_sinks_policy(model)
    if policy is not None:
        return policy.live
    config = getattr(model, "config", None)
    return config is not None and uses_gpt_oss_sinks(config)


def holds_unneutralized_sinks(model) -> bool:
    """Read off the tensors, so a model loaded without a policy answers with the sinks its checkpoint carries."""
    for _, attn in _gpt_oss_sink_attentions(model):
        sinks = getattr(attn, "sinks", None)
        if sinks is not None and bool((sinks.detach() > _NEUTRALIZED_SINK_FLOOR * torch.finfo(sinks.dtype).min).any()):
            return True
    return False


def neutralized_gpt_oss_sinks(model) -> dict:
    """``{checkpoint_key: neutralized sink}`` for each sink the FA2 reset set to ``None``, absent from every save."""
    config = getattr(model, "config", None)
    # Family gate first, as a per-layer-heterogeneous config (step3p7) raises on the bare head count.
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
    """Re-attach neutralized sinks for a plain ``save_pretrained``, restoring ``None`` as FA2 rejects a tensor."""
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
