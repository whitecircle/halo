"""Tree attention: every token of a tree-packed row sees exactly its own root-to-leaf path. Each sequence mixer of
an admitted family reads its part of the collator's plan (:mod:`halo.models.attention.tree_plan`): softmax
attention at the flash interface, in the gather or split form; a GatedDeltaNet layer at its causal convolution
and its delta-rule scan. Only backends whose kernels are pinned against every example run alone are admitted."""

import copy
import functools
import logging
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import numpy as np
import torch
from transformers import modeling_flash_attention_utils as flash_utils
from transformers.integrations.flash_attention import get_target_dtype
from transformers.models.gpt_oss import modeling_gpt_oss
from transformers.models.qwen3 import modeling_qwen3
from transformers.models.qwen3_5 import modeling_qwen3_5
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe
from transformers.models.qwen3_moe import modeling_qwen3_moe

from halo.models.attention.backend_setup import WARMUP_MIN_SEQ_LEN
from halo.models.attention.gpt_oss_sinks import uses_gpt_oss_sinks
from halo.models.attention.layout import resolve_head_dim, resolve_num_key_value_heads
from halo.models.attention.packed_position_ids import FlashInterfaceViewBase
from halo.models.attention.selection import effective_attn_implementation
from halo.models.attention.tree_plan import (
    GATHER_KEYS_PER_CALL,
    TREE_ATTENTION_KEY,
    FullAttentionForm,
    NodeTable,
    build_tree_attention_plan,
)
from halo.models.config_access import text_config
from halo.models.module_tree import backbone_with_layers, decoder_layers
from halo.models.segment_markers import require_segment_aware_kernels
from halo.models.unwrapping import base_transformers_model
from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _StatefulMixer:
    """Where a linear-attention layer keeps state along the sequence, as its modeling module spells it.

    ``conv`` and ``recurrence`` are the module-level functions the causal convolution and the
    delta-rule scan run through, ``layer_attr`` the decoder-layer attribute holding the mixer and
    ``conv_kernel_field`` the config field with the convolution's width.
    """

    conv: str
    recurrence: str
    layer_attr: str
    conv_kernel_field: str


@dataclass(frozen=True)
class TreeFamily:
    """How one family reads a tree row: the modeling module whose attention registry takes the view,
    and the stateful mixer its linear-attention layers are, where it has any."""

    module: ModuleType
    stateful: _StatefulMixer | None = None


_GATED_DELTA_NET = _StatefulMixer(
    conv="causal_conv1d_fn",
    recurrence="torch_chunk_gated_delta_rule",
    layer_attr="linear_attn",
    conv_kernel_field="linear_conv_kernel_dim",
)

# Families whose forward is verified against every example run alone (the CPU tree-attention oracle sweeps
# this table), keyed by the decoder's exact model type.
TREE_FAMILIES = {
    "gpt_oss": TreeFamily(modeling_gpt_oss),
    "qwen3": TreeFamily(modeling_qwen3),
    "qwen3_moe": TreeFamily(modeling_qwen3_moe),
    "qwen3_5_text": TreeFamily(modeling_qwen3_5, _GATED_DELTA_NET),
    "qwen3_5_moe_text": TreeFamily(modeling_qwen3_5_moe, _GATED_DELTA_NET),
}

# The one backend that returns the log-sum-exp the split form reads, with a gradient.
_SPLIT_FORM_IMPLEMENTATION = "flash_attention_4"

# The backends tree attention runs on: their varlen kernels take more keys than queries per sequence
# and align the causal mask and the window bottom-right.
TREE_ATTN_IMPLEMENTATIONS = (_SPLIT_FORM_IMPLEMENTATION, "flash_attention_2")

# The arguments a delta-rule scan takes its state and segments through, and the flag that has it
# normalize query and key itself, with the epsilon it then uses.
_RECURRENCE_STATE_ARGS = ("initial_state", "output_final_state", "cu_seqlens")
_RECURRENCE_L2NORM_FLAG = "use_qk_l2norm_in_kernel"
_RECURRENCE_L2NORM_EPS = 1e-6

# Depth of the chains the warm-ups run. The split form needs a parent and a child; the recurrence
# a state going out, one coming in and going out, one coming in — each a separate kernel compile.
_SPLIT_WARMUP_DEPTH = 2
_RECURRENCE_WARMUP_DEPTH = 3


class _GatherTokens(torch.autograd.Function):
    """``index_select`` along the token axis whose backward sums a token's copies in fp32.

    A shared token is gathered once per node below it, so its gradient is a sum of that many terms;
    accumulated in bf16 by the stock backward it is percents off at a fan-out of fifteen.
    """

    @staticmethod
    def forward(ctx, states, index, dim):
        ctx.save_for_backward(index)
        ctx.dim, ctx.shape, ctx.dtype = dim, states.shape, states.dtype
        return states.index_select(dim, index)

    @staticmethod
    def backward(ctx, grad):
        (index,) = ctx.saved_tensors
        summed = torch.zeros(ctx.shape, dtype=torch.float32, device=grad.device)
        # Widened a call's worth at a time: the gathered gradient can be many times the row.
        for start in range(0, index.numel(), GATHER_KEYS_PER_CALL):
            piece = index[start : start + GATHER_KEYS_PER_CALL]
            summed.index_add_(ctx.dim, piece, grad.narrow(ctx.dim, start, piece.numel()).float())
        return summed.to(ctx.dtype), None, None


class _L2Norm(torch.autograd.Function):
    """``x * rsqrt(sum(x**2) + eps)`` over the last dim: the delta-rule kernels' own query/key
    normalization, at their precision (fp32 under bf16 states) and keeping what they keep for the
    backward, the output and the per-vector scale."""

    @staticmethod
    def forward(ctx, states, eps):
        wide = states.to(torch.promote_types(states.dtype, torch.float32))
        scale = torch.rsqrt(wide.square().sum(-1, keepdim=True) + eps)
        normalized = (wide * scale).to(states.dtype)
        ctx.save_for_backward(normalized, scale)
        return normalized

    @staticmethod
    def backward(ctx, grad):
        normalized, scale = ctx.saved_tensors
        wide, grad_wide = normalized.to(scale.dtype), grad.to(scale.dtype)
        tangent = grad_wide - wide * (grad_wide * wide).sum(-1, keepdim=True)
        return (tangent * scale).to(grad.dtype), None


class _MergeAncestors(torch.autograd.Function):
    """Fold the ancestor pass into the per-node pass for the tokens that have ancestors.

    A softmax over two disjoint key sets is the mix of the two partial softmaxes weighted by their
    normalizers: with ``w = sigmoid(lse_own - lse_anc)`` the merged output is
    ``w * out_own + (1 - w) * out_anc``. ``lse_own`` already holds the sink, so the sink needs no term
    of its own. The backward is the closed form of that mix, which spares autograd three full-size
    passes over the attention output.
    """

    @staticmethod
    def forward(ctx, out, lse, out_anc, lse_anc, index):
        # The mix runs at the normalizers' precision (fp32 under a bf16 kernel) and rounds once.
        own, out_anc = out.index_select(0, index).to(lse.dtype), out_anc.to(lse.dtype)
        weight = torch.sigmoid(lse.index_select(1, index) - lse_anc).transpose(0, 1).unsqueeze(-1)
        ctx.save_for_backward(index, weight, own - out_anc)
        ctx.lse_shape = lse.shape
        return out.index_copy(0, index, torch.lerp(out_anc, own, weight).to(out.dtype))

    @staticmethod
    def backward(ctx, grad):
        index, weight, difference = ctx.saved_tensors
        merged = grad.index_select(0, index).to(weight.dtype)
        d_out = grad.index_copy(0, index, (merged * weight).to(grad.dtype))
        d_out_anc = (merged * (1 - weight)).to(grad.dtype)
        d_lse_own = ((merged * difference).sum(-1) * (weight * (1 - weight)).squeeze(-1)).transpose(0, 1)
        d_lse = torch.zeros(ctx.lse_shape, dtype=weight.dtype, device=grad.device).index_copy(1, index, d_lse_own)
        return d_out, d_lse, d_out_anc, -d_lse_own, None


def _gather_form(interface, module, query, key, value, plan, part, **kwargs):
    """The stock interface over per-node key sequences (window, sink and dtype handling are its own).

    Keys are gathered once, so a token shared across calls still sums its gradient in fp32; a part
    with ``chunks`` then attends one run of whole nodes per call, each node being its own sequence.
    """
    key, value = (_GatherTokens.apply(states, part["k_index"], 2) for states in (key, value))
    if "chunks" not in part:
        return interface(
            module,
            query,
            key,
            value,
            None,
            cu_seq_lens_q=plan["cu_seqlens_q"],
            cu_seq_lens_k=part["cu_seqlens_k"],
            max_length_q=plan["max_seqlen_q"],
            max_length_k=part["max_seqlen_k"],
            **kwargs,
        )
    outputs = []
    for chunk in part["chunks"]:
        queries, keys = slice(*chunk["q_span"]), slice(*chunk["k_span"])
        out, _ = interface(
            module,
            query[:, :, queries],
            key[:, :, keys],
            value[:, :, keys],
            None,
            cu_seq_lens_q=chunk["cu_seqlens_q"],
            cu_seq_lens_k=chunk["cu_seqlens_k"],
            max_length_q=chunk["max_seqlen_q"],
            max_length_k=chunk["max_seqlen_k"],
            **kwargs,
        )
        outputs.append(out)
    return torch.cat(outputs, dim=1), None


def _split_form(module, query, key, value, plan, *, dropout=0.0, scaling=None, s_aux=None, softcap=None, **_):
    part = plan["full"]
    (_, varlen, *_), (flash_kwargs, _) = flash_utils.lazy_import_flash_attention(module.config._attn_implementation)
    q, k, v = (states[0].transpose(0, 1) for states in (query, key, value))
    q, k, v = flash_utils.fa_peft_integration_check(q, k, v, get_target_dtype(query, module))
    shared = {"dropout": dropout, "softmax_scale": scaling, "softcap": softcap}

    out, lse = varlen(
        q,
        k,
        v,
        cu_seqlens_q=plan["cu_seqlens_q"],
        cu_seqlens_k=plan["cu_seqlens_q"],
        return_lse=True,
        **flash_kwargs(
            query_length=q.shape[0],
            key_length=k.shape[0],
            is_causal=True,
            s_aux=None if s_aux is None else s_aux.to(q.dtype),
            max_seqlen_q=plan["max_seqlen_q"],
            max_seqlen_k=plan["max_seqlen_q"],
            **shared,
        ),
    )
    index = part["q_index"]
    out_anc, lse_anc = varlen(
        q.index_select(0, index),
        _GatherTokens.apply(k, part["k_index"], 0),
        _GatherTokens.apply(v, part["k_index"], 0),
        cu_seqlens_q=part["cu_seqlens_q"],
        cu_seqlens_k=part["cu_seqlens_k"],
        return_lse=True,
        **flash_kwargs(
            query_length=index.shape[0],
            key_length=part["k_index"].shape[0],
            is_causal=False,
            max_seqlen_q=part["max_seqlen_q"],
            max_seqlen_k=part["max_seqlen_k"],
            **shared,
        ),
    )
    return _MergeAncestors.apply(out, lse, out_anc, lse_anc, index).unsqueeze(0), None


class TreeAttentionRegistry(FlashInterfaceViewBase):
    """Flash interfaces that run a batch carrying a tree plan through tree attention.

    A plan without attention parts describes a row of plain documents, which the stock interface
    already isolates by their position resets.
    """

    def wrap_flash_interface(self, fn):
        @functools.wraps(fn)
        def with_tree_plan(module, query, key, value, attention_mask, **kwargs):
            plan = kwargs.pop(TREE_ATTENTION_KEY, None)
            if plan is None or "full" not in plan:
                return fn(module, query, key, value, attention_mask, **kwargs)
            if kwargs.get("sliding_window") is not None:
                return _gather_form(fn, module, query, key, value, plan, plan["sliding"], **kwargs)
            if "q_index" in plan["full"]:
                return _split_form(module, query, key, value, plan, **kwargs)
            return _gather_form(fn, module, query, key, value, plan, plan["full"], **kwargs)

        return with_tree_plan


def _tree_conv(conv):
    """A causal convolution that gives each node of a tree row the last tokens of its path as context."""

    @functools.wraps(conv)
    def with_tree_plan(hidden_states, *args, **kwargs):
        plan = kwargs.pop(TREE_ATTENTION_KEY, None)
        if plan is None:
            return conv(hidden_states, *args, **kwargs)
        part = plan["conv"]
        # The CUDA kernel reads seq_idx from channel-last memory only: gather on the [B, T, C] view.
        behind = _GatherTokens.apply(hidden_states.transpose(1, 2), part["index"], 1).transpose(1, 2)
        out = conv(behind, *args, **{**kwargs, "seq_idx": part["seq_idx"]})
        return out.transpose(1, 2).index_select(1, part["own"]).transpose(1, 2)

    return with_tree_plan


def _tree_recurrence(recurrence):
    """A delta-rule scan that runs a tree row one depth at a time, each node from its parent's final state."""

    @functools.wraps(recurrence)
    def with_tree_plan(query, key, value, *, g, beta, **kwargs):
        plan = kwargs.pop(TREE_ATTENTION_KEY, None)
        if plan is None:
            return recurrence(query, key, value, g=g, beta=beta, **kwargs)
        kwargs = {name: arg for name, arg in kwargs.items() if name not in _RECURRENCE_STATE_ARGS}
        if kwargs.get(_RECURRENCE_L2NORM_FLAG):
            # Per token, so once over the row is what every depth would compute. The kernel keys a
            # compile on the token count it normalizes, which a depth changes on every call.
            query, key = (_L2Norm.apply(states, _RECURRENCE_L2NORM_EPS) for states in (query, key))
            kwargs[_RECURRENCE_L2NORM_FLAG] = False
        levels, outputs, state = plan["levels"], [], None
        for depth, level in enumerate(levels):
            index = level["index"]
            out, state = recurrence(
                query.index_select(1, index),
                key.index_select(1, index),
                value.index_select(1, index),
                g=g.index_select(1, index),
                beta=beta.index_select(1, index),
                initial_state=None if state is None else state.index_select(0, level["parent"]),
                output_final_state=depth + 1 < len(levels),
                cu_seqlens=level["cu_seqlens"],
                **kwargs,
            )
            outputs.append(out)
        return torch.cat(outputs, dim=1).index_select(1, plan["restore"]), None

    return with_tree_plan


def _require_tree_family(model_config) -> TreeFamily:
    """The family entry for this model, or the refusal of a family tree packing cannot run."""
    model_type = getattr(text_config(model_config), "model_type", None)
    family = TREE_FAMILIES.get(model_type)
    if family is None:
        raise ValueError(
            f"tree packing is not available for model_type={model_type!r}: only {sorted(TREE_FAMILIES)} "
            f"are verified against every example run alone. A family joins by passing that test."
        )
    return family


def resolve_tree_plan_options(model_config) -> dict[str, int | None]:
    """The model-dependent arguments of :func:`~halo.models.attention.tree_plan.build_tree_attention_plan`."""
    stateful, config = _require_tree_family(model_config).stateful, text_config(model_config)
    return {
        "sliding_window": getattr(config, "sliding_window", None),
        "conv_kernel": getattr(config, stateful.conv_kernel_field) if stateful else None,
    }


def _decoder_layer_modules(model, attr: str) -> list[torch.nn.Module]:
    """The ``attr`` module of every decoder layer that has one, under any wrapper."""
    backbone = backbone_with_layers(base_transformers_model(model))
    layers = decoder_layers(backbone) if backbone is not None else ()
    return [getattr(layer, attr) for layer in layers if hasattr(layer, attr)]


def reject_unsupported_tree_attention(
    model_config, attn_implementation: str | None, full_form: FullAttentionForm
) -> None:
    """Refuse, from the config and the attention backend it resolves to, what tree attention cannot run: a family
    outside :data:`TREE_FAMILIES`, a backend with no tree form, the split form off FA4, and linear-attention kernels
    that ignore segments. Needs no weights, so a run checks it before the load; :func:`install_tree_attention`
    repeats it on the loaded model."""
    _require_tree_family(model_config)
    if attn_implementation not in TREE_ATTN_IMPLEMENTATIONS:
        raise ValueError(
            f"tree packing is verified on {list(TREE_ATTN_IMPLEMENTATIONS)}, but this model resolved to "
            f"attn_implementation={attn_implementation!r}. Set attn_implementation to one of them."
        )
    if full_form == "split" and attn_implementation != _SPLIT_FORM_IMPLEMENTATION:
        raise ValueError(
            f"tree_full_attention='split' reads the attention kernel's log-sum-exp, which only "
            f"{_SPLIT_FORM_IMPLEMENTATION} returns with a gradient; this model resolved to "
            f"{attn_implementation!r}. Set tree_full_attention: gather."
        )
    require_segment_aware_kernels(model_config, "tree_packing")


def install_tree_attention(model, full_form: FullAttentionForm) -> None:
    """Install tree attention for ``model``'s family, or refuse what it cannot run. Idempotent.

    A plan that reaches a stock interface is dropped without a word and the row runs as one causal
    sequence, so everything that would leave the plan unread is refused here: what
    :func:`reject_unsupported_tree_attention` refuses, read off the loaded model, and attention layers built from
    another module than the one the view goes on (a remote-code class of the same model type).
    """
    reject_unsupported_tree_attention(model.config, effective_attn_implementation(model.config), full_form)
    family = _require_tree_family(model.config)
    module = family.module
    attentions = _decoder_layer_modules(model, "self_attn")
    foreign = sorted({type(layer).__module__ for layer in attentions} - {module.__name__})
    if foreign or not attentions:
        raise ValueError(
            f"tree packing installs on {module.__name__}, but this model's attention layers come from "
            f"{foreign or 'no self_attn module'}: its tree plan would never be read. Load the native "
            f"transformers class for this model type."
        )
    if isinstance(module.ALL_ATTENTION_FUNCTIONS, TreeAttentionRegistry):
        return
    module.ALL_ATTENTION_FUNCTIONS = TreeAttentionRegistry(module.ALL_ATTENTION_FUNCTIONS)
    if family.stateful is not None:
        for name, wrap in ((family.stateful.conv, _tree_conv), (family.stateful.recurrence, _tree_recurrence)):
            setattr(module, name, wrap(getattr(module, name)))
    if is_global_main_process():
        logger.info(f"Installed tree attention on {module.__name__} (full layers: {full_form})")


def _on_device(plan, device: torch.device):
    if isinstance(plan, dict):
        return {name: _on_device(part, device) for name, part in plan.items()}
    if isinstance(plan, list):
        return [_on_device(part, device) for part in plan]
    return plan.to(device) if torch.is_tensor(plan) else plan


def _warmup_plan(
    depth: int, device: torch.device, *, full_form: FullAttentionForm, conv_kernel: int | None = None
) -> dict:
    """The plan of two root-to-leaf chains ``depth`` nodes deep, so every kernel compiles against a
    real ``cu_seqlens``; nodes are long enough to land on the tiles a training row uses."""
    lengths = np.full(2 * depth, WARMUP_MIN_SEQ_LEN)
    parents = np.array([-1 if node % depth == 0 else node - 1 for node in range(2 * depth)])
    table = NodeTable(starts=np.cumsum(lengths) - lengths, lengths=lengths, parents=parents)
    plan = build_tree_attention_plan(table, sliding_window=None, full_form=full_form, conv_kernel=conv_kernel)
    return _on_device(plan, device)


def warm_split_form_kernels(model_config, *, dtype: torch.dtype) -> None:
    """Run the split form forward and backward over a small tree, compiling the kernels it adds: a
    non-causal pass and a backward that takes a log-sum-exp gradient."""
    device = torch.device("cuda", torch.cuda.current_device())
    config = text_config(model_config)
    heads, kv_heads = config.num_attention_heads, resolve_num_key_value_heads(model_config)
    head_dim = resolve_head_dim(model_config)
    plan = _warmup_plan(_SPLIT_WARMUP_DEPTH, device, full_form="split")
    tokens = int(plan["cu_seqlens_q"][-1])
    query, key, value = (
        torch.randn(1, n, tokens, head_dim, device=device, dtype=dtype, requires_grad=True)
        for n in (heads, kv_heads, kv_heads)
    )
    sinks = torch.randn(heads, device=device, dtype=dtype) if uses_gpt_oss_sinks(model_config) else None
    out, _ = _split_form(SimpleNamespace(config=config), query, key, value, plan, scaling=head_dim**-0.5, s_aux=sinks)
    out.sum().backward()
    torch.cuda.synchronize()


def warm_recurrent_mixer_kernels(model, *, dtype: torch.dtype) -> None:
    """Run one stateful linear-attention layer forward and backward over a small tree, compiling
    the recurrence kernels that take and return a state. A no-op for a family without one."""
    stateful = _require_tree_family(model.config).stateful
    if stateful is None:
        return
    config = text_config(model.config)
    mixer = copy.deepcopy(_decoder_layer_modules(model, stateful.layer_attr)[0]).to(dtype)
    device = next(mixer.parameters()).device
    plan = _warmup_plan(
        _RECURRENCE_WARMUP_DEPTH, device, full_form="gather", conv_kernel=getattr(config, stateful.conv_kernel_field)
    )
    tokens = int(plan["cu_seqlens_q"][-1])
    hidden_states = torch.randn(1, tokens, config.hidden_size, device=device, dtype=dtype, requires_grad=True)
    mixer(hidden_states=hidden_states, **{TREE_ATTENTION_KEY: plan}).sum().backward()
    torch.cuda.synchronize()


def warm_tree_kernels(model, *, dtype: torch.dtype, full_form: FullAttentionForm) -> None:
    """Pre-compile the kernels tree attention adds to the stock ones. Rank-local, best-effort.

    Left to the first batch that holds a branch, each of these compiles stalls that rank alone, so
    the caller runs this on every rank behind one fence
    (:func:`~halo.model_loading.parallel.whole_model.warm_tree_attention_kernels`). The gather form itself
    runs the flash kernels the stock warm-up already compiled.
    """
    try:
        if full_form == "split":
            warm_split_form_kernels(model.config, dtype=dtype)
        warm_recurrent_mixer_kernels(model, dtype=dtype)
        if is_global_main_process():
            logger.info(f"Tree attention kernel warmup done (full layers: {full_form}, dtype={dtype})")
    except Exception as e:
        # Every rank: the skip is rank-local, and a rank without the warm-up stalls later.
        logger.warning(f"Tree attention kernel warmup skipped (non-fatal): {type(e).__name__}: {e}")
