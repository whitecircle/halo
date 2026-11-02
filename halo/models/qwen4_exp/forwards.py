"""The upstream Qwen3.8-Flash-Next (``qwen4_exp``) forwards the toolkit replaces at load, and their installers.

Each installer is idempotent and checks the upstream signatures it replaces or re-derives, so a transformers release
that changed one fails at load instead of running a replacement written for another forward. The forwards apply to
every build once installed; only the n-gram table's placement follows the config stamp.
"""

from __future__ import annotations

import functools
import inspect
import logging

import torch
import torch.nn.functional as F
import transformers.models.qwen4_exp.modeling_qwen4_exp as qwen4_exp_modeling
from torch import nn
from transformers.masking_utils import create_recurrent_attention_mask

from halo.models.config_access import set_config_field_run_scoped
from halo.models.external_tables import EXTERNAL_TABLES_STAMP, reject_state_dict_save
from halo.models.qwen4_exp.documents import (
    document_ngram_ids,
    document_ple_forward,
    document_starts,
    ngram_rows,
    reject_unmarked_documents,
)
from halo.models.qwen4_exp.ngram_table import HostNGramTable
from halo.models.qwen4_exp.sparse_attention import qsa_attention_forward
from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)

# The n-gram table's checkpoint keys, which a host-table build leaves to the table attach: the hub's shards and
# the embedding weight transformers concatenates them into.
_NGRAM_TABLE_KEY_PATTERN = r"ngram_embedding\.(?:shard_\d+\.)?weight$"
# Marks the installed n-gram embedding forward.
_HOST_TABLE_ATTR = "_halo_host_ngram_table"
# Marks the installed document forwards.
_DOCUMENT_FORWARDS_ATTR = "_halo_document_forwards"
# The per-layer-type mask mapping upstream's model forward builds and reads, keyed by its ``layer_types``.
_MASK_KINDS = ("indexed_attention", "linear_attention")


def apply_qwen4_exp_patches(model_config) -> None:
    """Apply every Qwen4-Exp patch and stamp ``model_config`` (run-scoped) so its n-gram tables build in host
    memory."""
    patch_qwen4_exp_ngram_embedding()
    patch_qwen4_exp_document_forwards()
    set_config_field_run_scoped(model_config, EXTERNAL_TABLES_STAMP, True)


def _verify_upstream_parameters(function, expected: tuple[str, ...], patch: str) -> None:
    names = tuple(inspect.signature(function).parameters)
    if names != expected:
        raise RuntimeError(
            f"{function.__qualname__} takes {names} in this transformers release, but the toolkit's replacement "
            f"({patch}) is written for {expected}: re-derive it on this release's forward."
        )


def _host_table_forward(module: nn.Module, input_ids: torch.Tensor, past_key_values) -> torch.Tensor:
    """Upstream's ``Qwen4ExpTextNGramEmbedding.forward`` over a host table: the same history, cache update and
    ids, the rows from the table's own lookup."""
    input_ids = input_ids.long()
    resumed = past_key_values is not None and past_key_values.has_previous_state(module.layer_idx, state_idx=2)
    if resumed:
        previous = past_key_values.layers[module.layer_idx].conv_states[2].clone()
    else:
        previous = input_ids.new_full((input_ids.shape[0], module.context_len), module.eos_token_id)
    if past_key_values is not None:
        cached = input_ids
        # The cache pads a short first chunk with zeros; the history pads with eos.
        if not resumed and input_ids.shape[1] < module.context_len:
            cached = F.pad(cached, (module.context_len - input_ids.shape[1], 0), value=module.eos_token_id)
        past_key_values.update_conv_state(cached, module.layer_idx, state_idx=2, conv_kernel_size=module.context_len)
    history = torch.cat([previous, input_ids], dim=-1)
    one_document = document_starts(None, *history.shape, history.device)
    ids = document_ngram_ids(module, history, one_document)[:, -input_ids.shape[1] :]
    return ngram_rows(module, ids)


def patch_qwen4_exp_ngram_embedding() -> None:
    """Give ``Qwen4ExpTextNGramEmbedding`` a :class:`HostNGramTable` in place of its ``nn.Embedding`` when its
    config carries the external-tables stamp, the forward that reads it, and the model classes an unexpected-key
    pattern for the table keys the attach reads instead and a ``save_pretrained`` that refuses a model holding the
    table (:func:`~halo.models.external_tables.reject_state_dict_save`). An unstamped build keeps the stock module and
    forward.
    """
    embedding_cls = qwen4_exp_modeling.Qwen4ExpTextNGramEmbedding
    if getattr(embedding_cls.forward, _HOST_TABLE_ATTR, False):
        return
    patch = "patch_qwen4_exp_ngram_embedding"
    _verify_upstream_parameters(
        embedding_cls.__init__, ("self", "config", "embedding_dim", "layer_idx", "ple_layer_index"), patch
    )
    _verify_upstream_parameters(embedding_cls.forward, ("self", "input_ids", "past_key_values"), patch)
    _verify_upstream_parameters(embedding_cls._shift_right_ignore_eos, ("self", "token_ids", "shift"), patch)
    pretrained_cls = qwen4_exp_modeling.Qwen4ExpPreTrainedModel
    original_init, original_forward = embedding_cls.__init__, embedding_cls.forward
    original_save = pretrained_cls.save_pretrained

    @functools.wraps(original_init)
    def __init__(self, config, embedding_dim, layer_idx, ple_layer_index=0):
        original_init(self, config, embedding_dim, layer_idx, ple_layer_index)
        if not getattr(config, EXTERNAL_TABLES_STAMP, False):
            return
        stock = self.ngram_embedding
        if not isinstance(stock, nn.Embedding):
            raise TypeError(
                f"{embedding_cls.__name__}.ngram_embedding is a {type(stock).__name__} in this transformers "
                f"release, not the nn.Embedding the host n-gram table replaces."
            )
        del self.ngram_embedding
        self.ngram_embedding = HostNGramTable(
            stock.num_embeddings, stock.embedding_dim, config.split_ngram_parts, stock.weight.dtype
        )

    @functools.wraps(original_forward)
    def forward(self, input_ids, past_key_values):
        if not isinstance(self.ngram_embedding, HostNGramTable):
            return original_forward(self, input_ids, past_key_values)
        return _host_table_forward(self, input_ids, past_key_values)

    @functools.wraps(original_save)
    def save_pretrained(self, *args, **kwargs):
        reject_state_dict_save(self)
        return original_save(self, *args, **kwargs)

    setattr(forward, _HOST_TABLE_ATTR, True)
    embedding_cls.__init__ = __init__
    embedding_cls.forward = forward
    pretrained_cls.save_pretrained = save_pretrained
    pretrained_cls._keys_to_ignore_on_load_unexpected = [
        *pretrained_cls._keys_to_ignore_on_load_unexpected,
        _NGRAM_TABLE_KEY_PATTERN,
    ]
    if is_global_main_process():
        logger.info("Patched Qwen4ExpTextNGramEmbedding to hold its n-gram table in host memory")


def _verify_upstream_mask_kinds(forward) -> None:
    source = inspect.getsource(inspect.unwrap(forward))
    missing = [kind for kind in _MASK_KINDS if f'"{kind}"' not in source]
    if missing:
        raise RuntimeError(
            f"{forward.__qualname__} does not read the {missing} entries of its mask mapping in this transformers "
            f"release, but the toolkit's document forward (patch_qwen4_exp_document_forwards) hands it "
            f"{list(_MASK_KINDS)}: re-derive it on this release's forward."
        )


def _text_position_ids(position_ids: torch.Tensor | None) -> torch.Tensor | None:
    """The text positions upstream's model forward reads: a ``[batch, sequence]`` tensor itself, the text row of a
    ``[4 | 1, batch, sequence]`` one, none for the three M-RoPE rows alone or no tensor."""
    if position_ids is None or position_ids.ndim == 2:
        return position_ids
    return position_ids[0] if position_ids.shape[0] in (1, 4) else None


def _is_document_call(attention_mask, past_key_values) -> bool:
    """Whether a layer runs under the document model forward: no cache, and the 2-D padding mask (or none) its
    mapping carries where upstream's own forward hands every layer a 4-D mask."""
    return past_key_values is None and (attention_mask is None or attention_mask.ndim == 2)


def _document_decoder_forward(
    layer: nn.Module, hidden_states, position_embeddings, attention_mask, conv_mask, ple_input_ids, **kwargs
) -> torch.Tensor:
    """Upstream's ``Qwen4ExpTextDecoderLayer.forward`` for an uncached row, its PLE kept inside the ``seq_idx``
    documents (:func:`~halo.models.qwen4_exp.documents.document_ple_forward`)."""
    hidden_states = hidden_states + document_ple_forward(
        layer.ple, hidden_states, ple_input_ids, kwargs.get("seq_idx"), conv_mask
    )
    hidden_states, hyper_input, injection_weights = layer.attn_hyper_connection(hidden_states)
    if layer.layer_type == "linear_attention":
        hidden_states = layer.linear_attn(hidden_states, cache_params=None, attention_mask=conv_mask, **kwargs)
    else:
        hidden_states, _ = layer.self_attn(hidden_states, position_embeddings, attention_mask=attention_mask, **kwargs)
    hidden_states = hyper_input + (hidden_states.unsqueeze(-2) * injection_weights.unsqueeze(-1)).flatten(-2)
    hidden_states, hyper_input, injection_weights = layer.mlp_hyper_connection(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    return hyper_input + (hidden_states.unsqueeze(-2) * injection_weights.unsqueeze(-1)).flatten(-2)


def patch_qwen4_exp_document_forwards() -> None:
    """Run every uncached ``qwen4_exp`` forward per ``seq_idx`` document with memory linear in the row length.

    The model forward hands its layers the 2-D padding mask in place of upstream's ``[B, 1, T, T]`` masks and
    refuses a packed row (``position_ids`` restarting) whose documents no ``seq_idx`` marks; its layers then run
    the tiled QSA (:func:`~halo.models.qwen4_exp.sparse_attention.qsa_attention_forward`) and the document-aware
    PLE. A call with a cache, a ``use_cache`` request or a caller-built 4-D or per-layer-type mask keeps upstream's
    forwards.
    """
    model_cls = qwen4_exp_modeling.Qwen4ExpTextModel
    layer_cls = qwen4_exp_modeling.Qwen4ExpTextDecoderLayer
    attention_cls = qwen4_exp_modeling.Qwen4ExpTextAttention
    if getattr(model_cls.forward, _DOCUMENT_FORWARDS_ATTR, False):
        return
    patch = "patch_qwen4_exp_document_forwards"
    _verify_upstream_parameters(
        model_cls.forward,
        (
            "self",
            "input_ids",
            "attention_mask",
            "position_ids",
            "past_key_values",
            "inputs_embeds",
            "use_cache",
            "ple_input_ids",
            "kwargs",
        ),
        patch,
    )
    _verify_upstream_mask_kinds(model_cls.forward)
    _verify_upstream_parameters(
        layer_cls.forward,
        (
            "self",
            "hidden_states",
            "position_embeddings",
            "attention_mask",
            "conv_mask",
            "past_key_values",
            "ple_input_ids",
            "kwargs",
        ),
        patch,
    )
    _verify_upstream_parameters(
        qwen4_exp_modeling.Qwen4ExpTextPLELayer.forward,
        ("self", "hidden_states", "input_ids", "past_key_values", "conv_mask"),
        patch,
    )
    _verify_upstream_parameters(
        qwen4_exp_modeling.Qwen4ExpTextPLELayer._short_conv, ("self", "hidden_states", "past_key_values"), patch
    )
    _verify_upstream_parameters(qwen4_exp_modeling.Qwen4ExpTextGatedResidual.forward, ("self", "hyper_input"), patch)
    _verify_upstream_parameters(
        create_recurrent_attention_mask,
        ("config", "inputs_embeds", "attention_mask", "past_key_values", "kwargs"),
        patch,
    )
    _verify_upstream_parameters(
        attention_cls.forward,
        ("self", "hidden_states", "position_embeddings", "attention_mask", "past_key_values", "kwargs"),
        patch,
    )
    _verify_upstream_parameters(
        qwen4_exp_modeling.Qwen4ExpTextQSAIndexer.forward,
        ("self", "hidden_states", "position_embeddings", "attention_mask", "past_key_values"),
        patch,
    )
    original_model, original_layer, original_attention = model_cls.forward, layer_cls.forward, attention_cls.forward

    @functools.wraps(original_model)
    def model_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        ple_input_ids=None,
        **kwargs,
    ):
        cached = past_key_values is not None or bool(use_cache)
        if cached or isinstance(attention_mask, dict) or (attention_mask is not None and attention_mask.ndim != 2):
            return original_model(
                self,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                ple_input_ids=ple_input_ids,
                **kwargs,
            )
        seq_idx = kwargs.get("seq_idx")
        # Upstream reads position restarts as documents only without a padding mask.
        positions = _text_position_ids(position_ids) if attention_mask is None or seq_idx is not None else None
        reject_unmarked_documents(positions, seq_idx)
        rows = input_ids if input_ids is not None else inputs_embeds
        padding = create_recurrent_attention_mask(self.config, rows, attention_mask)
        return original_model(
            self,
            input_ids=input_ids,
            attention_mask=dict.fromkeys(_MASK_KINDS, padding),
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            ple_input_ids=ple_input_ids,
            **kwargs,
        )

    @functools.wraps(original_layer)
    def layer_forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        conv_mask=None,
        past_key_values=None,
        ple_input_ids=None,
        **kwargs,
    ):
        if self.ple is None or not _is_document_call(attention_mask, past_key_values):
            return original_layer(
                self,
                hidden_states,
                position_embeddings,
                attention_mask=attention_mask,
                conv_mask=conv_mask,
                past_key_values=past_key_values,
                ple_input_ids=ple_input_ids,
                **kwargs,
            )
        return _document_decoder_forward(
            self, hidden_states, position_embeddings, attention_mask, conv_mask, ple_input_ids, **kwargs
        )

    @functools.wraps(original_attention)
    def attention_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
        if not _is_document_call(attention_mask, past_key_values):
            return original_attention(
                self, hidden_states, position_embeddings, attention_mask, past_key_values=past_key_values, **kwargs
            )
        return qsa_attention_forward(self, hidden_states, position_embeddings, attention_mask, **kwargs)

    setattr(model_forward, _DOCUMENT_FORWARDS_ATTR, True)
    model_cls.forward = model_forward
    layer_cls.forward = layer_forward
    attention_cls.forward = attention_forward
    if is_global_main_process():
        logger.info("Patched the Qwen4ExpText model, decoder layer and attention forwards for per-document rows")
