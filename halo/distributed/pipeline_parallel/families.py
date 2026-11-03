"""Per-family pipeline-split contracts, keyed by the HF backbone class: the registered families and the
refused ones. A backbone no spec claims is refused by name.

The hyper-connection families' non-first stages run their own forward minus the embedding widen:
collapsing the widened residual stream mid-network is lossy.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_recurrent_attention_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, MoeModelOutputWithPast

from halo.models.config_access import text_config
from halo.models.module_tree import find_attr_name
from halo.models.qwen4_exp.forwards import document_forwards_installed, document_padding_mask


@dataclass(frozen=True)
class PPFamilySpec:
    """A family's pipeline-split contract.

    Attributes:
        hf_module_names: the backbone classes (the module holding the decoder-layer list) claimed.
        refusal: why the family is refused: the state a cut would sever.
        final_norm_attrs: the post-layers norm, probed in order; neutralized off the last stage.
        head_attrs: the model's task head, probed in order; held by the last stage only.
        tail_module_attrs: further post-layers backbone modules (the hyper-connection collapse
            ``hc_head``), neutralized off the last stage. Exact names; each must exist.
        layer_types_rebase_safe: every per-layer input is selected by the layer's own attributes
            rather than its list position, so stage offsets need not land on a ``layer_types`` period.
        stream_forward: the backbone forward a non-first stage runs; ``None`` = the upstream forward,
            driven through ``inputs_embeds``.
        token_ids_kwarg: the backbone kwarg a non-first stage takes the microbatch's token ids under, for
            layers that read them mid-network (per-layer embeddings); ``None`` = only stage 0 reads them.
        reject_severed_boundaries: ``(model, partition)`` raising when a boundary severs the
            family's cross-layer state.
        validated: the family is registered: its stages reproduce the unsplit model on the pipeline GPU tier.
    """

    hf_module_names: tuple[str, ...] = ()
    refusal: str | None = None
    final_norm_attrs: tuple[str, ...] = ("norm", "final_layernorm", "ln_f", "embedding_norm")
    head_attrs: tuple[str, ...] = ("lm_head", "score", "classifier")
    tail_module_attrs: tuple[str, ...] = ()
    layer_types_rebase_safe: bool = False
    stream_forward: Callable | None = None
    token_ids_kwarg: str | None = None
    reject_severed_boundaries: Callable[[nn.Module, list[tuple[int, int]]], None] | None = None
    validated: bool = False


def _boundary_stream(backbone, inputs_embeds: torch.Tensor | None, stream_shape: tuple[int, ...]) -> torch.Tensor:
    """The previous stage's hyper-connection stream, refused unless it is ``[B, S, *stream_shape]``.

    Anything else means the previous stage collapsed the stream, and widening it by replication would
    train on a lossy mean of the streams.
    """
    if inputs_embeds is None or tuple(inputs_embeds.shape[2:]) != stream_shape:
        got = "None" if inputs_embeds is None else f"shape {tuple(inputs_embeds.shape)}"
        raise RuntimeError(
            f"{type(backbone).__name__} is a non-first pipeline stage: its input must be the previous "
            f"stage's hyper-connection stream [batch, seq, {', '.join(map(str, stream_shape))}], got {got}. "
            f"Collapsing the widened residual mid-network is lossy, so this stage refuses anything "
            f"but the full stream."
        )
    return inputs_embeds


def _stacked_stream_shape(config) -> tuple[int, int]:
    """The ``(hc_mult, hidden)`` trailing shape of DeepSeek-V4's and GLM-5's ``[B, S, hc_mult, D]`` stream."""
    return config.hc_mult, config.hidden_size


def _stream_prologue(
    backbone, inputs_embeds, position_ids, past_key_values, use_cache
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Cache | None]:
    """``(stream, [B, S, D] view of one stream, position_ids, cache)`` for a stacked-stream mid-chain forward.

    The mask and rotary constructors read only batch, sequence, dtype and device off
    ``inputs_embeds``, so the one-stream view stands in for it.
    """
    hidden_states = _boundary_stream(backbone, inputs_embeds, _stacked_stream_shape(backbone.config))
    embeds_ref = hidden_states[:, :, 0]
    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=backbone.config)
    if position_ids is None:
        past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
        position_ids = (torch.arange(embeds_ref.shape[1], device=embeds_ref.device) + past_seen).unsqueeze(0)
    return hidden_states, embeds_ref, position_ids, past_key_values


def deepseek_v4_stream_forward(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    use_cache: bool | None = None,
    **kwargs,
) -> MoeModelOutputWithPast:
    """``DeepseekV4Model.forward`` (transformers ~5.20) minus the embedding widen."""
    hidden_states, embeds_ref, position_ids, past_key_values = _stream_prologue(
        self, inputs_embeds, position_ids, past_key_values, use_cache
    )
    if isinstance(attention_mask, dict):
        causal_mask = next(iter(attention_mask.values()))
    else:
        causal_mask = create_sliding_window_causal_mask(
            config=self.config,
            inputs_embeds=embeds_ref,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
    position_embeddings = {
        "main": self.rotary_emb(embeds_ref, position_ids=position_ids, layer_type="main"),
        "compress": self.rotary_emb(embeds_ref, position_ids=position_ids, layer_type="compress"),
    }

    # input_ids is None past stage 0; the hash-router layers that read it are confined to stage 0.
    for layer in self.layers:
        hidden_states = layer(
            hidden_states,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            attention_mask=causal_mask,
            input_ids=input_ids,
            past_key_values=past_key_values,
            **kwargs,
        )

    hidden_states = self.norm(self.hc_head(hidden_states))
    return MoeModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


def glm5_next_stream_forward(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    use_cache: bool | None = None,
    **kwargs,
) -> MoeModelOutputWithPast:
    """``Glm5NextTextModel.forward`` (transformers ~5.20) minus the embedding widen.

    Each layer's mask is keyed by its own ``block_type`` (set from its global index at construction)
    rather than ``config.layer_types[i]`` at loop position ``i``, which a sliced list re-bases.
    Upstream both mask entries hold the same tensor, so the two spellings agree today.
    """
    hidden_states, embeds_ref, position_ids, past_key_values = _stream_prologue(
        self, inputs_embeds, position_ids, past_key_values, use_cache
    )
    if not isinstance(causal_mask_mapping := attention_mask, dict):
        attention_mask = create_recurrent_attention_mask(
            config=self.config,
            inputs_embeds=embeds_ref,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
        # The indexer needs a mask even when none was passed (mirrors the upstream forward).
        if attention_mask is None:
            attention_mask = torch.ones(
                embeds_ref.shape[0], embeds_ref.shape[1], dtype=torch.bool, device=embeds_ref.device
            )
        attention_mask = attention_mask.bool()
        causal_mask_mapping = {
            "indexed_attention": attention_mask,
            "linear_attention": attention_mask,
        }

    # No stage begins on a "shared" DSA layer, so every indexer chain starts on its own stage.
    topk_indices = None
    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        hidden_states, topk_indices = decoder_layer(
            hidden_states,
            attention_mask=causal_mask_mapping[decoder_layer.block_type],
            position_ids=position_ids,
            position_embeddings=None,  # NoPE, as upstream
            input_ids=input_ids,
            past_key_values=past_key_values,
            prev_topk_indices=topk_indices,
            **kwargs,
        )

    hidden_states = self.norm(self.hc_head(hidden_states))
    return MoeModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


def qwen4_exp_stream_forward(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    use_cache: bool | None = None,
    ple_input_ids: torch.LongTensor | None = None,
    **kwargs,
) -> BaseModelOutputWithPast:
    """``Qwen4ExpTextModel.forward`` (transformers ~5.20) under the toolkit's document forward, minus the embedding
    widen: the received ``[B, S, hc_count * D]`` stream runs the layers, each reading the 2-D padding mask, and
    ``ple_input_ids`` (the microbatch's token ids, which the spec's ``token_ids_kwarg`` routes here) feed the
    per-layer embeddings. Uncached rows only, as a pipeline stage runs them."""
    config = self.config
    if past_key_values is not None or use_cache:
        raise RuntimeError(
            f"{type(self).__name__} runs as a non-first pipeline stage on uncached training rows only; it was "
            f"called with a cache (past_key_values={type(past_key_values).__name__}, use_cache={use_cache})."
        )
    if not document_forwards_installed():
        raise RuntimeError(
            f"{type(self).__name__} is a non-first pipeline stage, whose layers read the 2-D padding mask of the "
            f"toolkit's document forwards, but those are not installed: load the model through a Halo loader, which "
            f"applies the family patches (patch_qwen4_exp_document_forwards)."
        )
    hidden_states = _boundary_stream(self, inputs_embeds, (config.hc_count * config.hidden_size,))
    if ple_input_ids is None and any(layer.ple is not None for layer in self.layers):
        raise RuntimeError(
            f"{type(self).__name__} is a non-first pipeline stage holding a per-layer-embedding layer, which reads "
            f"the microbatch's token ids, but none reached it: the runner passes them as `token_ids`."
        )
    padding = document_padding_mask(config, hidden_states, attention_mask, position_ids, kwargs.get("seq_idx"))
    if ple_input_ids is not None and padding is not None:
        eos_token_id = config.eos_token_id[0] if isinstance(config.eos_token_id, list) else config.eos_token_id
        ple_input_ids = torch.where(padding.bool(), ple_input_ids, eos_token_id)

    # Upstream's M-RoPE rows: the text row leads a 4-row tensor, a 2-D one is the text row on every axis.
    if position_ids is None:
        position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        position_ids = position_ids.view(1, 1, -1).expand(4, hidden_states.shape[0], -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)
    if position_ids.shape[0] == 4:
        position_ids = position_ids[1:]
    elif position_ids.shape[0] == 1:
        position_ids = position_ids.expand(3, -1, -1)
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    for decoder_layer in self.layers[: config.num_hidden_layers]:
        hidden_states = decoder_layer(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=padding,
            conv_mask=padding,
            past_key_values=None,
            ple_input_ids=ple_input_ids,
            **kwargs,
        )

    return BaseModelOutputWithPast(last_hidden_state=self.hyper_connection_mixer(hidden_states))


def _reject_stranded_hash_routers(model: nn.Module, partition: list[tuple[int, int]]) -> None:
    """Every DeepSeek-V4 ``hash_moe`` layer must land on stage 0, the only stage holding the
    ``input_ids`` its ``tid2eid`` router reads."""
    mlp_layer_types = getattr(text_config(getattr(model, "config", None)), "mlp_layer_types", None) or ()
    hash_layers = [i for i, kind in enumerate(mlp_layer_types) if kind == "hash_moe"]
    stage0_hi = partition[0][1]
    stranded = [i for i in hash_layers if i >= stage0_hi]
    if stranded:
        raise ValueError(
            f"DeepSeek-V4 hash_moe layer(s) {stranded} fall outside pipeline stage 0 (layers "
            f"[0, {stage0_hi})). Their tid2eid routers consume input_ids, which only stage 0 "
            f"receives. Use pipeline_split to give stage 0 at least {max(hash_layers) + 1} layers."
        )


def _reject_severed_dsa_chains(model: nn.Module, partition: list[tuple[int, int]]) -> None:
    """No GLM-5 stage may begin on a "shared" DSA layer: it reuses the previous full indexer layer's
    top-k indices, handed layer to layer by the model loop."""
    indexer_types = getattr(text_config(getattr(model, "config", None)), "indexer_types", None) or ()
    severed = [lo for lo, _ in partition[1:] if lo < len(indexer_types) and indexer_types[lo] == "shared"]
    if severed:
        raise ValueError(
            f"GLM-5 pipeline stage boundary at layer(s) {severed} severs a DSA indexer chain: "
            f"config.indexer_types marks those layers 'shared', so they reuse the previous full "
            f"indexer layer's top-k indices from across the cut. Choose pipeline_split boundaries "
            f"that keep each full+shared indexer run on one stage."
        )


# A backbone no spec claims: the default probes, unregistered.
GENERIC_PP = PPFamilySpec()
# The registered families whose upstream forward splits contiguously as-is: the per-layer inputs are built once
# before the loop, and its only list-position read is ``config.layer_types[i]``, which validate_partition holds to
# whole periods. Inkling's embedding module carries ``embed_norm``, so a later stage drops the norm with it.
STOCK_FORWARD_PP = PPFamilySpec(
    hf_module_names=(
        "Glm4MoeLiteModel",
        "GptOssModel",
        "InklingTextModel",
        "LagunaModel",
        "Qwen3Model",
        "Qwen3MoeModel",
        "Qwen3_5TextModel",
        "Qwen3_5MoeTextModel",
        "Step3p7TextModel",
    ),
    validated=True,
)
ZAYA_PP = PPFamilySpec(
    hf_module_names=("ZayaModel",),
    refusal=(
        "Zaya's forward carries two tensors across every decoder-layer boundary: the fp32 residual "
        "stream (cast to float32 before the loop, de-cast only at the final norm) and the EDA "
        "router state (prev_router_hidden_states), which each layer's router adds to and forwards, "
        "so it accumulates down the stack and cannot be recomputed on a later stage. A stage cut "
        "would need a two-tensor, mixed-dtype P2P boundary the single-activation pipeline contract "
        "does not carry; the loop also selects layer_types and rotary embeddings by list position."
    ),
)
# Nothing in the loop indexes by list position (masks are loop-invariant, per-layer state hangs off
# each module's own layer_idx), so any split offset is exact.
DEEPSEEK_V4_PP = PPFamilySpec(
    hf_module_names=("DeepseekV4Model",),
    tail_module_attrs=("hc_head",),
    layer_types_rebase_safe=True,
    stream_forward=deepseek_v4_stream_forward,
    reject_severed_boundaries=_reject_stranded_hash_routers,
    validated=True,
)
GEMMA4_PP = PPFamilySpec(
    hf_module_names=("Gemma4TextModel",),
    refusal=(
        "Gemma4 slices its per-layer embeddings as per_layer_inputs[:, :, i, :] where i is the "
        "ENUMERATE position in the layer list, which slicing the list silently re-bases, and its "
        "trailing num_kv_shared_layers consume an earlier layer's K/V through a forward-threaded "
        "dict that a stage boundary would break."
    ),
)
# Only the composite Glm5NextForConditionalGeneration ships; a text-only run reaches this spec
# through the trainer's multimodal admission.
GLM5_NEXT_PP = PPFamilySpec(
    hf_module_names=("Glm5NextTextModel",),
    tail_module_attrs=("hc_head",),
    layer_types_rebase_safe=True,
    stream_forward=glm5_next_stream_forward,
    reject_severed_boundaries=_reject_severed_dsa_chains,
    validated=True,
)
# The collapse of the four-stream residual, ``hyper_connection_mixer``, is the backbone's only post-layers module
# (it holds the final norm). Every per-layer input hangs off the layer's own construction-time attributes
# (``layer_type``, ``ple``) and both mask kinds are one padding mask, so any split offset is exact; the PLE layers
# re-read the token ids, which reach every stage.
QWEN4_EXP_PP = PPFamilySpec(
    hf_module_names=("Qwen4ExpTextModel",),
    final_norm_attrs=("hyper_connection_mixer",),
    layer_types_rebase_safe=True,
    stream_forward=qwen4_exp_stream_forward,
    token_ids_kwarg="ple_input_ids",
    validated=True,
)
PP_FAMILY_SPECS = (STOCK_FORWARD_PP, ZAYA_PP, DEEPSEEK_V4_PP, GEMMA4_PP, GLM5_NEXT_PP, QWEN4_EXP_PP)
PP_SPEC_MAP: dict[str, PPFamilySpec] = {name: spec for spec in PP_FAMILY_SPECS for name in spec.hf_module_names}
if len(PP_SPEC_MAP) != sum(len(spec.hf_module_names) for spec in PP_FAMILY_SPECS):
    raise ValueError("Two PPFamilySpec constants in PP_FAMILY_SPECS claim one HF backbone class name.")


def resolve_pp_spec(backbone: nn.Module) -> PPFamilySpec:
    """The spec claiming ``backbone``'s class, or the unregistered :data:`GENERIC_PP`."""
    return PP_SPEC_MAP.get(type(backbone).__name__, GENERIC_PP)


def require_pp_attr(module: nn.Module, candidates: tuple[str, ...], what: str) -> str:
    """Name of the first attribute in ``candidates`` that ``module`` carries; the probe for the final
    norm, the task head and the family's tail modules.

    Raises:
        ValueError: none is present. Always fatal — a caller would otherwise skip a step (an
            un-neutralized norm, a head left on every stage) and produce wrong numbers.
    """
    name = find_attr_name(module, candidates)
    if name is None:
        raise ValueError(
            f"{type(module).__name__} exposes no {what} under any of {candidates}. Pipeline parallelism "
            f"needs it; declare the attribute on this family's PPFamilySpec if the layout differs."
        )
    return name
