"""Mid-chain stage forwards for hyper-connection backbones (DeepSeek-V4, GLM-5 ``glm5_next``).

These families thread an ``hc_mult``-widened residual stream ``[B, S, hc_mult, D]`` between decoder
layers: the model forward widens ``inputs_embeds`` by replication before the first layer and
collapses the streams through ``hc_head`` only after the last one. Mid-network the streams have
diverged (each layer mixes them through a doubly-stochastic ``comb``), so the stream itself is the
only exact boundary activation — a ``[B, S, D]`` collapse loses three quarters of the state, and the
upstream forwards widen whatever ``inputs_embeds`` they receive, unconditionally.

Each function here is that family's ``*TextModel``/``*Model`` forward with one change: the
already-widened stream from the previous stage is fed straight to the layer loop instead of being
re-widened. Mask construction, rotary embeddings, the layer loop's kwargs and the final
``norm(hc_head(...))`` mirror transformers' own forward for the pinned release, so a stage stays
numerically identical to the unsplit model. A family declares its own through
``PPModelSpec.STREAM_FORWARD``, which binds it per instance on non-first stages only; stage 0 keeps
the real forward (it is driven by ``input_ids`` and widens natively).
"""

from __future__ import annotations

import torch
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_recurrent_attention_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import MoeModelOutputWithPast


def _boundary_stream(backbone, inputs_embeds: torch.Tensor | None) -> torch.Tensor:
    """Validate and return the previous stage's hyper-connection stream.

    Raises on anything but ``[B, S, hc_mult, D]``: a 3-D tensor here means the previous stage
    collapsed the stream (or a caller drove this stage like an unsplit model), and widening it by
    replication would train on a lossy mean of the four streams.
    """
    hc_mult = backbone.config.hc_mult
    if inputs_embeds is None or inputs_embeds.dim() != 4 or inputs_embeds.shape[2] != hc_mult:
        got = "None" if inputs_embeds is None else f"shape {tuple(inputs_embeds.shape)}"
        raise RuntimeError(
            f"{type(backbone).__name__} is a non-first pipeline stage: its input must be the previous "
            f"stage's hyper-connection stream [batch, seq, hc_mult={hc_mult}, hidden], got {got}. "
            f"Collapsing the widened residual mid-network is lossy, so this stage refuses anything "
            f"but the full stream."
        )
    return inputs_embeds


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
    """``DeepseekV4Model.forward`` (transformers ~5.16) minus the embedding widen."""
    hidden_states = _boundary_stream(self, inputs_embeds)
    # [B, S, D] view of one stream: the mask and rotary constructors only read batch/seq/dtype/device
    # off ``inputs_embeds``, and the widened tensor's extra axis is not part of their contract.
    embeds_ref = hidden_states[:, :, 0]
    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)
    if position_ids is None:
        past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
        position_ids = torch.arange(embeds_ref.shape[1], device=embeds_ref.device) + past_seen
        position_ids = position_ids.unsqueeze(0)
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

    # ``input_ids`` is None past stage 0 by construction; the hash-router layers that consume it are
    # confined to stage 0 by DeepSeekV4PPSpec.validate_partition, so no layer here dereferences it.
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
    """``Glm5NextTextModel.forward`` (transformers ~5.16) minus the embedding widen.

    The other deliberate difference: the attention mask is selected by each layer's own
    ``block_type`` (set from its true global index at construction) instead of
    ``config.layer_types[i]`` with the loop position ``i`` — a sliced list re-bases ``i``, and the
    per-layer attribute is what makes this family's split offset-independent
    (``LAYER_TYPES_REBASE_SAFE``). Upstream both mask-map entries hold the same tensor, so the two
    spellings are equivalent today; keying by the layer keeps that true if they diverge.
    """
    hidden_states = _boundary_stream(self, inputs_embeds)
    embeds_ref = hidden_states[:, :, 0]
    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)
    if position_ids is None:
        past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
        position_ids = torch.arange(embeds_ref.shape[1], device=embeds_ref.device) + past_seen
        position_ids = position_ids.unsqueeze(0)
    if not isinstance(causal_mask_mapping := attention_mask, dict):
        attention_mask = create_recurrent_attention_mask(
            config=self.config,
            inputs_embeds=embeds_ref,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
        # Guarantee the mask to exist for the indexer (mirrors the upstream forward).
        if attention_mask is None:
            attention_mask = torch.ones(
                embeds_ref.shape[0], embeds_ref.shape[1], dtype=torch.bool, device=embeds_ref.device
            )
        attention_mask = attention_mask.bool()
        causal_mask_mapping = {
            "deepseek_sparse_attention": attention_mask,
            "linear_attention": attention_mask,
        }

    # topk starts as None: a stage never begins on a "shared" DSA layer (validate_partition), so the
    # indexer chain is stage-local by construction.
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
