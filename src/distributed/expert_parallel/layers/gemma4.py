"""EP wrapper for Gemma-4 routed experts (router lives in parent decoder)."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.distributed.expert_parallel.base_layer import EPMoELayerBase


class EPGemma4MoELayer(EPMoELayerBase):
    """EP wrapper for Gemma4 routed experts (``Gemma4TextExperts``) using DeepEP.

    Gemma4 inlines MoE compute inside ``Gemma4TextDecoderLayer.forward`` — no encapsulating
    ``*MoeBlock``. Only experts are replaced; the sibling ``Gemma4TextRouter`` stays FSDP-managed, so
    ``fp32_router`` has no effect and ``fp32_non_ep_params`` is not a substitute — it would upcast the
    router, and DeepEP's combine is bf16-only. The Gemma4 router stays bf16 under EP.
    Tokens arrive flat ``[B*S, H]`` with routing weights already pre-normalized/scaled.

    Expert TP supported via the base fused-GLU helper; TP/CP for Gemma4 attention not supported
    (KV-shared layers and ``attention_k_eq_v`` need bespoke handling).
    """

    HF_MODULE_NAMES = ("Gemma4TextExperts",)
    HF_MODEL_TYPES = ("gemma4", "gemma4_text")

    # The router is a sibling module in the decoder layer, not a child of the wrapped experts: nothing
    # is adopted here, and only expert grad-sync hooks register.
    _ROUTER_ATTR = None

    # Gemma4 spells its config field ``hidden_activation``, which the name-resolution chain does not
    # read, so the family fallback must be its own default rather than the roster's SiLU.
    _DEFAULT_ACTIVATION = "gelu_pytorch_tanh"

    # The sibling router passes only pre-weighted (indices, weights) — nothing to re-derive from.
    _supports_routing_replay = False

    # Unvalidated for this family and documented to fail at the first dispatch
    # (agent-docs/models/gemma4.md); refused at load rather than as a raw DeepEP C++ assert.
    _supports_fp32_non_ep_params = False

    # ``gate_up_proj`` is ``[E, 2M, H]`` — its length is the expert count.
    _NUM_EXPERTS_ATTR_PATHS = ("num_experts", "gate_up_proj")

    # The full-attention layers' head dim and KV-head count, which vLLM 0.26.0's Gemma 4 loader and
    # its transformers 5.14 config class read off these two flat keys (``head_dim`` /
    # ``num_key_value_heads`` stay the sliding-layer values).
    _LEGACY_PER_LAYER_CONFIG_KEYS = {
        "global_head_dim": ("full_attention", "head_dim"),
        "num_global_key_value_heads": ("full_attention", "num_key_value_heads"),
    }

    @classmethod
    def _find_experts_container(cls, layer: nn.Module) -> nn.Module:
        """The wrapped module IS the expert container — Gemma4 inlines MoE in its decoder layer, so
        there is no block to read one off."""
        return layer

    def _init_routing(self, original_layer: nn.Module) -> None:
        """The sibling ``Gemma4TextRouter`` hands the decision in already made, and
        ``Gemma4TextExperts`` carries no routing knob — there is nothing to read here."""

    def _init_summary_extras(self, original_layer: nn.Module) -> tuple[str, ...]:
        return ("router=sibling-module (FSDP-managed)",)

    def gather_expert_state_dict(self, device: str = "cpu", merge_lora: bool = False, retain: bool = True) -> dict:
        """Gather experts keyed bare (``gate_up_proj``/``down_proj``), not ``experts.``-prefixed.

        This wrapper replaces ``Gemma4TextExperts`` itself, so the base's ``experts.`` prefix would
        double to ``...experts.experts.gate_up_proj`` and fail to reload. Strip it to match the checkpoint.
        """
        return {
            key.removeprefix("experts."): tensor
            for key, tensor in super().gather_expert_state_dict(device, merge_lora=merge_lora, retain=retain).items()
        }

    @classmethod
    def merge_shards_to_hf(cls, prefix: str, params: dict) -> dict:
        """Same bare keying as the gather, applied to merged shards (see the base method)."""
        return {
            f"{prefix}.{key.removeprefix('experts.')}": tensor
            for key, tensor in cls._merge_fused_shards(params).items()
        }

    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Match ``Gemma4TextExperts.forward``: tokens are already flat ``[B*S, H]``."""
        return self._dispatch_compute_combine(
            hidden_states, top_k_index.long(), top_k_weights.float(), hidden_states.dtype
        )
