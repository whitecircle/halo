"""EP wrapper for NemotronH's MoE block: non-gated relu2 experts behind an optional latent projection."""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.distributed.expert_parallel.base_layer import EPGroupLimitedMoELayerBase


class EPNemotronHMoELayer(EPGroupLimitedMoELayerBase):
    """EP wrapper for ``NemotronHMoE`` using DeepEP.

    DeepSeek-V3 sigmoid routing with a correction bias, a shared expert on every token, and routed
    experts that are a plain ``down(act(up(x)))`` MLP with no gate half: ``up_proj [E, M, H_in]`` and
    ``down_proj [E, H_in, M]`` in ``F.linear`` convention, stored here in matmul convention. Where the
    block carries a latent projection (``moe_latent_size``), ``H_in`` is the latent width and tokens
    are projected before dispatch, so the all-to-all moves latent rows rather than hidden ones; without
    one the projections are ``nn.Identity``.
    """

    HF_MODULE_NAMES = ("NemotronHMoE",)
    HF_MODEL_TYPES = ("nemotron_h",)

    _supports_bias_balancing = True
    _NATIVE_BALANCING_BIAS_ATTR = "gate.e_score_correction_bias"

    _NUM_EXPERTS_ATTR_PATHS = ("experts.num_experts",)
    _EXPERT_WEIGHT_ATTR_ROOTS = ("up_proj", "down_proj")
    _HF_FUSED_EXPERT_KEYS = ("up_proj", "down_proj")
    _SHARED_EXPERT_ATTRS = ("shared_experts",)
    _SHARED_EXPERT_REQUIRED = True
    _LATENT_PROJ_ATTRS = ("fc1_latent_proj", "fc2_latent_proj")

    # The lazy loader's per-expert fusion map is keyed on gate/up/down GLU triples, where a bare
    # ``up_proj`` is the second GLU half; this layout loads through from_pretrained and is patched.
    _supports_lazy_loading = False
    _supports_weight_sync = False
    _WEIGHT_SYNC_REFUSAL_REASON = (
        "the non-gated per-expert export has not been validated end to end against a rollout engine's NemotronH loader"
    )

    @classmethod
    def _detect_hidden_dim(cls, router: nn.Module | None, experts: nn.Module) -> int:
        """The width the experts read, which is what DeepEP dispatches: the latent width when the block
        projects into one, the hidden size otherwise."""
        return experts.up_proj.shape[2]

    def _init_expert_compute(self, original_layer: nn.Module, experts: nn.Module) -> None:
        super()._init_expert_compute(original_layer, experts)
        for attr in self._LATENT_PROJ_ATTRS:
            setattr(self, attr, getattr(original_layer, attr))

    def _init_expert_params(self, experts: nn.Module, weights_already_sharded: bool = False) -> None:
        """This rank's ``[E_local, H_in, M]`` up and ``[E_local, M, H_in]`` down, the intermediate axis
        sliced to this rank's range under expert TP. With no gate half there is no pair to keep aligned,
        so ETP narrows each projection directly."""
        start, end = (0, self.experts_per_rank) if weights_already_sharded else (self.expert_start, self.expert_end)
        up = experts.up_proj.data[start:end].transpose(1, 2)
        down = experts.down_proj.data[start:end].transpose(1, 2)
        if self.expert_tp_size > 1:
            up, down = self._etp_narrow(up, 2), self._etp_narrow(down, 1)
        self.up_proj = nn.Parameter(up.contiguous())
        self.down_proj = nn.Parameter(down.contiguous())

    def replicated_named_params(self) -> list:
        return self._submodule_named_params(
            [(self._shared_expert_attr, getattr(self, self._shared_expert_attr))]
            + [(attr, getattr(self, attr)) for attr in self._LATENT_PROJ_ATTRS]
        )

    def _router_logits(self, flat: torch.Tensor) -> torch.Tensor:
        """The router matmul alone, in fp32 as ``NemotronHTopkRouter`` runs it. Calling the router would
        also run its own top-k, while selection stays here (balancing bias, routing replay)."""
        with torch.amp.autocast("cuda", enabled=False):
            return F.linear(flat.float(), self.gate.weight.float())

    def _dispatch_compute_combine(
        self, flat: torch.Tensor, experts: torch.Tensor, weights: torch.Tensor, input_dtype: torch.dtype
    ) -> torch.Tensor:
        latent = super()._dispatch_compute_combine(self.fc1_latent_proj(flat), experts, weights, input_dtype)
        return self.fc2_latent_proj(latent)

    def _expert_forward(self, idx: int, x: torch.Tensor) -> torch.Tensor:
        return self._expert_proj_single(idx, self.act_fn(self._expert_proj_single(idx, x, "up_proj")), "down_proj")

    def _compute_experts_gmm(
        self, tokens: torch.Tensor, experts: torch.Tensor, weights: torch.Tensor, output_dtype: torch.dtype
    ) -> torch.Tensor:
        def compute(sorted_tokens, offs, _eids):
            # autocast runs relu²'s square in fp32; grouped_mm takes one dtype.
            hidden = self.act_fn(self._expert_proj(sorted_tokens, "up_proj", offs, output_dtype)).to(output_dtype)
            return self._expert_proj(hidden, "down_proj", offs, output_dtype)

        return self._compute_experts_with_grouped_mm(tokens, experts, weights, output_dtype, compute)

    def _warm_expert_activation(self, gate_up: torch.Tensor) -> torch.Tensor:
        return self.act_fn(gate_up)

    def _init_summary_extras(self, original_layer: nn.Module) -> tuple[str, ...]:
        latent = not isinstance(self.fc1_latent_proj, nn.Identity)
        return (f"latent_dim={self.hidden_dim if latent else None}",)

    def gather_expert_state_dict(self, device: str = "cpu", merge_lora: bool = False, retain: bool = True) -> dict:
        """Full experts in the hub's per-expert layout (``experts.{i}.{up,down}_proj.weight``), the
        spelling transformers merges back on load and the rollout engines read."""
        self._reject_merge_lora_under_expert_tp(merge_lora)
        take = partial(self._materialize_expert_weight, merge_lora=merge_lora)
        up = self._ep_all_gather_cat(self._tp_all_gather_cat(take("up_proj"), dim=2))
        down = self._ep_all_gather_cat(self._tp_all_gather_cat(take("down_proj"), dim=1))
        if not retain:
            return {}
        return self._per_expert_state(up.to(device), down.to(device))

    @classmethod
    def merge_shards_to_hf(cls, prefix: str, params: dict) -> dict:
        """Inverse of :meth:`gather_expert_state_dict` over merged matmul-convention shards."""
        unexpected = set(params) - set(cls._EXPERT_WEIGHT_ATTR_ROOTS)
        if unexpected:
            raise ValueError(
                f"{cls.__name__} sharded merge found unexpected expert params {sorted(unexpected)} — "
                f"the per-expert split covers exactly {list(cls._EXPERT_WEIGHT_ATTR_ROOTS)}; refusing to drop them."
            )
        state = cls._per_expert_state(params["up_proj"], params["down_proj"])
        return {f"{prefix}.{key}": tensor for key, tensor in state.items()}

    @staticmethod
    def _per_expert_state(up: torch.Tensor, down: torch.Tensor) -> dict:
        """Matmul-convention ``up [E, H_in, M]`` / ``down [E, M, H_in]`` → one ``F.linear`` weight per expert."""
        state = {}
        for i in range(up.shape[0]):
            state[f"experts.{i}.up_proj.weight"] = up[i].transpose(0, 1).contiguous()
            state[f"experts.{i}.down_proj.weight"] = down[i].transpose(0, 1).contiguous()
        return state
