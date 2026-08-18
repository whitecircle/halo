"""Checkpoint gather and shard-merge for EP MoE layers.

:class:`EPExpertGatherMixin` is the export half of :class:`~src.distributed.expert_parallel.base_layer.EPMoELayerBase`:
it reassembles this rank's expert shards across the expert-TP and dispatch-EP groups into the layout
the family's HuggingFace checkpoint spells, and owns the shard-merge inverse of every gather it
performs. A family declares its layout (``_PER_EXPERT_UNFUSED_KEYS`` / ``_HUB_PER_EXPERT_KEYS``) or
overrides the gather AND its merge together — never one alone, which ``__init_subclass__`` refuses.

The instance gathers are COLLECTIVE over those groups: each rank must call them, and ``retain=False``
joins every gather without assembling the result. The classmethod merge half is pure — it runs
single-process in ``scripts/after_training/merge_ep_shards.py``, off tensors already on disk.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, distribute_tensor

from src.distributed.runtime import materialize_dtensor


class EPExpertGatherMixin:
    """Expert-weight export: gather to the family's checkpoint layout, and merge per-rank shards back.

    Mixed into :class:`EPMoELayerBase`, whose construction owns the state read here (the expert
    ranges, the expert-TP and dispatch groups, the grouped-LoRA adapters).
    """

    # (gate, up, down) per-expert hub names; the base gather splits the fused [gate; up] training
    # layout into them automatically.
    _PER_EXPERT_UNFUSED_KEYS: tuple[str, str, str] | None = None

    # The same names WITHOUT asserting the BASE gather does the split — families that emit per-expert
    # themselves (Qwen3, Bailing) or gather fused (Qwen3.5/3.6, DSv4). Read :meth:`hub_per_expert_keys`.
    _HUB_PER_EXPERT_KEYS: tuple[str, str, str] | None = None

    # Fused 3D expert tensor names as an HF CHECKPOINT spells them (dim 0 = expert); the lazy loader
    # slices these per rank and the shard merge accepts exactly these.
    _HF_FUSED_EXPERT_KEYS: tuple[str, ...] = (
        "gate_up_proj",
        "gate_up_proj_bias",
        "down_proj",
        "down_proj_bias",
    )

    def __init_subclass__(cls, **kwargs):
        """Enforce the layout contract this class declares, on every EP layer that mixes it in.

        Each raise closes a way for the declared layout and the gathered one to drift apart silently —
        a checkpoint whose expert keys no serving engine reads, with no shape, dtype or key error.
        """
        super().__init_subclass__(**kwargs)
        overrides_gather = "gather_expert_state_dict" in vars(cls)
        if getattr(cls, "_PER_EXPERT_UNFUSED_KEYS", None) is not None and overrides_gather:
            raise TypeError(
                f"{cls.__name__} resolves _PER_EXPERT_UNFUSED_KEYS (declared or inherited) while "
                f"defining its own gather_expert_state_dict — the override bypasses the base gather "
                f"that applies the per-expert split, so the declaration would be silently ignored. "
                f"Keep exactly one: the attribute (base gather splits automatically) or the override "
                f"(and set _PER_EXPERT_UNFUSED_KEYS = None on the overriding class)."
            )
        # Gather and merge are inverses: overriding one alone writes a checkpoint whose layout silently
        # differs from this family's own gathered save.
        if overrides_gather != ("merge_shards_to_hf" in vars(cls)):
            raise TypeError(
                f"{cls.__name__} overrides exactly one of gather_expert_state_dict / "
                f"merge_shards_to_hf. They are inverses: the sharded-merge output would no longer "
                f"match this family's gathered save. Override both, or neither."
            )
        if cls._PER_EXPERT_UNFUSED_KEYS is not None and cls._HUB_PER_EXPERT_KEYS is not None:
            raise TypeError(
                f"{cls.__name__} resolves both _PER_EXPERT_UNFUSED_KEYS and "
                f"_HUB_PER_EXPERT_KEYS (declared or inherited). They name the same thing — this "
                f"family's per-expert hub layout — but only the first also asserts that the BASE gather "
                f"performs the split, so keeping both lets them drift and leaves hub_per_expert_keys() "
                f"answering from whichever wins. A subclass that must switch paths sets the inherited "
                f"one to None explicitly."
            )

    @classmethod
    def hub_per_expert_keys(cls) -> tuple[str, str, str] | None:
        """``(gate, up, down)`` names this family's HF checkpoint stores ONE TENSOR PER EXPERT under, or
        ``None`` when its checkpoint keeps some other layout (fused halves, interleaved fused, a nested
        module per expert).

        Unioned over the two declarations, so a caller never has to know which gather path produced
        the layout. ``scripts/after_training/unfuse_moe_experts.py`` refuses a family answering ``None``:
        every per-expert name it could emit is one nothing reads.
        """
        return cls._PER_EXPERT_UNFUSED_KEYS or cls._HUB_PER_EXPERT_KEYS

    @staticmethod
    def _all_gather_cat(tensor: torch.Tensor, dim: int, group, world_size: int) -> torch.Tensor:
        """All-gather ``tensor`` across ``group`` and concat along ``dim`` (identity when ``world_size <= 1``).

        Shards are EQUAL by construction — :meth:`EPConfig.finalize_expert_assignment` rejects
        ``num_experts % ep_size`` and :meth:`_etp_shard_size` rejects an indivisible expert
        intermediate — so one collective suffices on this save / vLLM-sync hot path.

        ``dim == 0`` — every EP expert-axis gather — receives straight into one preallocated output:
        a shard list plus a ``cat`` holds the whole gathered tensor TWICE on every rank of the group,
        and the expert axis is exactly where a fine-grained MoE keeps its parameters. The other dims
        (expert-TP, which splits the intermediate) still need the list + concatenation.
        """
        if world_size <= 1:
            return tensor
        tensor = tensor.contiguous()
        if dim == 0:
            gathered = torch.empty(
                (tensor.shape[0] * world_size, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device
            )
            dist.all_gather_into_tensor(gathered, tensor, group=group)
            return gathered
        shards = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(shards, tensor, group=group)
        return torch.cat(shards, dim=dim)

    def _tp_all_gather_cat(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        """Reconstruct a full expert tensor from this rank's expert-TP shards."""
        return self._all_gather_cat(tensor, dim, self.expert_tp_group, self.expert_tp_size)

    def _ep_all_gather_cat(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather all experts across the dispatch-EP group.

        Always dim 0: EP shards the expert bank itself, so the concatenation axis is the expert axis
        by construction — unlike :meth:`_tp_all_gather_cat`, whose axis follows the projection."""
        return self._all_gather_cat(tensor, 0, self.ep_config.dispatch_ep_group, self.ep_config.ep_size)

    def _materialize_expert_weight(self, attr: str, *, merge_lora: bool = False) -> torch.Tensor:
        """This rank's full (FSDP-gathered) expert weight ``attr`` in matmul convention, with the grouped
        -LoRA delta ``scaling·(A@B)`` folded in when ``merge_lora`` and ``attr`` carries an adapter.

        The single fold seam every family gather routes base weights through, so the vLLM weight sync serves
        the TRAINED experts (else it ships the frozen base and the generator runs off-policy). Folds on the
        full PLAIN tensor :func:`~src.distributed.runtime.materialize_dtensor` returns — no sharding
        assumption. ``expert_tp_size==1`` only (expert-LoRA + ETP is rejected at config)."""
        base = materialize_dtensor(getattr(self, attr).data)
        if not merge_lora or attr not in self._expert_lora_attrs:
            return base
        lora_a = materialize_dtensor(getattr(self, f"{attr}_lora_A").data)
        lora_b = materialize_dtensor(getattr(self, f"{attr}_lora_B").data)
        delta = self.expert_lora_scaling * torch.bmm(lora_a.float(), lora_b.float())
        return (base.float() + delta).to(base.dtype)

    def _reject_merge_lora_under_expert_tp(self, merge_lora: bool) -> None:
        """Guard every gather against folding a REAL adapter under expert TP, where the sharded expert
        params have no fold seam.

        Gated on :attr:`has_expert_lora`, not on ``merge_lora`` alone: the vLLM sync asks for the fold
        unconditionally (it cannot know which layers carry adapters), so raising on the flag would reject
        every full fine-tune under ``expert_tp_size > 1``. Defense in depth — :class:`EPConfig` already
        refuses expert LoRA there — because the fold into the sharded layout is ambiguous, and failing
        beats shipping the frozen base experts to vLLM.
        """
        if merge_lora and self.has_expert_lora and self.expert_tp_size > 1:
            raise NotImplementedError(
                f"{type(self).__name__}: merge_lora is not implemented for the expert-TP gather "
                f"(expert_tp_size={self.expert_tp_size}) — the export would carry the FROZEN base "
                f"experts. Gather with expert_tp_size=1, or implement the fold for the sharded layout."
            )

    def gather_expert_state_dict(self, device: str = "cpu", merge_lora: bool = False, retain: bool = True) -> dict:
        """Gather this layer's EP-distributed experts into a checkpoint state dict, keyed relative to the
        EP layer (e.g. ``experts.gate_up_proj``).

        Returns the fused-GLU layout of :meth:`_gather_fused_expert_state_dict`; families whose hub
        checkpoint / vLLM loader store one tensor per expert declare ``_PER_EXPERT_UNFUSED_KEYS`` and
        the fused result is split into that layout here — no per-family override needed.
        ``merge_lora`` folds the expert-LoRA delta for the vLLM sync; off for checkpointing.

        ``retain=False`` enters every collective and returns ``{}``: the gathers are group-wide and a
        rank that skipped one would hang its peers, but only the writer needs the ASSEMBLED tensors. It
        is the post-gather layout work — split, re-interleave, transpose+``contiguous``, host copy —
        that every non-writing rank skips.
        """
        fused = self._gather_fused_expert_state_dict(device, merge_lora=merge_lora, retain=retain)
        if retain and self._PER_EXPERT_UNFUSED_KEYS is not None:
            return self._unfuse_fused_to_per_expert(fused)
        return fused

    def gather_expert_grads(self, device: str = "cpu") -> dict:
        """Grad-side sibling of :meth:`gather_expert_state_dict`: this layer's expert GRADIENTS
        reassembled over both expert axes into the canonical FUSED layout, whatever the family
        stores locally (the weight side additionally unfuses to the hub spelling; a grad check
        does not need that).

        One layout for every family is what makes an equivalence check mean anything: the expert
        parameter SET is layout-dependent (ETP splits the fused ``gate_up_proj``, grouped-GEMM splits
        it again into ``*_gmm``), so a check naming attributes compares nothing on a layout it did not
        anticipate — and reports success. Collective on the expert-TP and dispatch groups.
        """
        return self._gather_fused_experts(device, self._expert_grad)

    def _expert_grad(self, attr: str) -> torch.Tensor:
        """This rank's gradient for expert weight ``attr``.

        A missing grad means the backward never reached this expert; silently gathering zeros would
        turn an equivalence check into a tautology, so it raises instead. Full-gathered like the
        weight side, so an FSDP-sharded ep1 expert yields its whole gradient rather than a shard."""
        grad = getattr(self, attr).grad
        if grad is None:
            raise RuntimeError(
                f"{type(self).__name__}.{attr} has no gradient: the backward did not reach this expert parameter."
            )
        return materialize_dtensor(grad)

    def _gather_fused_expert_state_dict(
        self, device: str = "cpu", merge_lora: bool = False, retain: bool = True
    ) -> dict:
        """Fused-GLU contiguous-halves gather (``experts.gate_up_proj`` / ``experts.down_proj``)."""
        self._reject_merge_lora_under_expert_tp(merge_lora)
        return self._gather_fused_experts(
            device, partial(self._materialize_expert_weight, merge_lora=merge_lora), retain=retain
        )

    def _gather_fused_experts(self, device: str, take: Callable[[str], torch.Tensor], retain: bool = True) -> dict:
        """Reassemble the fused-GLU experts from whichever per-attribute tensor ``take`` returns.

        ``take`` is the only difference between the weight export and the gradient gather, so both
        share this one traversal of the expert axes. Tensors are stored in matmul convention (under
        ETP the gate/up halves live in separate params); reconstruct full tensors and transpose back
        to ``F.linear`` convention. ``retain=False`` runs every gather and drops the result before the
        transpose+``contiguous`` (see :meth:`gather_expert_state_dict`).
        """
        if self.expert_tp_size > 1:
            gate = self._tp_all_gather_cat(take("gate_proj"), dim=2)
            up = self._tp_all_gather_cat(take("up_proj"), dim=2)
            gate_up = torch.cat([gate, up], dim=2)
            down = self._tp_all_gather_cat(take("down_proj"), dim=1)
        elif not hasattr(self, "gate_up_proj"):
            # Separately-stored GLU (Qwen3, Bailing) — same shape the ETP branch rebuilds, so the fused
            # result is identical either way.
            gate_up = torch.cat([take("gate_proj"), take("up_proj")], dim=2)
            down = take("down_proj")
        else:
            gate_up = take("gate_up_proj")
            down = take("down_proj")
        gate_up = self._ep_all_gather_cat(gate_up)
        down = self._ep_all_gather_cat(down)
        if not retain:
            return {}
        return {
            "experts.gate_up_proj": gate_up.transpose(1, 2).contiguous().to(device),
            "experts.down_proj": down.transpose(1, 2).contiguous().to(device),
        }

    @classmethod
    def _unfuse_fused_to_per_expert(cls, fused: dict) -> dict:
        """Split the base fused gather into the per-expert hub layout named by
        ``_PER_EXPERT_UNFUSED_KEYS``.

        The fused ``gate_up_proj`` is the transformers conversion's ``[gate; up]`` concatenation
        along the output dim, so split the halves and the expert dim (no transpose — the fused
        gather already returns ``F.linear`` convention). A classmethod: the shard-merge side runs the
        SAME split with only the class in hand.
        """
        gate_key, up_key, down_key = cls._PER_EXPERT_UNFUSED_KEYS
        gate_up = fused["experts.gate_up_proj"]  # [E, 2M, H], halves [gate; up]
        down = fused["experts.down_proj"]  # [E, H, M]
        intermediate = gate_up.shape[1] // 2
        state = {}
        for i in range(gate_up.shape[0]):
            state[f"experts.{i}.{gate_key}.weight"] = gate_up[i, :intermediate].contiguous()
            state[f"experts.{i}.{up_key}.weight"] = gate_up[i, intermediate:].contiguous()
            state[f"experts.{i}.{down_key}.weight"] = down[i].contiguous()
        return state

    @classmethod
    def merge_shards_to_hf(cls, prefix: str, params: dict) -> dict:
        """Convert this family's CONCATENATED per-rank expert shards to its HF checkpoint layout.

        ``scripts/after_training/merge_ep_shards.py`` hands the merged runtime tensors (matmul
        convention, keyed by the wrapper attribute they were saved under) plus the module ``prefix``,
        and gets back the keys ``from_pretrained`` expects. On the class rather than in a table, so
        ``merged-from-sharded == gathered`` is structural: declaring the layout is what supports a family.

        Base = the fused-GLU path, mirroring :meth:`gather_expert_state_dict`: transpose to
        ``F.linear`` convention, then the same ``_PER_EXPERT_UNFUSED_KEYS`` split. A family overriding
        the gather must override this too (enforced in ``__init_subclass__``).
        """
        state = cls._merge_fused_shards(params)
        if cls._PER_EXPERT_UNFUSED_KEYS is not None:
            # The split emits weights only, so a fused bias here would be silently dropped.
            unexpected = set(state) - {"experts.gate_up_proj", "experts.down_proj"}
            if unexpected:
                raise ValueError(
                    f"{cls.__name__} sharded merge found unexpected expert params {sorted(unexpected)} — "
                    f"the per-expert split covers exactly gate_up_proj/down_proj; refusing to drop them."
                )
            state = cls._unfuse_fused_to_per_expert(state)
        return {f"{prefix}.{key}": tensor for key, tensor in state.items()}

    @classmethod
    def _merge_fused_shards(cls, params: dict) -> dict:
        """Merged fused expert shards (matmul convention ``[E,H,2M]`` / ``[E,M,H]``) → the fused
        ``F.linear`` layout :meth:`_gather_fused_expert_state_dict` returns, keyed relative to the EP
        layer. Biases pass through untransposed.

        Raises on any expert param outside :attr:`_HF_FUSED_EXPERT_KEYS` rather than dropping it: a
        silently discarded expert tensor produces a checkpoint that loads and is wrong.
        """
        unexpected = set(params) - set(cls._HF_FUSED_EXPERT_KEYS)
        if unexpected:
            raise ValueError(
                f"{cls.__name__} sharded merge found unexpected expert params {sorted(unexpected)} — "
                f"the fused merge covers exactly {sorted(cls._HF_FUSED_EXPERT_KEYS)}; refusing to drop them."
            )
        state = {}
        if "gate_up_proj" in params:
            state["experts.gate_up_proj"] = params["gate_up_proj"].transpose(1, 2).contiguous()
        if "gate_up_proj_bias" in params:
            state["experts.gate_up_proj_bias"] = params["gate_up_proj_bias"]
        if "down_proj" in params:
            state["experts.down_proj"] = params["down_proj"].transpose(1, 2).contiguous()
        if "down_proj_bias" in params:
            state["experts.down_proj_bias"] = params["down_proj_bias"]
        return state

    @classmethod
    def _merge_individual_glu_shards(cls, params: dict) -> dict:
        """Separately-stored gate/up/down shards → the family's per-expert hub layout (nn.Linear
        convention) — the merge-side inverse of :meth:`_gather_individual_glu_state_dict`, shared by
        Qwen3 and Bailing. Hub names come from ``_HUB_PER_EXPERT_KEYS``, so the two sides
        cannot spell the layout differently."""
        gate_key, up_key, down_key = cls._individual_glu_hub_keys()
        state = {}
        for i in range(params["gate_proj"].shape[0]):
            state[f"experts.{i}.{gate_key}.weight"] = params["gate_proj"][i].transpose(0, 1).contiguous()
            state[f"experts.{i}.{up_key}.weight"] = params["up_proj"][i].transpose(0, 1).contiguous()
            state[f"experts.{i}.{down_key}.weight"] = params["down_proj"][i].transpose(0, 1).contiguous()
        return state

    @classmethod
    def _individual_glu_hub_keys(cls) -> tuple[str, str, str]:
        """``_HUB_PER_EXPERT_KEYS``, or raise. A family routing its gather/merge through the
        separate-halves path without declaring the hub names would otherwise write experts under
        whatever the base happened to spell."""
        if cls._HUB_PER_EXPERT_KEYS is None:
            raise TypeError(
                f"{cls.__name__} uses the separate-halves per-expert gather/merge but declares no "
                f"_HUB_PER_EXPERT_KEYS, so the per-expert hub names are unknown."
            )
        return cls._HUB_PER_EXPERT_KEYS

    def _gather_separate_glu_full(self, merge_lora: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full (EP- and ETP-gathered) gate/up/down tensors for the separately-stored expert layouts.

        One traversal behind :meth:`_gather_individual_glu_state_dict`, kept separate so a family
        adding a second keying reuses the same gathers — a gather fix landing in one copy would
        silently desynchronize a family's exports.
        """
        self._reject_merge_lora_under_expert_tp(merge_lora)
        gate = self._ep_all_gather_cat(
            self._tp_all_gather_cat(self._materialize_expert_weight("gate_proj", merge_lora=merge_lora), 2)
        )
        up = self._ep_all_gather_cat(
            self._tp_all_gather_cat(self._materialize_expert_weight("up_proj", merge_lora=merge_lora), 2)
        )
        down = self._ep_all_gather_cat(
            self._tp_all_gather_cat(self._materialize_expert_weight("down_proj", merge_lora=merge_lora), 1)
        )
        return gate, up, down

    def _gather_individual_glu_state_dict(
        self, device: str = "cpu", merge_lora: bool = False, retain: bool = True
    ) -> dict:
        """Gather separately-stored gate/up/down expert weights into the family's per-expert hub layout
        ``experts.{i}.<gate|up|down>.weight`` (Qwen3, Bailing; names from
        ``_HUB_PER_EXPERT_KEYS``). ``merge_lora`` folds the expert-LoRA delta for the vLLM
        sync; ``retain=False`` gathers and drops (see :meth:`gather_expert_state_dict`)."""
        gate_key, up_key, down_key = self._individual_glu_hub_keys()
        gate, up, down = self._gather_separate_glu_full(merge_lora)
        if not retain:
            return {}
        state = {}
        for i in range(gate.shape[0]):
            state[f"experts.{i}.{gate_key}.weight"] = gate[i].transpose(0, 1).contiguous().to(device)
            state[f"experts.{i}.{up_key}.weight"] = up[i].transpose(0, 1).contiguous().to(device)
            state[f"experts.{i}.{down_key}.weight"] = down[i].transpose(0, 1).contiguous().to(device)
        return state

    def gather_fused_expert_state_dict(
        self, device: str = "cpu", merge_lora: bool = False, retain: bool = True
    ) -> dict:
        """This layer's experts in the CHECKPOINT-FUSED layout.

        A second layout exists only because the rollout engines disagree: vLLM's loader takes
        the per-expert tensors :meth:`gather_expert_state_dict` produces, SGLang's the fused ones
        transformers stores. Declared per family, since the fused spelling differs across the roster
        (interleaved, prefixed, ``linear_fc``).

        Raising is the base: :meth:`implements_fused_expert_layout` refuses a family declaring no fused
        layout before the engine is touched, so reaching this body means that gate was bypassed. An
        empty return cannot carry the refusal — ``retain=False`` legitimately returns ``{}`` off the sender.
        """
        raise NotImplementedError(
            f"{type(self).__name__} declares no fused expert layout, which this rollout engine "
            f"requires. Sending the per-expert layout instead is silently dropped or rejected on "
            f"arrival, with the engine already paused and partly written. Implement "
            f"gather_fused_expert_state_dict for this family, or use rollout_backend: vllm, which "
            f"takes the per-expert layout it already gathers."
        )

    @classmethod
    def implements_fused_expert_layout(cls) -> bool:
        """Whether this family declares :meth:`gather_fused_expert_state_dict`, or inherits the
        refusing default. Read before a sync starts, so a family with no fused layout is refused while
        the engine is still untouched."""
        return cls.gather_fused_expert_state_dict is not EPExpertGatherMixin.gather_fused_expert_state_dict

    def gather_expert_lora_state_dict(self, device: str = "cpu", retain: bool = True) -> dict:
        """Gather this layer's grouped LoRA adapters into a checkpoint dict, keyed relative to the layer.

        Adapters in matmul convention, gathered to full expert count across the dispatch-EP group.
        Never ETP-sharded: ``EPConfig`` refuses ``expert_lora`` with ``expert_tp_size > 1`` at
        construction. Returns ``{}`` when no expert LoRA. Collective: every rank must call it, and
        ``retain=False`` joins each gather without assembling the result (see
        :meth:`gather_expert_state_dict`)."""
        state: dict[str, torch.Tensor] = {}
        for attr in sorted(self._expert_lora_attrs):
            lora_a = materialize_dtensor(getattr(self, f"{attr}_lora_A").data)
            lora_b = materialize_dtensor(getattr(self, f"{attr}_lora_B").data)
            lora_a = self._ep_all_gather_cat(lora_a)
            lora_b = self._ep_all_gather_cat(lora_b)
            if not retain:
                continue
            state[f"experts.{attr}.lora_A"] = lora_a.contiguous().to(device)
            state[f"experts.{attr}.lora_B"] = lora_b.contiguous().to(device)
        return state

    def load_expert_lora_state_dict(self, layer_state: dict) -> None:
        """Load gathered grouped LoRA adapters (inverse of :meth:`gather_expert_lora_state_dict`).

        Slices this rank's expert range on dim 0, copies into the live adapter params. No expert-TP
        re-shard: ``EPConfig`` refuses ``expert_lora`` with ``expert_tp_size > 1`` at construction."""
        # A changed expert config between checkpoint and resume (GptOss renames attrs by grouped-GEMM /
        # ETP / arch) would otherwise leave adapters silently zero-init; an empty state is that mismatch too.
        expected_keys = {f"experts.{attr}.lora_{w}" for attr in self._expert_lora_attrs for w in ("A", "B")}
        if set(layer_state) != expected_keys:
            raise RuntimeError(
                f"Expert-LoRA resume mismatch in {type(self).__name__}: saved adapter keys "
                f"{sorted(layer_state)} do not match this layer's rebuilt adapters {sorted(expected_keys)}. "
                "Likely a use_grouped_gemm / expert_tp_size / GPU-arch change between checkpoint and "
                "resume — resume with the same expert configuration."
            )
        # sorted(): the DTensor branch below is a collective, and frozenset order varies per process
        # (unpinned PYTHONHASHSEED), so an unsorted loop lets ranks scatter in different orders.
        for attr in sorted(self._expert_lora_attrs):
            for which in ("A", "B"):
                key = f"experts.{attr}.lora_{which}"
                shard = layer_state[key][self.expert_start : self.expert_end]  # this rank's experts (dim 0)
                target = getattr(self, f"{attr}_lora_{which}")
                value = shard.to(dtype=target.dtype, device=target.device)
                # ep1 + fsdp_shard_ep1_experts makes these DTensors, where a plain copy_ raises; re-shard
                # onto the param's mesh first. No-op at ep_size>1 (plain, FSDP-ignored).
                if isinstance(target.data, DTensor):
                    value = distribute_tensor(value, target.data.device_mesh, target.data.placements)
                target.data.copy_(value)
