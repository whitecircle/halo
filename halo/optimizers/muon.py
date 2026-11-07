"""Muon over gram-newton-schulz: 2D+ hidden matrices take Newton-Schulz orthogonalization, 1D params and the
embedding/head an internal AdamWBF16. Fused Triton kernels replace upstream's per-batch loops and torch.compile."""

import contextlib
import logging
from collections.abc import Collection, Sequence
from typing import Any

import torch
import triton
import triton.language as tl
from gram_newton_schulz import Muon as UpstreamMuon
from gram_newton_schulz.muon.muon_utils.muon_matrix_split_utils import get_newton_schulz_inputs_from_gradients
from gram_newton_schulz.muon.muon_utils.muon_opt_utils import adjust_lr_rms_norm

from halo.config_validation import differs_from_default
from halo.models.introspection.module_tree import embedding_module
from halo.models.introspection.param_roles import EMBEDDING_HEAD_MARKERS
from halo.optimizers.adamw_bf16 import AdamWBF16
from halo.optimizers.param_groups import decay_groups
from halo.optimizers.step_skip import DeviceStepSkipMixin
from halo.optimizers.stochastic_rounding import (
    SR_BLOCK_SIZE,
    require_param_names,
    sr_param_index,
    sr_round_to_bf16,
    sr_seed_pair,
)
from halo.runtime.consensus import rank_consensus
from halo.runtime.dtensor import to_local
from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)

# Apart from the internal AdamWBF16's key, so neither's noise depends on how many params the other routed.
_SR_KEY = 0xC0FFEE
# Muon's recipe wants a shorter second-moment horizon than HF's adam_beta2 default (0.999).
DEFAULT_SCALAR_BETAS: tuple[float, float] = (0.9, 0.95)
# Same-shape params are stacked for the batched Newton-Schulz; the chunk size bounds that peak.
_GNS_CHUNK_SIZE = 32


def _muon_kernels_available() -> bool:
    """Probed by a call, since the quack/cutlass ABI can drift under a successful import."""
    if not torch.cuda.is_available():
        return False
    try:
        from quack.gemm_interface import gemm_symmetric  # noqa: PLC0415 — optional dep probe

        # Mirrors the real call, fp16 with float alpha/beta, as int literals raise a non-ABI type error.
        probe = torch.eye(16, device="cuda", dtype=torch.float16)
        gemm_symmetric(probe, probe, C=probe, alpha=1.0, beta=1.0)
        return True
    except Exception:
        return False


@triton.jit
def _momentum_nesterov_kernel(
    m_ptr,
    g_ptr,
    u_ptr,
    momentum_val,
    n_elements,
    skip_ptr,
    HAS_SKIP: tl.constexpr,
    NESTEROV: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused momentum update and Nesterov lookahead for one param."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    m_mask = mask
    if HAS_SKIP:
        m_mask = mask & (tl.load(skip_ptr) == 0)

    m = tl.load(m_ptr + offsets, mask=mask).to(tl.float32)
    g = tl.load(g_ptr + offsets, mask=mask).to(tl.float32)

    m = momentum_val * m + g
    tl.store(m_ptr + offsets, m.to(tl.bfloat16), mask=m_mask)

    if NESTEROV:  # noqa: SIM108  tl.constexpr branch in a @triton.jit kernel; keep explicit for codegen
        u = momentum_val * m + g
    else:
        u = m
    tl.store(u_ptr + offsets, u.to(tl.bfloat16), mask=mask)


@triton.jit
def _weight_decay_update_kernel(
    p_ptr,
    u_ptr,
    decay_factor,
    n_elements,
    seed,
    skip_ptr,
    HAS_SKIP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused ``p = p * decay - u`` with a stochastically rounded write. The orthogonalized update is ~lr, below the
    bf16 weight ULP, so nearest rounding would truncate it to zero."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    if HAS_SKIP:
        mask = mask & (tl.load(skip_ptr) == 0)

    p = tl.load(p_ptr + offsets, mask=mask).to(tl.float32)
    u = tl.load(u_ptr + offsets, mask=mask).to(tl.float32)

    p = p * decay_factor - u
    rand_noise = (tl.rand(seed, offsets) * 65536.0).to(tl.int32)
    tl.store(p_ptr + offsets, sr_round_to_bf16(p, rand_noise), mask=mask)


def _is_embedding_or_head(name: str) -> bool:
    """Fallback for models without embedding accessors. ``shared`` (T5's tied embedding) and ``head`` (a PP stage's
    relocated head) match whole path components, as a substring would catch MoE shared_expert matrices."""
    return any(marker in name for marker in EMBEDDING_HEAD_MARKERS) or not {"shared", "head"}.isdisjoint(
        name.split(".")
    )


def _embedding_and_head_param_ids(model: torch.nn.Module) -> set[int]:
    """Read from the module tree, so a family naming its embedding ``tok_embeddings`` or its head ``output`` routes
    correctly. A PP stage holding neither yields an empty set."""
    targets: list[torch.nn.Module] = []
    for which in ("input", "output"):
        # A stage/backbone without that end of the network raises rather than returning None.
        with contextlib.suppress(AttributeError, NotImplementedError):
            module = embedding_module(model, which)
            if isinstance(module, torch.nn.Module):
                targets.append(module)
    for attr in ("score", "classifier"):
        module = getattr(model, attr, None)
        if isinstance(module, torch.nn.Module):
            targets.append(module)
    return {id(p) for module in targets for p in module.parameters(recurse=True)}


def _advance_steps_and_collect_grads(state: dict, params: list, names: list[str]) -> tuple[list, list[int]]:
    """Count a step for every param and return those with grads, each with its SR seed. A param with no grad counts
    too, so replicas whose grad presence differs agree on its seed's step."""
    with_grad, seeds = [], []
    for p, name in zip(params, names, strict=True):
        param_state = state[p]
        param_state["step"] = param_state.get("step", 0) + 1
        if p.grad is None:
            continue
        if "momentum" not in param_state:
            param_state["momentum"] = torch.zeros_like(p)
        with_grad.append(p)
        seeds.append(sr_seed_pair(_SR_KEY, param_state["step"], sr_param_index(name))[0])
    return with_grad, seeds


def _resolve_adjust_lr_fn(adjust_lr):
    if adjust_lr == "rms_norm":
        return adjust_lr_rms_norm
    if callable(adjust_lr) or adjust_lr is None:
        return adjust_lr
    raise ValueError(
        f"Unsupported adjust_lr {adjust_lr!r} for the fused Muon step: expected 'rms_norm', a callable, or None."
    )


class Muon(DeviceStepSkipMixin, UpstreamMuon):
    """Upstream Muon with the fused Triton matrix step and a state dict that carries the scalar leg. A shard's
    ``OptimizerStateFingerprint`` records the class name, and a resume whose recorded class differs is rejected."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        require_param_names(self.param_groups, type(self).__name__)
        if self.scalar_optimizer is not None:
            # Upstream's torch.compile around the scalar step only adds graph breaks around AdamWBF16's kernel.
            self._compiled_scalar_step = self.scalar_optimizer.step

    def skip_next_step_if(self, flag: torch.Tensor) -> None:
        super().skip_next_step_if(flag)
        if self.scalar_optimizer is not None:
            self.scalar_optimizer.skip_next_step_if(flag)

    def settle_step(self, skipped: bool) -> None:
        super().settle_step(skipped)
        if self.scalar_optimizer is not None:
            self.scalar_optimizer.settle_step(skipped)

    def _muon_step(self, param_groups: list[dict]) -> None:
        skip = self._take_step_skip(param_groups)
        for group in param_groups:
            if group.get("param_split_fn") is not None or group.get("param_recombine_fn") is not None:
                raise ValueError(
                    "The fused Muon step orthogonalizes whole matrices in place: a param group's "
                    "param_split_fn / param_recombine_fn is not supported."
                )
            group_params, sr_seeds = _advance_steps_and_collect_grads(
                self.state, group["params"], group["param_names"]
            )
            if not group_params:
                continue
            updates = self._nesterov_updates(group_params, group["momentum"], group["nesterov"], skip)
            self._orthogonalize_in_place(
                updates, group["lr"], _resolve_adjust_lr_fn(group.get("adjust_lr", "rms_norm"))
            )
            decay_factor = 1.0 - group["lr"] * group["weight_decay"]
            for p, u, seed in zip(group_params, updates, sr_seeds, strict=True):
                p_local = to_local(p.detach())
                n = p_local.numel()
                _weight_decay_update_kernel[(triton.cdiv(n, SR_BLOCK_SIZE),)](
                    p_local.view(-1),
                    u.view(-1),
                    decay_factor,
                    n,
                    seed,
                    skip if skip is not None else p_local,
                    HAS_SKIP=skip is not None,
                    BLOCK_SIZE=SR_BLOCK_SIZE,
                )
                torch.autograd.graph.increment_version(p)  # the raw-pointer store above is invisible to ATen

    def _nesterov_updates(
        self, params: list, momentum: float, nesterov: bool, skip: torch.Tensor | None
    ) -> list[torch.Tensor]:
        updates = []
        for p in params:
            m_local = to_local(self.state[p]["momentum"])
            u = torch.empty_like(m_local)
            n = m_local.numel()
            _momentum_nesterov_kernel[(triton.cdiv(n, SR_BLOCK_SIZE),)](
                m_local.view(-1),
                to_local(p.grad).view(-1),
                u.view(-1),
                momentum,
                n,
                skip if skip is not None else m_local,
                HAS_SKIP=skip is not None,
                NESTEROV=nesterov,
                BLOCK_SIZE=SR_BLOCK_SIZE,
            )
            updates.append(u.view(m_local.shape))
        return updates

    def _orthogonalize_in_place(self, updates: list[torch.Tensor], lr: float, adjust_lr_fn) -> None:
        """Same-shape matrices run stacked in ``_GNS_CHUNK_SIZE`` chunks, each copied straight back into its views, as
        stacking a whole large MoE's expert updates reaches tens of GB."""
        views_by_shape, _, _ = get_newton_schulz_inputs_from_gradients(updates, None)
        for shape, views in views_by_shape.items():
            scale = lr if adjust_lr_fn is None else adjust_lr_fn(lr, shape)
            for start in range(0, len(views), _GNS_CHUNK_SIZE):
                chunk = views[start : start + _GNS_CHUNK_SIZE]
                torch._foreach_copy_(chunk, self.newton_schulz(torch.stack(chunk)).mul_(scale).unbind(0))

    def state_dict(self) -> dict:
        """Upstream serializes only ``self.state``, which would drop the AdamWBF16 moments on resume."""
        sd = super().state_dict()
        if self.scalar_optimizer is not None:
            index = {id(p): i for i, p in enumerate(p for g in self.param_groups for p in g["params"])}
            for p, st in self.scalar_optimizer.state.items():
                sd["state"][index[id(p)]] = st
        return sd

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        # Upstream's ``param_groups`` property shadows the slot the base restore writes, so restored hyperparameters
        # land in an unread attribute.
        for live, saved in zip(self.param_groups, state_dict["param_groups"], strict=True):
            live.update({k: v for k, v in saved.items() if k != "params"})
        if self.scalar_optimizer is None:
            return
        scalar_ids = {id(p) for g in self.scalar_optimizer.param_groups for p in g["params"]}
        for p in [p for p in self.state if id(p) in scalar_ids]:
            self.scalar_optimizer.state[p] = self.state.pop(p)


def build_muon_optimizer(
    model: torch.nn.Module,
    *,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    momentum: float = 0.95,
    nesterov: bool = True,
    scalar_lr: float | None = None,
    scalar_betas: tuple[float, float] = DEFAULT_SCALAR_BETAS,
    scalar_eps: float = 1e-8,
    scalar_weight_decay: float | None = None,
    ns_use_kernels: bool = True,
    ns_algorithm: str = "gram_newton_schulz",
    decay_parameters: Collection[str] | None = None,
) -> Muon:
    """``ns_use_kernels`` falls back to pure torch when the CUDA kernels cannot run on every rank."""
    if ns_use_kernels:
        # A mixed NS backend across ranks is equivalent but not bit-identical, which drifts replicas apart.
        kernels_ok = rank_consensus(_muon_kernels_available()).all_ok
        if not kernels_ok:
            logger.warning(
                "Muon Newton-Schulz CUDA kernels (quack/cutlass) are unavailable on at least one "
                "rank; all ranks fall back to the pure-torch orthogonalization path "
                "(ns_use_kernels=False) to keep replica updates bitwise identical."
            )
            ns_use_kernels = False

    scalar_lr = scalar_lr if scalar_lr is not None else lr
    scalar_weight_decay = scalar_weight_decay if scalar_weight_decay is not None else weight_decay

    muon_named: list[tuple[str, torch.nn.Parameter]] = []
    scalar_named: list[tuple[str, torch.nn.Parameter]] = []
    non_bf16_muon: list[str] = []
    embedding_head_ids = _embedding_and_head_param_ids(model)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Embedding/head rows index the sparsely updated vocab, where Newton-Schulz is empirically harmful.
        if param.ndim >= 2 and id(param) not in embedding_head_ids and not _is_embedding_or_head(name):
            if param.dtype != torch.bfloat16:
                non_bf16_muon.append(name)
            muon_named.append((name, param))
        else:
            scalar_named.append((name, param))

    # The fused step stores bf16 into the param pointer, so a non-bf16 param pays fp32 memory for bf16 math.
    if non_bf16_muon:
        raise ValueError(
            f"Muon requires bf16 hidden weights: {len(non_bf16_muon)} Muon-routed parameter(s) are not "
            f"bf16 (e.g. '{non_bf16_muon[0]}'). Disable fp32 master-weight knobs "
            f"(fp32_non_ep_params / fp32_router / fp32_experts) or choose a different optim."
        )

    scalar_optimizer = None
    scalar_groups = decay_groups(scalar_named, decay_parameters, scalar_weight_decay)
    if scalar_groups:
        # torch.optim.AdamW's nearest-rounded bf16 writes truncate every sub-ULP step (norm weights would never move),
        # and its foreach/fused paths reject the DTensor/plain mix under EP/TP.
        scalar_optimizer = AdamWBF16(
            scalar_groups,
            lr=scalar_lr,
            betas=scalar_betas,
            eps=scalar_eps,
            weight_decay=scalar_weight_decay,
        )

    # gram_newton_schulz keys its 3D->2D split on the first tensor's dim-0, so a 3D-first group whose 3D params differ
    # in dim-0 (qwen3.5) raises IndexError.
    muon_named.sort(key=lambda named: named[1].ndim)

    muon_groups = decay_groups(muon_named, decay_parameters, weight_decay)
    if not muon_groups:
        raise ValueError("No 2D+ parameters found for Muon. Muon requires at least one parameter with ndim >= 2.")

    return Muon(
        params=muon_groups,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        nesterov=nesterov,
        ns_use_kernels=ns_use_kernels,
        ns_algorithm=ns_algorithm,
        scalar_optimizer=scalar_optimizer,
    )


def muon_optimizer_from_args(model: torch.nn.Module, args: Any, decay_parameters: Sequence[str]):
    # The scalar leg's tuned betas differ from HF's, so a beta is forwarded only when set explicitly.
    default_beta1, default_beta2 = DEFAULT_SCALAR_BETAS
    scalar_betas = (
        args.adam_beta1 if differs_from_default(args, "adam_beta1") else default_beta1,
        args.adam_beta2 if differs_from_default(args, "adam_beta2") else default_beta2,
    )
    optimizer = build_muon_optimizer(
        model,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        scalar_betas=scalar_betas,
        scalar_eps=args.adam_epsilon,
        decay_parameters=decay_parameters,
    )
    if is_global_main_process():
        # The embedding and head are 2D yet routed to the scalar leg.
        scalar = optimizer.scalar_optimizer
        scalar_count = sum(p.numel() for g in scalar.param_groups for p in g["params"]) if scalar is not None else 0
        muon_count = sum(p.numel() for p in model.parameters() if p.requires_grad) - scalar_count
        logger.info(
            f"Muon optimizer: {muon_count / 1e6:.1f}M params (Newton-Schulz), "
            f"{scalar_count / 1e6:.1f}M params (AdamW scalar)"
        )
    return optimizer
