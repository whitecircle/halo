"""Wrapper for gram-newton-schulz's Muon with automatic parameter splitting: 2D+ params → Muon
(Newton-Schulz orthogonalization), 1D params and the embedding/head → an internal AdamW scalar
optimizer. Upstream's per-batch Python loops and torch.compile are replaced by fused Triton kernels
for the momentum and weight-decay updates (one launch per param).
"""

import contextlib
import logging
import random
from collections.abc import Collection, Sequence
from dataclasses import MISSING, fields
from typing import Any

import torch
import triton
import triton.language as tl
from gram_newton_schulz import Muon as UpstreamMuon
from gram_newton_schulz.muon.muon_utils.muon_matrix_split_utils import (
    get_newton_schulz_inputs_from_gradients,
    reconstruct_update_from_newton_schulz_outputs,
    scale_newton_schulz_outputs_with_adjusted_lr,
)
from gram_newton_schulz.muon.muon_utils.muon_opt_utils import (
    adjust_lr_rms_norm,
    get_or_initialize_muon_state,
)

from src.distributed.runtime import is_global_main_process, rank_consensus, to_local
from src.models.structure import EMBEDDING_HEAD_MARKERS
from src.optimizers.adamw_bf16 import BLOCK_SIZE, AdamWBF16
from src.optimizers.param_groups import decay_groups

logger = logging.getLogger(__name__)

# Rank-synchronized SR-seed stream, kept off global ``random`` (the data path advances that
# per-rank). It uses its own generator rather than the internal AdamWBF16's, so neither optimizer's
# noise depends on how many params the other routed.
_SR_RNG = random.Random(0xC0FFEE)

# Muon's recipe wants a shorter second-moment horizon than HF's adam_beta2 default (0.999).
DEFAULT_SCALAR_BETAS: tuple[float, float] = (0.9, 0.95)

# Same-shape params are stacked for the batched Newton-Schulz; the chunk size bounds that peak.
_GNS_CHUNK_SIZE = 32


def _muon_kernels_available() -> bool:
    """Whether the quack/cutlass Newton-Schulz kernel backend actually runs.

    A successful import is not sufficient, since the kernel ABI can drift and the call itself raise,
    so exercise it on a tiny tensor and let :func:`create_muon_optimizer` fall back to pure torch on
    any failure.
    """
    if not torch.cuda.is_available():
        return False
    try:
        from quack.gemm_interface import gemm_symmetric  # noqa: PLC0415 — optional dep probe

        # Mirror the real call: fp16, and FLOAT alpha/beta — int literals raise a non-ABI type error.
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
    NESTEROV: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused momentum update + nesterov computation for a single param."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    m = tl.load(m_ptr + offsets, mask=mask).to(tl.float32)
    g = tl.load(g_ptr + offsets, mask=mask).to(tl.float32)

    m = momentum_val * m + g
    tl.store(m_ptr + offsets, m.to(tl.bfloat16), mask=mask)

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
    BLOCK_SIZE: tl.constexpr,
):
    """Fused weight decay + param update with stochastic rounding: p = p*decay - u.

    The orthogonalized update has magnitude ~lr, below the bf16 weight ULP at typical learning
    rates, so nearest rounding would truncate it to zero. Stochastic rounding keeps the write
    unbiased.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    p = tl.load(p_ptr + offsets, mask=mask).to(tl.float32)
    u = tl.load(u_ptr + offsets, mask=mask).to(tl.float32)

    p = p * decay_factor - u

    p_bits = p.to(tl.int32, bitcast=True)
    rand_noise = (tl.rand(seed, offsets) * 65536.0).to(tl.int32)
    p_bits = (p_bits + rand_noise) & (-65536)
    tl.store(p_ptr + offsets, p_bits.to(tl.float32, bitcast=True).to(tl.bfloat16), mask=mask)


def _is_embedding_or_head(name: str) -> bool:
    """True if a parameter name refers to a token embedding or the output head (kept on AdamW under Muon).

    Name-based fallback for models exposing no embedding accessors;
    :func:`_embedding_and_head_param_ids` is the authoritative signal. T5's tied 'shared.weight'
    matches only as an exact path component, since a substring match would also catch MoE
    shared_expert FFN matrices, which belong on Muon.
    """
    # Exact component "head" catches a PP stage's relocated output head: vocab-indexed rows too.
    return any(marker in name for marker in EMBEDDING_HEAD_MARKERS) or not {"shared", "head"}.isdisjoint(
        name.split(".")
    )


def _embedding_and_head_param_ids(model: torch.nn.Module) -> set[int]:
    """Ids of every parameter owned by the token embedding, the output head, or a pooled score head.

    Read from the model's module tree rather than from parameter names, so a family naming its
    embedding ``tok_embeddings`` or its head ``output`` routes correctly without extending a marker
    list. Tied weights collapse to one id, and a PP stage holding neither returns an empty set.
    """
    targets: list[torch.nn.Module] = []
    for getter in ("get_input_embeddings", "get_output_embeddings"):
        fn = getattr(model, getter, None)
        if not callable(fn):
            continue
        # A stage/backbone without that end of the network raises rather than returning None.
        with contextlib.suppress(AttributeError, NotImplementedError):
            module = fn()
            if isinstance(module, torch.nn.Module):
                targets.append(module)
    for attr in ("score", "classifier"):
        module = getattr(model, attr, None)
        if isinstance(module, torch.nn.Module):
            targets.append(module)
    return {id(p) for module in targets for p in module.parameters(recurse=True)}


def _fused_momentum_nesterov(params, grads, momentums, momentum_val, nesterov):
    """Launch fused momentum+nesterov kernel for all params, return NS inputs."""
    ns_inputs = []
    for _, g, m in zip(params, grads, momentums, strict=True):
        m_local = to_local(m)
        g_local = to_local(g)
        u = torch.empty_like(m_local)
        n = m_local.numel()
        grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        _momentum_nesterov_kernel[grid](
            m_local.view(-1),
            g_local.view(-1),
            u.view(-1),
            momentum_val,
            n,
            NESTEROV=nesterov,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        ns_inputs.append(u.view(m_local.shape))
    return ns_inputs


def _collect_params_with_sr_seeds(params) -> tuple[list, list[int]]:
    """Split ``params`` into those with grads, drawing one SR seed per param unconditionally.

    Every param consumes its ``_SR_RNG`` draw in deterministic order, including when it is skipped
    for a missing grad, so rank-nonuniform grad presence cannot shift the seed stream and drift
    replicas apart.
    """
    with_grad, seeds = [], []
    for p in params:
        seed = _SR_RNG.randint(0, 2**30)
        if p.grad is not None:
            with_grad.append(p)
            seeds.append(seed)
    return with_grad, seeds


def _fused_weight_decay_update(params, updates, decay_factor, seeds):
    """Launch fused weight-decay + update kernel for all params (``seeds`` pre-drawn per param)."""
    for p, u, seed in zip(params, updates, seeds, strict=True):
        p_local = to_local(p.detach())
        u_flat = u.contiguous().view(-1)
        n = p_local.numel()
        grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        _weight_decay_update_kernel[grid](
            p_local.view(-1),
            u_flat,
            decay_factor,
            n,
            seed,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        torch.autograd.graph.increment_version(p)  # the raw-pointer store above is invisible to ATen


class Muon(UpstreamMuon):
    """Upstream Muon with the fused Triton matrix step and a scalar-optimizer-aware state dict.

    The class name is load-bearing: every optimizer shard records ``type(optimizer).__name__`` in its
    ``OptimizerStateFingerprint``, and a resume whose recorded class differs is rejected.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.scalar_optimizer is not None:
            # The scalar leg is AdamWBF16's own fused step; upstream's torch.compile wrapper around it
            # only adds graph breaks around a kernel launch.
            self._compiled_scalar_step = self.scalar_optimizer.step

    def _muon_step(self, param_groups: list[dict]) -> None:
        """Fused-Triton replacement for the upstream matrix step: fused momentum + nesterov, chunked
        GNS orthogonalization per shape group, fused weight decay + update.

        GNS batching stacks same-shape params (tens of GB on large MoE models), so it runs in
        ``_GNS_CHUNK_SIZE`` chunks whose entries are freed as they are consumed.
        """
        for group in param_groups:
            group_params, sr_seeds = _collect_params_with_sr_seeds(group["params"])
            if not group_params:
                continue

            lr = group["lr"]
            momentum_val = group["momentum"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            adjust_lr = group.get("adjust_lr", "rms_norm")
            param_split_fn = group.get("param_split_fn", None)
            param_recombine_fn = group.get("param_recombine_fn", None)

            if isinstance(adjust_lr, str) and adjust_lr == "rms_norm":
                adjust_lr_fn = adjust_lr_rms_norm
            elif callable(adjust_lr):
                adjust_lr_fn = adjust_lr
            elif adjust_lr is None:
                adjust_lr_fn = None
            else:
                # Upstream defines further string modes this step does not implement; reject them.
                raise ValueError(
                    f"Unsupported adjust_lr {adjust_lr!r} for the fused Muon step: "
                    f"expected 'rms_norm', a callable, or None."
                )

            grads = [p.grad for p in group_params]
            states = [get_or_initialize_muon_state(self.state, p) for p in group_params]
            momentums = [s["momentum"] for s in states]

            ns_inputs = _fused_momentum_nesterov(
                group_params,
                grads,
                momentums,
                momentum_val,
                nesterov,
            )

            ns_inputs_by_shape, shape_indices, split_metadata = get_newton_schulz_inputs_from_gradients(
                ns_inputs, param_split_fn
            )
            del ns_inputs  # entries now held by ns_inputs_by_shape

            orthogonalized_by_shape: dict[tuple, list[torch.Tensor]] = {}
            for shape, ns_inputs_for_shape in ns_inputs_by_shape.items():
                results_for_shape: list[torch.Tensor] = []
                for chunk_start in range(0, len(ns_inputs_for_shape), _GNS_CHUNK_SIZE):
                    chunk = ns_inputs_for_shape[chunk_start : chunk_start + _GNS_CHUNK_SIZE]
                    batched = torch.stack(chunk, dim=0)
                    for i in range(chunk_start, min(chunk_start + _GNS_CHUNK_SIZE, len(ns_inputs_for_shape))):
                        ns_inputs_for_shape[i] = None  # free so originals can be GC'd
                    result = self.newton_schulz(batched).clone()
                    del batched
                    results_for_shape.append(result)
                orthogonalized_by_shape[shape] = torch.cat(results_for_shape, dim=0)
                del results_for_shape

            del ns_inputs_by_shape

            orthogonalized_by_shape = scale_newton_schulz_outputs_with_adjusted_lr(
                orthogonalized_by_shape,
                lr,
                adjust_lr_fn,
            )
            orthogonalized = reconstruct_update_from_newton_schulz_outputs(
                orthogonalized_by_shape,
                shape_indices,
                split_metadata,
                param_recombine_fn,
            )
            del orthogonalized_by_shape

            decay_factor = 1.0 - lr * weight_decay
            _fused_weight_decay_update(group_params, orthogonalized, decay_factor, sr_seeds)

    def state_dict(self) -> dict:
        """Standard-shaped state dict with the scalar-optimizer state merged in.

        Upstream serializes only ``self.state``, so the internal AdamW moments would be absent from
        checkpoints and reset on resume; they are merged under the combined param_groups index space.
        """
        sd = super().state_dict()
        if self.scalar_optimizer is not None:
            index = {id(p): i for i, p in enumerate(p for g in self.param_groups for p in g["params"])}
            for p, st in self.scalar_optimizer.state.items():
                sd["state"][index[id(p)]] = st
        return sd

    def load_state_dict(self, state_dict: dict) -> None:
        """Inverse of :meth:`state_dict`: split restored per-param state back to the scalar optimizer."""
        super().load_state_dict(state_dict)
        # Upstream's ``param_groups`` property shadows the instance-dict slot the base restore writes,
        # so restored hyperparameters land in an unread attribute; copy them onto the live group dicts.
        for live, saved in zip(self.param_groups, state_dict["param_groups"], strict=True):
            live.update({k: v for k, v in saved.items() if k != "params"})
        if self.scalar_optimizer is None:
            return
        scalar_ids = {id(p) for g in self.scalar_optimizer.param_groups for p in g["params"]}
        for p in [p for p in self.state if id(p) in scalar_ids]:
            self.scalar_optimizer.state[p] = self.state.pop(p)


def create_muon_optimizer(
    model: torch.nn.Module,
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
    """Create a Muon optimizer with automatic 2D/non-2D parameter splitting.

    2D+ hidden weight matrices → Muon (Newton-Schulz); 1D params and the token embedding / output head →
    internal AdamW scalar optimizer. scalar_lr / scalar_weight_decay default to lr / weight_decay;
    ns_use_kernels auto-falls back to pure-torch when the CUDA kernels can't run.
    """
    if ns_use_kernels:
        # Per-rank probes can disagree (JIT hiccup, driver skew). A mixed NS backend is numerically
        # equivalent but not bit-identical, which drifts replicas apart, so the kernels are used
        # only when every rank has them.
        kernels_ok = rank_consensus(_muon_kernels_available())[0]
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

        # Route on identity first, name second: embedding/head rows index the sparsely-updated vocab,
        # where Newton-Schulz is empirically harmful. Names cover models exposing no accessors.
        if param.ndim >= 2 and id(param) not in embedding_head_ids and not _is_embedding_or_head(name):
            if param.dtype != torch.bfloat16:
                non_bf16_muon.append(name)
            muon_named.append((name, param))
        else:
            scalar_named.append((name, param))

    # The fused step stores bf16 into the param pointer, so a non-bf16 param pays fp32 memory for
    # bf16 math.
    if non_bf16_muon:
        raise ValueError(
            f"Muon requires bf16 hidden weights: {len(non_bf16_muon)} Muon-routed parameter(s) are not "
            f"bf16 (e.g. '{non_bf16_muon[0]}'). Disable fp32 master-weight knobs "
            f"(fp32_non_ep_params / fp32_router / fp32_experts) or choose a different optim."
        )

    scalar_optimizer = None
    scalar_groups = decay_groups(scalar_named, decay_parameters, scalar_weight_decay)
    if scalar_groups:
        # AdamWBF16 rather than torch.optim.AdamW: the scalar params are bf16 too, and a
        # nearest-rounding write truncates every sub-ULP step to zero (norm weights, ULP 3.9e-3,
        # would never move). It is also per-parameter, so the DTensor/plain mix under EP/TP that
        # foreach/fused AdamW rejects works.
        scalar_optimizer = AdamWBF16(
            scalar_groups,
            lr=scalar_lr,
            betas=scalar_betas,
            eps=scalar_eps,
            weight_decay=scalar_weight_decay,
        )

    # 2D first: gram_newton_schulz keys its 3D->2D split on the first tensor's dim-0, so a
    # 3D-first group whose 3D params differ in dim-0 (qwen3.5) raises IndexError. Per-param
    # updates are otherwise order-independent.
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


def _is_explicitly_set(args: Any, field_name: str) -> bool:
    """Whether ``args.<field_name>`` differs from its declared dataclass default (the user set it).

    The training-args classes are dataclasses and HF does not record which fields were provided, so
    comparing against the declared default is the only available signal.
    """
    for f in fields(args):
        if f.name == field_name:
            if f.default is MISSING:
                # Required and default_factory fields have no comparable literal default; a `!=`
                # against the MISSING sentinel would report every value as explicitly set.
                raise TypeError(f"{field_name} declares no literal default; _is_explicitly_set cannot probe it")
            return getattr(args, field_name) != f.default
    raise AttributeError(f"{type(args).__name__} declares no '{field_name}' field")


def build_muon_optimizer(model: torch.nn.Module, args: Any, decay_parameters: Sequence[str]):
    """Muon built from the training args: Newton-Schulz for 2D+ grads, internal AdamW for the rest."""
    # The Adam betas configure only Muon's internal scalar AdamW, whose tuned defaults differ from
    # HF's. Forward a beta only when explicitly set, so a Muon YAML does not retune that optimizer
    # by accident.
    default_beta1, default_beta2 = DEFAULT_SCALAR_BETAS
    scalar_betas = (
        args.adam_beta1 if _is_explicitly_set(args, "adam_beta1") else default_beta1,
        args.adam_beta2 if _is_explicitly_set(args, "adam_beta2") else default_beta2,
    )
    optimizer = create_muon_optimizer(
        model,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        scalar_betas=scalar_betas,
        scalar_eps=args.adam_epsilon,
        decay_parameters=decay_parameters,
    )
    if is_global_main_process():
        muon_count = sum(p.numel() for p in model.parameters() if p.requires_grad and p.ndim >= 2)
        scalar_count = sum(p.numel() for p in model.parameters() if p.requires_grad and p.ndim < 2)
        logger.info(
            f"Muon optimizer: {muon_count / 1e6:.1f}M params (Newton-Schulz), "
            f"{scalar_count / 1e6:.1f}M params (AdamW scalar)"
        )
    return optimizer
