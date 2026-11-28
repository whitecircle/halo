"""Keyed random init for a from-scratch run: each module drawn on CPU by the family's own ``_init_weights`` under a
seed hashed from the run's ``seed`` and the module's name, each fused expert bank one global expert at a time. The
weights depend only on the seed and the config, never on the layout, rank, GPU or global random state. They match
``from_config`` in distribution, not in value. A parameter the family's init leaves undrawn takes the init its
module's constructor runs, else transformers' default. See agent-docs/training-methods/pretraining.md."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers import initialization as hf_init

from halo.distributed.expert_parallel.families.roster import MOE_LAYER_MAP
from halo.distributed.expert_parallel.parallelize import ep_claimed_blocks
from halo.log import KEY_PREVIEW_COUNT
from halo.model_loading.lazy_safetensors.materialization import materialize_empty_like
from halo.model_loading.lazy_safetensors.meta_shell import build_config_meta_shell
from halo.models.moe.family_specs import find_moe_family_spec_for_config

logger = logging.getLogger(__name__)

InitWeights = Callable[[nn.Module], None]

# torch's CPU generator takes any 64-bit seed. 63 bits keeps the key a non-negative int.
_KEY_MASK = (1 << 63) - 1


def init_key(seed: int, module_name: str, expert: int | None = None) -> int:
    """The generator seed for one module, or for one global expert of a fused bank. sha256, not ``hash``, which Python
    salts per process."""
    text = f"{seed}\0{module_name}" if expert is None else f"{seed}\0{module_name}\0{expert}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little") & _KEY_MASK


@contextmanager
def _cpu_stream(key: int) -> Iterator[None]:
    """Draw from the CPU generator seeded with ``key``, its state restored after. Only the CPU generator: a
    ``torch.manual_seed`` would also reseed CUDA, which ``fork_rng(devices=[])`` does not restore."""
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(key)
        yield


def init_order(
    model: nn.Module, init_weights: InitWeights | None = None, prefix: str = ""
) -> Iterator[tuple[str, nn.Module, InitWeights]]:
    """``(name, module, _init_weights)``, children first, each module under the closest ``PreTrainedModel`` at or above
    it: the order and dispatch of transformers' ``initialize_weights``, so a parent's init still writes last.
    ``init_weights`` serves a root that is not a ``PreTrainedModel``, and ``prefix`` names the root in the full model."""

    def walk(module: nn.Module, name: str, init_fn: InitWeights):
        for child_name, child in module.named_children():
            child_init = child._init_weights if isinstance(child, PreTrainedModel) else init_fn
            yield from walk(child, f"{name}.{child_name}" if name else child_name, child_init)
        yield name, module, init_fn

    return walk(model, prefix, init_weights or model._init_weights)


def fused_expert_banks(model: nn.Module) -> dict[str, int]:
    """``{module name: expert count}`` for every fused expert container a family's EP layer claims. Keyed on the model
    class alone, never on the run's parallelism, so ep1 and ep8 draw the same way. A per-expert ``nn.ModuleList``
    already carries the expert index in its module names."""
    names = {id(module): name for name, module in model.named_modules()}
    banks: dict[str, int] = {}
    for _path, block in ep_claimed_blocks(model):
        layer_cls = MOE_LAYER_MAP[type(block).__name__]
        # The EP layer's own lookup: Gemma 4's claimed block is the container itself.
        experts = layer_cls._find_experts_container(block)
        if isinstance(experts, nn.ModuleList):
            continue
        banks[names[id(experts)]] = layer_cls.detect_num_experts(block)
    return banks


@contextmanager
def _parameters_swapped(module: nn.Module, tensors: dict[str, torch.Tensor]) -> Iterator[None]:
    """Point ``module``'s own parameters at ``tensors`` for one draw. An ``nn.Parameter`` over a view shares its
    storage, so an in-place init writes through to the bank."""
    saved = module._parameters
    module._parameters = {**saved, **{name: nn.Parameter(t, requires_grad=False) for name, t in tensors.items()}}
    try:
        yield
    finally:
        module._parameters = saved


def draw_module(
    module: nn.Module,
    init_weights: InitWeights,
    *,
    seed: int,
    name: str,
    num_experts: int | None = None,
    experts: range | None = None,
) -> None:
    """Draw ``module``'s own tensors in place.

    A fused bank (``num_experts`` set) is drawn expert by expert, each through a one-expert view keyed on its global
    index, so ``experts`` (default: all) can be any slice of the bank and each expert still gets the values the whole
    draw gives it. The bank's tensors without an expert axis are drawn once, keyed on the module alone. In each draw
    the other kind stands in as scratch of its full size, so every draw consumes the generator as the whole call
    would."""
    if num_experts is None:
        with _cpu_stream(init_key(seed, name)):
            init_weights(module)
        return
    own = {n: p for n, p in module._parameters.items() if p is not None}
    banked = {n: p for n, p in own.items() if p.dim() > 0 and p.shape[0] == num_experts}
    unbanked = {n: p for n, p in own.items() if n not in banked}
    if not banked:
        raise RuntimeError(
            f"{name} ({type(module).__name__}) is declared a fused bank of {num_experts} experts, but none of its "
            f"parameters {sorted(own)} has {num_experts} rows on dim 0."
        )
    if unbanked:
        scratch = {n: torch.empty_like(p) for n, p in banked.items()}
        with _parameters_swapped(module, scratch), _cpu_stream(init_key(seed, name)):
            init_weights(module)
    unbanked_scratch = {n: torch.empty_like(p) for n, p in unbanked.items()}
    for expert in experts if experts is not None else range(num_experts):
        views = {n: p.data[expert : expert + 1] for n, p in banked.items()}
        with _parameters_swapped(module, {**views, **unbanked_scratch}), _cpu_stream(init_key(seed, name, expert)):
            init_weights(module)


def _fallback_init(module: nn.Module, init_fn: InitWeights, *, seed: int, name: str) -> None:
    """Draw the parameters of ``module`` its family's init left undrawn as ``from_config`` leaves them: by the
    module's own ``reset_parameters``, the init its constructor runs (a remote-code router), else by transformers'
    default ``_init_weights`` (a remote-code norm whose constructor writes ones). Keyed as the module's own draw; what
    the family's init drew is kept."""
    own = dict(module.named_parameters(recurse=False))
    undrawn = {n for n, p in own.items() if p.is_floating_point() and bool(torch.isnan(p).any())}
    owner = getattr(init_fn, "__self__", None)
    reset = getattr(module, "reset_parameters", None)
    if not undrawn or not (callable(reset) or isinstance(owner, PreTrainedModel)):
        return
    drawn = {n: p.detach().clone() for n, p in own.items() if n not in undrawn}
    with _cpu_stream(init_key(seed, name)):
        if callable(reset):
            reset()
        else:
            PreTrainedModel._init_weights(owner, module)
    for n, value in drawn.items():
        own[n].copy_(value)


def keyed_init_(model: nn.Module, seed: int, *, init_weights: InitWeights | None = None, prefix: str = "") -> None:
    """Draw every tensor of ``model`` in place, under transformers' init guards. ``prefix`` is ``model``'s name in
    the full model, so a part built apart (an MTP head) draws what it would draw in place. A fused expert bank never
    falls back. A tied tensor is drawn once, by the first module holding it (the input embedding, before the head),
    as ``from_config`` draws before it ties; a later holder draws into zeroed scratch in its place."""
    banks = {f"{prefix}.{name}" if prefix else name: n for name, n in fused_expert_banks(model).items()}
    drawn: set[int] = set()
    with torch.no_grad(), hf_init.guard_torch_init_functions():
        for name, module, init_fn in init_order(model, init_weights, prefix):
            own = {n: p for n, p in module._parameters.items() if p is not None}
            tied = {n: torch.zeros_like(p) for n, p in own.items() if id(p) in drawn}
            with _parameters_swapped(module, tied):
                draw_module(module, init_fn, seed=seed, name=name, num_experts=banks.get(name))
                if name not in banks:
                    _fallback_init(module, init_fn, seed=seed, name=name)
            drawn.update(id(p) for p in own.values())


def draw_token_indexed_tables_(model: nn.Module, seed: int, *, prefix: str = "") -> None:
    """Draw the routing table the family indexes by token id (DeepSeek-V4's hash layers' ``gate.tid2eid``), which its
    ``_init_weights`` zeroes for the checkpoint to fill: from scratch that sends every token to expert 0 ``top_k``
    times, which the EP layer refuses and plain data parallelism trains as one expert. Each row takes ``top_k``
    distinct experts of its router's ``num_experts``, uniformly, keyed on the table's name."""
    config = getattr(model, "config", None)
    spec = None if config is None else find_moe_family_spec_for_config(config)
    table = None if spec is None else spec.token_indexed_table
    if table is None:
        return
    for name, buffer in model.named_buffers():
        if name != table and not name.endswith(f".{table}"):
            continue
        router = model.get_submodule(name.rpartition(".")[0])
        rows, top_k = buffer.shape
        with _cpu_stream(init_key(seed, f"{prefix}.{name}" if prefix else name)):
            picks = torch.rand(rows, router.num_experts).argsort(dim=-1)[:, :top_k]
        buffer.copy_(picks)


def undrawn_parameters(model: nn.Module) -> list[str]:
    """Float parameters still holding the NaN fill."""
    return [
        name
        for name, param in model.named_parameters()
        if param.is_floating_point() and bool(torch.isnan(param).any())
    ]


def draw_from_scratch_(
    model: nn.Module, seed: int, *, what: str, init_weights: InitWeights | None = None, prefix: str = ""
) -> None:
    """NaN-fill every float parameter of ``model``, :func:`keyed_init_` it, then refuse, naming them, the parameters
    neither its ``_init_weights`` nor the fallback drew: they would train from whatever the fill or the build left."""
    with torch.no_grad():
        for param in model.parameters():
            if param.is_floating_point():
                param.fill_(float("nan"))
    keyed_init_(model, seed, init_weights=init_weights, prefix=prefix)
    draw_token_indexed_tables_(model, seed, prefix=prefix)
    undrawn = undrawn_parameters(model)
    if undrawn:
        raise RuntimeError(
            f"init_from_scratch: {what}'s _init_weights leaves {len(undrawn)} parameter(s) undrawn, so it cannot be "
            f"built from scratch with a keyed init: {undrawn[:KEY_PREVIEW_COUNT]}. Train from a checkpoint instead: "
            f"build it once with from_config and save_pretrained, then load that."
        )


def build_keyed_model(
    model_class,
    config,
    dtype: torch.dtype,
    *,
    seed: int,
    trust_remote_code: bool,
    attn_implementation: str | None = None,
    revision: str | None = None,
) -> nn.Module:
    """A random-init ``model_class`` on CPU: the config-only meta shell (fp32 pins kept, remote code at ``revision``),
    every parameter given NaN-filled storage, tied pairs re-tied, then :func:`keyed_init_`. A parameter the family's
    ``_init_weights`` leaves undrawn is still NaN, and is refused by name. The global CPU generator is left as it was,
    so a from-scratch load shifts none of the run's later draws (shuffles, dropout)."""
    with torch.random.fork_rng(devices=[]):
        return _build_keyed_model(
            model_class,
            config,
            dtype,
            seed=seed,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
            revision=revision,
        )


def _build_keyed_model(
    model_class,
    config,
    dtype: torch.dtype,
    *,
    seed: int,
    trust_remote_code: bool,
    attn_implementation: str | None,
    revision: str | None,
) -> nn.Module:
    model = build_config_meta_shell(
        model_class,
        config,
        dtype,
        trust_remote_code=trust_remote_code,
        revision=revision,
        attn_implementation=attn_implementation,
    )
    # named_parameters yields a tied tensor once, and tie_weights then points its alias at the materialized one.
    for name, _ in list(model.named_parameters()):
        materialize_empty_like(model, name, "cpu", None)
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    on_meta = [name for name, param in model.named_parameters() if param.is_meta]
    if on_meta:
        raise RuntimeError(
            f"Random init of {type(model).__name__} left parameters on meta: {on_meta[:KEY_PREVIEW_COUNT]}"
        )
    draw_from_scratch_(model, seed, what=type(model).__name__)
    logger.info(f"Keyed random init of {type(model).__name__} with seed {seed}")
    return model
