#!/usr/bin/env python
"""Every EP wrapper must ADOPT its family's block, never restate it.

``EPMoELayerBase.__init__`` is one template: it resolves the router and the expert container off the
family's own declarations, then hands each step to a hook the family may override
(``_detect_hidden_dim`` / ``_init_routing`` / ``_init_shared_experts`` / ``_init_expert_compute`` /
``_init_expert_params``). What a family declares is therefore the ONLY thing standing between the
wrapper and the HF block it replaces, and every way of getting a declaration wrong is silent: a
``_ROUTER_ATTR`` pointing at the wrong module hooks and upcasts something the forward never calls; a
missing ``_SHARED_EXPERT_ATTRS`` drops the shared FFN from every token's output; an
``_init_routing`` override that stops consulting the router leaves ``routed_scaling_factor`` /
``norm_topk_prob`` / the group knobs at their neutral defaults; a restated activation or a mis-read
clamp bound trains the experts on a different function. None of those changes a shape, a dtype or a
key.

So this file compares the CONSTRUCTED wrapper against the block it wrapped, attribute by attribute,
over the whole registered roster (``test_every_registered_family_is_compared`` derives that roster
from ``ep_layer_classes()``, so a new family cannot join uncompared):

* the router the wrapper registered is the block's own module, under the family's declared name;
* every shared-expert module the block carries is adopted, under the name the CHECKPOINT spells;
* every routing / compute constant the wrapper holds equals what the block, its router, its expert
  container or its config declares — resolved by name, not by a per-family expectation table;
* the adopted native balancing slot carries the block's own tensor;
* this rank's expert parameters exist and cover the family's expert count.

A knob the wrapper does not hold at all is out of scope here (it has no value to compare); that a
family populates every knob its routing reads is
``test_ep_group_limited_routing_contract.py::test_every_consumed_knob_is_populated``.

The same template has to hold at scale, where a construction-time defect is a hang or an OOM rather
than a wrong number, so three properties of it are pinned here too: construction issues NO
collective (some loaders build layers before every rank's groups exist — a collective there deadlocks
the job, and at 512 ranks it deadlocks it silently); construction allocates NOTHING that grows with
world size (checked by building the same block at ``world_size`` 1, 512 and an NVL72 domain and
demanding byte-identical parameters); and the gathered export carries exactly the block's own expert
tensors, so no family's checkpoint moved.

``test_the_parity_check_is_sensitive_to_a_dropped_declaration`` is the anti-vacuity control: the
same checks against families whose declarations were deliberately reverted to the base defaults must
FAIL, or the assertions above are measuring nothing.

Run: ``python tests/cpu/parallelism/test_ep_prologue_parity.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import gc

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.distributed.expert_parallel.config import EPConfig
from src.distributed.expert_parallel.expert_weights import ep_layer_classes
from src.distributed.expert_parallel.layers.bailing import EPBailingMoELayer
from src.distributed.expert_parallel.layers.gemma4 import EPGemma4MoELayer
from src.distributed.expert_parallel.layers.laguna import EPLagunaMoELayer
from src.models.moe_balancing import ROUTER_TOPK_FIELDS, resolve_balancing_slot

# The real transformers blocks, built once per family and already parameterized with NON-neutral
# routing knobs — reusing them is what keeps this file comparing against upstream rather than
# against a fixture written to match the wrapper. Bailing is remote-code (transformers ships no
# module) and Laguna needs its own fixture, exactly as the numeric file records; both come from the
# suites that already own them.
from tests.cpu.parallelism.test_ep_expert_activation_and_weight_dtype import _BailingBlock, _laguna
from tests.cpu.parallelism.test_ep_vs_reference_equivalence_cpu import FAMILIES as _NUMERIC_FAMILIES

FAMILIES = [(name, wrapper, factory) for name, wrapper, factory, _inputs in _NUMERIC_FAMILIES] + [
    ("laguna", EPLagunaMoELayer, _laguna),
    ("bailing", EPBailingMoELayer, _BailingBlock),
]
_FIELDS = ("family", "wrapper", "factory")
_IDS = [family[0] for family in FAMILIES]

# Wrapper attribute → the spellings a block / router / expert container / config may declare it
# under. Resolved by NAME so the comparison stays a statement about the upstream module rather than
# a second copy of each family's expectations. A wrapper attribute NO source declares is a failure
# unless the family declared the knob optional: substituting a family default for a number the block
# owns is the silent-rename failure this file exists to catch.
_KNOB_SPELLINGS = {
    "top_k": ROUTER_TOPK_FIELDS,
    "n_routed_experts": ("n_routed_experts",),
    "n_group": ("n_group", "num_group"),
    "topk_group": ("topk_group",),
    "norm_topk_prob": ("norm_topk_prob",),
    "routed_scaling_factor": ("routed_scaling_factor", "moe_router_scaling_factor"),
    "expert_selection_fn": ("expert_selection_fn",),
    "use_expert_bias": ("use_expert_bias",),
    "is_hash": ("is_hash",),
    "score_fn": ("score_fn",),
    "alpha": ("alpha",),
    "limit": ("limit",),
    "swiglu_limit": ("swiglu_limit", "limit"),
    "n_shared_experts": ("n_shared_experts",),
    "route_scale": ("route_scale",),
}

# Knobs whose fallback is read off the LIVE tree rather than a constant (the ``required=False`` read
# in ``EPGroupLimitedMoELayerBase._init_routing``), so an absent declaration cannot become a wrong
# number and the strictness above does not apply.
_LIVE_TREE_FALLBACKS = {"n_routed_experts"}


def _ep_config(num_experts: int) -> EPConfig:
    # use_grouped_gemm off: the GptOss layout branches on it, and the loop layout is the one a CPU
    # box can also run a forward through.
    config = EPConfig(ep_size=1, world_size=1, gpus_per_node=1, use_grouped_gemm=False)
    config.finalize_expert_assignment(num_experts)
    return config


def _build(factory, wrapper, ep_config=None):
    """``(block, wrapper around it)``. The block is NOT deep-copied: adoption by reference is half of
    what is being checked, so the wrapper must see the very modules the block holds. Seeded, so two
    calls produce identical blocks — which is what lets the world-size sweep below compare them."""
    torch.manual_seed(0)
    block = factory()
    for parameter in block.parameters():
        nn.init.normal_(parameter, std=0.2)
    for _name, buffer in block.named_buffers():
        if buffer.is_floating_point():
            # The routing correction biases live here; left at zero a wrapper that never read one
            # would still match.
            nn.init.normal_(buffer, std=0.3)
    if ep_config is None:
        ep_config = _ep_config(wrapper.detect_num_experts(block))
    else:
        ep_config.finalize_expert_assignment(wrapper.detect_num_experts(block))
    return block, wrapper(block, ep_config).cpu()


def _resolve(root, path: str):
    """Dotted-path lookup, or ``None`` when any segment is absent."""
    for part in path.split("."):
        if not hasattr(root, part):
            return None
        root = getattr(root, part)
    return root


def _block_sources(block: nn.Module, wrapper) -> tuple:
    """Where a declaration may live, in resolution order: the block, its router, its expert
    container, its config."""
    router = wrapper._find_gate_or_router(block)
    experts = wrapper._find_experts_container(block)
    return (block, router, experts, getattr(block, "config", None))


def _declared(sources: tuple, spellings: tuple[str, ...]):
    """``(value, "source.spelling")`` for the first source declaring any spelling, else ``None``."""
    for name in spellings:
        for source in sources:
            if source is not None and hasattr(source, name):
                return getattr(source, name), f"{type(source).__name__}.{name}"
    return None


def _parity_failures(block: nn.Module, layer, wrapper) -> list[str]:
    """Everything the constructed ``layer`` disagrees with ``block`` about. Empty = parity."""
    failures: list[str] = []
    sources = _block_sources(block, wrapper)
    router, experts = sources[1], sources[2]

    # 1. Router: the wrapper must hold the block's OWN module under its declared name.
    if wrapper._ROUTER_ATTR is None:
        adopted = [name for name, child in layer.named_children() if child is router]
        if adopted:
            failures.append(f"declares no router but adopted one as {adopted}")
    elif getattr(layer, wrapper._ROUTER_ATTR, None) is not router:
        failures.append(f"router under '{wrapper._ROUTER_ATTR}' is not the block's own module")

    # 2. Every shared-expert module the block carries must be adopted, by the same name — that name
    #    is the export key, and an unadopted one silently vanishes from the output and the save.
    for name, child in block.named_children():
        if not name.startswith("shared_expert"):
            continue
        if getattr(layer, name, None) is not child:
            failures.append(f"block's '{name}' shared module was not adopted under that name")
    if layer._shared_expert_attr not in dict(block.named_children()) and any(
        name.startswith("shared_expert") for name, _ in block.named_children()
    ):
        failures.append(f"_shared_expert_attr='{layer._shared_expert_attr}' names no module on the block")

    # 3. Routing / compute constants: equal to whatever the block side declares, and declared at all.
    allowed_defaults = set(getattr(wrapper, "_OPTIONAL_ROUTING_KNOBS", ())) | _LIVE_TREE_FALLBACKS
    for attr, spellings in _KNOB_SPELLINGS.items():
        if not hasattr(layer, attr):
            continue
        declaration = _declared(sources, spellings)
        if declaration is None:
            if not set(spellings) & allowed_defaults:
                failures.append(
                    f"{attr}={getattr(layer, attr)!r} is a wrapper default — no source declares any of "
                    f"{spellings}, and the family does not list the knob in _OPTIONAL_ROUTING_KNOBS"
                )
            continue
        expected, origin = declaration
        held = getattr(layer, attr)
        same = held is expected if callable(expected) and not isinstance(expected, bool) else held == expected
        if not same:
            failures.append(f"{attr}={held!r} but the block declares {origin}={expected!r}")

    # 4. The activation, where the container carries one: identity, so a checkpoint's hidden_act
    #    cannot be overridden by the wrapper's own assumption.
    container_act = getattr(experts, "act_fn", None) if experts is not None else None
    if container_act is not None and hasattr(layer, "act_fn") and layer.act_fn is not container_act:
        failures.append("act_fn is not the expert container's own activation")

    # 5. The adopted native balancing slot must carry the block's tensor, at the same dotted path.
    if wrapper._NATIVE_BALANCING_BIAS_ATTR:
        target = resolve_balancing_slot(layer, wrapper._NATIVE_BALANCING_BIAS_ATTR)
        expected = _resolve(block, wrapper._NATIVE_BALANCING_BIAS_ATTR)
        if expected is not None:
            if target is None:
                failures.append(f"native balancing slot '{wrapper._NATIVE_BALANCING_BIAS_ATTR}' is absent")
            elif not torch.equal(getattr(*target).float(), expected.float()):
                failures.append(f"native balancing slot '{wrapper._NATIVE_BALANCING_BIAS_ATTR}' is not the block's")

    # 6. Expert weights exist and cover this rank's whole expert range — a layout probe that skipped
    #    a renamed tensor would leave the layer silently empty.
    expert_params = layer.expert_named_params()
    if not expert_params:
        failures.append("registered no expert parameters")
    else:
        for name, param in expert_params:
            if param.shape[0] != layer.experts_per_rank:
                failures.append(f"expert param '{name}' holds {param.shape[0]} of {layer.experts_per_rank} experts")
    return failures


@pytest.mark.parametrize(_FIELDS, FAMILIES, ids=_IDS)
def test_the_wrapper_adopts_the_block_it_replaces(family, wrapper, factory):
    block, layer = _build(factory, wrapper)
    failures = _parity_failures(block, layer, wrapper)
    assert not failures, f"{family}: the EP wrapper diverges from {type(block).__name__}:\n  " + "\n  ".join(failures)


@pytest.mark.parametrize(_FIELDS, FAMILIES, ids=_IDS)
def test_the_hidden_dim_and_expert_count_come_from_the_block(family, wrapper, factory):
    """Both are read before any module state exists, so nothing downstream can correct them: a wrong
    hidden dim sizes every DeepEP transport buffer, a wrong expert count mis-assigns the ranges."""
    block, layer = _build(factory, wrapper)
    experts = wrapper._find_experts_container(block)
    router = wrapper._find_gate_or_router(block)

    expected_hidden = router.weight.shape[1] if router is not None and hasattr(router, "weight") else None
    if expected_hidden is None:
        expected_hidden = experts.gate_up_proj.shape[2]
    assert layer.hidden_dim == expected_hidden, f"{family}: hidden_dim {layer.hidden_dim} != {expected_hidden}"

    param = layer.expert_named_params()[0][1]
    assert layer.num_experts == wrapper.detect_num_experts(block) == param.shape[0]


@pytest.mark.parametrize(_FIELDS, FAMILIES, ids=_IDS)
def test_top_k_is_set_exactly_where_the_family_has_a_router(family, wrapper, factory):
    """``top_k`` is the base prologue's one unconditional routing read; the routing-replay injector and
    every biased-selection path assume it. Only a family whose router is external can lack it — which
    the class already declares as ``_ROUTER_ATTR = None`` (Gemma4's sibling router), so the exception
    is read off that declaration rather than a roster this test would have to keep."""
    _block, layer = _build(factory, wrapper)
    assert hasattr(layer, "top_k") is (wrapper._ROUTER_ATTR is not None), (
        f"{family}: top_k presence ({hasattr(layer, 'top_k')}) contradicts _ROUTER_ATTR={wrapper._ROUTER_ATTR!r}"
    )


# Scale: what a construction-time defect costs at 512 ranks / an NVL72 domain.

# Every torch.distributed entry point that SYNCHRONIZES ranks. ``new_group`` is included: it is
# collective over the ranks forming it, so a layer creating one during construction hangs any loader
# that patches ranks at different times.
_COLLECTIVES = (
    "all_gather",
    "all_gather_into_tensor",
    "all_reduce",
    "all_to_all",
    "all_to_all_single",
    "barrier",
    "broadcast",
    "gather",
    "new_group",
    "reduce",
    "reduce_scatter",
    "reduce_scatter_tensor",
    "scatter",
)

# ep1 across a 64-node job and across one NVL72 domain. Both leave ``experts_fsdp_managed`` on (the
# ep_group_size==1 default), which is the DTensor-experts branch of the construction contract; the
# ep_size>1 plain-tensor branch needs real groups and is covered by the multi-rank EP suites.
_WORLD_SHAPES = ((1, 1), (512, 8), (72, 72))


@pytest.mark.parametrize(_FIELDS, FAMILIES, ids=_IDS)
def test_construction_issues_no_collective(family, wrapper, factory, monkeypatch):
    """Building a layer must not synchronize ranks.

    Some load paths patch the model before every rank has reached group setup, and a rank-uniform
    ordering is not guaranteed there — a collective inside ``__init__`` is a deadlock, not an error,
    and at 512 ranks it is a silent one. The template resolves everything from the wrapped module and
    the config, so this holds by construction; the point of the test is that it keeps holding as
    families add hooks (a gather, a broadcast of a routing table, a ``new_group`` for expert TP).
    """
    patched = 0
    for name in _COLLECTIVES:
        if hasattr(torch.distributed, name):

            def _banned(*args, _name=name, **kwargs):
                raise AssertionError(f"{family}: construction called torch.distributed.{_name}()")

            monkeypatch.setattr(torch.distributed, name, _banned)
            patched += 1

    assert patched >= len(_COLLECTIVES) - 1, f"only {patched} collectives were patched — the gate is porous"
    with pytest.raises(AssertionError, match="all_reduce"):
        torch.distributed.all_reduce(None)  # the patch is live, so a real call below would be caught

    _build(factory, wrapper)


@pytest.mark.parametrize(_FIELDS, FAMILIES, ids=_IDS)
def test_construction_allocates_nothing_that_grows_with_world_size(family, wrapper, factory):
    """The same block wrapped at world 1, 512 and an NVL72 domain must allocate the SAME tensors.

    Per-rank state sized by ``world_size`` (or by ``num_experts x world``) is what turns a working
    8-GPU config into an OOM at 512 ranks, and it is invisible at the scale anyone debugs at. The
    comparison is on the full parameter/buffer set — names, shapes, dtypes and values — so a
    world-sized allocation, a rank-dependent layout or a rank-dependent dtype all land here.
    """

    def snapshot(world_size: int, gpus_per_node: int) -> dict:
        config = EPConfig(ep_size=1, world_size=world_size, gpus_per_node=gpus_per_node, use_grouped_gemm=False)
        _block, layer = _build(factory, wrapper, ep_config=config)
        return {
            name: (tuple(tensor.shape), tensor.dtype, tensor.detach().clone())
            for name, tensor in list(layer.named_parameters()) + list(layer.named_buffers())
        }

    baseline = snapshot(*_WORLD_SHAPES[0])
    assert baseline, f"{family}: the wrapper allocated nothing — the comparison would be vacuous"

    for world_size, gpus_per_node in _WORLD_SHAPES[1:]:
        scaled = snapshot(world_size, gpus_per_node)
        assert set(scaled) == set(baseline), (
            f"{family} at world_size={world_size}: allocated a different tensor SET than at world 1 — "
            f"added {sorted(set(scaled) - set(baseline))}, dropped {sorted(set(baseline) - set(scaled))}"
        )
        for name, (shape, dtype, tensor) in baseline.items():
            scaled_shape, scaled_dtype, scaled_tensor = scaled[name]
            assert (scaled_shape, scaled_dtype) == (shape, dtype), (
                f"{family} at world_size={world_size}: '{name}' is {scaled_shape}/{scaled_dtype}, "
                f"was {shape}/{dtype} at world 1 — this allocation scales with the job"
            )
            assert torch.equal(scaled_tensor, tensor), f"{family} at world_size={world_size}: '{name}' changed value"


@pytest.mark.parametrize(_FIELDS, FAMILIES, ids=_IDS)
def test_the_managed_ep1_experts_branch_constructs_identically(family, wrapper, factory):
    """``fsdp_shard_ep1_experts`` (FSDP2 owns the replicated experts, so they become DTensors after
    sharding) must change nothing the wrapper ALLOCATES — it changes only who syncs the gradients.
    A layout that differed between the two would give the ep1 path a checkpoint the ep>1 path cannot
    read, which no shape or key check downstream would catch.
    """

    def snapshot(managed: bool):
        config = EPConfig(
            ep_size=1, world_size=8, gpus_per_node=8, use_grouped_gemm=False, fsdp_shard_ep1_experts=managed
        )
        _block, layer = _build(factory, wrapper, ep_config=config)
        return layer, {n: t.detach().clone() for n, t in layer.expert_named_params()}

    unmanaged_layer, unmanaged = snapshot(False)
    managed_layer, managed = snapshot(True)

    assert managed_layer.ep_config.experts_fsdp_managed and not unmanaged_layer.ep_config.experts_fsdp_managed
    assert set(managed) == set(unmanaged), f"{family}: fsdp_shard_ep1_experts changed the expert param SET"
    for name, tensor in unmanaged.items():
        assert torch.equal(managed[name], tensor), f"{family}: fsdp_shard_ep1_experts changed '{name}'"

    # FSDP2's reduce-scatter is the whole sync there, so every trainable param must count as synced.
    synced = managed_layer.synced_trainable_param_ids()
    unsynced = [n for n, p in managed_layer.named_parameters() if p.requires_grad and id(p) not in synced]
    assert not unsynced, f"{family}: managed-ep1 params outside synced_trainable_param_ids: {unsynced}"


@pytest.mark.parametrize(_FIELDS, FAMILIES, ids=_IDS)
def test_the_gathered_export_carries_exactly_the_blocks_expert_tensors(family, wrapper, factory):
    """The gathered checkpoint must hold the block's expert weights — all of them, and nothing else.

    Layout-agnostic on purpose: every family transposes, splits, stacks or interleaves on the way
    out, and all of those are permutations, so comparing the SORTED values against the block's own
    expert container covers each family's layout without restating it here. What it catches is a
    dropped tensor, a duplicated one, a rank's slice standing in for the whole bank, or a value that
    changed on the way through — the export defects that produce a checkpoint which loads and is
    wrong. The per-family KEY layouts are pinned by the checkpoint suite.
    """
    block, layer = _build(factory, wrapper)
    experts = wrapper._find_experts_container(block)

    exported = torch.cat([tensor.flatten().float() for tensor in layer.gather_expert_state_dict().values()])
    held = torch.cat([parameter.detach().flatten().float() for parameter in experts.parameters()])

    assert held.numel() > 0, f"{family}: the block's expert container holds no parameters — nothing compared"
    assert exported.numel() == held.numel(), (
        f"{family}: gathered {exported.numel()} expert values but the block holds {held.numel()} — "
        f"the export dropped or duplicated an expert tensor"
    )
    assert torch.equal(exported.sort().values, held.sort().values), (
        f"{family}: the gathered expert values are not the block's own"
    )


def test_every_registered_family_is_compared():
    """Derived from the class hierarchy, not a hand-kept list: a family joining the toolkit with a
    prologue nothing compares to its block lands here."""
    declaring = {cls.__name__ for cls in ep_layer_classes() if vars(cls).get("HF_MODULE_NAMES")}
    compared = {wrapper.__name__ for _family, wrapper, _factory in FAMILIES}
    missing = declaring - compared
    assert not missing, f"{sorted(missing)} declare an HF MoE block but no fixture compares their prologue to it"


def test_the_parity_check_is_sensitive_to_a_dropped_declaration():
    """Anti-vacuity: reverting a family's declarations to the base defaults — exactly what a
    hand-written prologue drifting from the block produces — must be CAUGHT, not tolerated."""
    families = {name: (wrapper, factory) for name, wrapper, factory in FAMILIES}

    # A family that stops declaring its shared expert: the block still builds one, the wrapper drops
    # it from every token's output, and nothing else in the toolkit notices.
    glm4_wrapper, glm4_factory = families["glm4_moe_lite"]
    forgot_shared = type("_ForgotSharedExpert", (glm4_wrapper,), {"_SHARED_EXPERT_ATTRS": ()})
    block, layer = _build(glm4_factory, forgot_shared)
    assert any("shared" in failure for failure in _parity_failures(block, layer, forgot_shared)), (
        "a wrapper that adopted no shared expert passed the parity check"
    )

    # A routing hook that reads the knobs off the BLOCK alone — the pre-5.14 resolution, before the
    # ``*TopkRouter`` refactor moved every one of them onto the gate. Each silently resolves to its
    # neutral default, which is a live wrong number (routed weights unscaled, group-limited
    # selection off) rather than a missing attribute.
    def _block_only_knobs(self, original_layer):
        self.n_routed_experts = getattr(original_layer, "n_routed_experts", self.num_experts)
        self.n_group = getattr(original_layer, "n_group", 1)
        self.topk_group = getattr(original_layer, "topk_group", 1)
        self.norm_topk_prob = getattr(original_layer, "norm_topk_prob", False)
        self.routed_scaling_factor = getattr(original_layer, "routed_scaling_factor", 1.0)
        self.top_k = self._find_top_k(original_layer)

    neutral_knobs = type("_NeutralRoutingKnobs", (glm4_wrapper,), {"_init_routing": _block_only_knobs})
    block, layer = _build(glm4_factory, neutral_knobs)
    assert any("routed_scaling_factor" in failure for failure in _parity_failures(block, layer, neutral_knobs)), (
        "a wrapper whose routing knobs fell back to their neutral defaults passed the parity check"
    )

    # A compute hook that RESTATES the family's activation instead of adopting the container's: the
    # shapes, dtypes and keys are all unchanged, and the experts train on a different non-linearity.
    def _restated_activation(self, original_layer, experts):
        self.act_fn = F.relu

    restated = type("_RestatedActivation", (glm4_wrapper,), {"_init_expert_compute": _restated_activation})
    block, layer = _build(glm4_factory, restated)
    assert any("act_fn" in failure for failure in _parity_failures(block, layer, restated)), (
        "a wrapper that restated its activation passed the parity check"
    )

    # A compute hook that INVENTS a constant no source declares — the shape a hand-picked clamp or
    # scale takes when the block it came from renamed or dropped it. Nothing is missing and nothing
    # raises: the layer simply computes with a number the checkpoint never carried.
    def _invented_limit(self, original_layer, experts):
        glm4_wrapper._init_expert_compute(self, original_layer, experts)
        self.limit = 7.0

    invented = type("_InventedLimit", (glm4_wrapper,), {"_init_expert_compute": _invented_limit})
    block, layer = _build(glm4_factory, invented)
    assert any("limit" in failure for failure in _parity_failures(block, layer, invented)), (
        "a wrapper holding a compute constant no source declares passed the parity check"
    )

    # Each stand-in joined ``EPMoELayerBase.__subclasses__`` the moment it was defined, and every
    # registry-derived roster in the session walks exactly that — so drop them here rather than
    # leave a later test's "which families declare X" answer holding four families nobody ships.
    stand_ins = {"_ForgotSharedExpert", "_NeutralRoutingKnobs", "_RestatedActivation", "_InventedLimit"}
    del forgot_shared, neutral_knobs, restated, invented, block, layer
    gc.collect()
    leaked = stand_ins & {cls.__name__ for cls in ep_layer_classes()}
    assert not leaked, f"this test's stand-in EP families outlived it: {sorted(leaked)}"


def test_gemma4_is_the_only_router_less_family():
    """The routerless branch of the template (no adoption, no routing read, expert-only grad hooks)
    exists for one family; a second one arriving silently would inherit Gemma4's exemptions above."""
    routerless = {
        cls.__name__ for cls in ep_layer_classes() if vars(cls).get("HF_MODULE_NAMES") and cls._ROUTER_ATTR is None
    }
    assert routerless == {EPGemma4MoELayer.__name__}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
