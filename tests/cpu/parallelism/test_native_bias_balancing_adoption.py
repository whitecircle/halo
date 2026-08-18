#!/usr/bin/env python
"""Base-class contracts of native bias-update adoption (``_NATIVE_BALANCING_BIAS_ATTR``).

Families whose selection math already consults a checkpoint-persistent tensor adopt that tensor as
the balancing state, so the trained bias exports with every gathered save instead of dying with the
run. The machinery lives on ``EPMoELayerBase``; these tests pin its edges for every declaring
family: adoption identity (updates land in the tensor the family's routing actually reads), the
single-application guarantee (``_balancing_bias`` returns None in native mode — the family's own
route already adds the adopted tensor), Parameter→buffer re-registration under the same state-dict
key (FSDP2 shards Parameters, which would break the callback's in-place updates), the fp32 upcast,
the meta fail-loud, slot materialization on config-gated families (LFM-2), the adoption of GPT-OSS's
trainable logit bias, and the side-buffer fallback when an instance lacks a non-materializable slot.

    python tests/cpu/parallelism/test_native_bias_balancing_adoption.py
"""

import pytest
import torch
import torch.nn as nn

from src.distributed.expert_parallel.balancing import EPRouterBalancingMixin
from src.distributed.expert_parallel.expert_weights import ep_layer_classes
from src.distributed.expert_parallel.layers.bailing import EPBailingMoELayer
from src.distributed.expert_parallel.layers.deepseek_v4 import EPDeepseekV4MoELayer
from src.distributed.expert_parallel.layers.glm4 import EPGlm4MoELayer
from src.distributed.expert_parallel.layers.glm5_next import EPGlm5NextMoELayer
from src.distributed.expert_parallel.layers.gpt_oss import EPGptOssMoELayer
from src.distributed.expert_parallel.layers.inkling import EPInklingMoELayer
from src.distributed.expert_parallel.layers.laguna import EPLagunaMoELayer
from src.distributed.expert_parallel.layers.lfm2 import EPLfm2MoELayer
from src.distributed.expert_parallel.layers.mistral4 import EPMistral4MoELayer
from src.distributed.expert_parallel.layers.qwen3 import EPQwen3MoELayer
from src.distributed.expert_parallel.layers.step3p7 import EPStep3p7MoELayer
from src.models.moe_balancing import is_transient_balancing_router, iter_balancing_routers

NUM_EXPERTS = 8
# Hub slots are bf16 everywhere; the adoption's fp32 upcast is load-bearing (1e-3 steps round away).
SLOT_DTYPE = torch.bfloat16


class _Gate(nn.Module):
    """Fake gate carrying the family's native slot as a buffer or a Parameter."""

    def __init__(
        self, attr: str | None, *, as_parameter: bool = False, meta: bool = False, requires_grad: bool = False
    ):
        super().__init__()
        if attr is None:
            return
        device = "meta" if meta else "cpu"
        tensor = torch.zeros(NUM_EXPERTS, dtype=SLOT_DTYPE, device=device)
        if as_parameter:
            setattr(self, attr, nn.Parameter(tensor, requires_grad=requires_grad))
        else:
            self.register_buffer(attr, tensor)


class _Shell(nn.Module):
    """A parent module, so router iteration has to walk the tree rather than read the layer itself."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner


def _bare_layer(cls, gate: nn.Module):
    """Wrapper skeleton without EP process-group state (the balancing seam needs none of it)."""
    layer = object.__new__(cls)
    nn.Module.__init__(layer)
    layer.gate = gate
    layer.num_experts = NUM_EXPERTS
    if cls is EPDeepseekV4MoELayer:
        layer.is_hash = False  # hash routing refuses balancing before adoption is reached
    return layer


# (wrapper class, native attr on the gate, hub slot is an nn.Parameter in that family's modeling)
FAMILIES = [
    (EPBailingMoELayer, "expert_bias", False),
    (EPGlm4MoELayer, "e_score_correction_bias", False),
    (EPGlm5NextMoELayer, "e_score_correction_bias", False),
    (EPLagunaMoELayer, "e_score_correction_bias", True),
    (EPDeepseekV4MoELayer, "e_score_correction_bias", False),
    (EPInklingMoELayer, "e_score_correction_bias", True),
    (EPStep3p7MoELayer, "e_score_correction_bias", False),
]
# LFM-2 declares the slot on the WRAPPER (and materializes it on use_expert_bias:false checkpoints),
# and GPT-OSS adopts the trainable logit-space `router.bias`; both get dedicated tests below.
_COVERED = {cls.__name__ for cls, _attr, _as_parameter in FAMILIES} | {
    EPLfm2MoELayer.__name__,
    EPGptOssMoELayer.__name__,
}


def test_every_family_with_a_native_slot_is_covered_here():
    """Derived from the class hierarchy, so a family that starts adopting a native slot cannot land
    without tests. Read as the EFFECTIVE attribute rather than ``vars(cls)``: adoption is inherited
    (Laguna takes GLM-4's), and an inheriting family still needs its own row — its hub slot may be a
    Parameter where the parent's is a buffer, which is a different re-registration path."""
    declaring = {cls.__name__ for cls in ep_layer_classes() if cls._NATIVE_BALANCING_BIAS_ATTR}
    assert declaring == _COVERED


@pytest.mark.parametrize("cls,attr,as_parameter", FAMILIES, ids=lambda v: getattr(v, "__name__", str(v)))
def test_adoption_identity_and_export_key(cls, attr, as_parameter):
    layer = _bare_layer(cls, _Gate(attr, as_parameter=as_parameter))
    assert not hasattr(layer, "balancing_biases"), "state must appear exactly at enable-time"

    assert layer.enable_bias_balancing() is True
    adopted = layer.balancing_biases
    assert adopted is getattr(layer.gate, attr), "updates must land in the tensor the gate reads"
    assert adopted.dtype == torch.float32, "1e-3 sign-steps round away in bf16"
    assert layer.expert_load_counter is None  # the slot iter_balancing_routers keys on

    # The slot stays in the gate's state_dict under its own key — that IS the export path.
    assert attr in layer.gate.state_dict()
    # Never a Parameter post-adoption: sharding it into a DTensor breaks in-place callback updates.
    assert attr not in dict(layer.gate.named_parameters())

    layer.balancing_biases.add_(torch.arange(NUM_EXPERTS, dtype=torch.float32))
    assert torch.equal(layer.gate.state_dict()[attr], torch.arange(NUM_EXPERTS, dtype=torch.float32))

    layer.balancing_biases.copy_(torch.full((NUM_EXPERTS,), 3.0))
    assert torch.equal(getattr(layer.gate, attr), torch.full((NUM_EXPERTS,), 3.0))


def test_lfm2_adopts_the_wrapper_level_buffer():
    layer = _bare_layer(EPLfm2MoELayer, _Gate(None))
    layer.use_expert_bias = True
    layer.register_buffer("expert_bias", torch.zeros(NUM_EXPERTS, dtype=torch.bfloat16))

    assert layer.enable_bias_balancing() is True
    assert layer.balancing_biases is layer.expert_bias
    assert layer.balancing_biases.dtype == torch.float32
    assert "expert_bias" in layer.state_dict()


def test_lfm2_without_native_slot_materializes_it():
    """A ``use_expert_bias: false`` checkpoint has no slot, but the slot is config-gated, not
    structural — enable creates the architecture's own zero buffer (a semantic no-op in the same
    sigmoid-score space the transient side-buffer would use) instead of training a bias no export
    carries."""
    layer = _bare_layer(EPLfm2MoELayer, _Gate(None))
    layer.use_expert_bias = False  # instance never registered the buffer

    assert layer.can_adopt_native_balancing() is True
    assert layer.enable_bias_balancing() is True
    assert layer.use_expert_bias, "wrapper routing must now apply the created slot"
    assert layer.balancing_biases is layer.expert_bias
    assert layer.balancing_biases.dtype == torch.float32
    assert torch.equal(layer.expert_bias, torch.zeros(NUM_EXPERTS)), "materialization must not shift routing"
    assert "expert_bias" in layer.state_dict(), "the materialized slot IS the export path"
    assert layer._balancing_bias(torch.zeros(1)) is None, "native mode — the route adds expert_bias itself"
    assert not is_transient_balancing_router(layer)


def test_gptoss_adopts_the_trainable_router_logit_bias():
    """GPT-OSS's slot is the hub router's own gradient-trained logit bias; adoption re-registers it
    as a buffer under the same key — frozen out of gradient training (the sign controller owns it),
    exported with every save, and read by ``F.linear`` exactly as vLLM/SGLang read it at serving."""
    layer = object.__new__(EPGptOssMoELayer)
    nn.Module.__init__(layer)
    layer.router = _Gate("bias", as_parameter=True, requires_grad=True)
    layer.num_experts = NUM_EXPERTS

    assert layer.can_adopt_native_balancing() is True
    assert layer.enable_bias_balancing() is True
    assert layer.balancing_biases is layer.router.bias, "updates must land in the tensor F.linear reads"
    assert layer.balancing_biases.dtype == torch.float32
    assert "bias" in layer.router.state_dict(), "the hub router key IS the export path"
    assert "bias" not in dict(layer.router.named_parameters()), "adoption freezes gradient training"
    assert layer._balancing_bias(torch.zeros(1)) is None, "the bias already sits inside the logits"
    assert not is_transient_balancing_router(layer)

    layer.balancing_biases.add_(torch.arange(NUM_EXPERTS, dtype=torch.float32))
    assert torch.equal(layer.router.state_dict()["bias"], torch.arange(NUM_EXPERTS, dtype=torch.float32))


def test_families_without_a_native_slot_cannot_adopt():
    """Qwen3 / Mistral4 gates carry no bias tensor and no materialization path — only the transient
    side-buffer is possible, which is exactly what the strict ``bias_update`` mode refuses."""
    for cls in (EPQwen3MoELayer, EPMistral4MoELayer):
        layer = object.__new__(cls)
        nn.Module.__init__(layer)
        assert layer.can_adopt_native_balancing() is False, cls.__name__


def test_native_mode_balancing_bias_returns_none():
    """The single-application guarantee: the family's own selection math already adds the adopted
    tensor, so the generic injection hook must contribute nothing on top of it."""
    layer = _bare_layer(EPGlm4MoELayer, _Gate("e_score_correction_bias"))
    layer.enable_bias_balancing()
    layer.balancing_biases.fill_(5.0)
    assert layer._balancing_bias(torch.zeros(NUM_EXPERTS)) is None


def test_meta_native_slot_fails_loud():
    layer = _bare_layer(EPGlm4MoELayer, _Gate("e_score_correction_bias", meta=True))
    with pytest.raises(RuntimeError, match="meta"):
        layer.enable_bias_balancing()


def _gate_declared_in(module_path: str) -> nn.Module:
    """A slot-less gate whose CLASS lives in ``module_path`` — the signal a family reads to tell an
    in-library router from a remote-code revision of the same architecture."""
    return type("_RevisionGate", (_Gate,), {"__module__": module_path})(None)


def test_glm4_tolerates_an_absent_slot_only_for_a_remote_code_router():
    """``Glm4MoeLiteTopkRouter`` and ``LagunaTopKRouter`` register ``e_score_correction_bias``
    unconditionally, so an absent slot on the in-library class is an upstream RENAME — selecting on
    unbiased scores would silently drop the pretrained correction, with no shape, dtype or key moving.
    Only the remote-code revisions this wrapper also serves may legally carry none, and a class-wide
    opt-out would disarm the read for both."""
    scores = torch.zeros(2, NUM_EXPERTS)

    in_library = _bare_layer(
        EPGlm4MoELayer, _gate_declared_in("transformers.models.glm4_moe_lite.modeling_glm4_moe_lite")
    )
    with pytest.raises(AttributeError, match="e_score_correction_bias"):
        in_library._selection_scores(scores)

    remote_code = _bare_layer(EPGlm4MoELayer, _gate_declared_in("transformers_modules.glm4.modeling_glm4_moe"))
    assert remote_code._selection_scores(scores) is scores, "a remote-code revision may ship no bias"


def test_lfm2_tolerates_an_absent_slot_only_where_the_flag_its_routing_reads_is_off():
    """LFM-2's slot is config-gated (``use_expert_bias``), not structural: absent with the flag off is
    a configuration, absent with it ON is a slot that went missing under the family's own routing.

    The flag is read off the WRAPPER, the one this family's ``route_tokens_to_experts`` and its slot
    materialization both key on — selection runs on layers that adopted no router module (the
    bias-injection callers build exactly that), so reaching into the router here would crash the
    forward instead of answering it."""
    scores = torch.zeros(2, NUM_EXPERTS)

    gated_off = _bare_layer(EPLfm2MoELayer, _Gate(None))
    del gated_off.gate  # the selection path never touches the router
    gated_off.use_expert_bias = False
    assert gated_off._selection_scores(scores) is scores

    asks_for_it = _bare_layer(EPLfm2MoELayer, _Gate(None))
    asks_for_it.use_expert_bias = True
    with pytest.raises(AttributeError, match="expert_bias"):
        asks_for_it._selection_scores(scores)


def test_lfm2_refuses_a_block_whose_router_asks_for_a_bias_it_does_not_carry():
    """The construction-time half of the pair above: the flag may only be False because the ROUTER
    says so. Silently flipping it off (as this once did) routes unbiased for the whole run, with the
    pretrained selection bias dropped and no shape, dtype or key moving."""
    router = nn.Module()
    router.norm_topk_prob = True
    router.routed_scaling_factor = 1.0
    router.top_k = 2
    layer = _bare_layer(EPLfm2MoELayer, router)

    block = nn.Module()  # a block carrying no expert_bias
    block.gate = router  # where _find_top_k reads it

    router.use_expert_bias = True
    with pytest.raises(AttributeError, match="use_expert_bias=True"):
        layer._init_routing(block)

    router.use_expert_bias = False
    layer._init_routing(block)
    assert layer.use_expert_bias is False


def test_absent_native_slot_falls_back_to_side_buffer():
    layer = _bare_layer(EPGlm4MoELayer, _Gate(None))  # hand-built gate without the slot
    assert layer.enable_bias_balancing() is True
    assert "balancing_biases" not in layer.state_dict()
    assert layer._balancing_bias(torch.zeros(1)) is not None


@pytest.mark.parametrize("cls,attr,as_parameter", FAMILIES, ids=lambda v: getattr(v, "__name__", str(v)))
def test_enable_is_idempotent_and_router_iteration_keys_on_presence(cls, attr, as_parameter):
    layer = _bare_layer(cls, _Gate(attr, as_parameter=as_parameter))

    shell = _Shell(layer)
    assert list(iter_balancing_routers(shell)) == []
    layer.enable_bias_balancing()
    adopted = layer.balancing_biases
    adopted.fill_(1.0)
    layer.enable_bias_balancing()
    assert layer.balancing_biases is adopted, "re-adoption would strand the callback on a dead alias"
    assert torch.equal(layer.balancing_biases, torch.ones(NUM_EXPERTS))
    assert list(iter_balancing_routers(shell)) == [layer]


def test_a_declared_selection_slot_that_vanished_raises_instead_of_routing_unbiased():
    """The families whose own selection reads the slot read it UNCONDITIONALLY upstream, so an absent
    one is a rename — and routing on without it drops the pretrained correction bias with no shape,
    dtype or key moving, and no warning outside the balancing enable path. GLM-4 and LFM-2 declare
    the absence legal (remote-code revisions ship none; ``use_expert_bias: false`` registers none)."""
    strict = _bare_layer(EPGlm5NextMoELayer, _Gate(None))  # declares gate.e_score_correction_bias
    with pytest.raises(AttributeError, match="e_score_correction_bias"):
        strict._selection_scores(torch.zeros(2, NUM_EXPERTS))

    tolerant = _bare_layer(EPGlm4MoELayer, _Gate(None))
    scores = torch.zeros(2, NUM_EXPERTS)
    assert tolerant._selection_scores(scores) is scores, "an optional slot's absence must route unbiased"


def test_exactly_one_family_materializes_its_native_balancing_slot():
    """``_can_materialize_native_balancing_slot`` defaults to "does this family implement the hook",
    which is what lets ``can_adopt_native_balancing`` promise an EXPORTING slot to strict
    ``bias_update``. A family that implements the hook without meaning to — or one that stops — moves
    that promise silently: the run then trains a bias no served copy reads. Derived from the override,
    so the roster follows the class hierarchy."""
    implementing = {
        cls.__name__
        for cls in ep_layer_classes()
        if cls._materialize_native_balancing_slot is not EPRouterBalancingMixin._materialize_native_balancing_slot
    }
    assert implementing == {"EPLfm2MoELayer"}, implementing

    lfm2 = _bare_layer(EPLfm2MoELayer, _Gate(None))
    assert lfm2._can_materialize_native_balancing_slot() is True
    glm4 = _bare_layer(EPGlm4MoELayer, _Gate("e_score_correction_bias"))
    assert glm4._can_materialize_native_balancing_slot() is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
