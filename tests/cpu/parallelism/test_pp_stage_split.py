#!/usr/bin/env python
"""Stage splitting: layer partition, family registration and rejection, and GLOBAL parameter naming.

The naming property is the load-bearing one. Slicing the decoder-layer list re-bases local indices
to 0, so stage 1's first layer is ``layers.0`` locally while being layer N globally. Any consumer
that reads a stage's ``state_dict`` — checkpoint saving, vLLM weight sync, adapter merging — would
otherwise write it under stage 0's name and silently OVERWRITE a different layer. So the stages'
state_dicts must partition the unsplit model's keys exactly: no collisions, no omissions.

Run: python tests/cpu/parallelism/test_pp_stage_split.py
"""

import pytest
import torch
from transformers import (
    GptOssConfig,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3ForSequenceClassification,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
    Qwen3MoeForSequenceClassification,
    Qwen3VLMoeConfig,
    Qwen3VLMoeForConditionalGeneration,
)
from transformers.models.step3p7.configuration_step3p7 import Step3p7TextConfig

from halo.distributed.pipeline_parallel import model_rules
from halo.distributed.pipeline_parallel.families import PP_SPEC_MAP
from halo.distributed.pipeline_parallel.model_rules import MTP_LAYER_COUNT_FIELDS, reject_unsupported_pp_model
from halo.distributed.pipeline_parallel.partition import (
    compute_layer_partition,
    head_cost_layer_equivalents,
    layer_types_period,
    validate_partition,
)
from halo.distributed.pipeline_parallel.stage import StageLayout, build_pipeline_stage
from halo.models.head_transform import IDENTITY_HEAD_TRANSFORM
from halo.models.moe.config_fields import ROUTER_TOPK_FIELDS
from halo.models.param_roles import persistent_buffers
from tests.common.models import TINY_GPTOSS_CONFIG, TINY_INKLING_PP_OVERRIDES, TINY_QWEN3_CONFIG, TINY_STEP3P7_CONFIG
from tests.common.pp_fakes import registered_pp_backbones
from tests.common.tiny_models import (
    TINY_DENSE_FAMILIES,
    TINY_MOE_FAMILIES,
    TINY_MOE_VLM_FAMILIES,
    TINY_UNWRAPPED_MOE_FAMILY,
    tiny_family_model,
)

# Qwen3 MoE balances through the HF router aux loss (auto → aux_loss) — the path a stage severs, so
# this file needs the coefficient LIVE at its transformers default. Deliberately NOT tests/common's
# TINY_QWEN3_MOE_CONFIG, which zeroes router_aux_loss_coef precisely to get past that gate.
_PP_SPLIT_MOE_CONFIG = {
    "vocab_size": 256,
    "hidden_size": 64,
    "intermediate_size": 128,
    "moe_intermediate_size": 32,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "decoder_sparse_step": 1,
    "max_position_embeddings": 128,
    "tie_word_embeddings": False,
}


# The fields head_cost_layer_equivalents reads, MoE-shaped; router width left to each test.
_MOE_COST_FIELDS = {
    "hidden_size": 256,
    "vocab_size": 4096,
    "num_attention_heads": 8,
    "moe_intermediate_size": 128,
    "num_experts": 64,
}


class _TopKConfig:
    """The whole config surface the head-cost model reads: plain attributes + ``get_text_config``.

    A real config cannot carry every top-k spelling (each family declares exactly one), so the cost
    model's field coverage is exercised against this stand-in.
    """

    def __init__(self, **fields):
        self.__dict__.update(fields)

    def get_text_config(self):
        return self


class _CompositeTopKConfig(_TopKConfig):
    """A wrapper (VLM-shaped) config carrying the decoder fields on ``text_config`` only."""

    def __init__(self, text_config):
        super().__init__(text_config=text_config)

    def get_text_config(self):
        return self.text_config


def _model(**overrides):
    torch.manual_seed(0)
    return Qwen3ForCausalLM(Qwen3Config(**{**TINY_QWEN3_CONFIG, **overrides}))


def _moe_model(**overrides):
    torch.manual_seed(0)
    return Qwen3MoeForCausalLM(Qwen3MoeConfig(**{**_PP_SPLIT_MOE_CONFIG, **overrides}))


def _inkling_model():
    return tiny_family_model(TINY_MOE_FAMILIES["inkling_text"], overrides=TINY_INKLING_PP_OVERRIDES)


def test_partition_is_contiguous_and_covers_every_layer():
    for n_layers, pp in [(8, 2), (8, 4), (12, 3), (7, 2), (10, 4), (5, 5)]:
        part = compute_layer_partition(n_layers, pp)
        assert len(part) == pp
        assert part[0][0] == 0 and part[-1][1] == n_layers, (n_layers, pp, part)
        for (_, prev_hi), (lo, _) in zip(part, part[1:], strict=False):
            assert prev_hi == lo, f"gap/overlap in {part}"
        assert all(hi > lo for lo, hi in part), f"empty stage in {part}"


def test_partition_is_balanced_and_favours_the_ends():
    """Remainder layers go to middle stages: stage 0 also carries the embedding, the last the head."""
    sizes = [hi - lo for lo, hi in compute_layer_partition(8, 2)]
    assert sizes == [4, 4]
    sizes = [hi - lo for lo, hi in compute_layer_partition(10, 4)]
    assert max(sizes) - min(sizes) <= 1, sizes
    assert sizes[1] >= sizes[0] and sizes[2] >= sizes[3], f"ends should not be heaviest: {sizes}"


def test_partition_rejects_more_stages_than_layers():
    with pytest.raises(ValueError, match="every stage needs at least one layer"):
        compute_layer_partition(3, 4)


def test_head_weighted_partition_shrinks_the_last_stage():
    """A head worth ~3 layers moves ~1.5 layers off the last stage of a 2-stage split; the total is
    preserved and every stage keeps >= 1 layer. Exact expectations pin the rounding."""
    part = compute_layer_partition(36, 2, head_layer_equivalents=3.2)
    assert [hi - lo for lo, hi in part] == [20, 16], part
    part = compute_layer_partition(36, 4, head_layer_equivalents=3.2)
    sizes = [hi - lo for lo, hi in part]
    assert sum(sizes) == 36 and sizes[-1] == 7 and min(sizes) >= 1, sizes
    # A huge head equivalent must clamp, never starve the last stage below one layer.
    part = compute_layer_partition(8, 2, head_layer_equivalents=100.0)
    assert [hi - lo for lo, hi in part] == [7, 1], part


def test_head_weighted_partition_respects_the_layer_types_period():
    """Boundaries must land on whole periods (position-indexed attention-mask families)."""
    # A 3-layer head cannot justify a 4-layer (whole-period) shift.
    part = compute_layer_partition(24, 2, head_layer_equivalents=3.0, boundary_period=4)
    assert [hi - lo for lo, hi in part] == [12, 12], part
    # An 8-layer head does: one whole period moves off the last stage.
    part = compute_layer_partition(24, 2, head_layer_equivalents=8.0, boundary_period=4)
    assert [hi - lo for lo, hi in part] == [16, 8], part
    # Indivisible layer count: boundaries stay whole periods and the ragged tail rides the LAST
    # stage — an un-snapped 13/13 "uniform" here puts the boundary mid-period, which
    # validate_partition then refuses, so the default partition would be a dead end.
    part = compute_layer_partition(26, 2, head_layer_equivalents=3.0, boundary_period=4)
    assert [hi - lo for lo, hi in part] == [16, 10], part
    assert part[1][0] % 4 == 0, part


def test_pp_split_override_wins_and_is_validated():
    part = compute_layer_partition(36, 2, head_layer_equivalents=3.2, pp_split=[20, 16])
    assert [hi - lo for lo, hi in part] == [20, 16], part
    with pytest.raises(ValueError, match="sums to"):
        compute_layer_partition(36, 2, pp_split=[20, 15])
    with pytest.raises(ValueError, match="entries"):
        compute_layer_partition(36, 2, pp_split=[36])
    with pytest.raises(ValueError, match=">= 1"):
        compute_layer_partition(36, 2, pp_split=[36, 0])


def test_head_cost_layer_equivalents_from_real_configs():
    """Dense qwen3-8B-scale: the 622M-param head is worth ~3 of its ~193M-param layers. A MoE
    counts per-token ACTIVATED expert params. A config missing the fields returns 0 (uniform)."""
    dense = Qwen3Config(
        hidden_size=4096,
        vocab_size=151936,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        intermediate_size=12288,
    )
    equiv = head_cost_layer_equivalents(dense)
    assert 2.5 < equiv < 4.0, equiv
    moe = GptOssConfig(**TINY_GPTOSS_CONFIG)
    assert head_cost_layer_equivalents(moe) > 0
    assert head_cost_layer_equivalents(None) == 0.0


@pytest.mark.parametrize("field", ROUTER_TOPK_FIELDS)
def test_head_cost_counts_activated_experts_under_every_topk_spelling(field):
    """The cost model must read the router width through the shared registry, not two spellings.

    ``ffn_mult`` scales the per-layer FFN term the head is weighed against, so a family whose config
    uses a spelling this gate missed (Zaya and the DeepSeek / Ling lineage spell it
    ``moe_router_topk``) would be costed as if it activated ONE expert per token — a head that
    dominates a layer, and a last stage silently starved of decoder layers.
    """
    unrouted = head_cost_layer_equivalents(_TopKConfig(**_MOE_COST_FIELDS))
    routed = head_cost_layer_equivalents(_TopKConfig(**_MOE_COST_FIELDS, **{field: 8}))

    assert routed > 0
    assert routed < unrouted / 2, (
        f"{field}=8 must raise the FFN term ~8x and shrink the head's layer-equivalents; "
        f"got {routed} against the unrouted fallback {unrouted}"
    )
    # Every spelling must agree — a per-spelling divergence is the split imbalance this guards.
    assert routed == pytest.approx(head_cost_layer_equivalents(_TopKConfig(**_MOE_COST_FIELDS, num_experts_per_tok=8)))


def test_head_cost_reads_the_topk_a_wrapper_hides_on_its_text_config():
    """A composite (VLM) config carries the router width on ``text_config`` only."""
    plain = _TopKConfig(**_MOE_COST_FIELDS, num_experts_per_tok=8)
    assert head_cost_layer_equivalents(_CompositeTopKConfig(plain)) == pytest.approx(
        head_cost_layer_equivalents(plain)
    )


def test_a_dense_config_is_never_routed_by_the_generation_top_k():
    """``top_k`` is the last-resort router spelling AND the legacy generation sampling parameter.

    Read as a router width on a dense model it would divide the head's layer-equivalents by the
    sampling k and hand the last stage far too many layers, so the lookup only applies where the
    config actually declares experts.
    """
    dense = {k: v for k, v in _MOE_COST_FIELDS.items() if k != "num_experts"}
    dense["intermediate_size"] = dense.pop("moe_intermediate_size")
    assert head_cost_layer_equivalents(_TopKConfig(**dense, top_k=50)) == pytest.approx(
        head_cost_layer_equivalents(_TopKConfig(**dense))
    )


def test_layer_types_period_detection():
    assert layer_types_period(GptOssConfig(**TINY_GPTOSS_CONFIG)) == 2  # sliding/full alternation
    assert layer_types_period(Qwen3Config(num_hidden_layers=8)) == 1  # uniform full attention
    assert layer_types_period(None) == 1


class _TextlessConfig:
    """A config whose ``get_text_config()`` yields nothing while it carries the decoder fields itself."""

    def __init__(self, layer_types):
        self.layer_types = layer_types

    def get_text_config(self):
        return None


def test_layer_type_gate_and_the_boundary_period_read_one_seam():
    """Both halves of the ``layer_types`` contract must see the same field on the same config.

    ``layer_types_period`` snaps the automatic partition to whole periods and
    ``validate_partition`` refuses the offsets that snapping does not produce — a manual
    ``pp_split`` is checked by the gate alone. A config the two read differently (``get_text_config``
    returning nothing while the fields sit on the wrapper) would leave that split un-snapped AND
    un-gated: every stage after the first silently swaps sliding-window and full causal masks.
    """
    config = _TextlessConfig(["full_attention", "sliding_attention"] * 4)
    assert layer_types_period(config) == 2, "premise: the period seam reads this config"

    model = _model()
    model.config = config
    with pytest.raises(ValueError, match="layer_types"):
        validate_partition(model, [(0, 1), (1, 8)])
    validate_partition(model, [(0, 2), (2, 8)])  # a whole period re-bases to a no-op


def test_head_cost_survives_per_layer_heterogeneous_attention():
    """step3p7 registers its attention fields per layer (64 full / 96 sliding heads on the hub), so
    the bare ``config.num_attention_heads`` read RAISES transformers'
    AmbiguousGlobalPerLayerAttributeError — a RuntimeError the getattr default does not swallow,
    which would stop ``resolve_layer_partition`` before any PP gate could fire. The cost model must
    resolve through the per-layer-aware seam instead (max: it is a cost ceiling)."""
    config = Step3p7TextConfig(**TINY_STEP3P7_CONFIG)
    with pytest.raises(RuntimeError):
        _ = config.num_attention_heads  # premise: the global read is ambiguous on this family
    assert head_cost_layer_equivalents(config) > 0


def test_ragged_layer_types_snap_to_whole_period_boundaries():
    """step3p7's real 45-layer list is 11 whole ``full,s,s,s`` periods plus one trailing layer, so
    no period tiles it exactly — period detection must use the same shift-invariance criterion the
    rebase gate enforces, and the automatic partition must put every boundary on a whole period
    with the ragged tail on the last stage. An even 23/22 pp2 split would start stage 1 at offset 23,
    which ``validate_partition`` refuses with no config knob to route around it."""
    period = ["full_attention", "sliding_attention", "sliding_attention", "sliding_attention"]
    config = _TextlessConfig((period * 12)[:45])
    assert layer_types_period(config) == 4

    part = compute_layer_partition(45, 2, head_layer_equivalents=1.0, boundary_period=4)
    assert part[-1][1] == 45 and all(lo % 4 == 0 for lo, _ in part), part
    model = _model()
    model.config = config
    validate_partition(model, part)  # every default boundary passes the gate


def test_stage_build_rejects_a_model_whose_input_embedding_cannot_be_located():
    """A non-first stage must DROP the embedding; an unlocatable one is fatal, never skipped.

    Left in place it stays in this stage's state_dict, and the merged checkpoint index then resolves
    the embedding to the last stage's untrained copy.
    """
    model = _model()
    model.get_input_embeddings = lambda: None
    with pytest.raises(ValueError, match="get_input_embeddings"):
        build_pipeline_stage(model, 1, 2)

    # An embedding the accessor hands back but the tree does not hold is the same fatal miss, and it
    # must be reported by the one identity lookup that names a stage's parameters.
    stray = _model()
    stray.get_input_embeddings = lambda: torch.nn.Embedding(4, 4)
    with pytest.raises(ValueError, match="is not a module of"):
        build_pipeline_stage(stray, 1, 2)


def test_global_names_partition_the_unsplit_model_exactly():
    """Union of every stage's GLOBAL names == the unsplit model's keys, with no key claimed twice.

    The mapping is applied explicitly at serialization boundaries, so this exercises
    global_parameter_name over each stage's LOCAL state_dict rather than the stage's state_dict()
    directly -- the module's own naming stays local on purpose (see global_parameter_name).
    """
    reference = set(_model().state_dict())

    pp_size = 2
    seen: dict[str, int] = {}
    for pp_rank in range(pp_size):
        stage = build_pipeline_stage(_model(), pp_rank, pp_size)
        for local in stage.state_dict():
            key = stage.global_parameter_name(local)
            assert key not in seen, f"{key} claimed by stage {seen[key]} and stage {pp_rank}"
            seen[key] = pp_rank

    assert set(seen) == reference, (
        f"missing from stages: {sorted(reference - set(seen))}; invented by stages: {sorted(set(seen) - reference)}"
    )


def test_module_naming_stays_local_so_namespace_joins_hold():
    """named_parameters() and state_dict() must agree, and must describe the REAL module tree.

    Overriding them to return global names silently breaks every consumer that joins names across
    APIs -- HF's weight-decay grouping intersects get_parameter_names() (named_children-derived)
    with named_parameters(), and torch DCP walks named_children() too. A stage must also be able to
    reload its own state_dict.
    """
    stage = build_pipeline_stage(_model(), 1, 2)
    param_names = {n for n, _ in stage.named_parameters()}
    assert param_names <= set(stage.state_dict()), "named_parameters and state_dict disagree"
    assert all(n.startswith(("model.", "head.")) for n in param_names), sorted(param_names)[:3]
    missing, unexpected = stage.load_state_dict(stage.state_dict(), strict=True)
    assert not missing and not unexpected


def test_later_stage_layers_keep_their_global_indices():
    """Stage 1's first decoder layer must NOT be named layers.0 — that is stage 0's layer."""
    stage1 = build_pipeline_stage(_model(), 1, 2)
    keys = [stage1.global_parameter_name(k) for k in stage1.state_dict() if ".layers." in k]
    indices = sorted({int(k.split(".layers.")[1].split(".")[0]) for k in keys})
    assert indices == [4, 5, 6, 7], f"stage 1 should own global layers 4-7, got {indices}"


def test_stage_zero_owns_embedding_and_last_owns_head():
    s0, s1 = build_pipeline_stage(_model(), 0, 2), build_pipeline_stage(_model(), 1, 2)
    first = {s0.global_parameter_name(k) for k in s0.state_dict()}
    last = {s1.global_parameter_name(k) for k in s1.state_dict()}
    assert any("embed_tokens" in k for k in first) and not any("embed_tokens" in k for k in last)
    assert any(k.startswith("lm_head") for k in last) and not any(k.startswith("lm_head") for k in first)
    assert "model.norm.weight" in last and "model.norm.weight" not in first


@registered_pp_backbones("InklingTextModel")
def test_an_embedding_norm_stays_on_stage_zero_with_the_embedding():
    """Inkling's ``embed_norm`` normalizes the token embedding only stage 0 computes: a later stage
    drops it with the embedding module that carries it (else every stage writes it), and its loader
    counts that module droppable, so a per-node checkpoint without stage 0's shard still loads there."""
    s0, s1 = (build_pipeline_stage(_inkling_model(), rank, 2, moe_balancing="none") for rank in (0, 1))
    assert "model.embed_tokens.embed_norm.weight" in s0.checkpoint_name_map()
    assert not any("embed_norm" in name for name in s1.checkpoint_name_map())
    assert "model.embed_tokens" in StageLayout.of(_inkling_model()).dropped_prefixes(False, True)
    assert "model.embed_tokens" not in StageLayout.of(_inkling_model()).dropped_prefixes(True, False)


def test_persistent_buffers_survive_the_split():
    """A persistent buffer on a later stage's layer must reach the save path, under its global name.

    ``persistent_buffers`` computes ``state_dict - named_parameters`` and then filters
    ``named_buffers``: it only yields anything while those three namespaces agree, and the name it
    yields is what a serializer maps. A buffer that falls out here (e.g. Gemma4 ``layer_scalar``) is
    silently dropped from the checkpoint.
    """
    model = _model()
    model.model.layers[5].register_buffer("layer_scalar", torch.ones(3), persistent=True)
    stage = build_pipeline_stage(model, 1, 2)

    local_names = dict(persistent_buffers(stage)).keys()
    assert local_names == {"model.layers.1.layer_scalar"}
    assert {stage.global_parameter_name(n) for n in local_names} == {"model.layers.5.layer_scalar"}


def test_global_parameter_name_round_trips_for_every_parameter():
    """Every stage-local name maps to a real name in the unsplit model."""
    reference = set(_model().state_dict())
    for pp_rank in range(2):
        stage = build_pipeline_stage(_model(), pp_rank, 2)
        for local, _ in stage.named_parameters():
            assert stage.global_parameter_name(local) in reference, local


def test_tied_embeddings_are_rejected():
    """embed_tokens and lm_head are one weight; split across ranks they diverge from step 1."""
    with pytest.raises(ValueError, match="tie_word_embeddings"):
        build_pipeline_stage(_model(tie_word_embeddings=True), 0, 2)


def test_tie_rejection_reads_the_weights_not_only_the_config():
    """A model whose head is tied to its embedding is rejected even with the config flag off."""
    model = _model()
    model.lm_head.weight = model.model.embed_tokens.weight
    with pytest.raises(ValueError, match="tie_word_embeddings"):
        reject_unsupported_pp_model(model)


def test_head_that_cannot_be_tied_is_accepted():
    """A score/classifier head inherits tie_word_embeddings from its base LM config, tying nothing."""
    config = Qwen3Config(**{**TINY_QWEN3_CONFIG, "tie_word_embeddings": True, "num_labels": 2})
    model = Qwen3ForSequenceClassification(config)
    assert model.get_output_embeddings() is None
    reject_unsupported_pp_model(model)


def test_untied_model_is_accepted():
    """The gate must not reject everything: an untied model splits normally."""
    model = _model()
    assert model.lm_head.weight is not model.model.embed_tokens.weight
    reject_unsupported_pp_model(model)


@registered_pp_backbones("Qwen3MoeModel")
def test_aux_loss_balanced_moe_is_rejected():
    """A stage bypasses *ForCausalLM.forward, where the router aux loss is added to the loss."""
    assert _moe_model().config.router_aux_loss_coef > 0
    with pytest.raises(ValueError, match="aux_loss"):
        build_pipeline_stage(_moe_model(), 0, 2)


@registered_pp_backbones("Qwen3MoeModel")
def test_bias_update_balanced_moe_is_accepted():
    """Families balanced by the aux-loss-free bias update do not touch the severed path.

    Mirrors what ``apply_balancing_mode``'s bias enable installs (plain attributes on the router), which
    is what ``moe_balancing: auto`` keys on to resolve ``bias_update``.
    """
    model = _moe_model()
    for layer in model.model.layers:
        layer.mlp.gate.balancing_biases = torch.zeros(model.config.num_experts)
        layer.mlp.gate.expert_load_counter = None
    reject_unsupported_pp_model(model)


@registered_pp_backbones("Qwen3MoeModel")
def test_explicit_non_aux_loss_balancing_is_accepted():
    """The gate judges the run's resolved strategy, not the family in the abstract."""
    for requested in ("bias_update", "none"):
        reject_unsupported_pp_model(_moe_model(), requested)


@registered_pp_backbones("Qwen3VLMoeTextModel")
def test_composite_vlm_moe_is_rejected():
    """A VLM carries the expert fields on text_config; a top-level-only read would miss the family.

    Requested explicitly: this wrapper's forward reads ``output_router_logits`` out of ``kwargs``
    and never from the config, so ``auto`` resolves it to ``none`` (nothing here can balance at all —
    ``resolve_balancing_mode``) and there would be no aux-loss strategy for a stage to sever.
    """
    config = Qwen3VLMoeConfig(
        text_config={**_PP_SPLIT_MOE_CONFIG, "num_hidden_layers": 2},
        vision_config={"hidden_size": 32, "intermediate_size": 64, "depth": 2, "num_heads": 2, "out_hidden_size": 64},
    )
    assert getattr(config, "num_experts", None) is None, "top-level config must not carry the expert fields"
    with pytest.raises(ValueError, match="aux_loss"):
        reject_unsupported_pp_model(Qwen3VLMoeForConditionalGeneration(config), "aux_loss")


@registered_pp_backbones("Qwen3MoeModel")
def test_aux_loss_gate_sees_a_coefficient_only_the_wrapper_declares(monkeypatch):
    """The coefficient is read through the wrapper-safe router seam, not off ``text_config`` alone.

    A composite config declaring ``router_aux_loss_coef`` at the top level while its text sub-config
    carries no such field reads back as 0 through a text-only probe — the gate then passes, PP
    severs ``*ForCausalLM.forward``, and the balancing term is dropped for the whole run with
    nothing raised. Every composite in the roster declares it on the text side (the direction a
    text-only read handles), so this pins the seam's other half rather than a live failure.
    """
    config = _CompositeTopKConfig(_TopKConfig(num_experts=4, num_experts_per_tok=2))
    config.router_aux_loss_coef = 0.001
    assert not hasattr(config.get_text_config(), "router_aux_loss_coef"), "premise: the text level is silent"

    model = _moe_model()
    model.config = config
    # The stand-in config builds no model, so the head-path verification (its own test file) is
    # stubbed out of this gate's way.
    monkeypatch.setattr(model_rules, "resolve_head_transform", lambda _model: IDENTITY_HEAD_TRANSFORM)
    with pytest.raises(ValueError, match="aux_loss"):
        reject_unsupported_pp_model(model, "aux_loss")


@registered_pp_backbones("Qwen3MoeModel")
def test_moe_sequence_classification_is_accepted():
    """Only the causal-LM forward adds the aux term; a score head never does, so PP drops nothing."""
    model = Qwen3MoeForSequenceClassification(Qwen3MoeConfig(**_PP_SPLIT_MOE_CONFIG, num_labels=2))
    assert model.config.router_aux_loss_coef > 0
    reject_unsupported_pp_model(model)


@registered_pp_backbones("Qwen3MoeModel")
def test_moe_without_an_aux_loss_term_is_accepted():
    """coef=0 means no aux term is added at all, so a stage drops nothing."""
    reject_unsupported_pp_model(_moe_model(router_aux_loss_coef=0.0))


def test_unsupported_families_are_rejected_with_a_mechanism():
    for backbone_name in ("ZayaModel", "Gemma4TextModel"):
        spec = PP_SPEC_MAP[backbone_name]
        assert len(spec.refusal or "") > 40, f"{backbone_name} rejects PP without a reason"


# Written out rather than read off PP_FAMILY_SPECS, so a family dropped from or slipped into the
# registry fails here.
REGISTERED_BACKBONES = {
    "GptOssModel",
    "Qwen3Model",
    "Qwen3_5TextModel",
    "Qwen3_5MoeTextModel",
    "Step3p7TextModel",
    "DeepseekV4Model",
    "Glm5NextTextModel",
}
_TINY_FAMILIES = {**TINY_DENSE_FAMILIES, **TINY_MOE_FAMILIES, **TINY_MOE_VLM_FAMILIES}
# The tiny families whose backbone is neither registered nor refused with a spec of its own.
_UNREGISTERED_FAMILIES = (
    "bailing_moe",
    "cohere2_moe",
    "glm4_moe_lite",
    "laguna",
    "lfm2_moe",
    "mistral4",
    "qwen3_moe",
    "qwen3_next",
    "qwen4_exp_text",
)


def test_the_registry_holds_exactly_the_validated_families():
    assert {name for name, spec in PP_SPEC_MAP.items() if spec.validated} == REGISTERED_BACKBONES
    assert all(PP_SPEC_MAP[name].refusal is None for name in REGISTERED_BACKBONES)


@pytest.mark.parametrize(
    "family", ["qwen3", "qwen3_5", "gpt_oss", "qwen3_5_moe_text", "qwen3_5_moe", "step3p7", "deepseek_v4", "glm5_next"]
)
def test_every_registered_family_passes_the_structure_gate(family):
    model = tiny_family_model(_TINY_FAMILIES[family])
    reject_unsupported_pp_model(model, "none")


@pytest.mark.parametrize("family", _UNREGISTERED_FAMILIES)
def test_an_unregistered_backbone_is_refused_by_name(family):
    """The verdict names the backbone class and the registered ones, so it reads the same on every
    rank: it depends on nothing but the class."""
    model = tiny_family_model(_TINY_FAMILIES[family], overrides={"tie_word_embeddings": False})
    backbone = type(model.get_decoder()).__name__
    assert backbone not in PP_SPEC_MAP, f"premise: {backbone} has no spec"
    with pytest.raises(ValueError, match=rf"^{backbone} is not registered for pipeline parallelism") as refusal:
        reject_unsupported_pp_model(model, "none")
    assert all(name in str(refusal.value) for name in REGISTERED_BACKBONES)
    with pytest.raises(ValueError, match="is not registered"):
        build_pipeline_stage(model, 0, 2, moe_balancing="none")


def test_a_backbone_without_a_family_wrapper_is_refused_by_name():
    with pytest.raises(ValueError, match=r"^MixtralModel is not registered"):
        reject_unsupported_pp_model(tiny_family_model(TINY_UNWRAPPED_MOE_FAMILY), "none")


def test_registration_opt_in_admits_and_restores():
    """The tests' opt-in admits an unregistered family for its block only."""
    model = _moe_model(router_aux_loss_coef=0.0)
    with registered_pp_backbones("Qwen3MoeModel"):
        reject_unsupported_pp_model(model)
    with pytest.raises(ValueError, match="Qwen3MoeModel is not registered"):
        reject_unsupported_pp_model(model)


def test_a_family_refusal_comes_before_the_registration_verdict():
    model = _inkling_model()
    with pytest.raises(ValueError, match=r"^InklingTextModel is refused for pipeline parallelism\. .*loader-only"):
        reject_unsupported_pp_model(model, "none")


def test_the_tie_refusal_comes_before_the_registration_verdict():
    """An unregistered tied model is told about the tie, the reason no registration would lift."""
    with pytest.raises(ValueError, match="tie_word_embeddings"):
        reject_unsupported_pp_model(_moe_model(tie_word_embeddings=True, router_aux_loss_coef=0.0))


def test_the_mtp_refusal_comes_before_the_registration_verdict():
    model = _moe_model(router_aux_loss_coef=0.0)
    model.config.num_nextn_predict_layers = 1
    model.model.layers.append(model.model.layers[-1])
    with pytest.raises(ValueError, match="num_nextn_predict_layers=1"):
        reject_unsupported_pp_model(model)


def test_model_without_a_layer_list_is_rejected():
    with pytest.raises(ValueError, match="decoder-layer list"):
        reject_unsupported_pp_model(torch.nn.Linear(4, 4))


# Written out rather than read from MTP_LAYER_COUNT_FIELDS: parametrizing over the constant under
# test would shrink the case list with it, so a dropped spelling would still pass.
MTP_FIELDS_IN_THE_WILD = ("num_nextn_predict_layers", "mtp_num_hidden_layers")


def test_mtp_gate_covers_every_known_spelling():
    assert set(MTP_FIELDS_IN_THE_WILD) <= set(MTP_LAYER_COUNT_FIELDS)


@pytest.mark.parametrize("field", MTP_FIELDS_IN_THE_WILD)
def test_every_mtp_spelling_rejects_a_live_tail(field):
    """Each MTP field must reject PP on its own — when the tail layers are actually BUILT.

    The families disagree on the spelling (``num_nextn_predict_layers`` for DeepSeek/Bailing,
    ``mtp_num_hidden_layers`` for Qwen3.5/3.6), and checking only one lets a config declaring the
    other through. Liveness is the decoder list outgrowing ``num_hidden_layers`` — exactly how the
    live-MTP construction (remote-code Bailing v3) appends its tail. The failure it guards is a
    mid-run crash on a non-first stage, not a config-time raise.
    """
    model = Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG))
    setattr(model.config, field, 1)
    model.model.layers.append(model.model.layers[-1])  # the appended MTP tail a live family builds
    with pytest.raises(ValueError, match=field):
        reject_unsupported_pp_model(model)


@pytest.mark.parametrize("field", MTP_FIELDS_IN_THE_WILD)
def test_mtp_field_declared_but_not_built_is_accepted(field):
    """The gate judges the BUILT tree, not the declaration.

    Step3p7, Qwen3.5/3.6, Inkling, DeepSeek-V4 and GLM-5 hub configs declare 1-3 MTP tail layers as
    checkpoint metadata while the load drops the weights (``_keys_to_ignore_on_load_unexpected``);
    a declaration-only gate would refuse every one of those checkpoints for layers that do not exist.
    """
    model = Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG))
    setattr(model.config, field, 3)
    assert len(model.model.layers) == model.config.num_hidden_layers, "premise: no tail was built"
    reject_unsupported_pp_model(model)


@pytest.mark.parametrize("field", MTP_FIELDS_IN_THE_WILD)
def test_mtp_field_declared_at_zero_is_accepted(field):
    """Anti-vacuity: the gate reads the COUNT, not the field's presence (Ling-mini-2.0 ships 0)."""
    model = Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG))
    setattr(model.config, field, 0)
    reject_unsupported_pp_model(model)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
