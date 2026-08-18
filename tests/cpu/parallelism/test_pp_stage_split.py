#!/usr/bin/env python
"""Stage splitting: layer partition, per-family rejection, and GLOBAL parameter naming.

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
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3ForSequenceClassification,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
    Qwen3MoeForSequenceClassification,
    Qwen3VLMoeConfig,
    Qwen3VLMoeForConditionalGeneration,
)

from src.distributed.pipeline_parallel.split import (
    MTP_LAYER_COUNT_FIELDS,
    PPModelSpec,
    compute_layer_partition,
    head_cost_layer_equivalents,
    layer_types_period,
    resolve_pp_spec,
    validate_model_supports_pp,
)
from src.distributed.pipeline_parallel.stage import build_pipeline_stage, module_path, reject_layer_type_rebase
from src.models.moe_balancing import ROUTER_TOPK_FIELDS
from src.models.structure import persistent_buffers
from tests.common.models import TINY_GPTOSS_CONFIG, TINY_QWEN3_CONFIG

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
    # reject_layer_type_rebase then refuses, so the default partition would be a dead end.
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
    from transformers import GptOssConfig, Qwen3Config

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
    from transformers import GptOssConfig, Qwen3Config

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
    ``reject_layer_type_rebase`` refuses the offsets that snapping does not produce — a manual
    ``pp_split`` is checked by the gate alone. A config the two read differently (``get_text_config``
    returning nothing while the fields sit on the wrapper) would leave that split un-snapped AND
    un-gated: every stage after the first silently swaps sliding-window and full causal masks.
    """
    config = _TextlessConfig(["full_attention", "sliding_attention"] * 4)
    assert layer_types_period(config) == 2, "premise: the period seam reads this config"

    model = _model()
    model.config = config
    with pytest.raises(ValueError, match="layer_types"):
        reject_layer_type_rebase(model, 1, 1)
    reject_layer_type_rebase(model, 2, 1)  # a whole period re-bases to a no-op


def test_head_cost_survives_per_layer_heterogeneous_attention():
    """step3p7 registers its attention fields per layer (64 full / 96 sliding heads on the hub), so
    the bare ``config.num_attention_heads`` read RAISES transformers'
    AmbiguousGlobalPerLayerAttributeError — a RuntimeError the getattr default does not swallow.
    Pre-fix this crashed ``resolve_layer_partition`` before any PP gate could fire; the cost model
    must resolve through the per-layer-aware seam instead (max: it is a cost ceiling)."""
    from transformers.models.step3p7.configuration_step3p7 import Step3p7TextConfig

    from tests.common.models import TINY_STEP3P7_CONFIG

    config = Step3p7TextConfig(**TINY_STEP3P7_CONFIG)
    with pytest.raises(RuntimeError):
        _ = config.num_attention_heads  # premise: the global read is ambiguous on this family
    assert head_cost_layer_equivalents(config) > 0


def test_ragged_layer_types_snap_to_whole_period_boundaries():
    """step3p7's real 45-layer list is 11 whole ``full,s,s,s`` periods plus one trailing layer, so
    no period tiles it exactly — period detection must use the same shift-invariance criterion the
    rebase gate enforces, and the automatic partition must put every boundary on a whole period
    with the ragged tail on the last stage. Pre-fix the default pp2 partition was 23/22, whose
    offset 23 ``reject_layer_type_rebase`` then loudly refused — a dead end with no config knob."""
    period = ["full_attention", "sliding_attention", "sliding_attention", "sliding_attention"]
    config = _TextlessConfig((period * 12)[:45])
    assert layer_types_period(config) == 4

    part = compute_layer_partition(45, 2, head_layer_equivalents=1.0, boundary_period=4)
    assert part[-1][1] == 45 and all(lo % 4 == 0 for lo, _ in part), part
    model = _model()
    model.config = config
    for stage_rank, (lo, _) in enumerate(part):
        reject_layer_type_rebase(model, lo, stage_rank)  # every default boundary passes the gate


def test_module_path_rejects_a_module_outside_the_tree():
    """The identity lookup behind global naming and the loader's droppable prefixes must fail loud.

    A bare ``next(...)`` raised ``StopIteration`` here, which reads as an exhausted iterator rather
    than as "this model does not contain that module".
    """
    with pytest.raises(ValueError, match="is not a module of"):
        module_path(_model(), torch.nn.Linear(4, 4))


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
        validate_model_supports_pp(model)


def test_head_that_cannot_be_tied_is_accepted():
    """A score/classifier head inherits tie_word_embeddings from its base LM config, tying nothing."""
    config = Qwen3Config(**{**TINY_QWEN3_CONFIG, "tie_word_embeddings": True, "num_labels": 2})
    model = Qwen3ForSequenceClassification(config)
    assert model.get_output_embeddings() is None
    validate_model_supports_pp(model)


def test_untied_model_is_accepted():
    """The gate must not reject everything: an untied model splits normally."""
    model = _model()
    assert model.lm_head.weight is not model.model.embed_tokens.weight
    validate_model_supports_pp(model)


def test_aux_loss_balanced_moe_is_rejected():
    """A stage bypasses *ForCausalLM.forward, where the router aux loss is added to the loss."""
    assert _moe_model().config.router_aux_loss_coef > 0
    with pytest.raises(ValueError, match="aux_loss"):
        build_pipeline_stage(_moe_model(), 0, 2)


def test_bias_update_balanced_moe_is_accepted():
    """Families balanced by the aux-loss-free bias update do not touch the severed path.

    Mirrors what ``_enable_router_bias_balancing`` installs (plain attributes on the router), which
    is what ``moe_balancing: auto`` keys on to resolve ``bias_update``.
    """
    model = _moe_model()
    for layer in model.model.layers:
        layer.mlp.gate.balancing_biases = torch.zeros(model.config.num_experts)
        layer.mlp.gate.expert_load_counter = None
    validate_model_supports_pp(model)


def test_explicit_non_aux_loss_balancing_is_accepted():
    """The gate judges the run's resolved strategy, not the family in the abstract."""
    for requested in ("bias_update", "none"):
        validate_model_supports_pp(_moe_model(), requested)


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
        validate_model_supports_pp(Qwen3VLMoeForConditionalGeneration(config), "aux_loss")


def test_aux_loss_gate_sees_a_coefficient_only_the_wrapper_declares():
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
    with pytest.raises(ValueError, match="aux_loss"):
        validate_model_supports_pp(model, "aux_loss")


def test_moe_sequence_classification_is_accepted():
    """Only the causal-LM forward adds the aux term; a score head never does, so PP drops nothing."""
    model = Qwen3MoeForSequenceClassification(Qwen3MoeConfig(**_PP_SPLIT_MOE_CONFIG, num_labels=2))
    assert model.config.router_aux_loss_coef > 0
    validate_model_supports_pp(model)


def test_moe_without_an_aux_loss_term_is_accepted():
    """coef=0 means no aux term is added at all, so a stage drops nothing."""
    validate_model_supports_pp(_moe_model(router_aux_loss_coef=0.0))


def test_unsupported_families_are_rejected_with_a_mechanism():
    from src.distributed.pipeline_parallel.split import PP_SPEC_MAP

    for backbone_name in ("ZayaModel", "Gemma4TextModel"):
        spec = PP_SPEC_MAP[backbone_name]
        assert spec.SUPPORTS_PP is False
        assert len(spec.UNSUPPORTED_REASON) > 40, f"{backbone_name} rejects PP without a reason"


def test_unclaimed_family_gets_the_permissive_generic_spec():
    """Absence of a spec means supported — new families cost no code."""
    assert resolve_pp_spec(_model().model) is PPModelSpec
    validate_model_supports_pp(_model())


def test_spec_subclass_must_justify_disabling_pp():
    """__init_subclass__ refuses a silent SUPPORTS_PP=False."""
    with pytest.raises(TypeError, match="declares no UNSUPPORTED_REASON"):

        class _Bad(PPModelSpec):
            HF_MODULE_NAMES = ("SomeModel",)
            SUPPORTS_PP = False


def test_model_without_a_layer_list_is_rejected():
    with pytest.raises(ValueError, match="decoder-layer list"):
        validate_model_supports_pp(torch.nn.Linear(4, 4))


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
        validate_model_supports_pp(model)


@pytest.mark.parametrize("field", MTP_FIELDS_IN_THE_WILD)
def test_mtp_field_declared_but_not_built_is_accepted(field):
    """The gate judges the BUILT tree, not the declaration.

    Step3p7, Qwen3.5/3.6, Inkling, DeepSeek-V4 and GLM-5 hub configs declare 1-3 MTP tail layers as
    checkpoint metadata while the load drops the weights (``_keys_to_ignore_on_load_unexpected``);
    a declaration-only gate refused every one of those checkpoints for layers that do not exist.
    """
    model = Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG))
    setattr(model.config, field, 3)
    assert len(model.model.layers) == model.config.num_hidden_layers, "premise: no tail was built"
    validate_model_supports_pp(model)


@pytest.mark.parametrize("field", MTP_FIELDS_IN_THE_WILD)
def test_mtp_field_declared_at_zero_is_accepted(field):
    """Anti-vacuity: the gate reads the COUNT, not the field's presence (Ling-mini-2.0 ships 0)."""
    model = Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG))
    setattr(model.config, field, 0)
    validate_model_supports_pp(model)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
