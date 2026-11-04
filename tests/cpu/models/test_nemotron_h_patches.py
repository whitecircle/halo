#!/usr/bin/env python
"""Nemotron-H's Mamba scan against upstream's and against each document run alone, and the values upstream's trainer
reads differently from the release and both engines. ``in_proj`` is scaled up where a row must cross: at stock init a
crossed boundary moves the logits by rounding alone."""

import pytest
import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file
from transformers import AutoModelForImageTextToText, NemotronHConfig, NemotronHForCausalLM
from transformers.models.jamba import JambaConfig
from transformers.models.nemotron_h import modeling_nemotron_h as nemotron_modeling
from transformers.models.nemotron_h_omni.modeling_nemotron_h_omni import NemotronH_Omni_Reasoning_V3
from transformers.models.radio import modeling_radio as radio_modeling
from transformers.models.radio.configuration_radio import RadioConfig

from halo.data.collators.collator_rules import reject_multi_document_rows
from halo.data.collators.factory import select_data_collator
from halo.model_loading.checkpoint_coverage import from_pretrained_verified
from halo.model_loading.lazy_safetensors.materialization import init_checkpoint_absent_modules
from halo.model_loading.model_preparation import apply_family_patches
from halo.models import nemotron_h
from halo.models.attention.tree import resolve_tree_plan_options
from halo.models.nemotron_h import apply_nemotron_h_patches
from halo.models.segment_markers import reject_pipeline_packed_rows, segment_markers_for
from tests.common.models import TINY_NEMOTRON_H_CONFIG
from tests.common.packed_flash_varlen import padding_tokenizer
from tests.common.tiny_models import TINY_MOE_VLM_FAMILIES, tiny_family_model

_SEED = 1234
_COMPOSITE = TINY_MOE_VLM_FAMILIES["nemotron_h_omni"]
_MIXER = nemotron_modeling.NemotronHMamba2Mixer
_SCALE = radio_modeling.RadioLayerScale
# The module's attributes as a fresh process holds them; workers collect before any test runs a patch.
_STOCK_ATTRIBUTES = {
    (nemotron_modeling, "mamba2_chunk_scan"): nemotron_modeling.mamba2_chunk_scan,
    (nemotron_modeling, "mamba2_split_conv1d_scan_combined"): nemotron_modeling.mamba2_split_conv1d_scan_combined,
    (_MIXER, "__init__"): _MIXER.__init__,
    (_MIXER, "forward"): _MIXER.forward,
    (_SCALE, "__init__"): _SCALE.__init__,
}
_STOCK_SCAN = nemotron_modeling.mamba2_chunk_scan
_HEADS, _HEAD_DIM, _STATE, _GROUPS, _CHUNK = 8, 8, 8, 2, 8
_LENGTHS = (20, 7, 33, 8, 1)
# fp32 rounding of a sum over a chunk, against upstream's and against a document alone.
_EXACT = 1e-5
# The whole-row and stock-scan controls, which must move the row well past any rounding.
_CROSSED = 1e-2
# What the floor moves on a mixer whose every time step sits under it.
_FLOOR_MOVES = 1e-3


@pytest.fixture
def stock(monkeypatch):
    for (owner, name), value in _STOCK_ATTRIBUTES.items():
        monkeypatch.setattr(owner, name, value)


@pytest.fixture
def patched(stock):
    apply_nemotron_h_patches()


def _rel(value: torch.Tensor, reference: torch.Tensor) -> float:
    return ((value - reference).abs().max() / reference.abs().max().clamp_min(1e-12)).item()


def _scan_inputs(batch: int, length: int, requires_grad: bool = False) -> dict:
    generator = torch.Generator().manual_seed(_SEED + length)
    inputs = {
        "hidden_states": torch.randn(batch, length, _HEADS, _HEAD_DIM, generator=generator),
        "dt": torch.randn(batch, length, _HEADS, generator=generator) - 3.0,
        "B": torch.randn(batch, length, _GROUPS, _STATE, generator=generator),
        "C": torch.randn(batch, length, _GROUPS, _STATE, generator=generator),
    }
    return {key: value.requires_grad_(requires_grad) for key, value in inputs.items()}


def _scan_arguments(**overrides) -> dict:
    arguments = {
        "A": -0.1 * torch.arange(1, _HEADS + 1, dtype=torch.float32),
        "chunk_size": _CHUNK,
        "D": torch.linspace(0.5, 1.5, _HEADS),
        "dt_bias": torch.linspace(-1.0, 1.0, _HEADS),
        "initial_states": None,
        "dt_softplus": True,
        "dt_limit": (0.0, float("inf")),
        "return_final_states": False,
    }
    return {**arguments, **overrides}


def _alone(scan, inputs: dict, spans: list[tuple[int, int, int]]) -> list[torch.Tensor]:
    """``scan`` per ``(row, start, stop)``, each span its own one-row input."""
    return [
        scan(**{key: value[row : row + 1, start:stop] for key, value in inputs.items()}, **_scan_arguments())
        for row, start, stop in spans
    ]


def _spans_of(lengths: tuple[int, ...], row: int = 0) -> list[tuple[int, int, int]]:
    edges = torch.tensor([0, *lengths]).cumsum(0).tolist()
    return [(row, start, stop) for start, stop in zip(edges[:-1], edges[1:], strict=True)]


def _seq_idx(lengths: tuple[int, ...]) -> torch.Tensor:
    return torch.cat([torch.full((length,), index) for index, length in enumerate(lengths)]).int().unsqueeze(0)


def _crossing_model(attn_implementation: str = "eager") -> NemotronHForCausalLM:
    torch.manual_seed(_SEED)
    model = NemotronHForCausalLM(NemotronHConfig(**TINY_NEMOTRON_H_CONFIG)).eval()
    model.config._attn_implementation = attn_implementation
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, _MIXER):
                module.in_proj.weight.mul_(50.0)
    return model


def _crossing_composite(attn_implementation: str = "eager"):
    torch.manual_seed(_SEED)
    model = tiny_family_model(_COMPOSITE).eval()
    model.config._attn_implementation = attn_implementation
    model.config.text_config._attn_implementation = attn_implementation
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, _MIXER):
                module.in_proj.weight.mul_(50.0)
    return model


def _row_ids(length: int) -> torch.Tensor:
    return torch.randint(4, 128, (1, length), generator=torch.Generator().manual_seed(_SEED))


def _packed_row(lengths: tuple[int, ...]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": _row_ids(sum(lengths)),
        "position_ids": torch.cat([torch.arange(length) for length in lengths]).unsqueeze(0),
        "seq_idx": _seq_idx(lengths),
    }


def _row_vs_alone(model, row: dict, lengths: tuple[int, ...], **kwargs) -> float:
    """The worst relative logit gap between the row's documents and each document run alone."""
    with torch.no_grad():
        packed = model(**row, use_cache=False, **kwargs).logits
        gaps = []
        for _, start, stop in _spans_of(lengths):
            alone = model(input_ids=row["input_ids"][:, start:stop], use_cache=False, **kwargs).logits
            gaps.append(_rel(packed[:, start:stop], alone))
    return max(gaps)


@pytest.mark.parametrize(
    ("batch", "length"),
    [(1, 1), (1, 7), (1, 8), (1, 9), (2, 24), (3, 37)],
    ids=["one_token", "under_a_chunk", "one_chunk", "over_a_chunk", "two_rows_whole_chunks", "three_rows_padded"],
)
def test_the_grouped_body_equals_upstreams(batch, length):
    """Output and every input's gradient, at each way a row meets the chunk grid."""
    inputs = _scan_inputs(batch, length, requires_grad=True)
    upstream = _STOCK_SCAN(**inputs, **_scan_arguments())
    upstream_grads = torch.autograd.grad(upstream.pow(2).sum(), list(inputs.values()))
    ours = nemotron_h._document_chunk_scan(**inputs, **_scan_arguments())
    grads = torch.autograd.grad(ours.pow(2).sum(), list(inputs.values()))
    assert ours.shape == upstream.shape
    assert _rel(ours, upstream) < _EXACT
    for grad, upstream_grad in zip(grads, upstream_grads, strict=True):
        assert _rel(grad, upstream_grad) < _EXACT


def test_the_grouped_body_carries_a_cached_state_as_upstreams():
    inputs = _scan_inputs(1, 24)
    upstream, upstream_final = _STOCK_SCAN(**inputs, **_scan_arguments(return_final_states=True))
    ours, final = nemotron_h._document_chunk_scan(**inputs, **_scan_arguments(return_final_states=True))
    assert _rel(ours, upstream) < _EXACT and _rel(final, upstream_final) < _EXACT
    start = torch.randn(1, _HEADS, _HEAD_DIM, _STATE, generator=torch.Generator().manual_seed(_SEED))
    upstream = _STOCK_SCAN(**inputs, **_scan_arguments(initial_states=start))
    ours = nemotron_h._document_chunk_scan(**inputs, **_scan_arguments(initial_states=start))
    assert _rel(ours, upstream) < _EXACT


def test_a_document_equals_itself_alone_and_the_whole_row_does_not():
    lengths = _LENGTHS
    inputs = _scan_inputs(1, sum(lengths), requires_grad=True)
    spans = _spans_of(lengths)
    alone = torch.cat(_alone(_STOCK_SCAN, inputs, spans), dim=1)
    documents = nemotron_h._document_chunk_scan(**inputs, **_scan_arguments(), seq_idx=_seq_idx(lengths))
    assert _rel(documents, alone) < _EXACT
    whole = _STOCK_SCAN(**inputs, **_scan_arguments())
    assert _rel(whole, alone) > _CROSSED

    tensors = list(inputs.values())
    alone_grads = torch.autograd.grad(alone.pow(2).sum(), tensors, retain_graph=True)
    for grad, reference in zip(torch.autograd.grad(documents.pow(2).sum(), tensors), alone_grads, strict=True):
        assert _rel(grad, reference) < _EXACT


def test_documents_of_several_rows_stay_on_their_rows():
    rows = ((9, 15), (24,), (3, 3, 18))
    seq_idx = torch.cat([_seq_idx(row) for row in rows])
    inputs = _scan_inputs(len(rows), 24)
    documents = nemotron_h._document_chunk_scan(**inputs, **_scan_arguments(), seq_idx=seq_idx)
    for row, lengths in enumerate(rows):
        spans = _spans_of(lengths, row)
        alone = torch.cat(_alone(_STOCK_SCAN, inputs, spans), dim=1)
        assert _rel(documents[row : row + 1], alone) < _EXACT


def test_a_row_of_one_document_takes_no_layout(monkeypatch):
    """The markers a one-document row carries change nothing, and cost it no split and stack."""
    monkeypatch.setattr(nemotron_h, "_scan_documents", lambda *_a, **_k: pytest.fail("re-laid one document"))
    inputs = _scan_inputs(2, 24)
    seq_idx = torch.zeros(2, 24, dtype=torch.int32)
    out = nemotron_h._document_chunk_scan(**inputs, **_scan_arguments(), seq_idx=seq_idx)
    assert _rel(out, _STOCK_SCAN(**inputs, **_scan_arguments())) < _EXACT


@pytest.mark.parametrize(
    ("lengths", "batches"),
    [
        ([128] * 64, 1),
        ([512] * 16, 1),
        ([2048] * 4, 1),
        ([4096, 2048, 1024, 512, 256, 128, 64, 64], 7),
        ([1], 1),
    ],
    ids=["many_short", "sixteen_even", "four_long", "skewed", "one"],
)
def test_documents_share_a_scan_only_within_the_padded_area(lengths, batches):
    planned = nemotron_h._padded_batches(lengths)
    assert len(planned) == batches
    assert sorted(index for batch in planned for index in batch) == list(range(len(lengths)))
    for batch in planned:
        members = [lengths[index] for index in batch]
        assert members[0] == max(members), "the padded width is its first member's"
        assert members[0] * len(members) <= nemotron_h._MAX_PADDED_AREA * sum(members)


def test_every_input_reaches_its_documents_through_one_split():
    """One slice per document would zero-fill the whole row per document in the backward, quadratic on a short-document
    row; a split's backward joins them at once."""
    lengths = (4,) * 16
    inputs = _scan_inputs(1, sum(lengths), requires_grad=True)
    out = nemotron_h._document_chunk_scan(**inputs, **_scan_arguments(), seq_idx=_seq_idx(lengths))
    seen, stack, names = set(), [out.grad_fn], []
    while stack:
        node = stack.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        names.append(type(node).__name__)
        stack.extend(next_node for next_node, _ in node.next_functions)
    assert names.count("SplitWithSizesBackward0") == len(inputs)


def test_a_cached_row_of_documents_is_refused():
    inputs = _scan_inputs(1, sum(_LENGTHS))
    with pytest.raises(ValueError, match="no one state"):
        nemotron_h._document_chunk_scan(
            **inputs, **_scan_arguments(return_final_states=True), seq_idx=_seq_idx(_LENGTHS)
        )


def test_every_mamba_layer_reads_one_host_copy_of_the_boundaries(patched, monkeypatch):
    calls = []
    derive = nemotron_h.row_documents
    monkeypatch.setattr(nemotron_h, "row_documents", lambda *args: calls.append(args) or derive(*args))
    row = _packed_row(_LENGTHS)
    with torch.no_grad():
        _crossing_model()(**row, use_cache=False)
    assert len(calls) == 1, "the tiny model holds two Mamba layers"


@pytest.mark.parametrize("attn_implementation", ["eager", "sdpa"])
@pytest.mark.parametrize("build", [_crossing_model, _crossing_composite], ids=["text_class", "composite"])
def test_a_packed_row_equals_its_documents_alone_through_the_whole_model(patched, attn_implementation, build):
    model = build(attn_implementation)
    assert _row_vs_alone(model, _packed_row(_LENGTHS), _LENGTHS) < _EXACT * 10


def test_without_its_markers_the_row_crosses(patched):
    row = _packed_row(_LENGTHS)
    del row["seq_idx"]
    assert _row_vs_alone(_crossing_model(), row, _LENGTHS) > _CROSSED


def test_upstreams_scan_crosses_under_the_same_markers(stock):
    """The conv keeps the row apart and upstream's scan swallows ``seq_idx``, so only the patch isolates it."""
    assert _row_vs_alone(_crossing_model(), _packed_row(_LENGTHS), _LENGTHS) > _CROSSED


def test_a_packed_row_under_reentrant_checkpointing_matches_the_plain_row(patched):
    """The recompute re-reads the memo and the conv, with gradients into every Mamba tensor."""
    row = _packed_row(_LENGTHS)
    row["labels"] = row["input_ids"].clone()
    gradients = []
    for checkpointing in (False, True):
        model = _crossing_model().train()
        if checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
        model(**row, use_cache=False).loss.backward()
        gradients.append(
            {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}
        )
    assert any("conv1d" in name for name in gradients[0]) and any("dt_bias" in name for name in gradients[0])
    assert gradients[0].keys() == gradients[1].keys()
    for name, plain in gradients[0].items():
        assert _rel(gradients[1][name], plain) < _EXACT * 10, name


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_padding_leaves_a_shorter_row_as_it_runs_alone(patched, padding_side):
    """The release's tokenizer pads left, which moves the real tokens off the chunk grid."""
    model = _crossing_model()
    long_ids, short_ids = _row_ids(23), _row_ids(9)
    pad = torch.zeros(1, long_ids.shape[1] - short_ids.shape[1], dtype=short_ids.dtype)
    padded = torch.cat([pad, short_ids] if padding_side == "left" else [short_ids, pad], dim=1)
    mask = torch.cat(
        [pad, torch.ones_like(short_ids)] if padding_side == "left" else [torch.ones_like(short_ids), pad], dim=1
    )
    with torch.no_grad():
        logits = model(
            input_ids=torch.cat([long_ids, padded]),
            attention_mask=torch.cat([torch.ones_like(long_ids), mask]),
            use_cache=False,
        ).logits
        alone = model(input_ids=short_ids, use_cache=False).logits
    real = logits[1:][mask.bool()].unsqueeze(0)
    assert _rel(real, alone) < _EXACT * 10


def test_autocast_leaves_the_scan_as_it_runs_outside(patched):
    inputs = _scan_inputs(1, 24)
    plain = nemotron_h._document_chunk_scan(**inputs, **_scan_arguments())
    with torch.autocast("cpu", dtype=torch.bfloat16):
        cast = nemotron_h._document_chunk_scan(**inputs, **_scan_arguments())
    assert torch.equal(cast, plain) and cast.dtype == torch.float32


def test_upstreams_body_moves_under_the_same_autocast():
    inputs = _scan_inputs(1, 24)
    plain = _STOCK_SCAN(**inputs, **_scan_arguments())
    with torch.autocast("cpu", dtype=torch.bfloat16):
        cast = _STOCK_SCAN(**inputs, **_scan_arguments())
    assert _rel(cast.float(), plain) > 1e-3


def _mixer_output(limit: tuple[float, float] | None = None) -> torch.Tensor:
    torch.manual_seed(_SEED)
    model = NemotronHForCausalLM(NemotronHConfig(**TINY_NEMOTRON_H_CONFIG)).eval()
    mixer = model.model.layers[0].mixer
    with torch.no_grad():
        # Every time step under the floor upstream clamps at.
        mixer.dt_bias.fill_(-10.0)
        mixer.in_proj.weight.mul_(50.0)
    assert torch.nn.functional.softplus(mixer.dt_bias).max() < model.config.time_step_min
    if limit is not None:
        mixer.time_step_limit = limit
    with torch.no_grad():
        return mixer(torch.randn(1, 24, model.config.hidden_size, generator=torch.Generator().manual_seed(_SEED)))


def test_the_mixer_honors_the_configs_limit(patched):
    assert _rel(_mixer_output(), _mixer_output(limit=(0.0, float("inf")))) == 0.0
    floor = NemotronHConfig(**TINY_NEMOTRON_H_CONFIG).time_step_min
    assert _rel(_mixer_output(), _mixer_output(limit=(floor, float("inf")))) > _FLOOR_MOVES
    config = NemotronHConfig(**{**TINY_NEMOTRON_H_CONFIG, "time_step_limit": [0.25, 4.0]})
    assert NemotronHForCausalLM(config).model.layers[0].mixer.time_step_limit == (0.25, 4.0)


def test_upstreams_mixer_clamps_at_the_floor_no_engine_reads(stock):
    model = NemotronHForCausalLM(NemotronHConfig(**TINY_NEMOTRON_H_CONFIG))
    assert model.model.layers[0].mixer.time_step_limit == (model.config.time_step_min, float("inf"))
    assert _rel(_mixer_output(), _mixer_output(limit=(0.0, float("inf")))) > _FLOOR_MOVES


def _conv_dim(config) -> int:
    return config.mamba_num_heads * config.mamba_head_dim + 2 * config.n_groups * config.ssm_state_size


def _conv_adapted():
    model = _crossing_model().train()
    adapted = get_peft_model(model, LoraConfig(r=_conv_dim(model.config), target_modules=["conv1d"]))
    return adapted, _row_ids(24)


def test_an_adapter_on_the_conv_is_refused(patched):
    adapted, ids = _conv_adapted()
    with pytest.raises(TypeError, match="conv1d out of lora_target_modules"):
        adapted(input_ids=ids, use_cache=False)


def test_upstream_lets_an_adapter_on_the_conv_do_nothing(stock):
    """Why it is refused: upstream reads the base weight, so the adapter's delta never reaches the row."""
    adapted, ids = _conv_adapted()
    with torch.no_grad():
        before = adapted(input_ids=ids, use_cache=False).logits
        for name, param in adapted.named_parameters():
            if "lora_" in name:
                param.fill_(1.0)
        after = adapted(input_ids=ids, use_cache=False).logits
    assert torch.equal(before, after)


@pytest.mark.parametrize("target", ["in_proj", "out_proj"])
def test_an_adapter_on_the_projections_reaches_the_row(patched, target):
    model = _crossing_model().train()
    adapted = get_peft_model(model, LoraConfig(r=2, target_modules=[target]))
    ids = _row_ids(24)
    with torch.no_grad():
        before = adapted(input_ids=ids, use_cache=False).logits
        for name, param in adapted.named_parameters():
            if f"{target}.lora_" in name:
                param.fill_(0.5)
        assert _rel(adapted(input_ids=ids, use_cache=False).logits, before) > _CROSSED


def test_a_fused_kernel_installed_later_cannot_take_the_row(stock, monkeypatch):
    """Upstream's mixer returns whatever the fused kernel gives, past the projections' adapters."""
    ids = _row_ids(24)
    sentinel = torch.full((1, 24, TINY_NEMOTRON_H_CONFIG["hidden_size"]), 7.0)
    monkeypatch.setattr(nemotron_modeling, "mamba2_split_conv1d_scan_combined", lambda *_a, **_k: sentinel)
    mixer = _crossing_model().train().model.layers[0].mixer
    hidden = torch.randn(1, 24, TINY_NEMOTRON_H_CONFIG["hidden_size"])
    assert mixer(hidden) is sentinel
    apply_nemotron_h_patches()
    assert (
        mixer(hidden) is not sentinel
        and _crossing_model().train()(input_ids=ids, use_cache=False).logits.isfinite().all()
    )


def _layer_scales(model) -> dict[str, torch.Tensor]:
    scales = {name: tensor for name, tensor in model.state_dict(keep_vars=True).items() if name.endswith("lambda1")}
    assert scales, "the tiny tower holds no layer scale, so nothing is under test"
    return scales


def _assert_no_scale_trains(model) -> None:
    for name, scale in _layer_scales(model).items():
        assert not scale.requires_grad and name not in dict(model.named_parameters()), name


def test_radio_layer_scale_trains_nothing_and_stays_in_the_checkpoint(patched):
    model = tiny_family_model(_COMPOSITE)
    _assert_no_scale_trains(model)
    value = model.config.vision_config.layerscale_value
    assert all(torch.all(scale == value) for scale in _layer_scales(model).values())


def test_upstreams_layer_scale_trains(stock):
    assert radio_modeling.RadioLayerScale(RadioConfig(hidden_size=8)).lambda1.requires_grad


def test_an_eager_load_of_a_checkpoint_storing_it_keeps_it_out_of_training(patched, tmp_path):
    """A load replaces a stored parameter with a trainable one whatever the module held, so a flag would not survive
    the toolkit's own export."""
    torch.manual_seed(_SEED)
    tiny_family_model(_COMPOSITE).save_pretrained(tmp_path)
    assert any(name.endswith("lambda1") for name in load_file(tmp_path / "model.safetensors"))
    _assert_no_scale_trains(AutoModelForImageTextToText.from_pretrained(tmp_path, dtype=torch.float32))


def test_an_eager_load_of_the_release_draws_it_at_its_value(patched, tmp_path):
    torch.manual_seed(_SEED)
    tiny_family_model(_COMPOSITE).save_pretrained(tmp_path)
    stored = load_file(tmp_path / "model.safetensors")
    save_file(
        {name: tensor for name, tensor in stored.items() if not name.endswith("lambda1")},
        tmp_path / "model.safetensors",
        metadata={"format": "pt"},
    )
    model = from_pretrained_verified(AutoModelForImageTextToText, str(tmp_path), dtype=torch.float32)
    _assert_no_scale_trains(model)
    assert all(torch.all(scale == 1.0) for scale in _layer_scales(model).values())


def test_the_lazy_draw_materializes_it_at_its_value(patched):
    with torch.device("meta"):
        model = tiny_family_model(_COMPOSITE)
    state = model.state_dict()
    absent = {key for key in state if key.endswith("lambda1")}
    init_checkpoint_absent_modules(model, set(state) - absent, "cpu", "unit", dtype=torch.bfloat16)
    _assert_no_scale_trains(model)
    assert all(not scale.is_meta and torch.all(scale.float() == 1.0) for scale in _layer_scales(model).values())


def test_a_training_step_leaves_the_layer_scale_alone(patched):
    """Through the tower, which a parameter frozen in name only would still accumulate into."""
    torch.manual_seed(_SEED)
    model = tiny_family_model(_COMPOSITE).train()
    optimizer = torch.optim.SGD([param for param in model.parameters() if param.requires_grad], lr=0.1)
    before = {name: scale.clone() for name, scale in _layer_scales(model).items()}
    height, width = 4, 4
    patches = height * width
    image_token = model.config.image_token_id
    ids = torch.cat([torch.full((1, patches // 4), image_token), torch.arange(4, 12).unsqueeze(0)], dim=1)
    pixels = torch.randn(patches, model.config.vision_config.patch_dim)
    grid = torch.tensor([[height, width]])
    loss = model(input_ids=ids, pixel_values=pixels, image_grid_hw=grid, labels=ids, use_cache=False).loss
    loss.backward()
    assert any(param.grad is not None and "vision_model" in name for name, param in model.named_parameters())
    optimizer.step()
    for name, scale in _layer_scales(model).items():
        assert torch.equal(scale, before[name]), name


@pytest.mark.parametrize(
    "owner_name",
    [
        (nemotron_modeling, "mamba2_chunk_scan"),
        (_MIXER, "__init__"),
        (_MIXER, "forward"),
        (_SCALE, "__init__"),
    ],
    ids=["chunk_scan", "mixer_init", "mixer_forward", "layer_scale_init"],
)
def test_a_moved_upstream_signature_refuses_every_patch(stock, monkeypatch, owner_name):
    owner, name = owner_name

    def moved(moved_argument):
        return moved_argument

    monkeypatch.setattr(owner, name, moved)
    with pytest.raises(RuntimeError, match="re-derive"):
        apply_nemotron_h_patches()
    fused = (nemotron_modeling, "mamba2_split_conv1d_scan_combined")
    assert getattr(*fused) is _STOCK_ATTRIBUTES[fused], "a refused patch must leave nothing half-applied"


def test_the_patches_apply_once(stock):
    apply_nemotron_h_patches()
    forward, init, scale_init = _MIXER.forward, _MIXER.__init__, _SCALE.__init__
    apply_nemotron_h_patches()
    assert (_MIXER.forward, _MIXER.__init__, _SCALE.__init__) == (forward, init, scale_init)


@pytest.mark.parametrize(
    "config",
    [NemotronHConfig(**TINY_NEMOTRON_H_CONFIG), tiny_family_model(_COMPOSITE).config],
    ids=["text_config", "composite_config"],
)
def test_the_family_branch_applies_them(stock, config):
    apply_family_patches(config, "sdpa")
    assert nemotron_modeling.mamba2_chunk_scan is nemotron_h._document_chunk_scan


def test_another_family_leaves_them_alone(stock):
    apply_family_patches(JambaConfig(), "sdpa")
    assert nemotron_modeling.mamba2_chunk_scan is _STOCK_SCAN


def test_the_packing_collator_marks_the_row_and_the_model_keeps_it_apart(patched):
    model = _crossing_model()
    collator = select_data_collator(
        padding_tokenizer(), train_on_completions_only=False, packing=True, model_config=model.config
    )
    ids = _row_ids(sum(_LENGTHS))[0].tolist()
    batch = collator.torch_call([{"input_ids": ids, "attention_mask": [1] * len(ids), "seq_lengths": list(_LENGTHS)}])
    assert batch["seq_idx"].tolist() == _seq_idx(_LENGTHS).tolist()
    row = {key: batch[key] for key in ("input_ids", "position_ids", "seq_idx")}
    assert _row_vs_alone(model, row, _LENGTHS) < _EXACT * 10


def test_a_padding_free_row_takes_the_markers_on_flash(patched):
    config = NemotronHConfig(**TINY_NEMOTRON_H_CONFIG)
    config._attn_implementation = "flash_attention_2"
    collator = select_data_collator(
        padding_tokenizer(), train_on_completions_only=False, padding_free=True, model_config=config
    )
    documents = [[5, 6, 7], [8, 9], [10, 11, 12, 13]]
    batch = collator([{"input_ids": document} for document in documents])
    assert batch["seq_idx"].tolist() == [[0, 0, 0, 1, 1, 2, 2, 2, 2]]
    assert batch["cu_seq_lens_q"].tolist() == [0, 3, 5, 9]


def test_multi_document_rows_pass_the_gates_with_tree_packing_still_refused():
    """The predicate reads any marker now, so what a marker does not open must stay shut: other Mamba families, tree
    packing."""
    config = NemotronHConfig(**TINY_NEMOTRON_H_CONFIG)
    for mode in ("packing", "padding_free"):
        reject_multi_document_rows(
            config,
            mode,
            attn_impl="flash_attention_2",
            allow_cross_document_linear_attention=False,
            keeps_packed_rows=mode == "packing",
            varlen_rows=True,
        )
    reject_pipeline_packed_rows(config)
    with pytest.raises(ValueError, match="tree packing is not available"):
        resolve_tree_plan_options(config)

    jamba = JambaConfig()
    assert segment_markers_for(jamba).seq_idx is False
    with pytest.raises(ValueError, match="linear-attention layers"):
        reject_pipeline_packed_rows(jamba)


@pytest.fixture
def stock_head_accessors(monkeypatch):
    """Upstream's, which read ``lm_head`` off the composite."""
    monkeypatch.delattr(NemotronH_Omni_Reasoning_V3, "get_output_embeddings")
    monkeypatch.delattr(NemotronH_Omni_Reasoning_V3, "set_output_embeddings")


def _tiny_composite():
    torch.manual_seed(_SEED)
    return tiny_family_model(_COMPOSITE)


def test_the_composite_answers_with_its_towers_head_under_no_family_patch(stock):
    model = _tiny_composite()
    assert model.get_output_embeddings() is model.language_model.lm_head


def test_upstreams_accessor_reports_no_head(stock_head_accessors):
    assert _tiny_composite().get_output_embeddings() is None


def test_a_resize_grows_the_composites_head_with_its_embedding():
    model = _tiny_composite()
    vocab = model.config.text_config.vocab_size
    head_rows = model.get_output_embeddings().weight.detach().clone()
    model.resize_token_embeddings(vocab + 8)
    assert model.config.text_config.vocab_size == vocab + 8
    assert model.get_input_embeddings().weight.shape[0] == vocab + 8
    assert torch.equal(model.get_output_embeddings().weight[:vocab], head_rows), "the kept rows moved"
    logits = model(input_ids=torch.arange(vocab, vocab + 8).unsqueeze(0), use_cache=False).logits
    assert logits.shape[-1] == vocab + 8


def test_upstreams_accessors_leave_the_head_behind_a_resize(stock_head_accessors):
    model = _tiny_composite()
    vocab = model.config.text_config.vocab_size
    model.resize_token_embeddings(vocab + 8)
    assert model.get_input_embeddings().weight.shape[0] == vocab + 8
    assert model.language_model.lm_head.out_features == vocab


def test_an_all_linear_adapter_leaves_the_composites_head_unadapted():
    """PEFT asks the accessor which linear is the head, and the chunked path refuses an adapted one."""
    peft = get_peft_model(_tiny_composite(), LoraConfig(r=2, target_modules="all-linear"))
    assert not hasattr(peft.base_model.model.language_model.lm_head, "base_layer")
    assert any("lora_A" in name for name, _ in peft.named_parameters())


def test_upstreams_accessor_would_adapt_the_head(stock_head_accessors):
    peft = get_peft_model(_tiny_composite(), LoraConfig(r=2, target_modules="all-linear"))
    assert hasattr(peft.base_model.model.language_model.lm_head, "base_layer")


def test_the_accessors_tie_nothing_and_change_no_saved_key(tmp_path, monkeypatch):
    model = _tiny_composite()
    model.tie_weights()
    assert model.get_output_embeddings().weight is not model.get_input_embeddings().weight
    model.save_pretrained(tmp_path / "patched")
    monkeypatch.delattr(NemotronH_Omni_Reasoning_V3, "get_output_embeddings")
    monkeypatch.delattr(NemotronH_Omni_Reasoning_V3, "set_output_embeddings")
    model.save_pretrained(tmp_path / "stock")
    patched = load_file(tmp_path / "patched" / "model.safetensors")
    stock_keys = load_file(tmp_path / "stock" / "model.safetensors")
    assert patched.keys() == stock_keys.keys()
    assert "language_model.lm_head.weight" in patched


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
