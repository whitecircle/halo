"""NemotronH family contract: registrations, the per-expert export, router balancing, the Mamba-2
kernel routing, and packed-row isolation through the Mamba-2 mixer.

Numeric parity of the EP forward with ``NemotronHMoE`` (latent and plain) is pinned with the rest of
the roster in ``tests/cpu/parallelism/test_ep_vs_reference_equivalence_cpu.py``.
"""

# The device-aware kernel-dispatch shim installs on `src` import and must land BEFORE the nemotron_h
# modeling module below, which binds transformers' hub-kernel fallback factory at import.
import src.models.patches.kernel_dispatch  # noqa: F401  # isort: skip

import copy
import importlib.util
import inspect

import pytest
import torch
import transformers.models.bamba.modeling_bamba as bamba_modeling
import transformers.models.nemotron_h.modeling_nemotron_h as nemotron_h_modeling
from safetensors.torch import save_file
from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHForCausalLM

from src.distributed.expert_parallel.expert_weights import ep_layer_class_by_model_type
from src.distributed.expert_parallel.layers.nemotron_h import EPNemotronHMoELayer
from src.distributed.expert_parallel.patching import MOE_LAYER_MAP
from src.models.moe_balancing import accepts_native_balancing_bias, resolve_balancing_mode
from src.models.patches.nemotron_h import patch_nemotron_h_mixer_segment_markers
from src.models.segment_markers import segment_marker_kwargs, segment_markers_for
from tests.common.models import TINY_NEMOTRON_H_CONFIG
from tests.common.parallelism import single_process_ep_config

E = TINY_NEMOTRON_H_CONFIG["n_routed_experts"]
VOCAB = TINY_NEMOTRON_H_CONFIG["vocab_size"]
NOISE_BOUND = 1e-5  # packed-isolation drift ceiling; a crossing state reads ~1e-2 and up
LEAK_FLOOR = 1e-3


def _tiny_model(**overrides) -> NemotronHForCausalLM:
    torch.manual_seed(0)
    return NemotronHForCausalLM(NemotronHConfig(**{**TINY_NEMOTRON_H_CONFIG, **overrides})).eval()


def _moe_blocks(model: NemotronHForCausalLM) -> list[tuple[int, torch.nn.Module]]:
    return [(index, block) for index, block in enumerate(model.model.layers) if block.block_type == "moe"]


def _wrap(block) -> EPNemotronHMoELayer:
    return EPNemotronHMoELayer(copy.deepcopy(block.mixer), single_process_ep_config(E)).cpu()


def test_registrations():
    assert MOE_LAYER_MAP["NemotronHMoE"] is EPNemotronHMoELayer
    assert ep_layer_class_by_model_type()["nemotron_h"] is EPNemotronHMoELayer
    assert EPNemotronHMoELayer._supports_bias_balancing
    assert not EPNemotronHMoELayer._supports_weight_sync
    assert not EPNemotronHMoELayer._supports_lazy_loading
    # The per-expert GLU-triple vocabulary has no non-gated form; the family's own gather writes it.
    assert EPNemotronHMoELayer.hub_per_expert_keys() is None


def test_gathered_export_reloads_through_from_pretrained(tmp_path):
    """The gathered experts, written under the hub's ``backbone.`` spelling, are exactly what
    ``from_pretrained`` merges back: the reload's logits equal the source model's."""
    model = _tiny_model()
    input_ids = torch.randint(0, VOCAB, (2, 12), generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        expected = model(input_ids=input_ids, use_cache=False).logits

    state = {
        f"backbone.{name.removeprefix('model.')}" if name.startswith("model.") else name: tensor.contiguous()
        for name, tensor in model.state_dict().items()
        if ".mixer.experts." not in name
    }
    for index, block in _moe_blocks(model):
        gathered = _wrap(block).gather_expert_state_dict()
        assert set(gathered) == {f"experts.{e}.{proj}.weight" for e in range(E) for proj in ("up_proj", "down_proj")}
        state.update({f"backbone.layers.{index}.mixer.{key}": tensor for key, tensor in gathered.items()})
    model.config.save_pretrained(tmp_path)
    save_file(state, str(tmp_path / "model.safetensors"))

    reloaded = NemotronHForCausalLM.from_pretrained(tmp_path, dtype=torch.float32).eval()
    with torch.no_grad():
        actual = reloaded(input_ids=input_ids, use_cache=False).logits
    assert torch.equal(actual, expected)


def test_sharded_merge_is_the_gathers_inverse():
    _index, block = _moe_blocks(_tiny_model())[0]
    layer = _wrap(block)
    gathered = layer.gather_expert_state_dict()
    merged = EPNemotronHMoELayer.merge_shards_to_hf(
        "prefix", {"up_proj": layer.up_proj.data, "down_proj": layer.down_proj.data}
    )
    assert set(merged) == {f"prefix.{key}" for key in gathered}
    for key, tensor in gathered.items():
        assert torch.equal(merged[f"prefix.{key}"], tensor), key

    with pytest.raises(ValueError, match="gate_up_proj"):
        EPNemotronHMoELayer.merge_shards_to_hf(
            "prefix", {"up_proj": layer.up_proj.data, "gate_up_proj": layer.up_proj.data}
        )


def test_balancing_auto_adopts_the_native_correction_bias():
    model = _tiny_model()
    for _index, block in _moe_blocks(model):
        block.mixer = _wrap(block)
    assert resolve_balancing_mode("auto", model, is_moe=True) == "bias_update"
    assert accepts_native_balancing_bias(model)
    assert all(block.mixer.can_adopt_native_balancing() for _index, block in _moe_blocks(model))


def _hub_route(function):
    """The Hub kernel a decorated modeling function sends its CUDA calls to; ``None`` = its own capture."""
    return inspect.getclosurevars(function).nonlocals["hub"]


@pytest.mark.skipif(
    importlib.util.find_spec("mamba_ssm") is not None, reason="transformers captures an installed mamba_ssm itself"
)
def test_only_the_nemotron_h_scan_kernels_route_to_the_hub_build():
    assert _hub_route(nemotron_h_modeling.mamba2_chunk_scan) is not None
    assert _hub_route(nemotron_h_modeling.mamba2_selective_state_update) is not None
    # The Hub build's fused op needs the causal_conv1d_cuda symbol the installed causal-conv1d dropped.
    assert _hub_route(nemotron_h_modeling.mamba2_split_conv1d_scan_combined) is None
    assert _hub_route(bamba_modeling.mamba2_chunk_scan) is None


def test_cpu_chunk_scan_fixup_matches_scanning_each_document_alone():
    """The CPU route of the packed scan restarts the state at each document, as the kernel does."""
    generator = torch.Generator().manual_seed(5)
    lens, heads, head_dim, groups, state = (5, 11), 4, 8, 2, 16
    length = sum(lens)

    def randn(*shape):
        return torch.randn(*shape, generator=generator)

    x = randn(1, length, heads, head_dim)
    dt = randn(1, length, heads)
    B = randn(1, length, groups, state)
    C = randn(1, length, groups, state)
    A = -torch.rand(heads, generator=generator)
    kwargs = {"D": randn(heads), "dt_bias": randn(heads), "dt_softplus": True, "chunk_size": 8}
    seq_idx = torch.cat([torch.full((n,), i, dtype=torch.int32) for i, n in enumerate(lens)]).unsqueeze(0)
    scan = nemotron_h_modeling.mamba2_chunk_scan

    packed = scan(x, dt, A, B, C, seq_idx=seq_idx, **kwargs)
    documents = (slice(0, lens[0]), slice(lens[0], length))
    alone = torch.cat([scan(x[:, d], dt[:, d], A, B[:, d], C[:, d], **kwargs) for d in documents], dim=1)
    assert torch.equal(packed, alone)
    crossed = scan(x, dt, A, B, C, **kwargs)
    assert (crossed[:, lens[0] :] - alone[:, lens[0] :]).abs().max().item() > LEAK_FLOOR, "the state never crossed"


def _doc_b_drift(model, *, with_seq_idx: bool, perturb_doc_b: bool = False) -> float:
    """Doc B's max logit drift when doc A changes (or doc B, the positive control) in a packed row."""
    lens = (7, 9)
    input_ids = torch.randint(4, VOCAB, (1, sum(lens)), generator=torch.Generator().manual_seed(7))
    position_ids = torch.cat([torch.arange(n) for n in lens]).unsqueeze(0)
    variant = input_ids.clone()
    changed = slice(lens[0], None) if perturb_doc_b else slice(None, lens[0])
    variant[0, changed] = (variant[0, changed] + 17) % VOCAB

    markers = segment_marker_kwargs(position_ids, segment_markers_for(model.config)) if with_seq_idx else {}
    with torch.no_grad():
        base = model(input_ids=input_ids, position_ids=position_ids, use_cache=False, **markers).logits
        swapped = model(input_ids=variant, position_ids=position_ids, use_cache=False, **markers).logits
    return (base[0, lens[0] :] - swapped[0, lens[0] :]).abs().max().item()


def test_packed_documents_stay_isolated_through_the_mamba_mixer():
    """Mamba + MoE layers only: the attention-free stack isolates a packed row through ``seq_idx``
    alone, so any drift is conv or SSM state crossing the boundary."""
    patch_nemotron_h_mixer_segment_markers()
    model = _tiny_model(layers_block_type=["mamba", "moe", "mamba", "moe"])
    assert segment_markers_for(model.config).seq_idx

    assert _doc_b_drift(model, with_seq_idx=True) < NOISE_BOUND
    assert _doc_b_drift(model, with_seq_idx=False) > LEAK_FLOOR, "the leak this test guards never occurs"
    assert _doc_b_drift(model, with_seq_idx=True, perturb_doc_b=True) > LEAK_FLOOR, "doc B is not read at all"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
