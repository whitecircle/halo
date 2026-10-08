#!/usr/bin/env python
"""Every whole-weight reload maps a hub-layout checkpoint onto the live model's names — across the roster.

transformers holds a MoE's experts fused in memory while the wrapper-less writers save them in the
layout the model was loaded from: one tensor per expert where the family's hub is per-expert. A reload
into an already-built model that matches checkpoint keys to live names must replay the load
conversion, or the experts keep the weights the model was built with (at scale the coverage gate
refuses the resume instead). Those reloads are:

  * a plain FSDP2 resume (``use_grouped_gemm: false``, no EP/CP/TP), whose policy is built from the
    BASE (``resolve_resume_weights_source`` keeps it) and then reads the checkpoint, and its
    best-model load;
  * a TP best-model load (a TP resume constructs from the checkpoint and skips the read);
  * a single-process / DDP resume and best-model load, which the base Trainer would read by raw key;
  * an ``init_from_scratch`` FSDP2 run, whose model is built from its config and records no load
    conversions, so both its save and its resume go through the family's registered ones — read off
    a model whose classes FSDP2 has swapped for its own ``FSDP<Name>``.

Each case saves through the FSDP2 writer from an FSDP2-wrapped model (CPU, a one-rank fake process
group) and reloads through :meth:`CheckpointLoader.load_model`, comparing every live tensor with the
trained one. The per-expert hubs are the cases that fail without the conversion; the fused hubs and the
dense models must load unchanged. The reload also reads layouts the model's own load never met, as
``from_pretrained`` does: the module layout an EP-gathered save writes, and a per-expert checkpoint of
the fused Qwen3.5/3.6 hub.

    python tests/cpu/checkpoint/test_hub_layout_resume.py
"""

# The device-aware kernel-dispatch shim must land before the modeling modules bind transformers'
# hub-kernel fallback at import, or a CPU forward of a conv / linear-attention family reaches CUDA.
import src.models.patches.kernel_dispatch  # noqa: F401  # isort: skip

import itertools
import os
import shutil
from types import SimpleNamespace

import pytest
import torch
from accelerate import PartialState
from safetensors.torch import save_file
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor
from transformers import Trainer, TrainingArguments

import src.distributed.expert_parallel.layers.roster  # noqa: F401  # the config export reads the EP roster
from scripts.after_training.unfuse_moe_experts import unfuse_checkpoint
from src.checkpoint.config_export import LOADED_WEIGHTS_FROM_ATTR
from src.distributed.checkpoint.context import CheckpointLoadContext
from src.distributed.checkpoint.loader import CheckpointLoader
from src.distributed.checkpoint.save import save_fsdp2_checkpoint
from src.distributed.fsdp import IdentityParamSet, apply_fsdp2_per_layer
from src.distributed.parallelism_config import ParallelismConfig
from src.models.structure import persistent_buffers
from src.training.environment import resolve_resume_weights_source
from tests.common.checkpoint_io import written_keys
from tests.common.distributed import fake_process_group_mesh
from tests.common.tiny_models import (
    TINY_DENSE_FAMILIES,
    TINY_MOE_FAMILIES,
    build_tiny_family_checkpoint,
    tiny_family_model,
)

PartialState()  # the save path logs through accelerate's logger

SHAPES = ("fsdp2_resume", "fsdp2_best_model", "tp_best_model", "unsharded_resume", "unsharded_best_model")
FAMILIES = {**TINY_MOE_FAMILIES, **TINY_DENSE_FAMILIES}


def _fused_hub(family, path: str) -> None:
    """The Qwen3.5/3.6 release layout: experts stored as the fused module tensors, so the load converts
    nothing — where ``save_pretrained`` of a model built from its config writes them per-expert."""
    torch.manual_seed(0)
    model = tiny_family_model(family).to(torch.bfloat16)
    model.config.save_pretrained(path)
    save_file(
        {key: tensor.contiguous() for key, tensor in model.state_dict().items()},
        os.path.join(path, "model.safetensors"),
        metadata={"format": "pt"},
    )


def _saved_hub(family, path: str) -> None:
    build_tiny_family_checkpoint(family, path, seed=0)


# Case name → (family, hub writer).
CASES = {name: (family, _saved_hub) for name, family in FAMILIES.items()}
CASES["qwen3_5_moe_text-fused_hub"] = (TINY_MOE_FAMILIES["qwen3_5_moe_text"], _fused_hub)


def _load(family, path: str):
    return family.load_class.from_pretrained(path, dtype=torch.bfloat16, trust_remote_code=family.trust_remote_code)


def _live_state(model) -> dict[str, torch.Tensor]:
    """Every parameter and persistent buffer by live name, a one-rank FSDP2 shard read as the whole tensor."""
    return {
        name: (t.to_local() if isinstance(t, DTensor) else t).detach().clone()
        for name, t in itertools.chain(model.named_parameters(), persistent_buffers(model))
    }


def _fsdp2(model, mesh):
    apply_fsdp2_per_layer(model, mesh, MixedPrecisionPolicy(), True, IdentityParamSet())
    return model


def _per_expert_hub(path: str) -> bool:
    return any(".experts.0." in key for key in written_keys(path))


def _perturbed(model, seed: int):
    """``model`` with every float parameter moved off its initial value, as training would."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in model.parameters():
            if param.is_floating_point():
                param.add_(torch.randn(param.shape, generator=generator).to(param.dtype))
    return model


def _save_fsdp2(model, mesh, checkpoint: str) -> None:
    """The FSDP2 writer's save of ``model``, wrapped as a plain FSDP2 run wraps it."""
    ctx = SimpleNamespace(
        model=_fsdp2(model, mesh), is_save_rank=True, max_shard_size="5GB", training_checkpoint=True, tokenizer=None
    )
    save_fsdp2_checkpoint(ctx, checkpoint)


@pytest.fixture(scope="module", params=sorted(CASES))
def trained(request, tmp_path_factory):
    """``(case, family, hub dir, checkpoint dir, trained live state)``: the tiny model's hub, then a
    "trained" copy of it saved by the FSDP2 writer — the save a plain FSDP2 run makes."""
    family, write_hub = CASES[request.param]
    hub = str(tmp_path_factory.mktemp(f"{request.param}_hub"))
    checkpoint = str(tmp_path_factory.mktemp(f"{request.param}_checkpoint"))
    write_hub(family, hub)
    with fake_process_group_mesh(0, 1) as mesh:
        model = _perturbed(_load(family, hub), seed=1)
        state = _live_state(model)
        _save_fsdp2(model, mesh, checkpoint)
    return request.param, family, hub, checkpoint, state


def _ctx(
    model, *, fsdp_wrapped=False, is_tp_mode=False, fallback=None, base_owns_whole_weight_load=False
) -> CheckpointLoadContext:
    return CheckpointLoadContext(
        model=model,
        optimizer=None,
        lr_scheduler=None,
        parallelism_config=None,
        is_pp_mode=False,
        is_cp_mode=False,
        is_tp_mode=is_tp_mode,
        has_ep_layers=False,
        fsdp_wrapped=fsdp_wrapped,
        tp_rank=0,
        tp_size=2 if is_tp_mode else 1,
        super_load_from_checkpoint=fallback,
        super_load_optimizer_and_scheduler=None,
        base_owns_whole_weight_load=base_owns_whole_weight_load,
    )


def _reload(shape: str, family, hub: str, checkpoint: str, tmp_path) -> dict[str, torch.Tensor]:
    """Build the policy from the hub as the shape's resume does, reload ``checkpoint`` into it, and
    return its live state."""
    best_model = shape.endswith("best_model")
    if shape.startswith("fsdp2"):
        with fake_process_group_mesh(0, 1) as mesh:
            model = _fsdp2(_load(family, hub), mesh)
            CheckpointLoader(_ctx(model, fsdp_wrapped=True)).load_model(checkpoint, model, for_best_model=best_model)
            return _live_state(model)
    model = _load(family, hub)
    if shape == "tp_best_model":
        CheckpointLoader(_ctx(model, is_tp_mode=True)).load_model(checkpoint, model, for_best_model=True)
        return _live_state(model)
    # The base Trainer's own loader is the fallback a single-process run would otherwise take.
    base_trainer = Trainer(model=model, args=TrainingArguments(output_dir=str(tmp_path), report_to=[], use_cpu=True))
    CheckpointLoader(_ctx(model, fallback=base_trainer._load_from_checkpoint)).load_model(
        checkpoint, model, for_best_model=best_model
    )
    return _live_state(model)


def test_a_plain_fsdp2_resume_builds_from_the_base(trained):
    """The premise of the FSDP2 rows: this shape's policy is built from the base, so the loader below is
    what puts the trained weights back."""
    _, _, hub, checkpoint, _ = trained
    pc = ParallelismConfig(use_grouped_gemm=False, nvlink_domain_size=1, max_concurrent_loading=0)
    assert resolve_resume_weights_source(checkpoint, SimpleNamespace(model_name_or_path=hub), pc) == hub


@pytest.mark.parametrize("shape", SHAPES)
def test_the_reload_restores_every_trained_tensor(trained, shape, tmp_path):
    name, family, hub, checkpoint, state = trained
    assert _per_expert_hub(checkpoint) == _per_expert_hub(hub), "premise: the writer saves the hub's layout"

    reloaded = _reload(shape, family, hub, checkpoint, tmp_path)

    assert reloaded.keys() == state.keys()
    stale = _stale(state, reloaded)
    assert not stale, f"{name}/{shape}: {len(stale)} tensors kept their construction values: {stale[:5]}"


def _stale(state: dict, reloaded: dict) -> list[str]:
    return sorted(key for key, tensor in state.items() if not torch.equal(reloaded[key], tensor))


def test_a_module_layout_checkpoint_passes_through(trained, tmp_path):
    """What an EP-gathered save writes for the families it keeps fused (Cohere2, DeepSeek-V4, GLM-5
    Next, Qwen3.5/3.6): the live names as they are, which a model whose load recorded the per-expert
    converters must read untouched."""
    name, family, hub, _, state = trained
    shutil.copy(os.path.join(hub, "config.json"), tmp_path)
    save_file({key: tensor.contiguous() for key, tensor in state.items()}, str(tmp_path / "model.safetensors"))

    stale = _stale(state, _reload("fsdp2_resume", family, hub, str(tmp_path), tmp_path))

    assert not stale, f"{name}: {len(stale)} tensors kept their construction values: {stale[:5]}"


def test_a_fused_hub_load_reads_a_per_expert_checkpoint(tmp_path):
    """The Qwen3.5/3.6 hub ships its experts fused, so its load records no conversion. A per-expert
    checkpoint of it (``unfuse_moe_experts.py``'s output) is still read, through the family's mapping,
    as ``from_pretrained`` reads it."""
    family = TINY_MOE_FAMILIES["qwen3_5_moe_text"]
    hub, fused, unfused = (str(tmp_path / name) for name in ("hub", "fused", "unfused"))
    _fused_hub(family, hub)
    with fake_process_group_mesh(0, 1) as mesh:
        model = _perturbed(_load(family, hub), seed=1)
        assert model._weight_conversions == [], "premise: the fused hub converts nothing"
        state = _live_state(model)
        _save_fsdp2(model, mesh, fused)
    unfuse_checkpoint(fused, unfused)
    assert _per_expert_hub(unfused), "premise: the experts are one tensor each"

    stale = _stale(state, _reload("fsdp2_resume", family, hub, unfused, tmp_path))

    assert not stale, f"{len(stale)} tensors kept their construction values: {stale[:5]}"


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_an_init_from_scratch_run_saves_the_hub_layout_and_resumes_from_it(name, tmp_path):
    family, checkpoint, reference = FAMILIES[name], str(tmp_path / "checkpoint"), str(tmp_path / "reference")
    torch.manual_seed(0)
    tiny_family_model(family).save_pretrained(reference)
    with fake_process_group_mesh(0, 1) as mesh:
        torch.manual_seed(0)
        model = _perturbed(tiny_family_model(family).to(torch.bfloat16), seed=1)
        state = _live_state(model)
        _save_fsdp2(model, mesh, checkpoint)
        assert written_keys(checkpoint) == written_keys(reference), "the save left the layout save_pretrained writes"

        torch.manual_seed(2)
        resumed = tiny_family_model(family).to(torch.bfloat16)
        # load_distributed_model's stamp for a model that read no weights.
        setattr(resumed, LOADED_WEIGHTS_FROM_ATTR, None)
        _fsdp2(resumed, mesh)
        CheckpointLoader(_ctx(resumed, fsdp_wrapped=True)).load_model(checkpoint, resumed)
        reloaded = _live_state(resumed)

    stale = _stale(state, reloaded)
    assert not stale, f"{name}: {len(stale)} tensors kept their construction values: {stale[:5]}"


def test_a_reload_the_base_loader_owns_stays_with_it(trained):
    """Accelerate-managed FSDP reads its own sharded format, and sentence-transformers rebuilds its whole
    pipeline: there the base loader is the one that reads the checkpoint, and the model is left to it."""
    _, family, hub, checkpoint, _ = trained
    model = _load(family, hub)
    before = _live_state(model)
    calls = []

    CheckpointLoader(
        _ctx(model, fallback=lambda *a, **k: calls.append(a), base_owns_whole_weight_load=True)
    ).load_model(checkpoint, model)

    assert calls == [(checkpoint, model)]
    assert all(torch.equal(tensor, before[key]) for key, tensor in _live_state(model).items())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
