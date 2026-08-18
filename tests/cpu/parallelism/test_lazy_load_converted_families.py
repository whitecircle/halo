#!/usr/bin/env python
"""The lazy loaders reproduce ``from_pretrained`` bitwise on every hub-converted family — on CPU.

A family whose hub checkpoint needs transformers' conversion mapping declares its key(s) on the EP
layer class (``_HUB_CONVERSION_KEYS``) and the lazy loaders replay that mapping per key. This drives
the REAL :func:`~src.distributed.expert_parallel.lazy_loader.load_ep_model_lazy` (EP patching
stubbed — it needs DeepEP and process groups) on a tiny checkpoint each family writes through its
own ``save_pretrained`` (the hub layout: vendor namespaces, split or per-expert projections, the
Step-3.5 vision tower's fused ``in_proj``), against the ``from_pretrained`` reference on the same
directory:

  * ep1 — every live tensor equals the reference (float parameters at the run's dtype, a buffer at
    the checkpoint's own dtype), the lazy gate admits the checkpoint, and every disk key is consumed;
  * ep2, rank 0 — every expert tensor equals the reference's FIRST half, sliced through the ranged
    read, which for a multi-source fan-in slices EVERY source (a whole-source read would fail the
    shape gate; a slice of the first source only would hand the second half garbage).

The roster is read off the registry — a family is covered once it declares its keys, is
lazy-loadable and has a tiny checkpoint builder here — so a family whose flag flips joins on its own.

    python tests/cpu/parallelism/test_lazy_load_converted_families.py
"""

import sys

import pytest
import torch
from transformers import AutoConfig
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextForConditionalGeneration
from transformers.models.step3p7.configuration_step3p7 import Step3p7Config
from transformers.models.step3p7.modeling_step3p7 import Step3p7ForConditionalGeneration

import src.distributed.expert_parallel.lazy_loader as ep_lazy_loader
from src.distributed.expert_parallel.config import EPConfig
from src.distributed.expert_parallel.expert_weights import ep_layer_class_by_model_type, is_expert_weight_attr
from src.distributed.expert_parallel.lazy_loader import lazy_loader_supports_checkpoint, load_ep_model_lazy
from src.models.loading.lazy_safetensors.weights import resolve_safetensors_index
from tests.common.models import (
    TINY_DSV4_CONFIG,
    TINY_GLM5_CONFIG,
    TINY_GLM5_VISION_CONFIG,
    TINY_STEP3P7_CONFIG,
    TINY_STEP3P7_VISION_CONFIG,
)

RUN_DTYPE = torch.bfloat16


def _glm5_next():
    config = Glm5NextConfig(text_config=dict(TINY_GLM5_CONFIG), vision_config=dict(TINY_GLM5_VISION_CONFIG))
    return Glm5NextForConditionalGeneration, Glm5NextForConditionalGeneration(config)


def _step3p7():
    config = Step3p7Config(text_config=dict(TINY_STEP3P7_CONFIG), vision_config=dict(TINY_STEP3P7_VISION_CONFIG))
    return Step3p7ForConditionalGeneration, Step3p7ForConditionalGeneration(config)


def _deepseek_v4():
    return DeepseekV4ForCausalLM, DeepseekV4ForCausalLM(DeepseekV4Config(**TINY_DSV4_CONFIG))


# model_type → (model class, tiny random-init instance) for the families with a tiny config.
_BUILDERS = {"glm5_next": _glm5_next, "step3p7": _step3p7, "deepseek_v4": _deepseek_v4}


def _lazy_converted_families() -> list[str]:
    registry = ep_layer_class_by_model_type()
    return sorted(
        mt
        for mt in _BUILDERS
        if (cls := registry.get(mt)) is not None and cls._HUB_CONVERSION_KEYS and cls._supports_lazy_loading
    )


# Families pinned on GPU instead (their own hub-layout checkpoint written by the test):
# ``tests/gpu/parallelism/ep/test_lazy_load_inkling.py``.
_GPU_PINNED_MODEL_TYPES = frozenset({"inkling_mm_model"})


def test_every_lazy_converted_family_is_pinned_here_or_on_gpu():
    """The parametrization reads its builders, so a family that declares ``_HUB_CONVERSION_KEYS``
    without a tiny builder would silently drop out of the resume oracle."""
    registry = ep_layer_class_by_model_type()
    declaring = {cls for cls in registry.values() if cls._HUB_CONVERSION_KEYS and cls._supports_lazy_loading}
    pinned = {registry[mt] for mt in (*_BUILDERS, *_GPU_PINNED_MODEL_TYPES)}
    assert declaring <= pinned, sorted(cls.__name__ for cls in declaring - pinned)


@pytest.fixture(scope="module", params=_lazy_converted_families())
def family(request, tmp_path_factory):
    """``(model_type, model class, checkpoint dir)`` — the tiny model saved in its HUB layout."""
    torch.manual_seed(0)
    model_class, model = _BUILDERS[request.param]()
    path = tmp_path_factory.mktemp(f"{request.param}_hub_ckpt")
    model.to(RUN_DTYPE).save_pretrained(path)
    return request.param, model_class, str(path)


@pytest.fixture(autouse=True)
def _stub_ep_patching(monkeypatch):
    """EP patching and buffer creation need DeepEP + process groups; the load itself does not."""
    monkeypatch.setattr(ep_lazy_loader, "patch_moe_model_for_ep", lambda model, *a, **k: model)
    monkeypatch.setattr(ep_lazy_loader, "create_ep_buffers", lambda *a, **k: None)


def _lazy(model_class, path: str, ep_size: int):
    ep_config = EPConfig(ep_size=ep_size, world_size=ep_size, gpus_per_node=ep_size)
    return load_ep_model_lazy(
        path,
        ep_config,
        AutoConfig.from_pretrained(path, trust_remote_code=False),
        dtype=RUN_DTYPE,
        trust_remote_code=False,
        model_class=model_class,
    )


@pytest.fixture(scope="module")
def reference(family):
    _, model_class, path = family
    model = model_class.from_pretrained(path, dtype=RUN_DTYPE)
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _mismatches(lazy_sd: dict, reference: dict, expert_slice: slice | None = None) -> list[str]:
    """Keys whose lazy tensor differs from the reference (expert tensors: from its ``expert_slice``).

    Compared at the narrower of the two dtypes: the lazy loader casts float PARAMETERS to the run's
    dtype where ``from_pretrained`` honors a family's fp32 pins, and keeps a float BUFFER at the
    checkpoint's dtype where ``from_pretrained`` casts it — both sides round the same stored value.
    """
    bad = []
    for key, lazy in lazy_sd.items():
        expected = reference[key]
        if expert_slice is not None and is_expert_weight_attr(key.rpartition(".")[2]) and expected.dim() == 3:
            expected = expected[expert_slice]
        if lazy.is_floating_point() and lazy.dtype != expected.dtype:
            narrow = min(lazy.dtype, expected.dtype, key=lambda dt: torch.finfo(dt).bits)
            lazy, expected = lazy.to(narrow), expected.to(narrow)
        if lazy.shape != expected.shape or not torch.equal(lazy, expected):
            bad.append(key)
    return bad


def test_ep1_lazy_load_equals_from_pretrained(family, reference):
    model_type, model_class, path = family
    assert lazy_loader_supports_checkpoint(path)

    model = _lazy(model_class, path, ep_size=1)
    lazy_sd = model.state_dict()
    assert set(lazy_sd) == set(reference)
    assert not _mismatches(lazy_sd, reference)
    # The loader's dtype contract: every float parameter at the run's dtype (FSDP2 refuses a mixed
    # shard group), buffers untouched.
    off_dtype = [name for name, p in model.named_parameters() if p.is_floating_point() and p.dtype != RUN_DTYPE]
    assert not off_dtype, off_dtype
    assert not any(t.is_meta for t in lazy_sd.values())


def test_ep1_load_consumes_every_disk_key(family, caplog):
    """No hub key may fall through the mapping: an unexpected-key warning here means a tensor the
    reference loads and the lazy model initializes from the meta shell instead."""
    _, model_class, path = family
    with caplog.at_level("WARNING", logger=ep_lazy_loader.logger.name):
        _lazy(model_class, path, ep_size=1)
    unexpected = [
        record.getMessage() for record in caplog.records if "align to no model tensor" in record.getMessage()
    ]
    assert not unexpected, unexpected


def test_ep2_rank0_loads_the_first_expert_half_of_every_source(family, reference):
    _, model_class, path = family
    model = _lazy(model_class, path, ep_size=2)
    lazy_sd = model.state_dict()
    num_experts = next(
        v.shape[0] for k, v in reference.items() if is_expert_weight_attr(k.rpartition(".")[2]) and v.dim() == 3
    )
    sliced = [k for k, v in lazy_sd.items() if is_expert_weight_attr(k.rpartition(".")[2]) and v.dim() == 3]
    assert sliced and all(lazy_sd[k].shape[0] == num_experts // 2 for k in sliced)
    assert not _mismatches(lazy_sd, reference, expert_slice=slice(0, num_experts // 2))


def test_the_hub_layout_is_not_the_canonical_one(family):
    """Anti-vacuity: the saved checkpoint really is in a namespace the model does not spell, so the
    equality above went through the conversion rather than a plain key-for-key read."""
    _, model_class, path = family
    weight_map, _ = resolve_safetensors_index(path)
    with torch.device("meta"):
        live = set(model_class.from_pretrained(path, device_map="meta").state_dict())
    assert set(weight_map) - live, "every disk key is a live key — nothing to convert"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
