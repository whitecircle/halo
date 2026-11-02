#!/usr/bin/env python
"""An adapter resume copies nothing until every rank has matched the file to the live adapters, key for key at the
saved shapes.

``restore_adapters`` copies a checkpoint's adapters into a model whose base stays as built: an EP/CP rebuild, or the
adapter-only checkpoint of a PEFT run. A file short of a tensor would resume it from initialization, a saved tensor no
live one takes would be dropped, a changed ``lora_r`` fails part-way through the copies, and a rank raising alone
leaves its peers in the next collective. Pinned on a mixed run (PEFT attention adapters with a ``modules_to_save``
head carrying a persistent buffer, beside native expert LoRA), its adapter written by the real saver: each defect
raises naming the keys with every live tensor unchanged, the complete file restores bit-equal, and of two gloo ranks
(one EP group of two) only one of which reads a short file, both raise before either copies.

    python tests/cpu/peft/test_adapter_restore_strict.py
"""

import json
import os
import shutil
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file

from halo.checkpoint.format import ADAPTER_SAFETENSORS_FILE
from halo.checkpoint.parallel.adapters import restore_adapters, save_peft_adapters
from halo.models.moe.expert_lora_spec import ExpertLoraSpec
from halo.models.param_roles import persistent_buffers
from tests.common.checkpoint_io import save_context
from tests.common.ep_stubs import StubEPLayerBase
from tests.common.gloo import run_gloo_ranks
from tests.common.tiny_models import TINY_DENSE_FAMILY, tiny_family_model

NUM_EXPERTS = 4
HIDDEN = 128
EXPERT_FFN = 64
SPEC = ExpertLoraSpec(r=8, alpha=16.0, projections=frozenset({"gate", "up", "down"}))
EXPERT_PATH = "base_model.model.model.layers.0.mlp"
EXPERT_KEY = f"{EXPERT_PATH}.experts.gate_up_proj.lora_A"
ATTENTION_KEY = "base_model.model.model.layers.1.self_attn.q_proj.lora_A.weight"
HEAD_KEY = "base_model.model.lm_head.weight"
BUFFER_KEY = "base_model.model.lm_head.balancing_bias"


class _ExpertLoraLayer(StubEPLayerBase):
    """An EP layer holding this rank's slice of native grouped expert LoRA on two expert weights."""

    def __init__(self, *, ep_rank: int = 0, ep_size: int = 1, expert_r: int = SPEC.r):
        super().__init__()
        local = NUM_EXPERTS // ep_size
        self.expert_start, self.expert_end = ep_rank * local, (ep_rank + 1) * local
        self.ep_config = SimpleNamespace(ep_size=ep_size, dispatch_ep_group=None, expert_lora=SPEC)
        shapes = {"gate_up_proj": (HIDDEN, 2 * EXPERT_FFN), "down_proj": (EXPERT_FFN, HIDDEN)}
        for attr, (k, n) in shapes.items():
            setattr(self, f"{attr}_lora_A", nn.Parameter(torch.zeros(local, k, expert_r)))
            setattr(self, f"{attr}_lora_B", nn.Parameter(torch.zeros(local, expert_r, n)))
        self._expert_lora_attrs = frozenset(shapes)


def _mixed_model(*, r: int = 4, ep_rank: int = 0, ep_size: int = 1, expert_r: int = SPEC.r, seed: int | None = None):
    """The tiny Qwen3 with PEFT LoRA on ``q_proj``/``v_proj``, ``lm_head`` in ``modules_to_save`` beside a persistent
    buffer, and native expert LoRA in layer 0's MLP; every trainable value drawn from ``seed`` (left at init when
    None)."""
    torch.manual_seed(0)
    base = tiny_family_model(TINY_DENSE_FAMILY, overrides={"num_hidden_layers": 2})
    base.lm_head.register_buffer("balancing_bias", torch.zeros(base.config.vocab_size))
    peft_model = get_peft_model(
        base, LoraConfig(r=r, lora_alpha=8, target_modules=["q_proj", "v_proj"], modules_to_save=["lm_head"])
    )
    peft_model.base_model.model.model.layers[0].mlp = _ExpertLoraLayer(
        ep_rank=ep_rank, ep_size=ep_size, expert_r=expert_r
    )
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for _, tensor in _live(peft_model):
                full = torch.randn(_full_shape(tensor, ep_size), generator=generator)
                tensor.copy_(full if full.shape == tensor.shape else _expert_slice(full, ep_rank, ep_size))
    return peft_model


def _full_shape(tensor: torch.Tensor, ep_size: int) -> tuple[int, ...]:
    """A 3-D expert adapter's shape across every expert; any other tensor's own."""
    return (tensor.shape[0] * ep_size, *tensor.shape[1:]) if tensor.dim() == 3 else tuple(tensor.shape)


def _expert_slice(full: torch.Tensor, ep_rank: int, ep_size: int) -> torch.Tensor:
    local = full.shape[0] // ep_size
    return full[ep_rank * local : (ep_rank + 1) * local]


def _live(model) -> list[tuple[str, torch.Tensor]]:
    """Every tensor the run trains or the adapter carries: the trainable parameters and the ``modules_to_save``
    copy's persistent buffer."""
    params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    return params + [(name, buffer) for name, buffer in persistent_buffers(model) if ".modules_to_save." in name]


def _snapshot(model) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in _live(model)}


def _changed(model, before: dict[str, torch.Tensor]) -> list[str]:
    return [name for name, tensor in _live(model) if not torch.equal(tensor, before[name])]


@pytest.fixture(scope="module")
def saved(tmp_path_factory) -> str:
    """The trained mixed run's adapter, written by the save every resume reads back (its live dtype kept)."""
    directory = str(tmp_path_factory.mktemp("adapter"))
    trained = _mixed_model(seed=1)
    context = save_context(
        trained,
        parallelism_config=SimpleNamespace(is_cp_mode=False, expert_lora=SPEC),
        has_expert_lora=True,
        training_checkpoint=True,
    )
    save_peft_adapters(context, directory)
    return directory


def _variant(saved: str, target: str, edit) -> str:
    """A copy of the saved adapter directory whose tensors ``edit`` rewrote."""
    shutil.copytree(saved, target)
    path = os.path.join(target, ADAPTER_SAFETENSORS_FILE)
    tensors = load_file(path)
    save_file(edit(dict(tensors)), path, metadata={"format": "pt"})
    return target


def _without(*keys):
    return lambda tensors: {key: value for key, value in tensors.items() if key not in keys}


def _reshaped(key: str):
    return lambda tensors: {**tensors, key: torch.zeros(*tensors[key].shape[:-1], tensors[key].shape[-1] + 1)}


def test_the_complete_file_restores_every_tensor_bit_equal(saved):
    trained, fresh = _mixed_model(seed=1), _mixed_model(seed=2)
    assert _changed(fresh, _snapshot(trained)), "premise: the fresh model already holds the trained values"

    assert restore_adapters(saved, fresh, is_cp_mode=False) is not None

    assert not _changed(fresh, _snapshot(trained))
    saved_keys = set(load_file(os.path.join(saved, ADAPTER_SAFETENSORS_FILE)))
    assert {EXPERT_KEY, ATTENTION_KEY, HEAD_KEY, BUFFER_KEY} <= saved_keys, "premise: the file holds each kind"


@pytest.mark.parametrize(
    ("edit", "named"),
    [
        (lambda tensors: {}, HEAD_KEY),
        (_without(ATTENTION_KEY), ATTENTION_KEY),
        (_without(HEAD_KEY), HEAD_KEY),
        (_without(BUFFER_KEY), BUFFER_KEY),
        (_without(EXPERT_KEY), EXPERT_KEY),
        (_reshaped(ATTENTION_KEY), ATTENTION_KEY),
        (_reshaped(EXPERT_KEY), EXPERT_KEY),
        (
            lambda tensors: {**tensors, ATTENTION_KEY.replace("q_proj", "k_proj"): tensors[ATTENTION_KEY].clone()},
            "k_proj",
        ),
        (lambda tensors: {key: value for key, value in tensors.items() if ".experts." not in key}, EXPERT_KEY),
    ],
    ids=[
        "empty_file",
        "missing_attention",
        "missing_modules_to_save",
        "missing_buffer",
        "missing_expert",
        "reshaped_attention",
        "reshaped_expert",
        "partly_unexpected",
        "no_expert_tensors_for_an_expert_lora_run",
    ],
)
def test_a_file_that_does_not_match_the_live_adapters_restores_nothing(saved, tmp_path, edit, named):
    variant = _variant(saved, str(tmp_path / "variant"), edit)
    fresh = _mixed_model(seed=2)
    before = _snapshot(fresh)

    with pytest.raises(RuntimeError, match=named):
        restore_adapters(variant, fresh, is_cp_mode=False)
    assert not _changed(fresh, before), "a tensor was written before the refusal"


@pytest.mark.parametrize(("r", "expert_r"), [(8, SPEC.r), (4, 2 * SPEC.r)], ids=["attention_r", "expert_r"])
def test_a_resume_under_another_lora_r_restores_nothing(saved, r, expert_r):
    """``r`` scales nothing, so the scaling refusal passes it; the adapters differ in shape alone."""
    fresh = _mixed_model(r=r, expert_r=expert_r, seed=2)
    before = _snapshot(fresh)

    with pytest.raises(RuntimeError, match="reshaped"):
        restore_adapters(saved, fresh, is_cp_mode=False)
    assert not _changed(fresh, before)


def test_expert_tensors_with_no_expert_layer_to_receive_them_are_refused(saved):
    fresh = _mixed_model(seed=2)
    fresh.base_model.model.model.layers[0].mlp = nn.Identity()
    before = _snapshot(fresh)

    with pytest.raises(RuntimeError, match="no live tensor takes"):
        restore_adapters(saved, fresh, is_cp_mode=False)
    assert not _changed(fresh, before)


def _restore_on_ep_rank(rank: int, saved: str, short: str, outcome_dir: str) -> None:
    """One rank of a two-way EP group: rank 1 reads an adapter short of one expert tensor, as a non-shared filesystem
    can leave it, unless ``short`` is empty."""
    model = _mixed_model(ep_rank=rank, ep_size=2, seed=2)
    trained = _mixed_model(ep_rank=rank, ep_size=2, seed=1)
    before = _snapshot(model)
    try:
        restore_adapters(short if rank == 1 and short else saved, model, is_cp_mode=False)
        outcome = "restored"
    except RuntimeError as e:
        outcome = str(e)
    result = {
        "outcome": outcome,
        "changed": _changed(model, before),
        "unrestored": _changed(model, _snapshot(trained)),
    }
    with open(os.path.join(outcome_dir, f"rank{rank}.json"), "w") as fh:
        json.dump(result, fh)


def _ep_outcomes(saved: str, short: str, outcome_dir) -> list[dict]:
    outcome_dir.mkdir()
    run_gloo_ranks(_restore_on_ep_rank, 2, saved, short, str(outcome_dir))
    return [json.loads((outcome_dir / f"rank{rank}.json").read_text()) for rank in range(2)]


def test_two_ep_ranks_restore_their_own_expert_slices(saved, tmp_path):
    outcomes = _ep_outcomes(saved, "", tmp_path / "outcomes")
    assert [outcome["outcome"] for outcome in outcomes] == ["restored", "restored"], outcomes
    assert [outcome["unrestored"] for outcome in outcomes] == [[], []], outcomes


def test_one_ranks_short_file_is_refused_on_every_rank_before_any_copy(saved, tmp_path):
    short = _variant(saved, str(tmp_path / "short"), _without(EXPERT_KEY))
    outcomes = _ep_outcomes(saved, short, tmp_path / "outcomes")
    assert outcomes[0]["outcome"] == outcomes[1]["outcome"], outcomes
    assert "failed on 1 of 2 rank(s) [1]" in outcomes[0]["outcome"] and EXPERT_KEY in outcomes[0]["outcome"]
    assert [outcome["changed"] for outcome in outcomes] == [[], []], "a rank copied before the joined refusal"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
