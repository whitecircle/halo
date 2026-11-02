#!/usr/bin/env python
"""The CP wrapper hands the chunk's ``attention_mask`` to the wrapped forward only where something besides attention
reads it.

Ulysses attention never reads the mask, while transformers' flash mask construction reads it back to the host
(``flash_attention_mask``'s ``.all()``), one device sync per forward for nothing. The router aux loss is the one
reader of the CP roster outside attention: it keeps padding out of its token statistics, so a forward computing it
(``output_router_logits``) keeps the mask. On two gloo ranks every roster family, its MoE families with the aux term
on, trains to the bitwise loss, aux term and gradients of the wrapper that always hands the mask over, on a batch
whose padding falls in one rank's chunk; dropping the mask from an aux-loss family moves its aux term there.
Every family whose forward computes the aux term runs with it on, every other one without.

Run: python tests/cpu/parallelism/test_cp_attention_mask_reach.py
"""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from accelerate import PartialState
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS

from halo.distributed.context_parallel.ulysses_model import UlyssesModel
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.distributed.parallelism_config import ParallelismConfig
from halo.kernels.logprobs import LABEL_IGNORE_INDEX
from tests.common.ep_reference import random_token_batch
from tests.common.equivalence.cp_equivalence import (
    BATCH,
    CP_SIZE,
    DEFAULT_ATTN_IMPLEMENTATION,
    FAMILIES,
    SEED,
    SEQ,
    tier_controls,
    tiny_model,
)
from tests.common.gloo import run_gloo_ranks
from tests.common.parallelism import single_process_ep_config

# Loud enough that padding counted into the router statistics moves the aux term past fp32 rounding.
MOE_AUX_OVERRIDES = {"output_router_logits": True, "router_aux_loss_coef": 0.1}
# The roster families whose forward returns ``aux_loss`` under ``output_router_logits``, run with it on; every other
# family runs without it.
AUX_LOSS_FAMILIES = ("gpt_oss", "qwen3_5_moe_text", "qwen3_moe")
# The last row's trailing pads, all inside the final rank's chunk.
PADS = SEQ // CP_SIZE // 2


def _batch(vocab_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ids, labels = random_token_batch(vocab_size, BATCH, SEQ, "cpu", seed=SEED)
    mask = torch.ones_like(ids)
    mask[-1, -PADS:] = 0
    labels[-1, -PADS:] = LABEL_IGNORE_INDEX
    return ids, mask, labels


def _wrapped(name: str, cp_config) -> UlyssesModel:
    family = FAMILIES[name]
    model = tiny_model(name, "cpu", torch.device("cpu"), **(MOE_AUX_OVERRIDES if name in AUX_LOSS_FAMILIES else {}))
    if family.moe:
        apply_ep_layers(model, single_process_ep_config())
    return UlyssesModel(model, cp_config).train()


def _step(name: str, cp_config) -> dict:
    """One training step's loss, aux term and gradients, the mask the wrapped forward received, and whether each
    flash mask construction (which reads a given mask back to the host) was given one."""
    wrapper = _wrapped(name, cp_config)
    received, flash_masks = [], []
    wrapper.model.register_forward_pre_hook(
        lambda _module, _args, kwargs: received.append(kwargs.get("attention_mask")), with_kwargs=True
    )
    flash_mask = ALL_MASK_ATTENTION_FUNCTIONS[DEFAULT_ATTN_IMPLEMENTATION]

    def recording_flash_mask(*args, attention_mask=None, **kwargs):
        flash_masks.append(attention_mask is not None)
        return flash_mask(*args, attention_mask=attention_mask, **kwargs)

    ids, mask, labels = _batch(wrapper.get_input_embeddings().num_embeddings)
    with patch.dict(ALL_MASK_ATTENTION_FUNCTIONS, {DEFAULT_ATTN_IMPLEMENTATION: recording_flash_mask}):
        outputs = wrapper(input_ids=ids, attention_mask=mask, labels=labels)
    outputs.loss.backward()
    aux = getattr(outputs, "aux_loss", None)
    return {
        "loss": outputs.loss.detach(),
        "aux": None if aux is None else aux.detach(),
        "grads": {n: p.grad.detach().clone() for n, p in wrapper.named_parameters() if p.grad is not None},
        "received_mask": [m is not None for m in received],
        "flash_masks": flash_masks,
    }


def _record_worker(rank: int, output: str) -> None:
    PartialState()
    cp_config = ParallelismConfig(cp_size=CP_SIZE).cp_config()
    always = patch.object(UlyssesModel, "_backbone_attention_mask", lambda self, mask, kwargs: mask)
    with tier_controls("cpu"):
        runs = {}
        for name in FAMILIES:
            runs[name] = {"current": _step(name, cp_config)}
            with always:
                runs[name]["always_masked"] = _step(name, cp_config)
    torch.save(runs, Path(output, f"rank{rank}.pt"))


@pytest.fixture(scope="module")
def recorded(tmp_path_factory) -> list[dict]:
    output = tmp_path_factory.mktemp("cp_mask_reach")
    run_gloo_ranks(_record_worker, CP_SIZE, str(output))
    return [torch.load(output / f"rank{rank}.pt", weights_only=False) for rank in range(CP_SIZE)]


@pytest.mark.parametrize("name", list(FAMILIES))
def test_the_step_is_bitwise_the_always_masked_ones(recorded, name):
    for rank, runs in enumerate(recorded):
        current, reference = runs[name]["current"], runs[name]["always_masked"]
        assert torch.equal(current["loss"], reference["loss"]), f"rank {rank}"
        assert (current["aux"] is None) == (reference["aux"] is None) == (name not in AUX_LOSS_FAMILIES)
        if current["aux"] is not None:
            assert torch.equal(current["aux"], reference["aux"]), f"rank {rank}: aux term"
        assert current["grads"].keys() == reference["grads"].keys()
        drifted = [n for n, g in reference["grads"].items() if not torch.equal(current["grads"][n], g)]
        assert drifted == [], f"rank {rank}: {drifted[:4]}"


@pytest.mark.parametrize("name", list(FAMILIES))
def test_the_mask_reaches_only_an_aux_loss_forward(recorded, name):
    for runs in recorded:
        assert runs[name]["current"]["received_mask"] == [name in AUX_LOSS_FAMILIES]
        assert runs[name]["always_masked"]["received_mask"] == [True]


@pytest.mark.parametrize("name", list(FAMILIES))
def test_flash_mask_construction_reads_a_mask_only_for_an_aux_loss_forward(recorded, name):
    """The host read itself: transformers' flash mask construction is handed a mask only where the aux loss keeps
    it. A family whose remote code builds its masks itself never reaches the construction."""
    for runs in recorded:
        current, reference = runs[name]["current"]["flash_masks"], runs[name]["always_masked"]["flash_masks"]
        assert len(current) == len(reference)
        assert all(current) if name in AUX_LOSS_FAMILIES else not any(current), current
        if reference:
            assert all(reference), "anti-vacuity: the always-masked wrapper hands each construction the mask"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
