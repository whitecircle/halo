"""Under accelerate's ``fsdp_cpu_ram_efficient_loading`` every rank loads the full weights and buffers, through a
plain load and the toolkit's policy and frozen loads alike: a frozen reference gets no broadcast to repair them."""

import json
import os

import pytest
import torch
from transformers import AutoModelForCausalLM
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from halo.model_loading.checkpoint_coverage import from_pretrained_verified
from halo.model_loading.parallel.frozen_models import load_frozen_auxiliary_model
from tests.common.gloo import run_gloo_ranks
from tests.common.models import TINY_QWEN3_CONFIG

RANKS = 2
ACCELERATE_FSDP_ENV = {"ACCELERATE_USE_FSDP": "true", "FSDP_CPU_RAM_EFFICIENT_LOADING": "true"}


def _abs_sum(model) -> tuple[float, float]:
    """Parameters, and the rotary ``inv_freq`` every load builds."""
    inv_freq = [b for name, b in model.named_buffers() if name.endswith("inv_freq") and "original" not in name]
    assert inv_freq, "the fixture has no rotary buffer to compare"
    return float(sum(p.detach().abs().sum() for p in model.parameters())), float(sum(b.abs().sum() for b in inv_freq))


def _load_on_rank(rank: int, checkpoint: str, out_dir: str) -> None:
    control = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.float32)
    policy = from_pretrained_verified(AutoModelForCausalLM, checkpoint, dtype=torch.float32)
    frozen = load_frozen_auxiliary_model(checkpoint, dtype=torch.float32, trust_remote_code=False, download_tag=None)
    record = {
        "control": _abs_sum(control),
        "policy": _abs_sum(policy),
        "frozen": _abs_sum(frozen),
    }
    with open(os.path.join(out_dir, f"rank{rank}.json"), "w") as handle:
        json.dump(record, handle)


@pytest.fixture(scope="module")
def per_rank(tmp_path_factory):
    checkpoint = tmp_path_factory.mktemp("tiny_qwen3")
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(CONFIG_MAPPING["qwen3"](**TINY_QWEN3_CONFIG))
    model.save_pretrained(checkpoint)
    out_dir = tmp_path_factory.mktemp("ranks")
    run_gloo_ranks(_load_on_rank, RANKS, str(checkpoint), str(out_dir), env=ACCELERATE_FSDP_ENV)
    records = [json.loads((out_dir / f"rank{rank}.json").read_text()) for rank in range(RANKS)]
    return _abs_sum(AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.float32)), records


@pytest.mark.parametrize("load", ["control", "policy", "frozen"])
def test_every_rank_loads_the_full_weights(per_rank, load):
    expected, records = per_rank
    for rank, record in enumerate(records):
        assert record[load] == pytest.approx(list(expected)), f"{load} on rank {rank}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
