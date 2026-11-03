"""The toolkit's weight funnel holds full weights on every rank under accelerate's rank-0-only loading, which zeroes
other ranks and never broadcasts to a frozen reference."""

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


def _abs_sum(model) -> float:
    return float(sum(p.detach().abs().sum() for p in model.parameters()))


def _load_on_rank(rank: int, checkpoint: str, out_dir: str) -> None:
    control = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.float32)
    policy = from_pretrained_verified(AutoModelForCausalLM, checkpoint, dtype=torch.float32)
    frozen = load_frozen_auxiliary_model(checkpoint, dtype=torch.float32, trust_remote_code=False, download_tag=None)
    record = {
        "control": _abs_sum(control),
        "policy": _abs_sum(policy),
        "frozen": _abs_sum(frozen),
        "flag_after": os.environ.get("FSDP_CPU_RAM_EFFICIENT_LOADING"),
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
    return _abs_sum(model), records


def test_the_control_load_is_zero_off_local_rank_0(per_rank):
    expected, records = per_rank
    assert records[0]["control"] == pytest.approx(expected)
    assert records[1]["control"] == 0.0


@pytest.mark.parametrize("load", ["policy", "frozen"])
def test_every_rank_loads_the_full_weights(per_rank, load):
    expected, records = per_rank
    for rank, record in enumerate(records):
        assert record[load] == pytest.approx(expected), f"{load} on rank {rank}"


def test_the_loading_flag_is_restored(per_rank):
    _, records = per_rank
    assert [record["flag_after"] for record in records] == ["true"] * RANKS


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
