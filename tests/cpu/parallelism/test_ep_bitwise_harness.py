"""The EP bitwise harness finds nothing between two runs of today's EP code, every case but the known ones records
what its variant owes, each seeded change to an EP layer shows up as a delta in the record it touches, and the
verdict fails when its negative control raises instead of moving a record.

Run: python tests/cpu/parallelism/test_ep_bitwise_harness.py
"""

import pytest
import torch

from src.distributed.expert_parallel import base_layer
from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.expert_parallel.config import EPConfig
from src.distributed.expert_parallel.expert_weights import ep_layer_class_by_model_type
from src.distributed.expert_parallel.grad_sync import create_expert_grad_hook
from tests.common.equivalence import ep_bitwise
from tests.common.equivalence.ep_bitwise import (
    CONSTRUCTION_VARIANTS,
    FAMILIES,
    KNOWN_RAISING,
    LAUNCHES,
    SEQ,
    VARIANTS,
    assess,
    control_case,
    produce,
    unexpected_raises,
)
from tests.common.equivalence.revisions import RAISED, run_in_tree
from tests.common.equivalence.tensor_records import (
    GRADIENT_PREFIX,
    case_of,
    compare_records,
    load_record,
    raised_records,
    record_deltas,
    record_paths,
)
from tests.common.models import TINY_QWEN3_MOE_CONFIG
from tests.common.repo import REPO_ROOT

CPU = LAUNCHES["cpu"][0]
GROUPED = LAUNCHES["gpu"][0]
MODULE = produce.__module__
SAMPLE_FAMILIES = ("qwen3_moe", "gpt_oss", "deepseek_v4")
ROUTER = "model.layers.0.mlp.gate.weight"
EXPERT_SHARD = "model.layers.0.mlp.gate_proj"


def _records(root, family: str, variant: str) -> dict:
    return load_record(root / CPU.label / family / variant / "rank0.pt")


def _produced(output, families, variants, **kwargs):
    produce(output, CPU, families, variants, **kwargs)
    return output


def _differences(baseline, mutated, family: str, variant: str) -> list[str]:
    """Every line by which the one changed record differs, the listing's cut aside."""
    [delta] = compare_records(baseline, mutated)
    assert delta["path"] == f"{CPU.label}/{family}/{variant}/rank0"
    return record_deltas(_records(baseline, family, variant), _records(mutated, family, variant))


def _hooks(record: dict, rank: int) -> list[str]:
    return record["orders"][f"rank{rank}/grad_hooks"]


@pytest.fixture(scope="module")
def roster(tmp_path_factory):
    """Every family through every CPU variant, once for the module (a few seconds)."""
    return _produced(tmp_path_factory.mktemp("roster"), FAMILIES, VARIANTS)


def test_only_the_known_cases_raise(roster):
    raised = raised_records(roster)
    assert {case_of(record) for record in raised} == {case for case in KNOWN_RAISING if case.startswith("cpu/")}
    assert {"cpu/gemma4_text/routing_replay", "grouped/zaya/routing_replay"} <= KNOWN_RAISING
    assert unexpected_raises(raised) == []
    assert unexpected_raises({**raised, "cpu/qwen3_moe/defaults/rank0": {}}) == ["cpu/qwen3_moe/defaults"]


def test_every_forwarding_case_records_its_loss_logits_and_gradients(roster):
    missing = []
    for record_path, path in record_paths(roster).items():
        family, variant = record_path.split("/")[1:3]
        if case_of(record_path) in KNOWN_RAISING or variant in CONSTRUCTION_VARIANTS:
            continue
        tensors = load_record(path)["tensors"]
        gradients = [name for name in tensors if name.startswith(GRADIENT_PREFIX)]
        if not {"loss", "logits"} <= tensors.keys() or not gradients or not torch.isfinite(tensors["loss"]):
            missing.append(record_path)
        replayable = ep_layer_class_by_model_type()[family]._supports_routing_replay
        if replayable and tensors.get("routing", torch.empty(0)).shape[:2] != (2, SEQ["cpu"]):
            missing.append(f"{record_path}: routing")
    assert missing == []


def test_each_forward_variant_records_what_sets_it_apart(roster):
    defaults, fp32 = _records(roster, "qwen3_moe", "defaults"), _records(roster, "qwen3_moe", "fp32_masters")
    dtypes = {line.split()[0]: line.split()[1] for line in fp32["orders"]["named_parameters"]}
    assert (dtypes[ROUTER], dtypes["model.embed_tokens.weight"]) == ("torch.float32", "torch.bfloat16")
    # FSDP-managed experts register no hook; FSDP-ignored ones without peers register an inert one.
    assert defaults["orders"]["grad_hooks"] == []
    assert fp32["orders"]["grad_hooks"] and all(line.endswith(" scale 1.0") for line in fp32["orders"]["grad_hooks"])

    lora = _records(roster, "qwen3_moe", "expert_lora")["tensors"]
    assert any("lora" in name and lora[name].any() for name in lora if name.startswith(GRADIENT_PREFIX))

    replayed = _records(roster, "qwen3_moe", "routing_replay")["tensors"]["routing"]
    assert not torch.equal(replayed, defaults["tensors"]["routing"])
    assert all(len(set(row.tolist())) == row.numel() for row in replayed.reshape(-1, replayed.shape[-1]))

    biased = _records(roster, "qwen3_moe", "bias_update")["tensors"]
    loads = [tensor for name, tensor in biased.items() if name.startswith("expert_load/")]
    routed = 2 * SEQ["cpu"] * TINY_QWEN3_MOE_CONFIG["num_experts_per_tok"]
    assert loads and all(int(counts.sum()) == routed for counts in loads)


def test_each_construction_variant_records_what_sets_it_apart(roster):
    hooked = _records(roster, "qwen3_moe", "in_backward_grad_sync")
    router_hooks = [line for line in _hooks(hooked, 0) if line.startswith(f"{ROUTER} ")]
    assert router_hooks == [f"{ROUTER} all_reduce(RedOpType.SUM, torch.float32, default group) scale 0.5"]
    assert f"{EXPERT_SHARD} scale 0.5" in _hooks(hooked, 1)

    deferred = _records(roster, "qwen3_moe", "deferred_grad_sync")
    assert [_hooks(deferred, rank) for rank in range(4)] == [[]] * 4

    saved = _hooks(_records(roster, "qwen3_moe", "modules_to_save"), 0)
    assert any(".gate.modules_to_save.default.weight all_reduce(" in line for line in saved)
    assert all(_records(roster, family, "in_backward_grad_sync")["orders"]["rank0/grad_hooks"] for family in FAMILIES)

    shards = _records(roster, "qwen3_moe", "etp_construction")["tensors"]
    assert not torch.equal(shards[f"rank0/{EXPERT_SHARD}"], shards[f"rank1/{EXPERT_SHARD}"])


def test_the_negative_control_skips_cases_known_to_raise():
    assert control_case(CPU, ["zaya"], ["routing_replay", "defaults"]) == ("zaya", "defaults")
    assert control_case(GROUPED, FAMILIES, list(CONSTRUCTION_VARIANTS)) is None


def test_two_producer_processes_match(tmp_path):
    """Base against base, each tree's producer in its own process as ``make verify-ep`` runs them."""
    outputs = [tmp_path / "first", tmp_path / "second"]
    for output in outputs:
        args = ["--produce", str(output), "--launch", CPU.label, "--families", *SAMPLE_FAMILIES]
        run_in_tree(REPO_ROOT, MODULE, *args)

    assert len(record_paths(outputs[0])) == len(SAMPLE_FAMILIES) * len(VARIANTS)
    assert compare_records(*outputs) == []


def test_a_rescaled_glu_combine_is_a_logit_and_gradient_delta(monkeypatch, tmp_path):
    baseline = _produced(tmp_path / "baseline", ["qwen3_moe"], ["defaults"])
    perturbed = _produced(tmp_path / "perturbed", ["qwen3_moe"], ["defaults"], perturb=True)
    combine = EPMoELayerBase._glu_combine
    monkeypatch.setattr(EPMoELayerBase, "_glu_combine", lambda self, gate, up: combine(self, gate, up) * (1 + 2**-12))
    mutated = _produced(tmp_path / "mutated", ["qwen3_moe"], ["defaults"])

    entries = _differences(baseline, mutated, "qwen3_moe", "defaults")

    assert "tensors/logits: bits differ" in " ".join(entries)
    assert any(entry.startswith(f"tensors/{GRADIENT_PREFIX}") for entry in entries)
    assert assess(baseline, mutated, perturbed)[0] == "fail"
    verdict, findings = assess(baseline, mutated, perturbed, accept=["cpu/qwen3_moe"])
    assert (verdict, len(findings["accepted"])) == ("pass", 1)


def test_a_router_master_at_another_dtype_is_a_dtype_delta(monkeypatch, tmp_path):
    baseline = _produced(tmp_path / "baseline", ["qwen3_moe"], ["in_backward_grad_sync"])

    def to_fp64(self, router_attr):
        setattr(self, router_attr, getattr(self, router_attr).double())

    monkeypatch.setattr(EPMoELayerBase, "_upcast_router_to_fp32", to_fp64)

    entries = _differences(
        baseline,
        _produced(tmp_path / "mutated", ["qwen3_moe"], ["in_backward_grad_sync"]),
        "qwen3_moe",
        "in_backward_grad_sync",
    )

    assert any(entry.startswith("orders/rank0/named_parameters") and "torch.float64" in entry for entry in entries)
    assert f"tensors/rank0/{ROUTER}: torch.float32[8, 256] != torch.float64[8, 256]" in entries


def test_unregistered_gradient_hooks_are_an_order_delta(monkeypatch, tmp_path):
    baseline = _produced(tmp_path / "baseline", ["gpt_oss"], ["in_backward_grad_sync"])
    monkeypatch.setattr(EPMoELayerBase, "_setup_gradient_sync", lambda self, router, expert_params: None)

    entries = _differences(
        baseline,
        _produced(tmp_path / "mutated", ["gpt_oss"], ["in_backward_grad_sync"]),
        "gpt_oss",
        "in_backward_grad_sync",
    )

    assert {entry.split("[")[0] for entry in entries} == {"orders/rank0/grad_hooks", "orders/rank1/grad_hooks"}


def test_a_router_given_the_expert_hook_is_an_order_delta(monkeypatch, tmp_path):
    """Router gradients that lose their DP all-reduce: the same hook count, a different hook."""
    baseline = _produced(tmp_path / "baseline", ["qwen3_moe"], ["in_backward_grad_sync"])
    monkeypatch.setattr(base_layer, "create_router_grad_hook", create_expert_grad_hook)

    entries = _differences(
        baseline,
        _produced(tmp_path / "mutated", ["qwen3_moe"], ["in_backward_grad_sync"]),
        "qwen3_moe",
        "in_backward_grad_sync",
    )

    assert {entry.split("[")[0] for entry in entries} == {"orders/rank0/grad_hooks", "orders/rank1/grad_hooks"}
    assert all("all_reduce" in entry.split(" != ")[0] for entry in entries)


def test_a_hook_under_a_deferred_sync_is_a_delta(monkeypatch, tmp_path):
    """In-backward hooks in a layout the post-backward sweep owns: here the expert hook refuses to build."""
    baseline = _produced(tmp_path / "baseline", ["qwen3_moe"], ["deferred_grad_sync"])
    init = EPConfig.__init__

    def undeferred(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self.defer_grad_sync = False

    monkeypatch.setattr(EPConfig, "__init__", undeferred)

    entries = _differences(
        baseline,
        _produced(tmp_path / "mutated", ["qwen3_moe"], ["deferred_grad_sync"]),
        "qwen3_moe",
        "deferred_grad_sync",
    )

    assert len(entries) == 1 and entries[0].startswith(f"{RAISED}: None != {{'type': 'RuntimeError'")


def test_a_saved_router_left_unhooked_is_an_order_delta(monkeypatch, tmp_path):
    baseline = _produced(tmp_path / "baseline", ["qwen3_moe"], ["modules_to_save"])
    monkeypatch.setattr(EPMoELayerBase, "reattach_router_grad_sync", lambda self: 0)

    entries = _differences(
        baseline, _produced(tmp_path / "mutated", ["qwen3_moe"], ["modules_to_save"]), "qwen3_moe", "modules_to_save"
    )

    assert {entry.split("[")[0] for entry in entries} == {"orders/rank0/grad_hooks", "orders/rank1/grad_hooks"}


def test_a_shifted_replay_selection_is_a_routing_delta(monkeypatch, tmp_path):
    baseline = _produced(tmp_path / "baseline", ["mistral4"], ["routing_replay"])
    replace = EPMoELayerBase._maybe_replace_selection
    monkeypatch.setattr(
        EPMoELayerBase,
        "_maybe_replace_selection",
        lambda self, indices: (replace(self, indices) + 1) % self.num_experts,
    )

    entries = _differences(
        baseline, _produced(tmp_path / "mutated", ["mistral4"], ["routing_replay"]), "mistral4", "routing_replay"
    )

    assert any(entry.startswith("tensors/routing:") for entry in entries)


def test_a_swapped_expert_tp_shard_is_a_parameter_delta(monkeypatch, tmp_path):
    baseline = _produced(tmp_path / "baseline", ["qwen3_moe"], ["etp_construction"])

    def other_rank_shard(self, tensor, dim):
        shard = self._etp_shard_size(tensor.shape[dim])
        return tensor.narrow(dim, (self.expert_tp_size - 1 - self.expert_tp_rank) * shard, shard)

    monkeypatch.setattr(EPMoELayerBase, "_etp_narrow", other_rank_shard)

    entries = _differences(
        baseline, _produced(tmp_path / "mutated", ["qwen3_moe"], ["etp_construction"]), "qwen3_moe", "etp_construction"
    )

    assert {entry.split("/")[1] for entry in entries} == {"rank0", "rank1"}
    assert all(entry.startswith("tensors/") for entry in entries)


def test_the_negative_control_differs_in_every_variant(tmp_path):
    baseline = _produced(tmp_path / "baseline", ["glm4_moe_lite"], VARIANTS)
    perturbed = _produced(tmp_path / "perturbed", ["glm4_moe_lite"], VARIANTS, perturb=True)

    deltas = compare_records(baseline, perturbed)

    assert sorted(case_of(delta["path"]) for delta in deltas) == sorted(
        f"{CPU.label}/glm4_moe_lite/{variant}" for variant in VARIANTS
    )
    assert assess(baseline, baseline, perturbed)[0] == "pass"


def test_a_negative_control_that_raises_fails_the_verdict(monkeypatch, tmp_path):
    baseline = _produced(tmp_path / "baseline", ["qwen3_moe"], ["defaults"])
    monkeypatch.setattr(ep_bitwise, "roll_expert_banks", lambda model: 0)
    control = _produced(tmp_path / "control", ["qwen3_moe"], ["defaults"], perturb=True)

    verdict, findings = assess(baseline, baseline, control)

    assert verdict == "fail"
    assert findings["negative_control"]["unseen"] == ["cpu/qwen3_moe/defaults/rank0"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
