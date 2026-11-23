#!/usr/bin/env python
"""Run-long per-expert token counts (``moe_expert_token_counts.json``): every optimizer step adds in, logged or not,
each token counts once per router, and a resume continues the totals. Totals are raw counts, so unlike the ``moe/*``
shares, a pass counted twice or a rank counted twice shows."""

from __future__ import annotations

import json
import logging
import os
from types import SimpleNamespace

import pytest
import torch
from transformers import GptOssConfig, GptOssForCausalLM

import halo.callbacks.balancing as balancing
import halo.callbacks.moe_metrics as moe_metrics
from halo.callbacks.balancing import BalancingBiasCallback
from halo.callbacks.moe_metrics import MoEMetricsCallback, merge_stage_counts, read_run_counts
from halo.checkpoint.format import MOE_EXPERT_TOKEN_COUNTS_FILE
from halo.distributed.parallelism_config import ParallelismConfig
from halo.models.moe.balancing import expert_load_token_weights
from halo.trainers.loop.checkpointing import CheckpointingMixin
from tests.common.gloo import run_gloo_ranks
from tests.common.models import TINY_GPTOSS_CONFIG

TOP_K = TINY_GPTOSS_CONFIG["num_experts_per_tok"]
NUM_LAYERS = TINY_GPTOSS_CONFIG["num_hidden_layers"]
NUM_EXPERTS = TINY_GPTOSS_CONFIG["num_local_experts"]
BATCH, SEQ = 2, 8


def _state(step: int) -> SimpleNamespace:
    return SimpleNamespace(global_step=step)


def _step_counts(step: int) -> list[torch.Tensor]:
    """Two layers of four experts, different every step."""
    return [torch.tensor([step, 0.0, 2.0, 1.0]), torch.tensor([0.0, step, 1.0, 3.0])]


def _drive(callback: MoEMetricsCallback, steps) -> torch.Tensor:
    """Each step's counters as its hooks would leave them; returns what the steps added up to."""
    total = torch.zeros(2, 4)
    for step in steps:
        callback._counters = _step_counts(step)
        total += torch.stack(_step_counts(step))
        callback.on_step_end(args=None, state=_state(step), control=None)
    return total


def test_unlogged_steps_add_to_the_run_totals():
    callback = MoEMetricsCallback(topk=1, log_every_n_steps=3)
    expected = _drive(callback, range(1, 6))
    record = callback.run_counts_record(5, token_replicas=1)
    assert record["counts"] == expected.long().tolist()
    assert (record["first_step"], record["last_step"], record["top_k"]) == (1, 5, 1)
    assert all(not counter.any() for counter in callback._counters), "the per-step counters were not zeroed"


def _tiny_gptoss() -> GptOssForCausalLM:
    torch.manual_seed(0)
    model = GptOssForCausalLM(GptOssConfig(**TINY_GPTOSS_CONFIG))
    model.train()
    return model


def _wired(model) -> MoEMetricsCallback:
    callback = MoEMetricsCallback(topk=TOP_K, enable_router_logits=True)
    callback.on_train_begin(args=None, state=None, control=None, model=model)
    assert len(callback._hook_handles) == NUM_LAYERS, "the callback did not hook every router"
    return callback


def _input_ids() -> torch.Tensor:
    return torch.randint(0, TINY_GPTOSS_CONFIG["vocab_size"], (BATCH, SEQ), generator=torch.Generator().manual_seed(1))


def _layer_totals(callback, step: int) -> list[int]:
    callback.on_step_end(args=None, state=_state(step), control=None)
    record = callback.run_counts_record(step, token_replicas=1)
    callback.on_train_end(args=None, state=None, control=None)
    assert len(record["counts"]) == NUM_LAYERS
    assert all(len(layer) == NUM_EXPERTS for layer in record["counts"])
    return [sum(layer) for layer in record["counts"]]


def test_each_layers_total_is_tokens_times_top_k():
    model = _tiny_gptoss()
    callback = _wired(model)
    model(input_ids=_input_ids()).logits.sum().backward()
    assert _layer_totals(callback, 1) == [BATCH * SEQ * TOP_K] * NUM_LAYERS


def test_a_checkpoint_recompute_adds_nothing():
    """MoE checkpointing is reentrant outside PP (``resolve_use_reentrant``): the router runs in the original pass,
    under no_grad, and again in the recompute."""
    model = _tiny_gptoss()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    callback = _wired(model)
    model(input_ids=_input_ids()).logits.sum().backward()
    assert _layer_totals(callback, 1) == [BATCH * SEQ * TOP_K] * NUM_LAYERS


def test_a_frozen_forward_adds_nothing():
    """A reference pass through the training model, as the GRPO scorer does, routes no training token."""
    model = _tiny_gptoss()
    callback = _wired(model)
    with torch.no_grad():
        model(input_ids=_input_ids())
    model(input_ids=_input_ids()).logits.sum().backward()
    assert _layer_totals(callback, 1) == [BATCH * SEQ * TOP_K] * NUM_LAYERS


def test_padding_positions_count():
    """The router sees every position and the experts compute every one, so a padded batch counts its padding."""
    model = _tiny_gptoss()
    callback = _wired(model)
    mask = torch.ones(BATCH, SEQ, dtype=torch.long)
    mask[0, SEQ // 2 :] = 0
    model(input_ids=_input_ids(), attention_mask=mask).logits.sum().backward()
    assert _layer_totals(callback, 1) == [BATCH * SEQ * TOP_K] * NUM_LAYERS


def test_a_model_declaring_no_router_counts_off_its_output_once(monkeypatch):
    """The fallback hook on the model output: a no-grad forward through the training model, as a reference scorer
    runs, adds nothing."""
    monkeypatch.setattr(moe_metrics, "declared_routers", lambda model: [])
    model = _tiny_gptoss()
    callback = MoEMetricsCallback(topk=TOP_K, enable_router_logits=True)
    callback.on_train_begin(args=None, state=None, control=None, model=model)
    assert len(callback._hook_handles) == 1, "the callback did not fall back to the model-output hook"
    with torch.no_grad():
        model(input_ids=_input_ids())
    model(input_ids=_input_ids()).logits.sum().backward()
    assert _layer_totals(callback, 1) == [BATCH * SEQ * TOP_K] * NUM_LAYERS


@pytest.mark.parametrize("hook", ["router", "model output"])
def test_a_tree_packed_token_counts_once_per_example(monkeypatch, hook):
    """SFT routes a tree-packed row under its leaf counts, and both hooks weigh each token by them, as the bias path's
    load recorder does."""
    if hook == "model output":
        monkeypatch.setattr(moe_metrics, "declared_routers", lambda model: [])
    model = _tiny_gptoss()
    callback = MoEMetricsCallback(topk=TOP_K, enable_router_logits=True)
    callback.on_train_begin(args=None, state=None, control=None, model=model)
    leaf_count = torch.ones(BATCH * SEQ)
    leaf_count[: SEQ // 2] = 3
    with expert_load_token_weights(leaf_count):
        model(input_ids=_input_ids()).logits.sum().backward()
    assert _layer_totals(callback, 1) == [int(leaf_count.sum()) * TOP_K] * NUM_LAYERS


def _reduce_two_ranks_counts(rank: int) -> None:
    """Each rank counts its own routing; the record holds both ranks' on every rank."""
    callback = MoEMetricsCallback(topk=1, log_every_n_steps=2)
    callback._counters = [torch.tensor([1.0 + rank, 0.0, 2.0, 0.0]), torch.tensor([0.0, rank, 0.0, 1.0])]
    callback.on_step_end(args=None, state=_state(1), control=None)
    record = callback.run_counts_record(1, token_replicas=1)
    assert record["counts"] == [[3, 0, 4, 0], [0, 1, 0, 2]], f"rank {rank}: {record['counts']}"


def test_the_record_sums_every_ranks_counts():
    run_gloo_ranks(_reduce_two_ranks_counts, 2)


class _Router(torch.nn.Module):
    def __init__(self, counts):
        super().__init__()
        self.balancing_biases = torch.zeros(len(counts))
        self.expert_load_counter = torch.tensor(counts, dtype=torch.float32)


def test_the_bias_callback_counts_its_own_slots_before_the_reduce(monkeypatch):
    """The bias update all-reduces every step. Its run totals take the local counts, which reduce once when read; the
    reduced ones would hold every other rank's tokens too."""

    def doubling_all_reduce(tensor, op=None, group=None):
        tensor.mul_(2)

    monkeypatch.setattr(balancing.dist, "all_reduce", doubling_all_reduce)
    monkeypatch.setattr(balancing, "get_global_world_size", lambda: 2)
    routers = [_Router([3.0, 1.0, 0.0, 4.0]), _Router([0.0, 2.0, 2.0, 4.0])]
    model = torch.nn.ModuleDict({"a": routers[0], "b": routers[1]})
    callback = BalancingBiasCallback(log_every_n_steps=3, top_k=2)
    for step in range(1, 5):
        routers[0].expert_load_counter.copy_(torch.tensor([3.0, 1.0, 0.0, 4.0]))
        routers[1].expert_load_counter.copy_(torch.tensor([0.0, 2.0, 2.0, 4.0]))
        callback.on_step_end(args=None, state=_state(step), control=None, model=model)
    assert [counts.tolist() for counts in callback._run_counts] == [[12, 4, 0, 16], [0, 8, 8, 16]]
    assert all(counts.dtype == torch.int64 for counts in callback._run_counts)


def test_stage_totals_join_in_stage_order_divide_out_partners_and_add_the_resume():
    stages = [[[2, 4]], None, [[6, 8], [2, 2]]]
    assert merge_stage_counts(stages, token_replicas=1, resumed=None) == [[2, 4], [6, 8], [2, 2]]
    assert merge_stage_counts(stages, token_replicas=2, resumed=None) == [[1, 2], [3, 4], [1, 1]]
    resumed = [[10, 10], [10, 10], [10, 10]]
    assert merge_stage_counts(stages, token_replicas=2, resumed=resumed) == [[11, 12], [13, 14], [11, 11]]
    assert merge_stage_counts([None, None], token_replicas=2, resumed=resumed) == resumed


def test_partners_that_disagree_are_named(caplog):
    with caplog.at_level(logging.WARNING):
        assert merge_stage_counts([[[3, 5]]], token_replicas=2, resumed=None) == [[2, 3]]
    assert "not a multiple of the 2 ranks" in caplog.text


def test_a_resume_from_another_model_is_refused():
    with pytest.raises(RuntimeError, match="came from another model"):
        merge_stage_counts([[[1, 2, 3]]], token_replicas=1, resumed=[[1, 2]])


@pytest.mark.parametrize(
    ("layout", "replicas"),
    [
        ({}, 1),
        ({"tp_size": 2}, 2),
        ({"cp_size": 2}, 1),
        ({"ep_size": 4}, 1),
        ({"expert_tp_size": 2}, 2),
        ({"ep_size": 2, "expert_tp_size": 2}, 2),
        ({"ep_size": 2, "tp_size": 2}, 2),
        ({"ep_size": 4, "cp_size": 2}, 1),
        ({"world_size": 8, "pp_size": 2, "ep_size": 2}, 1),
        ({"world_size": 8, "pp_size": 2, "expert_tp_size": 2}, 2),
    ],
)
def test_router_token_replicas_counts_the_ranks_sharing_a_tokens_routing(layout, replicas):
    """TP and expert-TP partners route one batch, while CP partners route different chunks of theirs. Two nodes of four for the PP rows, since a stage holds whole NVLink domains."""
    config = ParallelismConfig(**{"world_size": 4, "gpus_per_node": 4, **layout})
    assert config.router_token_replicas == replicas


class _Host(CheckpointingMixin):
    """The state the sidecar hooks read off a trainer."""

    def __init__(self, callbacks):
        self.callback_handler = SimpleNamespace(callbacks=callbacks)
        self.state = _state(0)
        self.parallelism_config = SimpleNamespace(router_token_replicas=1)

    def save(self, directory: str, step: int) -> dict:
        self.state.global_step = step
        os.makedirs(directory, exist_ok=True)
        self._persist_trainer_sidecars(directory)
        with open(os.path.join(directory, MOE_EXPERT_TOKEN_COUNTS_FILE)) as handle:
            return json.load(handle)


def test_a_resume_continues_the_count(tmp_path):
    first = MoEMetricsCallback(topk=2, exclude_last_slot=True)
    expected = _drive(first, range(1, 4))
    saved = _Host([first]).save(str(tmp_path / "checkpoint-3"), 3)
    assert saved["counts"] == expected.long().tolist()
    assert (saved["first_step"], saved["last_step"], saved["discard_slot_last"]) == (1, 3, True)

    resumed = MoEMetricsCallback(topk=2, exclude_last_slot=True)
    host = _Host([resumed])
    host._restore_trainer_sidecars(str(tmp_path / "checkpoint-3"))
    expected += _drive(resumed, range(4, 6))
    record = host.save(str(tmp_path / "checkpoint-5"), 5)
    assert record["counts"] == expected.long().tolist()
    assert (record["first_step"], record["last_step"]) == (1, 5)


def test_a_resume_without_the_file_counts_from_the_resume(tmp_path):
    os.makedirs(tmp_path / "checkpoint-6")
    callback = MoEMetricsCallback(topk=2)
    host = _Host([callback])
    host._restore_trainer_sidecars(str(tmp_path / "checkpoint-6"))
    expected = _drive(callback, range(7, 9))
    record = host.save(str(tmp_path / "checkpoint-8"), 8)
    assert record["counts"] == expected.long().tolist()
    assert (record["first_step"], record["last_step"]) == (7, 8)


def test_a_run_that_counted_nothing_writes_no_file(tmp_path):
    host = _Host([MoEMetricsCallback(topk=2)])
    host._persist_trainer_sidecars(str(tmp_path))
    assert not os.path.exists(tmp_path / MOE_EXPERT_TOKEN_COUNTS_FILE)


def _write_record(directory: str, format_version: int) -> None:
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, MOE_EXPERT_TOKEN_COUNTS_FILE), "w") as handle:
        json.dump({"format_version": format_version, "first_step": 1, "last_step": 2, "counts": [[1, 2]]}, handle)


def test_a_record_of_another_format_is_refused(tmp_path):
    _write_record(str(tmp_path), 2)
    with pytest.raises(ValueError, match="format_version 2"):
        read_run_counts(str(tmp_path / MOE_EXPERT_TOKEN_COUNTS_FILE))


def _resume_from_this_nodes_copy(rank: int, root: str) -> None:
    """Each rank reads its own node's copy, and only rank 1's is of another format."""
    callback = MoEMetricsCallback(topk=2)
    with pytest.raises(RuntimeError, match=f"delete {MOE_EXPERT_TOKEN_COUNTS_FILE}") as refused:
        _Host([callback])._restore_trainer_sidecars(os.path.join(root, f"node{rank}"))
    assert ("format_version 2" in str(refused.value.__cause__)) == (rank == 1)
    assert callback._resumed is None


def test_a_record_one_node_cannot_read_fails_every_rank(tmp_path):
    _write_record(str(tmp_path / "node0"), 1)
    _write_record(str(tmp_path / "node1"), 2)
    run_gloo_ranks(_resume_from_this_nodes_copy, 2, str(tmp_path))


def test_two_expert_load_callbacks_are_refused(tmp_path):
    host = _Host([MoEMetricsCallback(topk=2), BalancingBiasCallback()])
    with pytest.raises(ValueError, match="a run keeps one"):
        host._persist_trainer_sidecars(str(tmp_path))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
