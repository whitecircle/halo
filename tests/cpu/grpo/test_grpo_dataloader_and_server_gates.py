#!/usr/bin/env python
"""GRPO train-dataloader column pruning + the vLLM server-mode construction gates.

- the GRPO train dataloader builds its dataset/collator pair through the shared
  ``DataParallelDataLoaderMixin._loader_params`` seam (sized by GRPO's ``_train_loader_batch_size``
  hook), so a train dataset that is NOT a ``datasets.Dataset`` still honours
  ``remove_unused_columns`` — via the collator, the base ``Trainer`` contract. A diverged copy that
  only prunes when the dataset exposes ``column_names`` silently trains such a dataset on every
  column and fails here.
- ``_require_vllm_server_mode`` raises instead of no-opping when no training config reaches the
  ctor, refuses every non-server shape before TRL's vLLM client is swapped in, and accepts the
  server shape.
- TRL's tool-calling loop (``tools``, ``environment_factory``) is refused by keyword or position: it
  regenerates while any completion on the rank calls a tool, one collective generate per round, so
  ranks whose completions stop calling tools at different rounds desync. TRL's ``rollout_func`` is
  refused under TP/ETP, whose siblings would forward their own rollouts instead of their leader's.

    python tests/cpu/grpo/test_grpo_dataloader_and_server_gates.py
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
from datasets import Dataset
from torch.utils.data import SequentialSampler

from src.trainers.grpo.online import _ROLLOUT_FUNC_CTOR_POSITIONS, _TOOL_LOOP_CTOR_POSITIONS, DistributedGRPOTrainer

_ROWS = [{"prompt": "p", "unused": i} for i in range(6)]


def _raw_collator(batch):
    return batch


def _pruned_collator(batch):
    return batch


def _train_loader(dataset, *, remove_unused_columns: bool):
    """Run the real GRPO train-dataloader path over ``dataset``; returns (loader, prune_calls).

    Only the HF/accelerate seams are stubbed (column pruning, sampler, accelerate prepare); the
    dataset/collator decision under test is the production one.
    """
    trainer = object.__new__(DistributedGRPOTrainer)
    trainer.parallelism_config = SimpleNamespace(
        is_tp_mode=True, is_cp_mode=False, is_expert_tp_mode=False, is_pp_mode=False, non_dp_replication_factor=2
    )
    trainer._dataset_presharded = False
    trainer.train_dataset = dataset
    trainer._train_batch_size = 2
    trainer.data_collator = _raw_collator
    trainer.args = SimpleNamespace(
        remove_unused_columns=remove_unused_columns,
        steps_per_generation=3,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=False,
        dataloader_drop_last=False,
        dataloader_prefetch_factor=None,
        dataloader_multiprocessing_context=None,
        dataloader_in_order=True,
    )

    calls = {"collator": [], "dataset": []}

    def _prune_collator(collator, description):
        calls["collator"].append(description)
        return _pruned_collator

    def _prune_dataset(ds, description=None):
        calls["dataset"].append(description)
        return ds.remove_columns(["unused"])

    trainer._get_collator_with_removed_columns = _prune_collator
    trainer._remove_unused_columns = _prune_dataset
    trainer._get_train_sampler = lambda: SequentialSampler(dataset)
    trainer._prepare_dataloader = lambda loader: loader
    trainer.get_data_parallel_rank = lambda: 0

    return DistributedGRPOTrainer.get_train_dataloader(trainer), calls


def test_non_datasets_train_dataset_prunes_through_the_collator():
    """A plain-list train dataset has no ``column_names``, so pruning must reach the COLLATOR.

    A diverged copy that passes ``self.data_collator`` through untouched silently drops
    ``remove_unused_columns`` for every non-``datasets.Dataset`` train dataset.
    """
    loader, calls = _train_loader(list(_ROWS), remove_unused_columns=True)

    assert loader.collate_fn is _pruned_collator, (
        "remove_unused_columns was lost: the raw collator reached the DataLoader for a "
        "non-datasets.Dataset train dataset"
    )
    assert calls["collator"] == ["training"], f"expected one 'training' collator prune, got {calls['collator']}"
    assert calls["dataset"] == [], "a plain list has no columns to prune off the dataset"


def test_datasets_dataset_prunes_the_dataset_not_the_collator():
    """Anti-vacuity for the branch above: a real ``datasets.Dataset`` keeps the base contract —
    columns come off the dataset and the collator is passed through untouched."""
    loader, calls = _train_loader(Dataset.from_list(_ROWS), remove_unused_columns=True)

    assert loader.collate_fn is _raw_collator, "a datasets.Dataset must not have its collator rewrapped"
    assert calls["dataset"] == ["training"], f"expected one 'training' dataset prune, got {calls['dataset']}"
    assert "unused" not in loader.dataset.column_names, "the unused column survived dataset pruning"


def test_train_batch_covers_a_whole_generation_round():
    """The GRPO batch is ``per_device_train_batch_size * steps_per_generation`` — the whole round's
    prompts are fetched at once, not one micro-batch."""
    loader, _ = _train_loader(list(_ROWS), remove_unused_columns=False)
    assert loader.batch_size == 6, f"expected 2 * 3 rows per fetch, got {loader.batch_size}"


def test_missing_training_config_raises():
    """``ctor_config`` resolved no config: silently returning would leave in-process HF generation
    running under FSDP2/EP (slow and rank-divergent) with nothing in the log to say the gate never ran."""
    with pytest.raises(ValueError, match="GRPOConfig"):
        DistributedGRPOTrainer._require_vllm_server_mode(None)


@pytest.mark.parametrize(
    ("use_vllm", "vllm_mode", "match"),
    [
        (False, "server", "use_vllm=True"),
        (True, "colocate", "vllm_mode='server'"),
    ],
)
def test_non_server_configs_are_refused(use_vllm, vllm_mode, match):
    args = SimpleNamespace(use_vllm=use_vllm, vllm_mode=vllm_mode)
    with pytest.raises(ValueError, match=match):
        DistributedGRPOTrainer._require_vllm_server_mode(args)


def test_the_server_config_is_accepted():
    """Anti-over-rejection: the one shape the gate exists to let through."""
    DistributedGRPOTrainer._require_vllm_server_mode(SimpleNamespace(use_vllm=True, vllm_mode="server"))


def _calculator(expression: str) -> str:
    """A tool TRL would hand the policy."""
    return expression


@pytest.mark.parametrize("name", sorted(_TOOL_LOOP_CTOR_POSITIONS))
def test_the_tool_calling_loop_is_refused_by_keyword_and_by_position(name):
    with pytest.raises(ValueError, match=f"tool-calling loop .{name}."):
        DistributedGRPOTrainer._reject_tool_calling_loop((), {name: [_calculator]})
    positional = [None] * (_TOOL_LOOP_CTOR_POSITIONS[name] + 1)
    positional[_TOOL_LOOP_CTOR_POSITIONS[name]] = [_calculator]
    with pytest.raises(ValueError, match="Async GRPO with Environments"):
        DistributedGRPOTrainer._reject_tool_calling_loop(tuple(positional), {})


def test_no_tools_passes():
    DistributedGRPOTrainer._reject_tool_calling_loop((), {"tools": None, "environment_factory": None})
    DistributedGRPOTrainer._reject_tool_calling_loop((), {"tools": []})


def test_the_ctor_refuses_tools_before_trl_builds_anything():
    """Wired into ``__init__`` ahead of TRL's ctor, which would open the NCCL group to the server."""
    config = SimpleNamespace(use_vllm=True, vllm_mode="server")
    with (
        mock.patch.object(DistributedGRPOTrainer, "_begin_on_policy_init", lambda self, a, k: (config, k)),
        mock.patch.object(DistributedGRPOTrainer, "_resolve_advantage_hooks", side_effect=AssertionError("too late")),
        pytest.raises(ValueError, match="tool-calling loop"),
    ):
        DistributedGRPOTrainer(model=None, args=config, tools=[_calculator])


def _rollout(prompts, trainer):
    """A custom rollout TRL would call in place of the toolkit's generation."""
    return {"prompt_ids": [], "completion_ids": [], "logprobs": []}


def _parallel(tp: bool = False, etp: bool = False) -> SimpleNamespace:
    return SimpleNamespace(is_tp_mode=tp, is_expert_tp_mode=etp)


@pytest.mark.parametrize("config", [_parallel(tp=True), _parallel(etp=True)], ids=["tp", "etp"])
def test_rollout_func_is_refused_where_siblings_share_a_replica(config):
    """It replaces the generation that hands TP/ETP siblings their leader's completions."""
    host = SimpleNamespace(parallelism_config=config)
    with pytest.raises(ValueError, match="rollout_func is not supported under tensor or expert-tensor"):
        DistributedGRPOTrainer._reject_unshared_rollout_func(host, (), {"rollout_func": _rollout})
    positional = [None] * (_ROLLOUT_FUNC_CTOR_POSITIONS["rollout_func"] + 1)
    positional[-1] = _rollout
    with pytest.raises(ValueError, match="rollout_func"):
        DistributedGRPOTrainer._reject_unshared_rollout_func(host, tuple(positional), {})


def test_rollout_func_passes_off_tp_and_absent_under_tp():
    DistributedGRPOTrainer._reject_unshared_rollout_func(
        SimpleNamespace(parallelism_config=_parallel()), (), {"rollout_func": _rollout}
    )
    DistributedGRPOTrainer._reject_unshared_rollout_func(
        SimpleNamespace(parallelism_config=_parallel(tp=True)), (), {}
    )


def test_the_ctor_refuses_rollout_func_under_tp_before_trl_builds_anything():
    config = SimpleNamespace(use_vllm=True, vllm_mode="server")

    def begin(self, args, kwargs):
        self.parallelism_config = _parallel(tp=True)
        return config, kwargs

    with (
        mock.patch.object(DistributedGRPOTrainer, "_begin_on_policy_init", begin),
        mock.patch.object(DistributedGRPOTrainer, "_resolve_advantage_hooks", side_effect=AssertionError("too late")),
        pytest.raises(ValueError, match="rollout_func is not supported"),
    ):
        DistributedGRPOTrainer(model=None, args=config, rollout_func=_rollout)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
