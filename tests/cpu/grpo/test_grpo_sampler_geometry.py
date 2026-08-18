"""RepeatSampler geometry for the GRPO custom DP-sharded dataloader.

TRL sizes the train ``RepeatSampler`` from ``generation_batch_size`` (world-rate: one full prompt
block consumed per generation round across ``accelerator.num_processes``). The custom dataloader
(TP/ETP/pre-sharded) consumes only ``data_parallel_size`` batches per fetch, so under the
world-rate geometry each generation round re-rolls the leading ``dp/world`` fraction of every
prompt block and silently skips the rest. ``GRPOTrainDataLoaderMixin._get_train_sampler`` rebuilds
the sampler at the DP consumption rate; these tests assert the exact per-round index geometry.
"""

from types import SimpleNamespace

import pytest
from trl.trainer.utils import RepeatSampler

from src.trainers.grpo.mixins.dataloader import GRPOTrainDataLoaderMixin


class _TrlWorldRateBase:
    """Mirror of ``trl.GRPOTrainer._get_train_sampler`` (world-rate geometry)."""

    def _get_train_sampler(self, dataset=None):
        if dataset is None:
            dataset = self.train_dataset
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )


class _Trainer(GRPOTrainDataLoaderMixin, _TrlWorldRateBase):
    """Minimal composition exposing exactly the attributes the mixin reads."""

    def __init__(
        self,
        *,
        world_size,
        dp_size,
        num_generations,
        steps_per_generation,
        num_iterations,
        per_device_batch,
        dataset_size,
        presharded=False,
        custom=True,
    ):
        self.args = SimpleNamespace(
            steps_per_generation=steps_per_generation,
            generation_batch_size=per_device_batch * world_size * steps_per_generation,
            seed=42,
        )
        self._train_batch_size = per_device_batch
        self.accelerator = SimpleNamespace(num_processes=world_size)
        self.num_generations = num_generations
        self.num_iterations = num_iterations
        self.shuffle_dataset = False  # deterministic index geometry
        self.train_dataset = list(range(dataset_size))
        self._dataset_presharded = presharded
        self._dp_size = dp_size
        self._custom = custom

    def _needs_custom_dataloader(self):
        return self._custom

    def get_data_parallel_size(self):
        return self._dp_size


def _generated_prompts_per_round(sampler, *, loader_batch, shard_processes, consumer_ranks, generate_every):
    """Prompt sets each generation round actually trains on, in round order.

    Mirrors the runtime data path: accelerate's ``BatchSamplerShard`` hands per-rank fetch ``f`` on
    process ``p`` the underlying dataloader batch ``f * shard_processes + p``; TRL's
    ``_prepare_inputs`` generates only on fetches where ``step % generate_every == 0`` and discards
    the fetched content of every other step (buffer reuse). Stops at the first incomplete round.
    """
    stream = list(sampler)
    batches = [stream[i : i + loader_batch] for i in range(0, len(stream), loader_batch)]
    rounds = []
    for r in range(len(batches)):
        generated = set()
        for p in range(consumer_ranks):
            b = r * generate_every * shard_processes + p
            if b >= len(batches) or len(batches[b]) < loader_batch:
                return rounds
            generated.update(batches[b])
        rounds.append(generated)
    return rounds


@pytest.mark.parametrize("num_iterations", [1, 2])
def test_dp_geometry_covers_every_prompt_once_per_epoch(num_iterations):
    """world=8, tp=2 -> dp=4, G=4, spg=4: every prompt generated exactly once, in order."""
    trainer = _Trainer(
        world_size=8,
        dp_size=4,
        num_generations=4,
        steps_per_generation=4,
        num_iterations=num_iterations,
        per_device_batch=1,
        dataset_size=32,
    )
    sampler = trainer._get_train_sampler()
    rounds = _generated_prompts_per_round(
        sampler,
        loader_batch=4,  # _train_batch_size * steps_per_generation
        shard_processes=4,  # _prepare_dataloader shards over dp_size
        consumer_ranks=4,
        generate_every=4 * num_iterations,  # steps_per_generation * num_iterations
    )
    # 8 rounds of 4 unique prompts each: full coverage, no re-rolls, no skips. Under TRL's
    # world-rate geometry this is [{0..3}, {0..3}, {8..11}, {8..11}, ...] instead.
    assert rounds == [set(range(4 * r, 4 * r + 4)) for r in range(8)]


def test_world_rate_geometry_rerolls_and_skips_under_dp_consumption():
    """What the DP geometry prevents: a world-rate sampler consumed at DP rate skips half the data."""
    trainer = _Trainer(
        world_size=8,
        dp_size=4,
        num_generations=4,
        steps_per_generation=4,
        num_iterations=1,
        per_device_batch=1,
        dataset_size=32,
    )
    sampler = _TrlWorldRateBase._get_train_sampler(trainer)
    rounds = _generated_prompts_per_round(
        sampler, loader_batch=4, shard_processes=4, consumer_ranks=4, generate_every=4
    )
    # Each 8-prompt block: first half re-rolled every round of its window, second half never seen.
    assert rounds == [set(range(8 * (r // 2), 8 * (r // 2) + 4)) for r in range(8)]
    assert set().union(*rounds).isdisjoint({4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31})


def test_stock_path_keeps_trl_world_geometry():
    """Without the custom dataloader (EP-only / plain DP) the TRL sampler is used unchanged."""
    trainer = _Trainer(
        world_size=8,
        dp_size=8,
        num_generations=4,
        steps_per_generation=4,
        num_iterations=1,
        per_device_batch=1,
        dataset_size=32,
        custom=False,
    )
    sampler = trainer._get_train_sampler()
    assert (sampler.mini_repeat_count, sampler.batch_size, sampler.repeat_count) == (4, 8, 4)
    rounds = _generated_prompts_per_round(
        sampler, loader_batch=4, shard_processes=8, consumer_ranks=8, generate_every=4
    )
    assert rounds == [set(range(8 * r, 8 * r + 8)) for r in range(4)]


def test_presharded_geometry_is_per_rank():
    """Pre-sharded: each DP rank iterates its own shard, so the consumption rate is one rank's."""
    trainer = _Trainer(
        world_size=8,
        dp_size=4,
        num_generations=4,
        steps_per_generation=2,
        num_iterations=1,
        per_device_batch=2,
        dataset_size=8,  # this rank's shard
        presharded=True,
    )
    sampler = trainer._get_train_sampler()
    assert (sampler.mini_repeat_count, sampler.batch_size, sampler.repeat_count) == (4, 1, 2)
    rounds = _generated_prompts_per_round(
        sampler, loader_batch=4, shard_processes=1, consumer_ranks=1, generate_every=2
    )
    assert rounds == [{r} for r in range(8)]


def test_indivisible_dp_generation_batch_raises():
    """Fail loud when the DP-rate round cannot hold whole prompt groups."""
    trainer = _Trainer(
        world_size=8,
        dp_size=4,
        num_generations=8,
        steps_per_generation=4,
        num_iterations=1,
        per_device_batch=1,
        dataset_size=32,
        presharded=True,  # per-rank rate: 1 * 4 = 4, not divisible by 8
    )
    with pytest.raises(ValueError, match="divisible by num_generations"):
        trainer._get_train_sampler()


def test_tp_group_spanning_dp_ranks_raises():
    """world=8, tp=2, G=8, tb*spg=4: the DP-rate gate alone passes (16 % 8 == 0) but a prompt
    group spans two DP ranks — TRL's world-order reward gather would then regroup TP-duplicated
    blocks and silently corrupt the advantages. Must fail loud."""
    trainer = _Trainer(
        world_size=8,
        dp_size=4,
        num_generations=8,
        steps_per_generation=4,
        num_iterations=1,
        per_device_batch=1,
        dataset_size=32,
    )
    with pytest.raises(ValueError, match="spanning DP ranks"):
        trainer._get_train_sampler()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
