"""Train-dataloader geometry of the on-policy GRPO trainers.

:class:`GRPOTrainDataLoaderMixin` sizes the shared ``DataParallelDataLoaderMixin`` loader to the
GRPO train batch (``steps_per_generation`` micro-batches per generation round) and rebuilds the
sampler to match.
"""

from __future__ import annotations

from torch.utils.data import Dataset, Sampler
from trl.trainer.utils import RepeatSampler


class GRPOTrainDataLoaderMixin:
    """GRPO's train-loader geometry. Mixed in ahead of ``DistributedTrainerMixin``."""

    def _train_loader_batch_size(self) -> int:
        """One generation round per fetch: ``steps_per_generation`` micro-batches are drawn together."""
        return self._train_batch_size * self.args.steps_per_generation

    def _get_train_sampler(self, dataset: Dataset | None = None) -> Sampler:
        """``RepeatSampler`` sized to the custom DP-sharded dataloader's consumption rate.

        TRL derives the sampler geometry from ``generation_batch_size`` (world-rate: one full prompt
        block consumed per generation round across ``accelerator.num_processes``). The custom
        dataloader consumes only ``data_parallel_size`` batches per fetch (TP/ETP siblings replay the
        same DP slice), so under the world-rate geometry each generation round re-rolls only the
        leading ``dp/world`` fraction of every prompt block. Rebuild the sampler from the loader's
        per-round consumption; a pre-sharded dataset is consumed per DP rank, so the rate is one rank's.
        """
        if not self._needs_custom_dataloader():
            return super()._get_train_sampler(dataset)
        if dataset is None:
            dataset = self.train_dataset
        consumers = 1 if self._dataset_presharded else self.get_data_parallel_size()
        per_rank_rows = self._train_loader_batch_size()
        completions_per_round = per_rank_rows * consumers
        if completions_per_round % self.num_generations != 0:
            raise ValueError(
                f"The DP-rate generation batch ({completions_per_round} = per_device_train_batch_size "
                f"({self._train_batch_size}) * steps_per_generation ({self.args.steps_per_generation}) "
                f"* {'1 (pre-sharded per DP rank)' if self._dataset_presharded else f'data_parallel_size ({consumers})'}) "
                f"must be divisible by num_generations ({self.num_generations}), so every generation "
                f"round covers whole prompt groups."
            )
        if self.accelerator.num_processes != consumers and per_rank_rows % self.num_generations != 0:
            # TP/ETP siblings duplicate their DP slice into TRL's world-order gather; only whole groups regroup.
            raise ValueError(
                f"With TP/ETP active, the per-rank generation rows (per_device_train_batch_size "
                f"({self._train_batch_size}) * steps_per_generation ({self.args.steps_per_generation}) "
                f"= {per_rank_rows}) must be divisible by num_generations ({self.num_generations}): "
                f"a prompt group spanning DP ranks breaks TRL's world-gathered advantage grouping "
                f"once TP siblings inject duplicate blocks."
            )
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=completions_per_round // self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )
