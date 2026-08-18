"""TRL generation-buffer seam shared by the online and environmental GRPO trainers: the
accumulation-vs-generation cadence predicate, and the TP re-sync of the micro-batch drawn from the
buffer.
"""


class GRPOGenerationBufferMixin:
    """Generation-buffer helpers. Mixed in ahead of ``GRPOTrainer`` so ``super()`` reaches TRL's."""

    @property
    def _misaligned_accumulation(self) -> bool:
        """Whether ``gradient_accumulation_steps`` straddles the generation cadence.

        TRL refills the generation buffer every ``steps_per_generation * num_iterations`` steps; when
        the accumulation window does not divide that evenly, one optimizer step consumes micro-batches
        generated under two different policies, so TRL recomputes the old log-probs instead of reusing
        the detached ones. Single home for the predicate: the construction-time validators and the
        runtime recompute gates must agree, or a gate that never fires still looks enabled.
        """
        generate_every = self.args.steps_per_generation * self.num_iterations
        return self.args.gradient_accumulation_steps % generate_every != 0

    def _prepare_inputs(self, generation_batch):
        """Re-sync the selected micro-batch across the TP/ETP group after TRL's buffer shuffle.

        TRL shuffles the generation buffer with ``torch.randperm`` on the per-rank-seeded CPU
        generator (``set_seed(seed, device_specific=True)``), so TP siblings would buffer
        different row orders and all-reduce partial activations of different samples.
        """
        inputs = super()._prepare_inputs(generation_batch)
        return self._broadcast_tensors_from_tp_leader(inputs)
