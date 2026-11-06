"""Packed micro-batches of the environmental trainer: the shared round layout with its routing-mask filler, the
routing-replay pause around inert rows and its construction gates."""

import contextlib
from contextlib import AbstractContextManager
from typing import ClassVar

from halo.trainers.grpo.environmental.step_batch import ROUTING_MASKS_KEY, ROUTING_ROWS_KEY
from halo.trainers.grpo.packed_rounds import PackedRoundMixin


class PackedMicrobatchMixin(PackedRoundMixin):
    """Mixed in ahead of ``LengthOrderedRowsMixin`` (whose deal it replaces) and ``GenerationRoundMixin``. The host
    calls :meth:`_configure_packed_microbatches` once the model is placed, plans the round with :meth:`_plan_round` and
    lays it out with :meth:`_lay_out_packed_round` from the train round's batch build."""

    PACKED_FILLER_VALUES: ClassVar[dict[str, int]] = {
        **PackedRoundMixin.PACKED_FILLER_VALUES,
        ROUTING_MASKS_KEY: -1,
        ROUTING_ROWS_KEY: -1,
    }

    @property
    def _packed_microbatch_tokens(self) -> int | None:
        return self.async_config.packed_microbatch_tokens

    def _configure_packed_microbatches(self) -> None:
        super()._configure_packed_microbatches(length_ordered=self.async_config.length_ordered_microbatches)
        if self._packs_microbatches:
            self._orders_rows_by_length = False

    def _routing_replay_paused(self) -> AbstractContextManager[None]:
        return self._routing_injector.paused() if self._routing_injector is not None else contextlib.nullcontext()
