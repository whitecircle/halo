"""Packed micro-batches of the environmental trainer: the shared round layout with its routing-row and candidate fillers, the
routing-replay pause around inert rows and its construction gates."""

import contextlib
from contextlib import AbstractContextManager
from typing import ClassVar

from halo.environments.episodes.candidate_logprobs import NO_CANDIDATE_ID, NO_CANDIDATE_LOGP
from halo.trainers.grpo.environmental.step_batch import ROUTING_ROWS_KEY
from halo.trainers.grpo.objective.score_centering import TOPK_IDS_KEY, TOPK_LOGPS_KEY
from halo.trainers.grpo.rounds.packed_rounds import PackedRoundMixin


class PackedMicrobatchMixin(PackedRoundMixin):
    """Mixed in ahead of ``LengthOrderedRowsMixin`` (whose deal it replaces) and ``GenerationRoundMixin``. The host
    calls :meth:`_configure_packed_microbatches` once the model is placed, plans the round with :meth:`_plan_round` and
    lays it out with :meth:`_lay_out_packed_round` from the train round's batch build."""

    PACKED_FILLER_VALUES: ClassVar[dict[str, float]] = {
        **PackedRoundMixin.PACKED_FILLER_VALUES,
        ROUTING_ROWS_KEY: -1,
        TOPK_IDS_KEY: NO_CANDIDATE_ID,
        TOPK_LOGPS_KEY: NO_CANDIDATE_LOGP,
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
