"""Per-batch token accounting and attention-score work. Nothing syncs the host per micro-batch: counts sum over the
world in one all-reduce per log, and the attention work is read once per optimizer step."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import torch

from halo.kernels.logprobs import LABEL_IGNORE_INDEX
from halo.models.attention.tree_plan import TREE_ATTENTION_KEY
from halo.runtime.ranks import is_global_main_process
from halo.trainers.world_metrics import world_sums

logger = logging.getLogger(__name__)


def effective_loss_mask(masks: Mapping[str, torch.Tensor | Any]) -> torch.Tensor | None:
    """``completion_mask ∧ tool_mask`` where a ``tool_mask`` exists, else ``completion_mask``, which alone still counts
    tool-output tokens the loss never trains on; ``None`` without one."""
    completion_mask = masks.get("completion_mask")
    if completion_mask is None:
        return None
    tool_mask = masks.get("tool_mask")
    return completion_mask if tool_mask is None else completion_mask * tool_mask


class TokenMetricsMixin:
    # Set by the first ``training_step``: the efficiency callback binds here and hands over the layout.
    _attention_work_bound: bool = False
    _attention_layout = None
    # ``(model, non-embedding parameter count)`` for :meth:`floating_point_ops`, set on its first call.
    _flops_parameter_count: tuple | None = None
    # This rank's input tokens not yet summed into ``state.num_input_tokens_seen`` (HF's ``"all"`` mode).
    _unsummed_input_tokens: int = 0

    def _track_num_input_tokens(self, inputs) -> None:
        """In HF's ``"all"`` mode the count is a shape, so each micro-batch adds it on the host and its world sum lands
        where it is read (each log, save and measured step); HF would gather and read it on every micro-batch."""
        if self.args.include_num_input_tokens_seen != "all":
            super()._track_num_input_tokens(inputs)
            return
        main_input = getattr(self.model, "main_input_name", "input_ids")
        if main_input in inputs:
            self._unsummed_input_tokens += inputs[main_input].numel()

    def _sum_input_tokens(self) -> None:
        """COLLECTIVE in HF's ``"all"`` mode."""
        if self.args.include_num_input_tokens_seen == "all":
            (tokens,) = world_sums([self._unsummed_input_tokens])
            self.state.num_input_tokens_seen += int(tokens)
            self._unsummed_input_tokens = 0

    def store_flos(self) -> None:
        """COLLECTIVE off a logged step. A logged step's one all-reduce carries the count
        (:meth:`_drain_token_counts`); a save and the end of training sum it here, before ``trainer_state.json`` is
        written."""
        if self.control.should_log:
            return
        (flos,) = world_sums([self.current_flos])
        self.state.total_flos += flos
        self.current_flos = 0

    def floating_point_ops(self, inputs) -> int:
        """HF's estimate, ``6 × main-input tokens × non-embedding parameters``, with the parameter count walked once
        per model."""
        main_input = getattr(self.model, "main_input_name", "input_ids")
        if main_input not in inputs or not hasattr(self.model, "num_parameters"):
            return 0
        if self._flops_parameter_count is None or self._flops_parameter_count[0] is not self.model:
            self._flops_parameter_count = (self.model, self.model.num_parameters(exclude_embeddings=True))
        return 6 * inputs[main_input].numel() * self._flops_parameter_count[1]

    def _extract_document_lengths(self, inputs):
        """Zeros allowed; ``None`` without ``input_ids``. Reads ``cu_seq_lens_q``, the ``position_ids`` resets of
        packed rows, else one document per row. No host sync."""
        if not isinstance(inputs, Mapping):
            return None
        cu_seq_lens = inputs.get("cu_seq_lens_q")
        if cu_seq_lens is not None:
            return (cu_seq_lens[1:] - cu_seq_lens[:-1]).long()
        input_ids = inputs.get("input_ids")
        if input_ids is None:
            return None
        attention_mask = inputs.get("attention_mask")
        valid = attention_mask.bool() if attention_mask is not None else torch.ones_like(input_ids, dtype=torch.bool)
        position_ids = inputs.get("position_ids")
        if position_ids is None or position_ids.shape != input_ids.shape:
            return valid.sum(dim=1)
        # A document starts where the positions reset and at every row's first valid token, so rows
        # never merge; a running count of starts over the flattened batch numbers the documents.
        starts = ((position_ids == 0) | (valid.cumsum(dim=1) == 1)) & valid
        document = torch.cumsum(starts.flatten().long(), dim=0)
        lengths = torch.zeros(document.numel() + 1, dtype=torch.long, device=document.device)
        lengths.scatter_add_(0, document, valid.flatten().long())
        return lengths

    def _tree_node_lengths(self, inputs):
        """``(path length, own tokens)`` of every node of a tree-packed batch, else ``None``; a node attends along its
        root-to-leaf path, its last token's position plus one."""
        plan = inputs.get(TREE_ATTENTION_KEY) if isinstance(inputs, Mapping) else None
        if plan is None:
            return None
        bounds = plan["cu_seqlens_q"].long()
        return inputs["position_ids"][0, bounds[1:] - 1] + 1, bounds[1:] - bounds[:-1]

    def _accumulate_attention_flops(self, inputs):
        """Read once per optimizer step by :meth:`drain_step_work`."""
        if not self._attention_work_bound:
            # Offered to every callback that consumes measured attention work (duck-typed, so the trainer
            # package imports nothing from the callbacks); one answers with the layout it costs batches with.
            self._attention_work_bound = True
            for callback in self.callback_handler.callbacks:
                bind = getattr(callback, "bind_attention_work_source", None)
                if bind is not None:
                    self._attention_layout = bind(self)
        if self._attention_layout is None:
            return
        nodes = self._tree_node_lengths(inputs)
        lengths, own_tokens = nodes if nodes is not None else (self._extract_document_lengths(inputs), None)
        if lengths is None:
            return
        lengths = lengths.detach()
        flops = self._attention_layout.flops_for_documents(lengths, own_tokens)
        step = torch.stack((flops, (lengths > 0).sum().to(flops)))
        prev = getattr(self, "_local_attention_work_accum", None)
        self._local_attention_work_accum = step if prev is None else prev + step

    def drain_step_work(self) -> tuple[float, int]:
        """COLLECTIVE in HF's ``"all"`` mode, once per optimizer step. Rank-local work, as each pipeline stage costs
        documents through its own layer slice."""
        self._sum_input_tokens()
        accum = getattr(self, "_local_attention_work_accum", None)
        if accum is None:
            return 0.0, 0
        self._local_attention_work_accum = None
        flops, documents = accum.tolist()
        return flops, int(documents)

    def _extract_output_token_count(self, inputs):
        """Per-rank count of loss tokens; ``None`` for sequence-level and preference trainers, whose batches carry no
        token-level loss mask."""
        if not isinstance(inputs, Mapping):  # dict or transformers BatchEncoding (UserDict)
            return None
        labels, input_ids = inputs.get("labels"), inputs.get("input_ids")
        if labels is not None and input_ids is not None and labels.shape == input_ids.shape:
            return (labels != LABEL_IGNORE_INDEX).sum()
        loss_mask = effective_loss_mask(inputs)
        if loss_mask is None:
            loss_mask = inputs.get("completion_attention_mask")
        return loss_mask.sum() if loss_mask is not None else None

    def _accumulate_unmasked_output_tokens(self, local_count):
        if local_count is None:
            return
        # int64: a float accumulator would round a long window's count.
        local = local_count.detach().long()
        prev = getattr(self, "_local_unmasked_token_accum", None)
        self._local_unmasked_token_accum = local if prev is None else prev + local

    def _drain_token_counts(self):
        """COLLECTIVE. One all-reduce read back once, and all three reset. The loss-token total stays unset until some
        rank contributes, so sequence-level trainers log nothing."""
        accum = getattr(self, "_local_unmasked_token_accum", None)
        input_tokens, output_tokens, flos = world_sums(
            [self._unsummed_input_tokens, 0 if accum is None else accum, self.current_flos]
        )
        self.state.num_input_tokens_seen += int(input_tokens)
        self.state.total_flos += flos
        self._unsummed_input_tokens = 0
        self.current_flos = 0
        self._local_unmasked_token_accum = None
        output_tokens = int(output_tokens) // self.parallelism_config.non_dp_replication_factor
        previous = getattr(self, "_cumulative_unmasked_output_tokens", None)
        if previous is not None or output_tokens > 0:
            self._cumulative_unmasked_output_tokens = (previous or 0) + output_tokens

    def _warn_once_on_empty_labels(self, inputs: dict) -> None:
        """Warn once if a batch has no training signal (every label -100, pad or eos). First optimizer step only: the
        collator config is static, and ``.all()`` is a host sync."""
        if self._warned_empty_labels:
            return
        if getattr(self, "state", None) is not None and self.state.global_step > 0:
            return
        labels = inputs.get("labels")
        if labels is None:
            return
        non_training = labels == LABEL_IGNORE_INDEX
        tokenizer = getattr(self, "processing_class", None)
        for attr in ("pad_token_id", "eos_token_id"):
            token_id = getattr(tokenizer, attr, None)
            if token_id is not None:
                non_training = non_training | (labels == token_id)
        if non_training.all():
            self._warned_empty_labels = True
            if is_global_main_process():
                logger.warning(
                    "Batch contains no training signal — all labels are -100, pad, or eos. "
                    "This typically means the collator could not find the assistant_message_template "
                    "in the tokenized input, or the eval set is too small after packing. "
                    f"Labels shape: {list(labels.shape)}"
                )
