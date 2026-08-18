"""Keep structural (special) tokens inside the policy-gradient set under ``top_entropy_quantile``.

``top_entropy_quantile < 1.0`` trains only the highest-entropy completion tokens, but structural
tokens (role/channel markers, tool delimiters, BOS/EOS) are the lowest-entropy ones and would never
be reinforced — over a long run the template degrades and tool calls stop parsing. This unions the
special-token positions back into the entropy mask. Token set read from the tokenizer
(``all_special_ids`` + added tokens), so it is model-agnostic.
"""

import logging

import torch

from src.models.structure import resolve_tokenizer

logger = logging.getLogger(__name__)


class ProtectedTokenEntropyMixin:
    """Union special-token positions into ``top_entropy_quantile``'s high-entropy mask.

    Place BEFORE the trainer's other bases so it wins the MRO. No-op when
    ``top_entropy_quantile == 1.0`` (TRL never calls ``get_high_entropy_mask``).
    """

    def _protected_token_ids(self) -> torch.Tensor | None:
        """Special/added token ids for this tokenizer, cached. ``None`` when the tokenizer exposes none."""
        cached = getattr(self, "_protected_ids_cache", None)
        if cached is not None:
            # An EMPTY cache is "resolved, nothing to protect" — cached like any other answer so the
            # warning below is emitted once per run instead of once per micro-batch for the whole run.
            return cached if cached.numel() else None

        tokenizer = resolve_tokenizer(self.processing_class)
        ids: set[int] = set(tokenizer.all_special_ids or ())
        # Added tokens carry template scaffolding `all_special_ids` misses (ChatML/harmony, tool delimiters).
        added = getattr(tokenizer, "get_added_vocab", None)
        if callable(added):
            ids.update(added().values())
        self._protected_ids_cache = torch.tensor(sorted(ids), dtype=torch.long)
        if not ids:
            logger.warning("No special/added tokens found on the tokenizer; entropy mask left unprotected.")
            return None

        logger.info(f"Entropy mask protects {len(ids)} special/added tokens (always policy-trained).")
        return self._protected_ids_cache

    def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask, logits_to_keep, *args, **kwargs):
        # Stash completion ids for get_high_entropy_mask, called on the SAME micro-batch by _compute_loss.
        self._entropy_completion_ids = input_ids[:, -logits_to_keep:] if logits_to_keep else None
        return super()._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep, *args, **kwargs
        )

    def get_high_entropy_mask(self, entropies: torch.Tensor, mask: torch.Tensor, threshold: float) -> torch.Tensor:
        high_entropy = super().get_high_entropy_mask(entropies, mask, threshold)

        completion_ids = getattr(self, "_entropy_completion_ids", None)
        protected_ids = self._protected_token_ids()
        if completion_ids is None or protected_ids is None or completion_ids.shape != high_entropy.shape:
            return high_entropy

        if protected_ids.device != completion_ids.device:
            protected_ids = protected_ids.to(completion_ids.device)
            self._protected_ids_cache = protected_ids
        is_protected = torch.isin(completion_ids, protected_ids)
        return (high_entropy | is_protected) & mask.bool()
