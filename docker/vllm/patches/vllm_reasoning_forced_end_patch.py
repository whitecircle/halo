"""Let a reasoning parser name the sequence the thinking budget forces, apart from its end marker.

``ReasoningConfig.initialize_token_ids`` reads one string off the parser, ``reasoning_end_str``, and
uses it twice: as the natural end marker that stops the count and, unless ``--reasoning-config`` sets
``reasoning_end_str``, as the sequence forced when the budget runs out. A parser whose count ends on a
shorter marker than the transition it forces cannot express that. gpt-oss is one: its count ends at
any channel header, while a cut has to move the model into the final channel.

Fix: when the command line sets no forced string, adopt the parser class's
``reasoning_forced_end_str``; the natural end stays the parser's ``reasoning_end_str``. Parsers
without the attribute are untouched. Runs where the config is built (the API server); the engine
receives the resolved token ids with the config.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Stamped on the replacement so ``Dockerfile.vllm`` can assert the patch took.
PATCH_MARKER = "_halo_reasoning_forced_end"

_APPLIED = False


def apply() -> None:
    """Idempotently patch ``ReasoningConfig.initialize_token_ids`` to honour a parser's forced string."""
    global _APPLIED
    if _APPLIED:
        return

    from vllm.config import reasoning  # noqa: PLC0415 — vLLM-only lazy import

    config_cls = reasoning.ReasoningConfig
    original_initialize = config_cls.initialize_token_ids

    def initialize_token_ids(self, model_config):
        if self.reasoning_parser and not self.reasoning_end_str:
            parser_cls = reasoning.ReasoningParserManager.get_reasoning_parser(self.reasoning_parser)
            forced = getattr(parser_cls, "reasoning_forced_end_str", "")
            if forced:
                self.reasoning_end_str = forced
        return original_initialize(self, model_config)

    setattr(initialize_token_ids, PATCH_MARKER, True)
    config_cls.initialize_token_ids = initialize_token_ids
    _APPLIED = True
    logger.info("vllm_reasoning_forced_end_patch applied to %s", config_cls.__name__)
