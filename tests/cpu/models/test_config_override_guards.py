"""``model_config_overrides`` must reject a key it cannot honour, not accept and discard it.

The gate already raises on a key no config level declares, on the reasoning that a skipped override
trains with the stock value. Precision is the case that slips through it: ``dtype`` (and its
deprecated ``torch_dtype`` alias) IS declared by every transformers config, so the existence check
passes — and then every loader hands ``from_pretrained`` an explicit ``dtype=`` resolved from
the run's own ``bf16``/``fp16`` flags, which wins. The override is accepted, logged as applied, and
has no effect: a config asking for fp32 master weights trains in bf16 with nothing to show for it.

The gate's general accept/reject behaviour — a declared key written, a composite key written at the
level that declares it, an undeclared key raising — is pinned across every model type in
``tests/cpu/models/test_composite_config_fields.py``; only the precision refusal lives here.

    python tests/cpu/models/test_config_override_guards.py
"""

from __future__ import annotations

import pytest
from accelerate import PartialState
from transformers import LlamaConfig

from src.distributed.loading.model_loading import _apply_config_overrides

PartialState()  # the loader logs through accelerate's logger

# Spelled out rather than read off the production tuple: parametrizing over the very collection under
# test turns shrinking it into a silently dropped test node instead of a failure. transformers 5
# renamed ``torch_dtype`` to ``dtype`` and keeps the old name as a deprecated alias, so a config
# declares both and both must be refused.
PRECISION_KEYS = ["dtype", "torch_dtype"]


@pytest.mark.parametrize("key", PRECISION_KEYS)
def test_a_precision_override_is_refused(key):
    config = LlamaConfig()
    assert hasattr(config, key), (
        f"premise of this guard: {key!r} IS declared, so the undeclared-key check passes it through "
        f"and the override reaches a config field the explicit dtype= then overrules"
    )
    with pytest.raises(ValueError, match="bf16"):
        _apply_config_overrides(config, {key: "float16"})


def test_a_refused_precision_key_leaves_the_config_untouched():
    """Rejected before anything is written: a half-applied override is worse than none, because the
    run would carry some of what the config asked for and none of the rest."""
    config = LlamaConfig()
    original = config.max_position_embeddings
    with pytest.raises(ValueError):
        _apply_config_overrides(config, {"max_position_embeddings": 12345, "dtype": "float16"})
    assert config.max_position_embeddings == original
    assert config.dtype is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
