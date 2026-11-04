"""``model_config_overrides`` must reject a key it cannot honour, not accept and discard it.
Every config declares ``dtype``, and each loader's explicit ``dtype=`` wins, so an fp32 request trains in bf16."""

from __future__ import annotations

import pytest
from transformers import LlamaConfig

from halo.models.config_access import apply_config_overrides

# Spelled out: parametrizing over the production tuple turns shrinking it into a silently dropped test node.
# transformers 5 renamed ``torch_dtype`` to ``dtype`` and keeps the old name as an alias, so both must be refused.
PRECISION_KEYS = ["dtype", "torch_dtype"]


@pytest.mark.parametrize("key", PRECISION_KEYS)
def test_a_precision_override_is_refused(key):
    config = LlamaConfig()
    assert hasattr(config, key), (
        f"premise of this guard: {key!r} IS declared, so the undeclared-key check passes it through "
        f"and the override reaches a config field the explicit dtype= then overrules"
    )
    with pytest.raises(ValueError, match="bf16"):
        apply_config_overrides(config, {key: "float16"})


def test_a_refused_precision_key_leaves_the_config_untouched():
    """A half-applied override would carry some of what the config asked for and none of the rest."""
    config = LlamaConfig()
    original = config.max_position_embeddings
    with pytest.raises(ValueError):
        apply_config_overrides(config, {"max_position_embeddings": 12345, "dtype": "float16"})
    assert config.max_position_embeddings == original
    assert config.dtype is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
