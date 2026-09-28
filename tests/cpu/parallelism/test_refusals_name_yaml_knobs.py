#!/usr/bin/env python
"""Parallelism refusals name the knobs a user sets and link the docs for the trade-offs behind them.

Run: python tests/cpu/parallelism/test_refusals_name_yaml_knobs.py
"""

import pytest

from tests.common.parallelism import make_parallelism_config
from tests.common.utils import REPO_ROOT, load_script_module

EP1_SHARDING_DOC = "agent-docs/parallelism/data-parallelism.md#ep1-expert-sharding"


def test_ep1_sharding_refusal_points_at_the_docs_instead_of_a_benchmark():
    """The trade-off figure lives in the owning doc; the refusal links it rather than restating a
    number that drifts from the table."""
    with pytest.raises(ValueError) as err:
        make_parallelism_config(fsdp_shard_ep1_experts=False, tp_size=2, world_size=8, gpus_per_node=8)
    message = str(err.value)
    assert EP1_SHARDING_DOC in message, message
    assert "%" not in message, message
    page, anchor = EP1_SHARDING_DOC.split("#")
    check_links = load_script_module("scripts/docs/check_links.py")
    assert anchor in check_links.parse_markdown((REPO_ROOT / page).read_text()).anchors


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
