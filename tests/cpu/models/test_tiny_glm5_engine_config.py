"""The tiny GLM-5 Next checkpoint carries the release's config keys that both serving engines read."""

import json

import pytest
from transformers import Glm5NextConfig

from tests.common.models import TINY_GLM5_CONFIG, TINY_GLM5_VISION_CONFIG

pytestmark = pytest.mark.cpu

# Read by vLLM's glm5next and SGLang's glm5_next without a default, and not declared by transformers' class.
ENGINE_READ_RELEASE_KEYS = (
    "mhc",
    "index_kpool_compress",
    "mla_use_nope",
    "indexer_rope_interleave",
    "index_share_for_mtp_iteration",
    "num_nextn_predict_layers",
    "first_k_dense_replace",
    "scoring_func",
    "topk_method",
)


def _saved_text_config(tmp_path) -> dict:
    Glm5NextConfig(text_config=dict(TINY_GLM5_CONFIG), vision_config=dict(TINY_GLM5_VISION_CONFIG)).save_pretrained(
        tmp_path
    )
    return json.loads((tmp_path / "config.json").read_text())["text_config"]


def test_the_saved_config_holds_every_key_an_engine_reads(tmp_path):
    saved = _saved_text_config(tmp_path)
    missing = [key for key in ENGINE_READ_RELEASE_KEYS if key not in saved]
    assert not missing, f"engines raise AttributeError on {missing}"
    assert saved["mhc"] is True and saved["index_kpool_compress"] is True


def test_the_dense_prefix_count_matches_the_layer_types():
    leading = 0
    for kind in TINY_GLM5_CONFIG["mlp_layer_types"]:
        if kind != "dense":
            break
        leading += 1
    assert TINY_GLM5_CONFIG["first_k_dense_replace"] == leading
