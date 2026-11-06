"""The tiny GLM-5 Next checkpoint carries the release's config keys that both serving engines read."""

import json

import pytest
from transformers import Glm5NextConfig, Glm5NextTextConfig

from tests.common.models import GLM5_ENGINE_GATED_DIMS, TINY_GLM5_CONFIG, TINY_GLM5_VISION_CONFIG

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


def test_the_kernel_gated_dims_are_the_releases():
    release = Glm5NextTextConfig()
    drifted = {
        key: (TINY_GLM5_CONFIG[key], getattr(release, key))
        for key in GLM5_ENGINE_GATED_DIMS
        if TINY_GLM5_CONFIG[key] != getattr(release, key)
    }
    assert not drifted, f"tiny dims differ from the release (tiny, release): {drifted}"


def test_the_kernel_gated_dims_meet_the_engine_limits():
    dims = TINY_GLM5_CONFIG
    # vLLM FLASHINFER_MLA_SPARSE, native no-rope MLA: the one shape flashinfer ships.
    assert Glm5NextTextConfig(**dims).qk_rope_head_dim == 0
    assert (dims["qk_nope_head_dim"], dims["kv_lora_rank"]) == (256, 512)
    # SGLang's DSA index cache asserts a 128-wide index head; the kpool must tile its 64-token page and the top-k.
    assert dims["index_head_dim"] == 128
    assert 64 % dims["index_kpool"] == 0 and dims["index_topk"] % dims["index_kpool"] == 0
    assert 0 < dims["index_topk"] <= 2048
    # DeepGEMM's MQA logits take 32 or 64 indexer heads (vLLM zero-pads fewer).
    assert dims["index_n_heads"] in (32, 64)
    # FlashInfer's TRT-LLM sparse MLA kernel and FlashMLA take 16, 32, 64 or 128 heads.
    assert dims["num_attention_heads"] in (16, 32, 64, 128)
    assert dims["hidden_size"] % 256 == 0
