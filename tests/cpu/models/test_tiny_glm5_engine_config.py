"""The tiny GLM-5 Next checkpoint carries the release's config keys that both serving engines read."""

import json

import pytest
from transformers import Glm5NextConfig, Glm5NextTextConfig, Glm5NextVisionConfig

from tests.common.models import (
    GLM5_ENGINE_GATED_DIMS,
    GLM5_ENGINE_GATED_VISION_DIMS,
    TINY_GLM5_CONFIG,
    TINY_GLM5_VISION_CONFIG,
)

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


def test_the_vision_dims_are_the_releases():
    release = Glm5NextVisionConfig()
    drifted = {
        key: (TINY_GLM5_VISION_CONFIG[key], getattr(release, key))
        for key in GLM5_ENGINE_GATED_VISION_DIMS
        if TINY_GLM5_VISION_CONFIG[key] != getattr(release, key)
    }
    assert not drifted, f"tiny vision dims differ from the release (tiny, release): {drifted}"
    assert TINY_GLM5_VISION_CONFIG["hidden_size"] // TINY_GLM5_VISION_CONFIG["num_heads"] == (
        release.hidden_size // release.num_heads
    )


def test_the_vision_dims_meet_the_engine_limits():
    vision = TINY_GLM5_VISION_CONFIG
    # The processors flatten a patch to 3 x 2 x 14 x 14 = 1176 values (vLLM's profiling reports 37,632,000 values for
    # 32,000 patches); both patch embeds view each row as (channels, temporal, patch, patch).
    channels = Glm5NextVisionConfig().in_channels
    assert channels * vision["temporal_patch_size"] * vision["patch_size"] ** 2 == 1176
    # The processors resize to a multiple of patch x merge, so the square canvas must be one.
    assert vision["image_size"] % (vision["patch_size"] * vision["spatial_merge_size"]) == 0
    # SGLang builds the vision rotary table with rotary_dim = head_dim // 2 and vLLM with a 0.5 partial factor, both
    # over a 64-wide head; the attention kernels take 64 or 128.
    assert vision["hidden_size"] // vision["num_heads"] == 64
    # The projector's output width is the text hidden size; the heads split across the engines' tensor parallelism.
    assert vision["out_hidden_size"] == TINY_GLM5_CONFIG["hidden_size"]
    assert vision["num_heads"] % 8 == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
