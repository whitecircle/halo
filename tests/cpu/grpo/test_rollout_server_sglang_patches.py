#!/usr/bin/env python
"""The SGLang image patch must rewrite exactly the upstream code it names, and refuse anything else.

``docker/sglang/patches/patch_sglang_weight_updates.py`` runs at image build against the pinned
sglang; the fixtures below are the verbatim upstream code of that pin, so the build's pre-image
assertions are exercised here without the image. What is pinned:

  * the GLM gate rewrite leaves no fp32 cache and reads the parameter live;
  * the Gemma 4 router rewrite gives ``scale`` a loader that releases the fold latch;
  * a file without the pre-image, or with it twice, fails the build rather than patching around it;
  * ``--verify`` refuses the unpatched file.

    python tests/cpu/grpo/test_rollout_server_sglang_patches.py
"""

import pytest

from tests.common.utils import load_script_module

patches = load_script_module("docker/sglang/patches/patch_sglang_weight_updates.py")

# sglang 0.5.17 srt/models/glm4_moe_lite.py, class Glm4MoeLiteGate (the glm4_moe.py gate differs
# only by an ``is_nextn`` argument the rewrite does not touch).
GLM_GATE = """class Glm4MoeLiteGate(nn.Module):
    def __init__(
        self,
        config,
        prefix: str = "",
        is_nextn: bool = False,
    ):
        super().__init__()
        self.is_nextn = is_nextn
        self.weight = nn.Parameter(
            torch.empty((config.n_routed_experts, config.hidden_size))
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.empty((config.n_routed_experts), dtype=torch.float32)
        )
        # GLM requires FP32 gate projection; cache to avoid per-forward cast.
        # FIXME: if gate weight is updated at runtime (e.g. expert rebalancing), _weight_fp32 must be invalidated.
        self.register_buffer("_weight_fp32", None, persistent=False)

    def forward(self, hidden_states):
        if self._weight_fp32 is None:
            self._weight_fp32 = self.weight.data.to(torch.float32)
        logits = F.linear(hidden_states.to(torch.float32), self._weight_fp32, None)
        return logits
"""

# sglang 0.5.17 srt/models/gemma4_causal.py, the Gemma4Router lines the rewrite touches.
GEMMA4_ROUTER = '''from sglang.srt.model_loader.weight_utils import default_weight_loader


class Gemma4Router(nn.Module):
    def __init__(self, config, prefix: str = ""):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.norm = Gemma4RMSNorm(
            self.hidden_size, eps=config.rms_norm_eps, with_scale=False
        )
        # Per-dimension learned scale, applied after norm + root_size
        self.scale = nn.Parameter(torch.ones(self.hidden_size))
        self._scale_fused = False

    def fuse_scale(self):
        """Fold scale * root_size into norm.weight so forward needs no extra mul."""
        fused = (self.scale * self.root_size).to(self.norm.weight.dtype)
        self.norm.weight.data.copy_(fused)
        self._scale_fused = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._scale_fused:
            self.fuse_scale()
        x = self.norm(x)
        router_logits, _ = self.proj(x)
        return router_logits
'''


def test_glm_gate_reads_its_weight_live_after_the_rewrite():
    patched = patches.patch_glm_gate(GLM_GATE)
    assert "_weight_fp32" not in patched
    assert "dtype=torch.float32" in patched.split("e_score_correction_bias")[0], "the gate weight itself is fp32"
    assert "F.linear(hidden_states.to(torch.float32), self.weight, None)" in patched
    patches.verify_glm_gate(patched, "glm4_moe_lite.py")


def test_gemma4_scale_loader_releases_the_fold_latch():
    patched = patches.patch_gemma4_router(GEMMA4_ROUTER)
    assert "self.scale.weight_loader = self._load_scale" in patched
    loader = patched.split("def _load_scale")[1].split("def fuse_scale")[0]
    assert "default_weight_loader(param, loaded_weight)" in loader
    assert "self._scale_fused = False" in loader
    patches.verify_gemma4_router(patched, "gemma4_causal.py")


@pytest.mark.parametrize("patch", [patches.patch_glm_gate, patches.patch_gemma4_router])
def test_a_missing_pre_image_fails_the_build(patch):
    with pytest.raises(SystemExit, match="upstream changed"):
        patch("class Something(nn.Module):\n    pass\n")


def test_a_pre_image_seen_twice_fails_the_build():
    with pytest.raises(SystemExit, match="found 2"):
        patches.patch_glm_gate(GLM_GATE + GLM_GATE)


def test_an_already_patched_file_is_not_patched_twice():
    patched = patches.patch_glm_gate(GLM_GATE)
    with pytest.raises(SystemExit, match="upstream changed"):
        patches.patch_glm_gate(patched)


def test_verify_refuses_a_gate_whose_weight_was_left_in_the_default_dtype():
    """Cache stripped and forward rewritten but the parameter not fp32: ``F.linear`` would mix dtypes."""
    half_patched = patches._replace_once(GLM_GATE, patches._GLM_CACHE_BEFORE, "", "cache")
    half_patched = patches._replace_once(half_patched, patches._GLM_FORWARD_BEFORE, patches._GLM_FORWARD_AFTER, "fwd")
    with pytest.raises(SystemExit, match="hold its weight in fp32"):
        patches.verify_glm_gate(half_patched, "glm4_moe_lite.py")


def test_a_gemma4_file_without_the_scale_line_fails_the_build():
    with pytest.raises(SystemExit, match="upstream changed"):
        patches.patch_gemma4_router(GEMMA4_ROUTER.replace(patches._GEMMA4_SCALE_BEFORE, ""))


def test_verify_refuses_the_unpatched_files():
    with pytest.raises(SystemExit, match="still caches"):
        patches.verify_glm_gate(GLM_GATE, "glm4_moe.py")
    with pytest.raises(SystemExit, match="fold latch"):
        patches.verify_gemma4_router(GEMMA4_ROUTER, "gemma4_causal.py")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
