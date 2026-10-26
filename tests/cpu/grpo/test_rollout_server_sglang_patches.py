#!/usr/bin/env python
"""The SGLang image's edit table must rewrite exactly the upstream code it names, and refuse anything else.

``docker/sglang/patches/patch_sglang.py`` runs at image build against the pinned sglang; the
fixtures below are the verbatim upstream code of that pin, so the build's pre-image assertions are
exercised here without the image. What is pinned:

  * the GLM gate, which upstream now holds in fp32 and reads live, is asserted and never rewritten,
    and a gate that caches its weight again fails the build;
  * the Gemma 4 router rewrite gives ``scale`` a loader that releases the fold latch, and the build
    fails once the loader it calls is no longer imported;
  * the prefill-loop rewrite steps the log-prob offset over a skipped request before ``continue``;
  * ``--verify`` refuses the unpatched file.

The replace-once and pre/post-image mechanics are pinned in ``test_engine_source_patch.py``.

    python tests/cpu/grpo/test_rollout_server_sglang_patches.py
"""

import pytest

from tests.common.repo import load_script_module

# Registered so the table's own ``from engine_source_patch import ...`` resolves, as in the image.
load_script_module("docker/engine_source_patch.py", register=True)
patches = load_script_module("docker/sglang/patches/patch_sglang.py")

# sglang 0.5.22 srt/models/glm4_moe_lite.py, class Glm4MoeLiteGate (the glm4_moe.py gate differs
# only by an ``is_nextn`` argument the check does not read).
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
            torch.empty(
                (config.n_routed_experts, config.hidden_size), dtype=torch.float32
            )
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.empty((config.n_routed_experts), dtype=torch.float32)
        )

    def forward(self, hidden_states):
        logits = F.linear(hidden_states.to(torch.float32), self.weight, None)
        return logits
"""

# The gate's weight declared in the default dtype, as sglang 0.5.17 shipped it.
_GLM_WEIGHT_DEFAULT_DTYPE = """        self.weight = nn.Parameter(
            torch.empty((config.n_routed_experts, config.hidden_size))
        )
"""

# The same gate as sglang 0.5.17 shipped it, caching an fp32 copy of the weight on its first forward.
GLM_GATE_CACHING = """class Glm4MoeLiteGate(nn.Module):
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

# sglang 0.5.22 srt/models/gemma4_causal.py, the Gemma4Router lines the rewrite touches.
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
        """Returns raw router logits [T, E]."""
        if not self._scale_fused:
            self.fuse_scale()
        x = self.norm(x)
        router_logits, _ = self.proj(x)
        return router_logits
'''

# sglang 0.5.22 srt/managers/scheduler_components/batch_result_processor.py, the skip at the head of
# the process_batch_result_prefill loop body (its hidden-state capture elided).
PREFILL_LOOP = """                if (
                    req.finished() and req.inflight_middle_chunks <= 0
                ) or req.is_retracted:
                    # Decode req in a mixed batch, or a retracted req. Keep an
                    # aborted middle chunk in the chunked branch long enough to
                    # drain its accounting without streaming it.
                    continue

                if req.inflight_middle_chunks <= 0:
                    req.time_stats.set_prefill_finished_time()
"""


def _patched(text, path):
    for edit in patches.EDITS:
        if edit.path == path:
            text = edit.apply(text)
    return text


def _verify(text, path):
    for edit in patches.EDITS:
        if edit.path == path:
            edit.verify(text)


def test_the_pinned_glm_gate_reads_its_weight_live():
    patches.verify_glm_gate(GLM_GATE, "glm4_moe_lite.py")


def test_gemma4_scale_loader_releases_the_fold_latch():
    patched = _patched(GEMMA4_ROUTER, patches._GEMMA4_ROUTER_FILE)
    assert "self.scale.weight_loader = self._load_scale" in patched
    loader = patched.split("def _load_scale")[1].split("def fuse_scale")[0]
    assert "default_weight_loader(param, loaded_weight)" in loader
    assert "self._scale_fused = False" in loader
    _verify(patched, patches._GEMMA4_ROUTER_FILE)


def test_the_prefill_skip_steps_over_the_skipped_requests_log_probs():
    patched = _patched(PREFILL_LOOP, patches._PREFILL_RESULT_FILE)
    skip_branch = patched.split("or req.is_retracted:")[1].split("continue")[0]
    assert "if batch.return_logprob:" in skip_branch
    assert "logprob_pt += (" in skip_branch
    assert "calculate_num_input_logprobs(" in skip_branch
    assert "extend_input_len_per_req[i]" in skip_branch and "extend_logprob_start_len_per_req[i]" in skip_branch
    _verify(patched, patches._PREFILL_RESULT_FILE)


def test_verify_refuses_a_gate_whose_weight_is_in_the_default_dtype():
    """A gate reading its weight live but not holding it in fp32: ``F.linear`` would mix dtypes."""
    default_dtype = GLM_GATE.replace(patches._GLM_WEIGHT_AFTER, _GLM_WEIGHT_DEFAULT_DTYPE)
    with pytest.raises(SystemExit, match="hold its weight in fp32"):
        patches.verify_glm_gate(default_dtype, "glm4_moe_lite.py")


def test_the_build_fails_once_the_gemma4_loader_is_no_longer_imported(tmp_path):
    (tmp_path / "srt/models").mkdir(parents=True)
    for name in patches._GLM_GATE_FILES:
        (tmp_path / name).write_text(GLM_GATE)
    gemma4 = tmp_path / patches._GEMMA4_ROUTER_FILE
    gemma4.write_text(_patched(GEMMA4_ROUTER, patches._GEMMA4_ROUTER_FILE))
    patches.verify_upstream(tmp_path)
    gemma4.write_text(gemma4.read_text().replace("import default_weight_loader", "import maybe_remap_kv_scale_name"))
    with pytest.raises(SystemExit, match="default_weight_loader is no longer imported"):
        patches.verify_upstream(tmp_path)


def test_verify_refuses_the_unpatched_files():
    with pytest.raises(SystemExit, match="still caches"):
        patches.verify_glm_gate(GLM_GATE_CACHING, "glm4_moe.py")
    with pytest.raises(SystemExit, match="Gemma4 router scale: the post-image is not in place"):
        _verify(GEMMA4_ROUTER, patches._GEMMA4_ROUTER_FILE)
    with pytest.raises(SystemExit, match="prefill result skip: the post-image is not in place"):
        _verify(PREFILL_LOOP, patches._PREFILL_RESULT_FILE)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
