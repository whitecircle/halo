"""Source edits to the installed SGLang, applied at image build through ``engine_source_patch``.

python3 patch_sglang.py            # patch the installed sglang in place
python3 patch_sglang.py --verify   # assert the post-images only
"""

import ast
import pathlib

from engine_source_patch import Edit, main

_GLM_GATE_FILES = ("srt/models/glm4_moe.py", "srt/models/glm4_moe_lite.py")
_GEMMA4_ROUTER_FILE = "srt/models/gemma4_causal.py"
_PREFILL_RESULT_FILE = "srt/managers/scheduler_components/batch_result_processor.py"

_GLM_WEIGHT_AFTER = """        self.weight = nn.Parameter(
            torch.empty(
                (config.n_routed_experts, config.hidden_size), dtype=torch.float32
            )
        )
"""
_GLM_FORWARD_AFTER = """        logits = F.linear(hidden_states.to(torch.float32), self.weight, None)
"""

_GEMMA4_SCALE_BEFORE = """        self.scale = nn.Parameter(torch.ones(self.hidden_size))
"""
_GEMMA4_FUSE_BEFORE = """    def fuse_scale(self):
"""

_PREFILL_SKIP_BEFORE = """                if (
                    req.finished() and req.inflight_middle_chunks <= 0
                ) or req.is_retracted:
                    # Decode req in a mixed batch, or a retracted req. Keep an
                    # aborted middle chunk in the chunked branch long enough to
                    # drain its accounting without streaming it.
                    continue
"""
_PREFILL_SKIP_AFTER = """                if (
                    req.finished() and req.inflight_middle_chunks <= 0
                ) or req.is_retracted:
                    # The flat input-logprob array carries a block for every
                    # request in the batch, including this one, so step over it
                    # or every later request reads a shifted slice.
                    if batch.return_logprob:
                        assert extend_input_len_per_req is not None
                        assert extend_logprob_start_len_per_req is not None
                        logprob_pt += (
                            self.logprob_result_processor.calculate_num_input_logprobs(
                                req,
                                extend_input_len_per_req[i],
                                extend_logprob_start_len_per_req[i],
                            )
                        )
                    # Decode req in a mixed batch, or a retracted req. Keep an
                    # aborted middle chunk in the chunked branch long enough to
                    # drain its accounting without streaming it.
                    continue
"""

EDITS = (
    # /update_weights_from_distributed ends in model.load_weights, but the Gemma 4 router folds
    # ``scale`` into its norm weight once, behind a latch: a pushed ``scale`` would never reach routing.
    # A load of ``scale`` releases the latch.
    Edit(
        _GEMMA4_ROUTER_FILE,
        _GEMMA4_SCALE_BEFORE,
        _GEMMA4_SCALE_BEFORE + "        self.scale.weight_loader = self._load_scale\n",
        "Gemma4 router scale",
    ),
    Edit(
        _GEMMA4_ROUTER_FILE,
        _GEMMA4_FUSE_BEFORE,
        """    def _load_scale(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        default_weight_loader(param, loaded_weight)
        self._scale_fused = False

"""
        + _GEMMA4_FUSE_BEFORE,
        "Gemma4 router fuse_scale",
    ),
    # The prefill result loop reads each request's input log-probs from one flat array at a running
    # offset; upstream skips a retracted or mixed-batch-finished request without stepping over its
    # block, so every later request reads a shifted slice (a port of sgl-project/sglang#36939).
    Edit(_PREFILL_RESULT_FILE, _PREFILL_SKIP_BEFORE, _PREFILL_SKIP_AFTER, "prefill result skip"),
)


def verify_glm_gate(text: str, path: str) -> None:
    """The GLM-4 gate holds its weight in fp32 and reads it live, so a synced gate weight reaches routing."""
    if "_weight_fp32" in text:
        raise SystemExit(f"{path}: the GLM gate still caches its fp32 weight")
    if _GLM_WEIGHT_AFTER not in text or _GLM_FORWARD_AFTER not in text:
        raise SystemExit(f"{path}: the GLM gate does not hold its weight in fp32 and read it live")


def verify_upstream(root: pathlib.Path) -> None:
    """Assert the upstream code the edits rely on: the GLM-4 gates and the loader ``_load_scale`` calls."""
    for name in _GLM_GATE_FILES:
        verify_glm_gate((root / name).read_text(encoding="utf-8"), name)
    gemma4 = ast.parse((root / _GEMMA4_ROUTER_FILE).read_text(encoding="utf-8"))
    imported = {
        alias.asname or alias.name
        for node in ast.walk(gemma4)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    if "default_weight_loader" not in imported:
        raise SystemExit("Gemma4 router loader: default_weight_loader is no longer imported — upstream changed")


if __name__ == "__main__":
    main("sglang", EDITS, verify_upstream)
