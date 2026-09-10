"""Patch the pinned SGLang's model files so an online weight update reaches every served tensor.

``/update_weights_from_distributed`` ends in ``model.load_weights`` and nothing else, so a loader that
caches a derived form of a parameter keeps serving the cached one after the parameter is rewritten.
Two families do that in this release:

* GLM-4 MoE (``glm4_moe.py``, ``glm4_moe_lite.py``): the router gate keeps an fp32 copy of its
  weight, filled once on the first forward; a pushed gate weight lands in the parameter and routing
  never reads it. The gate is rewritten to hold its weight in fp32 and read it live.
* Gemma 4 (``gemma4_causal.py``, shared by the multimodal classes): the router folds ``scale`` into
  its norm weight once, behind a latch, so a pushed ``scale`` never reaches the fold. The latch is
  released by the parameter's own loader.

Every rewrite asserts its pre-image once and its post-image after, so an upstream change fails the
image build rather than shipping a server that silently serves stale routing.

    python3 patch_sglang_weight_updates.py            # patch the installed sglang in place
    python3 patch_sglang_weight_updates.py --verify   # assert the post-images only
"""

import argparse
import importlib.util
import pathlib

_GLM_GATE_FILES = ("glm4_moe.py", "glm4_moe_lite.py")
_GEMMA4_ROUTER_FILE = "gemma4_causal.py"

_GLM_WEIGHT_BEFORE = """        self.weight = nn.Parameter(
            torch.empty((config.n_routed_experts, config.hidden_size))
        )
"""
_GLM_WEIGHT_AFTER = """        self.weight = nn.Parameter(
            torch.empty(
                (config.n_routed_experts, config.hidden_size), dtype=torch.float32
            )
        )
"""
_GLM_CACHE_BEFORE = """        # GLM requires FP32 gate projection; cache to avoid per-forward cast.
        # FIXME: if gate weight is updated at runtime (e.g. expert rebalancing), _weight_fp32 must be invalidated.
        self.register_buffer("_weight_fp32", None, persistent=False)
"""
_GLM_FORWARD_BEFORE = """        if self._weight_fp32 is None:
            self._weight_fp32 = self.weight.data.to(torch.float32)
        logits = F.linear(hidden_states.to(torch.float32), self._weight_fp32, None)
"""
_GLM_FORWARD_AFTER = """        logits = F.linear(hidden_states.to(torch.float32), self.weight, None)
"""

_GEMMA4_SCALE_BEFORE = """        self.scale = nn.Parameter(torch.ones(self.hidden_size))
"""
_GEMMA4_SCALE_AFTER = """        self.scale = nn.Parameter(torch.ones(self.hidden_size))
        self.scale.weight_loader = self._load_scale
"""
_GEMMA4_FUSE_BEFORE = """    def fuse_scale(self):
"""
_GEMMA4_FUSE_AFTER = """    def _load_scale(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        default_weight_loader(param, loaded_weight)
        self._scale_fused = False

    def fuse_scale(self):
"""


def _replace_once(text: str, before: str, after: str, what: str) -> str:
    count = text.count(before)
    if count != 1:
        raise SystemExit(f"{what}: expected the pre-image exactly once, found {count} — upstream changed")
    return text.replace(before, after)


def patch_glm_gate(text: str) -> str:
    """The GLM router gate holds its weight in fp32 and reads it live."""
    text = _replace_once(text, _GLM_WEIGHT_BEFORE, _GLM_WEIGHT_AFTER, "GLM gate weight dtype")
    text = _replace_once(text, _GLM_CACHE_BEFORE, "", "GLM gate fp32 cache")
    return _replace_once(text, _GLM_FORWARD_BEFORE, _GLM_FORWARD_AFTER, "GLM gate forward")


def patch_gemma4_router(text: str) -> str:
    """A load of the Gemma 4 router's ``scale`` releases the fold latch."""
    if "default_weight_loader" not in text:
        raise SystemExit("Gemma4 router loader: default_weight_loader is no longer imported — upstream changed")
    text = _replace_once(text, _GEMMA4_SCALE_BEFORE, _GEMMA4_SCALE_AFTER, "Gemma4 router scale")
    return _replace_once(text, _GEMMA4_FUSE_BEFORE, _GEMMA4_FUSE_AFTER, "Gemma4 router fuse_scale")


def verify_glm_gate(text: str, path: pathlib.Path) -> None:
    if "_weight_fp32" in text:
        raise SystemExit(f"{path}: the GLM gate still caches its fp32 weight")
    if _GLM_WEIGHT_AFTER not in text or _GLM_FORWARD_AFTER not in text:
        raise SystemExit(f"{path}: the GLM gate does not hold its weight in fp32 and read it live")


def verify_gemma4_router(text: str, path: pathlib.Path) -> None:
    if _GEMMA4_SCALE_AFTER not in text or _GEMMA4_FUSE_AFTER not in text:
        raise SystemExit(f"{path}: the Gemma4 router scale loader does not release the fold latch")


def _models_dir() -> pathlib.Path:
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("sglang is not importable in this interpreter")
    return pathlib.Path(next(iter(spec.submodule_search_locations))) / "srt" / "models"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true", help="assert the post-images without rewriting")
    args = parser.parse_args()
    models = _models_dir()
    targets = [(models / name, patch_glm_gate, verify_glm_gate) for name in _GLM_GATE_FILES]
    targets.append((models / _GEMMA4_ROUTER_FILE, patch_gemma4_router, verify_gemma4_router))
    for path, patch, verify in targets:
        if not path.is_file():
            raise SystemExit(f"{path} is missing — upstream moved the model file")
        text = path.read_text(encoding="utf-8")
        if not args.verify:
            text = patch(text)
            path.write_text(text, encoding="utf-8")
        verify(text, path)
        print(f"{path.name}: {'verified' if args.verify else 'patched'}")


if __name__ == "__main__":
    main()
