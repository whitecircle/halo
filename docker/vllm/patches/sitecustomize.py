"""Auto-applied vLLM patches (CPython imports ``sitecustomize`` at interpreter startup).

Enabled by putting this directory on ``PYTHONPATH`` in the vLLM container, so the patches reach the
API server *and* every engine-core worker process (the workers are where weight loading and the
weight-transfer group live).

vLLM must not be imported here — that would drag torch in during ``site`` execution. Instead a
meta-path hook, installed at interpreter start, applies each patch the moment vLLM first imports the
module it targets, whenever that happens.
"""

import importlib
import importlib.abc
import importlib.util
import logging
import sys

# vLLM module → the patch module fixing it. One hook serves every entry; a new patch is a new row.
_PATCHES = {
    "vllm.model_executor.model_loader.reload.layerwise": "vllm_layerwise_reload_patch",
    "vllm.distributed.weight_transfer.nccl_engine": "vllm_weight_transfer_reinit_patch",
}


class _PatchOnImport(importlib.abc.MetaPathFinder):
    """Run each patch module's ``apply()`` right after its target module is executed."""

    def __init__(self, patches: dict[str, str]):
        self._pending = dict(patches)
        self._resolving: set[str] = set()

    def find_spec(self, fullname, path=None, target=None):
        patch_name = self._pending.get(fullname)
        if patch_name is None or fullname in self._resolving:
            return None

        # Re-entrancy guard rather than retiring the entry here: an import that fails part-way (a
        # transient error) would otherwise consume the one shot, and the retry would load the module
        # unpatched — silently reverting experts on every weight sync, or leaking a communicator per
        # trainer. The entry is retired below, once the target has actually executed.
        self._resolving.add(fullname)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            self._resolving.discard(fullname)
        if spec is None or spec.loader is None:
            return None

        original_exec_module = spec.loader.exec_module

        def exec_module(module):
            original_exec_module(module)
            self._pending.pop(fullname, None)
            if not self._pending and self in sys.meta_path:
                sys.meta_path.remove(self)
            try:
                importlib.import_module(patch_name).apply()
            except Exception:  # noqa: BLE001 — a broken patch must never break vLLM's own import
                logging.getLogger(__name__).exception("failed to apply vLLM patch %s", patch_name)

        spec.loader.exec_module = exec_module  # type: ignore[method-assign]
        return spec


sys.meta_path.insert(0, _PatchOnImport(_PATCHES))
