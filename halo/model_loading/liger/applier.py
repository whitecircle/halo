"""Liger appliers built from :class:`LigerFamilySpec` roles, and the deferred patch of remote modeling modules. Each
role subclasses the family's own class, keeping the class name lookups key on and avoiding the constructor variance
that blocks Liger's own swaps."""

from __future__ import annotations

import importlib
import inspect
import logging
import sys
from collections.abc import Callable
from types import ModuleType

from liger_kernel.transformers.auto_model import MODEL_TYPE_TO_APPLY_LIGER_FN

from halo.model_loading.liger.cross_entropy import patch_loss_utils_cross_entropy
from halo.model_loading.liger.family_specs import LigerFamilySpec
from halo.model_loading.liger.fused_modules import GLU_FLAGS, apply_fused_modules, apply_fused_modules_to_model
from halo.model_loading.remote_code import register_remote_class_hook
from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)

# (required class names, patch) armed for the modules transformers has not loaded yet.
_armed: list[tuple[tuple[str, ...], Callable[[ModuleType], object]]] = []


def _upstream_applier(spec: LigerFamilySpec) -> Callable:
    """The upstream Liger applier a delegating spec extends. Two distinct appliers mean the spec groups separate
    upstream families, one of which would be patched with the other's applier."""
    appliers = {model_type: MODEL_TYPE_TO_APPLY_LIGER_FN.get(model_type) for model_type in spec.model_types}
    missing = sorted(model_type for model_type, applier in appliers.items() if applier is None)
    if missing:
        raise ValueError(
            f"LigerFamilySpec for {spec.model_types} delegates to upstream Liger, which registers no "
            f"applier for {missing}. Declare the family's own roles instead, or drop those model types."
        )
    distinct = set(appliers.values())
    if len(distinct) != 1:
        raise ValueError(
            f"LigerFamilySpec for {spec.model_types} spans {sorted(fn.__qualname__ for fn in distinct)}; "
            f"split it so each spec delegates to one upstream applier."
        )
    return distinct.pop()


class LigerApplier:
    """The callable the dispatch resolves for one family, its signature built from the spec; ``upstream`` is a
    delegating spec's applier, run first."""

    def __init__(self, spec: LigerFamilySpec):
        self.spec = spec
        self.upstream = _upstream_applier(spec) if spec.delegates_to_upstream else None
        # Never handed to upstream as given: its CE branch rebinds `torch.nn.functional.cross_entropy`
        # for the whole process (the scoped patch serves CE instead), and the roles the spec took over.
        self.withheld = ("cross_entropy", *spec.upstream_off)
        # Resolved once, so the signature the dispatch reads and a call's defaults share one source.
        self.defaults = dict(self._declared_parameters())
        self.__signature__ = inspect.Signature(
            [
                inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=default)
                for name, default in self.defaults.items()
            ]
        )
        self.__qualname__ = f"apply_liger_kernel_to_{spec.model_types[0]}"
        self.__module__ = __name__

    def hands_upstream(self, flag: str) -> bool:
        """Whether a call passes ``flag`` on to upstream's applier, leaving that role to upstream."""
        return self.upstream is not None and flag in self.defaults and flag not in self.withheld

    def _declared_parameters(self):
        """One parameter per role that will run. Under delegation the names and defaults are upstream's, bar the
        toolkit's loss convention (CE on, FLCE off, the reverse of Liger's, because CE keeps the logits metrics
        readable)."""
        spec = self.spec
        loss_defaults = {"cross_entropy": not spec.flce_default, "fused_linear_cross_entropy": spec.flce_default}
        if self.upstream is not None:
            for name, parameter in inspect.signature(self.upstream).parameters.items():
                if name != "model":
                    yield name, loss_defaults.get(name, parameter.default)
        else:
            yield "rope", spec.rope
            yield "cross_entropy", loss_defaults["cross_entropy"]
            if spec.causal_lm:
                yield "fused_linear_cross_entropy", loss_defaults["fused_linear_cross_entropy"]
            if spec.rms_norm or spec.gated_rms_norm:
                yield "rms_norm", True
            if spec.glu_mlp:
                yield GLU_FLAGS[0], True
        yield "model", None

    def __call__(self, **kwargs) -> list[str]:
        spec = self.spec
        flags = dict(self.defaults)
        flags.update(kwargs)
        model = flags.pop("model", None)

        # Under delegation the rotary is upstream's where its own default serves the family; refuse
        # otherwise, since an upstream applier may only warn and patch nothing.
        upstream_serves_rope = self.upstream is not None and self.defaults.get("rope") is True
        if flags.get("rope") and not spec.rope and not upstream_serves_rope:
            raise NotImplementedError(
                f"rope is not implemented for {spec.model_types[0]}: its rotary (partial, "
                f"interleaved, mrope, YARN or per-layer-type) is not what Liger's kernel computes. "
                f"Remove rope from liger_kernel_config."
            )
        if flags.get("cross_entropy") and flags.get("fused_linear_cross_entropy"):
            raise ValueError("cross_entropy and fused_linear_cross_entropy cannot both be True.")

        patched: list[str] = []
        if self.upstream is not None:
            # Upstream keeps every declared role except the withheld ones; the scoped CE patch below serves the loss.
            upstream_flags = {**flags, **dict.fromkeys(self.withheld, False)}
            self.upstream(model=model, **upstream_flags)
            on = sorted(name for name, enabled in upstream_flags.items() if enabled)
            patched.append(f"{self.upstream.__name__}({', '.join(on)})")

        if spec.modeling_module:
            patched += apply_fused_modules(importlib.import_module(spec.modeling_module), spec, flags)
        if spec.remote_classes:
            # The modeling module does not exist until transformers loads the remote file.
            patch_remote_modules(spec.remote_classes, lambda module: apply_fused_modules(module, spec, flags))
            patched.append(f"armed for the remote module defining {spec.remote_classes[0]}")

        if flags.get("cross_entropy"):
            patch_loss_utils_cross_entropy()
            patched.append("CrossEntropy")

        if model is not None:
            apply_fused_modules_to_model(model, spec, flags)

        if is_global_main_process():
            logger.info(f"Liger Kernel applied to {spec.model_types[0]}: {', '.join(patched) or 'nothing'}")
        return patched


def build_liger_appliers(specs: tuple[LigerFamilySpec, ...]) -> dict[str, Callable]:
    """``model_type`` → applier for every spec; a family's aliases share one applier instance."""
    registry: dict[str, Callable] = {}
    for spec in specs:
        applier = LigerApplier(spec)
        for model_type in spec.model_types:
            if model_type in registry:
                raise ValueError(f"Two LigerFamilySpecs claim model_type {model_type!r}")
            registry[model_type] = applier
    return registry


def _declared_classes(module: ModuleType, class_names: tuple[str, ...]) -> set[str]:
    return {name for name in class_names if isinstance(getattr(module, name, None), type)}


def _fire(module: ModuleType) -> None:
    """Run every armed patch whose classes this module defines; a partial match warns, as repos share class names."""
    for class_names, patch in _armed:
        present = _declared_classes(module, class_names)
        if present == set(class_names):
            patch(module)
            if is_global_main_process():
                logger.info(f"Liger Kernel applied to remote modeling module {module.__name__}")
        elif present and is_global_main_process():
            logger.warning(
                f"Liger Kernel NOT applied to {module.__name__}: it defines {sorted(present)} but "
                f"not {sorted(set(class_names) - present)}. This revision has renamed a class the "
                f"family's LigerFamilySpec names — the run trains unfused until the spec is updated."
            )


def patch_remote_modules(class_names: tuple[str, ...], patch: Callable[[ModuleType], object]) -> None:
    """Apply ``patch`` to the remote module defining ``class_names``, now or when it loads. Re-applying Liger replaces
    the armed patch, so the latest application's flags fire."""
    entry = (class_names, patch)
    index = next((i for i, (names, _) in enumerate(_armed) if names == class_names), None)
    if index is None:
        _armed.append(entry)
    else:
        _armed[index] = entry
    register_remote_class_hook(_fire)
    for module in list(sys.modules.values()):
        if (
            isinstance(module, ModuleType)
            and module.__name__.startswith("transformers_modules.")
            and _declared_classes(module, class_names) == set(class_names)
        ):
            patch(module)
