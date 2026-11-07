"""Peeling framework, toolkit and PEFT wrappers off a live model, and respelling wrapped names as hub-model ones."""

import torch
from accelerate.utils import extract_model_from_parallel, is_peft_model
from peft import PeftModel

# The levels a ``PeftModel`` and ``UlyssesModel`` insert above the base model, in live names and saved adapter keys.
PEFT_BASE_MODEL_PREFIX = "base_model.model."
_CP_WRAPPER_MODULE_LEVEL = "model."
# The attribute a Ulysses attention wrapper keeps the module it replaced under: a level the hub tree lacks.
CP_ATTENTION_INNER_ATTR = "original_attention"


def unwrap_framework_wrappers(model: torch.nn.Module) -> torch.nn.Module:
    """An ``OptimizedModule`` names its parameters ``_orig_mod.*``, so a state dict through it writes a checkpoint
    ``from_pretrained`` and vLLM read zero tensors from; hence ``keep_torch_compile=False``."""
    return extract_model_from_parallel(model, recursive=True, keep_torch_compile=False)


def unwrap_toolkit_wrappers(model: torch.nn.Module) -> torch.nn.Module:
    """A PeftModel deliberately stays. A wrapper may yield inner names from ``named_parameters()`` while
    ``named_modules()`` still yields wrapper-prefixed paths, so a consumer cross-referencing the two must walk this."""
    model = unwrap_framework_wrappers(model)
    # On the type, as a delegating wrapper's __getattr__ would report the inner answer.
    inner_attr = getattr(type(model), "_toolkit_inner_model_attr", None)
    while inner_attr is not None:
        model = unwrap_framework_wrappers(getattr(model, inner_attr))
        inner_attr = getattr(type(model), "_toolkit_inner_model_attr", None)
    return model


def base_transformers_model(model: torch.nn.Module) -> torch.nn.Module:
    """Under every wrapper, PEFT included. A wrapper answers ``.model`` with what it wraps, so the same name read off
    the wrapper lands one level above the plain tree's."""
    base = unwrap_toolkit_wrappers(model)
    return unwrap_toolkit_wrappers(base.get_base_model()) if is_peft_model(base) else base


def plain_module_name(name: str) -> str:
    """PEFT and the Ulysses CP wrapper both deepen paths, so the PEFT prefix alone resolves nothing under PEFT+CP."""
    if name.startswith(f"{_CP_WRAPPER_MODULE_LEVEL}{_CP_WRAPPER_MODULE_LEVEL}"):
        name = name.removeprefix(_CP_WRAPPER_MODULE_LEVEL)
    else:
        cp_wrapped = f"{PEFT_BASE_MODEL_PREFIX}{_CP_WRAPPER_MODULE_LEVEL}{_CP_WRAPPER_MODULE_LEVEL}"
        if name.startswith(cp_wrapped):
            name = name.replace(cp_wrapped, f"{PEFT_BASE_MODEL_PREFIX}{_CP_WRAPPER_MODULE_LEVEL}", 1)
    return name.removeprefix(PEFT_BASE_MODEL_PREFIX)


def plain_peft_param_name(name: str, peft_prefix: str) -> str | None:
    """``None`` drops an adapter param, as the base weights are written with the delta folded in and emitting
    ``lora_A``/``lora_B`` too would apply it a second time on any PEFT-aware reload."""
    if peft_prefix in name or "original_module" in name:
        return None
    name = strip_base_layer_segment(name.removeprefix(PEFT_BASE_MODEL_PREFIX))
    return name.replace("modules_to_save.default.", "")


def strip_base_layer_segment(name: str) -> str:
    return name.replace(".base_layer", "")


def strip_peft_adapter_segment(name: str) -> str:
    """``...lora_A.default.weight`` becomes ``...lora_A.weight``. Every resume path must strip identically, as
    unmatched saved keys are dropped as unexpected by both adapter loaders."""
    name = name.replace(".modules_to_save.default.", ".").replace(".default.", ".")
    return name.removesuffix(".default")


def strip_cp_attention_level(key: str) -> str:
    """Every serializer reading a state dict off a Ulysses-patched tree must apply it, or ``from_pretrained`` leaves
    the attention weights randomly initialized."""
    return key.replace(f".{CP_ATTENTION_INNER_ATTR}.", ".")


def plain_adapter_key(key: str) -> str:
    """Strip CP-wrapper artifacts so an adapter key loads onto a non-CP model. :func:`plain_module_name` drops the PEFT
    prefix, which adapter keys must keep."""
    peft_cp = f"{PEFT_BASE_MODEL_PREFIX}{_CP_WRAPPER_MODULE_LEVEL}"
    new_key = strip_cp_attention_level(key)
    if new_key.startswith(_CP_WRAPPER_MODULE_LEVEL * 2):
        new_key = new_key.removeprefix(_CP_WRAPPER_MODULE_LEVEL)
    elif f"{peft_cp}{_CP_WRAPPER_MODULE_LEVEL}" in new_key:
        new_key = new_key.replace(f"{peft_cp}{_CP_WRAPPER_MODULE_LEVEL}", peft_cp)
    elif new_key.startswith(f"{peft_cp}lm_head."):
        # A ``modules_to_save`` lm_head sits above the backbone, one ``model.`` level shallower. A plain key never
        # spells lm_head at this depth, so the collapse is unambiguous and idempotent.
        new_key = new_key.replace(f"{peft_cp}lm_head.", f"{PEFT_BASE_MODEL_PREFIX}lm_head.", 1)
    return new_key


def find_peft_model(model) -> PeftModel | None:
    if isinstance(model, PeftModel):
        return model
    if hasattr(model, "model"):
        return find_peft_model(model.model)
    return None
