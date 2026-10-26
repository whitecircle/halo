"""Peeling framework, toolkit and PEFT wrappers off a live model, and respelling the names read off a
wrapped tree as plain hub-model ones. Rank-local.
"""

import torch
from accelerate.utils import extract_model_from_parallel, is_peft_model

# The module level a ``PeftModel`` inserts above the base model, in live names and saved adapter
# keys alike, and the one ``UlyssesModel`` inserts (its ``_toolkit_inner_model_attr``).
# Shared by every consumer that maps such a path back to the hub tree; a second copy drifts into an
# unresolvable key.
PEFT_BASE_MODEL_PREFIX = "base_model.model."
_CP_WRAPPER_MODULE_LEVEL = "model."
# The attribute a Ulysses attention wrapper keeps the module it replaced under: a level the hub tree lacks.
CP_ATTENTION_INNER_ATTR = "original_attention"


def unwrap_framework_wrappers(model: torch.nn.Module) -> torch.nn.Module:
    """Peel accelerate/DDP/FSDP **and** ``torch.compile``, leaving toolkit wrappers in place.

    ``keep_torch_compile=False`` is load-bearing: the ``OptimizedModule`` accelerate keeps by default
    names its parameters ``_orig_mod.*``, so a state dict taken through it writes a checkpoint
    ``from_pretrained`` and vLLM read zero tensors from, and an ``isinstance`` check for the CP
    wrapper fails through it. Neither raises.
    """
    return extract_model_from_parallel(model, recursive=True, keep_torch_compile=False)


def unwrap_toolkit_wrappers(model: torch.nn.Module) -> torch.nn.Module:
    """Peel every framework and toolkit wrapper down to the inner model (a PeftModel deliberately stays).

    Toolkit wrappers declare their inner-model attribute as ``_toolkit_inner_model_attr``, so a new
    wrapper type joins by declaring it. A wrapper may yield inner names from ``named_parameters()``
    while ``named_modules()`` still yields wrapper-prefixed paths, so a consumer cross-referencing
    the two must walk this.
    """
    model = unwrap_framework_wrappers(model)
    # type() lookup, not getattr: a delegating wrapper's __getattr__ would report the inner answer.
    inner_attr = getattr(type(model), "_toolkit_inner_model_attr", None)
    while inner_attr is not None:
        model = unwrap_framework_wrappers(getattr(model, inner_attr))
        inner_attr = getattr(type(model), "_toolkit_inner_model_attr", None)
    return model


def base_transformers_model(model: torch.nn.Module) -> torch.nn.Module:
    """The plain transformers model under every framework, toolkit **and** PEFT wrapper.

    :func:`unwrap_toolkit_wrappers` deliberately stops at a ``PeftModel``; this goes the one level
    further, for consumers that need the tree as transformers laid it out. Load-bearing for any
    attribute probe: a wrapper answers ``.model`` with what *it* wraps — a ``PeftModel`` through its
    tuner's ``__getattr__``, a toolkit wrapper directly — so the same name read off the wrapper lands
    one level above the plain tree's.
    """
    base = unwrap_toolkit_wrappers(model)
    return unwrap_toolkit_wrappers(base.get_base_model()) if is_peft_model(base) else base


def plain_module_name(name: str) -> str:
    """Map a module path read off a live wrapped tree to its plain hub-model spelling.

    Two toolkit-visible wrappers deepen every path beneath them: PEFT and the Ulysses CP wrapper,
    which ``unwrap_framework_wrappers`` keeps. Stripping only the PEFT prefix resolves nothing under
    PEFT+CP.
    """
    if name.startswith(f"{_CP_WRAPPER_MODULE_LEVEL}{_CP_WRAPPER_MODULE_LEVEL}"):
        name = name.removeprefix(_CP_WRAPPER_MODULE_LEVEL)
    else:
        cp_wrapped = f"{PEFT_BASE_MODEL_PREFIX}{_CP_WRAPPER_MODULE_LEVEL}{_CP_WRAPPER_MODULE_LEVEL}"
        if name.startswith(cp_wrapped):
            name = name.replace(cp_wrapped, f"{PEFT_BASE_MODEL_PREFIX}{_CP_WRAPPER_MODULE_LEVEL}", 1)
    return name.removeprefix(PEFT_BASE_MODEL_PREFIX)


def plain_peft_param_name(name: str, peft_prefix: str) -> str | None:
    """Map a ``PeftModel`` param name to its plain base-model name, or ``None`` to drop it.

    Adapter params are dropped rather than renamed because the base weights are written with the
    delta already folded in (:func:`~src.models.peft_adapters.lora_folded`), so emitting
    ``lora_A``/``lora_B`` too would apply it a second time on any PEFT-aware reload.
    """
    if peft_prefix in name or "original_module" in name:
        return None
    name = strip_base_layer_segment(name.removeprefix(PEFT_BASE_MODEL_PREFIX))
    return name.replace("modules_to_save.default.", "")


def strip_base_layer_segment(name: str) -> str:
    """Drop the level a PEFT tuner layer inserts above the module it wraps: ``q_proj.base_layer.weight``
    becomes ``q_proj.weight``."""
    return name.replace(".base_layer", "")


def strip_peft_adapter_segment(name: str) -> str:
    """Drop the ``default`` adapter-name segment from a live PEFT param name.

    ``...lora_A.default.weight`` becomes ``...lora_A.weight``, ``...modules_to_save.default.weight``
    becomes ``....weight``, and the ParameterDict spelling ``...lora_embedding_A.default`` matches
    what the adapter save writes, since a saved adapter file carries no adapter name. Every resume
    path rebuilding the saved-key to live-key map must strip identically, since unmatched saved keys
    are dropped as unexpected by both adapter loaders.
    """
    name = name.replace(".modules_to_save.default.", ".").replace(".default.", ".")
    return name.removesuffix(".default")


def strip_cp_attention_level(key: str) -> str:
    """Map a key read off a Ulysses-patched tree back to the hub layout by dropping the
    ``CP_ATTENTION_INNER_ATTR`` level. Every serializer reading a state dict out of such a tree (the CP and EP
    gathered saves, the pipeline stage shard, PEFT adapters) must apply it, or ``from_pretrained`` leaves the
    attention weights randomly initialized."""
    return key.replace(f".{CP_ATTENTION_INNER_ATTR}.", ".")
