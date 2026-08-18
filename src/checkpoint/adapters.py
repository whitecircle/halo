"""Saved PEFT adapters as an artifact: the file layout, the shape gates, and the merge into a base.

Format layer only — no ``torch.distributed`` — so the standalone merge tools and the parallel
adapter saver classify one artifact shape the same way. Producing an adapter from a live sharded
model (the DTensor gather, the CP key remap) belongs to
:mod:`src.distributed.checkpoint.peft`.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable

import torch
from accelerate.logging import get_logger
from peft import PeftConfig, PeftModel
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import PreTrainedModel

from src.checkpoint.format import (
    ADAPTER_CONFIG_FILE,
    ADAPTER_SAFETENSORS_FILE,
    ADAPTER_WEIGHT_NAMES,
    DEFAULT_MAX_SHARD_SIZE,
    cast_to_save_dtype,
)
from src.checkpoint.tool_io import (
    apply_training_sidecars,
    copy_training_sidecars,
    preflight_model_load_resources,
    reject_in_place_conversion,
    reject_sharded_checkpoint,
    save_full_checkpoint,
)
from src.models.loading.tokenizer_setup import resolve_peft_processing_class
from src.models.moe_balancing import is_balancing_state_key
from src.models.structure import strip_peft_adapter_segment

logger = get_logger(__name__)

# Where a merge parks the adapter's own config beside the merged weights: adapter_config.json in the
# output root would make every from_pretrained-based tool downstream read the merged model as an
# unmerged adapter and re-export the bare base.
MERGED_ADAPTER_CONFIG_DIR = "original_adapter_config"

# ``adapter_config.json`` markers for the two adapter shapes carrying native EP grouped expert
# deltas: the expert-only save (``save_ep_checkpoint``) and the mixed attention+expert save
# (``PeftAdapterSaver``). The merge guard keys on them — one spelling each, or it silently stops
# firing. Deliberately absent from PEFT's ``PEFT_TYPE_TO_CONFIG_MAPPING``: stock PEFT drops an
# unknown state-dict key silently, so a ``peft_type: LORA`` hybrid would load its attention half and
# discard every expert delta; an unknown ``peft_type`` makes that lookup raise, naming the marker.
EXPERT_LORA_PEFT_TYPE = "EXPERT_LORA"
MIXED_EXPERT_LORA_PEFT_TYPE = "LORA_WITH_EP_EXPERT_LORA"
EXPERT_LORA_PEFT_TYPES = frozenset({EXPERT_LORA_PEFT_TYPE, MIXED_EXPERT_LORA_PEFT_TYPE})


def is_expert_lora_key(key: str) -> bool:
    """Whether a saved adapter key is a native EP expert adapter (``<layer>.experts.<attr>.lora_{A,B}``).

    The saved-key counterpart of ``is_expert_weight_attr``, which classifies the live parameter name.
    Both the resume path and the PEFT-merge guard must agree on this namespace: a predicate that
    drifted would either drop expert adapters on resume or let ``merge_and_unload`` silently discard
    them.

    The trailing-segment test is what separates the two namespaces. A native adapter is stored as the
    whole grouped tensor, so its key ENDS at ``lora_A``/``lora_B``; stock PEFT wraps an ``nn.Linear``
    and always appends the tuner's own suffix (``.lora_A.default.weight``). Families that keep experts
    as a per-expert ``nn.ModuleList`` (Bailing/Ling) let PEFT adapt them directly when no EP wrapper is
    built, and those keys also carry ``.experts.`` — matching them here would route real PEFT adapters
    into the EP loader, which has nowhere to put them.
    """
    return ".experts." in key and key.endswith((".lora_A", ".lora_B"))


def adapter_weight_paths(adapter_dir: str) -> tuple[str, ...]:
    """The adapter weight files a directory may carry, in PEFT's own load-preference order.

    One spelling of the pair for every reader here — the expert-LoRA refusal, the resume — off the
    :data:`~src.checkpoint.format.ADAPTER_WEIGHT_NAMES` tuple that declares it: a hand-built
    pair drifts the day PEFT's fallback filename changes, and a reader that misses the ``.bin``
    fallback reads a saved adapter as absent.
    """
    return tuple(os.path.join(adapter_dir, name) for name in ADAPTER_WEIGHT_NAMES)


def _adapter_tensor_keys(path: str) -> list[str]:
    """The tensor names in an adapter file, without materializing its tensors."""
    if path.endswith(ADAPTER_SAFETENSORS_FILE):
        with safe_open(path, framework="pt") as handle:
            return list(handle.keys())
    return list(torch.load(path, map_location="cpu", weights_only=True))


def read_adapter_file(path: str) -> dict:
    """The adapter tensors at ``path`` — safetensors, or the legacy pickle PEFT still writes."""
    if path.endswith(ADAPTER_SAFETENSORS_FILE):
        return load_file(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def cast_adapter_state_to_save_dtype(state: dict) -> dict:
    """Adapter-file cast honouring the balancing export contract: balancing tensors (a
    ``modules_to_save`` router's bias riding along with the clone) stay at trained dtype, same
    as every full-checkpoint writer — a bf16-rounded bias flips near-tie top-k picks on serve.

    Adapter keys interpose PEFT segments between the module path and the slot
    (``gate.modules_to_save.default.e_score_correction_bias``); they are stripped for the check
    so the registry's dotted suffixes match the same tensors they match in a full checkpoint.
    """

    def _is_balancing(key: str) -> bool:
        return is_balancing_state_key(strip_peft_adapter_segment(key))

    return {k: v if _is_balancing(k) else cast_to_save_dtype(v) for k, v in state.items()}


def _expert_lora_merge_remedy(adapter_dir: str, *, mixed: bool) -> str:
    """What this adapter holds that a PEFT merge drops, and the save that folds it instead.

    Both shapes carry native EP grouped ``*.experts.<attr>.lora_{A,B}`` tensors, which PEFT cannot
    wrap and ``merge_and_unload`` discards, so both have the same remedy: re-save the run with
    ``merge_expert_lora_on_save``, which folds them inside
    :func:`~src.distributed.checkpoint.save.save_ep_checkpoint`'s gathered save. On the
    mixed shape that flag also routes the write past :class:`PeftAdapterSaver` and folds the
    attention half (``merged_adapters`` held open over it), so ``mixed`` only picks the wording.
    """
    if mixed:
        shape = (
            f"holds a MIXED attention+expert LoRA adapter (peft_type={MIXED_EXPERT_LORA_PEFT_TYPE}): "
            "alongside the stock PEFT attention tensors it carries native EP grouped expert deltas"
        )
        folds = "the expert deltas inside each family's expert gather, and the attention deltas with them"
    else:
        shape = (
            f"is a native EP expert-LoRA adapter (peft_type={EXPERT_LORA_PEFT_TYPE}): grouped "
            "*.experts.<attr>.lora_{A,B} tensors"
        )
        folds = "the expert deltas inside each family's expert gather"
    return (
        f"{adapter_dir} {shape} — which a PEFT merge silently drops. Re-save the run with "
        f"merge_expert_lora_on_save=True: the gathered EP save folds {folds}, producing a merged "
        "servable HF checkpoint."
    )


def assert_no_expert_lora_adapter(adapter_dir: str) -> None:
    """Refuse an adapter directory carrying native EP expert-LoRA deltas.

    EP expert adapters are plain ``<layer>.experts.<attr>.lora_{A,B}`` tensors (PEFT cannot wrap
    grouped expert weights), so ``PeftModel.merge_and_unload()`` drops every expert delta and yields a
    base-quality model that looks merged. Both writer shapes carry their
    :data:`EXPERT_LORA_PEFT_TYPES` marker — the expert-only adapter and the mixed attention+expert one
    :meth:`_save_with_expert_lora` writes. The verdict is the SHAPE, not the marker, so an adapter
    whose config is absent or unmarked is still classified by a tensor-key scan (safetensors or the
    ``.bin`` fallback); one carrying no expert keys loads as the plain adapter it is.

    The single home for the check — every tool that calls ``merge_and_unload`` on a saved adapter
    must run it.
    """
    cfg_path = os.path.join(adapter_dir, ADAPTER_CONFIG_FILE)
    if os.path.isfile(cfg_path):
        with open(cfg_path) as fh:
            peft_type = json.load(fh).get("peft_type")
        if peft_type in EXPERT_LORA_PEFT_TYPES:
            raise ValueError(_expert_lora_merge_remedy(adapter_dir, mixed=peft_type == MIXED_EXPERT_LORA_PEFT_TYPE))

    keys: list[str] = []
    for path in adapter_weight_paths(adapter_dir):
        if os.path.isfile(path):
            keys = _adapter_tensor_keys(path)
            break

    expert_keys = [key for key in keys if is_expert_lora_key(key)]
    if expert_keys:
        # Unmarked but expert-shaped. Any non-expert adapter tensor means an attention PeftModel was
        # present too, i.e. the mixed shape rather than the expert-only one.
        raise ValueError(_expert_lora_merge_remedy(adapter_dir, mixed=len(expert_keys) != len(keys)))


def merge_adapter_into_base(
    adapter_dir: str,
    output_dir: str,
    *,
    load_base_model: Callable[..., PreTrainedModel],
    tool: str,
    device_map: str | None = None,
    max_shard_size: str = DEFAULT_MAX_SHARD_SIZE,
    trust_remote_code: bool = True,
    prepare_for_save: Callable[[PreTrainedModel], None] | None = None,
    verbose: bool = True,
) -> PreTrainedModel:
    """Fold a saved PEFT adapter into its base model and write the merged checkpoint.

    The one gate order behind every merge tool; each gate answers a silent failure:

    * a per-rank EP/TP directory reads as whole while every expert tensor is one rank's slice —
      ``from_pretrained`` reports the real keys as MISSING and randomly initializes them, warning
      only. Checked on the adapter and again on the base, from which a merge takes its WEIGHTS;
    * an ``--output_dir`` pointing at the adapter or the base destroys that input, since the save
      deletes the weight files it does not overwrite;
    * ``merge_and_unload`` cannot fold a native EP expert-LoRA adapter at all — it drops every expert
      delta and yields a base-quality model that looks merged.

    ``excuse_task_head`` is read off the adapter's ``modules_to_save``: a classification head on a
    plain causal-LM base is the ONLY absence the tool's own ``load_base_model`` may excuse.
    ``prepare_for_save`` mutates the merged model before the save, which carries the BASE's aux
    files (``source_dir``) — the adapter's sidecars and its own config travel separately.
    """
    log = logger.info if verbose else lambda message: None

    reject_sharded_checkpoint(adapter_dir)
    reject_in_place_conversion(adapter_dir, output_dir)
    assert_no_expert_lora_adapter(adapter_dir)

    peft_config = PeftConfig.from_pretrained(adapter_dir)
    base_model_path = peft_config.base_model_name_or_path
    if base_model_path and os.path.isdir(base_model_path):
        reject_in_place_conversion(base_model_path, output_dir)
        reject_sharded_checkpoint(base_model_path)

    # Size preflight before the heavy load: without a device map the whole base lands in host RAM,
    # and the merged output is about the base checkpoint's size on disk.
    preflight_model_load_resources(base_model_path, output_dir, tool=tool, device_map=device_map)

    # Full processor for VLM adapters (image preprocessor + tokenizer + chat template), else a plain
    # tokenizer, taken from the adapter dir and falling back to the base — a tokenizer-only save
    # leaves an unloadable VLM.
    processing_class = resolve_peft_processing_class(adapter_dir, base_model_path, trust_remote_code=trust_remote_code)
    if processing_class is None:
        raise OSError(f"Could not load a processor or tokenizer from {adapter_dir} or {base_model_path}")

    log(f"Merging {adapter_dir} ({peft_config.peft_type}) into base model {base_model_path}...")
    base_model = load_base_model(base_model_path, excuse_task_head=bool(getattr(peft_config, "modules_to_save", None)))
    merged_model = PeftModel.from_pretrained(base_model, adapter_dir).merge_and_unload()

    for action in apply_training_sidecars(merged_model, adapter_dir):
        log(action)
    if prepare_for_save is not None:
        prepare_for_save(merged_model)

    log(f"Saving merged model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    save_full_checkpoint(
        merged_model,
        output_dir,
        processing_class=processing_class,
        source_dir=base_model_path,
        max_shard_size=max_shard_size,
    )
    copy_training_sidecars(adapter_dir, output_dir)
    peft_config.save_pretrained(os.path.join(output_dir, MERGED_ADAPTER_CONFIG_DIR))
    return merged_model
