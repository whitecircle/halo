"""Fold a saved PEFT adapter into its base and write the merged checkpoint, behind the input gates every adapter tool
shares; the causal-LM and classification merge drivers."""

from __future__ import annotations

import contextlib
import itertools
import logging
import os
from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import torch
from peft import PeftConfig, PeftModel
from peft.utils import get_peft_model_state_dict
from transformers import PreTrainedModel

from halo.checkpoint.adapters import (
    adapter_module_paths,
    find_adapter_weight_file,
    find_expert_lora_adapter_shape,
    read_adapter_config,
    read_adapter_file,
    reject_expert_lora_adapter,
)
from halo.checkpoint.format import (
    ADAPTER_CONFIG_FILE,
    DEFAULT_MAX_SHARD_SIZE,
    MERGED_ADAPTER_CONFIG_DIR,
    PROCESSOR_FILES,
)
from halo.checkpoint.reading import reject_partial_checkpoint
from halo.checkpoint.sidecars import apply_training_sidecars, copy_training_sidecars
from halo.checkpoint.tools.adapter_stream_merge import merge_adapter_streaming
from halo.checkpoint.tools.offline_export import (
    cast_norms_to_fp32,
    load_task_model,
    preflight_model_load_resources,
    reject_in_place_conversion,
    resolve_adapter_base_revision,
    save_full_checkpoint,
)
from halo.log import KEY_PREVIEW_COUNT
from halo.model_loading.model_preparation import apply_family_patches
from halo.model_loading.tokenizer_setup import (
    grow_vocabulary,
    load_processing_class,
    resolve_peft_processing_class,
    resolve_tokenizer,
)
from halo.models.introspection.config_access import config_export_ready
from halo.models.introspection.param_roles import persistent_buffers

logger = logging.getLogger(__name__)

# A sequence-classification head's size where a merge names none.
DEFAULT_NUM_LABELS = 1


class _AdapterBaseModel(NamedTuple):
    """The base an adapter applies to, and whether it is a multimodal checkpoint's text-only class."""

    model: PreTrainedModel
    text_only: bool


def _paths_absent_from(model: PreTrainedModel, paths: set[str]) -> list[str]:
    absent = []
    for path in sorted(paths):
        try:
            model.get_submodule(path)
        except AttributeError:
            absent.append(path)
    return absent


def load_base_for_adapter(
    adapter_dir: str,
    base_model_path: str,
    load_base_model: Callable[..., PreTrainedModel],
    *,
    excuse_task_head: bool,
    log,
) -> _AdapterBaseModel:
    """The base loaded through the class the adapter's keys address, since an adapter tensor keyed to no live module
    applies to nothing, which ``PeftModel.from_pretrained`` only warns about. A key set neither class carries raises."""
    paths = adapter_module_paths(adapter_dir)
    base_model = load_base_model(base_model_path, excuse_task_head=excuse_task_head)
    absent = _paths_absent_from(base_model, paths)
    if not absent:
        return _AdapterBaseModel(base_model, text_only=False)
    # A path both classes carry (a ``modules_to_save`` lm_head) resolves on the wrapper too, so any
    # absence is the text-only hypothesis.
    wrapper = type(base_model).__name__
    del base_model
    mismatch = (
        f"The adapter at {adapter_dir} addresses {len(absent)} module(s) that {wrapper} of {base_model_path} does "
        f"not have (e.g. {absent[0]}), so their adapter tensors would apply to nothing: the adapter and its "
        f"base_model_name_or_path do not belong together. Set base_model_name_or_path in {ADAPTER_CONFIG_FILE} to "
        f"the checkpoint the adapter was trained on."
    )
    try:
        text_only_model = load_base_model(base_model_path, excuse_task_head=excuse_task_head, text_only=True)
    except ValueError as no_text_only_class:
        raise ValueError(mismatch) from no_text_only_class
    if not _paths_absent_from(text_only_model, paths):
        log(
            f"The adapter addresses the text-only class of {wrapper}; reading the base as "
            f"{type(text_only_model).__name__}."
        )
        return _AdapterBaseModel(text_only_model, text_only=True)
    raise ValueError(mismatch)


def _embedding_weights(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """The weight the input embedding and output head compute with, by role, a tied pair once, through wrappers."""
    weights: dict[str, torch.Tensor] = {}
    for role, module in (
        ("input embedding", model.get_input_embeddings()),
        ("output head", model.get_output_embeddings()),
    ):
        if module is not None and all(module.weight is not weight for weight in weights.values()):
            weights[role] = module.weight
    return weights


def _grown_vocabulary(model: PreTrainedModel, adapter_processor, base_processor) -> int | None:
    """The adapter tokenizer's length where the run grew ``model``'s vocabulary to it, else None: more ids than
    ``model`` has rows, over a base whose own tokenizer fits them."""
    if adapter_processor is None:
        return None
    num_tokens = len(resolve_tokenizer(adapter_processor))
    rows = model.get_input_embeddings().weight.shape[0]
    if num_tokens <= rows:
        return None
    if base_processor is not None and len(resolve_tokenizer(base_processor)) > rows:
        return None
    return num_tokens


def _load_adapter(model: PreTrainedModel, adapter_dir: str, log) -> PeftModel:
    """``PeftModel.from_pretrained``, then every tensor of the adapter file copied onto the live tensor a PEFT save
    writes under its key where the load left another value. PEFT's load respells adapter keys through the base's
    checkpoint renames (transformers' conversion mapping), and a rename that does not hold on a live name (Laguna's
    ``shared_experts``, Inkling 2's ``*_sconv.weight_proj``) leaves a key no module has, whose tensor it drops with a
    warning. Raises for a tensor the wrap builds nothing for, or at another shape."""
    peft_model = PeftModel.from_pretrained(model, adapter_dir)
    saved = read_adapter_file(find_adapter_weight_file(adapter_dir))
    live = dict(itertools.chain(peft_model.named_parameters(remove_duplicate=False), persistent_buffers(peft_model)))
    # One process holds the whole model, so PEFT's selection hands back the live tensors themselves.
    targets = get_peft_model_state_dict(peft_model, state_dict=live, save_embedding_layers=True)
    unmatched = sorted(saved.keys() - targets.keys())
    reshaped = sorted(key for key in saved.keys() & targets.keys() if saved[key].shape != targets[key].shape)
    if unmatched or reshaped:
        raise ValueError(
            f"The adapter at {adapter_dir} holds {len(unmatched)} tensor(s) the PEFT wrap of this base builds no "
            f"adapter tensor for {unmatched[:KEY_PREVIEW_COUNT]} and {len(reshaped)} at another shape "
            f"{reshaped[:KEY_PREVIEW_COUNT]}, which the merged model would go without. The wrap resolves "
            f"adapter_config.json's target_modules and modules_to_save on this base, and PEFT matches a "
            f"modules_to_save entry against whole module names: `gate` takes no `shared_expert_gate`, which an earlier "
            f"PEFT matched by suffix and saved. Name every module the file holds there, or merge with "
            f"merge_peft_adapters.py --streaming, which folds the file's tensors by their saved names."
        )
    missed = sorted(key for key, tensor in saved.items() if not torch.equal(targets[key], tensor.to(targets[key])))
    if missed:
        log(
            f"PEFT's load left {len(missed)} adapter tensor(s) at other values than {adapter_dir} holds (e.g. "
            f"{missed[0]}): it respells adapter keys through this model type's checkpoint renames. Copying them in "
            f"under their saved names."
        )
        with torch.no_grad():
            for key in missed:
                targets[key].copy_(saved[key])
    return peft_model


def apply_adapter(adapter_dir: str, model: PreTrainedModel, *, adapter_processor, base_processor, log) -> PeftModel:
    """The saved adapter loaded onto ``model`` (:func:`_load_adapter`), first grown to a vocabulary the run grew. The
    added rows start as NaN, so an unset row is never served from uninitialized memory. An adapter lacking them has
    them rebuilt, with a warning, as the checkpoint rows' mean, matching the run's frozen mean-resize draw (that mean
    plus 1e-9 of the rows' covariance). The family patches come first, as PEFT's load runs the target check a patch
    lets past (Nemotron-H's ``out_proj``); they stamp run-scoped fields on ``model.config``, which a save writes only
    inside :func:`config_export_ready`."""
    apply_family_patches(model.config, None)
    num_tokens = _grown_vocabulary(model, adapter_processor, base_processor)
    if num_tokens is None:
        return _load_adapter(model, adapter_dir, log)
    rows = model.get_input_embeddings().weight.shape[0]
    log(
        f"Growing the base's embeddings from {rows} to {num_tokens} rows, the length of the adapter's tokenizer, "
        "which the run grew them to."
    )
    grow_vocabulary(model, num_tokens, mean_resizing=False)
    with torch.no_grad():
        for weight in _embedding_weights(model).values():
            weight[rows:] = float("nan")
    peft_model = _load_adapter(model, adapter_dir, log)
    unset = {role: weight for role, weight in _embedding_weights(peft_model).items() if weight[rows:].isnan().any()}
    if unset:
        logger.warning(
            f"The adapter at {adapter_dir} was trained on a vocabulary grown from {rows} to {num_tokens} rows (its "
            f"tokenizer's length) but holds no value for the added rows of the {' and '.join(unset)}: it was saved "
            f"before adapter saves carried them. Rebuilding them as the mean of the checkpoint's rows: the run drew "
            f"them by mean resizing and kept them frozen, so they match its rows to within that draw."
        )
        with torch.no_grad():
            for weight in unset.values():
                weight[rows:] = weight[:rows].mean(dim=0, dtype=torch.float32).to(weight.dtype)
    return peft_model


def adapter_input_gates(adapter_dir: str, output_dir: str) -> PeftConfig:
    """Refuse a partial directory (adapter and base), an output that is an input, and a native expert adapter
    ``merge_and_unload`` would drop, each of which raises nothing otherwise. Returns the adapter's config."""
    reject_partial_checkpoint(adapter_dir)
    reject_in_place_conversion(adapter_dir, output_dir)
    reject_expert_lora_adapter(adapter_dir)

    peft_config = PeftConfig.from_pretrained(adapter_dir)
    base_model_path = peft_config.base_model_name_or_path
    if base_model_path and os.path.isdir(base_model_path):
        reject_in_place_conversion(base_model_path, output_dir)
        reject_partial_checkpoint(base_model_path)
    return peft_config


def adapter_base_trust(adapter_dir: str, trust_remote_code: bool | None) -> bool:
    """``trust_remote_code`` where stated, else whether the adapter's base is a local directory: that is the operator's
    own artifact, while a Hub id names a repo whose code must not run because an adapter points at it."""
    if trust_remote_code is not None:
        return trust_remote_code
    # Read raw, before the input gates, which give a directory PEFT's own config class cannot read a better refusal.
    config = read_adapter_config(adapter_dir, missing_ok=True) or {}
    base_model_path = config.get("base_model_name_or_path")
    return bool(base_model_path) and os.path.isdir(base_model_path)


def merge_adapter_into_base(
    adapter_dir: str,
    output_dir: str,
    *,
    load_base_model: Callable[..., PreTrainedModel],
    tool: str,
    base_revision: str | None = None,
    device_map: str | None = None,
    max_shard_size: str = DEFAULT_MAX_SHARD_SIZE,
    trust_remote_code: bool = False,
    prepare_for_save: Callable[[PreTrainedModel], None] | None = None,
    verbose: bool = True,
) -> PreTrainedModel:
    """Fold a saved PEFT adapter into its base and write the merged checkpoint. ``load_base_model`` takes the base's
    path and the Hub revision to read it at (the commit the adapter records, or ``base_revision``), and may excuse a
    missing task head only when ``modules_to_save`` declares one; ``prepare_for_save`` mutates the merged model before
    the save."""
    log = logger.info if verbose else lambda message: None

    peft_config = adapter_input_gates(adapter_dir, output_dir)
    base_model_path = peft_config.base_model_name_or_path
    revision = resolve_adapter_base_revision(base_model_path, peft_config.revision, base_revision)

    preflight_model_load_resources(base_model_path, output_dir, tool=tool, device_map=device_map)

    log(f"Merging {adapter_dir} ({peft_config.peft_type}) into base model {base_model_path}...")
    base = load_base_for_adapter(
        adapter_dir,
        base_model_path,
        partial(load_base_model, revision=revision),
        excuse_task_head=bool(getattr(peft_config, "modules_to_save", None)),
        log=log,
    )
    adapter_processor = load_processing_class(adapter_dir, trust_remote_code=trust_remote_code)
    base_processor = load_processing_class(base_model_path, trust_remote_code=trust_remote_code, revision=revision)
    # A text-only merge writes what a text_only_model run exports: no multimodal processor files.
    processing_class = resolve_peft_processing_class(adapter_processor, base_processor, text_only=base.text_only)
    if processing_class is None:
        raise OSError(f"Could not load a processor or tokenizer from {adapter_dir} or {base_model_path}")
    merged_model = apply_adapter(
        adapter_dir, base.model, adapter_processor=adapter_processor, base_processor=base_processor, log=log
    ).merge_and_unload()

    for action in apply_training_sidecars(merged_model, adapter_dir):
        log(action)
    if prepare_for_save is not None:
        prepare_for_save(merged_model)

    log(f"Saving merged model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    with config_export_ready(merged_model.config):
        save_full_checkpoint(
            merged_model,
            output_dir,
            processing_class=processing_class,
            source_dir=base_model_path,
            source_revision=revision,
            include_resume_sidecars=False,
            max_shard_size=max_shard_size,
        )
    if base.text_only:
        for name in PROCESSOR_FILES:
            with contextlib.suppress(FileNotFoundError):
                os.remove(os.path.join(output_dir, name))
    copy_training_sidecars(adapter_dir, output_dir)
    peft_config.save_pretrained(os.path.join(output_dir, MERGED_ADAPTER_CONFIG_DIR))
    return merged_model


def load_task_base_model(
    base_model_path: str,
    task: str,
    dtype: torch.dtype,
    device_map: str | None,
    num_labels: int,
    *,
    excuse_task_head: bool = False,
    text_only: bool = False,
    trust_remote_code: bool = False,
    revision: str | None = None,
):
    """The base loaded as ``task`` through :func:`load_task_model`'s coverage gate, since a truncated directory would
    random-initialize absent tensors and the merge ship them. A causal-LM base's head is always excused, since its
    widest class adds one an encoder checkpoint never carries."""
    model_kwargs = {
        "dtype": dtype,
        "trust_remote_code": trust_remote_code,
        "revision": revision,
    }
    if device_map:
        model_kwargs["device_map"] = device_map

    if task == "classification":
        model_kwargs["num_labels"] = num_labels
    else:
        excuse_task_head = True
    return load_task_model(
        base_model_path, task, excuse_task_head=excuse_task_head, text_only=text_only, **model_kwargs
    )


def reject_in_memory_options(*, task: str, device_map: str | None, fp32_norms: bool) -> None:
    """Refuse what only a loaded model can honour, for a streaming merge."""
    refused = []
    if task != "causal_lm":
        refused.append(f"--task {task}")
    if device_map:
        refused.append(f"--device_map {device_map}")
    if fp32_norms:
        refused.append("--fp32_norms")
    if refused:
        raise ValueError(
            f"The streaming merge loads no model, so it cannot honour {', '.join(refused)}. Drop them, or merge a "
            f"dense adapter in memory (without --streaming)."
        )


def merge_peft_adapter(
    adapter_dir: str,
    output_dir: str,
    task: str = "causal_lm",
    dtype: torch.dtype = torch.bfloat16,
    device_map: str | None = None,
    num_labels: int = DEFAULT_NUM_LABELS,
    max_shard_size: str = DEFAULT_MAX_SHARD_SIZE,
    trust_remote_code: bool | None = None,
    fp32_norms: bool = False,
    streaming: bool = False,
    base_revision: str | None = None,
    verbose: bool = True,
):
    """Merge a PEFT adapter into its base model and save the result; ``fp32_norms`` writes norm leaves in fp32, and an
    unset ``trust_remote_code`` follows the base (:func:`adapter_base_trust`). A Hub base is read at the commit the
    adapter records, which ``base_revision`` overrides (:func:`resolve_adapter_base_revision`). ``streaming``, and any
    adapter holding native expert LoRA, folds shard by shard with no model loaded (:func:`merge_adapter_streaming`),
    which keeps the base's stored dtypes and so ignores ``dtype`` and refuses ``device_map``, ``fp32_norms`` and a
    classification head."""
    trust_remote_code = adapter_base_trust(adapter_dir, trust_remote_code)
    if streaming or find_expert_lora_adapter_shape(adapter_dir) is not None:
        reject_in_memory_options(task=task, device_map=device_map, fp32_norms=fp32_norms)
        if verbose:
            logger.info("Streaming merge: tensors keep the base's stored dtype, so --dtype is not applied")
        merge_adapter_streaming(
            adapter_dir,
            output_dir,
            base_revision=base_revision,
            max_shard_size=max_shard_size,
            trust_remote_code=trust_remote_code,
            verbose=verbose,
        )
        return
    if verbose:
        logger.info(f"Task: {task}, dtype: {dtype}")

    def load_base_model(
        base_model_path: str, *, excuse_task_head: bool, text_only: bool = False, revision: str | None = None
    ):
        return load_task_base_model(
            base_model_path,
            task,
            dtype,
            device_map,
            num_labels,
            excuse_task_head=excuse_task_head,
            text_only=text_only,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )

    merged_model = merge_adapter_into_base(
        adapter_dir,
        output_dir,
        load_base_model=load_base_model,
        tool="merge_peft_adapters",
        base_revision=base_revision,
        device_map=device_map,
        max_shard_size=max_shard_size,
        trust_remote_code=trust_remote_code,
        prepare_for_save=cast_norms_to_fp32 if fp32_norms else None,
        verbose=verbose,
    )

    if verbose:
        total_params = sum(p.numel() for p in merged_model.parameters())
        logger.info(f"Merged model saved to: {output_dir}")
        logger.info(f"Total parameters: {total_params:,}")

    del merged_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
