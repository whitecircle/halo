"""Fold a saved PEFT adapter into its base model and write the merged checkpoint, behind the input gates
every tool writing from a saved adapter shares."""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Callable
from typing import NamedTuple

from peft import PeftConfig, PeftModel
from transformers import PreTrainedModel

from src.checkpoint.adapters import adapter_module_paths, assert_no_expert_lora_adapter
from src.checkpoint.format import DEFAULT_MAX_SHARD_SIZE, MERGED_ADAPTER_CONFIG_DIR, PROCESSOR_FILES
from src.checkpoint.offline_export import (
    preflight_model_load_resources,
    reject_in_place_conversion,
    save_full_checkpoint,
)
from src.checkpoint.reading import reject_partial_checkpoint
from src.checkpoint.sidecars import apply_training_sidecars, copy_training_sidecars
from src.model_loading.tokenizer_setup import resolve_peft_processing_class

logger = logging.getLogger(__name__)


class AdapterBaseModel(NamedTuple):
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
) -> AdapterBaseModel:
    """The base loaded through the class whose module tree the adapter's keys address.

    ``PeftModel.from_pretrained`` only warns when saved keys name no live module, then merges nothing. An
    adapter trained through a multimodal checkpoint's text-only class is met through that class; a key set
    neither class carries means the base does not belong to the adapter.
    """
    paths = adapter_module_paths(adapter_dir)
    base_model = load_base_model(base_model_path, excuse_task_head=excuse_task_head)
    absent = _paths_absent_from(base_model, paths)
    if not absent:
        return AdapterBaseModel(base_model, text_only=False)
    # A path both classes carry (a ``modules_to_save`` lm_head) resolves on the wrapper too, so any
    # absence is the text-only hypothesis.
    wrapper = type(base_model).__name__
    del base_model
    mismatch = (
        f"The adapter at {adapter_dir} addresses {len(absent)} module(s) that {wrapper} loaded from "
        f"{base_model_path} does not have (e.g. {absent[0]}). PEFT would merge nothing there and the tool "
        f"would write the bare base; the adapter and its base_model_name_or_path do not belong together."
    )
    try:
        text_only_model = load_base_model(base_model_path, excuse_task_head=excuse_task_head, text_only=True)
    except ValueError as no_text_only_class:
        raise ValueError(mismatch) from no_text_only_class
    if not _paths_absent_from(text_only_model, paths):
        log(f"The adapter addresses the text-only class of {wrapper}; merging as {type(text_only_model).__name__}.")
        return AdapterBaseModel(text_only_model, text_only=True)
    raise ValueError(mismatch)


def adapter_input_gates(adapter_dir: str, output_dir: str) -> PeftConfig:
    """Refuse a saved adapter a tool cannot convert into ``output_dir``, then return its config.

    The gates every tool writing from a saved adapter shares, each covering a failure that raises
    nothing: a partial directory (adapter and base), an output that is an input, and a native expert
    adapter ``merge_and_unload`` would drop.
    """
    reject_partial_checkpoint(adapter_dir)
    reject_in_place_conversion(adapter_dir, output_dir)
    assert_no_expert_lora_adapter(adapter_dir)

    peft_config = PeftConfig.from_pretrained(adapter_dir)
    base_model_path = peft_config.base_model_name_or_path
    if base_model_path and os.path.isdir(base_model_path):
        reject_in_place_conversion(base_model_path, output_dir)
        reject_partial_checkpoint(base_model_path)
    return peft_config


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

    The input gates run before any load. ``load_base_model`` may excuse a missing task head only when
    the adapter's ``modules_to_save`` declares one. ``prepare_for_save`` mutates the merged model before
    the save, which carries the base's aux files; the adapter's sidecars and config travel separately.
    """
    log = logger.info if verbose else lambda message: None

    peft_config = adapter_input_gates(adapter_dir, output_dir)
    base_model_path = peft_config.base_model_name_or_path

    preflight_model_load_resources(base_model_path, output_dir, tool=tool, device_map=device_map)

    log(f"Merging {adapter_dir} ({peft_config.peft_type}) into base model {base_model_path}...")
    base = load_base_for_adapter(
        adapter_dir,
        base_model_path,
        load_base_model,
        excuse_task_head=bool(getattr(peft_config, "modules_to_save", None)),
        log=log,
    )
    # A text-only merge writes what a text_only_model run exports: no multimodal processor files.
    processing_class = resolve_peft_processing_class(
        adapter_dir, base_model_path, text_only=base.text_only, trust_remote_code=trust_remote_code
    )
    if processing_class is None:
        raise OSError(f"Could not load a processor or tokenizer from {adapter_dir} or {base_model_path}")
    merged_model = PeftModel.from_pretrained(base.model, adapter_dir).merge_and_unload()

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
