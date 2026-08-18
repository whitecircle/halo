"""Re-attach a multimodal base's vision tower to a text-only export, restoring the wrapper layout.

A ``text_only_model`` run trains and exports the CausalLM sibling of a VLM checkpoint (``model.*``
keys, ``*_text`` config, no vision tower), which vLLM 0.26.0 cannot load natively: its registry
carries only the multimodal wrapper class for Qwen3.5/3.6. (SGLang 0.5.17 registers the text
CausalLM classes and loads the export as-is.) This script rebuilds the wrapper-layout artifact: the
trained text weights are re-prefixed to ``model.language_model.*``, the untrained vision tower
(``model.visual.*``) and any MTP tail (``mtp.*``) stream back in from the base checkpoint, and the
composite config is regrafted with the trained text config, so the served model runs the trained
text weights (router balancing included) under the class the engines register.

The input must be a text-only export (its ``model_type`` names the ``*_text`` sub-config); a
wrapper-layout checkpoint needs no re-attachment and is refused. The base must be the multimodal
checkpoint the run trained from (same architecture); a base with no vision keys is refused.

Usage:
    python scripts/after_training/reattach_vision_tower.py \\
        --input_dir <text-only export> --model_id <multimodal base repo or dir> --output_dir <dst>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil

from transformers import AutoConfig

import src.distributed.expert_parallel.layers.roster  # noqa: F401 — registers the EP export roster the config finalizer requires
from scripts._common import add_hub_source_args, add_max_shard_size_arg, add_trust_remote_code_arg
from src.checkpoint.config_export import finalize_exported_config
from src.checkpoint.format import (
    DEFAULT_MAX_SHARD_SIZE,
    HF_STREAM_PART_PREFIX,
    copy_checkpoint_aux_files,
)
from src.checkpoint.shard_writer import StageShardWriter
from src.checkpoint.tool_io import (
    checkpoint_shard_files,
    iter_checkpoint_shard_entries,
    iter_checkpoint_tensors,
    preflight_resource_warning,
    reject_in_place_conversion,
    resolve_checkpoint_source,
)
from src.log import configure_cli_logging

configure_cli_logging()
logger = logging.getLogger(__name__)

TEXT_PREFIX = "model."
WRAPPER_TEXT_PREFIX = "model.language_model."
_PROCESSOR_FILES = ("processor_config.json", "preprocessor_config.json", "video_preprocessor_config.json")


def _require_text_only_export(input_dir: str, trust_remote_code: bool) -> AutoConfig:
    config = AutoConfig.from_pretrained(input_dir, trust_remote_code=trust_remote_code)
    if getattr(config, "vision_config", None) is not None or not str(config.model_type).endswith("_text"):
        raise ValueError(
            f"{input_dir} is not a text-only export (model_type={config.model_type!r}): it already "
            f"carries the multimodal layout and needs no re-attachment."
        )
    return config


def _grafted_composite_config(base_dir: str, text_config: AutoConfig, trust_remote_code: bool):
    """The base's composite config with its text sub-config replaced by the trained one."""
    composite = AutoConfig.from_pretrained(base_dir, trust_remote_code=trust_remote_code)
    if getattr(composite, "vision_config", None) is None:
        raise ValueError(
            f"the base checkpoint's config ({base_dir}) declares no vision_config — it is not the "
            f"multimodal checkpoint this export trained from."
        )
    # The graft is a plain attribute assignment, so the base is checked to be the family the export
    # came from: a text tower grafted into another family's wrapper yields a config that loads but
    # serves incorrect output.
    base_text_type = composite.get_text_config().model_type
    if base_text_type != text_config.model_type:
        raise ValueError(
            f"the base checkpoint ({base_dir}) wraps a {base_text_type!r} text tower, but the export is "
            f"{text_config.model_type!r} — not the multimodal checkpoint this export trained from."
        )
    composite.text_config = text_config
    return composite


def reattach_vision_tower(
    input_dir: str,
    model_id: str,
    output_dir: str,
    *,
    revision: str | None = None,
    trust_remote_code: bool = True,
    max_shard_size: str = DEFAULT_MAX_SHARD_SIZE,
) -> None:
    reject_in_place_conversion(input_dir, output_dir)
    text_config = _require_text_only_export(input_dir, trust_remote_code)
    base_dir = resolve_checkpoint_source(model_id, revision)
    # The base is streamed from while the writer's close removes every model*.safetensors it did not
    # write, so an output aimed at the base directory would replace the base with the artifact.
    reject_in_place_conversion(base_dir, output_dir)
    composite = _grafted_composite_config(base_dir, text_config, trust_remote_code)

    input_shards = checkpoint_shard_files(input_dir)
    base_shards = checkpoint_shard_files(base_dir)

    # Header-only, ahead of the output directory. The trained export supersedes the base's text tower
    # only where the base stores it under the wrapper prefix the export is re-prefixed to; a base
    # keeping a vendor namespace (renamed inside from_pretrained) would be carried whole beside the
    # trained tower, giving two text towers that collide on load. A base with nothing else to
    # contribute is not the multimodal checkpoint this export trained from.
    base_keys = [key for _shard, _reader, key in iter_checkpoint_shard_entries(base_dir)]
    superseded = {key for key in base_keys if key.startswith((WRAPPER_TEXT_PREFIX, "lm_head."))}
    if not any(key.startswith(WRAPPER_TEXT_PREFIX) for key in superseded):
        raise ValueError(
            f"the base checkpoint ({base_dir}) stores no text tower under {WRAPPER_TEXT_PREFIX}*, so the "
            f"trained export cannot supersede it — its text keys would be carried over beside the "
            f"trained ones. Re-attach onto a base saved in the wrapper layout."
        )
    if len(superseded) == len(base_keys):
        raise ValueError(
            f"the base checkpoint ({base_dir}) contributes no non-text tensors — nothing to "
            f"re-attach. Its weight layout does not match the expected multimodal wrapper."
        )

    writer = StageShardWriter(output_dir, HF_STREAM_PART_PREFIX, max_shard_size, enabled=True)
    preflight_resource_warning(
        "reattach_vision_tower",
        output_dir,
        disk_bytes=sum(os.path.getsize(s) for s in input_shards + base_shards),
        ram_bytes=writer.max_bytes,
    )
    os.makedirs(output_dir, exist_ok=True)

    trained = 0
    for key, tensor in iter_checkpoint_tensors(input_dir):
        if key.startswith(TEXT_PREFIX):
            key = WRAPPER_TEXT_PREFIX + key[len(TEXT_PREFIX) :]
        writer.add(key, tensor)  # lm_head.* and any other root keys pass through verbatim
        trained += 1

    carried = 0
    for key, tensor in iter_checkpoint_tensors(base_dir, predicate=lambda key: key not in superseded):
        writer.add(key, tensor)
        carried += 1
    writer.close_as_hf_checkpoint()

    # Tokenizer, template and similar files follow the trained export; the processor files come from
    # the base, which is the only source for them.
    copy_checkpoint_aux_files(input_dir, output_dir)
    for name in _PROCESSOR_FILES:
        src = os.path.join(base_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(output_dir, name))
    # The composite config's own architectures already name the wrapper class; write it last so the
    # artifact's config matches the wrapper-layout keys on disk.
    composite.save_pretrained(output_dir)
    finalize_exported_config(composite, output_dir, source=base_dir)
    logger.info(
        f"✓ Re-attached vision tower: {trained} trained tensor(s) re-prefixed, {carried} base "
        f"tensor(s) carried over -> {output_dir} ({json.dumps(composite.architectures)})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-attach a multimodal base's vision tower to a text-only export.")
    parser.add_argument("--input_dir", required=True, help="Text-only export directory.")
    add_hub_source_args(parser, source="The multimodal base checkpoint the run trained from")
    parser.add_argument("--output_dir", required=True, help="Output directory (wrapper layout).")
    add_max_shard_size_arg(parser)
    # Both configs load through it, and --model_id may be a Hub repo, so it is opt-in as on every
    # sibling whose source can be one.
    add_trust_remote_code_arg(parser, default=False)
    args = parser.parse_args()
    reattach_vision_tower(
        args.input_dir,
        args.model_id,
        args.output_dir,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
        max_shard_size=args.max_shard_size,
    )


if __name__ == "__main__":
    main()
