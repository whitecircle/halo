#!/usr/bin/env python
"""Merge an EP-sharded checkpoint into one ``from_pretrained``-loadable HuggingFace checkpoint.

Required for sharded EP saves (``save_sharded_ep: true``): each rank writes its own expert slices
under ``.shard_N`` suffixes for fast parallel I/O, a layout no loader reads (the EP loader refuses it
with these merge instructions). The merge writes each family's ``ExpertLayout`` through the same
``to_hub_expert_tensors`` the gathered save applies, so merged-from-sharded equals gathered.
A gathered save (``save_sharded_ep: false``) needs no merge; it is slower to write but directly
loadable. The merge carries the save's resume state, optimizer shards included, so the merged directory
resumes exactly on the topology that wrote it.

The merge is streamed: input shards are opened once and read group by group (one MoE layer's
experts, or one replicated tensor, at a time), each merged group is transformed and handed straight
to an incremental shard writer. Peak host RAM is one merged layer plus one pending output shard
rather than the whole model, so a 397B artifact merges on an ordinary host.

Output is HuggingFace-style sharded safetensors (``model-0000N-of-000XX.safetensors`` +
``model.safetensors.index.json``); ``--max_shard_size`` sets the per-file cap.

Usage:
    python scripts/after_training/merge_ep_shards.py \\
        --input_dir /path/to/ep_checkpoint --output_dir /path/to/merged_checkpoint
"""

import argparse
import logging
import os
import re
import shutil
from collections import defaultdict
from contextlib import ExitStack

import torch
from safetensors import safe_open

from halo.checkpoint.format import DEFAULT_MAX_SHARD_SIZE, EP_SHARD_KEY_RE, ep_shard_filename, is_ep_shard
from halo.checkpoint.offline_export import (
    copy_checkpoint_aux_files,
    preflight_resource_warning,
    reject_in_place_conversion,
    reject_sibling_adapter,
    resume_state_sources,
)
from halo.checkpoint.reading import detect_model_type, header_nbytes, is_ep_sharded_checkpoint, read_checkpoint_index
from halo.checkpoint.writing import StreamingShardWriter
from halo.log import configure_cli_logging
from halo.models.moe.expert_layout import to_hub_expert_tensors
from halo.models.moe.family_specs import (
    MoEFamilySpec,
    expert_weight_roots,
    find_moe_family_spec,
    moe_model_types,
    shared_expert_attrs,
)
from scripts.cli_arguments import add_max_shard_size_arg

configure_cli_logging()
logger = logging.getLogger(__name__)

# Union of every family's expert-weight roots (MoEFamilySpec.expert_weight_roots), derived rather than
# hand-listed so a new family's expert shards cannot pass through as non-expert weights.
_EP_EXPERT_SUFFIXES = sorted(expert_weight_roots(), key=lambda s: (-len(s), s))

# Expert keys are ...{container}.{suffix}. The container is captured verbatim (mlp, feed_forward, or
# the block itself) so transforms re-emit experts under the same container.
# Captures (prefix_with_container, param_suffix).
_EP_EXPERT_PATTERN = re.compile(r"^(.+)\.(" + "|".join(re.escape(s) for s in _EP_EXPERT_SUFFIXES) + r")$")


def _group_expert_weights(merged_weights: dict) -> tuple[dict, dict]:
    """Separate expert weights from non-expert weights and group by MoE layer.

    Values pass through untouched, so this classifies tensors and read plans alike (the streaming
    merge groups keys before any tensor is materialized). A declared shared-expert container
    (:func:`~halo.models.moe.family_specs.shared_expert_attrs`) is not an expert group,
    whatever its parameters are called.

    Returns:
        expert_groups: {layer_prefix: {param_suffix: value}} for expert weights
        non_expert: {key: value} for all other weights
    """
    expert_groups = defaultdict(dict)
    non_expert = {}

    shared = shared_expert_attrs()
    for key, value in merged_weights.items():
        match = _EP_EXPERT_PATTERN.match(key)
        # A shared expert's raw parameters can share a routed suffix; they are never sharded.
        if match and match.group(1).rsplit(".", 1)[-1] not in shared:
            layer_prefix = match.group(1)
            param_suffix = match.group(2)
            expert_groups[layer_prefix][param_suffix] = value
        else:
            non_expert[key] = value

    return dict(expert_groups), non_expert


def _resolve_merge_transform(
    model_type: str, has_expert_groups: bool, had_expert_shards: bool, total_tensors: int, verbose: bool
) -> MoEFamilySpec | None:
    """The family spec whose ``expert_layout`` this checkpoint merges into, or ``None`` for pass-through.

    Raises the two dead ends instead of writing an unloadable checkpoint: merged shards that match
    no declared expert attribute, and a ``model_type`` no family spec claims.
    """
    if not has_expert_groups:
        # Only expert params are sharded, so merged shards matching nothing means the suffix union
        # missed this family.
        if had_expert_shards:
            raise ValueError(
                f"Merged {total_tensors} tensors from per-rank expert shards, but none matched an "
                f"EP expert attribute ({', '.join(_EP_EXPERT_SUFFIXES)}). The merged checkpoint would "
                f"carry EP-internal names that from_pretrained cannot load. This family's expert "
                f"attributes are missing from its spec's expert_weight_roots."
            )
        if verbose:
            logger.info("No EP expert weights detected, skipping post-processing")
        return None

    spec = find_moe_family_spec(model_type)
    if spec is None:
        raise ValueError(
            f"merge_ep_shards cannot resolve model_type '{model_type}' to a MoE family spec, so it has "
            f"no HF-layout transform; merging would produce an unloadable checkpoint. Supported: "
            f"{', '.join(sorted(moe_model_types()))}. Save with save_sharded_ep: false (gathered, "
            "the default) — that writes a directly loadable checkpoint via each layer's "
            "gather_layer_state_dict()."
        )
    return spec


def merge_ep_shards(
    input_dir: str,
    output_dir: str,
    verbose: bool = True,
    max_shard_size: str = DEFAULT_MAX_SHARD_SIZE,
    delete_input_shards: bool = False,
):
    """
    Merge EP-sharded checkpoint into a single model, streaming one tensor group at a time.

    Args:
        input_dir: Path to EP-sharded checkpoint
        output_dir: Path to save merged checkpoint
        verbose: Log progress messages
        max_shard_size: Maximum shard size for output safetensors
        delete_input_shards: Delete input shard files after the merged checkpoint is fully saved
    """
    # Every input gate runs before the output directory is created, so a refused merge writes nothing.
    reject_in_place_conversion(input_dir, output_dir)
    reject_sibling_adapter(input_dir)

    model_type = detect_model_type(input_dir)
    if verbose:
        logger.info(f"Detected model type: {model_type}")

    metadata = read_checkpoint_index(input_dir, missing_ok=True).get("metadata", {})
    # A gathered (or PP/TP) checkpoint already carries HF-layout expert keys matching the pattern
    # below, so merging one doubles the ``experts.experts.`` prefix and from_pretrained then reads
    # those keys as missing.
    if not is_ep_sharded_checkpoint(input_dir) or "ep_size" not in metadata:
        raise ValueError(
            f"{input_dir} is not a per-rank EP-sharded checkpoint: no index there declares the "
            f"EP save's format marker and ep_size (index metadata {metadata!r}). Only a directory "
            f"written by a run with save_sharded_ep: true can be merged — a gathered EP save, a "
            f"pipeline-stage save, or an already-merged checkpoint is loadable as it stands, and "
            f"merging it would rewrite its expert weights into an unloadable layout."
        )
    ep_size = metadata["ep_size"]

    if verbose:
        logger.info("EP checkpoint info:")
        logger.info(f"  EP size: {ep_size}")
        logger.info(f"  Format: {metadata.get('format', 'unknown')}")

    # EP shards only: a bare .safetensors glob would pull in adapter_model.safetensors or a stale
    # model.safetensors.
    shard_files = sorted(f for f in os.listdir(input_dir) if is_ep_shard(f))
    if not shard_files:
        raise FileNotFoundError(f"No EP shards ({ep_shard_filename(0, ep_size)} and siblings) found in {input_dir}")
    if verbose:
        logger.info(f"  Shard files: {shard_files}")
    resume_files = resume_state_sources([input_dir])

    with ExitStack() as stack:
        # Open every input shard once and read tensors lazily: only the group being merged is ever
        # resident, so peak host RAM is one gathered MoE layer, not the model.
        readers = {
            name: stack.enter_context(safe_open(os.path.join(input_dir, name), framework="pt")) for name in shard_files
        }
        key_reader = {}
        for name, reader in readers.items():
            keys = reader.keys()
            for key in keys:
                key_reader[key] = reader
            if verbose:
                logger.info(f"  Opened {name}: {len(keys)} tensors")

        shard_groups: dict[str, dict[int, str]] = defaultdict(dict)
        passthrough: list[str] = []
        for key in key_reader:
            match = EP_SHARD_KEY_RE.match(key)
            if match:
                shard_groups[match.group(1)][int(match.group(2))] = key
            else:
                passthrough.append(key)

        for base_name, shards in shard_groups.items():
            if len(shards) != ep_size:
                raise ValueError(
                    f"Expert tensor '{base_name}' has {len(shards)} shards but the index reports "
                    f"ep_size={ep_size}. Concatenating an incomplete shard set would silently drop "
                    f"experts and corrupt the checkpoint. Ensure all {ep_size} per-rank shards "
                    f"({ep_shard_filename(0, ep_size)} through {ep_shard_filename(ep_size - 1, ep_size)}) "
                    f"are present in "
                    f"'{input_dir}'. On a non-shared multi-node filesystem the shards are scattered "
                    f"across nodes' local disks and must be gathered into one directory first — or "
                    f"save gathered (save_sharded_ep: false) instead."
                )

        # Logical output tensors in the reference order: passthrough keys first, concatenated shard
        # groups after, matching the order loading every file into one dict produces.
        logical = dict.fromkeys(passthrough)
        logical.update(dict.fromkeys(shard_groups))
        expert_groups, non_expert = _group_expert_weights(logical)
        spec = _resolve_merge_transform(model_type, bool(expert_groups), bool(shard_groups), len(logical), verbose)

        def input_keys(name: str) -> tuple[str, ...]:
            if name in shard_groups:
                shards = shard_groups[name]
                return tuple(shards[i] for i in sorted(shards))
            return (name,)

        def read_merged(name: str) -> torch.Tensor:
            """Materialize one logical tensor: a verbatim read, or the ep_size shard concat."""
            tensors = [key_reader[key].get_tensor(key) for key in input_keys(name)]
            if name not in shard_groups:
                return tensors[0]
            # dim 0 is the expert dimension
            merged = torch.cat(tensors, dim=0)
            if verbose:
                logger.info(f"  Merged {name}: {len(tensors)} shards -> {list(merged.shape)}")
            return merged

        def logical_nbytes(name: str) -> int:
            return sum(header_nbytes(key_reader[key].get_slice(key)) for key in input_keys(name))

        writer = StreamingShardWriter(output_dir, max_shard_size, enabled=True)

        # Peak RAM is about twice the largest group (the concatenated inputs plus the transform's
        # copies) on top of the writer's pending output shard. On disk the merged artifact is the
        # input shards' size, since tensors are re-emitted at their stored dtype, plus the resume state.
        largest_group = max(
            [
                sum(logical_nbytes(f"{prefix}.{suffix}") for suffix in params)
                for prefix, params in expert_groups.items()
            ]
            + [logical_nbytes(name) for name in non_expert],
            default=0,
        )
        preflight_resource_warning(
            "merge_ep_shards",
            output_dir,
            disk_bytes=sum(os.path.getsize(os.path.join(input_dir, f)) for f in (*shard_files, *resume_files)),
            ram_bytes=2 * largest_group + writer.max_bytes,
        )

        os.makedirs(output_dir, exist_ok=True)

        def emit(state: dict) -> None:
            """Rename to hub spelling and stage, at the dtype the shards store.

            The sharded writer already applied ``save_dtype_caster`` (the module-tree keep-set of
            norms, balancing tensors and the family's fp32 pins that the gathered save uses; nothing
            cast in a training checkpoint, whose fp32 masters the merge keeps so it still resumes), so
            the stored dtype is the one to write. A second, name-only cast here has no model tree to
            derive the pins from and would fold them to bf16 (GLM-5 Next's linear-attention
            ``A_log``/``dt_bias``, Inkling's short convolutions, DeepSeek-V4's ``attn_hc``), so
            merged-from-sharded would no longer match a gathered save.
            """
            if spec is not None:
                state = {spec.to_hub_key(key): tensor for key, tensor in state.items()}
            for key, tensor in state.items():
                writer.add(key, tensor)

        for name in non_expert:
            emit({name: read_merged(name)})

        if verbose and expert_groups:
            logger.info(f"Post-processing {len(expert_groups)} MoE layers for {model_type}")
        total_tensors = len(non_expert)
        for prefix, params in sorted(expert_groups.items()):
            merged_params = {suffix: read_merged(f"{prefix}.{suffix}") for suffix in params}
            transformed = {
                f"{prefix}.{key}": tensor
                for key, tensor in to_hub_expert_tensors(spec.expert_layout, merged_params, device="cpu").items()
            }
            del merged_params
            if verbose:
                for new_key, new_tensor in sorted(transformed.items()):
                    logger.info(f"  {new_key}: {list(new_tensor.shape)}")
            total_tensors += len(transformed)
            emit(transformed)
            del transformed

        if verbose:
            logger.info(f"Total merged weights: {total_tensors}")

        writer.close_as_hf_checkpoint()
    if verbose:
        logger.info(f"Saved merged weights to {output_dir}")

    # The optimizer meta lands after every shard and before the aux copy brings trainer_state.json, so a
    # merge that died midway leaves a directory the resume refuses rather than one that warm-restarts.
    for name in resume_files:
        shutil.copy2(os.path.join(input_dir, name), os.path.join(output_dir, name))
    # The inputs go LAST, once the merged weights and their index are on disk, so a merge that died
    # earlier leaves the only copy of the weights where it found them. generation_config.json rides
    # across verbatim: the format gate admits only this toolkit's sharded writer's output, which
    # emits that file only for a generating model off an already-sanitized config.
    copy_checkpoint_aux_files(input_dir, output_dir, verbose=verbose)
    if delete_input_shards:
        for shard_file in shard_files:
            os.remove(os.path.join(input_dir, shard_file))
        if verbose:
            logger.info(f"Deleted {len(shard_files)} input shard files (--delete_input_shards)")
    print(f"\n✓ Merged EP checkpoint saved to: {output_dir}")  # noqa: T201 — CLI-facing


def main():
    parser = argparse.ArgumentParser(
        description="Merge EP-sharded checkpoint into standard HuggingFace format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Required after a run with save_sharded_ep: true, whose per-rank shards load only once merged.
A run with save_sharded_ep: false writes a gathered checkpoint that needs no merge.

Examples:
    # Merge Qwen3 MoE EP checkpoint
    python scripts/after_training/merge_ep_shards.py \\
        --input_dir checkpoints/qwen3-moe-ep \\
        --output_dir checkpoints/qwen3-moe-merged

    # Merge GptOss EP checkpoint
    python scripts/after_training/merge_ep_shards.py \\
        --input_dir checkpoints/gpt-oss-ep \\
        --output_dir checkpoints/gpt-oss-merged
        """,
    )
    parser.add_argument("--input_dir", required=True, help="Path to the per-rank sharded checkpoint")
    parser.add_argument("--output_dir", required=True, help="Path to save the merged checkpoint")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    add_max_shard_size_arg(parser)
    parser.add_argument(
        "--delete_input_shards",
        action="store_true",
        help="Delete the input shard files after the merged checkpoint is fully saved (not before: "
        "a failed merge must not destroy the only copy of the weights). Peak disk usage therefore "
        "briefly holds input shards + merged output simultaneously.",
    )
    args = parser.parse_args()

    merge_ep_shards(
        args.input_dir,
        args.output_dir,
        verbose=not args.quiet,
        max_shard_size=args.max_shard_size,
        delete_input_shards=args.delete_input_shards,
    )


if __name__ == "__main__":
    main()
