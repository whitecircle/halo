"""Merge a per-rank EP-sharded save into one ``from_pretrained``-loadable checkpoint, streaming one MoE layer's experts
at a time through the family's ``ExpertLayout``, the transform the gathered save applies, and carry its resume
state."""

import logging
import os
import re
from collections import defaultdict
from contextlib import ExitStack

import torch
from safetensors import safe_open

from halo.checkpoint.atomic import link_or_copy_file
from halo.checkpoint.format import DEFAULT_MAX_SHARD_SIZE, EP_SHARD_KEY_RE, ep_shard_filename, is_numbered_shard_file
from halo.checkpoint.reading import detect_model_type, header_nbytes, is_ep_sharded_checkpoint, read_checkpoint_index
from halo.checkpoint.tools.offline_export import (
    copy_checkpoint_aux_files,
    cross_device_bytes,
    preflight_resource_warning,
    reject_in_place_conversion,
    reject_sibling_adapter,
    resume_state_sources,
)
from halo.checkpoint.writing import StreamingShardWriter
from halo.models.moe.expert_layout import to_hub_expert_tensors
from halo.models.moe.family_specs import (
    MoEFamilySpec,
    expert_weight_roots,
    find_moe_family_spec,
    moe_model_types,
    shared_expert_attrs,
)

logger = logging.getLogger(__name__)

# Derived so a new family's expert shards cannot pass through as non-expert weights.
EP_EXPERT_SUFFIXES = tuple(sorted(expert_weight_roots(), key=lambda s: (-len(s), s)))

# Captures (prefix_with_container, param_suffix); the container (mlp, feed_forward, the block itself) stays verbatim so
# transforms re-emit experts under it.
EP_EXPERT_PATTERN = re.compile(r"^(.+)\.(" + "|".join(re.escape(s) for s in EP_EXPERT_SUFFIXES) + r")$")


def group_expert_weights(merged_weights: dict) -> tuple[dict, dict]:
    """Split into ``{layer_prefix: {param_suffix: value}}`` expert groups and the rest; values pass through untouched,
    so this classifies tensors and read plans alike."""
    expert_groups = defaultdict(dict)
    non_expert = {}

    shared = shared_expert_attrs()
    for key, value in merged_weights.items():
        match = EP_EXPERT_PATTERN.match(key)
        # A shared expert's raw parameters can share a routed suffix; they are never sharded.
        if match and match.group(1).rsplit(".", 1)[-1] not in shared:
            layer_prefix = match.group(1)
            param_suffix = match.group(2)
            expert_groups[layer_prefix][param_suffix] = value
        else:
            non_expert[key] = value

    return dict(expert_groups), non_expert


def resolve_merge_family_spec(
    model_type: str, has_expert_groups: bool, had_expert_shards: bool, total_tensors: int, verbose: bool
) -> MoEFamilySpec | None:
    """The family spec whose ``expert_layout`` this checkpoint merges into, or ``None`` for pass-through; raises rather
    than write an unloadable checkpoint."""
    if not has_expert_groups:
        # Only expert params are sharded, so merged shards matching nothing means the suffix union missed this family.
        if had_expert_shards:
            raise ValueError(
                f"Merged {total_tensors} tensors from per-rank expert shards, but none matched an "
                f"EP expert attribute ({', '.join(EP_EXPERT_SUFFIXES)}). The merged checkpoint would "
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


def _ep_save_layout(input_dir: str, verbose: bool) -> tuple[str, int, list[str]]:
    """``(model_type, ep_size, shard files)`` of a per-rank EP save, refusing any other directory."""
    model_type = detect_model_type(input_dir)
    if verbose:
        logger.info(f"Detected model type: {model_type}")

    metadata = read_checkpoint_index(input_dir, missing_ok=True).get("metadata", {})
    # A gathered (or PP/TP) checkpoint already carries HF-layout expert keys, so merging one doubles the
    # ``experts.experts.`` prefix and from_pretrained reads them as missing.
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

    # EP shards only, since a bare glob would pull in adapter_model.safetensors or a stale model.safetensors.
    shard_files = sorted(f for f in os.listdir(input_dir) if is_numbered_shard_file(f))
    if not shard_files:
        raise FileNotFoundError(f"No EP shards ({ep_shard_filename(0, ep_size)} and siblings) found in {input_dir}")
    if verbose:
        logger.info(f"  Shard files: {shard_files}")
    return model_type, ep_size, shard_files


class _RankShards:
    """A per-rank EP save read lazily: a logical key is one tensor or ``ep_size`` slices joined on the expert dim."""

    def __init__(self, key_reader: dict, shard_groups: dict[str, dict[int, str]], verbose: bool):
        self.key_reader = key_reader
        self.shard_groups = shard_groups
        self.verbose = verbose

    def logical_keys(self) -> dict:
        """Passthrough keys first, concatenated groups after, the order loading every file into one dict produces."""
        logical = dict.fromkeys(key for key in self.key_reader if not EP_SHARD_KEY_RE.match(key))
        logical.update(dict.fromkeys(self.shard_groups))
        return logical

    def read(self, name: str) -> torch.Tensor:
        tensors = [self.key_reader[key].get_tensor(key) for key in self._input_keys(name)]
        if name not in self.shard_groups:
            return tensors[0]
        merged = torch.cat(tensors, dim=0)
        if self.verbose:
            logger.info(f"  Merged {name}: {len(tensors)} shards -> {list(merged.shape)}")
        return merged

    def nbytes(self, name: str) -> int:
        return sum(header_nbytes(self.key_reader[key].get_slice(key)) for key in self._input_keys(name))

    def _input_keys(self, name: str) -> tuple[str, ...]:
        if name in self.shard_groups:
            shards = self.shard_groups[name]
            return tuple(shards[i] for i in sorted(shards))
        return (name,)


def _open_rank_shards(
    stack: ExitStack, input_dir: str, shard_files: list[str], ep_size: int, verbose: bool
) -> _RankShards:
    """Open every input shard once on ``stack`` and group the per-rank slices, refusing a group missing a rank. Reads
    are lazy, so peak host RAM is one gathered MoE layer."""
    key_reader = {}
    for name in shard_files:
        reader = stack.enter_context(safe_open(os.path.join(input_dir, name), framework="pt"))
        keys = reader.keys()
        for key in keys:
            key_reader[key] = reader
        if verbose:
            logger.info(f"  Opened {name}: {len(keys)} tensors")

    shard_groups: dict[str, dict[int, str]] = defaultdict(dict)
    for key in key_reader:
        match = EP_SHARD_KEY_RE.match(key)
        if match:
            shard_groups[match.group(1)][int(match.group(2))] = key

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
    return _RankShards(key_reader, dict(shard_groups), verbose)


def _write_merged_tensors(
    shards: _RankShards,
    expert_groups: dict,
    non_expert: dict,
    spec: MoEFamilySpec | None,
    writer: StreamingShardWriter,
    model_type: str,
    verbose: bool,
) -> int:
    """Stream the non-expert tensors, then each layer through ``ExpertLayout``, under hub spellings; returns the tensor
    count. Written at the stored dtype, since the sharded writer already cast and a name-only cast here has no module
    tree to derive fp32 pins from (GLM-5 Next's ``A_log``/``dt_bias``, Inkling's short convolutions, DeepSeek-V4's
    ``attn_hc``)."""

    def emit(state: dict) -> None:
        if spec is not None:
            state = {spec.to_hub_key(key): tensor for key, tensor in state.items()}
        for key, tensor in state.items():
            writer.add(key, tensor)

    for name in non_expert:
        emit({name: shards.read(name)})

    if verbose and expert_groups:
        logger.info(f"Post-processing {len(expert_groups)} MoE layers for {model_type}")
    total_tensors = len(non_expert)
    for prefix, params in sorted(expert_groups.items()):
        merged_params = {suffix: shards.read(f"{prefix}.{suffix}") for suffix in params}
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
    return total_tensors


def merge_ep_shards(
    input_dir: str,
    output_dir: str,
    verbose: bool = True,
    max_shard_size: str = DEFAULT_MAX_SHARD_SIZE,
    delete_input_shards: bool = False,
):
    """Merge an EP-sharded checkpoint into one model, one tensor group at a time. Resume files and the optimizer meta
    are hard-linked where the filesystem allows, else copied; a training save into either directory replaces a linked
    file rather than writing through it."""
    # Every input gate runs before the output directory is created, so a refused merge writes nothing.
    reject_in_place_conversion(input_dir, output_dir)
    reject_sibling_adapter(input_dir)
    model_type, ep_size, shard_files = _ep_save_layout(input_dir, verbose)
    resume_files = resume_state_sources([input_dir])

    with ExitStack() as stack:
        shards = _open_rank_shards(stack, input_dir, shard_files, ep_size, verbose)
        logical = shards.logical_keys()
        expert_groups, non_expert = group_expert_weights(logical)
        spec = resolve_merge_family_spec(
            model_type, bool(expert_groups), bool(shards.shard_groups), len(logical), verbose
        )

        writer = stack.enter_context(StreamingShardWriter(output_dir, max_shard_size, enabled=True))
        # Peak RAM is about twice the largest group on top of the writer's staged parts. On disk the artifact is the
        # input shards' size, as tensors keep their stored dtype, plus the resume state it cannot hard-link.
        largest_group = max(
            [sum(shards.nbytes(f"{prefix}.{suffix}") for suffix in params) for prefix, params in expert_groups.items()]
            + [shards.nbytes(name) for name in non_expert],
            default=0,
        )
        preflight_resource_warning(
            "merge_ep_shards",
            output_dir,
            disk_bytes=sum(os.path.getsize(os.path.join(input_dir, f)) for f in shard_files)
            + cross_device_bytes([os.path.join(input_dir, name) for name in resume_files], output_dir),
            ram_bytes=2 * largest_group + writer.max_resident_bytes,
        )

        os.makedirs(output_dir, exist_ok=True)
        total_tensors = _write_merged_tensors(shards, expert_groups, non_expert, spec, writer, model_type, verbose)
        if verbose:
            logger.info(f"Total merged weights: {total_tensors}")
        writer.close_as_hf_checkpoint()
    if verbose:
        logger.info(f"Saved merged weights to {output_dir}")

    # The optimizer meta lands after every shard and before the aux copy brings trainer_state.json, so a
    # merge that died midway leaves a directory the resume refuses rather than one that warm-restarts.
    for name in resume_files:
        link_or_copy_file(os.path.join(input_dir, name), os.path.join(output_dir, name))
    # The inputs go LAST, so a merge that died earlier leaves the only copy where it found it. generation_config.json
    # rides across verbatim, since the format gate admits only this toolkit's writer, which emits it off a sanitized
    # config.
    copy_checkpoint_aux_files(input_dir, output_dir, carried=resume_files, verbose=verbose)
    if delete_input_shards:
        for shard_file in shard_files:
            os.remove(os.path.join(input_dir, shard_file))
        if verbose:
            logger.info(f"Deleted {len(shard_files)} input shard files (--delete_input_shards)")
