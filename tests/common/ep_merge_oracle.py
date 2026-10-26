"""Whole-dict oracle for the streamed EP-shard merge.

``merge_ep_shards`` streams: it groups, transforms and renames one MoE layer at a time so peak host
RAM is a single layer, which lets a 397B artifact merge on an ordinary host. Streaming failures occur
at group or flush boundaries, which an assertion about one layer alone cannot see. This module
applies the same grouping and the same family layout to the whole state dict at once, and the
equivalence test pins the streamed output to it key for key and byte for byte.

Test-only. The tool never calls this; a second in-tool path holding the whole model in memory would
defeat the streaming the tool exists for.
"""

from halo.checkpoint.ep_shard_merge import group_expert_weights, resolve_merge_transform
from halo.models.moe.expert_layout import to_hub_expert_tensors


def post_process_merged_weights(
    merged_weights: dict,
    model_type: str,
    verbose: bool = True,
    had_expert_shards: bool = False,
) -> dict:
    """Convert merged EP weights from runtime format to HuggingFace checkpoint format.

    EP wrappers store expert weights in matmul convention with architecture-specific parameter names.
    This groups expert weights by MoE layer and applies the ``ExpertLayout`` of the family spec the
    checkpoint's ``model_type`` resolves to. ``had_expert_shards`` says whether the caller
    actually merged ``.shard_N`` keys, which turns "nothing matched" from a pass-through into an
    error.
    """
    expert_groups, result = group_expert_weights(merged_weights)

    spec = resolve_merge_transform(model_type, bool(expert_groups), had_expert_shards, len(merged_weights), verbose)
    if spec is None:
        return merged_weights

    if verbose:
        print(f"\nPost-processing {len(expert_groups)} MoE layers for {model_type}")

    for prefix, params in sorted(expert_groups.items()):
        hub = to_hub_expert_tensors(spec.expert_layout, params, device="cpu")
        transformed = {f"{prefix}.{key}": tensor for key, tensor in hub.items()}
        result.update(transformed)
        if verbose:
            for new_key, new_tensor in sorted(transformed.items()):
                print(f"  {new_key}: {list(new_tensor.shape)}")

    # Per-rank shards carry live module names while the gather writes hub names; mirror it so a merged
    # checkpoint matches a gathered one.
    return {spec.to_hub_key(key): tensor for key, tensor in result.items()}
