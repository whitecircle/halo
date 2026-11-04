"""Whole-dict oracle for the streamed EP-shard merge: same grouping and family layout at once, pinned to the stream
byte for byte. Test-only: a second in-tool path holding the whole model would defeat the streaming the tool exists
for."""

from halo.checkpoint.tools.ep_shard_merge import group_expert_weights, resolve_merge_family_spec
from halo.models.moe.expert_layout import to_hub_expert_tensors


def post_process_merged_weights(
    merged_weights: dict,
    model_type: str,
    verbose: bool = True,
    had_expert_shards: bool = False,
) -> dict:
    """Convert merged EP weights to HF checkpoint format; ``had_expert_shards`` turns nothing matching from a
    pass-through into an error."""
    expert_groups, result = group_expert_weights(merged_weights)

    spec = resolve_merge_family_spec(model_type, bool(expert_groups), had_expert_shards, len(merged_weights), verbose)
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

    # Per-rank shards carry live module names while the gather writes hub names; mirror it so the two match.
    return {spec.to_hub_key(key): tensor for key, tensor in result.items()}
