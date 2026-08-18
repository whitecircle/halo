"""Whole-dict oracle for the streamed EP-shard merge.

``merge_ep_shards`` streams: it groups, transforms and renames one MoE layer at a time so peak host
RAM is a single layer, which is what lets a 397B artifact merge on an ordinary host. Streaming is
also where a merge goes wrong — at a group or flush boundary — and a bug there is invisible to any
assertion made about one layer alone. This module is the comparison that can see it: the same
grouping and the same class-owned transform applied to the WHOLE state dict at once, which the
equivalence test pins the streamed output to key for key and byte for byte.

Test-only. The tool itself never calls this; a second in-tool code path holding the whole model in
memory would defeat the streaming the tool exists for.
"""

from scripts.after_training.merge_ep_shards import _group_expert_weights, _resolve_merge_transform
from src.distributed.expert_parallel.expert_weights import to_hub_layer_key


def post_process_merged_weights(
    merged_weights: dict,
    model_type: str,
    verbose: bool = True,
    had_expert_shards: bool = False,
) -> dict:
    """Convert merged EP weights from runtime format to HuggingFace checkpoint format.

    EP wrappers store expert weights in matmul convention with architecture-specific parameter names.
    This groups expert weights by MoE layer and applies the transform registered for the EP layer
    class the checkpoint's ``model_type`` resolves to. ``had_expert_shards`` says whether the caller
    actually merged ``.shard_N`` keys, which turns "nothing matched" from a pass-through into an
    error.
    """
    expert_groups, result = _group_expert_weights(merged_weights)

    layer_cls = _resolve_merge_transform(
        model_type, bool(expert_groups), had_expert_shards, len(merged_weights), verbose
    )
    if layer_cls is None:
        return merged_weights

    if verbose:
        print(f"\nPost-processing {len(expert_groups)} MoE layers for {model_type}")

    for prefix, params in sorted(expert_groups.items()):
        transformed = layer_cls.merge_shards_to_hf(prefix, params)
        result.update(transformed)
        if verbose:
            for new_key, new_tensor in sorted(transformed.items()):
                print(f"  {new_key}: {list(new_tensor.shape)}")

    # Per-rank shards carry live MODULE names, the gather writes HUB names — mirror it so merged == gathered.
    return {to_hub_layer_key(key, layer_cls): tensor for key, tensor in result.items()}
