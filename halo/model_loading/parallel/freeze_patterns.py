"""Partial fine-tuning by name pattern: freeze or unfreeze the matching parameters of a loaded model."""

import fnmatch

from halo.distributed.pipeline_parallel.stage import PP_STAGE_PARTITION_ATTR
from halo.models.module_tree import DECODER_LAYER_LIST_ATTRS
from halo.runtime.consensus import reject_across_ranks


def _pins_a_decoder_layer_index(pattern: str) -> bool:
    """Whether a pattern selects decoder layers by position: the segment after a layer list carries a digit, while
    ``*.mlp.experts.0.*`` and ``model.layers.*`` mean the same on every stage."""
    segments = pattern.split(".")
    return any(
        segment in DECODER_LAYER_LIST_ATTRS and any(char.isdigit() for char in next_segment)
        for segment, next_segment in zip(segments, segments[1:], strict=False)
    )


def _reject_layer_indexed_patterns_under_pp(model, patterns, flag: str) -> None:
    """Reject layer-indexed patterns on a pipeline stage, whose layers are re-based to index 0: a global
    index matches nothing on most stages and the wrong layer on the rest."""
    if getattr(model, PP_STAGE_PARTITION_ATTR, None) is None:
        return
    indexed = [pattern for pattern in patterns if _pins_a_decoder_layer_index(pattern)]
    if indexed:
        raise ValueError(
            f"{flag} carries layer-indexed patterns {indexed} under pipeline parallelism. Each stage "
            f"holds only its own layers, re-based to index 0, so a global layer index selects nothing "
            f"on most stages and the wrong layer on the rest. Use index-free patterns (e.g. "
            f"'*.self_attn.sinks'), or train without pipeline parallelism to target layers by index."
        )


def freeze_modules_by_patterns(model, patterns):
    """COLLECTIVE: freeze parameters whose full names match an fnmatch pattern. Raises on every rank when a pattern
    matches nothing, as a typo would leave what it meant to hold still training with no other symptom."""
    _reject_layer_indexed_patterns_under_pp(model, patterns, "freeze_layers_patterns")
    unmatched = set(patterns)
    for param_name, param in model.named_parameters():
        for pattern in patterns:
            if fnmatch.fnmatch(param_name, pattern):
                param.requires_grad = False
                unmatched.discard(pattern)

    reason = None
    if unmatched:
        param_names = [name for name, _ in model.named_parameters()]
        reason = (
            f"freeze_layers_patterns {sorted(unmatched)} matched no parameter of "
            f"{type(model).__name__}, so whatever they name would keep training — a typo here has "
            f"no other symptom. Patterns are fnmatch against full PARAMETER names (not module "
            f"names, unlike unfreeze_layers_patterns); e.g. {param_names[:5]}. Under pipeline "
            f"parallelism this model is one stage, so name only parameters every stage carries."
        )
    # Under PP a stage-local raise would leave peer stages hanging in the next collective.
    reject_across_ranks(reason, "freeze_layers_patterns", ValueError)


def unfreeze_modules_by_patterns(model, patterns):
    """COLLECTIVE: freeze every parameter, then unfreeze modules whose full names match an fnmatch pattern. Raises on
    every rank when nothing is left trainable, as the optimizer accepts empty parameter groups."""
    _reject_layer_indexed_patterns_under_pp(model, patterns, "unfreeze_layers_patterns")
    for param in model.parameters():
        param.requires_grad = False

    for module_name, module in model.named_modules():
        for pattern in patterns:
            if fnmatch.fnmatch(module_name, pattern):
                for param in module.parameters():
                    param.requires_grad = True
                break

    reason = None
    if not any(param.requires_grad for param in model.parameters()):
        module_names = [name for name, _ in model.named_modules() if name]
        reason = (
            f"unfreeze_layers_patterns {list(patterns)} matched no parameter-bearing module of "
            f"{type(model).__name__}, so every parameter stayed frozen and the run would train "
            f"nothing. Patterns are fnmatch against full MODULE names (not parameter names); e.g. "
            f"{module_names[:5]}."
        )
    reject_across_ranks(reason, "unfreeze_layers_patterns", ValueError)
