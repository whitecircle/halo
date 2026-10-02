"""CP-GRPO numerical oracles shared by dense and expert-parallel GPU suites."""

from unittest.mock import patch

import torch

from src.data.spans import LABEL_IGNORE_INDEX
from src.distributed.context_parallel.config import cp_boundary_shift
from src.trainers.grpo.objective.offline import offline_loss
from tests.common.tolerances import TOL


def full_row_loss(logps, shifted_labels, advantages, group_sizes):
    """Independent unsharded GRPO reduction over exactly the supervised targets."""
    valid = shifted_labels != LABEL_IGNORE_INDEX
    token_loss = -logps * advantages.unsqueeze(1)
    weights = 1.0 / group_sizes.float()
    return ((token_loss * valid).sum(1) / valid.sum(1).clamp(min=1) * weights).sum() / weights.sum()


def ep_gradient_parameter_names(model, ep_layers):
    """Routed expert/router parameters by identity; shared experts retain the dense CP bounds."""
    routed_ids = {id(parameter) for layer in ep_layers for _, parameter in layer.expert_named_params()}
    for layer in ep_layers:
        router = layer._live_router_module()
        if router is not None:
            routed_ids.update(id(parameter) for parameter in router.parameters())
    return {name for name, parameter in model.named_parameters() if id(parameter) in routed_ids}


def gradient_agreement(
    actual,
    expected,
    *,
    cosine_min=TOL.cp_grad_cosine_min,
    norm_ratio_band=(1.0 - TOL.cp_grad_norm_rtol, 1.0 + TOL.cp_grad_norm_rtol),
):
    """Exact parameter coverage, each parameter's direction and scale, and total gradient scale."""
    same_keys = actual.keys() == expected.keys()
    if not same_keys or not expected:
        return same_keys, False, False, float("nan"), float("nan")
    cosines = []
    parameter_norms_match = True
    actual_squared_norm = 0.0
    expected_squared_norm = 0.0
    for name, reference in expected.items():
        current = actual[name]
        if current.shape != reference.shape or not current.isfinite().all() or not reference.isfinite().all():
            return same_keys, False, False, float("nan"), float("nan")
        reference_norm = reference.norm().item()
        current_norm = current.norm().item()
        actual_squared_norm += current_norm**2
        expected_squared_norm += reference_norm**2
        if reference_norm == 0.0:
            cosines.append(1.0 if current_norm == 0.0 else 0.0)
            parameter_norms_match &= current_norm == 0.0
        else:
            parameter_norms_match &= norm_ratio_band[0] < current_norm / reference_norm < norm_ratio_band[1]
            cosine = (
                (current.flatten() @ reference.flatten()).item() / (current_norm * reference_norm)
                if current_norm
                else 0.0
            )
            cosines.append(cosine)
    minimum_cosine = min(cosines)
    norm_ratio = (actual_squared_norm / expected_squared_norm) ** 0.5 if expected_squared_norm else float("inf")
    return (
        same_keys,
        minimum_cosine >= cosine_min,
        parameter_norms_match and norm_ratio_band[0] < norm_ratio < norm_ratio_band[1],
        minimum_cosine,
        norm_ratio,
    )


def optimizer_step_agreement(actual, expected, initial, **gradient_bounds):
    """Compare the applied update, not large unchanged weights that hide an omitted optimizer step."""
    if actual.keys() != expected.keys() or expected.keys() != initial.keys():
        return False
    actual_update = torch.cat([(actual[name] - initial[name]).flatten() for name in sorted(initial)])
    expected_update = torch.cat([(expected[name] - initial[name]).flatten() for name in sorted(initial)])
    _, direction_matches, norm_matches, _, _ = gradient_agreement(
        {"update": actual_update}, {"update": expected_update}, **gradient_bounds
    )
    return direction_matches and norm_matches


def boundary_loss_negative_control(scorer, model, ids, mask, advantages, group_sizes, full_logps, cp_config):
    """Dropping the shard-edge targets must break the same numeric loss oracle that passes normally.

    Only CP4's internal boundaries are supervised, so the mutation deletes the whole objective
    rather than diluting a few lost targets among dozens of other completion tokens.
    """
    if cp_config.cp_size != 4:
        raise ValueError("the boundary-only fixture requires CP4")
    boundary_labels = torch.full_like(ids, LABEL_IGNORE_INDEX)
    chunk = ids.size(1) // cp_config.cp_size
    boundary_labels[:, (2 * chunk, 3 * chunk)] = ids[:, (2 * chunk, 3 * chunk)]
    expected = full_row_loss(full_logps, boundary_labels[:, 1:], advantages, group_sizes)

    def score_loss():
        logps, shifted = scorer._cp_chunked_logps(model, ids, mask, boundary_labels)
        return offline_loss(
            -logps * advantages.unsqueeze(1),
            shifted != LABEL_IGNORE_INDEX,
            group_sizes,
            loss_type="grpo",
            max_completion_length=ids.size(1),
            cp_config=cp_config,
        )

    def drop_boundary(hidden, local_labels, _boundary_label, _is_last_rank):
        return cp_boundary_shift(hidden, local_labels, None, True)

    with torch.no_grad():
        correct = score_loss()
        with patch("src.distributed.context_parallel.config.cp_boundary_shift", side_effect=drop_boundary):
            broken = score_loss()
    correct_error = abs((correct - expected).item())
    broken_error = abs((broken - expected).item())
    return (
        correct_error < TOL.parallel_vs_baseline_loss_abs,
        broken_error > TOL.control_min_loss_shift(),
        correct_error,
        broken_error,
    )
