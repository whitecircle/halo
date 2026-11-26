"""Stand-ins for the online trainer's packed rounds: its real scoring, buffer and loss on a model, built without
``__init__``, and the build TRL's round runs around the old-policy pass."""

import torch

from halo.trainers.grpo.online import DistributedOnlineGRPOTrainer
from tests.common.length_deal import loss_host, stacked


def online_host(model, *, slots: int, budget: int, accelerator, packed: bool, loss_type: str = "dapo"):
    """The online trainer as :func:`~tests.common.length_deal.loss_host` builds the environmental one, packing or not."""
    host = loss_host(model, chunks=slots, accelerator=accelerator, loss_type=loss_type)
    host.__class__ = DistributedOnlineGRPOTrainer
    host._packed_microbatch_tokens = budget
    host._packed_dispatch_ceiling = None
    host._packs_microbatches = packed
    host._orders_rows_by_length = False
    host._round_store, host._round_plan = None, None
    host.pad_token_id = 0
    host.use_liger_kernel = False
    host._get_tp_or_etp_process_group = lambda: None
    return host


def old_policy_pass(host, batch: dict) -> torch.Tensor:
    """What TRL's round build does with the whole round: one no-grad scoring call at ``per_device_train_batch_size``."""
    ids, mask, keep = stacked(batch)
    with torch.no_grad():
        logps, _, _ = host._get_per_token_logps_and_entropies(
            host.model, ids, mask, keep, batch_size=1, num_images=None, num_tiles=None
        )
    return logps


def build_online_round(host, batch: dict) -> dict:
    """The online build around TRL's: the old-policy pass inside the armed block, then the layout under packing."""
    with host._scoring_round_in_bins():
        old = old_policy_pass(host, batch)
    result = dict(batch) | {"old_per_token_logps": old}
    return host._lay_out_packed_round(result, host._take_round_plan(result)) if host._packs_microbatches else result


def train_online_window(host, batch: dict, seed: int = 7) -> tuple[float, list[torch.Tensor]]:
    """One optimizer step's micro-steps through ``compute_loss`` and TRL's buffer: ``(summed loss, gradients)``."""
    host._step, host._buffered_inputs = 0, None
    host._generate_and_score_completions = lambda _inputs: build_online_round(host, batch)
    for param in host.model.parameters():
        param.grad = None
    torch.manual_seed(seed)
    total = 0.0
    for _ in range(host.args.gradient_accumulation_steps):
        inputs = host._prepare_inputs({})
        host._step += 1
        loss = DistributedOnlineGRPOTrainer.compute_loss(host, host.model, inputs)
        loss.backward()
        total += loss.item()
    grads = [p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad for p in host.model.parameters()]
    return total, [grad.float().clone() for grad in grads]
