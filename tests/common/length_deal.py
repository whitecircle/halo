"""Stand-ins for the length-ordered deal's suites: the environmental trainer's scoring path and TRL's real buffer and
loss, built without ``__init__``."""

import time
import types
from collections import defaultdict

import torch
import torch.distributed as dist
from transformers import LlamaConfig, LlamaForCausalLM
from trl import GRPOTrainer

from halo.trainers.grpo.environmental.trainer import DistributedAsyncEnvironmentalGRPOTrainer
from halo.trainers.grpo.microbatch_plan import MicrobatchPlan

MAX_COMPLETION_LENGTH = 64

# The packed-microbatch suites' rounds: 16 rows over 4 slots, budgets at which first-fit-decreasing needs 3 bins, so
# the window keeps one no-op slot and the rows-per-bin rescale is not 1.
PACKED_ROWS, PACKED_SLOTS = 16, 4
PACKED_DENSE_ROW_TOKENS = [128 + 128 * i for i in range(PACKED_ROWS)]
PACKED_DENSE_BUDGET = 6144
PACKED_MOE_ROW_TOKENS = [64 + 24 * i for i in range(PACKED_ROWS)]
PACKED_MOE_BUDGET = 1536


def tiny_llama(vocab: int, seed: int) -> LlamaForCausalLM:
    """A seeded two-layer SDPA Llama, small enough to forward on the host."""
    config = LlamaConfig(
        vocab_size=vocab,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(seed)
    return LlamaForCausalLM(config)


def single_process_accelerator() -> types.SimpleNamespace:
    """The accelerator surface the scoring path and TRL's loss read, for one process."""
    return types.SimpleNamespace(
        unwrap_model=lambda model: model,
        gather=lambda x: x,
        reduce=lambda x, reduction: x,
        num_processes=1,
        sync_gradients=True,
        is_main_process=False,
        device=torch.device("cpu"),
    )


def _all_gathered(tensor: torch.Tensor) -> torch.Tensor:
    rows = tensor.reshape(-1).contiguous() if tensor.dim() == 0 else tensor.contiguous()
    parts = [torch.empty_like(rows) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, rows)
    return torch.cat(parts)


def _all_reduced(tensor: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
    out = tensor.clone()
    dist.all_reduce(out)
    return out


def world_accelerator() -> types.SimpleNamespace:
    """The same surface over the initialized process group."""
    return types.SimpleNamespace(
        unwrap_model=lambda model: model,
        gather=_all_gathered,
        reduce=_all_reduced,
        num_processes=dist.get_world_size(),
        sync_gradients=True,
        is_main_process=dist.get_rank() == 0,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )


def ragged_round(
    prompt_lens: list[int], completion_lens: list[int], *, vocab: int, seed: int, device="cpu"
) -> dict[str, torch.Tensor]:
    """A round of left-padded prompts and right-padded completions, with centred advantages and an IS ratio, as the
    batch build hands it to TRL."""
    gen = torch.Generator().manual_seed(seed)
    rows, prompt_width, completion_width = len(prompt_lens), max(prompt_lens), max(completion_lens)
    prompt_mask = (torch.arange(prompt_width) >= prompt_width - torch.tensor(prompt_lens)[:, None]).long()
    completion_mask = (torch.arange(completion_width) < torch.tensor(completion_lens)[:, None]).long()
    batch = {
        "prompt_ids": torch.randint(1, vocab, (rows, prompt_width), generator=gen) * prompt_mask,
        "prompt_mask": prompt_mask,
        "completion_ids": torch.randint(1, vocab, (rows, completion_width), generator=gen) * completion_mask,
        "completion_mask": completion_mask,
        "tool_mask": completion_mask.clone(),
        "advantages": torch.rand(rows, generator=gen) * 2 - 1,
        "importance_sampling_ratio": 0.5 + torch.rand(rows, completion_width, generator=gen),
        "row_id": torch.arange(rows),
        "old_per_token_logps": None,
        "num_items_in_batch": completion_mask.sum(),
    }
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def sampling_side_round(*, vocab: int, close: int, rows: int, seed: int = 5) -> dict:
    """A ragged round with a forced close, an untrainable row and a tool-output token, for suites that weigh tokens by
    the engine's log-probs."""
    gen = torch.Generator().manual_seed(seed)
    batch = ragged_round(
        torch.randint(2, 6, (rows,), generator=gen).tolist(),
        torch.randint(3, 9, (rows,), generator=gen).tolist(),
        vocab=vocab,
        seed=seed,
    )
    batch["completion_ids"][1, 1] = close
    batch["completion_ids"][3, 0] = close
    batch["tool_mask"][2, 1] = 0
    batch["advantages"][4] = 0.0
    return batch


def stacked(batch: dict) -> tuple[torch.Tensor, torch.Tensor, int]:
    """``(input_ids, attention_mask, logits_to_keep)`` of a round, as the no-grad pass forwards it."""
    ids = torch.cat([batch["prompt_ids"], batch["completion_ids"]], 1)
    mask = torch.cat([batch["prompt_mask"], batch["completion_mask"]], 1)
    return ids, mask, batch["completion_ids"].size(1)


def recompute_logps(model, batch: dict) -> torch.Tensor:
    """The old-policy pass's recompute of a round: every row, no grad, no entropy."""
    host = packed_host(model, slots=batch["advantages"].size(0), accelerator=single_process_accelerator())
    host._packs_microbatches = False
    ids, mask, keep = stacked(batch)
    with torch.no_grad():
        return host._get_per_token_logps_and_entropies(model, ids, mask, keep, compute_entropy=False)[0]


def scoring_host(model, ref_model, *, chunks: int, accelerator):
    """The trainer's real scoring path, the chunked dense sweep at batch 1, the deal on."""
    host = object.__new__(DistributedAsyncEnvironmentalGRPOTrainer)
    host.model, host.ref_model = model, ref_model
    host.args = types.SimpleNamespace(
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        steps_per_generation=chunks,
        gradient_accumulation_steps=chunks,
        delta=None,
    )
    host.accelerator = accelerator
    host.temperature = 1.0
    host.beta = 0.0
    host.num_iterations = 1
    host._use_chunked_grpo_logprobs = True
    host._is_correction = True
    host._routing_injector = None
    host._rollout_routing_replay = False
    host._replay_row_spans = True
    host._orders_rows_by_length = True
    return host


def loss_host(model, *, chunks: int, accelerator, loss_type: str = "dapo"):
    """:func:`scoring_host` with TRL's buffer and ``loss_type`` state, IS on; ``luspo`` at sequence-level IS,
    ``dr_grpo`` by :data:`MAX_COMPLETION_LENGTH`."""
    host = scoring_host(model, None, chunks=chunks, accelerator=accelerator)
    host.args.sapo_temperature_pos, host.args.sapo_temperature_neg = 1.0, 1.05
    host._step = 0
    host._buffered_inputs = None
    host._broadcast_tensors_from_tp_leader = lambda inputs: inputs
    host.current_gradient_accumulation_steps = chunks
    host.aux_loss_enabled = False
    host.top_entropy_quantile = 1.0
    host.off_policy_mask_threshold = None
    host.importance_sampling_level = "sequence" if loss_type == "luspo" else "token"
    host.loss_type = loss_type
    host.max_completion_length = MAX_COMPLETION_LENGTH
    host.epsilon_low, host.epsilon_high = 0.2, 0.28
    host.use_vllm = host.vllm_importance_sampling_correction = True
    host._entropy_bonus_enabled = False
    host._metrics = {"train": defaultdict(list)}
    return host


def train_round(host, batch: dict, *, deal: bool, seed: int = 7) -> tuple[float, dict[int, torch.Tensor]]:
    """One round through TRL's buffer and loss, backward per micro-batch: ``(summed loss, row id -> per-token
    log-probs)``."""
    host._orders_rows_by_length = deal
    host._step, host._buffered_inputs = 0, None
    host._generate_and_score_completions = lambda _inputs: dict(batch)
    scored = []
    scorer = DistributedAsyncEnvironmentalGRPOTrainer._get_per_token_logps_and_entropies

    def recording(*args, **kwargs):
        out = scorer(host, *args, **kwargs)
        scored.append(out[0].detach())
        return out

    host._get_per_token_logps_and_entropies = recording
    torch.manual_seed(seed)
    total, row_logps = 0.0, {}
    try:
        for _ in range(host.args.steps_per_generation):
            inputs = host._prepare_inputs({})
            host._step += 1
            loss = GRPOTrainer._compute_loss(host, host.model, inputs)
            loss.backward()
            total += loss.item()
            for row_id, values in zip(inputs["row_id"].tolist(), scored[-1], strict=True):
                row_logps[row_id] = values
    finally:
        del host._get_per_token_logps_and_entropies
    return total, row_logps


def packed_host(model, *, slots: int, iterations: int = 1, loss_type: str = "dapo", accelerator):
    """:func:`loss_host` as the packed trainer sees a window: ``slots`` micro-steps of one generation cycle,
    ``iterations`` cycles to the optimizer step, the round held in the store."""
    host = loss_host(model, chunks=slots, accelerator=accelerator, loss_type=loss_type)
    host.num_iterations = iterations
    host.current_gradient_accumulation_steps = host.args.gradient_accumulation_steps = slots * iterations
    host._packs_microbatches = True
    host._round_store = None
    host._orders_rows_by_length = False
    host.pad_token_id = 0
    host.use_liger_kernel = False
    host._get_tp_or_etp_process_group = lambda: None
    return host


def train_window(
    host, batch: dict, plan: MicrobatchPlan | None = None, seed: int = 7
) -> tuple[float, list[torch.Tensor]]:
    """One optimizer step's micro-steps through the trainer's own ``compute_loss`` and TRL's buffer: a round one row per
    chunk, or under ``plan`` its packed slots. ``(summed loss, parameter gradients)``."""
    host._step, host._buffered_inputs = 0, None
    if plan is None:
        host._generate_and_score_completions = lambda _inputs: dict(batch)
    else:
        host._generate_and_score_completions = lambda _inputs: host._lay_out_packed_round(dict(batch), plan)
    for param in host.model.parameters():
        param.grad = None
    torch.manual_seed(seed)
    total = 0.0
    for _ in range(host.args.gradient_accumulation_steps):
        inputs = host._prepare_inputs({})
        host._step += 1
        loss = type(host).compute_loss(host, host.model, inputs)
        loss.backward()
        total += loss.item()
    grads = [p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad for p in host.model.parameters()]
    return total, [grad.float().clone() for grad in grads]


def with_world_normalizer(batch: dict) -> dict:
    """``batch`` with the world's loss-token count as its DAPO normalizer, as the batch build gathers it."""
    world_items = batch["num_items_in_batch"].clone()
    dist.all_reduce(world_items)
    return batch | {"num_items_in_batch": world_items}


def timed_no_grad_pass(host, batch: dict, order: torch.Tensor | None, repeats: int) -> float:
    """The best wall time of ``repeats`` no-grad recomputes with rows forwarded in ``order``, every rank fenced in and
    out."""
    ids, mask, keep = stacked(batch)
    best = float("inf")
    for _ in range(repeats):
        dist.barrier()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            host._recompute_logps_and_routing(ids, mask, keep, None, "train", order)
        torch.cuda.synchronize()
        dist.barrier()
        best = min(best, time.perf_counter() - start)
    return best
