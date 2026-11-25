"""Offline-GRPO GPU-suite fixtures and full-logits oracles, never the trainer's chunked path, so its defects cannot
cancel out."""

import argparse
import contextlib
import itertools
import math
import random
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, NamedTuple

import torch
import torch.distributed as dist
from datasets import Dataset, concatenate_datasets
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen3MoeConfig, Qwen3MoeForCausalLM

from halo.callbacks.moe_metrics import _selected_experts
from halo.checkpoint.reading import load_full_state_dict
from halo.configs.grpo_offline import OfflineGRPOConfig
from halo.data.collators.offline_grpo import REF_PER_TOKEN_LOGPS_COLUMN
from halo.distributed.fsdp import reshard_fsdp2_modules
from halo.model_loading.finalization import finalize_loaded_model
from halo.model_loading.liger.dispatch import apply_liger_kernel
from halo.model_loading.parallel.loader import load_distributed_model
from halo.models.attention.gpt_oss_sinks import SinksPolicy, apply_sinks_policy
from halo.models.introspection.config_access import get_config_field
from halo.models.moe.config_fields import resolve_moe_routing
from halo.models.moe.router_logits import declared_routers
from halo.trainers.grpo.objective.logratio import KL_LOGRATIO_CLAMP
from halo.trainers.grpo.offline.trainer import OfflineGRPOTrainer
from tests.common.checkpoint_io import loading_problems
from tests.common.memory import cleanup_memory, finish_phase
from tests.common.models import TINY_QWEN3_MOE_CONFIG
from tests.common.numerics import max_or_nan
from tests.common.rank_log import log
from tests.common.tolerances import TOL

if TYPE_CHECKING:
    from tests.common.tiny_models import TinyFamily

OFFLINE_WORDS = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet")
# The offline vocabulary past its special tokens.
OFFLINE_TEXT_WORDS = ("question", "answer", "yes", "no", "maybe", *OFFLINE_WORDS)
OFFLINE_VOCAB = {word: index for index, word in enumerate(("<pad>", "<eos>", "<unk>", *OFFLINE_TEXT_WORDS))}
OFFLINE_QWEN3_MOE_CONFIG = TINY_QWEN3_MOE_CONFIG | {
    "vocab_size": len(OFFLINE_VOCAB),
    "num_hidden_layers": 2,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "head_dim": 32,
    "pad_token_id": OFFLINE_VOCAB["<pad>"],
    "eos_token_id": OFFLINE_VOCAB["<eos>"],
}
# Mistral 4's query scale steps at this position; these rows cross it through their left pads alone.
MISTRAL4_SCALE_STEP = 32
MISTRAL4_STEPPED_ROPE = {
    "rope_type": "yarn",
    "rope_theta": 10000.0,
    "factor": 128.0,
    "original_max_position_embeddings": MISTRAL4_SCALE_STEP,
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
    "llama_4_scaling_beta": 0.1,
}
EP_LOADINGS = ("lazy", "eager")
# An exported checkpoint must score this to finite logits after a stock reload.
EXPORT_PROBE_WORDS = ("question", "alpha", "answer", "yes", "<eos>")
# Mismatching tensors an export comparison names before it stops listing.
EXPORT_MISMATCH_PREVIEW = 12
# Random-word groups a KL suite trains beside its template rows, which share prompt positions: one such position inside
# the bf16 routing noise fails every templated row at once. Over 21 CPU seeds of each Ulysses MoE family, the fewest of
# the 144 rows beside 8 template groups with routing headroom for a KL check was 6 (Mistral 4). More groups do not
# raise it, since the noise is a maximum over every row: at 128, Mistral 4's base checkpoint fell from 7 such rows to 5.
KL_ROUTING_POOL_GROUPS = 64
# The routing margin, in measured bf16 router noises, a row needs before it is compared against the trainer's kernels.
ROUTING_HEADROOM = 3


class RoutingClearRows(NamedTuple):
    """``rows`` to collate; ``cleared``, in dataset order, every row keeping fp32's selection at every router past twice
    ``noise``, the largest bf16-vs-fp32 router-logit difference the scan measured, which holds the eager oracle's own
    kernels to fp32's choice; ``headroom``, those of them whose margin is at least :data:`ROUTING_HEADROOM` times
    ``noise``, which also holds any kernel moving router logits less than half that many times ``noise``; ``clear``
    when ``rows`` all come from ``headroom``."""

    rows: tuple[int, ...]
    clear: bool
    noise: float
    cleared: tuple[int, ...]
    headroom: tuple[int, ...]


def ep_loading_parser() -> argparse.ArgumentParser:
    """The ``--ep-loading`` flag of the EP suites, read by the suite and the CPU check that manifest rows cover both
    loaders."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep-loading", choices=EP_LOADINGS, default=EP_LOADINGS[0])
    return parser


def make_offline_tokenizer():
    """A local-only tokenizer whose IDs and EOS behavior match the tiny checkpoint."""
    tokenizer = Tokenizer(WordLevel(OFFLINE_VOCAB, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
        model_max_length=128,
    )


def save_offline_moe_base(path, seed):
    """A shared local-only tiny MoE checkpoint for full-FT and native expert-LoRA GPU lifecycles."""
    torch.manual_seed(seed)
    model = Qwen3MoeForCausalLM(Qwen3MoeConfig(**OFFLINE_QWEN3_MOE_CONFIG))
    model.to(torch.bfloat16).save_pretrained(path)
    make_offline_tokenizer().save_pretrained(path)


def offline_grpo_dataset(groups, offset=0):
    """Two unequal-length, oppositely rewarded completions for each distinct prompt group."""
    records = []
    for index in range(groups):
        word = OFFLINE_WORDS[(index + offset) % len(OFFLINE_WORDS)]
        other = OFFLINE_WORDS[(index + offset + 1) % len(OFFLINE_WORDS)]
        records.append(
            {
                "prompt": f"question {word} answer",
                "completions": [f"yes {word}", f"no {other} maybe"],
                "rewards": [1.0, -1.0],
            }
        )
    return Dataset.from_list(records)


def random_word_groups(groups, seed) -> Dataset:
    """``groups`` oppositely rewarded completion pairs of random offline words, one to three in a prompt and one or two
    in a completion, so no position is common to every row."""
    rng = random.Random(seed)

    def words(fewest, most):
        return " ".join(rng.choices(OFFLINE_TEXT_WORDS, k=rng.randint(fewest, most)))

    return Dataset.from_list(
        [
            {"prompt": words(1, 3), "completions": [words(1, 2), words(1, 2)], "rewards": [1.0, -1.0]}
            for _ in range(groups)
        ]
    )


def kl_oracle_training_set(template_groups, seed) -> Dataset:
    """:func:`offline_grpo_dataset` followed by :data:`KL_ROUTING_POOL_GROUPS` random-word groups: the rows
    :func:`routing_clear_rows` draws a KL batch from."""
    return concatenate_datasets(
        [offline_grpo_dataset(template_groups), random_word_groups(KL_ROUTING_POOL_GROUPS, seed)]
    )


def mixed_length_groups(groups) -> Dataset:
    """Long prompts and short ones, so a batch holding both left-pads a short row past Mistral 4's scale step and
    DeepSeek-V4's compressor windows."""
    records = []
    for index in range(groups):
        word = OFFLINE_WORDS[index % len(OFFLINE_WORDS)]
        repeats = 1 if index % 2 else MISTRAL4_SCALE_STEP
        records.append(
            {
                "prompt": " ".join(["question", *[word] * repeats, "answer"]),
                "completions": [f"yes {word}", f"no {word} maybe"],
                "rewards": [1.0, -1.0],
            }
        )
    return Dataset.from_list(records)


def load_liger_class_patched_model(model_path, *, attn_implementation="flash_attention_4"):
    """``model_path`` at bf16 with Liger's class-level patch applied before the load; a late instance patch misses
    Qwen3's q/k norms."""
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    apply_liger_kernel(config, liger_kernel_config={"cross_entropy": False, "fused_linear_cross_entropy": False})
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
    )


def eager_oracle(source, device="cpu", dtype=torch.bfloat16, *, family: "TinyFamily | None" = None):
    """``source`` at ``dtype`` with full-row eager attention and sinks neutralized, in eval mode, through ``family``'s
    loader (the auto causal-LM class without one). Finalized as production loads are: a remote-code rotary table is
    otherwise the uninitialized memory transformers 5 leaves in non-persistent buffers."""
    model, info = (AutoModelForCausalLM if family is None else family.load_class).from_pretrained(
        source,
        dtype=dtype,
        attn_implementation="eager",
        output_loading_info=True,
        trust_remote_code=family is not None and family.trust_remote_code,
    )
    assert not loading_problems(info), f"incomplete oracle/export checkpoint: {loading_problems(info)}"
    model.to(device)
    finalize_loaded_model(model, source)
    apply_sinks_policy(model, model.config, policy=SinksPolicy.NEUTRALIZED, attn_implementation="eager")
    return model.eval()


def offline_grpo_config(
    output_dir,
    *,
    steps,
    save_steps,
    seed,
    kl_beta,
    learning_rate=1e-3,
    sequence_length=32,
    evaluate=True,
    save=True,
    **overrides,
) -> OfflineGRPOConfig:
    """The suites' shared run: two rows per device, chunked log-probs, ``grpo`` with REINFORCE; ``overrides`` sets any
    other field."""
    fields = {
        "output_dir": output_dir,
        "max_steps": steps,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "learning_rate": learning_rate,
        "bf16": True,
        "gradient_checkpointing": True,
        "use_liger_kernel": False,
        "kl_beta": kl_beta,
        "use_chunked_grpo_logprobs": True,
        "loss_type": "grpo",
        "policy_gradient_formulation": "reinforce",
        "min_log_prob": None,
        "max_grad_norm": 0.0,
        "logging_steps": 1,
        "eval_strategy": "steps" if evaluate else "no",
        "eval_steps": 1,
        "save_strategy": "steps" if save else "no",
        "save_steps": save_steps,
        "save_total_limit": steps,
        "report_to": "none",
        "max_prompt_length": sequence_length,
        "max_completion_length": sequence_length,
        "remove_unused_columns": False,
        "dataloader_drop_last": True,
        "dataloader_num_workers": 0,
        "seed": seed,
        "data_seed": seed,
    }
    return OfflineGRPOConfig(**(fields | overrides))


def offline_grpo_trainer(
    ctx, model, parallelism, args, train, evaluation=None, *, checkpoint=None, tokenizer=None, **trainer_kwargs
) -> OfflineGRPOTrainer:
    """``OfflineGRPOTrainer`` on ``model``, balancing off; DeepEP buffers are also released at teardown for a body that
    raises early."""
    trainer = OfflineGRPOTrainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=evaluation,
        processing_class=make_offline_tokenizer() if tokenizer is None else tokenizer,
        parallelism_config=parallelism,
        resume_checkpoint=checkpoint,
        moe_balancing="none",
        **trainer_kwargs,
    )
    ctx.on_teardown(trainer.cleanup_ep)
    return trainer


def build_offline_grpo_trainer(
    ctx,
    source,
    parallelism,
    args,
    train,
    evaluation=None,
    *,
    checkpoint=None,
    tokenizer=None,
    trust_remote_code: bool = False,
) -> OfflineGRPOTrainer:
    """:func:`offline_grpo_trainer` through the production loader at bf16 with FA2; a resume also requires every stored
    fp32 master restored."""
    model, _ = load_distributed_model(
        model_name_or_path=source,
        parallelism_config=parallelism,
        dtype=torch.bfloat16,
        trust_remote_code=trust_remote_code,
        attn_implementation="flash_attention_2",
        use_liger_kernel=False,
        preserve_checkpoint_precision=checkpoint is not None,
    )
    return offline_grpo_trainer(
        ctx, model, parallelism, args, train, evaluation, checkpoint=checkpoint, tokenizer=tokenizer
    )


def token_logps(model, ids, attention_mask=None) -> torch.Tensor:
    """fp32 log-probs of each next token under ``model``'s full-vocabulary logits, ``[B, S-1]``."""
    logits = model(input_ids=ids, attention_mask=attention_mask, use_cache=False).logits[:, :-1].float()
    return logits.log_softmax(-1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)


def completion_logps(model, batch) -> torch.Tensor:
    """:func:`token_logps` of a non-CP batch's prompt + completion rows, cut to the completion columns."""
    ids = torch.cat([batch["prompt_input_ids"], batch["completion_input_ids"]], dim=1)
    attention_mask = torch.cat([batch["prompt_attention_mask"], batch["completion_attention_mask"]], dim=1)
    return token_logps(model, ids, attention_mask)[:, -batch["completion_input_ids"].size(1) :]


def sweep_batches(rows: int, width: int, dp_size: int) -> list[range]:
    """The dataset rows of each batch a reference sweep collates: ``width`` rows in order within each of ``dp_size``
    contiguous data-parallel shards, the last batch of a shard ``width`` does not divide holding the remainder."""
    bounds = [shard * rows // dp_size for shard in range(dp_size + 1)]
    return [
        range(start, min(start + width, end))
        for begin, end in itertools.pairwise(bounds)
        for start in range(begin, end, width)
    ]


def reference_oracle_rows(oracle) -> list[torch.Tensor]:
    """Each row's completion log-probs under ``oracle``, a ``kl_beta=0`` trainer released here: what a reference sweep
    must reproduce. Collated in the sweep's :func:`sweep_batches`, a short one filled to full width with copies of its
    last row as the sweep fills it, so a row's padding, and any routing near-tie with it, comes out alike on both
    sides."""
    dataset, width = oracle.train_dataset, oracle.args.per_device_train_batch_size
    dp_size, _ = oracle.dp_shard_geometry()
    oracle.model.eval()
    rows = []
    for indices in sweep_batches(len(dataset), width, dp_size):
        features = list(dataset.select(indices))
        batch = oracle._prepare_inputs(oracle.data_collator(features + features[-1:] * (width - len(features))))
        with torch.no_grad():
            logps = completion_logps(oracle.model, batch)
        real = len(features)
        rows.extend(
            row[: int(mask.sum())].cpu().clone()
            for row, mask in zip(logps[:real], batch["completion_attention_mask"][:real], strict=True)
        )
    finish_phase(oracle)
    return rows


def swept_reference_error(swept_rows, expected_rows) -> float:
    """Worst per-token gap between a sweep's rows and the oracle's, NaN-propagating; infinite with no rows."""
    return max_or_nan(
        (
            (torch.as_tensor(swept, dtype=torch.float32) - expected).abs().max().item()
            for swept, expected in zip(swept_rows, expected_rows, strict=True)
        ),
        default=math.inf,
    )


def _router_outputs(model, input_ids, attention_mask) -> list[tuple[torch.Tensor, torch.Tensor | None]]:
    """Each router's fp32 ``[B, S, experts]`` logits and ``[B, S, k]`` selected ids (None where it returns logits
    alone), in forward order. A model declaring no router is read off its forward's ``router_logits``."""
    routers = declared_routers(model)
    outputs = []

    def record(router, module, inputs, output):
        logits = output[router.logits_index] if isinstance(output, (tuple, list)) else output
        outputs.append((logits, None if router.folds_discard_slot else _selected_experts(output, router.logits_index)))

    handles = [router.module.register_forward_hook(partial(record, router)) for router in routers]
    try:
        with torch.no_grad():
            result = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                **({} if routers else {"output_router_logits": True}),
            )
    finally:
        for handle in handles:
            handle.remove()
    if not routers:
        outputs = [
            (layer[0] if isinstance(layer, (tuple, list)) else layer, _selected_experts(layer, 0))
            for layer in getattr(result, "router_logits", None) or ()
        ]
    shape = input_ids.shape
    return [
        (logits.float().reshape(*shape, -1), None if ids is None else ids.reshape(*shape, -1))
        for logits, ids in outputs
    ]


def _selection(logits, ids, top_k) -> torch.Tensor:
    return (logits.topk(top_k, dim=-1).indices if ids is None else ids).sort(-1).values


def _selection_margin(logits, selection, groups, picked_groups) -> torch.Tensor:
    """Per position, the logit shift that could first change the router's own ``selection``: its lowest logit's lead
    over every expert competing with it, and, for a router picking ``picked_groups`` of ``groups`` contiguous expert
    blocks by their top two scores, each picked block's first and second logits over every other block's. A bound for
    scores strictly increasing in the logit with no selection bias added after it; a router ranking scores it rounded to
    bf16 (Mistral 4's softmax) can swap logits up to one bf16 score step apart, which it does not bound."""
    chosen = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, selection, True)
    picked = chosen.unflatten(-1, (groups, -1)).any(-1)
    # One picked block leaves only its own experts in the race; with more, a picked block may hold no selection.
    competing = (
        picked.repeat_interleave(logits.size(-1) // groups, dim=-1) if picked_groups == 1 else torch.ones_like(chosen)
    )
    lead = logits.masked_fill(~chosen, math.inf).amin(-1) - logits.masked_fill(chosen | ~competing, -math.inf).amax(-1)
    top_two = logits.unflatten(-1, (groups, -1)).topk(2, dim=-1).values
    block_lead = top_two.masked_fill(~picked[..., None], math.inf).amin(-2) - top_two.masked_fill(
        picked[..., None], -math.inf
    ).amax(-2)
    return torch.minimum(lead, block_lead.amin(-1))


def routing_clear_rows(load_oracle: Callable[[torch.dtype], torch.nn.Module], dataset, count=2) -> RoutingClearRows:
    """Every row of ``dataset`` whose expert choices at every router outlast the bf16 kernel noise of the weights
    ``load_oracle(dtype)`` loads, those of them with :data:`ROUTING_HEADROOM`, and ``count`` rows with headroom for
    a KL batch: those with twice the headroom first, then the most completion tokens, the tokens a KL check scores
    (EOS included), then the wider margin. Where fewer have headroom, ``clear`` is False and the widest-margin rows,
    clearing ones first, fill in. A dense model clears every row and keeps the first. Raises when a model with routed
    experts exposes no router logits.

    The bar is the eager oracle's measured noise: a row clears where the bf16 oracle keeps fp32's choices and every
    router's fp32 margin exceeds twice the largest bf16-vs-fp32 logit difference at each position a scored log-prob
    reads, which holds two forwards moving logits no further than the eager oracle to one choice. The trainer's kernels
    go unmeasured; only the headroom covers them."""
    pairs = [(row["prompt_input_ids"], row["completion_input_ids"]) for row in dataset]
    rows = [prompt + completion for prompt, completion in pairs]
    scored = [len(completion) for _, completion in pairs]
    lengths = torch.tensor([len(row) for row in rows])
    exact_model = load_oracle(torch.float32)
    routing = resolve_moe_routing(exact_model)
    if not routing.routes_between_experts and not declared_routers(exact_model):
        every = tuple(range(len(rows)))
        return RoutingClearRows(every[:count], True, 0.0, every, every)
    groups = get_config_field(exact_model.config, "n_group") or 1
    picked_groups = get_config_field(exact_model.config, "topk_group") or 1
    input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(row) for row in rows], batch_first=True)
    positions = torch.arange(input_ids.size(1))
    attention_mask = (positions < lengths[:, None]).long()
    device = exact_model.device
    exact = _router_outputs(exact_model, input_ids.to(device), attention_mask.to(device))
    del exact_model
    if not exact:
        raise ValueError(
            "The oracle's routing scan found no router logits on a model with routed experts: it declares no "
            "router_logits recorder and its forward returns no router_logits, so the margins that keep the trainer's "
            "and the oracle's expert choices alike cannot be read. Register the family's router under "
            "_can_record_outputs['router_logits']."
        )
    noisy = _router_outputs(load_oracle(torch.bfloat16), input_ids.to(device), attention_mask.to(device))
    # A row's last real position predicts past the row, so no scored log-prob reads it.
    checked = (positions < (lengths - 1)[:, None]).to(device)
    agree = torch.ones(len(rows), dtype=torch.bool, device=device)
    margin = torch.full((len(rows),), math.inf, device=device)
    for (logits, ids), (noisy_logits, noisy_ids) in zip(exact, noisy, strict=True):
        selection = _selection(logits, ids, routing.top_k)
        agree &= ((selection == _selection(noisy_logits, noisy_ids, routing.top_k)).all(-1) | ~checked).all(-1)
        position_margin = _selection_margin(logits, selection, groups, picked_groups)
        margin = torch.minimum(margin, position_margin.masked_fill(~checked, math.inf).amin(-1))
    # Past a flip, a row's later routers move by the flip's effect, not by kernel noise.
    trusted = checked & agree[:, None]
    noise = max(
        float((noisy_logits - logits).abs().amax(-1)[trusted].max()) if trusted.any() else 0.0
        for (logits, _), (noisy_logits, _) in zip(exact, noisy, strict=True)
    )
    margins = margin.tolist()
    clears = [kept and lead > 2 * noise for kept, lead in zip(agree.tolist(), margins, strict=True)]
    cleared = tuple(index for index, clear in enumerate(clears) if clear)
    headroom = tuple(index for index in cleared if margins[index] >= ROUTING_HEADROOM * noise)
    if len(headroom) >= count:
        order = sorted(
            headroom,
            key=lambda index: (margins[index] < 2 * ROUTING_HEADROOM * noise, -scored[index], -margins[index]),
        )
    else:
        order = sorted(range(len(rows)), key=lambda index: (not clears[index], -margins[index]))
    chosen = tuple(order[:count])
    leads = ", ".join(f"{margins[index] / (noise or math.nan):.3g}" for index in chosen)
    log(
        f"Routing scan: {len(cleared)}/{len(rows)} rows keep fp32's routing at every router past twice the bf16 noise "
        f"{noise:.3g}, {len(headroom)} by {ROUTING_HEADROOM} times it or more; rows {chosen} lead it by {leads} times and "
        f"score {sum(scored[index] for index in chosen)} completion tokens"
    )
    return RoutingClearRows(chosen, len(headroom) >= count, noise, cleared, headroom)


def _double_output_head(model) -> None:
    with torch.no_grad():
        model.get_output_embeddings().weight.mul_(2)


def _doubled_head_oracle(load_oracle, dtype):
    model = load_oracle(dtype)
    _double_output_head(model)
    return model


def agree_kl_rows(ctx, load_oracle, dataset) -> RoutingClearRows:
    """COLLECTIVE. Rank 0's :func:`routing_clear_rows` of ``load_oracle``'s weights under the doubled output head every
    KL check compares, which every rank collates alike: a tied head doubles the input embeddings too, moving routers."""
    payload = [routing_clear_rows(partial(_doubled_head_oracle, load_oracle), dataset) if ctx.rank == 0 else None]
    cleanup_memory()
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def pure_kl_batch(trainer, rows) -> dict:
    """Training ``rows``, collated and placed, with zero advantages: the loss on this batch is the KL term alone."""
    batch = trainer._prepare_inputs(trainer.data_collator([trainer.train_dataset[index] for index in rows]))
    batch["advantage"] = torch.zeros_like(batch["advantage"])
    return batch


@contextlib.contextmanager
def doubled_output_head(model):
    """``model`` in eval mode with its output head doubled for the body: a policy away from the reference, so the KL
    term is nonzero."""
    reshard_fsdp2_modules(model)
    original = model.get_output_embeddings().weight.detach().clone()
    was_training = model.training
    model.eval()
    try:
        _double_output_head(model)
        yield model
    finally:
        reshard_fsdp2_modules(model)
        with torch.no_grad():
            model.get_output_embeddings().weight.copy_(original)
        model.train(was_training)


def pure_kl_objective(policy, reference, valid, group_size, beta) -> torch.Tensor:
    """The offline objective at zero advantage, written out independently: per-token ``k3`` KL with a detached
    log-ratio cap, weighted ``1 / group_size``."""
    delta = torch.minimum(reference, policy.detach() + KL_LOGRATIO_CLAMP) - policy
    per_row = ((delta.exp() - delta - 1) * valid).sum(1) / valid.sum(1).clamp(min=1)
    weights = group_size.float().reciprocal()
    return beta * (per_row * weights).sum() / weights.sum()


def full_logits_kl(model, batch, beta) -> torch.Tensor:
    """:func:`pure_kl_objective` of full-logits log-probs against the batch's run-start reference column."""
    return pure_kl_objective(
        completion_logps(model, batch),
        batch[REF_PER_TOKEN_LOGPS_COLUMN],
        batch["completion_attention_mask"],
        batch["group_size"],
        beta,
    )


def doubled_head_kl_verdict(trainer, batch, expected_fn, label) -> tuple[bool, bool]:
    """``(oracle nonzero, trainer loss matches it)`` on the doubled head; ``expected_fn(model)`` may ignore the model
    for a separate reload."""
    with doubled_output_head(trainer.model) as model, torch.no_grad():
        expected = expected_fn(model)
        actual = trainer.compute_loss(model, batch)
    error = abs((actual - expected).item())
    log(f"{label}: expected={expected.item():.5g}, error={error:.5g}")
    return bool(expected > 0), error < TOL.exact_objective_rel * expected.item()


def export_scores_finite(path, device) -> bool:
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="eager").to(device)
    prompt = torch.tensor([[OFFLINE_VOCAB[word] for word in EXPORT_PROBE_WORDS]], device=device)
    with torch.no_grad():
        finite = bool(model(prompt).logits.isfinite().all())
    del model
    cleanup_memory()
    return finite


def resumed_export_verdict(continuous, resumed, checkpoint, device) -> tuple[bool, bool, bool]:
    """Rank 0's verdicts: ``(bit-exact to the continuous export, moved off the checkpoint, finite after a stock
    reload)``. The move is read at the export's dtype, so rounding stored fp32 masters alone is not an update."""
    before, after = load_full_state_dict(continuous), load_full_state_dict(resumed)
    exact = before.keys() == after.keys() and all(torch.equal(before[name], after[name]) for name in before)
    if not exact:
        mismatches = []
        for name in sorted(before.keys() | after.keys()):
            if name not in before or name not in after:
                mismatches.append(f"{name}: missing")
            elif not torch.equal(before[name], after[name]):
                error = (before[name].float() - after[name].float()).abs()
                mismatches.append(
                    f"{name}: max={error.max().item():.8g}, changed={torch.count_nonzero(error).item()}/{error.numel()}"
                )
        log(f"resumed export mismatches ({len(mismatches)}): " + "; ".join(mismatches[:EXPORT_MISMATCH_PREVIEW]))
    previous = load_full_state_dict(checkpoint)
    stepped = previous.keys() == after.keys() and any(
        not torch.equal(previous[name].to(after[name].dtype), after[name]) for name in after
    )
    return exact, stepped, export_scores_finite(resumed, device)
