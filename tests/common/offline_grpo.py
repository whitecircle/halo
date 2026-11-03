"""Offline-GRPO GPU-suite fixtures and full-logits oracles, never the trainer's chunked path, so its defects cannot
cancel out."""

import argparse
import contextlib
import math

import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen3MoeConfig, Qwen3MoeForCausalLM

from halo.checkpoint.reading import load_full_state_dict
from halo.configs.grpo_offline import OfflineGRPOConfig
from halo.data.collators.offline_grpo import REF_PER_TOKEN_LOGPS_COLUMN
from halo.distributed.fsdp import reshard_fsdp2_modules
from halo.model_loading.liger.dispatch import apply_liger_kernel
from halo.model_loading.parallel.loader import load_distributed_model
from halo.trainers.grpo.objective.logratio import KL_LOGRATIO_CLAMP
from halo.trainers.grpo.offline.trainer import OfflineGRPOTrainer
from tests.common.memory import cleanup_memory, finish_phase
from tests.common.models import TINY_QWEN3_MOE_CONFIG
from tests.common.numerics import max_or_nan
from tests.common.rank_log import log
from tests.common.tolerances import TOL

OFFLINE_WORDS = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet")
OFFLINE_VOCAB = {
    word: index
    for index, word in enumerate(
        ("<pad>", "<eos>", "<unk>", "question", "answer", "yes", "no", "maybe", *OFFLINE_WORDS)
    )
}
OFFLINE_QWEN3_MOE_CONFIG = TINY_QWEN3_MOE_CONFIG | {
    "vocab_size": len(OFFLINE_VOCAB),
    "num_hidden_layers": 2,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "head_dim": 32,
    "pad_token_id": OFFLINE_VOCAB["<pad>"],
    "eos_token_id": OFFLINE_VOCAB["<eos>"],
}
EP_LOADINGS = ("lazy", "eager")
# An exported checkpoint must score this to finite logits after a stock reload.
EXPORT_PROBE_WORDS = ("question", "alpha", "answer", "yes", "<eos>")
# Mismatching tensors an export comparison names before it stops listing.
EXPORT_MISMATCH_PREVIEW = 12


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


def reference_oracle_rows(oracle) -> list[torch.Tensor]:
    """Each row's completion log-probs under ``oracle``, a ``kl_beta=0`` trainer released here: what a reference sweep
    must reproduce."""
    batch = oracle._prepare_inputs(oracle.data_collator(list(oracle.train_dataset)))
    oracle.model.eval()
    with torch.no_grad():
        logps = completion_logps(oracle.model, batch)
    rows = [
        row[: int(mask.sum())].cpu().clone()
        for row, mask in zip(logps, batch["completion_attention_mask"], strict=True)
    ]
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


def pure_kl_batch(trainer) -> dict:
    """The first two training rows, collated and placed, with zero advantages: the loss on this batch is the KL term
    alone."""
    batch = trainer._prepare_inputs(trainer.data_collator([trainer.train_dataset[index] for index in range(2)]))
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
        with torch.no_grad():
            model.get_output_embeddings().weight.mul_(2)
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
