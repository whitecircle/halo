"""Shared pieces of the PP suites driving a real trainer."""

import math
import os

import torch
import torch.distributed as dist
from datasets import Dataset
from torch.distributed.tensor import DTensor
from transformers import (
    AutoTokenizer,
    Qwen3Config,
    Qwen3ForSequenceClassification,
    TrainerCallback,
    default_data_collator,
)

from halo.callbacks.balancing import configure_moe_balancing
from halo.callbacks.factory import build_run_callbacks
from halo.configs.script_arguments import CommonScriptArguments
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.checkpoint_io import OPTIMIZER_SHARD_FILE, save_schedule
from tests.common.models import QWEN3_0_6B, TINY_QWEN3_CONFIG
from tests.common.optimizer_state import local_optimizer_state
from tests.common.pp_equivalence import causal_lm_loss, count_checkpoint_calls, dp_slice_batch
from tests.common.rank_log import log
from tests.common.tolerances import TOL

PAD_ID = 0


def expect_construction_error(trainer_cls, name: str, checks: dict, match: str, **trainer_kwargs) -> None:
    """Construct ``trainer_cls`` on every rank expecting a uniform pre-collective ValueError naming ``match``; records
    ``checks[name]``."""
    try:
        trainer_cls(**trainer_kwargs)
        checks[name] = False
    except ValueError as exc:
        checks[name] = match in str(exc)
        if not checks[name]:
            log(f"[{name}] raised without the expected mechanism: {exc}")


def build_pp_args(config_cls, output_dir: str, *, save_every: int | None = None, **fields):
    """``config_cls`` with what PP suites share: accumulation as the microbatch count, constant LR, every step logged,
    fp32."""
    return config_cls(
        **{
            "output_dir": output_dir,
            "gradient_accumulation_steps": 2,
            "learning_rate": 5e-3,
            "lr_scheduler_type": "constant",
            "logging_steps": 1,
            **save_schedule(save_every),
            "report_to": [],
            "logging_nan_inf_filter": False,
            "seed": 42,
            "bf16": False,
            **fields,
        }
    )


def build_sft_rows(vocab_size: int, n_rows: int = 16, seed: int = 7, seq_len: int = 64) -> Dataset:
    """Pre-tokenized rows with ragged ``-100`` prefixes, so replicas hold unequal counts and exercise the
    ``num_items_in_batch`` -> G/dp threading."""
    g = torch.Generator().manual_seed(seed)
    rows = torch.randint(4, vocab_size, (n_rows, seq_len), generator=g)
    labels = rows.clone()
    for i in range(labels.size(0)):
        labels[i, : 4 + (i % 11)] = -100
    return Dataset.from_dict(
        {
            "input_ids": rows.tolist(),
            "labels": labels.tolist(),
            "attention_mask": torch.ones_like(rows).tolist(),
        }
    )


def tiny_vocab_tokenizer():
    """The Qwen3 tokenizer padding with id 0: its own pad id is outside tiny vocabs and sends the stage-0 embedding
    gather out of bounds."""
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
    tokenizer.pad_token = tokenizer.convert_ids_to_tokens(PAD_ID)
    return tokenizer


def build_seq_cls_model(num_labels: int, seed: int = 1234) -> Qwen3ForSequenceClassification:
    torch.manual_seed(seed)
    config = Qwen3Config(**TINY_QWEN3_CONFIG, num_labels=num_labels, pad_token_id=PAD_ID, eos_token_id=1)
    return Qwen3ForSequenceClassification(config)


def right_pad(ids: list[list[int]], length: int, device=None) -> dict:
    input_ids = torch.full((len(ids), length), PAD_ID, dtype=torch.long)
    attention_mask = torch.zeros((len(ids), length), dtype=torch.long)
    for i, row in enumerate(ids):
        input_ids[i, : len(row)] = torch.tensor(row)
        attention_mask[i, : len(row)] = 1
    return {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device)}


def build_pp_trainer(trainer_cls, *, also_record=None, **trainer_kwargs):
    """``trainer_cls`` with its runner's ``step`` and ``eval_loss`` spied; returns ``(trainer, steps, evals)``.
    ``num_items`` is NaN without a normalizer, so a normalizer check fails rather than raises."""
    trainer = trainer_cls(**trainer_kwargs)
    runner = trainer._pp_runner
    steps, evals = [], []
    for method, calls in (("step", steps), ("eval_loss", evals)):
        drive = getattr(runner, method)

        def spying(input_ids, labels, _drive=drive, _calls=calls, **kw):
            normalizer = kw.get("num_items_in_batch")
            _calls.append(
                {
                    "input_ids": input_ids.clone(),
                    "labels": labels.clone(),
                    "attention_mask": kw["attention_mask"].clone(),
                    "num_items": math.nan if normalizer is None else float(normalizer),
                    **{key: value.clone() for key, value in (kw.get("extra_targets") or {}).items()},
                    **(also_record(trainer) if also_record else {}),
                }
            )
            return _drive(input_ids, labels, **kw)

        setattr(runner, method, spying)
    return trainer, steps, evals


def enable_gradient_checkpointing_as_train_does(trainer) -> None:
    """Enable checkpointing as ``Trainer.train`` does, for a suite driving ``training_step`` without ``train``; a no-op
    where the args ask for none."""
    if not trainer.args.gradient_checkpointing:
        return
    kwargs = dict(trainer.args.gradient_checkpointing_kwargs or {})
    every_n_layers = kwargs.pop("every_n_layers", 1)
    offload = kwargs.pop("offload", False)
    trainer.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs=kwargs or None, every_n_layers=every_n_layers, offload=offload
    )


class CheckpointCallsInTrain(TrainerCallback):
    """Counts stage layers' checkpoint calls during ``train()``, wrapped at ``on_train_begin``, the first hook after
    checkpointing is enabled."""

    def __init__(self, trainer):
        self._trainer = trainer
        self._calls = [0]
        trainer.add_callback(self)

    @property
    def calls(self) -> int:
        return self._calls[0]

    def on_train_begin(self, args, state, control, **kwargs):
        if self._trainer.model.is_gradient_checkpointing:
            self._calls = count_checkpoint_calls(self._trainer.model)


def losses_identical_across_ranks(losses: list[float]) -> bool:
    """COLLECTIVE (world all-gather). Whether every rank logged the same ``losses``, to
    ``TOL.pp_rank_identical_loss_abs``."""
    local = torch.tensor(losses, device=torch.cuda.current_device())
    gathered = [torch.zeros_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    return all(torch.allclose(local, peer, atol=TOL.pp_rank_identical_loss_abs) for peer in gathered)


def build_bias_balanced_sft_trainer(model, args, config, tokenizer, train_dataset) -> DistributedSFTTrainer:
    """The real SFT trainer over an EP model with ``bias_update`` balancing wired as the scripts do."""
    apply_ep_layers(model, config.ep_config())
    script_args = CommonScriptArguments(moe_balancing="bias_update")
    mode = configure_moe_balancing(script_args, model, config)
    return DistributedSFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=default_data_collator,
        callbacks=build_run_callbacks(script_args, args, model, config, balancing_mode=mode),
        parallelism_config=config,
        moe_balancing=script_args.moe_balancing,
    )


def probe_step(trainer, rows: Dataset, batch_size: int):
    """One more pipeline step on this replica's slice with no optimizer update: the loss a reload must reproduce.
    Returns ``(batch, loss)``."""
    batch = dp_slice_batch(
        torch.tensor(rows["input_ids"]),
        torch.tensor(rows["labels"]),
        trainer.parallelism_config.get_data_parallel_rank(),
        batch_size,
    )
    trainer.state.global_step += 1  # The EP sweep's double-run guard keys on the step.
    return batch, trainer.training_step(trainer.model, dict(batch))


def reload_reproduces_probe_loss(model, batch: dict, rows: Dataset, dp_size: int, probe_loss, device) -> bool:
    """Whether the reloaded ``model`` scores the probe under the pipeline's normalizer. Loose (5e-2): the run's bf16
    kernels differ from the reload's."""
    with torch.no_grad():
        logits = model(
            input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device)
        ).logits
    # G is the probe step's DP-global count, not the dataset's.
    global_labels = torch.tensor(rows["labels"][: dp_size * batch["labels"].size(0)])
    global_count = float((global_labels[:, 1:] != -100).sum())
    expected = float(causal_lm_loss(logits, batch["labels"].to(device), reduction="sum") / (global_count / dp_size))
    rel = abs(expected - float(probe_loss)) / max(abs(expected), 1e-9)
    log(f"[reload-value] probe={float(probe_loss):.6f} reloaded={expected:.6f} rel={rel:.3e}")
    return rel <= 5e-2


def restored_moments_match_shard(stage, optimizer, checkpoint: str, rank: int, fqn_prefixes=("",)) -> tuple[bool, int]:
    """Whether every moment in this rank's shard under ``fqn_prefixes`` equals the live optimizer's, bitwise, and how
    many params it covers (none fails)."""
    shard = torch.load(
        os.path.join(checkpoint, OPTIMIZER_SHARD_FILE.format(rank=rank)), map_location="cpu", weights_only=False
    )
    live = local_optimizer_state(stage, optimizer)["state"]
    fqns = [fqn for fqn in shard["state"] if fqn.startswith(fqn_prefixes)]
    ok = bool(fqns)
    for fqn in fqns:
        if fqn not in live:
            log(f"[optim] live state missing for {fqn}")
            ok = False
            continue
        for key, saved in shard["state"][fqn].items():
            if not torch.is_tensor(saved):
                continue
            saved = saved.to_local() if isinstance(saved, DTensor) else saved
            if key not in live[fqn] or not torch.equal(saved.cpu(), live[fqn][key]):
                log(f"[optim] moment mismatch at {fqn}.{key}")
                ok = False
    return ok, len(fqns)
