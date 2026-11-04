#!/usr/bin/env python
"""A released head fit on real weights under EP8: what its held-out acceptance does at the safe learning rate, hard and
soft, from a stale head and from the trunk's own, and whether the gate catches a fit that hurts it.

Every fit trains head_only on first-turn ultrachat rows at batch 8 in total, constant after warmup, and scores 24
held-out rows at step 0 and every 30 steps. ``stale_*`` starts Qwen3.6-35B's trunk on Qwen3.5-35B's head, ``own_*`` on
its own. ``detached`` runs a fine-tune with the clip off, whose trunk must stay what a run with no head makes it.

    torchrun --nproc_per_node=8 tests/gpu/trainers/sft/test_sft_mtp_head_real.py --row stale_soft

``--export_dir`` keeps the fit's last checkpoint, the one a server then reads. Bars hold at the default weights;
``HALO_TEST_MTP_REAL_MODEL`` and ``HALO_TEST_MTP_REAL_STALE_HEAD`` point a row elsewhere, where it records only.

``--release`` runs an ``own_*``, ``detached`` or ``untrained_export`` row on a Nemotron-H release through its knob
(``HALO_TEST_NEMOTRON_MODEL``, ``HALO_TEST_NEMOTRON_LIGHTNING_MODEL``), recording only, on rows rendered with its empty
think block. ``--text_only`` takes Super VL's text class, and ``--attn`` the attention a row runs. ``untrained_export``
saves a load's head with no step, and holds it to the release's keys and bytes.

    HALO_TEST_NEMOTRON_LIGHTNING_MODEL=/ckpt torchrun --nproc_per_node=8 tests/gpu/trainers/sft/test_sft_mtp_head_real.py \\
        --release lightning --row own_soft --attn flash_attention_4 --export_dir $D/lightning-fit
"""

import argparse
import hashlib
import math
import os
from dataclasses import dataclass, field, replace

import torch
import torch.distributed as dist
from datasets import load_from_disk
from transformers import AutoTokenizer
from trl import SFTConfig

from halo.callbacks.mtp import MtpAcceptanceGate
from halo.checkpoint.reading import iter_checkpoint_tensors, resolve_checkpoint_weights
from halo.distributed.fsdp import reshard_fsdp2_modules
from halo.distributed.parallelism_config import ParallelismConfig
from halo.entrypoints.model_setup import load_vlm_processor
from halo.model_loading.finalization import finalize_loaded_model
from halo.model_loading.liger.dispatch import LIGER_APPLIED_CONFIG_ATTR, finalize_liger_after_direct_load
from halo.model_loading.model_preparation import auto_load_model
from halo.model_loading.parallel.loader import load_distributed_model
from halo.models.mtp.acceptance import MTP_ACCEPTANCE_GREEDY, MTP_ACCEPTANCE_TV
from halo.models.mtp.head import mtp_head_path
from halo.models.mtp.spec import (
    MtpHeadRequest,
    MtpLossType,
    MtpObjective,
    MtpScope,
    require_mtp_head_spec,
    stored_head_prefix,
)
from halo.runtime.collectives import barrier, broadcast_from_rank0
from halo.runtime.consensus import rank_consensus
from halo.runtime.dtensor import to_local
from halo.runtime.filesystem import is_fs_save_rank
from halo.runtime.ranks import is_global_main_process
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.distributed import pin_deterministic_ep_dispatch, shared_output_dir
from tests.common.harness import gpu_test_main
from tests.common.memory import cleanup_memory
from tests.common.mtp_heads import collate_tokenized_rows
from tests.common.mtp_real_rows import (
    REAL_BARS_APPLY,
    REAL_MODEL,
    REAL_STALE_HEAD,
    AcceptanceBars,
    acceptance_checks,
    cached_source,
    first_turn_rows,
    off_by_one_rows_dataset,
    rows_dataset,
    ultrachat_conversations,
)
from tests.common.nemotron_h import NEMOTRON_RELEASES, release_model_id
from tests.common.rank_log import log

_WEIGHT = 1.0
_SEED = 42
_LENGTH = 256
_MAX_PROMPT = 96
_TRAIN_ROWS = 480
_HELD_OUT_ROWS = 24
_SAMPLED_TRAIN_ROWS = 240
_SAMPLED_REPLY = 128
_SAMPLE_BATCH = 16
_EVAL_EVERY = 30
# The positions 24 first-turn rows of 256 tokens score, which a different stream or tokenizer would miss.
_HUMAN_STREAM_POSITIONS = 4954
_STALE_START = {"first_tv": 0.776, "first_greedy": 0.653, "scored_positions": _HUMAN_STREAM_POSITIONS}
_OWN_START = {"first_tv": 0.819, "first_greedy": 0.675, "scored_positions": _HUMAN_STREAM_POSITIONS}
# A fine-tune at the SFT recipe's schedule; its loss curve is per step.
_DETACHED_STEPS = 200
_DETACHED_LR = 5e-6
_DETACHED_WARMUP = 32
_DETACHED_EVAL_EVERY = 50
# The next-token loss recorded two ways on one forward's float sums.
_SAME_CURVE_BAR = 1e-5
_ACCEPTANCE_DROP_BAR = 0.02


@dataclass(frozen=True)
class _Fit:
    loss: MtpLossType = "soft"
    start: str = "stale"
    learning_rate: float = 1e-5
    warmup_steps: int = 20
    steps: int = 150
    attn: str = "sdpa"
    text_only: bool = False
    own_samples: bool = False
    off_by_one: bool = False
    bars: AcceptanceBars = field(default_factory=AcceptanceBars)


_FITS = {
    "stale_soft": _Fit(bars=AcceptanceBars(**_STALE_START, last_tv_at_least=0.815, gate_stops=False)),
    "stale_soft_fused": _Fit(
        text_only=True, bars=AcceptanceBars(**_STALE_START, last_tv_at_least=0.815, gate_stops=False)
    ),
    "stale_soft_fa4": _Fit(
        attn="flash_attention_4", bars=AcceptanceBars(**_STALE_START, last_tv_at_least=0.815, gate_stops=False)
    ),
    "stale_hard": _Fit(loss="hard", bars=AcceptanceBars(**_STALE_START)),
    "own_soft": _Fit(start="own", bars=AcceptanceBars(**_OWN_START, gate_stops=False)),
    "own_hard": _Fit(start="own", loss="hard", bars=AcceptanceBars(**_OWN_START)),
    # Fresh samples, so the start is held loosely and the end by what the fit gained.
    "own_samples_soft": _Fit(
        own_samples=True,
        bars=AcceptanceBars(
            first_tv=0.869, first_greedy=0.845, first_window=0.02, tv_gain_at_least=0.013, gate_stops=False
        ),
    ),
    "oversized_lr": _Fit(
        learning_rate=1e-4, warmup_steps=0, steps=60, bars=AcceptanceBars(**_STALE_START, gate_stops=True)
    ),
    "off_by_one": _Fit(loss="hard", off_by_one=True, bars=AcceptanceBars(last_tv_at_most=0.05, gate_stops=True)),
}
_DETACHED_ROW = "detached"
_UNTRAINED_ROW = "untrained_export"
# A reply under Nemotron-H's generation prompt reads as what its template renders with no reasoning.
_RELEASE_TEMPLATE = {"enable_thinking": False}
_STALE_START_NAME = "stale"
_TEXT_RELEASE = "lightning"


def _checkpoint(cli) -> str:
    return release_model_id(cli.release) if cli.release else REAL_MODEL


def _local_digests(model) -> dict[str, str]:
    """This rank's slice of each parameter it holds any of. Past the reshard, as a forward leaves a gathered copy
    registered that the optimizer's last step did not write."""
    reshard_fsdp2_modules(model)
    return {
        name: hashlib.sha256(local.detach().float().cpu().numpy().tobytes()).hexdigest()
        for name, param in model.named_parameters()
        if (local := to_local(param)).numel()
    }


def _head_path(model) -> str:
    return f"{mtp_head_path(model, require_mtp_head_spec(model.config))}."


def _split(digests: dict[str, str], prefix: str) -> tuple[dict[str, str], dict[str, str]]:
    """The head's digests and the trunk's."""
    head = {name: digest for name, digest in digests.items() if name.startswith(prefix)}
    return head, {name: digest for name, digest in digests.items() if name not in head}


def _every_tensor_moved_on_some_rank(ctx, before: dict[str, str], after: dict[str, str]) -> bool:
    """COLLECTIVE. At a fit's learning rate stochastic rounding moves a norm's weights a few elements at a time: 0 of a
    256-element ``q_norm`` in two steps, so one rank's slice can hold still while the tensor moves."""
    moved = {name for name in before if before[name] != after[name]}
    gathered: list = [(set(before), moved)]
    if ctx.world_size > 1:
        gathered = [None] * ctx.world_size
        dist.all_gather_object(gathered, (set(before), moved))
    held = set().union(*(names for names, _moved in gathered))
    stayed = held - set().union(*(rank_moved for _names, rank_moved in gathered))
    if stayed:
        log(f"  head tensors that kept their bytes on every rank: {sorted(stayed)}")
    return bool(held) and not stayed


def _human_streams(cli, tokenizer) -> tuple[list, list]:
    template = _RELEASE_TEMPLATE if cli.release else None
    rows = {"length": _LENGTH, "max_prompt": _MAX_PROMPT, "template_kwargs": template}
    return (
        first_turn_rows(ultrachat_conversations("train_gen"), tokenizer, _TRAIN_ROWS, **rows),
        first_turn_rows(ultrachat_conversations("test_gen"), tokenizer, _HELD_OUT_ROWS, **rows),
    )


def _sample_replies(ctx, model, tokenizer, prompts: list[list[int]]) -> list[tuple[list[int], int]]:
    """COLLECTIVE: the trunk's own replies at temperature 1 over this rank's slice, gathered. A reply stops at its first
    end-of-turn token."""
    eos = model.generation_config.eos_token_id
    stops = {eos} if isinstance(eos, int) else set(eos)
    pad = tokenizer.pad_token_id
    local = prompts[ctx.rank :: ctx.world_size]
    rows: list[tuple[list[int], int]] = []
    for offset in range(0, len(local), _SAMPLE_BATCH):
        batch = local[offset : offset + _SAMPLE_BATCH]
        width = max(len(prompt) for prompt in batch)
        input_ids = torch.tensor([[pad] * (width - len(prompt)) + prompt for prompt in batch], device=ctx.device)
        attention_mask = torch.tensor(
            [[0] * (width - len(prompt)) + [1] * len(prompt) for prompt in batch], device=ctx.device
        )
        sampled = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            max_new_tokens=_SAMPLED_REPLY,
            pad_token_id=pad,
        )[:, width:].tolist()
        for prompt, reply in zip(batch, sampled, strict=True):
            end = next((index + 1 for index, token in enumerate(reply) if token in stops), len(reply))
            rows.append((prompt + reply[:end], len(prompt)))
    gathered: list = [None] * ctx.world_size
    dist.all_gather_object(gathered, rows)
    return [row for rank_rows in gathered for row in rank_rows]


def _own_sample_streams(ctx, cli, tokenizer, train: list, held_out: list) -> tuple[list, list]:
    """COLLECTIVE: the same prompts, each with a reply the trunk sampled, as a served stream's next token comes from the
    trunk itself rather than a human. Each rank samples on a plain load of its own, freed before the fit's."""
    torch.manual_seed(_SEED + ctx.rank)
    model = auto_load_model(
        _checkpoint(cli), dtype=torch.bfloat16, attn_implementation="sdpa", text_only=True, device_map={"": ctx.device}
    )
    finalize_loaded_model(model)
    train_prompts = [ids[:start] for ids, start in train[:_SAMPLED_TRAIN_ROWS]]
    held_out_prompts = [ids[:start] for ids, start in held_out]
    streams = (
        _sample_replies(ctx, model, tokenizer, train_prompts),
        _sample_replies(ctx, model, tokenizer, held_out_prompts),
    )
    del model
    cleanup_memory()
    return streams


def _load_streams(ctx, cli, tokenizer, fit: _Fit):
    """COLLECTIVE: the train and held-out datasets; ``--stream_dir`` reads prepared ones."""
    if cli.stream_dir:
        splits = load_from_disk(cli.stream_dir)
        return splits["train"], splits["test"]
    streams = _human_streams(cli, tokenizer) if is_global_main_process() else None
    train, held_out = broadcast_from_rank0(streams)
    if fit.own_samples:
        train, held_out = _own_sample_streams(ctx, cli, tokenizer, train, held_out)
    return off_by_one_rows_dataset(train) if fit.off_by_one else rows_dataset(train), rows_dataset(held_out)


def _run(
    cli,
    fit: _Fit,
    scope: MtpScope | None,
    streams,
    tokenizer,
    output: str,
    *,
    gate_gain: float | None = None,
    **config_fields,
):
    """COLLECTIVE: one run over a fresh load; ``scope`` ``None`` builds no head. ``(trainer, model, gate)``."""
    fused = cli.kernel == "fused"
    liger = {"use_liger_kernel": fused, "liger_kernel_config": {"fused_linear_cross_entropy": True} if fused else None}
    parallelism = ParallelismConfig(ep_size=cli.ep_size)
    request = None
    if scope is not None:
        request = MtpHeadRequest(
            MtpObjective(_WEIGHT, fit.loss, scope), REAL_STALE_HEAD if fit.start == _STALE_START_NAME else None
        )
    model, _ = load_distributed_model(
        _checkpoint(cli),
        parallelism,
        dtype=torch.bfloat16,
        attn_implementation=cli.attn or fit.attn,
        trust_remote_code=False,
        text_only_model=fit.text_only,
        mtp_head=request,
        **liger,
    )
    config = SFTConfig(
        output_dir=output,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        bf16=True,
        gradient_checkpointing=scope != "head_only",
        eval_strategy="steps",
        eval_on_start=True,
        logging_steps=1,
        report_to="none",
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        seed=_SEED,
        **liger,
        **config_fields,
    )
    finalize_liger_after_direct_load(config, fused, model)
    gate = None if gate_gain is None else MtpAcceptanceGate(gate_gain)
    trainer = DistributedSFTTrainer(
        model=model,
        args=config,
        train_dataset=streams[0],
        eval_dataset=streams[1],
        processing_class=_export_processing_class(cli, tokenizer),
        data_collator=collate_tokenized_rows,
        parallelism_config=parallelism,
        callbacks=[gate] if gate is not None else None,
    )
    return trainer, model, gate


def _export_processing_class(cli, tokenizer):
    """COLLECTIVE. The processor a composite's export needs for an engine to load it, as the SFT script passes it."""
    source = argparse.Namespace(model_name_or_path=_checkpoint(cli), model_revision=None, trust_remote_code=False)
    return load_vlm_processor(source, required=False) or tokenizer


def _readings(trainer) -> list[dict]:
    return [entry for entry in trainer.state.log_history if f"eval_{MTP_ACCEPTANCE_TV}" in entry]


def _curve(readings: list[dict]) -> list[tuple[int, float, float]]:
    return [
        (entry["step"], entry[f"eval_{MTP_ACCEPTANCE_TV}"], entry[f"eval_{MTP_ACCEPTANCE_GREEDY}"])
        for entry in readings
    ]


def _finished(trainer, model) -> None:
    trainer.cleanup_ep()
    del trainer, model
    cleanup_memory()
    barrier()


def _stored_keys(checkpoint: str) -> set[str]:
    return set(resolve_checkpoint_weights(checkpoint).require_safetensors("checkpoint").weight_map)


def _export_holds_the_head(checkpoint: str, release: str, prefix: str, config) -> bool:
    """On a save rank, ``checkpoint`` holds every tensor of the release's head outside a fused bank's, whose layout is the
    save's own, at the level this run's class writes it. Bytes and the bank's keys are the tiny rows'."""
    if not is_fs_save_rank():
        return True
    stored = _stored_keys(release)
    level = stored_head_prefix(stored, config, prefix[:-1])
    head = {f"{prefix}{key[len(level) :]}" for key in stored if key.startswith(level) and ".mlp.experts." not in key}
    return bool(head) and head <= _stored_keys(checkpoint)


def _head_differs_from_the_release(checkpoint: str, release: str, prefix: str, config) -> list[str]:
    """On a save rank, each head tensor whose key, dtype or bytes leave the release's, at the level this run's class
    writes it."""
    if not is_fs_save_rank():
        return []
    level = stored_head_prefix(_stored_keys(release), config, prefix[:-1])
    stored = {
        f"{prefix}{key[len(level) :]}": tensor
        for key, tensor in iter_checkpoint_tensors(release, predicate=lambda key: key.startswith(level))
    }
    written = dict(iter_checkpoint_tensors(checkpoint, predicate=lambda key: key.startswith(prefix)))
    apart = set(stored) ^ set(written)
    apart.update(
        key
        for key in stored.keys() & written.keys()
        if stored[key].dtype != written[key].dtype or not torch.equal(stored[key], written[key])
    )
    return sorted(apart)


def _untrained_export_row(cli, streams, tokenizer, root: str) -> dict:
    """COLLECTIVE. What an export writes of an untrained tail, per expert with its router bias in fp32, is what the
    release stores."""
    output = cli.export_dir or os.path.join(root, cli.row)
    fit = _Fit(start="own", text_only=cli.text_only)
    trainer, model, _ = _run(cli, fit, "head_only", streams, tokenizer, output, max_steps=1, eval_steps=_EVAL_EVERY)
    prefix, config = _head_path(model), model.config
    trainer.save_model(output)
    _finished(trainer, model)
    apart = _head_differs_from_the_release(output, cached_source(_checkpoint(cli)), prefix, config)
    log(f"  {_UNTRAINED_ROW}: {len(apart)} head tensors left the release's bytes: {apart[:4]}")
    return {"checks": {"the_export_holds_the_release_head_byte_for_byte": rank_consensus(not apart).all_ok}}


def _fit_row(ctx, cli, fit: _Fit, streams, tokenizer, root: str) -> dict:
    steps = cli.steps or fit.steps
    output = cli.export_dir or os.path.join(root, cli.row)
    trainer, model, gate = _run(
        cli,
        fit,
        "head_only",
        streams,
        tokenizer,
        output,
        gate_gain=0.0,
        max_steps=steps,
        learning_rate=cli.learning_rate or fit.learning_rate,
        warmup_steps=fit.warmup_steps,
        lr_scheduler_type="constant_with_warmup",
        eval_steps=min(_EVAL_EVERY, steps),
        save_strategy="steps" if cli.export_dir else "no",
        save_steps=steps,
    )
    prefix = _head_path(model)
    fused_kernel = (getattr(model.config, LIGER_APPLIED_CONFIG_ATTR, None) or {}).get("fused_linear_cross_entropy")
    head_before, trunk_before = _split(_local_digests(model), prefix)
    trainer.train()
    readings = _readings(trainer)
    head_after, trunk_after = _split(_local_digests(model), prefix)
    log(f"  {cli.row}: step, acceptance, greedy {_curve(readings)}; gate stopped: {gate.stopped}")
    if gate.stopped:
        log(f"  {cli.row}: {gate.stop_reason}")
    _finished(trainer, model)

    bars = fit.bars if REAL_BARS_APPLY and not cli.release else AcceptanceBars()
    checks = acceptance_checks(readings, gate.stopped, bars)
    checks["every_reading_is_finite"] = all(math.isfinite(entry[f"eval_{MTP_ACCEPTANCE_TV}"]) for entry in readings)
    if (fit.text_only or cli.release == _TEXT_RELEASE) and cli.kernel == "fused":
        checks["the_term_took_the_fused_kernel"] = bool(fused_kernel)
    checks["the_trunk_kept_its_bytes"] = rank_consensus(trunk_before == trunk_after).all_ok
    checks["every_head_tensor_moved"] = _every_tensor_moved_on_some_rank(ctx, head_before, head_after)
    if cli.export_dir:
        release = cached_source(REAL_STALE_HEAD if fit.start == _STALE_START_NAME else _checkpoint(cli))
        checkpoint = os.path.join(cli.export_dir, f"checkpoint-{steps}")
        checks["the_export_holds_the_head"] = rank_consensus(
            _export_holds_the_head(checkpoint, release, prefix, model.config)
        ).all_ok
    return {"checks": checks, "metrics": {"acceptance": _curve(readings), "gate_stopped": gate.stopped}}


def _detached_row(ctx, cli, streams, tokenizer, root: str) -> dict:
    """COLLECTIVE. The clip off and determinism on, so each detached run's trunk bytes must be the run with no head's
    and its next-token curve the other head loss's, while the head moves and the trunk moved from its start. The run
    with no head logs ``loss``, a mean over the ranks, against ``ntp_ce``'s world fraction, so its curve compares to
    neither."""
    pin_deterministic_ep_dispatch()
    steps = cli.steps or _DETACHED_STEPS
    schedule = {
        "max_steps": steps,
        "learning_rate": cli.learning_rate or _DETACHED_LR,
        "warmup_steps": _DETACHED_WARMUP,
        "lr_scheduler_type": "cosine",
        "eval_steps": min(_DETACHED_EVAL_EVERY, steps),
        "save_strategy": "no",
        "max_grad_norm": 0,
        "full_determinism": True,
    }
    fit = _Fit(start="own", text_only=cli.text_only)
    trainer, model, _ = _run(cli, fit, None, streams, tokenizer, os.path.join(root, "off"), **schedule)
    start = _local_digests(model)
    trainer.train()
    off_trunk = _local_digests(model)
    _finished(trainer, model)

    checks = {"the_run_with_no_head_moved_its_trunk": rank_consensus(start != off_trunk).all_ok}
    metrics: dict = {}
    next_token_curves: dict[str, list[float]] = {}
    for loss in ("soft", "hard"):
        fit = _Fit(start="own", loss=loss, text_only=cli.text_only)
        output = os.path.join(root, f"detached_{loss}")
        trainer, model, _ = _run(cli, fit, "detached", streams, tokenizer, output, **schedule)
        prefix = _head_path(model)
        head_before, _trunk_before = _split(_local_digests(model), prefix)
        trainer.train()
        history = [entry for entry in trainer.state.log_history if "loss" in entry]
        readings = _readings(trainer)
        head_after, trunk = _split(_local_digests(model), prefix)
        _finished(trainer, model)
        next_token_curves[loss] = [entry["ntp_ce"] for entry in history]
        log(f"  detached {loss}: step, acceptance, greedy {_curve(readings)}")
        differing = sorted(name for name in off_trunk.keys() | trunk.keys() if off_trunk.get(name) != trunk.get(name))
        if differing:
            log(
                f"  detached {loss}: {len(differing)} trunk tensors left the run with no head's bytes: {differing[:4]}"
            )
        checks[f"{loss}_trunk_bytes_are_the_run_with_no_head"] = rank_consensus(not differing).all_ok
        checks[f"{loss}_every_head_tensor_moved"] = _every_tensor_moved_on_some_rank(ctx, head_before, head_after)
        checks[f"{loss}_scored_at_start_and_end"] = (
            len(readings) >= 2 and readings[0]["step"] == 0 and readings[-1]["step"] == steps
        )
        metrics[loss] = _curve(readings)
        if loss == "soft" and REAL_BARS_APPLY and not cli.release:
            checks["soft_acceptance_tracks_the_moving_trunk"] = (
                readings[-1][f"eval_{MTP_ACCEPTANCE_TV}"]
                >= readings[0][f"eval_{MTP_ACCEPTANCE_TV}"] - _ACCEPTANCE_DROP_BAR
            )
    gap = max(
        abs(soft - hard) / abs(hard)
        for soft, hard in zip(next_token_curves["soft"], next_token_curves["hard"], strict=True)
    )
    log(f"  detached: worst gap between the next-token curves of soft and hard {gap:.3e}")
    checks["next_token_curve_is_the_same_under_either_head_loss"] = gap <= _SAME_CURVE_BAR
    return {"checks": checks, "metrics": metrics}


@gpu_test_main(min_world_size=1)
def run(ctx) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--row", choices=(*sorted(_FITS), _DETACHED_ROW, _UNTRAINED_ROW), required=True)
    parser.add_argument("--ep_size", type=int, default=8)
    parser.add_argument("--kernel", choices=("fused", "chunked"), default="fused")
    parser.add_argument("--export_dir", default=None, help="Keep the fit's last checkpoint here.")
    parser.add_argument("--stream_dir", default=None, help="A saved DatasetDict of pre-tokenized train and test rows.")
    parser.add_argument("--steps", type=int, default=None, help="Stop early, for a short look; a bar may not hold.")
    parser.add_argument("--learning_rate", type=float, default=None, help="Replace the row's, for such a look.")
    parser.add_argument("--release", choices=sorted(NEMOTRON_RELEASES), default=None, help="A Nemotron-H release.")
    parser.add_argument("--text_only", action="store_true", help="Load a composite through its text class.")
    parser.add_argument("--attn", default=None, help="Replace the row's attention.")
    cli = parser.parse_args()
    fit = _FITS.get(cli.row, _Fit(start="own"))
    if cli.text_only:
        fit = replace(fit, text_only=True)
    if cli.release and fit.start == _STALE_START_NAME:
        parser.error(f"--release has no stale head to start from; run an own_* or {_DETACHED_ROW} row.")
    for source in (_checkpoint(cli), *(() if cli.release else (REAL_STALE_HEAD,))):
        if cached_source(source) is None:
            log(
                f"SKIP: {source} is not in the local Hub cache (HALO_TEST_MTP_REAL_MODEL, HALO_TEST_MTP_REAL_STALE_HEAD, "
                "HALO_TEST_NEMOTRON_MODEL, HALO_TEST_NEMOTRON_LIGHTNING_MODEL)"
            )
            raise SystemExit(0)

    tokenizer = AutoTokenizer.from_pretrained(_checkpoint(cli))
    streams = _load_streams(ctx, cli, tokenizer, fit)
    log(f"  {cli.row}: {len(streams[0])} train rows, {len(streams[1])} held out")
    root = shared_output_dir(ctx)
    if cli.row == _DETACHED_ROW:
        return _detached_row(ctx, cli, streams, tokenizer, root)
    if cli.row == _UNTRAINED_ROW:
        return _untrained_export_row(cli, streams, tokenizer, root)
    return _fit_row(ctx, cli, fit, streams, tokenizer, root)


if __name__ == "__main__":
    raise SystemExit(run())
