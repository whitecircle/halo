#!/usr/bin/env python
"""Qwen3.8-Flash-Next's release checkpoint cut to its first ``--layers`` decoder layers, under EP across 8 GPUs: what
the tiny-model suites cannot show at the release's widths.

``--mode layers``: the lazy EP load reads only those layers (each rank's peak host RSS below ``LOAD_RSS_GIB``) and
attaches the n-gram table outside the module state; on a ``--seq``-token document the tiled QSA selects
transformers' key blocks, a near-tie swapping at most one block pair on a few queries, and computes its attention
within the bf16 bar; each document of a packed row computes its logits alone; and the EP loss on one batch is the
unsharded (``ep_size = 1``) model's. A whole-model logits bar admits the few positions a bf16 near-tie routes to
another expert (``ROUTING_FLIP_MAX_FRACTION``); a leak or a wrong weight moves most of them.

``--mode export``: after ``--steps`` SGD steps on a batch every rank shares, the gathered save carries the hub's
keys (the n-gram table's ``shard_*`` parts, no MTP head), ``split_ngram_parts`` and no run stamp, and a stock
``from_pretrained`` of it on the CPU computes the trained model's logits. At ``--steps 0`` every exported tensor
is the release's own, byte for byte: with ``--layers 48 --export-dir DIR`` that is the identity export a served
comparison against the release starts from. Rank 0's stock reload and byte comparison of the whole release outlast
the default NCCL watchdog its peers wait under: raise ``DIST_NCCL_TIMEOUT_MINUTES`` for that row.

``--mode resume``: full fine-tuning from a gathered save of the cut model checkpoints every 3 steps, and a trainer
built from step 3's checkpoint trains to step 6 with its optimizer restored bit for bit, the unbroken run's losses
replayed and its final weights reproduced; the final checkpoint serves through stock transformers
(:func:`~tests.common.resume_export.run_resume_export`).

The release is read from the local Hub cache (``HALO_TEST_QWEN4_EXP_MODEL``: a repo id or a local directory); the
run skips without it.

    torchrun --nproc_per_node=8 tests/gpu/models/test_qwen4_exp_real_checkpoint.py --mode layers --layers 4 --seq 4096
    torchrun --nproc_per_node=8 tests/gpu/models/test_qwen4_exp_real_checkpoint.py --mode export
    torchrun --nproc_per_node=8 tests/gpu/models/test_qwen4_exp_real_checkpoint.py --mode resume
"""

import argparse
import json
import os
import sys

import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.masking_utils import create_causal_mask
from trl import SFTConfig

from halo.checkpoint.format import SAFETENSORS_INDEX_FILE
from halo.checkpoint.parallel.ep import save_ep_gathered_checkpoint
from halo.checkpoint.parallel.save import save_tokenizer
from halo.checkpoint.reading import resolve_checkpoint_weights
from halo.distributed.expert_parallel.families.qwen4_exp import EPQwen4ExpMoELayer
from halo.distributed.parallelism_config import ParallelismConfig
from halo.env import env_str
from halo.model_loading.parallel.lazy_plan import is_lazy_loadable
from halo.model_loading.parallel.loader import load_distributed_model
from halo.models.config_access import text_config
from halo.models.external_tables import EXTERNAL_TABLES_STAMP, named_external_tables
from halo.models.qwen4_exp.sparse_attention import qsa_attention_forward, select_qsa_keys
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.checkpoint_io import loading_problems, save_context, save_schedule
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import pin_deterministic_ep_dispatch, shared_output_dir
from tests.common.ep_reference import random_token_batch
from tests.common.harness import gpu_test_main
from tests.common.memory import cleanup_memory, host_peak_rss_gib, reset_host_peak_rss
from tests.common.models import QWEN3_8_FLASH_NEXT
from tests.common.qwen4_exp import packed_markers, packed_positions, selection_keys
from tests.common.rank_log import log
from tests.common.resume_export import ResumeExportRoute, run_resume_export

MODEL = env_str("HALO_TEST_QWEN4_EXP_MODEL", QWEN3_8_FLASH_NEXT)
SEED = 0
# Peak host RSS of one rank's lazy EP load of the cut model: its experts' slices, not the release's 340 GB.
LOAD_RSS_GIB = 40.0
# The QSA parity document: wider than the 2,048-key budget, so most of its queries select.
QSA_DOCUMENT = 3000
# Queries whose tiled selection may differ from transformers' (near-ties the two matmul shapes round apart), and
# how far: one block pair swapped.
QSA_TIE_ROWS_MAX_FRACTION = 1e-2
QSA_TIE_SWAPPED_BLOCKS = 1
# bf16 bars, as fractions of the reference's largest magnitude: the tiled attention against transformers', a packed
# document against itself alone, and a CPU reload against the trained model on the GPU.
QSA_OUTPUT_REL_MAX = 2e-2
ISOLATED_REL_MAX = 2e-2
EXPORT_LOGITS_REL_MAX = 1e-2
# Positions of a whole-model comparison allowed past its bar: the tokens a bf16 near-tie routes to another expert.
ROUTING_FLIP_MAX_FRACTION = 2e-2
# EP's loss against the unsharded model's on one batch.
EP_LOSS_ABS_MAX = 1e-2
LOSS_TOKENS = 1024
PROBE_TOKENS = 256
EXPORT_LR = 1e-3
RESUME_MAX_LENGTH = 512
RESUME_LR = 1e-4
SAVE_STEP = 3
TOTAL_STEPS = 2 * SAVE_STEP


def _release_directory() -> str | None:
    """The release's local directory: ``MODEL`` itself, or its Hub snapshot in the local cache."""
    if os.path.isdir(MODEL):
        return MODEL
    try:
        return snapshot_download(MODEL, local_files_only=True)
    except LocalEntryNotFoundError:
        return None


def _depth_overrides(source: str, layers: int) -> dict | None:
    """``model_config_overrides`` keeping the first ``layers`` decoder layers of ``source`` (none for all)."""
    config = text_config(AutoConfig.from_pretrained(source))
    if layers >= config.num_hidden_layers:
        return None
    return {"num_hidden_layers": layers, "layer_types": config.layer_types[:layers]}


def _load(source: str, parallelism_config: ParallelismConfig, layers: int):
    """COLLECTIVE. ``source`` cut to ``layers`` as the training scripts load it: its text class, on SDPA."""
    return load_distributed_model(
        source,
        parallelism_config,
        dtype=torch.bfloat16,
        trust_remote_code=False,
        attn_implementation="sdpa",
        model_config_overrides=_depth_overrides(source, layers),
        text_only_model=True,
    )


def _load_checks(model, source: str, layers: int, rss_gib: float) -> dict[str, bool]:
    tables = [table for _, table in named_external_tables(model)]
    return {
        "lazy_route_admitted": is_lazy_loadable(source) is True,
        "every_layer_ep_wrapped": sum(isinstance(m, EPQwen4ExpMoELayer) for m in model.modules()) == layers,
        "load_reads_only_the_cut_layers": rss_gib < LOAD_RSS_GIB,
        "ngram_table_attached_outside_parameters": len(tables) == 1
        and tables[0].is_attached
        and not any("ngram_embedding" in name for name, _ in model.named_parameters()),
    }


def _relative(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.float() - expected.float()).abs().max() / expected.float().abs().max()).item()


def _beyond(actual: torch.Tensor, expected: torch.Tensor, bar: float) -> float:
    """The fraction of ``[..., positions, vocab]`` logit rows whose largest difference exceeds ``bar`` of the
    reference's largest magnitude."""
    difference = (actual.float() - expected.float()).abs().amax(-1)
    return (difference > bar * expected.float().abs().max()).float().mean().item()


@torch.no_grad()
def _qsa_checks(model, device) -> tuple[dict[str, bool], dict[str, float]]:
    """The first sparse layer's tiled QSA against transformers' own, on one random document of real width."""
    config = model.config
    index = config.layer_types.index("indexed_attention")
    attention = model.model.layers[index].self_attn
    generator = torch.Generator(device=device).manual_seed(SEED)
    hidden = torch.randn(1, QSA_DOCUMENT, config.hidden_size, device=device, generator=generator, dtype=torch.bfloat16)
    positions = torch.arange(QSA_DOCUMENT, device=device)[None]
    position_embeddings = model.model.rotary_emb(hidden, positions.expand(3, -1, -1))
    mask = create_causal_mask(
        config=config,
        inputs_embeds=hidden,
        attention_mask=None,
        past_key_values=None,
        position_ids=positions,
        allow_is_causal_skip=False,
    )
    stock_keys = (attention.indexer(hidden, position_embeddings, mask, None) & mask)[:, 0].cpu()
    selection = select_qsa_keys(attention.indexer, hidden, position_embeddings)
    differing = (stock_keys != selection_keys(selection, 1, QSA_DOCUMENT)).sum(-1).flatten()
    tie_rows = int((differing > 0).sum())
    stock_output, _ = attention(hidden, position_embeddings, mask)
    tiled_output, _ = qsa_attention_forward(attention, hidden, position_embeddings)
    output_rel = _relative(tiled_output, stock_output)
    sparse_queries = int(((selection.blocks >= 0).sum(-1) == attention.indexer.block_topk).sum())
    log(
        f"  QSA layer {index}: {sparse_queries} queries at the full budget, {tie_rows} select differently "
        f"(at most {int(differing.max())} keys); output {output_rel:.3e} of its scale"
    )
    swap = 2 * QSA_TIE_SWAPPED_BLOCKS * selection.compress_ratio
    return {
        "qsa_document_crosses_the_budget": sparse_queries > 0,
        "qsa_selects_transformers_keys_outside_tie_rows": tie_rows <= QSA_TIE_ROWS_MAX_FRACTION * QSA_DOCUMENT
        and int(differing.max()) <= swap,
        "qsa_output_matches_transformers": output_rel <= QSA_OUTPUT_REL_MAX,
    }, {"qsa_tie_rows": tie_rows, "qsa_output_rel": output_rel}


def _logits(model, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
    with torch.no_grad():
        return model(input_ids=input_ids, use_cache=False, **kwargs).logits[0]


def _packing_checks(model, seq: int, device) -> tuple[dict[str, bool], dict[str, float]]:
    """COLLECTIVE. Each document of a packed row, the middle one past the indexer budget, against itself alone."""
    lengths = [seq // 6, seq - 2 * (seq // 6), seq // 6]
    generator = torch.Generator().manual_seed(SEED + 1)
    ids = torch.randint(0, model.config.vocab_size, (1, seq), generator=generator).to(device)
    positions = packed_positions([lengths]).to(device)
    packed = _logits(model, ids, position_ids=positions, **packed_markers(model.config, positions))
    worst, beyond, start = 0.0, 0.0, 0
    for length in lengths:
        alone = _logits(model, ids[:, start : start + length])
        worst = max(worst, _relative(packed[start : start + length], alone))
        beyond = max(beyond, _beyond(packed[start : start + length], alone, ISOLATED_REL_MAX))
        start += length
    log(
        f"  packed {lengths}: each document against itself alone {worst:.3e} of its scale, {beyond:.2%} of positions "
        f"past {ISOLATED_REL_MAX:.0e}"
    )
    return {"packed_documents_compute_their_logits_alone": beyond <= ROUTING_FLIP_MAX_FRACTION}, {
        "packed_vs_alone_rel": worst,
        "packed_vs_alone_positions_beyond": beyond,
    }


def _loss(model, batch: tuple[torch.Tensor, torch.Tensor]) -> float:
    with torch.no_grad():
        return model(input_ids=batch[0], labels=batch[1], use_cache=False).loss.item()


def run_layers(ctx, source: str, args) -> dict:
    reset_host_peak_rss()
    model, _ = _load(source, ParallelismConfig(ep_size=ctx.world_size), args.layers)
    rss_gib = host_peak_rss_gib()
    log(f"  lazy EP load of {args.layers} layers: peak host RSS {rss_gib:.1f} GiB on rank {ctx.rank}")
    model.eval()
    checks = _load_checks(model, source, args.layers, rss_gib)
    metrics = {"load_peak_host_rss_gib": rss_gib}
    qsa = {}
    if ctx.rank == 0:
        qsa, qsa_metrics = _qsa_checks(model, ctx.device)
        metrics |= qsa_metrics
    checks |= ctx.broadcast_checks(qsa)
    packing, packing_metrics = _packing_checks(model, args.seq, ctx.device)
    checks |= packing
    metrics |= packing_metrics

    batch = random_token_batch(model.config.vocab_size, batch=1, seq=LOSS_TOKENS, device=ctx.device, seed=SEED + 2)
    ep_loss = _loss(model, batch)
    del model
    cleanup_memory()
    reference, _ = _load(source, ParallelismConfig(ep_size=1), args.layers)
    reference_loss = _loss(reference.eval(), batch)
    log(f"  loss on one batch: EP {ep_loss:.6f}, unsharded {reference_loss:.6f}")
    checks["ep_loss_matches_unsharded"] = abs(ep_loss - reference_loss) <= EP_LOSS_ABS_MAX
    metrics |= {"ep_loss": ep_loss, "unsharded_loss": reference_loss}
    return {"checks": checks, "metrics": metrics}


def _hub_key(key: str, hub_keys: set[str]) -> str | None:
    """The release key an exported key stores, read through the text tower's ``language_model`` level."""
    for candidate in (key, key.replace("model.", "model.language_model.", 1)):
        if candidate in hub_keys:
            return candidate
    return None


def _identity_checks(export: str, source: str) -> dict[str, bool]:
    """Whether every tensor of ``export`` is the release's own under its key, byte for byte. Streams one tensor at a
    time."""
    exported, released = (resolve_checkpoint_weights(path).weight_map for path in (export, source))
    hub_keys = set(released)
    unmatched, differing = [], []
    for key, name in exported.items():
        hub_key = _hub_key(key, hub_keys)
        if hub_key is None:
            unmatched.append(key)
            continue
        with (
            safe_open(os.path.join(export, name), framework="pt") as written,
            safe_open(os.path.join(source, released[hub_key]), framework="pt") as release,
        ):
            if not torch.equal(written.get_tensor(key), release.get_tensor(hub_key)):
                differing.append(key)
    log(f"  identity export: {len(unmatched)} keys the release lacks, {len(differing)} tensors differ")
    return {"export_keys_are_the_release_keys": not unmatched, "export_tensors_are_the_release_tensors": not differing}


def _artifact_checks(export: str, source: str, probe: torch.Tensor, trained_logits: torch.Tensor) -> dict[str, bool]:
    """The gathered save as the hub layout serves it, and a stock CPU load of it computing ``trained_logits``.
    Rank 0."""
    with open(os.path.join(export, SAFETENSORS_INDEX_FILE)) as handle:
        keys = set(json.load(handle)["weight_map"])
    with open(os.path.join(export, "config.json")) as handle:
        exported_config = handle.read()
    parts = text_config(AutoConfig.from_pretrained(export)).split_ngram_parts
    release_parts = text_config(AutoConfig.from_pretrained(source)).split_ngram_parts
    table_keys = {key for key in keys if ".ngram_embedding." in key}
    checks = {
        "table_written_as_the_release_parts": parts == release_parts
        and len(table_keys) == release_parts
        and all(".ngram_embedding.shard_" in key for key in table_keys),
        "no_mtp_or_vision_keys": not any(key.startswith("mtp") or ".visual." in key for key in keys),
        "no_run_stamp_in_config": EXTERNAL_TABLES_STAMP not in exported_config,
    }
    reloaded, info = AutoModelForCausalLM.from_pretrained(export, dtype=torch.bfloat16, output_loading_info=True)
    problems = loading_problems(info)
    reloaded_logits = _logits(reloaded.eval(), probe.cpu())
    reload_rel = _relative(reloaded_logits, trained_logits)
    beyond = _beyond(reloaded_logits, trained_logits, EXPORT_LOGITS_REL_MAX)
    log(
        f"  stock CPU reload: problems {problems or 'none'}, logits {reload_rel:.3e} of their scale, {beyond:.2%} of "
        f"positions past {EXPORT_LOGITS_REL_MAX:.0e}"
    )
    checks["stock_reload_loads_every_weight"] = not problems
    checks["stock_reload_computes_the_trained_logits"] = beyond <= ROUTING_FLIP_MAX_FRACTION
    return checks


def _save_gathered(ctx, model, parallelism_config: ParallelismConfig, tokenizer, directory: str) -> None:
    """COLLECTIVE. The gathered EP save of ``model`` and its tokenizer into ``directory``, as a run's save writes it."""
    save = save_context(model, parallelism_config, tokenizer=tokenizer, has_ep_layers=True)
    save_tokenizer(save, directory)
    save_ep_gathered_checkpoint(save, directory)
    ctx.barrier()


def run_export(ctx, source: str, args) -> dict:
    parallelism_config = ParallelismConfig(ep_size=ctx.world_size)
    model, tokenizer = _load(source, parallelism_config, args.layers)
    if args.steps:
        model.train()
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=EXPORT_LR)
        batch = random_token_batch(model.config.vocab_size, batch=1, seq=LOSS_TOKENS, device=ctx.device, seed=SEED)
        for step in range(args.steps):
            loss = model(input_ids=batch[0], labels=batch[1], use_cache=False).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            log(f"  step {step + 1}: loss {loss.item():.6f}")
    probe = torch.randint(
        0, model.config.vocab_size, (1, PROBE_TOKENS), generator=torch.Generator().manual_seed(SEED + 3)
    ).to(ctx.device)
    trained_logits = _logits(model.eval(), probe).float().cpu()

    export = args.export_dir or os.path.join(shared_output_dir(ctx), "export")
    _save_gathered(ctx, model, parallelism_config, tokenizer, export)
    del model
    cleanup_memory()
    checks = {}
    if ctx.rank == 0:
        checks = _artifact_checks(export, source, probe, trained_logits)
        if not args.steps:
            checks |= _identity_checks(export, source)
    return {"checks": ctx.broadcast_checks(checks)}


def run_resume(ctx, source: str, args) -> dict:
    # The resumed run replays the unbroken one exactly, which DeepEP's atomic receive order would not.
    pin_deterministic_ep_dispatch()
    parallelism_config = ParallelismConfig(ep_size=ctx.world_size)
    root = shared_output_dir(ctx)
    base = os.path.join(root, "base")
    model, tokenizer = _load(source, parallelism_config, args.layers)
    _save_gathered(ctx, model, parallelism_config, tokenizer, base)
    del model
    cleanup_memory()
    train_dataset = create_sft_dataset(32, tokenizer, seed=SEED)

    def build_trainer(model, output_dir, *, max_steps, save_every, resume_checkpoint):
        return DistributedSFTTrainer(
            model=model,
            args=SFTConfig(
                output_dir=output_dir,
                max_steps=max_steps,
                **save_schedule(save_every),
                per_device_train_batch_size=1,
                learning_rate=RESUME_LR,
                bf16=True,
                gradient_checkpointing=True,
                logging_steps=1,
                report_to=[],
                logging_nan_inf_filter=False,
                max_length=RESUME_MAX_LENGTH,
                dataloader_drop_last=True,
                dataloader_num_workers=0,
            ),
            train_dataset=train_dataset,
            processing_class=tokenizer,
            parallelism_config=parallelism_config,
        )

    route = ResumeExportRoute(build_trainer, AutoModelForCausalLM, dtype=torch.bfloat16, attn_implementation="sdpa")
    checks, metrics = run_resume_export(
        ctx, route, parallelism_config, base=base, root=root, save_step=SAVE_STEP, total_steps=TOTAL_STEPS
    )
    return {"checks": checks, "metrics": metrics}


MODES = {"layers": run_layers, "export": run_export, "resume": run_resume}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=sorted(MODES), required=True)
    parser.add_argument("--layers", type=int, default=4, help="decoder layers kept from the release (>= 2: PLE)")
    parser.add_argument("--seq", type=int, default=4096, help="packed row width of --mode layers")
    parser.add_argument("--steps", type=int, default=1, help="SGD steps before --mode export saves")
    parser.add_argument("--export-dir", default=None, help="where --mode export writes and keeps its save")
    return parser.parse_args()


@gpu_test_main(exact_world_size=8, prefix="qwen4_exp_real_checkpoint")
def run(ctx) -> dict:
    args = _arguments()
    source = _release_directory()
    log(f"=== {MODEL} at {source}, first {args.layers} layers, --mode {args.mode}")
    return MODES[args.mode](ctx, source, args)


if __name__ == "__main__":
    if _release_directory() is None:
        log(f"SKIP: {MODEL} is not in the local Hub cache (set HALO_TEST_QWEN4_EXP_MODEL to a present checkpoint)")
        sys.exit(0)
    run()
