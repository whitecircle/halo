#!/usr/bin/env python
"""Qwen3.8-Flash-Next's release cut to ``--layers`` decoder layers under EP on 8 GPUs: what tiny models can't show.

Packing runs in fp32: bf16 rounding flips top-10-of-512 routing near-ties, and the flips compound past any logits bar.
Routing flips also reach every position of the export's CPU load at full depth, so only the mean NLL holds there. Rank
0's reload (~3.5 min per row at full depth) outlasts the NCCL watchdog: raise ``DIST_NCCL_TIMEOUT_MINUTES``. Skips
without the local release.
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

from halo.checkpoint.format import SAFETENSORS_INDEX_FILE
from halo.checkpoint.reading import resolve_checkpoint_weights
from halo.distributed.expert_parallel.families.qwen4_exp import EPQwen4ExpMoELayer
from halo.distributed.parallelism_config import ParallelismConfig
from halo.env import env_str
from halo.model_loading.parallel.lazy_plan import is_lazy_loadable
from halo.model_loading.parallel.loader import load_distributed_model
from halo.models.config_access import text_config
from halo.models.external_tables import EXTERNAL_TABLES_STAMP, named_external_tables
from halo.models.qwen4_exp.sparse_attention import qsa_attention_forward, select_qsa_keys
from tests.common.checkpoint_io import loading_problems
from tests.common.datasets import repository_text_rows
from tests.common.distributed import shared_output_dir
from tests.common.ep_reference import random_token_batch
from tests.common.harness import gpu_test_main
from tests.common.memory import cleanup_memory, host_peak_rss_gib, reset_host_peak_rss
from tests.common.models import QWEN3_8_FLASH_NEXT
from tests.common.qwen4_exp import packed_markers, packed_positions, selection_keys
from tests.common.rank_log import log
from tests.common.resume_export import run_release_resume, save_gathered_ep

MODEL = env_str("HALO_TEST_QWEN4_EXP_MODEL", QWEN3_8_FLASH_NEXT)
SEED = 0
# Peak host RSS of one rank's lazy EP load of the cut model: its experts' slices, not the release's 340 GB.
LOAD_RSS_GIB = 40.0
# Wider than the 2,048-key budget, so most of its queries select.
QSA_DOCUMENT = 3000
# Queries whose tiled selection may differ from transformers' (near-ties the two matmul shapes round apart): one block
# pair swapped.
QSA_TIE_ROWS_MAX_FRACTION = 1e-2
QSA_TIE_SWAPPED_BLOCKS = 1
# Fraction of the reference's largest magnitude, in bf16.
QSA_OUTPUT_REL_MAX = 2e-2
# Fraction of the alone logits' largest magnitude in fp32, and the positions allowed past it: tokens a near-tie still
# routes to another expert.
ISOLATED_REL_MAX = 2e-2
ROUTING_FLIP_MAX_FRACTION = 2e-2
EP_LOSS_ABS_MAX = 1e-2
LOSS_TOKENS = 1024
EXPORT_LR = 1e-3
# How far mean NLLs may part, in nats: the same weights part by 0.0002 on the four-layer cut (CPU bf16 vs fp32) and by
# 0.0066 at full depth (EP vs the served release), one 2048-token row alone by up to 0.014 there.
EXPORT_TEXT_ROWS = 4
EXPORT_TEXT_LENGTH = 2048
EXPORT_NLL_ABS_MAX = 2e-2
LOGPROB_DIFF_QUANTILES = {"p50": 0.5, "p90": 0.9, "p99": 0.99}
# Per exported depth, where the floor is measured. A mean NLL hides a corrupted layer whose errors cancel (one layer's
# expert bank reversed on the cut: gap +0.002, p50 0.17). On the cut the CPU forward in bf16 vs fp32 parts by p50 0.010
# / p90 0.034, and reversing any one layer's experts moves them to p50 >= 0.14 / p90 >= 0.47. Routing flips widen the
# floor past the cut (p50 0.066 / p90 0.52 at full depth), so other depths get the mean bar, with byte identity at
# --steps 0.
EXPORT_LOGPROB_DIFF_MAX = {4: {"p50": 0.05, "p90": 0.15}}


def _release_directory() -> str | None:
    if os.path.isdir(MODEL):
        return MODEL
    try:
        return snapshot_download(MODEL, local_files_only=True)
    except LocalEntryNotFoundError:
        return None


def _depth_overrides(source: str, layers: int) -> dict | None:
    config = text_config(AutoConfig.from_pretrained(source))
    if layers >= config.num_hidden_layers:
        return None
    return {"num_hidden_layers": layers, "layer_types": config.layer_types[:layers]}


def _load(source: str, parallelism_config: ParallelismConfig, layers: int, dtype: torch.dtype = torch.bfloat16):
    """COLLECTIVE. ``source`` cut to ``layers`` as the training scripts load it: its text class, on SDPA."""
    return load_distributed_model(
        source,
        parallelism_config,
        dtype=dtype,
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
    """Fraction of logit rows whose largest difference exceeds ``bar`` of the reference's largest magnitude."""
    difference = (actual.float() - expected.float()).abs().amax(-1)
    return (difference > bar * expected.float().abs().max()).float().mean().item()


@torch.no_grad()
def _qsa_checks(model, device) -> tuple[dict[str, bool], dict[str, float]]:
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
    """Each document of a packed row against itself alone, and the negative arm: without the ``cu_seq_lens`` set the
    GatedDeltaNet layers carry each document into the next."""
    lengths = [seq // 6, seq - 2 * (seq // 6), seq // 6]
    generator = torch.Generator().manual_seed(SEED + 1)
    ids = torch.randint(0, model.config.vocab_size, (1, seq), generator=generator).to(device)
    positions = packed_positions([lengths]).to(device)
    packed = _logits(model, ids, position_ids=positions, **packed_markers(model.config, positions))
    crossing = _logits(
        model, ids, position_ids=positions, **packed_markers(model.config, positions, cu_seq_lens=False)
    )
    worst, isolated, crossed, start = 0.0, [], [], 0
    for length in lengths:
        document = slice(start, start + length)
        alone = _logits(model, ids[:, document])
        worst = max(worst, _relative(packed[document], alone))
        isolated.append(_beyond(packed[document], alone, ISOLATED_REL_MAX))
        crossed.append(_beyond(crossing[document], alone, ISOLATED_REL_MAX))
        start += length
    log(
        f"  packed {lengths} in {model.dtype}: each document against itself alone {worst:.3e} of its scale; positions past "
        f"{ISOLATED_REL_MAX:.0e} per document {' / '.join(f'{v:.2%}' for v in isolated)}, without cu_seq_lens "
        f"{' / '.join(f'{v:.2%}' for v in crossed)}"
    )
    return {
        "packed_documents_compute_their_logits_alone": max(isolated) <= ROUTING_FLIP_MAX_FRACTION,
        # The first document has nothing before it to leak from.
        "packed_row_without_cu_seq_lens_leaks_past_the_first_document": crossed[0] <= ROUTING_FLIP_MAX_FRACTION
        and min(crossed[1:]) > ROUTING_FLIP_MAX_FRACTION,
    }, {
        "packed_vs_alone_rel": worst,
        "packed_vs_alone_positions_beyond": max(isolated),
        "packed_without_cu_seq_lens_positions_beyond": min(crossed[1:]),
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

    batch = random_token_batch(model.config.vocab_size, batch=1, seq=LOSS_TOKENS, device=ctx.device, seed=SEED + 2)
    ep_loss = _loss(model, batch)
    del model
    cleanup_memory()
    reference, _ = _load(source, ParallelismConfig(ep_size=1), args.layers)
    reference_loss = _loss(reference.eval(), batch)
    log(f"  loss on one batch: EP {ep_loss:.6f}, unsharded {reference_loss:.6f}")
    checks["ep_loss_matches_unsharded"] = abs(ep_loss - reference_loss) <= EP_LOSS_ABS_MAX
    metrics |= {"ep_loss": ep_loss, "unsharded_loss": reference_loss}
    del reference
    cleanup_memory()

    # Every rank holds the collective load, so each runs the packing forwards itself.
    reference, _ = _load(source, ParallelismConfig(ep_size=1), args.layers, dtype=torch.float32)
    packing, packing_metrics = _packing_checks(reference.eval(), args.seq, ctx.device)
    checks |= packing
    metrics |= packing_metrics
    return {"checks": checks, "metrics": metrics}


def _hub_key(key: str, hub_keys: set[str]) -> str | None:
    """The release key an exported key stores, read through the text tower's ``language_model`` level."""
    for candidate in (key, key.replace("model.", "model.language_model.", 1)):
        if candidate in hub_keys:
            return candidate
    return None


def _identity_checks(export: str, source: str) -> dict[str, bool]:
    """Whether every tensor is the release's own under its key, byte for byte, streaming one at a time."""
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


def _token_logprobs(model, rows: torch.Tensor) -> torch.Tensor:
    """Fp32 log-probs per next token, on the CPU, one row per forward."""
    scores = []
    for row in rows:
        logits = _logits(model, row[None]).float()
        scores.append(torch.log_softmax(logits[:-1], dim=-1).gather(-1, row[1:, None])[:, 0].cpu())
    return torch.stack(scores)


def _artifact_checks(
    export: str, source: str, rows: torch.Tensor, trained: torch.Tensor
) -> tuple[dict[str, bool], dict[str, float]]:
    """Rank 0. The gathered save in the hub layout, and a stock CPU load of it scoring ``rows`` at the trained model's
    mean NLL and, at a depth with a measured floor, token by token (:data:`EXPORT_LOGPROB_DIFF_MAX`)."""
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
    reloaded_scores = _token_logprobs(reloaded.eval(), rows)
    reload_nll, trained_nll = -reloaded_scores.mean().item(), -trained.mean().item()
    row_gaps = trained.mean(-1) - reloaded_scores.mean(-1)
    gap = reload_nll - trained_nll
    per_token = (reloaded_scores - trained).abs().flatten()
    levels = torch.tensor(list(LOGPROB_DIFF_QUANTILES.values()))
    quantiles = dict(zip(LOGPROB_DIFF_QUANTILES, torch.quantile(per_token, levels).tolist(), strict=True))
    log(
        f"  stock CPU reload: problems {problems or 'none'}; mean NLL on {per_token.numel()} real-text tokens "
        f"{reload_nll:.5f} vs trained {trained_nll:.5f} (gap {gap:+.5f}, per row [{row_gaps.min().item():+.4f}, "
        f"{row_gaps.max().item():+.4f}]); per-token |logprob difference| "
        + ", ".join(f"{name} {value:.4f}" for name, value in quantiles.items())
    )
    checks["stock_reload_loads_every_weight"] = not problems
    checks["stock_reload_scores_real_text_at_the_trained_nll"] = abs(gap) <= EXPORT_NLL_ABS_MAX
    depth = text_config(reloaded.config).num_hidden_layers
    if (bars := EXPORT_LOGPROB_DIFF_MAX.get(depth)) is not None:
        checks["stock_reload_scores_each_token_at_the_trained_logprob"] = all(
            quantiles[name] <= bar for name, bar in bars.items()
        )
        log(f"  per-token bars at {depth} layers: {bars}")
    return checks, {
        "export_reload_nll": reload_nll,
        "export_trained_nll": trained_nll,
        "export_nll_gap": gap,
        **{f"export_logprob_abs_diff_{name}": value for name, value in quantiles.items()},
    }


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
    rows = torch.tensor(repository_text_rows(tokenizer, EXPORT_TEXT_ROWS, EXPORT_TEXT_LENGTH))
    trained = _token_logprobs(model.eval(), rows.to(ctx.device))

    export = args.export_dir or os.path.join(shared_output_dir(ctx), "export")
    save_gathered_ep(ctx, model, parallelism_config, tokenizer, export)
    del model
    cleanup_memory()
    checks, metrics = {}, {}
    if ctx.rank == 0:
        checks, metrics = _artifact_checks(export, source, rows, trained)
        if not args.steps:
            checks |= _identity_checks(export, source)
    return {"checks": ctx.broadcast_checks(checks), "metrics": metrics}


def run_resume(ctx, source: str, args) -> dict:
    return run_release_resume(ctx, lambda parallelism_config: _load(source, parallelism_config, args.layers))


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
