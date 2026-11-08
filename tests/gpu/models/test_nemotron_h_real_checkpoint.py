#!/usr/bin/env python
"""A Nemotron-H release under EP on 8 GPUs: what tiny models can't show. ``--release`` picks Super VL, a composite, or
Lightning, whose tower sits at the checkpoint's root and carries no latent.

``--mode layers`` cuts it to its first layers: the lazy route, the fp32 bias as stored, EP against one rank, and packed
documents against themselves alone. ``--mode export`` writes it on either route and holds every tensor to the release,
byte for byte. ``--mode resume`` replays a run exactly. ``--mode profile`` times the whole tower per block kind, the
throughput the docs state and the scan's share of it. Skips without the local release.
"""

import argparse
import json
import math
import os
import statistics
import sys
import time

import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
from transformers.models.nemotron_h import modeling_nemotron_h as nemotron_modeling

from halo.checkpoint.reading import resolve_checkpoint_weights
from halo.distributed.expert_parallel.families.nemotron_h import EPNemotronHMoELayer
from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.lazy_plan import is_lazy_loadable
from halo.model_loading.parallel.loader import load_distributed_model
from halo.models.attention.selection import effective_attn_implementation
from halo.models.families.nemotron_h import _document_chunk_scan
from halo.models.gradient_checkpointing import enable_scoped_gradient_checkpointing
from halo.models.introspection.config_access import text_config
from halo.models.segment_markers import SegmentMarkers, segment_marker_kwargs
from tests.common.checkpoint_io import loading_problems
from tests.common.datasets import repository_text_rows
from tests.common.distributed import shared_output_dir
from tests.common.harness import gpu_test_main, skip_unless_blackwell
from tests.common.memory import cleanup_memory, host_peak_rss_gib, reset_host_peak_rss
from tests.common.nemotron_h import NEMOTRON_RELEASES, expected_hub_keys, hub_text_prefix, release_model_id
from tests.common.rank_log import log
from tests.common.resume_export import run_release_resume, save_gathered_ep

SEED = 0
BIAS_SLOT = "e_score_correction_bias"
# Each composite layer scale is one the release does not store.
_LAYER_SCALE = "lambda1"
# Peak host RSS of one rank's lazy EP load of the cut model: its experts' slices, not the release's 249 GB.
LOAD_RSS_GIB = 40.0
# Whole nats apart under a lost expert slice; rounding and the routing it flips part them by far less.
EP_LOSS_ABS_MAX = 1e-2
LOSS_TOKENS = 1024
# Fraction of the alone logits' largest magnitude in fp32, and the positions allowed past it: tokens a near-tie still
# routes to another expert.
ISOLATED_REL_MAX = 2e-2
ROUTING_FLIP_MAX_FRACTION = 2e-2
EXPORT_TEXT_ROWS = 2
EXPORT_TEXT_LENGTH = 2048
EXPORT_NLL_ABS_MAX = 2e-2
# Random weights score ln(131072) = 11.8 on any text; the release reads prose near 1.
TRAINED_TEXT_LOSS_MAX = 4.0
# Mean |logit difference|: without its marker a Mamba layer alone moves by whole percents of its output, far above what
# the kernels' two row shapes round apart.
UNMARKED_OVER_MARKED_MIN = 3.0


def _release_directory(model: str) -> str | None:
    if os.path.isdir(model):
        return model
    try:
        return snapshot_download(model, local_files_only=True)
    except LocalEntryNotFoundError:
        return None


def _depth_overrides(source: str, layers: int | None) -> dict | None:
    """The release's first ``layers`` layers; upstream reads the count and the list apart."""
    config = text_config(AutoConfig.from_pretrained(source))
    if layers is None or layers >= config.num_hidden_layers:
        return None
    return {"num_hidden_layers": layers, "layers_block_type": config.layers_block_type[:layers]}


def _load(
    source: str,
    parallelism_config: ParallelismConfig,
    layers: int | None,
    *,
    dtype: torch.dtype = torch.bfloat16,
    text_only: bool = True,
):
    """COLLECTIVE. As the training scripts load it, on the attention a run with it unset resolves to."""
    return load_distributed_model(
        source,
        parallelism_config,
        dtype=dtype,
        trust_remote_code=False,
        model_config_overrides=_depth_overrides(source, layers),
        text_only_model=text_only,
    )


def _stored_tensor(source: str, key: str) -> torch.Tensor:
    weights = resolve_checkpoint_weights(source)
    with safe_open(os.path.join(source, weights.weight_map[key]), framework="pt") as reader:
        return reader.get_tensor(key)


def _load_checks(model, source: str, rss_gib: float) -> dict[str, bool]:
    mixers = [module for module in model.modules() if isinstance(module, nemotron_modeling.NemotronHMamba2Mixer)]
    block_types = text_config(model.config).layers_block_type
    return {
        "lazy_route_admitted": is_lazy_loadable(source) is True,
        "every_moe_layer_ep_wrapped": sum(isinstance(m, EPNemotronHMoELayer) for m in model.modules())
        == block_types.count("moe"),
        "load_reads_only_the_cut_layers": rss_gib < LOAD_RSS_GIB,
        "scan_is_the_toolkits": nemotron_modeling.mamba2_chunk_scan is _document_chunk_scan,
        "mixers_take_the_configs_time_step_limit": bool(mixers)
        and all(mixer.time_step_limit == (0.0, math.inf) for mixer in mixers),
        "attention_resolves_to_fa4": effective_attn_implementation(model.config) == "flash_attention_4",
    }


def _bias_checks(model, source: str) -> dict[str, bool]:
    """The tensor routing selects on is the release's, at the precision it stores, or the bf16 cast rounds it past its
    spread."""
    biases = {path: getattr(layer.gate, BIAS_SLOT) for path, layer in named_ep_layers(model)}
    prefix = hub_text_prefix(resolve_checkpoint_weights(source).weight_map)
    stored = {
        path: _stored_tensor(source, f"{prefix}backbone.{path.removeprefix('model.')}.gate.{BIAS_SLOT}")
        for path in biases
    }
    return {
        "bias_in_fp32": all(bias.dtype == torch.float32 for bias in biases.values()),
        "bias_is_the_release_s": all(torch.equal(biases[path].cpu(), stored[path]) for path in biases),
        "bias_finer_than_bf16": any(not torch.equal(bias, bias.bfloat16().float()) for bias in biases.values()),
    }


def _text_rows(tokenizer, rows: int, length: int, device=None) -> torch.Tensor:
    return torch.tensor(repository_text_rows(tokenizer, rows, length), device=device)


def _loss(model, input_ids: torch.Tensor) -> float:
    with torch.no_grad():
        return model(input_ids=input_ids, labels=input_ids, use_cache=False).loss.item()


def _logits(model, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
    with torch.no_grad():
        return model(input_ids=input_ids, use_cache=False, **kwargs).logits[0].float()


def _beyond(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Fraction of logit rows whose largest difference exceeds the bar of the reference's largest magnitude."""
    return ((actual - expected).abs().amax(-1) > ISOLATED_REL_MAX * expected.abs().max()).float().mean().item()


def _mean_gap(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (actual - expected).abs().mean().item()


def _packing_checks(model, input_ids: torch.Tensor) -> tuple[dict[str, bool], dict[str, float]]:
    """Each document of a packed row against itself alone, and the negative arm: without ``seq_idx`` the Mamba layers
    carry each document into the next."""
    length = input_ids.shape[1]
    lengths = [length // 6, length - 2 * (length // 6), length // 6]
    positions = torch.cat([torch.arange(document) for document in lengths]).unsqueeze(0).to(input_ids.device)
    marked = _logits(
        model, input_ids, position_ids=positions, **segment_marker_kwargs(positions, SegmentMarkers(seq_idx=True))
    )
    crossing = _logits(model, input_ids, position_ids=positions)
    isolated, marked_gaps, crossed_gaps, start = [], [], [], 0
    for document in lengths:
        span = slice(start, start + document)
        alone = _logits(model, input_ids[:, span])
        isolated.append(_beyond(marked[span], alone))
        marked_gaps.append(_mean_gap(marked[span], alone))
        crossed_gaps.append(_mean_gap(crossing[span], alone))
        start += document
    log(f"  packed {lengths}: positions past the bar {isolated}, mean gap {marked_gaps}, no seq_idx {crossed_gaps}")
    # The first document has nothing before it to leak from.
    return {
        "packed_documents_compute_their_logits_alone": max(isolated) <= ROUTING_FLIP_MAX_FRACTION,
        "packed_row_without_seq_idx_leaks_past_the_first_document": all(
            crossed >= UNMARKED_OVER_MARKED_MIN * gap
            for crossed, gap in zip(crossed_gaps[1:], marked_gaps[1:], strict=True)
        ),
    }, {
        "packed_positions_beyond": max(isolated),
        "packed_mean_gap": max(marked_gaps),
        "unmarked_mean_gap": min(crossed_gaps[1:]),
    }


def run_layers(ctx, source: str, args) -> dict:
    reset_host_peak_rss()
    model, tokenizer = _load(source, ParallelismConfig(ep_size=ctx.world_size), args.layers)
    rss_gib = host_peak_rss_gib()
    log(f"  lazy EP load of {args.layers} layers: peak host RSS {rss_gib:.1f} GiB on rank {ctx.rank}")
    checks = _load_checks(model.eval(), source, rss_gib) | _bias_checks(model, source)
    metrics = {"load_peak_host_rss_gib": rss_gib}
    batch = _text_rows(tokenizer, 1, LOSS_TOKENS, ctx.device)
    ep_loss = _loss(model, batch)
    del model
    cleanup_memory()

    reference, _ = _load(source, ParallelismConfig(ep_size=1), args.layers)
    reference_loss = _loss(reference.eval(), batch)
    log(f"  loss on real text: EP {ep_loss:.6f}, one rank {reference_loss:.6f}")
    checks["ep_loss_matches_one_rank"] = abs(ep_loss - reference_loss) <= EP_LOSS_ABS_MAX
    metrics |= {"ep_loss": ep_loss, "one_rank_loss": reference_loss}
    del reference
    cleanup_memory()

    # Every rank holds the collective load. In fp32, as bf16 rounding flips top-22 near-ties past any logits bar.
    reference, _ = _load(source, ParallelismConfig(ep_size=1), args.layers, dtype=torch.float32)
    packing, packing_metrics = _packing_checks(reference.eval(), _text_rows(tokenizer, 1, args.seq, ctx.device))
    return {"checks": checks | packing, "metrics": metrics | packing_metrics}


def _identity_checks(export: str, source: str, layers: int | None, text_only: bool) -> dict[str, bool]:
    """Rank 0, streaming one tensor at a time. Every tensor is the release's under its own name and dtype."""
    exported, released = (resolve_checkpoint_weights(path).weight_map for path in (export, source))
    level = hub_text_prefix(released) if text_only else ""
    hub_names = {f"{level}{key}": key for key in exported if _LAYER_SCALE not in key}
    missing = expected_hub_keys(set(released), layers, text_only) - hub_names.keys()
    unknown = hub_names.keys() - released.keys()
    differing = []
    for hub_key in hub_names.keys() - unknown:
        key = hub_names[hub_key]
        with (
            safe_open(os.path.join(export, exported[key]), framework="pt") as written,
            safe_open(os.path.join(source, released[hub_key]), framework="pt") as release,
        ):
            ours, theirs = written.get_tensor(key), release.get_tensor(hub_key)
        if ours.dtype != theirs.dtype or not torch.equal(ours, theirs):
            differing.append(key)
    log(f"  identity: {len(missing)} keys missing, {len(unknown)} the release lacks, {len(differing)} tensors differ")
    return {
        "export_stores_every_release_key_of_its_route": not missing,
        "export_keys_are_the_release_s": not unknown,
        "export_tensors_are_the_release_s": not differing,
    }


def _config_checks(export: str, source: str, layers: int | None, text_only: bool) -> dict[str, bool]:
    """A full-depth export carries the release's config; the trainer's ``use_cache`` write is every export's. A text-only
    export of a composite keeps transformers' schema for the text class, while a text-only release's own schema, which
    SGLang needs, carries through a cut too, but for the cut's layer list."""
    with open(os.path.join(export, "config.json")) as handle:
        written = json.load(handle)
    with open(os.path.join(source, "config.json")) as handle:
        release = json.load(handle)
    composite = bool(hub_text_prefix(resolve_checkpoint_weights(source).weight_map))
    if composite and text_only:
        return {
            "text_export_is_the_text_class": written["model_type"] == "nemotron_h" and "auto_map" not in written,
        }
    for payload in (written, release):
        payload.pop("use_cache", None)
        payload.pop("transformers_version", None)
    if composite:
        checks = {"composite_export_carries_the_remote_modules": "auto_map" in written}
        if layers is None:
            checks["composite_config_is_the_release_s"] = written == release
        return checks
    if layers is not None:
        written.pop("layers_block_type")
        release.pop("layers_block_type")
    return {"text_release_export_is_the_release_config": written == release}


def _token_scores(model, rows: torch.Tensor) -> torch.Tensor:
    """Fp32 log-probs per next token, one row per forward."""
    scores = []
    for row in rows:
        logits = _logits(model, row[None])
        scores.append(torch.log_softmax(logits[:-1], dim=-1).gather(-1, row[1:, None])[:, 0].cpu())
    return torch.stack(scores)


def _reload_checks(export: str, rows: torch.Tensor, trained: torch.Tensor, text_only: bool) -> dict[str, bool]:
    """Rank 0. A CPU load of the export scores real text at the trained model's mean NLL. In this process, so under the
    family patches: a stock process keeps upstream's time-step floor, which the docs name."""
    model_class = AutoModelForCausalLM if text_only else AutoModelForImageTextToText
    reloaded, info = model_class.from_pretrained(export, dtype=torch.bfloat16, output_loading_info=True)
    problems = loading_problems(info)
    gap = (trained.mean() - _token_scores(reloaded.eval(), rows.cpu()).mean()).abs().item()
    log(f"  CPU reload: problems {problems or 'none'}, |mean NLL gap| {gap:.5f}")
    return {"reload_loads_every_weight": not problems, "reload_scores_at_the_trained_nll": gap <= EXPORT_NLL_ABS_MAX}


def run_export(ctx, source: str, args) -> dict:
    parallelism_config = ParallelismConfig(ep_size=ctx.world_size)
    text_only = args.route == "text"
    model, tokenizer = _load(source, parallelism_config, args.layers, text_only=text_only)
    rows = _text_rows(tokenizer, EXPORT_TEXT_ROWS, EXPORT_TEXT_LENGTH, ctx.device)
    trained = _token_scores(model.eval(), rows)
    export = args.export_dir or os.path.join(shared_output_dir(ctx), "export")
    save_gathered_ep(ctx, model, parallelism_config, tokenizer, export)
    del model
    cleanup_memory()
    checks = {}
    if ctx.rank == 0:
        checks = _identity_checks(export, source, args.layers, text_only)
        checks |= _config_checks(export, source, args.layers, text_only)
        checks |= _reload_checks(export, rows, trained, text_only)
    return {"checks": ctx.broadcast_checks(checks), "metrics": {}}


def run_resume(ctx, source: str, args) -> dict:
    return run_release_resume(ctx, lambda parallelism_config: _load(source, parallelism_config, args.layers))


class _BlockClock:
    """Seconds per block kind over one reentrant step, off a pre-hook on every block: a forward's interval ends at the
    next block's start, and a recompute runs its block's own graph back before the next recompute starts. Each stamp
    syncs, so only the shares are read off this step, never its total."""

    def __init__(self, blocks, final_norm):
        self.seconds: dict[str, float] = {}
        self.block_starts = 0
        self._open: tuple[str, float] | None = None
        self._handles = [block.register_forward_pre_hook(self._block_hook(block.block_type)) for block in blocks]
        self._handles.append(final_norm.register_forward_pre_hook(lambda _module, _args: self.stamp("head")))

    def _block_hook(self, kind: str):
        def hook(_module, _args) -> None:
            self.block_starts += 1
            self.stamp(kind)

        return hook

    def stamp(self, kind: str | None) -> None:
        torch.cuda.synchronize()
        now = time.perf_counter()
        if self._open is not None:
            self.seconds[self._open[0]] = self.seconds.get(self._open[0], 0.0) + now - self._open[1]
        self._open = None if kind is None else (kind, now)

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()


def _loss_at_upstream_floor(model, ids: torch.Tensor) -> float:
    """The clamp upstream's trainer applies and no engine does, which moves one layer's output by whole percents."""
    mixers = [module for module in model.modules() if isinstance(module, nemotron_modeling.NemotronHMamba2Mixer)]
    limits = [mixer.time_step_limit for mixer in mixers]
    for mixer in mixers:
        mixer.time_step_limit = (text_config(model.config).time_step_min, math.inf)
    try:
        return _loss(model, ids)
    finally:
        for mixer, limit in zip(mixers, limits, strict=True):
            mixer.time_step_limit = limit


def run_profile(ctx, source: str, args) -> dict:
    """The whole tower at ``--seq`` tokens per rank: real text, reentrant checkpointing, the plain head, no optimizer
    and no FSDP2 reduce."""
    model, tokenizer = _load(source, ParallelismConfig(ep_size=ctx.world_size), None)
    model.train()
    enable_scoped_gradient_checkpointing(model, gradient_checkpointing_kwargs={"use_reentrant": True})
    # One row per rank, as a training step holds, so each rank dispatches its own tokens.
    ids = _text_rows(tokenizer, ctx.world_size, args.seq, ctx.device)[ctx.rank : ctx.rank + 1]
    if args.frozen_experts:
        for _, layer in named_ep_layers(model):
            for _, parameter in layer.expert_named_params():
                parameter.requires_grad_(False)

    def step() -> float:
        loss = model(input_ids=ids, labels=ids, use_cache=False).loss
        loss.backward()
        model.zero_grad(set_to_none=True)
        return loss.item()

    for _ in range(2):
        step()
    torch.cuda.empty_cache()
    weights = torch.cuda.memory_allocated()
    grads = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    torch.cuda.reset_peak_memory_stats()
    times, losses = [], []
    for _ in range(args.iters):
        ctx.barrier()
        torch.cuda.synchronize()
        start = time.perf_counter()
        losses.append(step())
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    step_s = statistics.median(times)
    peak, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()

    blocks = [module for module in model.modules() if isinstance(module, nemotron_modeling.NemotronHBlock)]
    clock = _BlockClock(blocks, model.model.norm_f)
    loss = model(input_ids=ids, labels=ids, use_cache=False).loss
    # The head's backward runs ahead of the first recompute.
    clock.stamp("head")
    loss.backward()
    clock.stamp(None)
    clock.remove()
    model.zero_grad(set_to_none=True)
    # In eval, so no checkpoint scope opens for a backward that never comes.
    model.eval()
    limit_loss, floor_loss = _loss(model, ids), _loss_at_upstream_floor(model, ids)
    total = sum(clock.seconds.values())
    shares = {f"share_{kind}": seconds / total for kind, seconds in clock.seconds.items()}
    metrics = {
        "tokens_per_rank": args.seq,
        "step_s": step_s,
        "tokens_per_second_per_rank": args.seq / step_s,
        "weights_gib": weights / 2**30,
        "grads_gib": grads / 2**30,
        "peak_allocated_gib": peak / 2**30,
        "peak_reserved_gib": reserved / 2**30,
        "activations_mb_per_token": (peak - weights - grads) / args.seq / 2**20,
        "loss": losses[-1],
        "loss_at_config_time_step_limit": limit_loss,
        "loss_at_upstream_time_step_floor": floor_loss,
        **shares,
    }
    log(f"  profile: {metrics}")
    return {
        "checks": {
            "real_text_scores_as_trained_weights": losses[-1] <= TRAINED_TEXT_LOSS_MAX,
            "config_time_step_limit_scores_no_worse": limit_loss <= floor_loss + EP_LOSS_ABS_MAX,
            "every_block_ran_and_recomputed": clock.block_starts == 2 * len(blocks),
            "every_kind_timed": {"linear_attention", "moe", "full_attention", "head"} == clock.seconds.keys(),
        },
        "metrics": metrics,
    }


MODES = {"layers": run_layers, "export": run_export, "resume": run_resume, "profile": run_profile}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=sorted(MODES), required=True)
    parser.add_argument("--release", choices=sorted(NEMOTRON_RELEASES), default="super")
    parser.add_argument(
        "--layers",
        type=int,
        default=9,
        help="layers --mode layers/export/resume keep; 88 is all of Super, 52 Lightning",
    )
    parser.add_argument("--seq", type=int, default=4096, help="packed row width of layers, tokens per rank of profile")
    parser.add_argument("--route", choices=("text", "composite"), default="text", help="the class --mode export loads")
    parser.add_argument("--export-dir", default=None, help="where --mode export writes and keeps its save")
    parser.add_argument("--iters", type=int, default=5, help="timed steps of --mode profile")
    parser.add_argument("--frozen-experts", action="store_true", help="--mode profile with the experts frozen")
    return parser.parse_args()


@gpu_test_main(exact_world_size=8, prefix="nemotron_h_real_checkpoint")
def run(ctx) -> dict:
    args = _arguments()
    model = release_model_id(args.release)
    source = _release_directory(model)
    if args.route == "composite" and not hub_text_prefix(resolve_checkpoint_weights(source).weight_map):
        raise ValueError(f"{model} is a text-only release, so it has no composite route")
    if args.layers >= text_config(AutoConfig.from_pretrained(source)).num_hidden_layers:
        args.layers = None
    log(f"=== {model} at {source}, --mode {args.mode}, --layers {args.layers}")
    return MODES[args.mode](ctx, source, args)


if __name__ == "__main__":
    skip_unless_blackwell("Nemotron-H's release row")
    _release = _arguments().release
    _model = release_model_id(_release)
    if _release_directory(_model) is None:
        log(
            f"SKIP: {_model} is not in the local Hub cache (set {NEMOTRON_RELEASES[_release][1]} to a present checkpoint)"
        )
        sys.exit(0)
    run()
