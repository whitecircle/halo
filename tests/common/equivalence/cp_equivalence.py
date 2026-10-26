"""CP equivalence harness: Ulysses attention and the CP model wrapper on one tiny model per Ulysses wrapper class,
compared between two commits (``make verify-cp BASE=<sha>``) into ``evidence/cp_equivalence-<head sha>.json``.
The base tree runs twice: a record it repeats is compared bit for bit, one it does not with each gradient inside
the base's own spread. The GPU tier runs bf16 with CP's own flash kernel made deterministic; the CPU tier runs
fp32 on gloo with a plain-torch attention under the flash contract.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
from accelerate import PartialState
from transformers.trainer_utils import enable_full_determinism

from src.distributed.context_parallel import base_layer as cp_attention
from src.distributed.context_parallel.base_layer import UlyssesAttentionBase
from src.distributed.context_parallel.config import split_sequence_for_cp
from src.distributed.context_parallel.layers.registry import WRAPPER_CLASS_MAP
from src.distributed.context_parallel.wrapper import UlyssesCPModelWrapper
from src.distributed.expert_parallel.patching import patch_moe_model_for_ep
from src.distributed.parallelism_config import ParallelismConfig
from src.models.attention.gpt_oss_sinks import SinksPolicy, apply_sinks_policy
from tests.common.distributed import init_distributed, teardown_distributed
from tests.common.ep_reference import random_token_batch
from tests.common.equivalence.revisions import (
    RAISED,
    compare_outputs,
    export_tree,
    quiet_logs,
    resolve_commit,
    run_in_tree,
    split_accepted,
    unseen_controls,
    write_evidence,
)
from tests.common.equivalence.tensor_records import (
    GRADIENT_PREFIX,
    Orders,
    Tensors,
    load_record,
    max_abs_difference,
    raised_records,
    record_case,
    record_deltas,
    record_paths,
)
from tests.common.models import TINY_MISTRAL4_CONFIG, TINY_QWEN35_MOE_CONFIG
from tests.common.parallelism import single_process_ep_config
from tests.common.repo import REPO_ROOT
from tests.common.tiny_models import TINY_DENSE_FAMILY, TINY_MOE_FAMILIES, TinyFamily, tiny_family_model

CP_SIZE = 2
SEED = 1234
BATCH = 2
# Past the tiny sliding windows (64), so the windowed layers mask.
SEQ = 128
LEVELS = ("attention", "model")
DTYPE = {"cpu": torch.float32, "gpu": torch.bfloat16}
DEFAULT_ATTN_IMPLEMENTATION = "flash_attention_2"
# Where Mistral4's llama-4 query scale steps up: inside SEQ, so the scale varies along the sequence (it is 1
# at every position below the tiny config's own 256).
MISTRAL4_SCALE_PERIOD = 32


@dataclass(frozen=True)
class CPFamily:
    """A tiny model one Ulysses wrapper class runs on: its builder, the text-config fields that make every layer
    one the wrapper takes, and whether its MoE blocks take the EP wrappers."""

    tiny: TinyFamily
    overrides: dict = field(default_factory=dict)
    moe: bool = True


FAMILIES: dict[str, CPFamily] = {
    "qwen3": CPFamily(TINY_DENSE_FAMILY, moe=False),
    "gpt_oss": CPFamily(TINY_MOE_FAMILIES["gpt_oss"]),
    "bailing_moe": CPFamily(TINY_MOE_FAMILIES["bailing_moe"]),
    "cohere2_moe": CPFamily(TINY_MOE_FAMILIES["cohere2_moe"]),
    "glm4_moe_lite": CPFamily(TINY_MOE_FAMILIES["glm4_moe_lite"]),
    "mistral4": CPFamily(
        TINY_MOE_FAMILIES["mistral4"],
        {
            "rope_parameters": {
                **TINY_MISTRAL4_CONFIG["rope_parameters"],
                "original_max_position_embeddings": MISTRAL4_SCALE_PERIOD,
            }
        },
    ),
    # Full attention in every layer: the gated DeltaNet layers run a sequence-axis scan Ulysses refuses.
    "qwen3_5_moe_text": CPFamily(
        TINY_MOE_FAMILIES["qwen3_5_moe_text"],
        {"layer_types": ["full_attention"] * TINY_QWEN35_MOE_CONFIG["num_hidden_layers"]},
    ),
}


class _Captured(Exception):
    """Ends a forward once the first attention layer's inputs are in hand."""


def reference_flash_attn(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1)):
    """``flash_attn_func``'s contract in plain torch, for the keywords CP passes: ``[B, S, H, D]`` in and out,
    native GQA, and the causal mask and sliding window aligned to the bottom right of the score matrix."""
    if dropout_p:
        raise ValueError(f"the reference attention runs no dropout, got dropout_p={dropout_p}")
    groups = q.shape[2] // k.shape[2]
    k, v = k.repeat_interleave(groups, dim=2), v.repeat_interleave(groups, dim=2)
    scale = softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * scale
    rows = torch.arange(q.shape[1], device=q.device)[:, None] + (k.shape[1] - q.shape[1])
    cols = torch.arange(k.shape[1], device=q.device)[None, :]
    allowed = torch.ones(q.shape[1], k.shape[1], dtype=torch.bool, device=q.device)
    left, right = window_size
    if causal:
        allowed &= cols <= rows
    if left >= 0:
        allowed &= cols >= rows - left
    if right >= 0:
        allowed &= cols <= rows + right
    probs = scores.masked_fill(~allowed, float("-inf")).softmax(dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", probs, v)


def gloo_all_to_all(output_tensor_list, input_tensor_list, group=None, async_op=False):
    """``dist.all_to_all`` over ``dist.all_to_all_single``, the form the gloo backend implements."""
    if async_op:
        raise ValueError("the gloo all_to_all stands in for synchronous calls only")
    input_sizes = [tensor.numel() for tensor in input_tensor_list]
    output_sizes = [tensor.numel() for tensor in output_tensor_list]
    flat_input = torch.cat([tensor.reshape(-1) for tensor in input_tensor_list])
    flat_output = flat_input.new_empty(sum(output_sizes))
    dist.all_to_all_single(flat_output, flat_input, output_sizes, input_sizes, group=group)
    for output, chunk in zip(output_tensor_list, flat_output.split(output_sizes), strict=True):
        output.copy_(chunk.view_as(output))


def deterministic_kernels(probe):
    """``get_flash_attn_func`` over ``probe``: the kernel ``probe`` picks for the family, called with
    ``deterministic=True``, since CP passes no such flag and FA2, FA3 and FA4 all default to an atomic dQ."""

    def get_flash_attn_func(allow_fa4: bool = True):
        return functools.partial(probe(allow_fa4), deterministic=True)

    return get_flash_attn_func


@contextlib.contextmanager
def tier_controls(tier: str) -> Iterator[None]:
    """The kernel and transport CP runs on in ``tier``: on GPU its own flash kernel made deterministic under
    HF's ``full_determinism``; on CPU :func:`reference_flash_attn`, as flash has no CPU build, and
    :func:`gloo_all_to_all`."""
    if tier == "cpu":
        with (
            patch.object(cp_attention, "get_flash_attn_func", lambda allow_fa4=True: reference_flash_attn),
            patch.object(dist, "all_to_all", gloo_all_to_all),
        ):
            yield
        return
    enable_full_determinism(SEED)
    with patch.object(cp_attention, "get_flash_attn_func", deterministic_kernels(cp_attention.get_flash_attn_func)):
        yield


def _tiny_model(name: str, tier: str, device) -> torch.nn.Module:
    """``name``'s seeded tiny model on ``device`` at the tier's dtype, labelled with the attention CP validates,
    its GPT-OSS sinks neutralized as the CP loaders leave them."""
    family = FAMILIES[name]
    torch.manual_seed(SEED)
    model = tiny_family_model(family.tiny, overrides=family.overrides)
    label = family.tiny.cp_attn_implementation or DEFAULT_ATTN_IMPLEMENTATION
    model.config._attn_implementation = label
    apply_sinks_policy(model, model.config, policy=SinksPolicy.NEUTRALIZED, attn_implementation=label)
    return model.to(device=device, dtype=DTYPE[tier])


def _first_attention(module: torch.nn.Module) -> UlyssesAttentionBase:
    return next(child for child in module.modules() if isinstance(child, UlyssesAttentionBase))


def _nudge(attention: UlyssesAttentionBase) -> None:
    """The negative control: the first element of the attention's first weight moved by a few ulps."""
    weight = next(attention.parameters())
    finfo = torch.finfo(weight.dtype)
    with torch.no_grad():
        first = weight.view(-1)[:1]
        first.add_(first.abs().clamp_min(finfo.tiny) * 4 * finfo.eps)


def _decoder_attention_inputs(model: torch.nn.Module, input_ids, position_ids) -> dict:
    """The keyword inputs the decoder hands its first attention layer on ``input_ids``, the forward stopped
    there."""
    attention = next(module for module in model.modules() if type(module).__name__ in WRAPPER_CLASS_MAP)
    captured = {}

    def capture(_module, _args, kwargs):
        captured.update(kwargs)
        raise _Captured

    handle = attention.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        with torch.no_grad(), contextlib.suppress(_Captured):
            model(input_ids=input_ids, position_ids=position_ids)
    finally:
        handle.remove()
    return captured


def _attention_case(name: str, cp_config, *, tier: str, device, perturb: bool) -> tuple[Tensors, Orders]:
    model = _tiny_model(name, tier, device)
    hidden_size = model.get_input_embeddings().embedding_dim
    generator = torch.Generator().manual_seed(SEED)
    vocab_size = model.get_input_embeddings().num_embeddings
    input_ids = torch.randint(0, vocab_size, (BATCH, SEQ), generator=generator).to(device)
    positions = torch.arange(SEQ, device=device).unsqueeze(0).expand(BATCH, -1)
    local_positions = split_sequence_for_cp(positions, cp_config)
    inputs = _decoder_attention_inputs(model, split_sequence_for_cp(input_ids, cp_config), local_positions)
    attention = _first_attention(UlyssesCPModelWrapper(model, cp_config))
    if perturb:
        _nudge(attention)
    hidden = torch.randn(BATCH, SEQ, hidden_size, generator=generator).to(device=device, dtype=DTYPE[tier])
    local_hidden = split_sequence_for_cp(hidden, cp_config).requires_grad_()
    attention.global_position_ids = positions
    output = attention(
        hidden_states=local_hidden,
        position_embeddings=inputs["position_embeddings"],
        position_ids=inputs.get("position_ids"),
    )[0]
    cotangent = torch.randn(BATCH, SEQ, output.shape[-1], generator=generator).to(device=device, dtype=output.dtype)
    (output.float() * split_sequence_for_cp(cotangent, cp_config).float()).sum().backward()
    tensors = {"output": output, f"{GRADIENT_PREFIX}hidden_states": local_hidden.grad}
    named = list(attention.named_parameters())
    tensors |= {f"{GRADIENT_PREFIX}{param_name}": param.grad for param_name, param in named if param.grad is not None}
    return tensors, {"attention_parameters": [param_name for param_name, _ in named]}


def _model_case(name: str, cp_config, *, tier: str, device, perturb: bool) -> tuple[Tensors, Orders]:
    model = _tiny_model(name, tier, device)
    if FAMILIES[name].moe:
        patch_moe_model_for_ep(model, single_process_ep_config(use_grouped_gemm=tier == "gpu"))
    wrapper = UlyssesCPModelWrapper(model, cp_config).train()
    if perturb:
        _nudge(_first_attention(wrapper))
    vocab_size = model.get_input_embeddings().num_embeddings
    input_ids, labels = random_token_batch(vocab_size, BATCH, SEQ, device, seed=SEED)
    outputs = wrapper(input_ids=input_ids, labels=labels)
    outputs.loss.backward()
    named = list(wrapper.named_parameters())
    tensors = {"loss": outputs.loss, "logits": outputs.logits}
    tensors |= {f"{GRADIENT_PREFIX}{param_name}": param.grad for param_name, param in named if param.grad is not None}
    return tensors, {"named_parameters": [param_name for param_name, _ in named]}


CASE_BODIES = {"attention": _attention_case, "model": _model_case}


def record_rank(output: Path, families, levels, *, tier: str, device, perturb: bool = False) -> None:
    """Record this rank's cases under ``output/<family>/<level>``, on the default process group as the CP
    group; the tier's controls are the caller's. Accelerate's process state comes first: the GPT-OSS sinks
    policy logs through it."""
    PartialState()
    cp_config = ParallelismConfig(cp_size=CP_SIZE).cp_config()
    rank = dist.get_rank()
    for name in families:
        for level in levels:
            body = functools.partial(CASE_BODIES[level], name, cp_config, tier=tier, device=device, perturb=perturb)
            record_case(output, f"{name}/{level}", rank, body)


def produce(output: Path, families, levels, *, tier: str, perturb: bool) -> None:
    """:func:`record_rank` as one ``torchrun`` rank: the process group, the tier's controls and the teardown."""
    with quiet_logs():
        if tier == "gpu":
            _, _, local_rank = init_distributed()
            device = torch.device("cuda", local_rank)
        else:
            dist.init_process_group("gloo")
            device = torch.device("cpu")
        try:
            with tier_controls(tier):
                record_rank(output, families, levels, tier=tier, device=device, perturb=perturb)
        finally:
            teardown_distributed()


def _without_gradients(record: Mapping) -> dict:
    tensors = {name: tensor for name, tensor in record["tensors"].items() if not name.startswith(GRADIENT_PREFIX)}
    return {"orders": record["orders"], "tensors": tensors}


def judged_deltas(base: Mapping, repeat: Mapping, head: Mapping) -> list[str]:
    """``head``'s differences from ``base``: bit for bit where ``repeat`` matches ``base``; otherwise everything
    but the gradients bit for bit and each gradient within its ``base``-vs-``repeat`` max-abs envelope (zero
    for one ``repeat`` lacks)."""
    if not record_deltas(base, repeat) or any(RAISED in record for record in (base, repeat, head)):
        return record_deltas(base, head)
    deltas = record_deltas(_without_gradients(base), _without_gradients(head))
    base_tensors, repeat_tensors, head_tensors = base["tensors"], repeat["tensors"], head["tensors"]
    for name in sorted(name for name in base_tensors.keys() | head_tensors.keys() if name.startswith(GRADIENT_PREFIX)):
        if name not in base_tensors or name not in head_tensors:
            deltas.append(f"tensors/{name}: only in {'base' if name in base_tensors else 'head'}")
            continue
        envelope = max_abs_difference(base_tensors[name], repeat_tensors[name]) if name in repeat_tensors else 0.0
        found = max_abs_difference(base_tensors[name], head_tensors[name])
        if not found <= envelope:
            deltas.append(f"tensors/{name}: max |delta| {found:.3e} outside the base envelope {envelope:.3e}")
    return deltas


def judge(base_root: Path, repeat_root: Path, head_root: Path) -> tuple[list[dict], list[str]]:
    """``head``'s records against ``base``'s, each pair by :func:`judged_deltas`, as
    :func:`~tests.common.equivalence.revisions.compare_outputs` lists them, and the records the envelope judged
    (those the base tree's two runs did not repeat). Raises when those two runs recorded different records."""
    base_paths, repeat_paths = record_paths(base_root), record_paths(repeat_root)
    if base_paths.keys() != repeat_paths.keys():
        raise RuntimeError(
            f"the base tree's two runs recorded different records: {sorted(base_paths.keys() ^ repeat_paths.keys())}"
        )
    base_runs = {path: (load_record(base_paths[path]), load_record(repeat_paths[path])) for path in base_paths}
    enveloped = sorted(path for path, (base, repeat) in base_runs.items() if record_deltas(base, repeat))
    deltas = compare_outputs(
        base_runs, record_paths(head_root), lambda runs, head: judged_deltas(*runs, load_record(head))
    )
    return deltas, enveloped


def assess(base: Path, repeat: Path, head: Path, control: Path, accept=()) -> tuple[str, dict]:
    """The verdict on four recorded outputs and its findings: it fails on a :func:`judge` delta outside the cases
    ``accept`` names, on any head raise, and on a control record that matches its base one or raised."""
    compared, enveloped = judge(base, repeat, head)
    deltas, accepted = split_accepted(compared, accept)
    raised = raised_records(head)
    control_records = sorted(record_paths(control))
    unseen = unseen_controls(judge(base, repeat, control)[0], control_records, raised_records(control))
    findings = {
        "enveloped": enveloped,
        "raised": raised,
        "negative_control": {"records": control_records, "unseen": unseen},
        "accepted": accepted,
        "deltas": deltas,
    }
    return "fail" if deltas or raised or unseen else "pass", findings


def _produce_in(tree: Path, output: Path, families, levels, tier: str, *, perturb: bool = False) -> None:
    flags = ["--produce", str(output), "--tier", tier, "--families", *families, "--levels", *levels]
    run_in_tree(tree, __spec__.name, *flags, *(["--perturb"] if perturb else []), nproc=CP_SIZE)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare Ulysses CP forward and backward between two commits.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--base-ref", help="the commit the change is compared with")
    mode.add_argument("--produce", type=Path, help="record this rank's cases under the directory (under torchrun)")
    parser.add_argument("--head-ref", default="HEAD", help="the commit under test (default: HEAD)")
    parser.add_argument("--tier", choices=sorted(DTYPE), default="cpu")
    parser.add_argument("--families", nargs="+", choices=list(FAMILIES), default=list(FAMILIES))
    parser.add_argument("--levels", nargs="+", choices=LEVELS, default=list(LEVELS))
    parser.add_argument(
        "--accept", nargs="+", default=[], metavar="CASE", help="cases whose deltas a flagged change declares"
    )
    parser.add_argument("--perturb", action="store_true", help="move one attention weight (the negative control)")
    parser.add_argument("--image", help="the image the run used, recorded in the evidence")
    args = parser.parse_args(argv)
    families, levels = args.families, args.levels
    if args.produce is not None:
        produce(args.produce, families, levels, tier=args.tier, perturb=args.perturb)
        return 0

    base, head = resolve_commit(REPO_ROOT, args.base_ref), resolve_commit(REPO_ROOT, args.head_ref)
    with tempfile.TemporaryDirectory() as scratch:
        base_tree = export_tree(REPO_ROOT, base, Path(scratch, "base"))
        head_tree = export_tree(REPO_ROOT, head, Path(scratch, "head"))
        outputs = {label: Path(scratch, f"{label}-records") for label in ("base", "repeat", "head", "control")}
        _produce_in(base_tree, outputs["base"], families, levels, args.tier)
        _produce_in(base_tree, outputs["repeat"], families, levels, args.tier)
        _produce_in(head_tree, outputs["head"], families, levels, args.tier)
        _produce_in(head_tree, outputs["control"], families[:1], levels, args.tier, perturb=True)
        verdict, assessed = assess(
            outputs["base"], outputs["repeat"], outputs["head"], outputs["control"], args.accept
        )
    findings = {"tier": args.tier, "families": families, "levels": levels, **assessed}
    path = write_evidence("cp_equivalence", base=base, head=head, image=args.image, verdict=verdict, findings=findings)
    print(  # noqa: T201 — the CLI's report
        f"CP equivalence ({args.tier}): {verdict}, {len(assessed['deltas'])} deltas, {len(assessed['accepted'])} "
        f"accepted, {len(assessed['enveloped'])} records judged by envelope, {len(assessed['raised'])} raising, "
        f"negative control unseen in {len(assessed['negative_control']['unseen'])} ({path})"
    )
    return 0 if verdict == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
