"""Checkpoint artifact harness: what every saver writes, compared between two commits (``make verify-artifacts
BASE=<sha>``) into ``evidence/checkpoint_artifacts-<head sha>.json``. Each tree saves a tiny model of every family
through every saver mode one CPU process runs, at fp32 and bf16, as an export and as a training checkpoint, and
the outputs are compared file by file: file sets and permission bits, safetensors headers and tensor digests,
JSON by value, torch files by key and tensor, any other file by its bytes.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import stat
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from accelerate import PartialState
from peft import LoraConfig, get_peft_model
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor

from halo.checkpoint.parallel.adapters import save_resume_adapter
from halo.checkpoint.parallel.optimizer import OptimizerShardStore
from halo.checkpoint.parallel.save import save_checkpoint
from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.distributed.fsdp import IdentityParamSet, apply_fsdp2_per_layer, mixed_precision_policy
from halo.model_loading.dtype import cast_parameters_to_run_dtype
from halo.models.moe.expert_lora_spec import ExpertLoraSpec
from tests.common.adapters import randomize_adapters
from tests.common.checkpoint_io import save_context, shard_store_context
from tests.common.cp_model import unpatched_cp_wrapper
from tests.common.equivalence.revisions import (
    RAISED,
    common_families,
    compare_outputs,
    digest,
    export_tree,
    quiet_logs,
    raised_description,
    resolve_commit,
    run_in_tree,
    split_accepted,
    write_evidence,
)
from tests.common.parallelism import make_parallelism_config, single_process_ep_config
from tests.common.repo import REPO_ROOT
from tests.common.tiny_models import (
    TINY_DENSE_FAMILIES,
    TINY_MOE_FAMILIES,
    TINY_MOE_VLM_FAMILIES,
    TinyFamily,
    tiny_family_model,
    word_level_tokenizer,
)

FAMILIES: dict[str, TinyFamily] = {**TINY_DENSE_FAMILIES, **TINY_MOE_FAMILIES, **TINY_MOE_VLM_FAMILIES}
RUN_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
# The modes whose training checkpoint carries the per-rank optimizer shards.
FULL_WEIGHT_MODES = ("fsdp2", "cp", "cp_fsdp2", "tp", "ep1", "ep")
CP_MODES = ("cp", "cp_fsdp2")
EXPERT_LORA_MODES = ("ep_expert_lora", "ep_expert_lora_merged")
# Written in place of a case's files when its save raises: the exception's type and message.
RAISED_FILE = f"{RAISED}.json"
# Reseeded before the adapters are drawn, so their values do not hang on how much randomness the model
# build and the EP patch consumed.
ADAPTER_SEED = 1
# Both trees' producers run under one string hash seed: a set's iteration order follows it, and reaches
# adapter_config.json's target_modules and the tensor names of a safetensors refusal.
PRODUCER_ENV = {"PYTHONHASHSEED": "0"}


@dataclass(frozen=True)
class ArtifactCase:
    """One save: a tiny family's model at one run dtype through one saver mode, as an export or a training
    checkpoint."""

    family: str
    mode: str
    dtype: str
    training: bool

    @property
    def path(self) -> str:
        return f"{self.family}/{self.mode}/{self.dtype}/{'training' if self.training else 'export'}"


def _modes(name: str, family: TinyFamily) -> tuple[str, ...]:
    cp = CP_MODES if family.ulysses_cp else ()
    if name in TINY_DENSE_FAMILIES:
        return ("fsdp2", *cp, "tp", "peft")
    return ("fsdp2", *cp, "ep1", "ep", *EXPERT_LORA_MODES)


ARTIFACT_CASES = tuple(
    ArtifactCase(name, mode, dtype, training)
    for name, family in FAMILIES.items()
    for mode in _modes(name, family)
    for dtype in RUN_DTYPES
    for training in (False, True)
)


def _context(model, parallelism_config=None, *, training: bool, **fields):
    return save_context(
        model,
        parallelism_config,
        is_save_rank=True,
        tokenizer=word_level_tokenizer(),
        training_checkpoint=training,
        **fields,
    )


def _shard(model, mesh: DeviceMesh, dtype: torch.dtype, ignored: Iterable = ()) -> None:
    """FSDP2 per layer under the run's mixed-precision policy, as a run's DP wrap applies it."""
    policy = mixed_precision_policy(dtype, fp32_reduce=False, cast_forward_inputs=True)
    apply_fsdp2_per_layer(model, mesh, policy, False, IdentityParamSet(ignored))


def _save_optimizer_shards(model, parallelism_config, directory: Path) -> None:
    """One AdamW step on a fixed gradient per parameter, then the shard store's save."""
    params = [param for param in model.parameters() if param.requires_grad]
    for index, param in enumerate(params):
        param.grad = param.detach() * (index + 1) / len(params)
    optimizer = torch.optim.AdamW(params, lr=1e-2, foreach=False)
    optimizer.step()
    OptimizerShardStore(shard_store_context(model, optimizer, parallelism_config)).save(str(directory))


def _save_case(case: ArtifactCase, directory: Path, mesh: DeviceMesh) -> None:
    family, dtype = FAMILIES[case.family], RUN_DTYPES[case.dtype]
    torch.manual_seed(0)
    model = tiny_family_model(family)
    cast_parameters_to_run_dtype(model, dtype)
    training = case.training
    if case.mode == "peft":
        targets = list(family.attention_targets or ("q_proj", "v_proj"))
        model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=targets))
        torch.manual_seed(ADAPTER_SEED)
        randomize_adapters(model)
        _shard(model, mesh, dtype)
        save_checkpoint(_context(model, training=training, toolkit_fsdp2=True), str(directory))
        return
    if case.mode in ("fsdp2", "cp_fsdp2"):
        _shard(model, mesh, dtype)
    config = None
    if case.mode == "fsdp2":
        flags = {"toolkit_fsdp2": True}
    elif case.mode in CP_MODES:
        config = make_parallelism_config(world_size=2, gpus_per_node=2, cp_size=2)
        flags = {"cp_wrapper": unpatched_cp_wrapper(model)}
    elif case.mode == "tp":
        config = make_parallelism_config(world_size=2, gpus_per_node=2, tp_size=2)
        flags = {}
    else:
        expert_lora = ExpertLoraSpec(r=8, alpha=16.0) if case.mode in EXPERT_LORA_MODES else None
        ep_config = single_process_ep_config(expert_lora=expert_lora, fp32_router=dtype != torch.float32)
        apply_ep_layers(model, ep_config)
        if expert_lora is not None:
            torch.manual_seed(ADAPTER_SEED)
            randomize_adapters(model)
        ep_params = [param for _, layer in named_ep_layers(model) for param in layer.parameters()]
        _shard(model, mesh, dtype, ignored=() if case.mode == "ep1" else ep_params)
        config = make_parallelism_config(
            world_size=1,
            gpus_per_node=1,
            expert_lora=expert_lora,
            merge_expert_lora_on_save=case.mode == "ep_expert_lora_merged",
        )
        flags = {"has_ep_layers": True, "has_expert_lora": expert_lora is not None}
    ctx = _context(model, config, training=training, **flags)
    if not save_checkpoint(ctx, str(directory)):
        raise RuntimeError(f"no saver owns the {case.mode} save")
    if training and case.mode == "ep_expert_lora_merged":
        save_resume_adapter(ctx, str(directory))
    if training and case.mode in FULL_WEIGHT_MODES:
        # The shards record the one-process topology they are written on, whichever mode the save took.
        _save_optimizer_shards(model, make_parallelism_config(world_size=1, gpus_per_node=1), directory)


@contextlib.contextmanager
def _single_rank_group(scratch: Path) -> Iterator[DeviceMesh]:
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{scratch / 'process_group'}")
    try:
        PartialState()
        yield init_device_mesh("cpu", (1,))
    finally:
        dist.destroy_process_group()


def produce(output: Path, cases: Iterable[ArtifactCase] = ARTIFACT_CASES) -> None:
    """Save every case under ``output/<case path>``, on a single-rank gloo group this process holds for the
    duration; a case that raises writes :data:`RAISED_FILE` there instead."""
    output.mkdir(parents=True, exist_ok=True)
    with quiet_logs(), tempfile.TemporaryDirectory() as scratch, _single_rank_group(Path(scratch)) as mesh:
        for case in cases:
            directory = output / case.path
            try:
                _save_case(case, directory, mesh)
            except Exception as error:
                shutil.rmtree(directory, ignore_errors=True)
                directory.mkdir(parents=True)
                raised = json.dumps(raised_description(error, output), indent=1)
                (directory / RAISED_FILE).write_text(raised + "\n", encoding="utf-8")


def _safetensors(path: Path) -> dict:
    """A safetensors file's metadata and each tensor's dtype, shape and byte digest, read off its header."""
    data = path.read_bytes()
    header_size = int.from_bytes(data[:8], "little")
    header = json.loads(data[8 : 8 + header_size])
    body = data[8 + header_size :]
    metadata = header.pop("__metadata__", {})
    tensors = {}
    for name, info in sorted(header.items()):
        begin, end = info["data_offsets"]
        tensors[name] = [info["dtype"], info["shape"], digest(body[begin:end])]
    return {"metadata": metadata, "tensors": tensors}


def _leaf(value) -> object:
    if isinstance(value, DTensor):
        return [f"DTensor{list(value.placements)}", *_leaf(value.to_local())]
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return [str(tensor.dtype), list(tensor.shape), digest(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())]
    return value if value is None or isinstance(value, (bool, int, float, str)) else repr(value)


def _flatten(value, prefix: str = "") -> dict[str, object]:
    """A torch file's content as ``key path -> leaf``, a tensor as its dtype, shape and byte digest."""
    if isinstance(value, Mapping):
        items = [(str(key), item) for key, item in value.items()]
    elif isinstance(value, (list, tuple)):
        items = [(str(index), item) for index, item in enumerate(value)]
    else:
        return {prefix: _leaf(value)}
    if not items:
        return {prefix: type(value).__name__}
    flat = {}
    for key, item in items:
        flat |= _flatten(item, f"{prefix}/{key}" if prefix else key)
    return flat


def _content(path: Path) -> object:
    if path.suffix == ".safetensors":
        return _safetensors(path)
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix in (".pt", ".bin"):
        return _flatten(torch.load(path, map_location="cpu", weights_only=True))
    return {"sha256": digest(path.read_bytes())}


def describe(root: Path) -> dict[str, dict]:
    """Each file under ``root``, by relative path, as the comparison reads it: its permission bits (octal) and
    its content."""
    return {
        str(path.relative_to(root)): {"mode": f"{stat.S_IMODE(path.stat().st_mode):o}", "content": _content(path)}
        for path in sorted(path for path in root.rglob("*") if path.is_file())
    }


def raised_cases(description: Mapping[str, dict]) -> dict[str, object]:
    """The cases of a described output that raised, by case path, with what they raised."""
    return {
        path.rpartition("/")[0]: described["content"]
        for path, described in description.items()
        if path.rpartition("/")[2] == RAISED_FILE
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare what every saver writes between two commits.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--base-ref", help="the commit the change is compared with")
    mode.add_argument("--produce", type=Path, help="write this tree's artifacts under the directory")
    parser.add_argument("--head-ref", default="HEAD", help="the commit under test (default: HEAD)")
    parser.add_argument("--families", nargs="+", choices=list(FAMILIES), help="save only these families' cases")
    parser.add_argument(
        "--accept", nargs="+", default=[], metavar="CASE", help="cases whose deltas a flagged change declares"
    )
    parser.add_argument("--image", help="the image the run used, recorded in the evidence")
    args = parser.parse_args(argv)
    families = args.families or list(FAMILIES)
    if args.produce is not None:
        produce(args.produce, [case for case in ARTIFACT_CASES if case.family in families])
        return 0

    base, head = resolve_commit(REPO_ROOT, args.base_ref), resolve_commit(REPO_ROOT, args.head_ref)
    with tempfile.TemporaryDirectory() as scratch:
        trees = {
            label: export_tree(REPO_ROOT, commit, Path(scratch, label))
            for label, commit in (("base", base), ("head", head))
        }
        families, skipped = common_families(trees, __spec__.name, families)
        outputs = {}
        for label, tree in trees.items():
            outputs[label] = Path(scratch, f"{label}-artifacts")
            run_in_tree(
                tree, __spec__.name, "--produce", str(outputs[label]), "--families", *families, env=PRODUCER_ENV
            )
        before, after = describe(outputs["base"]), describe(outputs["head"])
    deltas, accepted = split_accepted(compare_outputs(before, after), args.accept)
    raised = raised_cases(after)
    verdict = "fail" if deltas or raised else "pass"
    findings = {
        "families": families,
        "skipped": skipped,
        "cases": sum(case.family in families for case in ARTIFACT_CASES),
        "files": len(after),
        "raised": raised,
        "accepted": accepted,
        "deltas": deltas,
    }
    path = write_evidence(
        "checkpoint_artifacts", base=base, head=head, image=args.image, verdict=verdict, findings=findings
    )
    print(  # noqa: T201 — the CLI's report
        f"checkpoint artifacts: {verdict}, {findings['files']} files, {len(deltas)} deltas, {len(accepted)} "
        f"accepted, {len(raised)} cases raising, skipped as absent from a tree's roster: {skipped or 'none'} ({path})"
    )
    return 0 if verdict == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
