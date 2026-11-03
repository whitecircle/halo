"""Efficiency harness: what ``EfficiencyCallback`` reports, compared between two commits (``make verify-efficiency
BASE=<sha>``). Each tree drives its own callback through the same scripted steps on a scripted B300 and clock; the
cases both run are compared by value."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, get_args
from unittest.mock import patch

import torch
import transformers
from peft import LoraConfig, get_peft_model
from torch.distributed.device_mesh import init_device_mesh
from transformers.distributed.tensor_parallel import apply_tensor_parallelism

from halo.callbacks.efficiency import EfficiencyCallback
from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.distributed.fsdp import IdentityParamSet, apply_fsdp2_per_layer, mixed_precision_policy
from halo.distributed.mesh import build_dp_tp_mesh, tp_submesh
from halo.distributed.pipeline_parallel.stage import build_pipeline_stage
from halo.distributed.tensor_parallel.parallelize_attention import apply_tp_to_attention_only
from halo.kernels.lowp.spec import LowpSpec
from tests.common.distributed import fake_process_group
from tests.common.equivalence.revisions import (
    RAISED,
    compare_outputs,
    export_tree,
    raised_description,
    resolve_commit,
    run_in_tree,
    split_accepted,
    unseen_controls,
    write_evidence,
)
from tests.common.parallelism import make_parallelism_config
from tests.common.repo import REPO_ROOT
from tests.common.reporting import snapshot_efficiency
from tests.common.tiny_models import (
    TINY_DENSE_FAMILIES,
    TINY_MOE_FAMILIES,
    TINY_MOE_VLM_FAMILIES,
    TinyFamily,
    tiny_family_model,
)

DENSE_FAMILIES: dict[str, TinyFamily] = dict(TINY_DENSE_FAMILIES)
MOE_FAMILIES: dict[str, TinyFamily] = {**TINY_MOE_FAMILIES, **TINY_MOE_VLM_FAMILIES}
FAMILIES = {**DENSE_FAMILIES, **MOE_FAMILIES}
REPRESENTATIVE_FAMILIES = ("qwen3", "qwen3_moe", "gpt_oss")
SCENARIO_FAMILY = "qwen3_moe"
# Whose records a case captures, wherever the callback's own modules live in it.
PACKAGE = EfficiencyCallback.__module__.partition(".")[0]
SEED = 0
WARMUP_STEPS = 1
PER_DEVICE_BATCH = 2
GRADIENT_ACCUMULATION = 2
MAX_LENGTH = 1024
FULL_MODEL_PARAMS = 5e8
STEP_SECONDS = (2.5e-6, 3.1e-6, 2.7e-6, 2.9e-6, 3.3e-6)
# HF's counter advances by these times the world size.
STEP_TOKENS = (3500, 3200, 3900, 3700, 3600)
# A step whose counted delta no length bound allows, and the step it lands on.
OUTLIER_TOKENS = 10**6
OUTLIER_STEP = 3
STEP_DOCUMENTS = ((1024, 1024, 900, 552), (800, 800, 800, 800), (1024, 1024, 1024, 828), (2000, 1700), (3600,))
# Host time before each step begins, carried from the previous step's end, and the negative control's tick.
BATCH_FETCH_SECONDS = 4e-7
CLOCK_TICK = 1e-7
PERTURBED_STEP = len(STEP_SECONDS) - 1
ALLOCATED_GIB = (1.25, 1.5, 1.375, 1.625, 1.5)
GIB = 1024**3
DEVICE_NAME = {"b300": "NVIDIA B300", "a100": "NVIDIA A100-SXM4-80GB"}

TokenFeed = Literal["counted", "uncounted", "outlier"]
ModelEdit = Literal["family", "lora", "unreadable_top_k", "single_expert", "no_top_k", "layerless", "parameterless"]


@dataclass(frozen=True)
class Shape:
    """A parallel layout as ``rank`` of ``world_size``: its ``ParallelismConfig`` fields and whether FSDP2 shards what
    the EP layers leave."""

    world_size: int
    fields: dict = field(default_factory=dict)
    fsdp: bool = False
    gpus_per_node: int | None = None
    rank: int = 0


SHAPES = {
    "single": Shape(1),
    "ep2": Shape(2, {"ep_size": 2}),
    "ep4": Shape(4, {"ep_size": 4}),
    "etp2": Shape(2, {"expert_tp_size": 2}),
    "ep2_etp2": Shape(4, {"ep_size": 2, "expert_tp_size": 2}),
    # Dense: transformers' own TP plan; MoE: attention-only TP, as the loaders shard each.
    "tp2": Shape(2, {"tp_size": 2}),
    "cp2": Shape(2, {"cp_size": 2}),
    "ep2_cp2": Shape(2, {"ep_size": 2, "cp_size": 2}),
    "ep2_tp2": Shape(2, {"ep_size": 2, "tp_size": 2}),
    # ep_size 1: FSDP2 shards the experts too, so every parameter is a DTensor.
    "fsdp2": Shape(2, fsdp=True),
    # The EP layers stay plain rank-local tensors beside DTensor peers.
    "ep2_fsdp2": Shape(2, {"ep_size": 2}, fsdp=True),
    # A stage must be a whole NVLink domain of two ranks or more, so two stages take two two-rank nodes.
    "pp2_first": Shape(4, {"pp_size": 2}, gpus_per_node=2),
    "pp2_last": Shape(4, {"pp_size": 2}, gpus_per_node=2, rank=2),
    "ep2_pp2_first": Shape(4, {"ep_size": 2, "pp_size": 2}, gpus_per_node=2),
    "ep2_pp2_last": Shape(4, {"ep_size": 2, "pp_size": 2}, gpus_per_node=2, rank=2),
    "etp2_pp2_first": Shape(4, {"expert_tp_size": 2, "pp_size": 2}, gpus_per_node=2),
    "etp2_pp2_last": Shape(4, {"expert_tp_size": 2, "pp_size": 2}, gpus_per_node=2, rank=2),
}
DENSE_SHAPES = ("single", "tp2", "cp2", "fsdp2", "pp2_first", "pp2_last")


@dataclass(frozen=True)
class Scenario:
    """What a case changes from the defaults."""

    device: str | None = "b300"
    bf16: bool = True
    tf32: bool = False
    lowp_precision: str = "bf16"
    full_model_params: float | None = None
    report_mfu_diagnostics: bool = True
    measured_attention: bool = True
    model_at_init: bool = True
    tokens: TokenFeed = "counted"
    model: ModelEdit = "family"

    def __post_init__(self):
        for name, choices in (("tokens", get_args(TokenFeed)), ("model", get_args(ModelEdit))):
            if getattr(self, name) not in choices:
                raise ValueError(f"Scenario.{name}={getattr(self, name)!r} is none of {choices}")


SCENARIOS = {
    "default": Scenario(),
    "full_params": Scenario(full_model_params=FULL_MODEL_PARAMS),
    "no_diagnostics": Scenario(report_mfu_diagnostics=False),
    "config_attention": Scenario(measured_attention=False),
    "lora": Scenario(model="lora"),
    "no_device": Scenario(device=None),
    "no_peak": Scenario(device="a100", lowp_precision="fp8"),
    "fp8": Scenario(lowp_precision="fp8"),
    "fp32": Scenario(bf16=False),
    "tf32": Scenario(bf16=False, tf32=True),
    # The model handed over at train begin only, so the first setup reads the precision off the args alone.
    "late_model": Scenario(bf16=False, model_at_init=False),
    "token_fallback": Scenario(tokens="uncounted"),
    "token_outlier": Scenario(tokens="outlier"),
    # The MoE estimate raises after the parameter counts: a top-k field int() refuses.
    "estimate_raises": Scenario(model="unreadable_top_k"),
    "single_expert": Scenario(model="single_expert"),
    "no_top_k": Scenario(model="no_top_k"),
    # No decoder layer list, so the attention term comes off the config's depth.
    "layerless": Scenario(model="layerless"),
    # No parameters at all, so the projection term falls back to 12·d²·L.
    "parameterless": Scenario(model="parameterless"),
}


@dataclass(frozen=True)
class EfficiencyCase:
    family: str
    shape: str
    scenario: str

    @property
    def path(self) -> str:
        return f"{self.family}/{self.shape}/{self.scenario}"


def _cases() -> tuple[EfficiencyCase, ...]:
    cases = [EfficiencyCase(family, "single", "default") for family in FAMILIES]
    cases += [EfficiencyCase(family, "ep2_fsdp2", "default") for family in MOE_FAMILIES]
    for family in REPRESENTATIVE_FAMILIES:
        shapes = DENSE_SHAPES if family in DENSE_FAMILIES else tuple(SHAPES)
        cases += [EfficiencyCase(family, shape, "default") for shape in shapes if shape not in ("single", "ep2_fsdp2")]
    cases += [EfficiencyCase(SCENARIO_FAMILY, "single", scenario) for scenario in SCENARIOS if scenario != "default"]
    # A stage computes only its share of the config's full depth.
    cases.append(EfficiencyCase(SCENARIO_FAMILY, "pp2_first", "layerless"))
    return tuple(cases)


CASES = _cases()


class _ScriptedEvent:
    """A CUDA event the scripted device reaches as it is recorded, at the clock's reading."""

    def __init__(self, device: ScriptedDevice):
        self.device = device
        self.at = 0.0

    def record(self, *args, **kwargs) -> None:
        self.at = self.device.now

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        pass

    def elapsed_time(self, end: _ScriptedEvent) -> float:
        return (end.at - self.at) * 1000.0


class ScriptedDevice:
    """The CUDA calls the callback makes, answered by a GPU of the scenario's name; the driver sets clock readings per
    hook."""

    def __init__(self, name: str | None):
        self.name = name
        self.now = 0.0
        self.step = 0

    def perf_counter(self) -> float:
        return self.now

    def allocated(self, *args, **kwargs) -> int:
        return int(ALLOCATED_GIB[self.step] * GIB)

    def event(self, *args, **kwargs) -> _ScriptedEvent:
        return _ScriptedEvent(self)

    @contextlib.contextmanager
    def installed(self) -> Iterator[None]:
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(time, "perf_counter", self.perf_counter))
            stack.enter_context(patch.object(torch.cuda, "is_available", lambda: self.name is not None))
            if self.name is not None:
                calls = {
                    "get_device_name": lambda *args, **kwargs: DEVICE_NAME[self.name],
                    "Event": self.event,
                    "synchronize": lambda *args, **kwargs: None,
                    "reset_peak_memory_stats": lambda *args, **kwargs: None,
                    "memory_allocated": self.allocated,
                    "memory_reserved": lambda *args, **kwargs: 2 * self.allocated(),
                    "max_memory_allocated": lambda *args, **kwargs: 3 * self.allocated() // 2,
                }
                for name, call in calls.items():
                    stack.enter_context(patch.object(torch.cuda, name, call))
            yield


class AttentionWork:
    """The per-step attention work a trainer measures, costed through the layout the callback hands back."""

    def __init__(self):
        self.layout = None
        self.documents: tuple[int, ...] = ()

    def drain_step_work(self) -> tuple[float, int]:
        documents, self.documents = self.documents, ()
        if self.layout is None or not documents:
            return 0.0, 0
        return float(self.layout.flops_for_documents(torch.tensor(documents))), len(documents)


class _PackageRecords(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.lines: list[list[str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith(f"{PACKAGE}."):
            self.lines.append([record.levelname, record.getMessage()])


@contextlib.contextmanager
def captured_records() -> Iterator[list[list[str]]]:
    """Every INFO-and-above record under :data:`PACKAGE` while inside, in place of the root handlers."""
    root = logging.getLogger()
    handler = _PackageRecords()
    handlers, level = root.handlers[:], root.level
    root.handlers, root.level = [handler], logging.INFO
    try:
        yield handler.lines
    finally:
        root.handlers, root.level = handlers, level


def _apply_tensor_parallelism(model: torch.nn.Module, family: str, parallelism) -> None:
    """The TP sharding the loaders apply: attention-only on MoE, transformers' own plan on dense."""
    mesh = build_dp_tp_mesh(parallelism.tp_size, parallelism.data_parallel_size, device_type="cpu")
    if family in MOE_FAMILIES:
        apply_tp_to_attention_only(model, mesh)
    else:
        apply_tensor_parallelism(model, tp_submesh(mesh))


def _model(case: EfficiencyCase, scenario: Scenario, parallelism) -> torch.nn.Module:
    family = FAMILIES[case.family]
    torch.manual_seed(SEED)
    overrides = {"num_experts": 1, "num_experts_per_tok": 1} if scenario.model == "single_expert" else None
    model = tiny_family_model(family, overrides=overrides)
    if scenario.model in ("layerless", "parameterless"):
        shell = torch.nn.Module()
        shell.config = model.config
        if scenario.model == "layerless":
            shell.proj = torch.nn.Linear(model.config.get_text_config().hidden_size, 1)
        return shell
    model.to(torch.bfloat16 if scenario.bf16 else torch.float32)
    if parallelism.tp_size > 1:
        _apply_tensor_parallelism(model, case.family, parallelism)
    if case.family in MOE_FAMILIES and parallelism.needs_ep_layers:
        apply_ep_layers(model, parallelism.ep_config())
    if parallelism.pp_size > 1:
        model = build_pipeline_stage(model, parallelism.pp_rank, parallelism.pp_size, moe_balancing="none")
    if scenario.model == "lora":
        targets = list(family.attention_targets or ("q_proj", "v_proj"))
        model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=targets))
    if SHAPES[case.shape].fsdp:
        ignored = (
            ()
            if parallelism.experts_fsdp_managed
            else [p for _, layer in named_ep_layers(model) for p in layer.parameters()]
        )
        policy = mixed_precision_policy(
            torch.bfloat16 if scenario.bf16 else torch.float32, fp32_reduce=False, cast_forward_inputs=True
        )
        apply_fsdp2_per_layer(
            model, init_device_mesh("cpu", (parallelism.world_size,)), policy, False, IdentityParamSet(ignored)
        )
    text = model.config.get_text_config()
    if scenario.model == "unreadable_top_k":
        # Past the config's own field validation, which refuses the value on assignment.
        text.__dict__["num_experts_per_tok"] = "two"
    elif scenario.model == "no_top_k":
        text.num_experts_per_tok = 0
    return model


def _training_args(scenario: Scenario, output_dir: str) -> transformers.TrainingArguments:
    args = transformers.TrainingArguments(
        output_dir=output_dir,
        use_cpu=True,
        bf16=scenario.bf16,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        logging_steps=1,
        include_num_input_tokens_seen="no" if scenario.tokens == "uncounted" else "all",
        report_to=[],
    )
    if scenario.tf32:
        # After construction: the arguments refuse tf32 with no Ampere GPU, and the callback reads only the flag.
        args.tf32 = True
    return args


def _step_tokens(scenario: Scenario, step: int, world_size: int) -> int:
    if scenario.tokens == "uncounted":
        return 0
    tokens = OUTLIER_TOKENS if scenario.tokens == "outlier" and step == OUTLIER_STEP else STEP_TOKENS[step]
    return tokens * world_size


def run_case(case: EfficiencyCase, scratch: str, *, perturb: bool = False) -> dict:
    shape, scenario = SHAPES[case.shape], SCENARIOS[case.scenario]
    with fake_process_group(shape.rank, shape.world_size):
        parallelism = make_parallelism_config(
            world_size=shape.world_size,
            gpus_per_node=shape.gpus_per_node or shape.world_size,
            rank=shape.rank,
            lowp=LowpSpec(precision=scenario.lowp_precision),
            **shape.fields,
        )
        model = _model(case, scenario, parallelism)
        args = _training_args(scenario, scratch)
        device, work = ScriptedDevice(scenario.device), AttentionWork()
        state, control = transformers.TrainerState(), transformers.TrainerControl()
        steps = []
        with device.installed(), captured_records() as records:
            callback = EfficiencyCallback(
                parallelism,
                n_warmup_steps=WARMUP_STEPS,
                num_full_model_params=scenario.full_model_params,
                report_mfu_diagnostics=scenario.report_mfu_diagnostics,
                script_args=SimpleNamespace(max_length=MAX_LENGTH),
            )
            callback.on_init_end(args, state, control, **({"model": model} if scenario.model_at_init else {}))
            callback.on_train_begin(args, state, control, model=model)
            if scenario.measured_attention:
                work.layout = callback.bind_attention_work_source(work)
            for step, seconds in enumerate(STEP_SECONDS):
                device.step = step
                device.now += BATCH_FETCH_SECONDS
                callback.on_step_begin(args, state, control, model=model)
                work.documents = STEP_DOCUMENTS[step]
                state.global_step += 1
                state.num_input_tokens_seen += _step_tokens(scenario, step, shape.world_size)
                device.now += seconds + (CLOCK_TICK if perturb and step == PERTURBED_STEP else 0.0)
                callback.on_step_end(args, state, control, model=model)
                logs: dict[str, float] = {}
                callback.on_log(args, state, control, logs=logs, model=model)
                steps.append(logs)
            snapshot = snapshot_efficiency(callback)
    return {"steps": steps, "records": records, "snapshot": snapshot}


def produce(output: Path, cases, *, perturb: bool = False) -> None:
    """Write each case's report to ``output/<case path>.json``, or what it raised."""
    verbosity = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()
    try:
        with tempfile.TemporaryDirectory() as scratch:
            for case in cases:
                try:
                    report = run_case(case, scratch, perturb=perturb)
                except Exception as error:
                    report = {RAISED: raised_description(error, scratch)}
                path = output / f"{case.path}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    finally:
        transformers.logging.set_verbosity(verbosity)


def read_reports(root: Path) -> dict[str, dict]:
    return {
        str(path.relative_to(root).with_suffix("")): json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.rglob("*.json"))
    }


def raised_reports(reports: dict[str, dict]) -> dict[str, dict]:
    return {path: report[RAISED] for path, report in reports.items() if RAISED in report}


def assess(reports: dict[str, dict[str, dict]], accept=()) -> tuple[str, dict]:
    """The verdict and findings on the three reports; only cases both trees ran are compared. Fails on a delta outside
    ``accept``, any head raise, or a control report that matches its base or raised."""
    base, head, control = reports["base"], reports["head"], reports["control"]
    shared = base.keys() & head.keys()
    compared = compare_outputs({path: base[path] for path in shared}, {path: head[path] for path in shared})
    deltas, accepted = split_accepted(compared, accept)
    raised = raised_reports(head)
    unseen = unseen_controls(compare_outputs(base, control), control, raised_reports(control))
    findings = {
        "cases": len(shared),
        "new": sorted(head.keys() - shared),
        "dropped": sorted(base.keys() - shared),
        "raised": raised,
        "negative_control": {"cases": sorted(control), "unseen": unseen},
        "accepted": accepted,
        "deltas": deltas,
    }
    return "fail" if deltas or raised or unseen else "pass", findings


def _produce_in(tree: Path, output: Path, case_paths: list[str] | None, *, perturb: bool = False) -> dict[str, dict]:
    """Run ``tree``'s own producer over ``case_paths`` (its whole roster when None) and read back its reports."""
    args = ["--produce", str(output), *(["--cases", *case_paths] if case_paths else [])]
    run_in_tree(tree, __spec__.name, *args, *(["--perturb"] if perturb else []))
    return read_reports(output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare what the EfficiencyCallback reports between two commits.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--base-ref", help="the commit the change is compared with")
    mode.add_argument("--produce", type=Path, help="write this tree's case reports under the directory")
    parser.add_argument("--head-ref", default="HEAD", help="the commit under test (default: HEAD)")
    by_path = {case.path: case for case in CASES}
    parser.add_argument(
        "--cases", nargs="+", choices=list(by_path), metavar="CASE", help="run only these (default: every case)"
    )
    parser.add_argument(
        "--accept", nargs="+", default=[], metavar="CASE", help="cases whose deltas a flagged change declares"
    )
    parser.add_argument("--perturb", action="store_true", help="move one step's clock (the negative control)")
    parser.add_argument("--image", help="the image the run used, recorded in the evidence")
    args = parser.parse_args(argv)
    if args.produce is not None:
        produce(args.produce, [by_path[path] for path in args.cases or by_path], perturb=args.perturb)
        return 0

    base, head = resolve_commit(REPO_ROOT, args.base_ref), resolve_commit(REPO_ROOT, args.head_ref)
    with tempfile.TemporaryDirectory() as scratch:
        trees = {
            label: export_tree(REPO_ROOT, commit, Path(scratch, label))
            for label, commit in (("base", base), ("head", head))
        }
        reports = {
            label: _produce_in(trees[label], Path(scratch, f"{label}-reports"), args.cases)
            for label in ("base", "head")
        }
        shared = sorted(reports["base"].keys() & reports["head"].keys())
        if not shared:
            raise SystemExit("efficiency callback: the base and head rosters share no case to compare")
        reports["control"] = _produce_in(trees["head"], Path(scratch, "control-reports"), shared[:1], perturb=True)
    verdict, findings = assess(reports, args.accept)
    path = write_evidence(
        "efficiency_callback", base=base, head=head, image=args.image, verdict=verdict, findings=findings
    )
    print(
        f"efficiency callback: {verdict}, {findings['cases']} cases compared ({len(findings['new'])} new, "
        f"{len(findings['dropped'])} dropped), {len(findings['deltas'])} deltas, {len(findings['accepted'])} "
        f"accepted, {len(findings['raised'])} raising, negative control unseen in "
        f"{len(findings['negative_control']['unseen'])} ({path})"
    )
    return 0 if verdict == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
