#!/usr/bin/env python
"""Protocol-measurement wrapper around Halo's scripts/training/sft.py (no Halo source edits).

Injects one TrainerCallback via halo.entrypoints.script.configure_balancing_and_build_callbacks that records,
per optimizer step: wall time (cuda-synchronized, consecutive on_step_end deltas), loss, grad_norm,
trained tokens (non-pad label tokens and input tokens, summed over ranks); plus peak memory
(max over ranks). With $BENCH_CANON set it also verifies the data the model receives, as the other
frameworks' harnesses do: the canonical row each rank trains at every step (order_verification,
expected row 2*(k-1)+rank), and the first 4 micro-batches' input_ids and labels diffed token by token
against their expected canonical rows (data_verification). Rank 0 writes the JSON given by $BENCH_OUT.

Usage: torchrun ... bench_sft.py <config.yaml> [extra halo args]   (run from the Halo checkout)
"""

import json
import os
import runpy
import sys
import time

import torch
import torch.distributed as dist
from transformers import TrainerCallback

TREE = os.getcwd()  # both launchers run from the Halo checkout
sys.path.insert(0, TREE)

import halo.entrypoints.script as script_runner  # noqa: E402

_tok_counter = {"input": 0, "label": 0}
N_VERIFY = 4
_verify = {"rows": [], "diffs": []}  # this rank: canonical row per micro-batch; per-micro-batch mismatch counts
_canon = {}


def _canonical():
    """Canonical rows from $BENCH_CANON: by input_ids and by row id (loaded once, lazily)."""
    if not _canon and os.environ.get("BENCH_CANON"):
        with open(os.environ["BENCH_CANON"]) as f:
            for line in f:
                row = json.loads(line)
                _canon.setdefault("by_ids", {})[tuple(row["input_ids"])] = row["row"]
                _canon.setdefault("rows", {})[row["row"]] = row
    return _canon


def _verify_micro_batch(ids: torch.Tensor, labels: torch.Tensor | None) -> None:
    canon = _canonical()
    if not canon:
        return
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    for b in range(ids.shape[0]):
        got = ids[b].tolist()
        row = canon["by_ids"].get(tuple(got), -1)
        _verify["rows"].append(row)
        k = len(_verify["rows"]) - 1
        if k < N_VERIFY:
            expected = canon["rows"].get(world * k + rank)
            if expected is None:
                _verify["diffs"].append(None)
                continue
            id_diff = sum(a != e for a, e in zip(got, expected["input_ids"], strict=False)) + abs(
                len(got) - len(expected["input_ids"])
            )
            lab_diff = 0
            if labels is not None:
                lab = labels[b].tolist()
                lab_diff = sum(a != e for a, e in zip(lab, expected["labels"], strict=False)) + abs(
                    len(lab) - len(expected["labels"])
                )
            _verify["diffs"].append(
                {"expected_row": world * k + rank, "input_ids_mismatches": id_diff, "label_mismatches": lab_diff}
            )


class ProtocolBenchCallback(TrainerCallback):
    def __init__(self):
        self.t_prev = None
        self.t_begin = None
        self.step_times, self.losses, self.grad_norms, self.step_tokens = [], [], [], []
        self._tok_at_prev = 0

    def on_train_begin(self, args, state, control, **kw):
        torch.cuda.synchronize()
        self.t_prev = time.perf_counter()

    def on_step_end(self, args, state, control, **kw):
        torch.cuda.synchronize()
        t = time.perf_counter()
        self.step_times.append(t - self.t_prev)
        self.t_prev = t
        self.step_tokens.append(_tok_counter["input"] - self._tok_at_prev)
        self._tok_at_prev = _tok_counter["input"]
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"[bench] step {state.global_step} time {self.step_times[-1]:.4f}s", flush=True)

    def on_log(self, args, state, control, logs=None, **kw):
        if logs and "loss" in logs:
            self.losses.append(float(logs["loss"]))
            gn = logs.get("grad_norm")
            self.grad_norms.append(float(gn) if gn is not None else None)

    def on_train_end(self, args, state, control, **kw):
        dev = torch.device("cuda", torch.cuda.current_device())
        mem = torch.tensor(
            [torch.cuda.max_memory_allocated() / 2**30, torch.cuda.max_memory_reserved() / 2**30], device=dev
        )
        toks = torch.tensor(self.step_tokens, dtype=torch.float64, device=dev)
        lab = torch.tensor([_tok_counter["label"]], dtype=torch.float64, device=dev)
        gathered = [_verify]
        if dist.is_initialized():
            dist.all_reduce(mem, op=dist.ReduceOp.MAX)
            dist.all_reduce(toks)
            dist.all_reduce(lab)
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, _verify)
        if int(os.environ.get("RANK", "0")) != 0:
            return
        st = self.step_times
        toks = toks.tolist()
        steady = slice(5, len(st))
        wall = sum(st[steady])
        out = {
            "num_steps": len(st),
            "step_times_s": st,
            "losses": self.losses,
            "grad_norms": self.grad_norms,
            "tokens_per_step_list": toks,
            "tokens_per_step": toks[5] if len(toks) > 5 else None,
            "steady_wall_s": wall,
            "steady_tokens": sum(toks[steady]),
            "cluster_tokens_per_second": (sum(toks[steady]) / wall) if len(st) > 5 else None,
            "label_tokens_total": lab.item(),
            "peak_mem_allocated_gib": mem[0].item(),
            "peak_mem_reserved_gib": mem[1].item(),
        }
        if _canonical():
            world = len(gathered)
            rows = [g["rows"] for g in gathered]
            order_ok = all(r == world * k + rank for rank, rs in enumerate(rows) for k, r in enumerate(rs))
            diffs = [d for g in gathered for d in g["diffs"]]
            data_ok = all(d is not None and not d["input_ids_mismatches"] and not d["label_mismatches"] for d in diffs)
            out["order_verification"] = {
                "canonical_row_per_step_by_rank": rows,
                "expected": "rank r trains canonical row world*(k-1)+r at step k",
                "result": "exact match" if order_ok else "MISMATCH",
            }
            out["data_verification"] = {
                "per_micro_batch": diffs,
                "result": "exact match" if data_ok else "MISMATCH",
            }
        path = os.environ.get("BENCH_OUT")
        if path:
            with open(path, "w") as f:
                json.dump(out, f, indent=1)
        print(
            "[bench] RESULT "
            + json.dumps(
                {
                    k: out[k]
                    for k in (
                        "cluster_tokens_per_second",
                        "peak_mem_allocated_gib",
                        "peak_mem_reserved_gib",
                        "tokens_per_step",
                    )
                }
                | {k: out[k]["result"] for k in ("order_verification", "data_verification") if k in out}
            ),
            flush=True,
        )


_orig_build = script_runner.configure_balancing_and_build_callbacks


def _build(*a, **kw):
    cbs = _orig_build(*a, **kw)
    cbs.append(ProtocolBenchCallback())
    return cbs


vars(script_runner)["configure_balancing_and_build_callbacks"] = _build

# Count tokens fed to the model per micro-batch by wrapping the Halo mixin's training_step.
from halo.trainers.distributed_trainer import DistributedTrainerMixin  # noqa: E402

_orig_training_step = DistributedTrainerMixin.training_step


def _counting_training_step(self, model, inputs, num_items_in_batch=None):
    ids = inputs.get("input_ids")
    if ids is not None:
        am = inputs.get("attention_mask")
        _tok_counter["input"] += int(am.sum().item()) if am is not None and am.dim() == 2 else ids.numel()
        _verify_micro_batch(ids, inputs.get("labels"))
    lbl = inputs.get("labels")
    if lbl is not None:
        _tok_counter["label"] += int((lbl != -100).sum().item())
    return _orig_training_step(self, model, inputs, num_items_in_batch)


DistributedTrainerMixin.training_step = _counting_training_step

sys.argv = [os.path.join(TREE, "scripts/training/sft.py")] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
