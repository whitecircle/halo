#!/usr/bin/env python
"""Protocol-measurement wrapper around Halo's scripts/training/sft.py (no Halo source edits).

Injects one TrainerCallback via src.training.script_runner.build_training_callbacks that records,
per optimizer step: wall time (cuda-synchronized, consecutive on_step_end deltas), loss, grad_norm,
trained tokens (non-pad label tokens and input tokens, summed over ranks); plus peak memory
(max over ranks). Rank 0 writes the JSON given by $BENCH_OUT.

Usage: torchrun ... bench_sft.py <config.yaml> [extra halo args]   (HALO_TREE on PYTHONPATH, cwd)
"""

import json
import os
import runpy
import sys
import time

import torch
import torch.distributed as dist
from transformers import TrainerCallback

TREE = os.environ.get("HALO_TREE", os.getcwd())
sys.path.insert(0, TREE)

import src.training.script_runner as script_runner  # noqa: E402

_tok_counter = {"input": 0, "label": 0}


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
        if dist.is_initialized():
            dist.all_reduce(mem, op=dist.ReduceOp.MAX)
            dist.all_reduce(toks)
            dist.all_reduce(lab)
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
            ),
            flush=True,
        )


_orig_build = script_runner.build_training_callbacks


def _build(*a, **kw):
    cbs = _orig_build(*a, **kw)
    cbs.append(ProtocolBenchCallback())
    return cbs


script_runner.build_training_callbacks = _build

# Count tokens fed to the model per micro-batch by wrapping the Halo mixin's training_step.
from src.trainers.mixins.base import DistributedTrainerMixin  # noqa: E402

_orig_training_step = DistributedTrainerMixin.training_step


def _counting_training_step(self, model, inputs, num_items_in_batch=None):
    ids = inputs.get("input_ids")
    if ids is not None:
        am = inputs.get("attention_mask")
        _tok_counter["input"] += int(am.sum().item()) if am is not None and am.dim() == 2 else ids.numel()
    lbl = inputs.get("labels")
    if lbl is not None:
        _tok_counter["label"] += int((lbl != -100).sum().item())
    return _orig_training_step(self, model, inputs, num_items_in_batch)


DistributedTrainerMixin.training_step = _counting_training_step

sys.argv = [os.path.join(TREE, "scripts/training/sft.py")] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
