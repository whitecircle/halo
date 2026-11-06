"""Axolotl plugin: per-step timing / loss / grad-norm / token / peak-memory recorder for the
Gemma-4 throughput protocol. Writes JSON to $BENCH_OUT from rank 0 at train end."""

import json
import os
import time

import torch
import torch.distributed as dist
from axolotl.integrations.base import BasePlugin
from transformers import TrainerCallback

BENCH_TMP = os.environ.get("BENCH_TMP", os.path.join(os.environ["BENCH_ROOT"], "tmp"))  # see paths.env


class _BenchCallback(TrainerCallback):
    def __init__(self, counter):
        self.counter = counter
        self.t_prev = None
        self.step_times, self.step_tokens, self.step_nonpad = [], [], []
        self.losses, self.grad_norms = {}, {}
        self.check = None
        self.tw_losses, self.mean_of_means, self.rows_per_step, self.mism_per_step = [], [], [], []
        self.nlab_global = []

    def on_train_begin(self, args, state, control, **kw):
        torch.cuda.reset_peak_memory_stats()

    def on_step_begin(self, args, state, control, **kw):
        if self.t_prev is None:
            torch.cuda.synchronize()
            self.t_prev = time.perf_counter()

    def on_step_end(self, args, state, control, **kw):
        torch.cuda.synchronize()
        t = time.perf_counter()
        self.step_times.append(t - self.t_prev)
        self.t_prev = t
        tok = torch.tensor([self.counter["tok"], self.counter["nonpad"]], device="cuda", dtype=torch.float64)
        if dist.is_initialized():
            dist.all_reduce(tok)
        self.step_tokens.append(int(tok[0].item()))
        self.step_nonpad.append(int(tok[1].item()))
        self.counter["tok"] = 0
        if self.check is not None and self.check["sumce"]:
            k = len(self.step_times) - 1
            world = dist.get_world_size() if dist.is_initialized() else 1
            loc = torch.tensor(
                [
                    self.check["sumce"][k],
                    self.check["nlab"][k] if self.check["nlab"] else 0,
                    self.check["data_mismatch"][k] if self.check["data_mismatch"] else 0,
                ],
                device="cuda",
                dtype=torch.float64,
            )
            allv = [torch.zeros_like(loc) for _ in range(world)]
            if dist.is_initialized():
                dist.all_gather(allv, loc)
            else:
                allv = [loc]
            sc = sum(x[0].item() for x in allv)
            nl = sum(x[1].item() for x in allv)
            self.tw_losses.append(sc / nl if nl else None)
            self.nlab_global.append(int(nl))
            self.mean_of_means.append(sum(x[0].item() / x[1].item() for x in allv) / world if nl else None)
            rows = torch.tensor(
                [float(self.check["rows"][k]) if self.check["rows"] else -1.0], device="cuda", dtype=torch.float64
            )
            allr = [torch.zeros_like(rows) for _ in range(world)]
            if dist.is_initialized():
                dist.all_gather(allr, rows)
            else:
                allr = [rows]
            self.rows_per_step.append([int(x.item()) for x in allr])
            self.mism_per_step.append(int(sum(x[2].item() for x in allv)))
        self.counter["nonpad"] = 0
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                f"[bench] step {state.global_step} time {self.step_times[-1]:.3f}s tokens {self.step_tokens[-1]} "
                f"alloc {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB",
                flush=True,
            )

    def on_log(self, args, state, control, logs=None, **kw):
        if logs:
            if "loss" in logs:
                self.losses[state.global_step] = float(logs["loss"])
            if "grad_norm" in logs:
                self.grad_norms[state.global_step] = float(logs["grad_norm"])

    def on_train_end(self, args, state, control, **kw):
        mem = torch.tensor(
            [torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()], device="cuda", dtype=torch.float64
        )
        if dist.is_initialized():
            dist.all_reduce(mem, op=dist.ReduceOp.MAX)
        self.mem = mem
        # per-rank logged grad norms (HF computes clip_grad_norm_ on every rank; under pure EP/DDP the
        # DDP-ignored local expert shards make this rank-local)
        n = len(self.step_times)
        g = torch.tensor(
            [self.grad_norms.get(i + 1, float("nan")) for i in range(n)], device="cuda", dtype=torch.float64
        )
        world = dist.get_world_size() if dist.is_initialized() else 1
        allg = [torch.zeros_like(g) for _ in range(world)]
        if dist.is_initialized():
            dist.all_gather(allg, g)
        else:
            allg = [g]
        self.grad_norms_by_rank = [x.tolist() for x in allg]
        if int(os.environ.get("RANK", "0")) == 0:
            self._write()
        if os.environ.get("BENCH_EXIT_AFTER") == "1":
            # Protocol: no checkpoint saving. Axolotl always saves the final model after train();
            # skip it (outside the measured window) to avoid writing 50 GB per run.
            if dist.is_initialized():
                dist.barrier()
            print("[bench] exiting before axolotl final save", flush=True)
            os._exit(0)

    def _write(self):
        mem = self.mem
        n = len(self.step_times)
        steady = slice(5, n)
        wall = sum(self.step_times[steady])
        toks = sum(self.step_tokens[steady])
        out = {
            "num_steps": n,
            "step_times_s": self.step_times,
            "step_tokens": self.step_tokens,
            "step_nonpad_tokens": self.step_nonpad,
            "losses": [self.losses.get(i + 1) for i in range(n)],
            "grad_norms": [self.grad_norms.get(i + 1) for i in range(n)],
            "losses_token_weighted": self.tw_losses,
            "losses_mean_of_rank_means": self.mean_of_means,
            "rows_per_step_by_rank": self.rows_per_step,
            "data_mismatches_per_step": self.mism_per_step,
            "loss_scale_trainer": self.check["loss_scale"] if self.check else None,
            "loss_mode": self.check.get("loss_mode") if self.check else None,
            "grad_norms_by_rank": getattr(self, "grad_norms_by_rank", None),
            "label_tokens_per_step_by_rank_sum": self.nlab_global,
            "n_global_labels_rank0": self.check["n_global"] if self.check else None,
            "steady_wall_s": wall,
            "steady_tokens": toks,
            "cluster_tokens_per_second": toks / wall if wall else None,
            "peak_mem_allocated_gib": mem[0].item() / 2**30,
            "peak_mem_reserved_gib": mem[1].item() / 2**30,
        }
        path = os.environ.get("BENCH_OUT", f"{BENCH_TMP}/bench_out.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=1)
        print(f"[bench] RESULT {json.dumps({k: v for k, v in out.items() if not isinstance(v, list)})}", flush=True)


class BenchPlugin(BasePlugin):
    def __init__(self):
        super().__init__()
        self.counter = {"tok": 0, "nonpad": 0}
        self.check = {
            "rows": [],
            "data_mismatch": [],
            "sumce": [],
            "nlab": [],
            "n_global": [],
            "loss_raw": [],
            "loss_scale": None,
        }
        self.canon = None
        path = os.environ.get("BENCH_CANON")
        if path:
            # canonical rows: input_ids + labels, keyed by row id and by input_ids hash
            self.canon = {"by_hash": {}, "rows": []}
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    self.canon["rows"].append((r["input_ids"], r["labels"]))
                    self.canon["by_hash"][hash(tuple(r["input_ids"]))] = r["row"]

    def pre_model_load(self, cfg):
        print(
            f"[bench] pre_model_load: dist.is_initialized={dist.is_initialized()} RANK={os.environ.get('RANK')}",
            flush=True,
        )

    def post_model_build(self, cfg, model):
        print(f"[bench] post_model_build: dist.is_initialized={dist.is_initialized()}", flush=True)

    def post_model_load(self, cfg, model):
        n, shapes = 0, set()
        for _name, mod in model.named_modules():
            gp = getattr(mod, "gate_up_proj", None)
            if isinstance(gp, torch.nn.Parameter) and gp.dim() == 3:
                n += 1
                shapes.add((tuple(gp.shape), getattr(mod, "num_experts", None)))
        print(
            f"[bench] post_model_load: rank={os.environ.get('RANK')} experts_modules={n} gate_up_shapes/num_experts={shapes}",
            flush=True,
        )

    def post_trainer_create(self, cfg, trainer):
        orig = trainer.training_step
        counter = self.counter

        chk, canon = self.check, self.canon
        rank = int(os.environ.get("RANK", "0"))
        world = int(os.environ.get("WORLD_SIZE", "1"))

        def training_step(model, inputs, *a, **kw):
            ids = inputs.get("input_ids")
            if canon is not None and ids is not None:
                step = len(chk["rows"])  # 0-based optimizer step (GA=1, micro-batch 1)
                got_ids = ids[0].tolist()
                got_lab = inputs["labels"][0].tolist()
                row = canon["by_hash"].get(hash(tuple(got_ids)), -1)
                exp = world * step + rank
                ei, el = canon["rows"][exp]
                mism = (
                    sum(a_ != b_ for a_, b_ in zip(got_ids, ei, strict=False))
                    + abs(len(got_ids) - len(ei))
                    + sum(a_ != b_ for a_, b_ in zip(got_lab, el, strict=False))
                    + abs(len(got_lab) - len(el))
                )
                chk["rows"].append(row)
                chk["data_mismatch"].append(mism)
                if step < 3:
                    print(
                        f"[bench-check] step {step + 1} rank {rank} canonical_row={row} expected={exp} "
                        f"input_ids+labels mismatches={mism} first_ids={got_ids[:6]} n_labels={sum(x != -100 for x in got_lab)}",
                        flush=True,
                    )
                chk["nlab"].append(int(inputs["labels"][..., 1:].ne(-100).sum().item()))
            if ids is not None:
                counter["tok"] += ids.numel()
                am = inputs.get("attention_mask")
                counter["nonpad"] += int(am.sum().item()) if am is not None else ids.numel()
            return orig(model, inputs, *a, **kw)

        trainer.training_step = training_step

        orig_cl = trainer.compute_loss

        def compute_loss(model, inputs, *a, **kw):
            out = orig_cl(model, inputs, *a, **kw)
            loss = out[0] if isinstance(out, tuple) else out
            n = kw.get("num_items_in_batch")
            if n is None and len(a) >= 2:
                n = a[1]
            scale = 1
            args = trainer.args
            if args.average_tokens_across_devices and trainer.model_accepts_loss_kwargs and n is not None:
                scale = trainer.accelerator.num_processes
            chk["loss_scale"] = scale
            chk["loss_raw"].append(float(loss.detach().float().item()))
            chk["n_global"].append(int(n.item()) if torch.is_tensor(n) else n)
            if n is not None:
                # model loss = sum_CE_local / N_global (CCE reduction='sum' / num_items_in_batch), trainer * scale
                chk["sumce"].append(float(loss.detach().float().item()) / scale * float(n))
                chk["loss_mode"] = "sum/num_items_in_batch"
            else:
                # axolotl sets model_accepts_loss_kwargs=False for Gemma 4 -> no num_items_in_batch: the loss is the
                # rank-local mean CE over its shifted label tokens (CCE reduction='mean'); sum_CE_local = mean * n_local
                chk["sumce"].append(float(loss.detach().float().item()) * float(chk["nlab"][-1]))
                chk["loss_mode"] = "local mean (no num_items_in_batch)"
            return out

        trainer.compute_loss = compute_loss

    def add_callbacks_post_trainer(self, cfg, trainer):
        cb = _BenchCallback(self.counter)
        cb.check = self.check
        return [cb]
