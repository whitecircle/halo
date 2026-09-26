"""the protocol instrumentation for HF-Trainer-based frameworks (ms-swift, unsloth).

install() patches transformers.Trainer so every trainer gets:
  * timing callback (cuda-synchronized step ends; throughput = tokens in steps 6..N / wall time of steps 6..N),
  * data verification: the input_ids/labels each rank actually receives in its first 4 micro-steps are diffed
    token-by-token against canonical_tokens.jsonl row 2*(k-1)+rank,
  * order verification: canonical row id (lookup by input_ids hash) consumed per rank for every step,
  * token-weighted loss: independent fp32 CE sum + label count per rank from the model's own logits (forward
    hook on the root model) or, when the framework returns no logits (Unsloth fused CE), from the final norm
    output through lm_head + Gemma final-logit softcapping; all-reduced over ranks -> losses_token_weighted[].
Rank 0 writes $BENCH_OUT.
"""

import functools
import hashlib
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformers
from transformers import TrainerCallback

BENCH_ROOT = os.environ["BENCH_ROOT"]  # see paths.env

CANON = os.environ.get("BENCH_CANON", f"{BENCH_ROOT}/data/canonical_tokens.jsonl")
N_VERIFY = 4


def _world():
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def _rank():
    return dist.get_rank() if _world() > 1 else 0


def _h(ids):
    return hashlib.sha1(json.dumps([int(x) for x in ids]).encode()).hexdigest()


def load_canonical():
    rows = [json.loads(line) for line in Path(CANON).read_text().splitlines()]
    return rows, {_h(r["input_ids"]): r["row"] for r in rows}


class BenchCallback(TrainerCallback):
    def __init__(self):
        self.step_end, self.tokens, self.label_tokens = [], [], []
        self._cur_tokens = 0
        self._cur_label_tokens = 0
        self.logs = {}
        self.t0 = None
        self.impl = {}
        self.canon, self.hash2row = load_canonical()
        self.micro = 0
        self.verify = []  # per micro-step dicts (first N_VERIFY)
        self.rows_seen = []  # canonical row id per step (this rank)
        self._step_rows = []
        self.ce_sum, self.ce_cnt = [], []
        self._cur_ce_sum = 0.0
        self._cur_ce_cnt = 0
        self.step_loss_local = []  # value returned by the framework's training_step on this rank
        self._cur_step_loss = 0.0
        self._labels = None
        self._hidden = None
        self.loss_source = None

    # ---- token-weighted CE -------------------------------------------------------------------------
    @torch.no_grad()
    def _ce_from_logits(self, logits, labels):
        # causal shift inside each sequence: (B, T) -> predict t+1 from t
        lab2 = labels[..., 1:].reshape(-1)
        lg = logits[:, :-1].reshape(-1, logits.shape[-1])
        tot = torch.zeros((), dtype=torch.float64, device=lg.device)
        for s in range(0, lg.shape[0], 512):
            tot += F.cross_entropy(
                lg[s : s + 512].float(), lab2[s : s + 512], ignore_index=-100, reduction="sum"
            ).double()
        return tot.item(), int((lab2 != -100).sum().item())

    def _install_loss_hooks(self, model):
        cb = self

        def root_hook(mod, args, kwargs, out):
            if not mod.training or cb._labels is None:
                return
            logits = getattr(out, "logits", None)
            vocab = mod.config.get_text_config().vocab_size
            if (
                isinstance(logits, torch.Tensor)
                and logits.dim() == 3
                and logits.shape[-1] == vocab
                and logits.shape[1] == cb._labels.shape[1]
            ):
                s, c = cb._ce_from_logits(logits.detach(), cb._labels)
                cb.loss_source = "model logits (root forward hook)"
            elif cb._hidden is not None:
                head = mod.get_output_embeddings()
                w = head.weight
                h = cb._hidden
                cap = mod.config.get_text_config().final_logit_softcapping
                lab2 = cb._labels[..., 1:].reshape(-1)
                hs = h[:, :-1].reshape(-1, h.shape[-1])
                tot = torch.zeros((), dtype=torch.float64, device=h.device)
                with torch.no_grad():
                    for st in range(0, hs.shape[0], 512):
                        lg = F.linear(hs[st : st + 512], w)
                        if cap is not None:
                            lg = torch.tanh(lg / cap) * cap
                        tot += F.cross_entropy(
                            lg.float(), lab2[st : st + 512], ignore_index=-100, reduction="sum"
                        ).double()
                s, c = tot.item(), int((lab2 != -100).sum().item())
                cb.loss_source = "final-norm hidden states @ lm_head.weight + softcap (framework returns no logits)"
            else:
                return
            cb._cur_ce_sum += s
            cb._cur_ce_cnt += c
            cb._hidden = None

        model.register_forward_hook(root_hook, with_kwargs=True)
        # final norm of the text model (fallback path)
        norm = None
        for name, m in model.named_modules():
            if name.endswith("language_model.norm") or name == "model.norm":
                norm = m
        if norm is not None:

            def norm_hook(mod, args, out):
                if cb._labels is not None and torch.is_grad_enabled():
                    cb._hidden = out.detach()

            norm.register_forward_hook(norm_hook)

    # ---- callbacks ---------------------------------------------------------------------------------
    def on_train_begin(self, args, state, control, model=None, optimizer=None, **kw):
        self.opt_info = {
            "trainer_args": {
                k: getattr(args, k, None)
                for k in (
                    "optim",
                    "learning_rate",
                    "adam_beta1",
                    "adam_beta2",
                    "adam_epsilon",
                    "weight_decay",
                    "lr_scheduler_type",
                    "warmup_steps",
                    "warmup_ratio",
                    "max_grad_norm",
                    "train_sampling_strategy",
                )
            }
        }
        if optimizer is not None:
            opt = getattr(optimizer, "optimizer", optimizer)
            self.opt_info["optimizer_class"] = f"{type(opt).__module__}.{type(opt).__name__}"
            self.opt_info["param_groups"] = [
                {
                    k: (
                        v
                        if isinstance(v, (int, float, str, bool, type(None)))
                        else list(v)
                        if isinstance(v, tuple)
                        else str(v)
                    )
                    for k, v in g.items()
                    if k != "params"
                }
                | {"n_params": len(g["params"])}
                for g in opt.param_groups
            ]
        if model is not None:
            cfgs = [model.config] + [getattr(model.config, "text_config", None)]
            self.impl["experts_implementation"] = [getattr(c, "_experts_implementation", None) for c in cfgs if c]
            self.impl["attn_implementation"] = [getattr(c, "_attn_implementation", None) for c in cfgs if c]
            self.impl["model_forward"] = getattr(getattr(model, "forward", None), "__module__", None)
            if os.environ.get("BENCH_TW_HOOK", "1") == "1":
                self._install_loss_hooks(model)
        self.host_load = {"train_begin": Path("/proc/loadavg").read_text().split()[:3]}
        torch.cuda.synchronize()
        self.t0 = time.perf_counter()

    def on_step_end(self, args, state, control, **kw):
        torch.cuda.synchronize()
        self.step_end.append(time.perf_counter())
        if len(self.step_end) == int(os.environ.get("BENCH_WARMUP", 5)):
            self.host_load["timed_window_start"] = Path("/proc/loadavg").read_text().split()[:3]
        self.tokens.append(self._cur_tokens)
        self.label_tokens.append(self._cur_label_tokens)
        self.ce_sum.append(self._cur_ce_sum)
        self.ce_cnt.append(self._cur_ce_cnt)
        self.rows_seen.append(self._step_rows)
        self.step_loss_local.append(self._cur_step_loss)
        self._cur_step_loss = 0.0
        self._cur_tokens = self._cur_label_tokens = self._cur_ce_cnt = 0
        self._cur_ce_sum = 0.0
        self._step_rows = []

    def on_log(self, args, state, control, logs=None, **kw):
        if logs and "loss" in logs:
            self.logs[state.global_step] = {"loss": logs.get("loss"), "grad_norm": logs.get("grad_norm")}

    def record_inputs(self, inputs):
        ids = inputs["input_ids"]
        labels = inputs.get("labels")
        self._labels = labels
        am = inputs.get("attention_mask")
        self._cur_tokens += (
            int(am.sum().item()) if isinstance(am, torch.Tensor) and am.dim() == 2 else int(ids.numel())
        )
        if isinstance(labels, torch.Tensor):
            self._cur_label_tokens += int((labels != -100).sum().item())
        ids_l = ids[0].tolist()
        row = self.hash2row.get(_h(ids_l))
        self._step_rows.append(row)
        if self.micro < N_VERIFY:
            k = self.micro + 1
            exp = self.canon[2 * (k - 1) + _rank()]
            lab_l = labels[0].tolist() if isinstance(labels, torch.Tensor) else None
            ids_mis = sum(a != b for a, b in zip(ids_l, exp["input_ids"], strict=False)) + abs(
                len(ids_l) - len(exp["input_ids"])
            )
            lab_mis = None
            if lab_l is not None:
                lab_mis = sum(a != b for a, b in zip(lab_l, exp["labels"], strict=False)) + abs(
                    len(lab_l) - len(exp["labels"])
                )
            self.verify.append(
                {
                    "rank": _rank(),
                    "micro_step": k,
                    "input_keys": sorted(inputs.keys()),
                    "expected_row": exp["row"],
                    "matched_row_by_hash": row,
                    "len": len(ids_l),
                    "input_ids_mismatches": ids_mis,
                    "labels_mismatches": lab_mis,
                    "n_label_tokens": None if lab_l is None else sum(x != -100 for x in lab_l),
                    "input_ids_head": ids_l[:8],
                    "labels_nonmasked_first_idx": None
                    if lab_l is None
                    else next((i for i, x in enumerate(lab_l) if x != -100), None),
                }
            )
        self.micro += 1

    def on_train_end(self, args, state, control, **kw):
        self.host_load["train_end"] = Path("/proc/loadavg").read_text().split()[:3]
        dev = torch.device("cuda", torch.cuda.current_device())
        ws = _world()
        tok = torch.tensor(self.tokens, dtype=torch.float64, device=dev)
        lab = torch.tensor(self.label_tokens, dtype=torch.float64, device=dev)
        ces = torch.tensor(self.ce_sum, dtype=torch.float64, device=dev)
        cec = torch.tensor(self.ce_cnt, dtype=torch.float64, device=dev)
        loc = torch.tensor([self.step_loss_local, self.label_tokens], dtype=torch.float64, device=dev)
        mem = torch.tensor(
            [torch.cuda.max_memory_allocated() / 2**30, torch.cuda.max_memory_reserved() / 2**30], device=dev
        )
        rows = torch.tensor(
            [[r if r is not None else -1 for r in rs][:1] or [-1] for rs in self.rows_seen], device=dev
        ).view(-1)
        if ws > 1:
            for t in (tok, lab, ces, cec):
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
            dist.all_reduce(mem, op=dist.ReduceOp.MAX)
            all_rows = [torch.zeros_like(rows) for _ in range(ws)]
            dist.all_gather(all_rows, rows)
            all_loc = [torch.zeros_like(loc) for _ in range(ws)]
            dist.all_gather(all_loc, loc)
            vobj = [None] * ws
            dist.all_gather_object(vobj, self.verify)
        else:
            all_rows, vobj, all_loc = [rows], [self.verify], [loc]
        if _rank() != 0:
            return
        tok = tok.tolist()
        times = [self.step_end[0] - self.t0] + [b - a for a, b in zip(self.step_end, self.step_end[1:], strict=False)]
        n = len(times)
        skip = int(os.environ.get("BENCH_WARMUP", 5))
        thr = sum(tok[skip:]) / (self.step_end[-1] - self.step_end[skip - 1]) if n > skip else None
        verify = [v for vs in vobj for v in vs]
        # per-rank framework step loss (training_step return) and per-rank label count (labels != -100, unshifted)
        per_rank = {
            f"rank{r}": {"step_loss": all_loc[r][0].tolist(), "label_tokens": all_loc[r][1].tolist()}
            for r in range(len(all_loc))
        }
        L = torch.stack([a[0] for a in all_loc])
        C = torch.stack([a[1] for a in all_loc])  # labels[0] is always -100, so shifted count == count
        tw_from_local_means = ((L * C).sum(0) / C.sum(0)).tolist()
        exact = all(v["input_ids_mismatches"] == 0 and v["labels_mismatches"] == 0 for v in verify)
        order = {f"rank{r}": [int(x) for x in all_rows[r].tolist()] for r in range(len(all_rows))}
        expected = {f"rank{r}": [2 * k + r for k in range(n)] for r in range(len(all_rows))}
        out = {
            "num_steps": n,
            "world_size": ws,
            "cluster_tokens_per_second": thr,
            "tokens_per_step": tok,
            "loss_tokens_per_step": lab.tolist(),
            "step_times_s": times,
            "losses": [self.logs.get(i + 1, {}).get("loss") for i in range(n)],
            "losses_token_weighted": [s / c if c else None for s, c in zip(ces.tolist(), cec.tolist(), strict=False)],
            "token_weighted_label_count": cec.tolist(),
            "token_weighted_loss_source": self.loss_source,
            "losses_token_weighted_from_rank_means": tw_from_local_means,
            "per_rank_framework_loss": per_rank,
            "grad_norms": [self.logs.get(i + 1, {}).get("grad_norm") for i in range(n)],
            "peak_mem_allocated_gib": mem[0].item(),
            "peak_mem_reserved_gib": mem[1].item(),
            "impl": self.impl,
            "host_load_1_5_15min": self.host_load,
            "throughput_valid": max(float(v[0]) for v in self.host_load.values()) <= 20.0,
            "optimizer_effective": getattr(self, "opt_info", None),
            "data_verification": {"result": "exact match" if exact else "MISMATCH", "per_micro_step": verify},
            "order_verification": {
                "canonical_row_per_step": order,
                "expected": expected,
                "result": "exact match" if order == expected else "MISMATCH",
            },
        }
        print(
            "BENCH_RESULT "
            + json.dumps({k: out[k] for k in ("cluster_tokens_per_second", "peak_mem_allocated_gib")})
            + " DATA "
            + out["data_verification"]["result"]
            + " ORDER "
            + out["order_verification"]["result"],
            flush=True,
        )
        if os.environ.get("BENCH_OUT"):
            Path(os.environ["BENCH_OUT"]).write_text(json.dumps(out, indent=1))


def install():
    orig_init = transformers.Trainer.__init__

    @functools.wraps(orig_init)
    def __init__(self, *a, **kw):
        orig_init(self, *a, **kw)
        cb = BenchCallback()
        self.add_callback(cb)
        orig_step = self.training_step

        def training_step(model, inputs, *args, **kwargs):
            cb.record_inputs(inputs)
            try:
                out = orig_step(model, inputs, *args, **kwargs)
                cb._cur_step_loss += (
                    float(out.detach().float().item()) if isinstance(out, torch.Tensor) else float("nan")
                )
                return out
            finally:
                cb._labels = None

        self.training_step = training_step

    transformers.Trainer.__init__ = __init__
