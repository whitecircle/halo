"""Thin benchmark driver around AutoModel's FinetuneRecipeForVLM.

Runs the stock recipe (setup + run_train_validation_loop) unchanged, and only wraps
log_train_metrics to record per-step wall time (host perf_counter after the optimizer
step, which the recipe already syncs via loss.item()), loss, grad norm and tokens.
At the end, gathers max_memory_allocated / max_memory_reserved across ranks.
Writes JSON to $BENCH_OUT on rank 0.
"""

import json
import os
import time

import nemo_automodel.components.loss.masked_ce as mce
import torch
import torch.distributed as dist
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.vlm.finetune import FinetuneRecipeForVLM


def main():
    cfg = parse_args_and_load_config()
    recipe = FinetuneRecipeForVLM(cfg)
    recipe.setup()

    opt = recipe.optimizer[0]
    g = opt.param_groups[0]
    eff_opt = {
        "cls": f"{type(opt).__module__}.{type(opt).__name__}",
        "lr": g["lr"],
        "betas": list(g["betas"]),
        "eps": g["eps"],
        "weight_decay": sorted({pg["weight_decay"] for pg in opt.param_groups}),
        "fused": g.get("fused"),
        "n_param_groups": len(opt.param_groups),
        "lr_scheduler": None if recipe.lr_scheduler is None else str(type(recipe.lr_scheduler[0])),
        "max_grad_norm": recipe.max_grad_norm,
        "param_dtypes": sorted({str(p.dtype) for pg in opt.param_groups for p in pg["params"]}),
    }
    if dist.get_rank() == 0:
        print("EFFECTIVE_OPTIMIZER", json.dumps(eff_opt), flush=True)
    records = []
    orig_log = recipe.log_train_metrics

    def log_hook(log_data):
        torch.cuda.synchronize()
        now = time.perf_counter()
        m = log_data.metrics
        gn = m["grad_norm"]
        gn = float(gn.item() if torch.is_tensor(gn) else gn)
        records.append(
            {
                "step": int(log_data.step),
                "wall": now,
                "loss": float(m["loss"]),
                "grad_norm": gn,
                "tokens": int(m["num_tokens_per_step"]),
                "label_tokens": int(m["num_label_tokens"]),
                "recipe_tps": float(m["tps"]),
            }
        )
        return orig_log(log_data)

    recipe.log_train_metrics = log_hook

    # data verification: capture the exact tensors handed to the model for the first 4 steps on this rank,
    # and an independent token-weighted CE (all-reduced sum-CE / label count) from the model logits.
    seen = []
    tw = []
    orig_fb = recipe._forward_backward_step

    def fb_hook(idx, batch, **kw):
        if len(seen) < 4:
            seen.append((batch["input_ids"][0].tolist(), batch["labels"][0].tolist()))
        return orig_fb(idx, batch, **kw)

    recipe._forward_backward_step = fb_hook

    orig_ce_call = mce.MaskedCrossEntropy.forward
    acc = {"sum": 0.0, "n": 0}

    def ce_call(self, logits, labels, *a, **k):
        with torch.no_grad():
            lg = logits.reshape(-1, logits.shape[-1]).float()
            lb = labels.reshape(-1).to(lg.device)
            m = lb != -100
            s = torch.nn.functional.cross_entropy(lg[m], lb[m], reduction="sum") if m.any() else lg.sum() * 0
            acc["sum"] += float(s)
            acc["n"] += int(m.sum())
        return orig_ce_call(self, logits, labels, *a, **k)

    tw_check = os.environ.get("BENCH_TW_CHECK", "0") == "1"
    if tw_check:
        mce.MaskedCrossEntropy.forward = ce_call
    orig_log2 = recipe.log_train_metrics

    def log_hook2(log_data):
        t = torch.tensor([acc["sum"], float(acc["n"])], device="cuda", dtype=torch.float64)
        dist.all_reduce(t)
        tw.append(float(t[0] / t[1]))
        acc["sum"] = 0.0
        acc["n"] = 0
        return orig_log2(log_data)

    if tw_check:
        recipe.log_train_metrics = log_hook2
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    recipe.run_train_validation_loop()
    torch.cuda.synchronize()

    alloc = torch.tensor([torch.cuda.max_memory_allocated() / 2**30], device="cuda")
    resv = torch.tensor([torch.cuda.max_memory_reserved() / 2**30], device="cuda")
    per_rank_alloc = [torch.zeros_like(alloc) for _ in range(dist.get_world_size())]
    per_rank_resv = [torch.zeros_like(resv) for _ in range(dist.get_world_size())]
    dist.all_gather(per_rank_alloc, alloc)
    dist.all_gather(per_rank_resv, resv)

    # data verification vs canonical rows (expected row for step k, rank r = 2(k-1)+r)
    canon = {}
    with open(os.environ["BENCH_CANON"]) as f:
        for line in f:
            r = json.loads(line)
            if r["row"] < 8:
                canon[r["row"]] = r
    rank = dist.get_rank()
    mism = 0
    detail = []
    for k, (ids, lab) in enumerate(seen):
        c = canon[2 * k + rank]
        exp_lab = c["labels"][1:] + [-100]
        n_ids = sum(a != b for a, b in zip(ids, c["input_ids"], strict=False)) + abs(len(ids) - len(c["input_ids"]))
        n_lab = sum((a != -100) != (b != -100) or (a != b) for a, b in zip(lab, exp_lab, strict=False)) + abs(
            len(lab) - len(exp_lab)
        )
        mism += n_ids + n_lab
        detail.append(
            {
                "step": k + 1,
                "rank": rank,
                "canonical_row": 2 * k + rank,
                "len": len(ids),
                "input_id_mismatches": n_ids,
                "label_mismatches": n_lab,
            }
        )
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, {"mism": mism, "detail": detail})

    if dist.get_rank() == 0:
        step_times = []
        prev = t_start
        for r in records:
            step_times.append(r["wall"] - prev)
            prev = r["wall"]
        warm = 5
        meas = records[warm:]
        wall = meas[-1]["wall"] - records[warm - 1]["wall"] if len(records) > warm else float("nan")
        toks = sum(r["tokens"] for r in meas)
        out = {
            "num_steps": len(records),
            "step_times_s": step_times,
            "losses": [r["loss"] for r in records],
            "grad_norms": [r["grad_norm"] for r in records],
            "tokens_per_step": [r["tokens"] for r in records],
            "label_tokens_per_step": [r["label_tokens"] for r in records],
            "recipe_tps": [r["recipe_tps"] for r in records],
            "measured_steps": f"{warm + 1}-{len(records)}",
            "measured_tokens": toks,
            "measured_wall_s": wall,
            "cluster_tokens_per_second": toks / wall if wall == wall else None,
            "peak_mem_allocated_gib_per_rank": [float(x.item()) for x in per_rank_alloc],
            "peak_mem_reserved_gib_per_rank": [float(x.item()) for x in per_rank_resv],
            "effective_optimizer": eff_opt,
            "losses_token_weighted": tw if tw_check else [r["loss"] for r in records],
            "losses_token_weighted_source": "independent all-reduced sum-CE/label-count hook on model logits"
            if tw_check
            else "recipe loss (MaskedCrossEntropy sum / global label count, dp all-reduce sum) = token-weighted by construction",
            "data_verification": {
                "result": "exact match"
                if sum(g["mism"] for g in gathered) == 0
                else f"{sum(g['mism'] for g in gathered)} mismatches",
                "per_row": [d for g in gathered for d in g["detail"]],
            },
            "peak_mem_allocated_gib": max(float(x.item()) for x in per_rank_alloc),
            "peak_mem_reserved_gib": max(float(x.item()) for x in per_rank_resv),
        }
        path = os.environ.get("BENCH_OUT", "bench_result.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=1)
        print(
            "BENCH_RESULT",
            json.dumps(
                {k: out[k] for k in ("cluster_tokens_per_second", "peak_mem_allocated_gib", "peak_mem_reserved_gib")}
            ),
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
