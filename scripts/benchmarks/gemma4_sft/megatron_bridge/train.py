"""Benchmark run of train_gemma4.py on the canonical tokens (same model / parallel / optimizer config, see build_config there).

Data: canonical pre-tokenized rows ($BENCH_ROOT/data/canonical_tokens.jsonl) fed directly through a
Bridge DatasetProvider - no tokenizer, template or masking inside Bridge. Megatron uses next-token labels, so each
row is converted exactly like Bridge's own build_shifted_labels_and_loss_mask: tokens = input_ids,
labels[t] = input_ids[t+1] where canonical labels[t+1] != -100 (else -100), loss_mask = labels != -100.
Order: dataloader_type="single" -> MegatronPretrainingSampler (sequential, no shuffle); with MBS 1 / DP 2 the
global batch of step k is rows [2(k-1), 2(k-1)+1] and DP rank r takes row 2(k-1)+r.
Logged per step: Bridge "lm loss" plus an independently all-reduced token-weighted CE (sum CE / label count
over both ranks), the canonical row id each rank consumed, and a check of the first 4 rows against the file.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import train_gemma4 as base
from megatron.bridge.data.base import DatasetProvider
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.vlm_step import forward_step as vlm_forward_step

BENCH_ROOT = os.environ["BENCH_ROOT"]  # see paths.env

CANON = os.environ.get("MB_CANON", os.environ.get("BENCH_CANON", f"{BENCH_ROOT}/data/canonical_tokens.jsonl"))
SEQ = base.SEQ
IGNORE = -100


def load_canonical(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    assert [r["row"] for r in rows] == list(range(len(rows)))
    return rows


def to_megatron_sample(row):
    ids = torch.tensor(row["input_ids"], dtype=torch.long)
    lab = torch.tensor(row["labels"], dtype=torch.long)
    assert ids.numel() == SEQ and lab.numel() == SEQ
    labels = torch.full_like(ids, IGNORE)
    keep = lab[1:] != IGNORE
    labels[:-1] = torch.where(keep, ids[1:], torch.full_like(ids[1:], IGNORE))
    return {
        "row": torch.tensor(row["row"], dtype=torch.long),
        "tokens": ids,
        "input_ids": ids,
        "position_ids": torch.arange(SEQ, dtype=torch.long),
        "labels": labels,
        "loss_mask": (labels != IGNORE).float(),
    }


class CanonicalDataset(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return to_megatron_sample(self.rows[i])

    @staticmethod
    def collate_fn(samples):
        return {k: torch.stack([s[k] for s in samples]) for k in samples[0]}


@dataclass(kw_only=True)
class CanonicalProvider(DatasetProvider):
    seq_length: int = SEQ
    skip_getting_attention_mask_from_dataset: bool = True

    def build_datasets(self, context):
        return CanonicalDataset(load_canonical(CANON)), None, None


# ---- per-step instrumentation shared between forward_step and the callback ----------------------
LRS = []
STATE = {"ce_sum": 0.0, "ntok": 0.0, "rows_this_step": [], "rows": [], "verify": [], "canon": None}


def _verify(batch):
    """Compare the tensors Bridge hands to the model with the canonical row (first 4 rows per rank)."""
    if len(STATE["verify"]) >= 4:
        return
    if STATE["canon"] is None:
        STATE["canon"] = load_canonical(CANON)
    for b in range(batch["tokens"].shape[0]):
        rid = int(batch["row"][b])
        c = STATE["canon"][rid]
        ids = torch.tensor(c["input_ids"])
        lab = torch.tensor(c["labels"])
        tok = batch["tokens"][b].cpu()
        got_lab = batch["labels"][b].cpu()
        got_mask = batch["loss_mask"][b].cpu() > 0
        exp_mask = lab[1:] != IGNORE  # model position t predicts canonical position t+1
        STATE["verify"].append(
            {
                "row": rid,
                "input_ids_mismatches": int((tok != ids).sum()),
                "label_mask_mismatches": int((got_mask[:-1] != exp_mask).sum()) + int(got_mask[-1]),
                "label_value_mismatches": int((got_lab[:-1][exp_mask] != lab[1:][exp_mask]).sum()),
                "label_tokens": int(exp_mask.sum()),
                "input_ids_sha1": hashlib.sha1(tok.numpy().tobytes()).hexdigest()[:12],
            }
        )


class _TapIterator:
    def __init__(self, it):
        self.it = it

    def __iter__(self):
        return self

    def __next__(self):
        batch = next(self.it)
        STATE["rows_this_step"].extend(int(r) for r in batch["row"])
        _verify(batch)
        return batch


def forward_step(state, data_iterator, model, return_schedule_plan=False):
    output, loss_fn = vlm_forward_step(state, _TapIterator(data_iterator), model, return_schedule_plan)

    def wrapped(*args, **kwargs):
        loss, num_tokens, report = loss_fn(*args, **kwargs)
        STATE["ce_sum"] += float(report["lm loss"][0])
        STATE["ntok"] += float(report["lm loss"][1])
        return loss, num_tokens, report

    return output, wrapped


class CanonicalBenchCallback(base.BenchCallback):
    def __init__(self):
        super().__init__()
        self.tw_losses = []

    def on_train_step_start(self, context):
        STATE["ce_sum"], STATE["ntok"], STATE["rows_this_step"] = 0.0, 0.0, []
        super().on_train_step_start(context)

    def on_train_step_end(self, context):
        super().on_train_step_end(context)
        t = torch.tensor([STATE["ce_sum"], STATE["ntok"]], device="cuda", dtype=torch.float64)
        torch.distributed.all_reduce(t, group=torch.distributed.group.WORLD)
        world = torch.distributed.get_world_size()
        # ranks in the same DP replica set (EP only) all hold distinct rows here: DP = world at EP2/TP1
        self.tw_losses.append(float(t[0] / t[1]))
        if context.optimizer is not None:
            LRS.append(context.optimizer.param_groups[0]["lr"])
        rows = [None] * world
        torch.distributed.all_gather_object(rows, STATE["rows_this_step"])
        STATE["rows"].append(rows)
        if torch.distributed.get_rank() == 0:
            print(
                f"[bench] step {len(self.tw_losses)} token_weighted_loss {self.tw_losses[-1]:.6f} rows/rank {rows}",
                flush=True,
            )

    def on_train_end(self, context):
        super().on_train_end(context)
        world = torch.distributed.get_world_size()
        ver = [None] * world
        torch.distributed.all_gather_object(ver, STATE["verify"])
        if torch.distributed.get_rank() != 0:
            return
        out = base.OUT_JSON
        result = json.loads(Path(out).read_text())
        expected = [[2 * k + r for r in range(world)] for k in range(len(STATE["rows"]))]
        consumed = [[rr[0] for rr in step] for step in STATE["rows"]]
        mism = sum(
            v["input_ids_mismatches"] + v["label_mask_mismatches"] + v["label_value_mismatches"]
            for rank in ver
            for v in rank
        )
        result.update(
            {
                "losses_token_weighted": self.tw_losses,
                "loss_logged_by_framework": "Megatron 'lm loss' = sum CE / sum label tokens all-reduced over DP "
                "(token-weighted); losses_token_weighted recomputed independently",
                "rows_consumed_per_step": consumed,
                "order_verification": {
                    "expected_step_k_rank_r": "row 2(k-1)+r",
                    "first_3_steps": consumed[:3],
                    "all_steps_match": consumed == expected,
                },
                "data_verification": {
                    "result": "exact match" if mism == 0 else f"{mism} mismatches",
                    "checked": "first 4 microbatches per rank: tokens vs canonical input_ids; labels/loss_mask vs "
                    "canonical labels shifted by one (Megatron next-token convention)",
                    "per_rank": ver,
                },
                "canonical_file": CANON,
                "optimizer_effective": {
                    k: getattr(context.state.cfg.optimizer, k, None)
                    for k in (
                        "optimizer",
                        "lr",
                        "min_lr",
                        "adam_beta1",
                        "adam_beta2",
                        "adam_eps",
                        "weight_decay",
                        "clip_grad",
                        "use_distributed_optimizer",
                        "use_precision_aware_optimizer",
                    )
                }
                | {
                    "lr_decay_style": context.state.cfg.scheduler.lr_decay_style,
                    "lr_warmup_iters": context.state.cfg.scheduler.lr_warmup_iters,
                    "start_weight_decay": context.state.cfg.scheduler.start_weight_decay,
                    "end_weight_decay": context.state.cfg.scheduler.end_weight_decay,
                    "lr_per_step": LRS,
                },
            }
        )
        Path(out).write_text(json.dumps(result, indent=2))
        print(
            "[bench] VERIFY",
            json.dumps({k: result[k] for k in ("order_verification", "data_verification")})[:2000],
            flush=True,
        )


def build_config_canonical():
    cfg = base.build_config()
    old = cfg.dataset
    cfg.dataset = CanonicalProvider(
        dataloader_type="single",  # MegatronPretrainingSampler: sequential, no shuffle
        num_workers=0,
        persistent_workers=False,
        data_sharding=True,
        pin_memory=True,
        drop_last=True,
    )
    del old
    # PROTOCOL.txt optimizer: AdamW (0.9, 0.999), eps 1e-8, weight_decay 0.0, lr 1e-5 constant, clip 1.0
    o = cfg.optimizer
    o.optimizer = "adam"  # Megatron "adam" = decoupled-weight-decay AdamW (TE FusedAdam, adam_w_mode)
    o.adam_beta1, o.adam_beta2, o.adam_eps = 0.9, 0.999, 1e-8
    o.weight_decay = 0.0
    o.lr = o.min_lr = 1e-5
    o.clip_grad = 1.0
    s = cfg.scheduler
    s.lr_decay_style = "constant"
    s.lr_warmup_iters = 0
    s.lr_warmup_init = 1e-5
    s.start_weight_decay = s.end_weight_decay = 0.0
    s.weight_decay_incr_style = "constant"
    return cfg


if __name__ == "__main__":
    finetune(build_config_canonical(), forward_step, callbacks=[CanonicalBenchCallback()])
