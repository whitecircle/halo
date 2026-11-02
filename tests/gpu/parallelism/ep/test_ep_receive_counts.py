#!/usr/bin/env python
"""On real DeepEP dispatch, the receive counts each backend hands the expert compute are the received ids'
own per-expert counts, and the expert compute they size is sync-free and bit-identical to its device-sized
reference.

The grouped sort compacts DeepEP's ``-1`` padding with a ``nonzero_static`` sized by the dispatch's host-side
counts (V2 ``EPHandle.num_recv_tokens_per_expert_list``, V1's fourth dispatch output), and the per-expert loop
selects each expert's slots with a ``nonzero_static`` of that expert's count; a count that is off by one drops
slots or pads with one that gathers the wrong row. The CPU suite
(``tests/cpu/parallelism/test_ep_receive_counts.py``) can only fake DeepEP, so this pins the contract on the
installed extension: every tiny family's model at ``ep_size = world / expert_tp_size``, one forward and
backward per rank on a batch of its own, with every ``_compute_experts`` call audited on the path that layer
runs:

  * the counts are a tuple of ``experts_per_rank`` ints equal to the bincount of the received local ids;
  * the grouped sort, or the whole per-expert loop, run on that call's receive raises on no host sync
    (``set_sync_debug_mode("error")``) and equals the ``nonzero`` sort's six outputs, or the ``torch.where``
    loop's output, bit for bit;
  * every call ran the path the row expects (``--expect-path``);
  * somewhere in the run the receive carried ``-1`` padding (top-1 families carry none: a received token's
    one expert is local).

An EP group spanning NVLink domains (two nodes, ``ep_size = world``) runs DeepEP's hybrid dispatch kernel, which
reports the counts on its own code path.

    torchrun --nproc_per_node=2 tests/gpu/parallelism/ep/test_ep_receive_counts.py --backend elastic
    torchrun --nproc_per_node=2 tests/gpu/parallelism/ep/test_ep_receive_counts.py --backend legacy
    torchrun --nproc_per_node=2 tests/gpu/parallelism/ep/test_ep_receive_counts.py --backend elastic \
        --no-grouped-gemm --expect-path loop
    torchrun --nproc_per_node=4 tests/gpu/parallelism/ep/test_ep_receive_counts.py --backend elastic \
        --family gpt_oss --expert-tp-size 2
"""

import argparse
import contextlib

import torch

from halo.distributed.expert_parallel.deepep.capacity import bump_forward_generation
from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase, named_ep_layers
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.distributed.parallelism_config import ParallelismConfig
from halo.hardware import supports_grouped_mm
from tests.common.ep_reference import (
    grouped_sort_mismatches,
    nonzero_grouped_sort,
    random_token_batch,
    receive_counts,
    where_expert_loop,
)
from tests.common.harness import gpu_test_main
from tests.common.rank_log import log, log_all
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model

SEED = 42
BATCH, SEQ = 2, 64

_COMPUTE = EPMoELayerBase._compute_experts


@contextlib.contextmanager
def sync_is_an_error():
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode("default")


def audit(layer, tokens, experts, weights, output_dtype, recv_counts) -> dict:
    """One ``_compute_experts`` call's verdicts on the path ``layer`` runs. Never raises: a rank that raised here
    would leave its peers in the next dispatch."""
    path = "grouped" if layer._grouped_mm_enabled() else "loop"
    record = {
        "path": path,
        "counts_match": isinstance(recv_counts, tuple)
        and recv_counts == receive_counts(experts, layer.experts_per_rank),
        "padded": bool((experts < 0).any()),
        "sync_free": False,
        "bitwise": False,
        "detail": f"{path} counts={recv_counts}",
    }
    if tokens.shape[0] == 0:
        record.update(sync_free=True, bitwise=True)
        return record
    with torch.no_grad():
        torch.cuda.synchronize()
        try:
            with sync_is_an_error():
                if path == "grouped":
                    got = EPMoELayerBase._sort_tokens_for_grouped_mm(
                        layer, tokens, experts, weights, recv_counts=recv_counts
                    )
                else:
                    got = _COMPUTE(layer, tokens, experts, weights, output_dtype, recv_counts=recv_counts)
            torch.cuda.synchronize()
        except RuntimeError as exc:
            record["detail"] += f"; raised {exc}"
            return record
        record["sync_free"] = True
        if path == "grouped":
            mismatched = grouped_sort_mismatches(got, nonzero_grouped_sort(layer, tokens, experts, weights))
        else:
            want = where_expert_loop(layer, tokens, experts, weights, output_dtype)
            mismatched = [] if torch.equal(got, want) else ["output"]
        record["bitwise"] = not mismatched
        record["detail"] += f"; mismatched={mismatched}"
    return record


@contextlib.contextmanager
def audited_compute(records: list[dict]):
    def compute(self, tokens, experts, weights, output_dtype, *, recv_counts):
        records.append(audit(self, tokens, experts, weights, output_dtype, recv_counts))
        return _COMPUTE(self, tokens, experts, weights, output_dtype, recv_counts=recv_counts)

    EPMoELayerBase._compute_experts = compute
    try:
        yield
    finally:
        EPMoELayerBase._compute_experts = _COMPUTE


def run_family(ctx, family: str) -> tuple[dict[str, bool], bool]:
    """COLLECTIVE. ``family``'s checks under the row's backend and expert compute, and whether any receive
    carried padding."""
    pc = ParallelismConfig(
        ep_size=ctx.world_size // ARGS.expert_tp_size,
        expert_tp_size=ARGS.expert_tp_size,
        use_grouped_gemm=ARGS.grouped_gemm,
        ep_buffer_backend=ARGS.backend,
    )
    torch.manual_seed(SEED)
    model = tiny_family_model(TINY_MOE_FAMILIES[family]).to(ctx.device, torch.bfloat16)
    model = apply_ep_layers(model, pc.ep_config()).train()
    layers = named_ep_layers(model)
    vocab = model.get_input_embeddings().num_embeddings
    input_ids, labels = random_token_batch(vocab, BATCH, SEQ, ctx.device, seed=SEED + ctx.rank)

    records: list[dict] = []
    with audited_compute(records):
        bump_forward_generation()
        model(input_ids=input_ids, labels=labels).loss.backward()
    torch.cuda.synchronize()

    for index, record in enumerate(records):
        if not (record["counts_match"] and record["sync_free"] and record["bitwise"]):
            log_all(f"  {family} call {index} rank {ctx.rank}: {record}")
    log(f"  {family}: {len(records)} expert computes over {len(layers)} EP layers")
    path = ARGS.expect_path
    checks = {
        f"{family}_runs_the_{path}_path": all(r["path"] == path for r in records),
        f"{family}_every_layer_computed": len(records) >= len(layers) > 0,
        f"{family}_counts_match_received_ids": all(r["counts_match"] for r in records),
        f"{family}_{path}_sync_free": all(r["sync_free"] for r in records),
        f"{family}_{path}_equals_reference": all(r["bitwise"] for r in records),
    }
    del model
    return checks, any(r["padded"] for r in records)


parser = argparse.ArgumentParser()
parser.add_argument("--backend", choices=("elastic", "legacy"), required=True)
parser.add_argument("--family", choices=sorted(TINY_MOE_FAMILIES), action="append")
parser.add_argument("--expert-tp-size", type=int, choices=(1, 2), default=1)
parser.add_argument("--grouped-gemm", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--expect-path", choices=("grouped", "loop"), default="grouped")
ARGS = parser.parse_args()


@gpu_test_main(min_world_size=2, prefix="ep_receive_counts")
def run(ctx):
    torch.cuda.set_device(ctx.device)
    checks = {"grouped_mm_supported": supports_grouped_mm() or not ARGS.grouped_gemm}
    padded = False
    for family in ARGS.family or TINY_MOE_FAMILIES:
        family_checks, family_padded = run_family(ctx, family)
        checks |= family_checks
        padded |= family_padded
    checks["some_receive_carried_padding"] = padded
    return {"checks": checks}


if __name__ == "__main__":
    run()
