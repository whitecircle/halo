#!/usr/bin/env python
"""Online GRPO / SDPG against a live vLLM server with a **dense** policy under TP and FSDP2 DP.

The MoE half of this pair covers EP/ETP; this one covers the two axes a dense policy reaches —
toolkit tensor parallelism, where the sync must gather DTensor shards, and plain FSDP2 data
parallelism, where a PEFT merge runs over DTensor adapters — plus the refusal that keeps LoRA off TP.

Rows (``--trainer {online,sdpg} --mode M [--resume]``):

    full_tp2            full fine-tune at ``tp_size=2`` — the served policy must move
    lora_fsdp           attention PEFT under plain FSDP2 DP — the merge path over DTensor adapters
    lora_tp2_rejected   ``tp_size=2`` + attention LoRA must raise at construction, naming TP
    --resume            the checkpoint's weights must generate the first resumed rollout

Requires the vLLM container serving the SAME dense checkpoint on a GPU outside
``CUDA_VISIBLE_DEVICES`` (a rank cannot NCCL broadcast to itself).

Usage:
    HALO_TEST_VLLM_DENSE_SERVER_URL=http://localhost:8010 NCCL_IB_DISABLE=1 NCCL_NET=Socket \
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
        tests/gpu/trainers/grpo/test_online_grpo_vllm_dense_e2e.py --trainer online --mode lora_fsdp
"""

import argparse

from src.env import env_int, env_str
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.online_grpo_e2e import modes_for, run_online_grpo_e2e

MODEL_NAME = env_str("HALO_TEST_ONLINE_GRPO_DENSE_MODEL", QWEN3_0_6B)
# Its own endpoint: the MoE half of this pair serves a different checkpoint on VLLM_SERVER_URL,
# and every row here asserts on logprobs only this model's server can produce.
SERVER_URL = env_str("HALO_TEST_VLLM_DENSE_SERVER_URL") or env_str("VLLM_SERVER_URL") or "http://localhost:8010"
# Trainer-side NCCL weight-transfer port. A resume row rebinds it for phase 2, which phase 1's
# close_communicator makes available again on both ends.
GROUP_PORT = env_int("HALO_TEST_VLLM_GROUP_PORT", 51380)


def _row_group_port(base: int, trainer: str, mode: str, resume: bool, family: str) -> int:
    """A deterministic per-row port: 22 rows share one pass back-to-back, and a listener freed by
    one row sits in TIME_WAIT while the next binds — so every row owns two ports (phase 2 of a
    resume binds ``+1``)."""
    rows = [(t, m, r) for t in ("online", "sdpg") for m in modes_for(family) for r in (False, True)]
    return base + 2 * rows.index((trainer, mode, resume))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer", choices=("online", "sdpg"), required=True)
    parser.add_argument("--mode", choices=modes_for("dense"), required=True)
    parser.add_argument("--resume", action="store_true", help="drive the two-phase resume invariant")
    return parser.parse_args()


def run(ctx) -> dict:
    args = _parse_args()
    return run_online_grpo_e2e(
        ctx,
        trainer_kind=args.trainer,
        mode=args.mode,
        resume=args.resume,
        server_url=SERVER_URL,
        group_port=_row_group_port(GROUP_PORT, args.trainer, args.mode, args.resume, "dense"),
        model_name=MODEL_NAME,
    )


main = gpu_test_main(exact_world_size=2, prefix="online_grpo_vllm_dense_e2e")(run)

if __name__ == "__main__":
    main()
