#!/usr/bin/env python
"""Online GRPO / SDPG against a live vLLM server with a **MoE** policy under EP and ETP.

Every row asserts on the SERVER's behaviour, not on the trainer's tensors: the gather-side weight-sync
tests check what ``gather_and_send_weights`` produces, never that a real engine applied it, and a
server that accepts an expert update and silently reverts it makes every expert sync a no-op — the
served policy frozen for a whole RL run while the training loss moves normally.

Rows (``--trainer {online,sdpg} --mode M [--resume]``):

    full_ep2         full fine-tune at ``ep_size=2`` — the EP expert gather into the engine
    lora_ep2         attention PEFT under EP — the sync's merge → strip → unmerge
    expert_lora_ep2  native grouped expert LoRA — the ``merge_lora=True`` fold in the expert gather,
                     the only route an expert adapter has into the engine
    lora_etp2        attention PEFT at ``ep_size=1, expert_tp_size=2``, plus the config-time refusal
                     of native expert LoRA under expert TP
    --resume         the checkpoint's weights must generate the first resumed rollout

Requires the vLLM container serving the SAME MoE checkpoint with ``--moe-backend triton``, on a GPU
outside ``CUDA_VISIBLE_DEVICES`` (a rank cannot NCCL broadcast to itself), and 2 GPUs of >=140 GB.

Usage:
    VLLM_SERVER_URL=http://localhost:8000 NCCL_IB_DISABLE=1 NCCL_NET=Socket \
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
        tests/gpu/trainers/grpo/test_online_grpo_vllm_moe_e2e.py --trainer online --mode lora_ep2
"""

import argparse

from src.env import env_int, env_str
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_30B_A3B
from tests.common.online_grpo_e2e import modes_for, run_online_grpo_e2e

MODEL_NAME = env_str("HALO_TEST_ONLINE_GRPO_MOE_MODEL", QWEN3_30B_A3B)
SERVER_URL = env_str("VLLM_SERVER_URL") or "http://localhost:8000"
# Trainer-side NCCL weight-transfer port. A resume row rebinds it for phase 2, which phase 1's
# close_communicator makes available again on both ends.
GROUP_PORT = env_int("HALO_TEST_VLLM_GROUP_PORT", 51340)


def _row_group_port(base: int, trainer: str, mode: str, resume: bool, family: str) -> int:
    """A deterministic per-row port: 22 rows share one pass back-to-back, and a listener freed by
    one row sits in TIME_WAIT while the next binds — so every row owns two ports (phase 2 of a
    resume binds ``+1``)."""
    rows = [(t, m, r) for t in ("online", "sdpg") for m in modes_for(family) for r in (False, True)]
    return base + 2 * rows.index((trainer, mode, resume))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer", choices=("online", "sdpg"), required=True)
    parser.add_argument("--mode", choices=modes_for("moe"), required=True)
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
        group_port=_row_group_port(GROUP_PORT, args.trainer, args.mode, args.resume, "moe"),
        model_name=MODEL_NAME,
    )


main = gpu_test_main(exact_world_size=2, prefix="online_grpo_vllm_moe_e2e")(run)

if __name__ == "__main__":
    main()
