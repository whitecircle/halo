#!/usr/bin/env python
"""TP+DP correctness — the same loss / grad-norm equivalence as ``test_tp_correctness.py``, at dp>1.

The distinction matters because FSDP2 erases the signal the classification keys on. Under pure TP
everything ``tp_plan`` shards is a DTensor whose grad reduces itself; the leftovers are plain tensors
the type check reaches — a TP replica, and the per-head ``q_norm``/``k_norm`` whose gradient covers
only this rank's heads and must be SUMmed, not averaged. Once FSDP2 wraps the stage for DP, those
leftovers become 1-D ``dp`` DTensors too, so a grad-norm bucket or a replicated-grad sync that reads
the tensor type skips them — leaving each on ``1/tp_size`` of its gradient and mis-counting the
global norm. Only a dp>1 run exercises that.

Every rank receives the identical fixed batch, so the DP average reproduces the single-batch
gradient and the global norm must still equal the unsharded single-GPU baseline.

Run with 4 GPUs:
    torchrun --nproc_per_node=4 \
        tests/gpu/parallelism/tp/test_tp_dp_correctness.py
"""

import sys

from tests.gpu.parallelism.tp.test_tp_correctness import main

if __name__ == "__main__":
    sys.exit(main())
