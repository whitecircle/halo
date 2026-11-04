#!/usr/bin/env python
"""TP+DP keeps ``test_tp_correctness.py``'s loss and grad-norm equivalence at dp>1.

FSDP2 turns leftover plain tensors (TP replicas, per-head ``q_norm``/``k_norm``) into 1-D ``dp`` DTensors, so a
grad-norm bucket or sync reading the tensor type skips them, leaving each on ``1/tp_size`` of its gradient.
"""

import sys

from tests.gpu.parallelism.tp.test_tp_correctness import main

if __name__ == "__main__":
    sys.exit(main())
