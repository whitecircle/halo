"""Job-wide DeepEP knobs, read once from the environment.

Imports no backend and no process group, so the rank-uniform env check reads them without loading
DeepEP, and a test reloads this module alone to re-resolve them.
"""

from halo.env import env_flag, env_int, env_positive_int

# 0 restores the per-layer capacity all-reduce, for a model whose later MoE layers dispatch more
# tokens than its first.
EP_CAPACITY_DEDUP = env_flag("HALO_EP_CAPACITY_DEDUP", True)
# Device-side spin budget for the dispatch/combine NVLink barrier, in seconds (DeepEP's own default
# is 100). The clock starts when a rank enters the barrier, so it bounds rank skew rather than idle
# time; save paths and teardown first align ranks on host-blocking collectives, under the larger
# process-group timeout. Raise it only for a legitimately skewed phase that can outlast it.
DEEPEP_DEFAULT_GPU_TIMEOUT_S = 100
DEEPEP_GPU_TIMEOUT_S = env_positive_int("HALO_DEEPEP_GPU_TIMEOUT_SECONDS", DEEPEP_DEFAULT_GPU_TIMEOUT_S)
# Dispatch/combine SM count (0 = auto), honoured by both backends.
DEEPEP_NUM_SMS = env_int("HALO_DEEPEP_NUM_SMS", None) or 0
# RDMA queue-pair count (0 = auto); more QPs help the latency-bound all-to-all on EFA Gin. The
# elastic backend only: V1 takes no per-call QP count, so the legacy backend warns instead.
DEEPEP_NUM_QPS = env_int("HALO_DEEPEP_NUM_QPS", None) or 0
# SM count when get_theoretical_num_sms divides by zero on an inter-node Blackwell topology.
DEEPEP_DEFAULT_INTERNODE_NUM_SMS = 24
