"""Common test utilities for tests/.

Usage from any test file:
    from tests.common.utils import log, log_all, cleanup_memory, gpu_mem_gb
    from tests.common.distributed import init_distributed, teardown_distributed
    from tests.common.models import MODEL_CONFIGS, GPT_OSS_20B
    from tests.common.datasets import create_sft_dataset, create_preference_dataset
"""
