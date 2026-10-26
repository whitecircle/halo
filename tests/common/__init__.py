"""Shared test modules for tests/, one concern per module.

Usage from any test file:
    from tests.common.rank_log import log, log_all
    from tests.common.memory import cleanup_memory, gpu_mem_gb
    from tests.common.distributed import init_distributed, teardown_distributed
    from tests.common.models import MODEL_CONFIGS, GPT_OSS_20B
    from tests.common.datasets import create_sft_dataset, create_preference_dataset
"""
