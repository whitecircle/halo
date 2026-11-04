"""Process-wide defaults that must be set before torch and transformers import."""

import os

from halo.env import env_positive_int
from halo.log import configure_root_logging

# Tokenizer parallelism deadlocks against dataset multiprocessing; setdefault so a user export wins.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
configure_root_logging()

import torch  # noqa: E402  after the logging setup above

# One CPU thread per rank: ranks share a host and compute runs on the GPU, so extra threads contend.
torch.set_num_threads(env_positive_int("HALO_TORCH_NUM_THREADS", 1))

import transformers  # noqa: E402

transformers.logging.set_verbosity_info()
transformers.logging.enable_default_handler()
transformers.logging.enable_explicit_format()

# Must precede any transformers modeling import, which binds the hub-kernel fallback factory at import time.
import halo.kernels.hub_kernel_dispatch  # noqa: E402, F401
