"""Package init: the process-wide defaults that must be set before torch/transformers import."""

import os

from src.env import env_int
from src.log import configure_root_logging

# Tokenizer parallelism deadlocks against dataset multiprocessing; setdefault so a user export wins.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
configure_root_logging()

import torch  # noqa: E402  after the logging setup above

# One CPU thread per rank: ranks share a host and compute runs on the GPU, so extra threads contend.
torch.set_num_threads(env_int("HALO_TORCH_NUM_THREADS", 1))

import transformers  # noqa: E402

transformers.logging.set_verbosity_info()
transformers.logging.enable_default_handler()
transformers.logging.enable_explicit_format()

# Install the device-aware dispatch shim before any toolkit module imports a transformers modeling
# module, which binds the hub-kernel fallback factory at import time.
import src.models.patches.kernel_dispatch  # noqa: E402, F401
