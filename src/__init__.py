"""Package init: the process-wide defaults that must be set before torch/transformers import."""

import os

from src.env import env_int
from src.log import configure_root_logging

# Tokenizer parallelism deadlocks against dataset multiprocessing; setdefault so a user export wins.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
configure_root_logging()

import torch  # noqa: E402  after the logging setup above

# One CPU thread per rank: ranks share a host, compute is on GPU, so extra threads only contend.
torch.set_num_threads(env_int("HALO_TORCH_NUM_THREADS", 1))

import transformers  # noqa: E402

transformers.logging.set_verbosity_info()
transformers.logging.enable_default_handler()
transformers.logging.enable_explicit_format()

# Before any toolkit module imports a transformers modeling module (they bind the hub-kernel
# fallback factory at import): the device-aware dispatch shim.
import src.models.patches.kernel_dispatch  # noqa: E402, F401
