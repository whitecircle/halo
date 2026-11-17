"""Workaround plugin (NOT part of Axolotl): initialize torch.distributed before model build.

Axolotl 0.19.0's ExpertParallelPlugin shards experts in post_model_build via _resolve_ep_group, which
returns None when torch.distributed is not initialized yet. In the pure-EP (expert_parallel_size ==
world_size, no fsdp_config) path nothing has initialized the process group at that point, so
shard_expert_weights() returns 0 and EP silently degrades to replicated DDP ("no Experts modules were
detected for sharding ... DeepEP dispatch/combine will run as a no-op"). Initializing the default NCCL
group here (torchrun env://) makes the EP plugin see the 2-rank group; accelerate reuses it later.
List this plugin BEFORE axolotl.integrations.expert_parallel.ExpertParallelPlugin.
"""

import os

import torch
import torch.distributed as dist
from axolotl.integrations.base import BasePlugin


class EPDistInitPlugin(BasePlugin):
    def pre_model_load(self, cfg):
        if not dist.is_initialized() and int(os.environ.get("WORLD_SIZE", "1")) > 1:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
            print(
                f"[ep_dist_init] initialized process group rank={dist.get_rank()} world={dist.get_world_size()}",
                flush=True,
            )
