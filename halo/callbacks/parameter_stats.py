"""Pre-training parameter and dtype summary. Counts are per rank (``local_numel``): on the wrapped model a plain
``numel()`` would report FSDP2/TP DTensors globally but FSDP-ignored EP experts locally."""

import logging
import re
from collections import defaultdict

import torch
from transformers import TrainerCallback

from halo.runtime.dtensor import local_numel
from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)


def count_model_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return this rank's (total_params, trainable_params)."""
    total_params = sum(local_numel(p) for p in model.parameters())
    trainable_params = sum(local_numel(p) for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def _normalize_module_name(module_name: str) -> str:
    """Collapse numeric indices so repeated layers group: ``model.layers.0.self_attn`` -> ``…layers.X…``."""
    if not module_name:
        return "<root>"
    return re.sub(r"\b\d+\b", "X", module_name)


def _module_trainable_stats(model: torch.nn.Module) -> dict[str, tuple[int, int]]:
    stats: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
    for param_name, param in model.named_parameters():
        numel = local_numel(param)
        tally = stats[_normalize_module_name(param_name)]
        tally[0] += numel
        tally[1] += numel if param.requires_grad else 0
    return {name: tuple(tally) for name, tally in stats.items()}


def _dtype_stats(model: torch.nn.Module) -> dict[torch.dtype, tuple[int, int, int]]:
    """``dtype -> (total, trainable, memory_bytes)``."""
    stats: defaultdict[torch.dtype, list[int]] = defaultdict(lambda: [0, 0, 0])
    for param in model.parameters():
        numel = local_numel(param)
        tally = stats[param.dtype]
        tally[0] += numel
        tally[1] += numel if param.requires_grad else 0
        tally[2] += numel * param.element_size()
    return {dtype: tuple(tally) for dtype, tally in stats.items()}


def _format_memory_size(bytes_size: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


class ParameterStatsCallback(TrainerCallback):
    """Log param counts, deduplicated module groups with trainable %, and a dtype breakdown with memory usage."""

    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs["model"]

        if not is_global_main_process():
            return

        total_params, trainable_params = count_model_parameters(model)
        percent = 100 * trainable_params / total_params if total_params > 0 else 0

        logger.info("\n===== Model view inside Trainer =====")
        logger.info(model)

        logger.info(f"Total number of parameters (this rank): {total_params:,}")
        logger.info(f"Number of trainable parameters (this rank): {trainable_params:,}")
        logger.info(f"Percentage of trainable parameters: {percent:.2f}%\n")

        logger.info("\n===== Model parameter statistics (per rank) =====")

        module_stats = _module_trainable_stats(model)
        module_stats_percent = []
        for mod_name, (mod_total, mod_trainable) in module_stats.items():
            mod_percent = 100 * mod_trainable / mod_total if mod_total > 0 else 0
            module_stats_percent.append((mod_name, mod_total, mod_trainable, mod_percent))

        module_stats_percent.sort(key=lambda x: x[3], reverse=True)

        logger.info("List of module groups with normalized names:")
        for mod_name, mod_total, mod_trainable, mod_percent in module_stats_percent:
            logger.info(f"  {mod_name:30s} - trainable: {mod_trainable:,} / {mod_total:,} ({mod_percent:.2f}%)")

        logger.info("\n===== Model parameter dtypes =====")

        dtype_stats = _dtype_stats(model)
        if dtype_stats:
            logger.info("Parameter dtypes breakdown:")
            total_memory = 0
            for dtype, (total_params, trainable_params, memory_bytes) in dtype_stats.items():
                total_memory += memory_bytes
                trainable_percent = 100 * trainable_params / total_params if total_params > 0 else 0
                logger.info(
                    f"  {str(dtype):15s} - total: {total_params:,} params, "
                    f"trainable: {trainable_params:,} ({trainable_percent:.2f}%), "
                    f"memory: {_format_memory_size(memory_bytes)}"
                )
            logger.info(f"Total parameter memory: {_format_memory_size(total_memory)}")

        logger.info("========================================\n")
