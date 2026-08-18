"""Pre-training parameter / dtype summary.

Counts are per rank throughout (``local_numel``): the callback runs on the wrapped model, where a
plain ``numel()`` would report FSDP2/TP DTensor params globally while EP experts, which are
FSDP-ignored plain tensors, report locally, mixing two scopes in one sum.
"""

import re

import torch
from accelerate import PartialState
from transformers import TrainerCallback
from transformers.utils import logging

from src.distributed.runtime import local_numel

logger = logging.get_logger(__name__)


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
    """Per-module ``(total, trainable)`` counts, keyed by normalized name."""
    stats: dict[str, tuple[int, int]] = {}
    for param_name, param in model.named_parameters():
        total = local_numel(param)
        trainable = total if param.requires_grad else 0
        norm_name = _normalize_module_name(param_name)
        if norm_name in stats:
            prev_total, prev_trainable = stats[norm_name]
            stats[norm_name] = (prev_total + total, prev_trainable + trainable)
        else:
            stats[norm_name] = (total, trainable)
    return stats


def _dtype_stats(model: torch.nn.Module) -> dict[torch.dtype, tuple[int, int, int]]:
    """Parameter counts grouped by dtype: ``dtype -> (total, trainable, memory_bytes)``."""
    dtype_stats = {}

    for _param_name, param in model.named_parameters():
        dtype = param.dtype
        total_params = local_numel(param)
        trainable_params = total_params if param.requires_grad else 0

        memory_bytes = total_params * param.element_size()

        if dtype in dtype_stats:
            prev_total, prev_trainable, prev_memory = dtype_stats[dtype]
            dtype_stats[dtype] = (
                prev_total + total_params,
                prev_trainable + trainable_params,
                prev_memory + memory_bytes,
            )
        else:
            dtype_stats[dtype] = (total_params, trainable_params, memory_bytes)

    return dtype_stats


def _format_memory_size(bytes_size: float) -> str:
    """Human-readable byte count."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


class ParameterStatsCallback(TrainerCallback):
    """Log before training: total/trainable param counts (+%), deduplicated module groups with
    per-group trainable %, and a dtype breakdown with memory usage.
    """

    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs["model"]

        if not PartialState().is_main_process:
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
