"""The adapter versions on disk under ``rollout_adapter_dir``, which every rollout server reads at the trainer's path:
a version is published whole, and its directory is removed when the version leaves the servers."""

import json
import logging
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path

import torch
from safetensors.torch import save_file

from halo.checkpoint.format import ADAPTER_CONFIG_FILE, ADAPTER_SAFETENSORS_FILE

logger = logging.getLogger(__name__)


def publish_adapter_directory(root: str | Path, name: str, tensors: Mapping[str, torch.Tensor], config: dict) -> Path:
    """``tensors`` and ``config`` as the PEFT adapter directory ``root/name``, staged beside it and renamed into place, so
    an engine never reads a half-written one. Every user may read it, as a server may run as another user. A write that
    raises leaves no staging directory."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{name}.", dir=root))
    try:
        save_file(dict(tensors), staged / ADAPTER_SAFETENSORS_FILE)
        (staged / ADAPTER_CONFIG_FILE).write_text(json.dumps(config))
        for written in staged.iterdir():
            written.chmod(0o644)
        staged.chmod(0o755)
        published = root / name
        os.rename(staged, published)
    except BaseException:
        shutil.rmtree(staged, ignore_errors=True)
        raise
    return published


def remove_adapter_directories(root: str | Path, names: Iterable[str]) -> None:
    """Removes the published directory of each version in ``names``; one that was never published is skipped, and one
    that cannot be removed is logged and left."""
    for name in names:
        try:
            shutil.rmtree(Path(root) / name)
        except FileNotFoundError:
            continue
        except OSError as error:
            logger.warning(f"Could not remove the adapter version at {Path(root) / name}: {error}")
