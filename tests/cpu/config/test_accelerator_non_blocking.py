#!/usr/bin/env python
"""A parsed config copies its pinned batches to the device without blocking. transformers defaults
``accelerator_config.non_blocking`` to False, and a blocking copy waits for the kernels already queued. The
parser merges the key into a user's dict, never over a key they set or into a JSON file they name."""

import json

import pytest
import torch
from torch.utils.data import DataLoader
from transformers import Trainer
from trl import SFTConfig

from halo.configs.parser import _TOOLKIT_KEY_DEFAULTS, HaloArgumentParser
from halo.trainers.accelerator import build_plain_accelerator
from halo.trainers.dataloaders import DataParallelLoaderMixin

_ROWS = [{"x": torch.zeros(2)} for _ in range(4)]


def _config(tmp_path, body: str = "") -> SFTConfig:
    path = tmp_path / "config.yaml"
    path.write_text(f"output_dir: {tmp_path / 'out'}\nbf16: false\nuse_liger_kernel: false\n{body}")
    parser = HaloArgumentParser((SFTConfig,))
    (config,) = parser.parse_dict(parser.load_values([str(path)]))
    return config


def _toolkit_loader(config: SFTConfig) -> DataLoader:
    """The train loader the toolkit prepares on its own accelerator (any custom parallelism)."""
    host = object.__new__(DataParallelLoaderMixin)
    host.accelerator = build_plain_accelerator(config)
    host.dp_shard_geometry = lambda: (1, 0)
    return host._prepare_dataloader(DataLoader(_ROWS, batch_size=2, pin_memory=config.dataloader_pin_memory))


def _base_trainer_loader(config: SFTConfig) -> DataLoader:
    """The train loader the base Trainer prepares on its own accelerator (an accelerate-managed launch)."""
    return Trainer(model=torch.nn.Linear(2, 2), args=config, train_dataset=_ROWS).get_train_dataloader()


@pytest.mark.parametrize("prepare", [_toolkit_loader, _base_trainer_loader], ids=["toolkit", "base_trainer"])
def test_a_pinned_loader_copies_without_blocking_by_default(tmp_path, prepare):
    config = _config(tmp_path)
    assert config.dataloader_pin_memory, "premise: transformers pins host batches by default"

    assert prepare(config)._non_blocking is True


@pytest.mark.parametrize("prepare", [_toolkit_loader, _base_trainer_loader], ids=["toolkit", "base_trainer"])
@pytest.mark.parametrize("unpinned", ["dataloader_pin_memory: false\n", "use_cpu: true\n"], ids=["unpinned", "cpu"])
def test_an_unpinned_loader_keeps_blocking_copies(tmp_path, prepare, unpinned):
    """A non-blocking copy from pageable memory is staged and synchronous anyway."""
    config = _config(tmp_path, unpinned)
    assert not config.dataloader_pin_memory, "premise: the loader pins nothing"

    assert prepare(config)._non_blocking is False


def test_the_users_false_wins(tmp_path):
    config = _config(tmp_path, "accelerator_config:\n  non_blocking: false\n")

    assert _toolkit_loader(config)._non_blocking is False


def test_the_default_merges_into_the_users_dict(tmp_path):
    config = _config(tmp_path, "accelerator_config:\n  even_batches: false\n")

    assert config.accelerator_config.non_blocking is True
    assert config.accelerator_config.even_batches is False, "the user's own key was dropped by the merge"


def test_a_json_file_config_is_left_whole(tmp_path):
    path = tmp_path / "accelerator.json"
    path.write_text(json.dumps({"even_batches": False}))
    config = _config(tmp_path, f"accelerator_config: {path}\n")

    assert config.accelerator_config.non_blocking is False
    assert config.accelerator_config.even_batches is False


def test_every_declared_key_default_is_applied(tmp_path):
    """Derived from the registry, so a key default added later cannot ship unexercised."""
    config = _config(tmp_path)

    for (name, key), (expected, _) in _TOOLKIT_KEY_DEFAULTS.items():
        field = getattr(config, name)
        value = field.get(key) if isinstance(field, dict) else getattr(field, key)
        assert value == expected, f"toolkit default {name}.{key}={expected} was not applied"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
