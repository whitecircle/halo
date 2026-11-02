#!/usr/bin/env python
"""A model a trainer is handed as a path string, and a frozen reference or teacher, builds on the config the family
patches were applied to, whether or not a backend is pinned.

``resolve_trainer_model`` builds the model a trainer constructor was given as a path (SMPO, offline GRPO, a
teacher-distillation student); ``load_frozen_auxiliary_model`` builds every frozen reference and teacher. Zaya's
patches are process-wide and key on the family alone: the router's load recording, which its native
``balancing_biases`` buffer needs, and the gradient-checkpointing refusal (backward recompute through the CCA convs
faults in cuDNN); they are checked in a fresh interpreter, since an earlier test in this process may already have
installed them. Qwen3.8-Flash-Next's patches also stamp the config, and only a build handed that config holds its
n-gram table as a frozen host table instead of a trainable embedding. A build handed a config passes every keyword
it does not read to the model's ``__init__``, so ``model_init_kwargs`` entries that name config fields go onto the
config first.

    python tests/cpu/model_loading/test_trainer_path_family_patches.py
"""

import inspect
import json
import re
import subprocess
import sys
import types

import pytest
import torch
from transformers import PreTrainedModel, Qwen3Config

import scripts.training.embedding as embedding_script
from halo.configs.embedding import EmbeddingConfig
from halo.configs.script_arguments import ModelArguments
from halo.model_loading.parallel import whole_model
from halo.model_loading.parallel.frozen_models import load_frozen_auxiliary_model
from halo.models.qwen4_exp.ngram_table import HostNGramTable
from tests.common.qwen4_exp import NGRAM_TABLE_PATH, build_tiny_qwen4_exp_checkpoint
from tests.common.repo import REPO_ROOT
from tests.common.tiny_models import TINY_MOE_FAMILIES, build_tiny_family_checkpoint

_PROBE = """
import json, sys
from types import SimpleNamespace
import transformers.models.zaya.modeling_zaya as zaya_modeling
from halo.model_loading.parallel.whole_model import resolve_trainer_model
before = zaya_modeling.ZayaPreTrainedModel.supports_gradient_checkpointing
args = SimpleNamespace(model_init_kwargs=json.loads(sys.argv[2]), gradient_checkpointing=False)
model = resolve_trainer_model(sys.argv[1], args, parallelism_config=None)
print(json.dumps({
    "checkpointing_before": before,
    "checkpointing": type(model).supports_gradient_checkpointing,
    "load_recording": bool(getattr(zaya_modeling.ZayaRouter.forward, "_halo_load_recording", False)),
}))
"""
_PINNED_IDS = ["unpinned", "pinned"]


@pytest.fixture(scope="module")
def zaya_checkpoint(tmp_path_factory) -> str:
    directory = str(tmp_path_factory.mktemp("zaya"))
    build_tiny_family_checkpoint(TINY_MOE_FAMILIES["zaya"], directory)
    return directory


@pytest.fixture(scope="module")
def flash_next_checkpoint(tmp_path_factory) -> str:
    directory = str(tmp_path_factory.mktemp("qwen4_exp"))
    build_tiny_qwen4_exp_checkpoint(directory)
    return directory


def _assert_host_table(model: PreTrainedModel) -> None:
    """``model`` holds its n-gram table as the attached host table, with no n-gram parameter to train."""
    table = model.get_submodule(NGRAM_TABLE_PATH)
    assert isinstance(table, HostNGramTable), type(table).__name__
    assert table.is_attached
    assert [name for name, _ in model.named_parameters() if "ngram" in name] == []


@pytest.mark.parametrize("model_init_kwargs", [{}, {"attn_implementation": "eager"}], ids=_PINNED_IDS)
def test_a_trainer_path_zaya_load_installs_the_family_patches(zaya_checkpoint, model_init_kwargs):
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, zaya_checkpoint, json.dumps(model_init_kwargs)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    probe = json.loads(result.stdout.strip().splitlines()[-1])
    assert probe["checkpointing_before"] is not False, "premise: the fresh interpreter starts unpatched"
    assert probe == {
        "checkpointing_before": probe["checkpointing_before"],
        "checkpointing": False,
        "load_recording": True,
    }


@pytest.mark.parametrize("model_init_kwargs", [{}, {"attn_implementation": "sdpa"}], ids=_PINNED_IDS)
def test_a_trainer_path_flash_next_load_builds_its_host_table(flash_next_checkpoint, model_init_kwargs):
    """The trainer path builds on the stamped config, so the table is the host one, attached from the checkpoint
    every rank agreed on, and not a trainable embedding the size of the release's 95 GiB table."""
    args = types.SimpleNamespace(model_init_kwargs=dict(model_init_kwargs), gradient_checkpointing=False)
    _assert_host_table(whole_model.resolve_trainer_model(flash_next_checkpoint, args, parallelism_config=None))


@pytest.mark.parametrize("download_tag", [None, "reference_model"], ids=["single_process", "agreed_source"])
def test_a_frozen_flash_next_load_builds_its_host_table(flash_next_checkpoint, download_tag):
    """A frozen reference or teacher builds on its own stamped config too, and attaches the table from the directory
    it read the weights from."""
    model = load_frozen_auxiliary_model(
        flash_next_checkpoint,
        dtype=torch.float32,
        trust_remote_code=False,
        attn_implementation="sdpa",
        download_tag=download_tag,
    )
    _assert_host_table(model)


def test_model_init_kwargs_that_name_config_fields_reach_the_built_config(flash_next_checkpoint):
    """Entries naming a config field go onto the config the build receives, as ``from_pretrained`` applies them to a
    config it reads itself; handed to the build beside the config they would reach the model's ``__init__``, which
    takes no such keyword."""
    args = types.SimpleNamespace(
        model_init_kwargs={"use_cache": False, "dtype": "float32", "attn_implementation": "sdpa"},
        gradient_checkpointing=False,
    )
    model = whole_model.resolve_trainer_model(flash_next_checkpoint, args, parallelism_config=None)
    assert model.config.use_cache is False
    assert model.dtype == torch.float32
    _assert_host_table(model)


def test_the_load_arguments_kept_off_the_config_are_those_from_pretrained_reads_before_its_config():
    """``from_pretrained`` takes its own arguments out of its keywords before the config reads the rest, so the
    split keeps every one of them that also names a config field (on a config carrying every optional one) off the
    config: the dtype the load casts to and the quantization it applies."""
    source = inspect.getsource(PreTrainedModel.from_pretrained)
    arguments = set(inspect.signature(PreTrainedModel.from_pretrained).parameters) | set(
        re.findall(r'kwargs\.pop\("(\w+)"', source)
    )
    config = Qwen3Config()
    config.quantization_config = {"quant_method": "bitsandbytes"}
    assert {name for name in arguments if hasattr(config, name)} == whole_model._LOAD_ARGUMENTS_NAMING_CONFIG_FIELDS


def test_the_embedding_path_refuses_a_backbone_with_external_tables(flash_next_checkpoint, tmp_path):
    """sentence-transformers reads the backbone's config itself, so the stamp the family patches set never reaches
    its build: the load refuses before it, rather than train the table as an embedding on every rank."""
    runtime = types.SimpleNamespace(
        parallelism_config=types.SimpleNamespace(has_ep_group=False, is_tp_mode=False, max_concurrent_loading=None),
        model_source=flash_next_checkpoint,
    )
    with pytest.raises(ValueError, match="external tables"):
        embedding_script.build_sentence_transformer(
            runtime,
            EmbeddingConfig(output_dir=str(tmp_path), bf16=False, use_cpu=True),
            ModelArguments(model_name_or_path=flash_next_checkpoint),
            types.SimpleNamespace(reset_sinks=True, train_sinks=False),
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
