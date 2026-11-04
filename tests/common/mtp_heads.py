"""Tiny models with a live MTP head on the real family classes, the rows that exercise it, a release-layout tail on disk,
and the entry script's config and corpus for a run with one."""

import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from datasets import Dataset, DatasetDict
from safetensors.torch import save_file
from torch import nn

from halo.checkpoint.reading import iter_checkpoint_tensors
from halo.kernels.logprobs import LABEL_IGNORE_INDEX
from halo.model_loading.liger.cross_entropy import head_token_ce_sum
from halo.models.config_access import text_config
from halo.models.module_tree import backbone_with_layers
from halo.models.mtp.head import MtpForwardTerm, attach_mtp_head_module, build_mtp_head, mtp_head_path
from halo.models.mtp.spec import MtpObjective, require_mtp_head_spec
from halo.models.param_roles import declared_fp32_pins, fp32_pin_matches
from tests.common.models import QWEN3_0_6B
from tests.common.ports import free_port
from tests.common.repo import REPO_ROOT
from tests.common.tiny_models import (
    TINY_DENSE_FAMILIES,
    TINY_MOE_FAMILIES,
    TINY_MOE_VLM_FAMILIES,
    TINY_NEMOTRON_H_NO_LATENT_FAMILY,
    build_tiny_family_checkpoint,
    tiny_family_model,
)
from tests.common.tokenizers import load_cached_tokenizer

MTP_TINY_FAMILIES = {
    "qwen3_5": TINY_DENSE_FAMILIES["qwen3_5"],
    "qwen3_5_moe_text": TINY_MOE_FAMILIES["qwen3_5_moe_text"],
    "qwen3_5_moe": TINY_MOE_VLM_FAMILIES["qwen3_5_moe"],
    "qwen3_next": TINY_MOE_FAMILIES["qwen3_next"],
    "glm4_moe_lite": TINY_MOE_FAMILIES["glm4_moe_lite"],
    "nemotron_h": TINY_MOE_FAMILIES["nemotron_h"],
    "nemotron_h_no_latent": TINY_NEMOTRON_H_NO_LATENT_FAMILY,
    "nemotron_h_omni": TINY_MOE_VLM_FAMILIES["nemotron_h_omni"],
}
# The GPU rows' shard layouts, each with the ranks it runs on. ``fsdp`` leaves a MoE's experts as its class builds them,
# ``fsdp_zero3`` and ``ep1`` take the EP layer at one rank per group, and all three hold every parameter as a DTensor.
MTP_GPU_MODES: dict[str, tuple[dict, int]] = {
    "single": ({}, 1),
    "fsdp": ({"use_grouped_gemm": False}, 2),
    "fsdp_zero3": ({"fsdp_reshard_after_forward": True}, 2),
    "ep": ({"ep_size": 2}, 2),
    "ep_etp": ({"ep_size": 2, "expert_tp_size": 2}, 4),
    "etp": ({"expert_tp_size": 2}, 2),
    # The fine-tune recipes' masters, which a head trains beside and ``head_only`` refuses.
    "ep_fp32_masters": (
        {"ep_size": 2, "fp32_non_ep_params": True, "ep_fp32_experts": True, "ep_fp32_router": True},
        2,
    ),
    "ep1": ({}, 2),
    "ep1_ignored": ({"fsdp_shard_ep1_experts": False}, 2),
    "ep1_fp32_router": ({"ep_fp32_router": True}, 2),
}
# The CPU named in place of the GPU: the loader's device map, the process group's backend and the shard mesh. The one
# script serves one process and gloo ranks.
_ENTRY_ON_CPU = """
import functools, runpy, sys
import torch
import torch.distributed as dist
from halo.distributed import fsdp
from halo.distributed.mesh import build_dp_mesh
from halo.model_loading.parallel import loader
loader.get_local_rank = lambda: "cpu"
fsdp.build_dp_mesh = functools.partial(build_dp_mesh, device_type="cpu")
init_process_group = dist.init_process_group
dist.init_process_group = lambda *args, **kwargs: init_process_group(**{**kwargs, "backend": "gloo"})
"""
_ENTRY = (
    _ENTRY_ON_CPU
    + """
sys.argv = ["sft.py", sys.argv[1]]
runpy.run_path("scripts/training/sft.py", run_name="__main__")
"""
)
# A GPU row's own run(), through the harness that launches it: what no CPU test of its parts reaches.
_GPU_ROW_ENTRY = (
    _ENTRY_ON_CPU
    + """
import trl
from tests.common import harness
trl.SFTConfig = functools.partial(trl.SFTConfig, use_cpu=True)
_ctx_init = harness.Ctx.__init__
def _ctx_on_cpu(self, *args, **kwargs):
    _ctx_init(self, *args, **kwargs)
    self.device = torch.device("cpu")
harness.Ctx.__init__ = _ctx_on_cpu
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""
)


def _eager_experts(model: nn.Module) -> None:
    for module in model.modules():
        if hasattr(module, "config"):
            module.config._experts_implementation = "eager"
    text_config(model.config)._experts_implementation = "eager"


def tiny_model_with_head(
    name: str, *, dtype: torch.dtype = torch.float64, overrides: dict | None = None
) -> tuple[nn.Module, nn.Module]:
    """Family ``name``'s seeded tiny model and its head, attached with the model's own init, the same weights on every
    call. No hook yet, so a caller's arm picks its objective."""
    torch.manual_seed(7)
    model = tiny_family_model(MTP_TINY_FAMILIES[name], overrides=overrides).to(dtype)
    _eager_experts(model)
    spec = require_mtp_head_spec(model.config)
    head = build_mtp_head(model, spec, device="cpu")
    torch.manual_seed(11)
    head.apply(backbone_with_layers(model)._init_weights)
    head.to(dtype)
    attach_mtp_head_module(model, head, spec)
    return model.train(), head


def install_term(model: nn.Module, head: nn.Module, **objective) -> MtpForwardTerm:
    return MtpForwardTerm(
        model,
        head,
        require_mtp_head_spec(model.config),
        MtpObjective(**objective),
        head_token_ce_sum(model, fused=False),
    )


def padded_rows(seed: int = 3) -> dict[str, torch.Tensor]:
    """Two rows with a masked prompt, one right-padded."""
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(4, 90, (2, 24), generator=generator)
    labels = input_ids.clone()
    labels[:, :5] = LABEL_IGNORE_INDEX
    labels[1, 20:] = LABEL_IGNORE_INDEX
    attention_mask = torch.ones_like(input_ids)
    attention_mask[1, 20:] = 0
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask, "use_cache": False}


def packed_rows(seed: int = 3) -> dict[str, torch.Tensor]:
    """One row of two documents restarting at 12, each first label masked as the packing collator does."""
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(4, 90, (1, 24), generator=generator)
    position_ids = torch.cat([torch.arange(12), torch.arange(12)])[None]
    labels = input_ids.clone()
    labels[position_ids == 0] = LABEL_IGNORE_INDEX
    return {"input_ids": input_ids, "labels": labels, "position_ids": position_ids, "use_cache": False}


def _release_layout(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Fused experts split into one tensor per expert, as the Qwen3.5, GLM-4 and Nemotron-H tails ship them."""
    split = {}
    for key, tensor in tensors.items():
        if key.endswith("experts.gate_up_proj"):
            prefix = key[: -len("gate_up_proj")]
            half = tensor.shape[1] // 2
            for index, expert in enumerate(tensor):
                split[f"{prefix}{index}.gate_proj.weight"] = expert[:half].clone()
                split[f"{prefix}{index}.up_proj.weight"] = expert[half:].clone()
        elif key.endswith(("experts.down_proj", "experts.up_proj")):
            prefix, _, bank = key.rpartition(".")
            for index, expert in enumerate(tensor):
                split[f"{prefix}.{index}.{bank}.weight"] = expert.clone()
        else:
            split[key] = tensor
    return split


def write_checkpoint_with_tail(
    name: str, target: Path, *, per_expert: bool, seed: int = 123, edit=None, overrides: dict | None = None
) -> dict:
    """Family ``name``'s trunk beside a seeded release-layout tail, saved bf16 as a release stores it; returns the tail
    as stored in the head's own (fused) keys. ``edit`` rewrites the stored tail's dict before the write."""
    torch.manual_seed(1)
    model = tiny_family_model(MTP_TINY_FAMILIES[name], overrides=overrides).to(torch.bfloat16)
    spec = require_mtp_head_spec(model.config)
    path = mtp_head_path(model, spec)
    head = build_mtp_head(model, spec, device="cpu")
    torch.manual_seed(seed)
    fused = {}
    pinned = fp32_pin_matches(head.state_dict(), declared_fp32_pins(model)) if spec.tail_keeps_fp32_pins else ()
    for key, tensor in head.state_dict().items():
        drawn = torch.randn_like(tensor.float()) * 0.02 if tensor.dim() > 1 else torch.ones_like(tensor.float())
        fused[f"{path}.{key}"] = drawn if key in pinned else drawn.to(torch.bfloat16)
    stored = _release_layout(fused) if per_expert else dict(fused)
    for key, copied in spec.trunk_copies:
        embedding = model.get_input_embeddings() if copied == "input" else model.get_output_embeddings()
        stored[f"{path}.{key}"] = embedding.weight.detach().clone()
    if edit is not None:
        edit(stored)
    target.mkdir(parents=True)
    trunk = {key: value.clone() for key, value in model.state_dict().items()}
    save_file({**trunk, **stored}, str(target / "model.safetensors"), metadata={"format": "pt"})
    model.config.save_pretrained(target)
    return fused


def mtp_mode_needs_bf16(mode: str) -> bool:
    """DeepEP's dispatch takes 2-byte tokens, and the stock AdamW a fp32 run selects refuses plain experts beside
    DTensors."""
    axes = MTP_GPU_MODES[mode][0]
    return (
        axes.get("ep_size", 1) > 1 or axes.get("expert_tp_size", 1) > 1 or axes.get("fsdp_shard_ep1_experts") is False
    )


def tokenized_rows(vocab_size: int, count: int, *, seed: int, length: int, prompt: int) -> Dataset:
    """``count`` random rows of ``length`` tokens, the first ``prompt`` masked, for a trainer with no tokenizing path."""
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(4, vocab_size, (count, length), generator=generator)
    labels = input_ids.clone()
    labels[:, :prompt] = LABEL_IGNORE_INDEX
    return Dataset.from_dict(
        {"input_ids": input_ids.tolist(), "labels": labels.tolist(), "attention_mask": [[1] * length] * count}
    )


def packed_token_rows(vocab_size: int, count: int, *, seed: int, length: int, documents: int) -> Dataset:
    """``count`` random rows of ``documents`` equal documents, as the packing pass leaves them for its collator."""
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(4, vocab_size, (count, length), generator=generator)
    return Dataset.from_dict(
        {"input_ids": input_ids.tolist(), "seq_lengths": [[length // documents] * documents] * count}
    )


def collate_tokenized_rows(features: list[dict]) -> dict[str, torch.Tensor]:
    return {
        key: torch.tensor([feature[key] for feature in features], dtype=torch.long)
        for key in ("input_ids", "labels", "attention_mask")
    }


def write_entry_corpus(path: Path) -> str:
    rows = [
        {"prompt": [{"role": "user", "content": f"What is {i} + {i}?"}, {"role": "assistant", "content": f"{2 * i}"}]}
        for i in range(16)
    ]
    DatasetDict({"train": Dataset.from_list(rows[:12]), "test": Dataset.from_list(rows[12:])}).save_to_disk(path)
    return str(path)


def write_entry_run(name: str, target: Path, corpus: str, *, steps: int, **fields) -> Path:
    """Family ``name``'s tiny base and a release-layout tail under ``target``, kept if a run already wrote them, and the
    CPU config training on them; returns the config's path. ``fields`` override it."""
    if not (target / "base").exists():
        build_tiny_family_checkpoint(MTP_TINY_FAMILIES[name], str(target / "base"), load_cached_tokenizer(QWEN3_0_6B))
        write_checkpoint_with_tail(name, target / "tail", per_expert=True)
    config = target / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model_name_or_path": str(target / "base"),
                "dataset": corpus,
                "output_dir": str(target / "run"),
                "attn_implementation": "sdpa",
                "assistant_message_template": "<|im_start|>assistant\n",
                "max_length": 64,
                "per_device_train_batch_size": 2,
                "learning_rate": 1e-3,
                "lr_scheduler_type": "constant",
                "logging_steps": 1,
                "report_to": "none",
                "max_steps": steps,
                "save_steps": steps,
                "bf16": False,
                "use_cpu": True,
                "use_liger_kernel": False,
                "gradient_checkpointing": False,
                "mtp_head_source": str(target / "tail"),
                **fields,
            }
        )
    )
    return config


def _run_on_cpu(
    entry: Path, source: str, *argv: str, ranks: int, env: dict | None = None
) -> subprocess.CompletedProcess:
    entry.write_text(source)
    launcher = (
        [sys.executable]
        if ranks == 1
        else [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={ranks}",
            f"--master_port={free_port()}",
        ]
    )
    return subprocess.run(
        [*launcher, str(entry), *argv],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "PYTHONPATH": str(REPO_ROOT),
            **(env or {}),
        },
        capture_output=True,
        text=True,
        check=False,
    )


def run_entry(config: Path, *, ranks: int = 1) -> subprocess.CompletedProcess:
    """The SFT entry script on ``config``, on one process or ``ranks`` gloo ranks. Never raises on a failed run."""
    return _run_on_cpu(config.parent / "entry.py", _ENTRY, str(config), ranks=ranks)


def run_gpu_row(
    script: str, *flags: str, work: Path, ranks: int = 2, env: dict | None = None
) -> subprocess.CompletedProcess:
    """A ``tests/gpu`` script's row on ``ranks`` gloo ranks, ``TMPDIR`` under ``work``. Never raises on a failed run."""
    scratch = work / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    return _run_on_cpu(
        work / "row_entry.py", _GPU_ROW_ENTRY, script, *flags, ranks=ranks, env={"TMPDIR": str(scratch), **(env or {})}
    )


def saved_tensors(checkpoint) -> dict[str, torch.Tensor]:
    return dict(iter_checkpoint_tensors(str(checkpoint)))


def moved_keys(saved: dict, reference: dict) -> list[str]:
    return [
        key
        for key, value in saved.items()
        if key in reference and not torch.equal(value.float(), reference[key].float())
    ]
