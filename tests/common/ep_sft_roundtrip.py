"""The EP SFT round trip per-family suites share: load at ``ep_size = world``, train, save gathered, reload plain, and
match the loss."""

import math
import os
from collections.abc import Mapping
from types import MappingProxyType

import torch
from transformers import AutoTokenizer, PreTrainedModel
from trl import SFTConfig

from halo.configs.script_arguments import CommonScriptArguments
from halo.distributed.parallelism_config import ParallelismConfig
from halo.entrypoints.script import ScriptContext, configure_balancing_and_build_callbacks
from halo.model_loading.parallel.lazy_plan import is_lazy_loadable
from halo.model_loading.parallel.loader import load_distributed_model
from halo.models.module_tree import backbone_with_layers
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.checkpoint_io import fixed_batch_loss, safetensors_state_dict
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import ensure_model_downloaded, shared_output_dir
from tests.common.ep_reference import random_token_batch
from tests.common.memory import cleanup_memory
from tests.common.models import QWEN3_0_6B
from tests.common.rank_log import log
from tests.common.tiny_models import TINY_MOE_FAMILIES, shared_tiny_family_checkpoint
from tests.common.tolerances import TOL
from tests.common.training_checks import training_run_checks

SEED = 42
NUM_TRAIN_STEPS = 3
MAX_SEQ_LENGTH = 256


def _native_bias_on_disk(ep_layers: list, save_dir: str) -> dict[str, bool]:
    """Each EP layer's distinctive bias is written once at fp32; found by value, since a family's hub spelling differs
    from its module path."""
    written = safetensors_state_dict(save_dir)
    matches = [
        [t for t in written.values() if t.shape == bias.shape and torch.equal(t.float(), bias)]
        for bias in (ep.gate.e_score_correction_bias.detach().float().cpu() for ep in ep_layers)
    ]
    log(f"native bias on disk: {[[str(t.dtype) for t in found] for found in matches]}")
    return {
        "native_bias_written_once_per_layer": all(len(found) == 1 for found in matches),
        "native_bias_fp32_on_disk": all(len(found) == 1 and found[0].dtype == torch.float32 for found in matches),
    }


class EPSftRoundTrip:
    """One family's round trip: a subclass names the family and overrides what differs; each hook returns the checks it
    adds. ``native_bias_layer`` marks a family whose EP layers can adopt ``e_score_correction_bias``: a distinctive
    bf16-exact value must survive the save at fp32."""

    family: str
    ep_layer_cls: type
    num_ep_layers: int
    attn_implementation: str = "eager"
    load_liger_kernel: bool = False
    train_liger_kernel: bool = False
    composite_token_ids: Mapping[str, int] = MappingProxyType({})
    hub_conversion: bool = False
    native_bias_layer: int | None = None
    balancing_callbacks: bool = False
    reload_in_bf16: bool = False

    def after_load(self, model: PreTrainedModel, ep_layers: list) -> dict[str, bool]:
        return {}

    def after_train(self, ep_layers: list, device: torch.device) -> dict[str, bool]:
        """Checks after training, before the fixed-batch loss; may set state the save must carry."""
        return {}

    def after_save(self, ep_layers: list, base_dir: str, save_dir: str) -> dict[str, bool]:
        return {}

    def after_reload(self, reloaded: PreTrainedModel, ep_layers: list) -> dict[str, bool]:
        return {}

    def _pin_composite_token_ids(self, model: PreTrainedModel) -> None:
        model.config.update(dict(self.composite_token_ids))

    def run(self, ctx) -> dict:
        checks: dict[str, bool] = {}
        metrics: dict[str, float] = {}
        device = ctx.device
        tiny = TINY_MOE_FAMILIES[self.family]

        ensure_model_downloaded(QWEN3_0_6B, ctx.rank)
        tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
        base_dir = shared_tiny_family_checkpoint(
            ctx,
            tiny,
            f"{self.family}_sft_roundtrip",
            tokenizer,
            SEED,
            edit=self._pin_composite_token_ids if self.composite_token_ids else None,
        )
        # Rank 0's dir on every rank: the gathered save is written once and read back by every rank.
        save_dir = os.path.join(shared_output_dir(ctx), "trained")

        if self.hub_conversion:
            # The hub layout is lazy-loadable, so every rank takes the lazy route.
            checks["lazy_loading_admitted"] = is_lazy_loadable(base_dir) is True

        pc = ParallelismConfig(ep_size=ctx.world_size)
        model, _ = load_distributed_model(
            model_name_or_path=base_dir,
            parallelism_config=pc,
            dtype=torch.bfloat16,
            attn_implementation=self.attn_implementation,
            use_liger_kernel=self.load_liger_kernel,
            trust_remote_code=True,
        )
        ep_layers = [m for m in model.modules() if isinstance(m, self.ep_layer_cls)]
        checks["ep_layers_patched"] = len(ep_layers) == self.num_ep_layers
        if self.native_bias_layer is not None:
            checks["native_bias_adoptable"] = all(ep.can_adopt_native_balancing() for ep in ep_layers)
        checks |= self.after_load(model, ep_layers)
        vocab_size = model.config.get_text_config().vocab_size

        train_dataset = create_sft_dataset(16, tokenizer, seed=SEED)
        sft_config = SFTConfig(
            output_dir=ctx.output_dir,
            max_steps=NUM_TRAIN_STEPS,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            learning_rate=1e-5,
            warmup_steps=1,
            max_length=MAX_SEQ_LENGTH,
            bf16=True,
            gradient_checkpointing=False,
            use_liger_kernel=self.train_liger_kernel,
            logging_steps=1,
            eval_strategy="no",
            save_strategy="no",
            report_to=[],
            logging_nan_inf_filter=False,
            dataloader_num_workers=0,
            dataloader_drop_last=True,
            remove_unused_columns=False,
        )
        # A trainer built bare never enables balancing; the scripts' ``configure_balancing_and_build_callbacks`` does.
        context = ScriptContext(pc, ctx.local_rank, None, base_dir, DistributedSFTTrainer)
        callbacks = (
            configure_balancing_and_build_callbacks(CommonScriptArguments(), sft_config, model, context)
            if self.balancing_callbacks
            else None
        )
        trainer = DistributedSFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            parallelism_config=pc,
            callbacks=callbacks,
        )
        ctx.on_teardown(trainer.cleanup_ep)
        result = trainer.train()
        metrics["final_train_loss"] = result.training_loss
        checks |= training_run_checks(result, trainer, NUM_TRAIN_STEPS)
        checks |= self.after_train(ep_layers, device)
        if self.native_bias_layer is not None:
            # A zero-init bias reloads as zeros whether or not the save wrote it, so carry a value no init produces.
            with torch.no_grad():
                for offset, ep in enumerate(ep_layers):
                    ep.gate.e_score_correction_bias.copy_(torch.arange(ep.num_experts, device=device) * 0.125 + offset)

        ids, labels = random_token_batch(vocab_size, batch=2, seq=64, device=device, seed=SEED + 7)
        ep_loss = fixed_batch_loss(model, ids, labels)
        metrics["ep_loss_post_train"] = ep_loss
        checks["ep_loss_finite"] = math.isfinite(ep_loss)

        ctx.barrier()
        trainer.save_model(save_dir)
        ctx.barrier()
        checks |= self.after_save(ep_layers, base_dir, save_dir)
        if self.native_bias_layer is not None:
            checks |= _native_bias_on_disk(ep_layers, save_dir)

        reloaded = (
            tiny.load_class.from_pretrained(save_dir, dtype=torch.bfloat16, attn_implementation="eager")
            .to(device=device)
            .eval()
        )
        if self.reload_in_bf16:
            reloaded.to(torch.bfloat16)
        if self.native_bias_layer is not None:
            reloaded_bias = (
                backbone_with_layers(reloaded).layers[self.native_bias_layer].mlp.gate.e_score_correction_bias
            )
            checks["e_score_bias_roundtrip"] = torch.equal(
                reloaded_bias.float().cpu(), ep_layers[0].gate.e_score_correction_bias.float().cpu()
            )
        checks |= self.after_reload(reloaded, ep_layers)
        rl_loss = fixed_batch_loss(reloaded, ids, labels)
        metrics["reload_loss"] = rl_loss
        delta = abs(rl_loss - ep_loss)
        metrics["reload_loss_delta"] = delta
        log(f"EP loss {ep_loss:.6f} vs reloaded plain-HF loss {rl_loss:.6f} (|Δ|={delta:.2e})")
        # Only the EP forward's numerics separate them. An expert-layout bug here moves the loss by less than the
        # bound, so this pins the round-trip, not the expert layout.
        checks["reload_loss_matches"] = delta < TOL.parallel_vs_baseline_loss_abs

        del reloaded
        cleanup_memory()
        return {"checks": checks, "metrics": metrics}
