"""What the replica-start tests share: a tiny model whose copies the ranks must start alike, and the trainer's
parallel-mode ladder run on it without a trainer.

The base model is drawn from one shared seed, as a checkpoint load gives every rank the same weights; the LoRA
adapters on the attention projections and one row of the frozen input embedding (the row a vocabulary resize adds)
are drawn from each rank's own stream (``ADAPTER_SEED + rank``), as they are whenever the streams have parted before
the draw.
"""

from types import SimpleNamespace

import torch
from peft import LoraConfig, PeftModel, get_peft_model

from halo.trainers.parallelism.ep_layers import EPLayersMixin
from halo.trainers.parallelism.mode_setup import ModeSetupMixin
from tests.common.adapters import mixer_linear_leaves
from tests.common.tiny_models import TinyFamily, tiny_family_model

BASE_SEED = 0
# Each rank's adapter stream: ``ADAPTER_SEED + rank``.
ADAPTER_SEED = 100


class ModeLadder(ModeSetupMixin, EPLayersMixin):
    """The trainer's mode ladder on a bare model: the state ``_wrap_for_parallel_mode`` reads, and no clip."""

    _accelerate_manages_wrap = False

    def __init__(self, model, parallelism_config, ep_config=None):
        self.model = model
        self.parallelism_config = parallelism_config
        self._ep_config = ep_config
        self.args = SimpleNamespace(bf16=False, fp16=False)
        self._param_id_caches = {}

    def _top_level_model(self):
        return self.model

    def _install_ep_clip(self) -> None:
        pass

    def _install_qlora_clip(self) -> None:
        pass


def per_rank_lora_model(family: TinyFamily, rank: int) -> PeftModel:
    """``family``'s tiny model from :data:`BASE_SEED`, with LoRA on its attention projections (its token mixers'
    linears where the family names none) and its last input-embedding row drawn from this rank's own stream."""
    torch.manual_seed(BASE_SEED)
    model = tiny_family_model(family)
    targets = list(family.attention_targets) or mixer_linear_leaves(model)
    torch.manual_seed(ADAPTER_SEED + rank)
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=targets))
    with torch.no_grad():
        model.get_input_embeddings().weight[-1].normal_()
    return model


def per_rank_draws(model: PeftModel) -> list[torch.Tensor]:
    """The tensors :func:`per_rank_lora_model` drew per rank: every ``lora_A`` and the last input-embedding row."""
    lora_a = [param.data for name, param in model.named_parameters() if "lora_A" in name]
    return [*lora_a, model.get_input_embeddings().weight.data[-1]]
