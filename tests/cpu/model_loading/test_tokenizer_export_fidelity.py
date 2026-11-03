"""What ``configure_run_tokenizer`` may change about the exported tokenizer: ``save_pretrained`` writes the live
``model_max_length``, and transformers' default replaces the extra-special list."""

from __future__ import annotations

import ast
import json
import types

import pytest
import torch.nn as nn
from tokenizers import Tokenizer, models, pre_tokenizers
from torch.distributed.tensor import Shard, distribute_tensor
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from halo.configs.script_arguments import CommonScriptArguments
from halo.distributed.tensor_parallel.hf_plan import is_input_embedding_tp_sharded
from halo.model_loading.tokenizer_setup import configure_run_tokenizer, pristine_model_max_length
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from tests.common.distributed import fake_process_group_mesh
from tests.common.models import TINY_QWEN3_CONFIG
from tests.common.repo import REPO_ROOT

MODEL_CONTEXT = 8192
RUN_BUDGET = 512
CONTROL_TOKENS = ["<|im_start|>", "<|im_end|>"]


def _tokenizer() -> PreTrainedTokenizerFast:
    vocab = {token: index for index, token in enumerate(["<unk>", "<pad>", "<eos>", *CONTROL_TOKENS, "hi", "there"])}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>",
        extra_special_tokens=list(CONTROL_TOKENS),
        model_max_length=MODEL_CONTEXT,
    )


def _written_config(output_dir) -> dict:
    return json.loads((output_dir / "tokenizer_config.json").read_text(encoding="utf-8"))


def _saved_config(tokenizer, tmp_path) -> dict:
    tokenizer.save_pretrained(tmp_path)
    return _written_config(tmp_path)


def test_the_export_carries_the_tokenizers_own_bound(tmp_path):
    tokenizer = configure_run_tokenizer(
        CommonScriptArguments(), None, _tokenizer(), RUN_BUDGET, embeddings_sharded=is_input_embedding_tp_sharded
    )
    with pristine_model_max_length(tokenizer):
        assert _saved_config(tokenizer, tmp_path)["model_max_length"] == MODEL_CONTEXT
    assert tokenizer.model_max_length == RUN_BUDGET, "the run keeps truncating at its budget after a checkpoint"


def test_a_second_pin_does_not_overwrite_the_snapshot(tmp_path):
    """A script runs the seam once per model against one tokenizer; the second pass must not snapshot the first's
    budget."""
    tokenizer = _tokenizer()
    configure_run_tokenizer(
        CommonScriptArguments(), None, tokenizer, RUN_BUDGET, embeddings_sharded=is_input_embedding_tp_sharded
    )
    configure_run_tokenizer(
        CommonScriptArguments(), None, tokenizer, RUN_BUDGET, embeddings_sharded=is_input_embedding_tp_sharded
    )
    with pristine_model_max_length(tokenizer):
        assert _saved_config(tokenizer, tmp_path)["model_max_length"] == MODEL_CONTEXT


def test_a_processor_reaches_its_nested_tokenizer():
    """A VLM trainer holds the processor, whose save delegates to the pinned tokenizer."""
    tokenizer = configure_run_tokenizer(
        CommonScriptArguments(), None, _tokenizer(), RUN_BUDGET, embeddings_sharded=is_input_embedding_tp_sharded
    )
    processor = types.SimpleNamespace(tokenizer=tokenizer)
    with pristine_model_max_length(processor):
        assert tokenizer.model_max_length == MODEL_CONTEXT
    assert tokenizer.model_max_length == RUN_BUDGET


def test_an_unpinned_tokenizer_is_left_alone():
    """The GRPO family pins nothing unless both halves of its budget are bounded."""
    tokenizer = _tokenizer()
    with pristine_model_max_length(tokenizer):
        assert tokenizer.model_max_length == MODEL_CONTEXT
    assert tokenizer.model_max_length == MODEL_CONTEXT


class _Fsdp2SaveTrainer:
    """A plain FSDP2 run, the cheapest in-training writer that saves the tokenizer; the model is a bare ``Linear``."""

    _fsdp_wrapped = True
    _accelerate_manages_fsdp = False
    _pp_wrapper_state = None
    _writing_training_checkpoint = False
    save_sharded_ep = False
    save_max_shard_size = None

    def _ep_layers(self):
        return []

    def __init__(self, tokenizer, output_dir: str):
        self.model = nn.Linear(4, 4)
        self.processing_class = tokenizer
        self.args = types.SimpleNamespace(output_dir=output_dir)
        self.parallelism_config = types.SimpleNamespace(
            is_pp_mode=False,
            is_cp_mode=False,
            is_tp_mode=False,
            has_ep_group=False,
            tp_size=1,
            merge_expert_lora_on_save=False,
            expert_lora=None,
        )
        self._device_mesh = None
        self.accelerator = types.SimpleNamespace(process_index=0)

    save_model = DistributedTrainerMixin.save_model
    _write_model_payload = DistributedTrainerMixin._write_model_payload
    _checkpoint_context = DistributedTrainerMixin._checkpoint_context
    _find_cp_wrapper = DistributedTrainerMixin._find_cp_wrapper
    _get_tp_rank = DistributedTrainerMixin._get_tp_rank
    _mark_model_save_collectives_done = DistributedTrainerMixin._mark_model_save_collectives_done

    def _top_level_model(self):
        # The double runs unwrapped, so the top-level model is self.model.
        return self.model


def test_a_trainer_save_exports_the_models_own_bound(tmp_path):
    """``save_model`` is the one gate all in-training writers pass through. Driven through a real save, since a
    source check holds when the manager is misapplied."""
    tokenizer = configure_run_tokenizer(
        CommonScriptArguments(), None, _tokenizer(), RUN_BUDGET, embeddings_sharded=is_input_embedding_tp_sharded
    )
    # Premise: there is a pin to keep off disk.
    assert tokenizer.model_max_length == RUN_BUDGET

    _Fsdp2SaveTrainer(tokenizer, str(tmp_path)).save_model(str(tmp_path))

    assert _written_config(tmp_path)["model_max_length"] == MODEL_CONTEXT, (
        "the run's sequence budget was exported as the served context window"
    )
    assert tokenizer.model_max_length == RUN_BUDGET, "the run must keep truncating at its budget after the save"


def test_added_special_tokens_keep_the_checkpoints_own(tmp_path):
    tokenizer = configure_run_tokenizer(
        CommonScriptArguments(added_special_tokens=["<newtok>"]),
        None,
        _tokenizer(),
        RUN_BUDGET,
        embeddings_sharded=is_input_embedding_tp_sharded,
    )
    assert tokenizer.extra_special_tokens == [*CONTROL_TOKENS, "<newtok>"]
    for control in CONTROL_TOKENS:
        assert control in tokenizer.all_special_tokens, "the trainers' special-token masks read this set"

    with pristine_model_max_length(tokenizer):
        exported = _saved_config(tokenizer, tmp_path)["extra_special_tokens"]
    assert exported == [*CONTROL_TOKENS, "<newtok>"]


def test_re_adding_an_existing_special_token_does_not_duplicate_it():
    tokenizer = configure_run_tokenizer(
        CommonScriptArguments(added_special_tokens=[CONTROL_TOKENS[0]]),
        None,
        _tokenizer(),
        RUN_BUDGET,
        embeddings_sharded=is_input_embedding_tp_sharded,
    )
    assert tokenizer.extra_special_tokens == CONTROL_TOKENS


def test_growing_the_vocabulary_under_tp_is_refused():
    """``resize_token_embeddings`` cannot re-shard the DTensor; through the production predicate, so one that stops
    recognizing it drops the refusal."""
    tokenizer = _tokenizer()
    model = Qwen3ForCausalLM(Qwen3Config(**{**TINY_QWEN3_CONFIG, "vocab_size": len(tokenizer)}))
    embedding = model.get_input_embeddings()
    with fake_process_group_mesh(rank=0, world_size=2) as mesh:
        embedding.weight = nn.Parameter(
            distribute_tensor(embedding.weight.detach(), mesh, [Shard(0)], src_data_rank=None)
        )
        with pytest.raises(ValueError, match="patch_vocab.py"):
            configure_run_tokenizer(
                CommonScriptArguments(added_special_tokens=["<newtok>"]),
                model,
                tokenizer,
                RUN_BUDGET,
                embeddings_sharded=is_input_embedding_tp_sharded,
            )


def test_every_production_caller_passes_the_sharding_predicate():
    """``embeddings_sharded`` is a required keyword, but its ``TypeError`` fires only behind a whole model load, so
    this names such a call site first."""
    calls, missing = 0, []
    assert (REPO_ROOT / "halo").is_dir() and (REPO_ROOT / "scripts").is_dir(), "the sweep lost a root"
    for path in sorted((*(REPO_ROOT / "halo").rglob("*.py"), *(REPO_ROOT / "scripts").rglob("*.py"))):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == configure_run_tokenizer.__name__:
                calls += 1
                if "embeddings_sharded" not in {kw.arg for kw in node.keywords}:
                    missing.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert calls, f"the sweep found no call to {configure_run_tokenizer.__name__}"
    assert not missing, "these call sites lose the TP vocab-grow refusal: " + ", ".join(missing)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
