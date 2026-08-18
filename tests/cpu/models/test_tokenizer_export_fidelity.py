"""What ``setup_model_and_tokenizer`` may change about the tokenizer a run then EXPORTS.

The seam mutates the tokenizer for the run's benefit, and both mutations otherwise reach the
checkpoint:

  * ``model_max_length`` is pinned to the run's sequence budget because HF resolves every
    ``truncation=True, max_length=None`` call against it. ``save_pretrained`` writes the LIVE
    attribute into ``tokenizer_config.json`` — it adds the key to its own save set whether or not
    the tokenizer was constructed with one — so an SFT run at ``max_length: 40000`` exports a
    262k-context model whose served context is 40k, with nothing marking the cap as a training
    budget. ``pristine_model_max_length`` is what keeps the pin off disk.
  * ``--added_special_tokens`` on transformers' default REPLACES the extra-special list rather than
    extending it. Adding one token therefore drops every control token the checkpoint ships from
    ``all_special_ids`` (which the trainers build their special-token masks from) and from the
    exported ``tokenizer_config.json``.

Built on a real ``PreTrainedTokenizerFast`` rather than a stub, because the save format is exactly
what is under test.

    python tests/cpu/models/test_tokenizer_export_fidelity.py
"""

from __future__ import annotations

import ast
import json
import pathlib
import types

import pytest
import torch.nn as nn
from accelerate import PartialState
from tokenizers import Tokenizer, models, pre_tokenizers
from torch.distributed.tensor import Shard, distribute_tensor
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from src.args.common_script_args import CommonScriptArguments
from src.distributed.tensor_parallel.state_dict import input_embeddings_tp_sharded
from src.models.loading.tokenizer_setup import pristine_model_max_length, setup_model_and_tokenizer
from src.trainers.mixins.base import DistributedTrainerMixin
from tests.common.distributed import fake_process_group_mesh
from tests.common.models import TINY_QWEN3_CONFIG

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

PartialState()  # the setup seam logs through accelerate's logger

MODEL_CONTEXT = 8192
RUN_BUDGET = 512
CONTROL_TOKENS = ["<|im_start|>", "<|im_end|>"]


def _tokenizer() -> PreTrainedTokenizerFast:
    """A real fast tokenizer with the shape that matters: its own context bound and control tokens."""
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


# model_max_length: pinned for the run, never for the export


def test_the_export_carries_the_tokenizers_own_bound(tmp_path):
    tokenizer = setup_model_and_tokenizer(CommonScriptArguments(), None, _tokenizer(), RUN_BUDGET)
    with pristine_model_max_length(tokenizer):
        assert _saved_config(tokenizer, tmp_path)["model_max_length"] == MODEL_CONTEXT
    assert tokenizer.model_max_length == RUN_BUDGET, "the run keeps truncating at its budget after a checkpoint"


def test_a_second_pin_does_not_overwrite_the_snapshot(tmp_path):
    """A preference or distillation script runs the seam once per model against one tokenizer; the
    second pass must not record the first pass's budget as the tokenizer's own bound."""
    tokenizer = _tokenizer()
    setup_model_and_tokenizer(CommonScriptArguments(), None, tokenizer, RUN_BUDGET)
    setup_model_and_tokenizer(CommonScriptArguments(), None, tokenizer, RUN_BUDGET)
    with pristine_model_max_length(tokenizer):
        assert _saved_config(tokenizer, tmp_path)["model_max_length"] == MODEL_CONTEXT


def test_a_processor_reaches_its_nested_tokenizer():
    """A VLM run hands the trainer the processor, whose save delegates to the tokenizer the seam
    pinned — so the restore has to reach through it."""
    tokenizer = setup_model_and_tokenizer(CommonScriptArguments(), None, _tokenizer(), RUN_BUDGET)
    processor = types.SimpleNamespace(tokenizer=tokenizer)
    with pristine_model_max_length(processor):
        assert tokenizer.model_max_length == MODEL_CONTEXT
    assert tokenizer.model_max_length == RUN_BUDGET


def test_an_unpinned_tokenizer_is_left_alone():
    """The GRPO family pins nothing unless both halves of its budget are bounded, so there is
    nothing to keep off disk and nothing to restore."""
    tokenizer = _tokenizer()
    with pristine_model_max_length(tokenizer):
        assert tokenizer.model_max_length == MODEL_CONTEXT
    assert tokenizer.model_max_length == MODEL_CONTEXT


class _Fsdp2SaveTrainer:
    """Only the trainer surface ``save_model`` reads, with the real mixin methods bound to it.

    Shaped as a plain mixin-managed FSDP2 run — no EP/CP/TP/PP, no adapters — so
    ``select_checkpoint_saver`` picks ``save_fsdp2_checkpoint``: the cheapest of the in-training
    writers that saves the tokenizer, and the one an ordinary ``torchrun`` data-parallel run takes.
    The model is a bare ``Linear`` because only the tokenizer half of the write is under test.
    """

    _has_ep_layers = False
    _fsdp_wrapped = True
    _accelerate_manages_fsdp = False
    _pp_wrapper_state = None
    save_sharded_ep = False

    def __init__(self, tokenizer, output_dir: str):
        self.model = nn.Linear(4, 4)
        self.processing_class = tokenizer
        self.args = types.SimpleNamespace(output_dir=output_dir, save_max_shard_size=None)
        self.parallelism_config = types.SimpleNamespace(
            is_pp_mode=False,
            is_cp_mode=False,
            is_tp_mode=False,
            is_ep_tp_mode=False,
            tp_size=1,
            merge_expert_lora_on_save=False,
            expert_lora=None,
        )
        self.parallel_dims = types.SimpleNamespace(tp_local_rank=lambda: 0)

    save_model = DistributedTrainerMixin.save_model
    _checkpoint_context = DistributedTrainerMixin._checkpoint_context
    _persist_router_balancing_biases = DistributedTrainerMixin._persist_router_balancing_biases
    _find_cp_wrapper = DistributedTrainerMixin._find_cp_wrapper
    _get_tp_rank = DistributedTrainerMixin._get_tp_rank
    _mark_model_save_collectives_done = DistributedTrainerMixin._mark_model_save_collectives_done

    def _top_level_model(self):
        # The double runs unwrapped (no accelerate/DDP/compile), so the top-level model IS self.model.
        return self.model


def test_a_trainer_save_exports_the_models_own_bound(tmp_path):
    """``save_model`` is the one gate all in-training writers pass through (the strategies, the PEFT
    adapter saver, and the base Trainer fall-through), so the restore has to be on THAT path rather
    than merely available to it.

    Driven through a real save and read back off disk: the claim is about the number in the file the
    run ships, which a check that ``save_model``'s source mentions the context manager cannot make —
    it holds just as well when the manager is applied to the wrong object or after the write."""
    tokenizer = setup_model_and_tokenizer(CommonScriptArguments(), None, _tokenizer(), RUN_BUDGET)
    # Premise: there IS a pin to keep off disk, so the assertion below is not the tokenizer's default.
    assert tokenizer.model_max_length == RUN_BUDGET

    _Fsdp2SaveTrainer(tokenizer, str(tmp_path)).save_model(str(tmp_path))

    assert _written_config(tmp_path)["model_max_length"] == MODEL_CONTEXT, (
        "the run's sequence budget was exported as the served context window"
    )
    assert tokenizer.model_max_length == RUN_BUDGET, "the run must keep truncating at its budget after the save"


# added_special_tokens: a union, not a replacement


def test_added_special_tokens_keep_the_checkpoints_own(tmp_path):
    tokenizer = setup_model_and_tokenizer(
        CommonScriptArguments(added_special_tokens=["<newtok>"]), None, _tokenizer(), RUN_BUDGET
    )
    assert tokenizer.extra_special_tokens == [*CONTROL_TOKENS, "<newtok>"]
    for control in CONTROL_TOKENS:
        assert control in tokenizer.all_special_tokens, "the trainers' special-token masks read this set"

    with pristine_model_max_length(tokenizer):
        exported = _saved_config(tokenizer, tmp_path)["extra_special_tokens"]
    assert exported == [*CONTROL_TOKENS, "<newtok>"]


def test_re_adding_an_existing_special_token_does_not_duplicate_it():
    tokenizer = setup_model_and_tokenizer(
        CommonScriptArguments(added_special_tokens=[CONTROL_TOKENS[0]]), None, _tokenizer(), RUN_BUDGET
    )
    assert tokenizer.extra_special_tokens == CONTROL_TOKENS


def test_growing_the_vocabulary_under_tp_is_refused():
    """HF-native TP shards the embedding into a DTensor that ``resize_token_embeddings`` cannot re-shard;
    the seam names the pre-run tool instead of dying inside a torch sharding-strategy lookup.

    Through the production predicate, since the sharding-agnostic seam takes the verdict as a
    parameter: a predicate that stopped recognizing the DTensor would drop the refusal silently.
    """
    tokenizer = _tokenizer()
    model = Qwen3ForCausalLM(Qwen3Config(**{**TINY_QWEN3_CONFIG, "vocab_size": len(tokenizer)}))
    embedding = model.get_input_embeddings()
    with fake_process_group_mesh(rank=0, world_size=2) as mesh:
        embedding.weight = nn.Parameter(
            distribute_tensor(embedding.weight.detach(), mesh, [Shard(0)], src_data_rank=None)
        )
        with pytest.raises(ValueError, match="patch_vocab.py"):
            setup_model_and_tokenizer(
                CommonScriptArguments(added_special_tokens=["<newtok>"]),
                model,
                tokenizer,
                RUN_BUDGET,
                embeddings_sharded=input_embeddings_tp_sharded,
            )


def test_every_production_caller_passes_the_sharding_predicate():
    """The refusal above only fires where the caller supplies the verdict.

    ``setup_model_and_tokenizer`` is sharding-agnostic by construction (no ``torch.distributed``), so
    a call site that omits ``embeddings_sharded`` silently grows a TP-sharded embedding instead of
    naming ``patch_vocab.py``. Source-level because reaching the seam any other way needs a whole
    parallel model load.
    """
    missing = []
    for path in sorted((*(REPO_ROOT / "src").rglob("*.py"), *(REPO_ROOT / "scripts").rglob("*.py"))):
        if path == REPO_ROOT / "src/models/loading/tokenizer_setup.py":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            called = isinstance(node, ast.Call) and getattr(node.func, "id", None) == "setup_model_and_tokenizer"
            if called and "embeddings_sharded" not in {kw.arg for kw in node.keywords}:
                missing.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert not missing, "these call sites lose the TP vocab-grow refusal: " + ", ".join(missing)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
