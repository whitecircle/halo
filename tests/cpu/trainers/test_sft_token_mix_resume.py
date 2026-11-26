#!/usr/bin/env python
"""``DistributedSFTTrainer`` trains on a token mix, a map-style torch dataset TRL's own type check refuses unless its
preparation is skipped; a stop right after a save commits leaves a checkpoint the resume accepts; and on per-node output
storage each node's checkpoints carry the mix's fingerprint, and a node holding another run's checkpoint fails every
rank's resume."""

import dataclasses
import datetime
import json
import os
from functools import partial
from unittest.mock import patch

import numpy as np
import pytest
import torch
from transformers import TrainerCallback
from trl import SFTConfig

from halo.callbacks.token_mix_checkpoint import TokenMixCheckpointCallback, find_saved_token_mix
from halo.checkpoint.format import TRAINER_STATE_FILE
from halo.checkpoint.parallel.resume_source import resolve_resume_checkpoint
from halo.configs.sft import SFTScriptArguments
from halo.data.collators.packing import DataCollatorWithPacking
from halo.data.sources.token_mix import TokenMix, TokenSource
from halo.distributed import fsdp, mesh
from halo.entrypoints.data_setup import disable_trl_dataset_prep, load_token_sources
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.datatrove_shards import write_token_folder
from tests.common.gloo import run_gloo_ranks
from tests.common.offline_grpo import OFFLINE_VOCAB, make_offline_tokenizer
from tests.common.parallelism import make_parallelism_config
from tests.common.tiny_models import TINY_DENSE_FAMILY, tiny_family_model

BATCH = 2
GRAD_ACCUM = 2
MAX_LENGTH = 16
MAX_STEPS = 6
SAVE_STEP = 3
PG_TIMEOUT = datetime.timedelta(seconds=240)


def _sources(root) -> list[TokenSource]:
    """Two folders of documents of random lengths over the tiny vocabulary, past the special ids."""
    rng = np.random.default_rng(0)
    sources = []
    for name, tokens, weight in (("web", 900, 3.0), ("code", 300, 1.0)):
        cuts = np.unique(np.concatenate(([0, tokens], rng.integers(1, tokens, size=tokens // 9))))
        documents = [
            rng.integers(3, len(OFFLINE_VOCAB), size=end - begin).tolist()
            for begin, end in zip(cuts[:-1], cuts[1:], strict=True)
        ]
        write_token_folder(root / name, {"00000.ds": documents}, token_bytes=2)
        sources.append(TokenSource(name, str(root / name), weight))
    return sources


def _mix(sources, world: int = 1, seed: int = 0) -> TokenMix:
    rows_per_step = BATCH * GRAD_ACCUM * world
    return TokenMix(sources, max_length=MAX_LENGTH, num_steps=MAX_STEPS, rows_per_step=rows_per_step, seed=seed)


def _script_args(sources) -> SFTScriptArguments:
    return SFTScriptArguments(
        train_on_completions_only=False, token_sources=[dataclasses.asdict(source) for source in sources]
    )


def _trainer(
    output_dir: str, mix: TokenMix, *, rank: int = 0, world: int = 1, first_callbacks: tuple = (), **training
) -> DistributedSFTTrainer:
    """``first_callbacks`` run ahead of the mix's own; ``training`` overrides fields of the run's config."""
    torch.manual_seed(0)
    model = tiny_family_model(
        TINY_DENSE_FAMILY, overrides={"num_hidden_layers": 2, "vocab_size": len(OFFLINE_VOCAB), "pad_token_id": 0}
    )
    tokenizer = make_offline_tokenizer()
    args = SFTConfig(
        output_dir=output_dir,
        use_cpu=True,
        bf16=False,
        max_length=MAX_LENGTH,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_steps=MAX_STEPS,
        logging_steps=1,
        save_strategy="steps",
        save_steps=SAVE_STEP,
        gradient_checkpointing=False,
        use_liger_kernel=False,
        dataloader_num_workers=0,
        train_sampling_strategy="sequential",
        packing=True,
        report_to="none",
        disable_tqdm=True,
        seed=0,
        **training,
    )
    disable_trl_dataset_prep(args)
    with patch.object(fsdp, "build_dp_mesh", partial(mesh.build_dp_mesh, device_type="cpu")):
        return DistributedSFTTrainer(
            model=model,
            args=args,
            train_dataset=mix,
            processing_class=tokenizer,
            data_collator=DataCollatorWithPacking(tokenizer),
            parallelism_config=make_parallelism_config(world_size=world, gpus_per_node=world, rank=rank),
            callbacks=[*first_callbacks, TokenMixCheckpointCallback(mix.fingerprint())],
        )


class _Stopped(Exception):
    """The run stopped as a kill at that point would stop it."""


class _StopAtSave(TrainerCallback):
    """Stops the run at step ``step``'s first save hook: the checkpoint's trainer state is committed, older checkpoints
    are rotated out, and no later hook has run."""

    def __init__(self, step: int):
        self.step = step

    def on_save(self, args, state, control, **kwargs):
        if state.global_step == self.step:
            raise _Stopped


def _stopped_worker(rank: int, root: str, sources: list[TokenSource]) -> None:
    """One rank trains to MAX_STEPS keeping one checkpoint, stops at the last save's first hook, and records what is left
    and whether the resume's check accepts the newest complete checkpoint."""
    mix = _mix(sources)
    output_dir = f"{root}/stopped"
    trainer = _trainer(output_dir, mix, first_callbacks=(_StopAtSave(MAX_STEPS),), save_total_limit=1)
    try:
        trainer.train()
        stopped = False
    except _Stopped:
        stopped = True
    checkpoint = resolve_resume_checkpoint(output_dir)
    try:
        load_token_sources(
            _script_args(sources), trainer.args, trainer.parallelism_config, resume_checkpoint=checkpoint
        )
        verdict = "accepted"
    except ValueError as error:
        verdict = str(error)
    result = {
        "stopped": stopped,
        "left": sorted(name for name in os.listdir(output_dir) if name.startswith("checkpoint-")),
        "resumes_from": os.path.basename(checkpoint),
        "verdict": verdict,
    }
    with open(f"{root}/stopped.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle)


def test_a_stop_right_after_a_save_commits_leaves_a_checkpoint_the_resume_accepts(tmp_path):
    """With one checkpoint kept, the stop leaves the newest alone on disk: its trainer state already holds the mix."""
    run_gloo_ranks(_stopped_worker, 1, str(tmp_path), _sources(tmp_path), pg_timeout=PG_TIMEOUT)
    result = json.loads((tmp_path / "stopped.json").read_text())
    assert result["stopped"], "premise: the run stopped at the last save"
    assert result["left"] == [f"checkpoint-{MAX_STEPS}"] == [result["resumes_from"]]
    assert result["verdict"] == "accepted"


def _per_node_worker(rank: int, root: str, sources: list[TokenSource]) -> None:
    """Rank ``rank`` is node ``rank``'s only rank and writes that node's own output directory. It trains, records
    whether each of its node's checkpoints carries the live mix, then resumes from its node's newest checkpoint, which
    on node 1 a run of another seed wrote under the same name."""
    os.environ["LOCAL_RANK"] = "0"
    mix = _mix(sources, world=2)
    output_dir = f"{root}/node{rank}/run"
    trainer = _trainer(output_dir, mix, rank=rank, world=2)
    trainer.train()
    carries = {
        name: find_saved_token_mix(os.path.join(output_dir, name)) == mix.fingerprint()
        for name in sorted(os.listdir(output_dir))
        if name.startswith("checkpoint-")
    }
    checkpoint = resolve_resume_checkpoint(output_dir)
    if rank == 1:
        state_path = os.path.join(checkpoint, TRAINER_STATE_FILE)
        with open(state_path, encoding="utf-8") as handle:
            state = json.load(handle)
        other_run = _mix(sources, world=2, seed=1).fingerprint()
        state["stateful_callbacks"]["TokenMixCheckpointCallback"]["args"]["fingerprint"] = other_run
        with open(state_path, "w", encoding="utf-8") as handle:
            json.dump(state, handle)
    try:
        load_token_sources(
            _script_args(sources), trainer.args, trainer.parallelism_config, resume_checkpoint=checkpoint
        )
        verdict = "accepted"
    except ValueError as error:
        verdict = str(error)
    with open(f"{root}/node{rank}.json", "w", encoding="utf-8") as handle:
        json.dump({"carries": carries, "verdict": verdict}, handle)


def test_each_node_saves_the_mix_and_one_node_holding_another_runs_checkpoint_fails_every_rank(tmp_path):
    """Per-node output storage: each node's save rank writes its own trainer state, and every rank reads its own
    node's copy of the checkpoint, so one node's verdict must reach the others."""
    run_gloo_ranks(
        _per_node_worker,
        2,
        str(tmp_path),
        _sources(tmp_path),
        pg_timeout=PG_TIMEOUT,
        env={"LOCAL_WORLD_SIZE": "1", "DIST_OUTPUT_SHARED_FILESYSTEM": "0"},
    )
    results = [json.loads((tmp_path / f"node{rank}.json").read_text()) for rank in range(2)]
    saved = {f"checkpoint-{step}": True for step in (SAVE_STEP, MAX_STEPS)}
    assert [result["carries"] for result in results] == [saved, saved]
    assert results[0]["verdict"] == results[1]["verdict"]
    assert (
        "Checking the token mix against the resumed checkpoint failed on 1 of 2 rank(s) [1]" in results[0]["verdict"]
    )
    assert "seed 1 → 0" in results[0]["verdict"]


def test_a_row_dataset_without_skipped_preparation_is_refused(tmp_path):
    """TRL would map over the mix as if it were a ``datasets.Dataset``."""
    mix = _mix(_sources(tmp_path))
    args = SFTConfig(output_dir=str(tmp_path / "run"), use_cpu=True, bf16=False, report_to="none")
    with pytest.raises(ValueError, match="map-style TokenMix of collated rows"):
        DistributedSFTTrainer(
            model=tiny_family_model(TINY_DENSE_FAMILY, overrides={"vocab_size": len(OFFLINE_VOCAB)}),
            args=args,
            train_dataset=mix,
            processing_class=make_offline_tokenizer(),
            parallelism_config=make_parallelism_config(world_size=1, gpus_per_node=1),
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
