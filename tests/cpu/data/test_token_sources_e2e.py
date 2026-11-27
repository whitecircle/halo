#!/usr/bin/env python
"""``scripts/training/sft.py`` on token_sources, end to end on two data-parallel CPU ranks with only the policy load
replaced by a tiny local model: each step trains its slice of the mix split across the ranks, each held-out folder is
scored as ``eval_<name>_loss``, every checkpoint carries the mix's fingerprint under the callback's name, a resume reads
the rows that follow, and a held-out folder holding an id past the model's vocabulary is refused before any step
trains."""

import datetime
import json
import pickle
import sys
from functools import partial
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from halo.checkpoint.format import TRAINER_STATE_FILE
from halo.data.sources.token_mix import TokenMix, TokenSource
from halo.distributed import fsdp, mesh
from halo.entrypoints.model_setup import PolicyLoad
from halo.model_loading.finalization import finalize_loaded_model
from tests.common.datatrove_shards import write_token_folder
from tests.common.gloo import run_gloo_ranks
from tests.common.offline_grpo import OFFLINE_VOCAB, make_offline_tokenizer
from tests.common.repo import load_script_module
from tests.common.tiny_models import TINY_DENSE_FAMILY, tiny_family_model

WORLD = 2
MAX_LENGTH = 16
BATCH = 2
GRAD_ACCUM = 2
ROWS_PER_STEP = BATCH * GRAD_ACCUM * WORLD
SAVE_STEPS = 2
FIRST_STEPS = 4
RESUMED_STEPS = 6
SEED = 0
# Each folder's name, token count and weight; a held-out folder takes none.
TRAINED = {"web": (900, 3.0), "code": (300, 1.0)}
HELD_OUT = {"web-val": (96, None), "code-val": (64, None)}
PG_TIMEOUT = datetime.timedelta(seconds=240)


def _write_folder(path: Path, tokens: int, rng, *, high: int | None = None) -> Path:
    """Documents of random lengths over the tiny vocabulary past its special ids; ``high`` ends every document."""
    cuts = np.unique(np.concatenate(([0, tokens], rng.integers(1, tokens, size=tokens // 9))))
    documents = [
        rng.integers(3, len(OFFLINE_VOCAB), size=end - begin).tolist()
        for begin, end in zip(cuts[:-1], cuts[1:], strict=True)
    ]
    if high is not None:
        documents = [[*document[:-1], high] for document in documents]
    return write_token_folder(path, {"00000.ds": documents}, token_bytes=2)


def _sources_yaml(key: str, folders: dict[str, Path]) -> str:
    entries = [
        f"  - {{name: {name}, path: {json.dumps(str(path))}"
        + ("}" if name in HELD_OUT else f", weight: {TRAINED[name][1]}}}")
        for name, path in folders.items()
    ]
    return f"{key}:\n" + "\n".join(entries) + "\n"


def _config(root: Path, output_dir: str, *, max_steps: int, held_out: dict[str, Path], resume: bool = False) -> str:
    trained = {name: root / "tok" / name for name in TRAINED}
    body = (
        f"model_name_or_path: {root / 'model'}\noutput_dir: {root / output_dir}\nmax_steps: {max_steps}\n"
        f"max_length: {MAX_LENGTH}\npacking: true\ntrain_sampling_strategy: sequential\n"
        f"train_on_completions_only: false\nper_device_train_batch_size: {BATCH}\n"
        f"gradient_accumulation_steps: {GRAD_ACCUM}\nper_device_eval_batch_size: {BATCH}\nsave_strategy: steps\n"
        f"save_steps: {SAVE_STEPS}\neval_strategy: steps\neval_steps: {SAVE_STEPS}\nlogging_steps: 1\nuse_cpu: true\n"
        f"bf16: false\nreport_to: none\ngradient_checkpointing: false\nuse_liger_kernel: false\nseed: {SEED}\n"
        f"disable_tqdm: true\n{_sources_yaml('token_sources', trained)}{_sources_yaml('eval_token_sources', held_out)}"
    )
    if resume:
        body += "resume_from_checkpoint: true\n"
    path = root / f"{output_dir}-{max_steps}.yaml"
    path.write_text(body)
    return str(path)


def _load_policy(context, _training_config, model_config, *_args, **_kwargs) -> PolicyLoad:
    """The policy from the weights the run resolved (the base, or the checkpoint it resumes), on the CPU."""
    model = AutoModelForCausalLM.from_pretrained(context.model_source)
    finalize_loaded_model(model)
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    return PolicyLoad(model, tokenizer, tokenizer, is_vlm=False)


def _main(module, config: str) -> dict:
    """One ``main()`` of the script on ``config``: the rows each optimizer step trained on this rank, by its zero-based
    index, or the error that ended it."""
    rows: dict[int, list[list[int]]] = {}
    run_trainer = module.run_trainer

    def recording_run(trainer, context, **kwargs):
        compute_loss = trainer.compute_loss

        def recording(model, inputs, *args, **loss_kwargs):
            if model.training:
                flat = inputs["input_ids"][0].tolist()
                step = rows.setdefault(trainer.state.global_step, [])
                step += [flat[start : start + MAX_LENGTH] for start in range(0, len(flat), MAX_LENGTH)]
            return compute_loss(model, inputs, *args, **loss_kwargs)

        trainer.compute_loss = recording
        run_trainer(trainer, context, **kwargs)

    with (
        mock.patch.object(module, "load_policy_model", side_effect=_load_policy),
        mock.patch.object(module, "run_trainer", side_effect=recording_run),
        mock.patch("halo.entrypoints.script.install_log_tee"),
        mock.patch.object(sys, "argv", ["sft.py", config]),
    ):
        try:
            module.main()
        except Exception as error:  # recorded for the test that expects it; the others read the rows
            return {"error": f"{type(error).__name__}: {error}", "rows": rows}
    return {"error": None, "rows": rows}


def _worker(rank: int, root: str, configs: dict[str, str]) -> None:
    """Every rank runs the script on each config in turn, on the CPU mesh."""
    module = load_script_module("scripts/training/sft.py", "halo_test_sft_token_sources_e2e")
    with mock.patch.object(fsdp, "build_dp_mesh", partial(mesh.build_dp_mesh, device_type="cpu")):
        runs = {label: _main(module, config) for label, config in configs.items()}
    with open(f"{root}/runs.{rank}.pkl", "wb") as handle:
        pickle.dump(runs, handle)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("token_sources_e2e")
    torch.manual_seed(SEED)
    tokenizer = make_offline_tokenizer()
    tiny_family_model(
        TINY_DENSE_FAMILY,
        overrides={"num_hidden_layers": 2, "vocab_size": len(OFFLINE_VOCAB), "pad_token_id": tokenizer.pad_token_id},
    ).save_pretrained(root / "model")
    tokenizer.save_pretrained(root / "model")
    rng = np.random.default_rng(SEED)
    for name, (tokens, _) in {**TRAINED, **HELD_OUT}.items():
        _write_folder(root / "tok" / name, tokens, rng)
    _write_folder(root / "tok" / "high", HELD_OUT["code-val"][0], rng, high=len(OFFLINE_VOCAB))
    held_out = {name: root / "tok" / name for name in HELD_OUT}
    # The trained folders' ids all fit, so only the script's hand-off of the held-out folders can raise in the last.
    configs = {
        "first": _config(root, "run", max_steps=FIRST_STEPS, held_out=held_out),
        "resumed": _config(root, "run", max_steps=RESUMED_STEPS, held_out=held_out, resume=True),
        "high id": _config(
            root, "high", max_steps=FIRST_STEPS, held_out={**held_out, "code-val": root / "tok" / "high"}
        ),
    }
    run_gloo_ranks(_worker, WORLD, str(root), configs, pg_timeout=PG_TIMEOUT)
    runs = []
    for rank in range(WORLD):
        with open(root / f"runs.{rank}.pkl", "rb") as handle:
            runs.append(pickle.load(handle))
    return root, runs


def _mix(root: Path) -> TokenMix:
    sources = [TokenSource(name, str(root / "tok" / name), weight) for name, (_, weight) in TRAINED.items()]
    return TokenMix(sources, max_length=MAX_LENGTH, num_steps=RESUMED_STEPS, rows_per_step=ROWS_PER_STEP, seed=SEED)


def _assert_steps_read_their_slices(mix: TokenMix, runs: list[dict], phase: str, steps: range) -> None:
    """Each step's rows over the ranks are its slice of the mix, each rank holding its own share."""
    assert sorted(runs[0][phase]["rows"]) == list(steps), phase
    for step in steps:
        held = sorted(row for run in runs for row in run[phase]["rows"][step])
        expected = sorted(mix[index]["input_ids"] for index in range(step * ROWS_PER_STEP, (step + 1) * ROWS_PER_STEP))
        assert held == expected, (phase, step)
        assert all(len(run[phase]["rows"][step]) == ROWS_PER_STEP // WORLD for run in runs), (phase, step)


def test_each_step_trains_its_slice_of_the_mix_split_across_the_ranks(run):
    root, runs = run
    assert [run["first"]["error"] for run in runs] == [None] * WORLD
    _assert_steps_read_their_slices(_mix(root), runs, "first", range(FIRST_STEPS))


def test_every_checkpoint_carries_the_fingerprint_under_the_callbacks_name(run):
    """The entry's key is the checkpoint format: a renamed callback would leave every saved checkpoint unresumable."""
    root, _ = run
    fingerprint = json.loads(json.dumps(_mix(root).fingerprint()))
    for step in range(SAVE_STEPS, RESUMED_STEPS + 1, SAVE_STEPS):
        state = json.loads((root / "run" / f"checkpoint-{step}" / TRAINER_STATE_FILE).read_text())
        assert state["stateful_callbacks"]["TokenMixCheckpointCallback"]["args"]["fingerprint"] == fingerprint, step


def test_each_held_out_folder_is_scored_at_every_evaluation(run):
    root, _ = run
    history = json.loads((root / "run" / f"checkpoint-{FIRST_STEPS}" / TRAINER_STATE_FILE).read_text())["log_history"]
    for name in HELD_OUT:
        scored = [entry["step"] for entry in history if f"eval_{name}_loss" in entry]
        assert scored == list(range(SAVE_STEPS, FIRST_STEPS + 1, SAVE_STEPS)), name


def test_a_resume_continues_from_the_saved_step_onto_the_rows_that_follow(run):
    root, runs = run
    assert [run["resumed"]["error"] for run in runs] == [None] * WORLD
    _assert_steps_read_their_slices(_mix(root), runs, "resumed", range(FIRST_STEPS, RESUMED_STEPS))


def test_a_held_out_folder_holding_an_id_past_the_vocabulary_is_refused_before_training(run):
    root, runs = run
    errors = [run["high id"]["error"] for run in runs]
    assert errors[0] == errors[1] and errors[0].startswith("ValueError: "), errors
    assert f"Token source folder {root / 'tok' / 'high'}: window" in errors[0]
    assert f"past the model's vocabulary of {len(OFFLINE_VOCAB)}" in errors[0]
    assert all(run["high id"]["rows"] == {} for run in runs)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
