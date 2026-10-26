#!/usr/bin/env python
"""A training entry script end to end, launched as a user launches it: ``halo launch <method> <recipe> -n N``
on a local dataset, two steps saving each, then a resume to step three. A method generating through a live vLLM
server trains the dense tier's served checkpoint against it; every other one a tiny local model.

The first launch must save both steps complete. The resume must pick ``checkpoint-2``, leave the checkpoints
before it untouched (a restart from step 0 would rewrite them) and log steps one to three. The run must have had
a gradient, and the last checkpoint must load as the method's export (``from_pretrained``, or a
``SentenceTransformer``) with every weight it should hold, moved off the base model. One manifest row per
method; the row runs this file as one process (``launcher="python"``), and it starts the ranks.

    HALO_TEST_NPROC=2 python tests/gpu/trainers/other/test_entry_scripts.py --method preference/dpo
"""

import argparse
from pathlib import Path

from halo.env import env_int
from tests.common.entry_scripts import (
    CASES,
    FIRST_STEPS,
    RESUME_LOG,
    RESUMED_STEPS,
    SERVED_MODEL,
    build_tiny_model,
    checkpoint_dir,
    export_checks,
    first_launch_overrides,
    had_a_learning_signal,
    launch_command,
    resume_overrides,
    run_launch,
    state_stamps,
    trained_steps,
    write_run_config,
)
from tests.common.harness import gpu_test_main
from tests.common.ports import free_port
from tests.common.rank_log import log
from tests.common.weight_sync import VLLM_DENSE_SERVER_URL, weight_transfer_port

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--method", choices=sorted(CASES), required=True)
ARGS = parser.parse_args()
NPROC = env_int("HALO_TEST_NPROC", 2)


def _launch(config: Path, overrides: list[str], launch: int):
    """Launch number ``launch``; a method generating through an engine binds a weight-transfer port of its own."""
    server_fields = CASES[ARGS.method].server_fields
    if server_fields is not None:
        overrides = [*overrides, f"--{server_fields[1]}={weight_transfer_port('HALO_TEST_VLLM_GROUP_PORT', launch)}"]
    port = free_port() if NPROC > 1 else None
    return run_launch(launch_command(ARGS.method, config, nproc=NPROC, port=port, overrides=overrides))


@gpu_test_main(prefix="entry_scripts", partial_state=False, process_group=False)
def run(ctx):
    case = CASES[ARGS.method]
    scratch = Path(ctx.output_dir)
    served = case.server_fields is not None
    model = SERVED_MODEL if served else str(build_tiny_model(scratch / "model"))
    dataset_dir = scratch / "data"
    case.rows().save_to_disk(str(dataset_dir))
    config = write_run_config(
        ARGS.method, scratch, model=model, dataset_dir=str(dataset_dir), server_url=VLLM_DENSE_SERVER_URL
    )
    output = scratch / "output"

    log(f"{ARGS.method}: {FIRST_STEPS} steps on {NPROC} ranks")
    saved = range(1, FIRST_STEPS + 1)
    checks = {"first_launch_exits_cleanly": _launch(config, first_launch_overrides(), 0).returncode == 0}
    checks["first_launch_saves_every_step"] = all(
        (checkpoint_dir(output, step) / "trainer_state.json").is_file() for step in saved
    )
    if not all(checks.values()):
        return {"checks": checks}

    log(f"{ARGS.method}: resume to step {RESUMED_STEPS}")
    stamps = state_stamps(output, saved)
    resumed = _launch(config, resume_overrides(), 1)
    checks["resume_exits_cleanly"] = resumed.returncode == 0
    checks["resume_picks_the_last_checkpoint"] = f"{RESUME_LOG}{checkpoint_dir(output, FIRST_STEPS)}" in resumed.stdout
    checks["resume_leaves_the_saved_checkpoints_alone"] = state_stamps(output, saved) == stamps
    last = checkpoint_dir(output, RESUMED_STEPS)
    checks["resume_logs_every_step_once"] = (last / "trainer_state.json").is_file() and trained_steps(last) == list(
        range(1, RESUMED_STEPS + 1)
    )
    if checks["resume_logs_every_step_once"]:
        learned = had_a_learning_signal(last)
        checks["run_had_a_learning_signal"] = learned
        checks.update(export_checks(case.export, last, model, learned=learned))
    return {"checks": checks}


if __name__ == "__main__":
    run()
