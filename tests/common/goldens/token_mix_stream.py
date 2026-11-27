"""Token mix stream golden: the source and window each sample of a fixed mix over the recorded datatrove folders reads,
and a digest of the rows. A checkpoint resumes by sample index, so a change to the order resumes runs onto other rows
and comes with a raised ``ORDER_VERSION``."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from halo.data.sources.token_mix import ORDER_VERSION, PATTERN_PERIOD, TokenMix, TokenSource
from tests.common.datatrove_shards import write_recorded_folder
from tests.common.goldens import regen_main

GOLDEN_PATH = Path(__file__).with_suffix(".json")
# Three periods: the pattern's repeat, and hundreds of passes over both sources' few windows.
SAMPLES = 3 * PATTERN_PERIOD
# Samples written out as '<source>:<window>'; the digests hold the rest.
SHOWN_SAMPLES = 64
MAX_LENGTH = 8
SEED = 1234
# Each source is one task's file of the recorded 'web' folder, with its weight: one tokenizer, so one token width.
TASK_FILES = {"web-0": ("00000_unshuffled.ds", 7.0), "web-1": ("00001_unshuffled.ds", 3.0)}


def render() -> str:
    with tempfile.TemporaryDirectory() as scratch:
        sources = [
            TokenSource(name, str(write_recorded_folder("web", Path(scratch) / name, data_file=data_file)), weight)
            for name, (data_file, weight) in TASK_FILES.items()
        ]
        mix = TokenMix(sources, max_length=MAX_LENGTH, num_steps=SAMPLES, rows_per_step=1, seed=SEED)
        located = [mix.locate(index) for index in range(SAMPLES)]
        rows = hashlib.blake2b(digest_size=16)
        for index in range(SAMPLES):
            rows.update(json.dumps(mix[index], separators=(",", ":")).encode())
        record = {
            "order_version": ORDER_VERSION,
            "config": {
                "weights": {name: weight for name, (_, weight) in TASK_FILES.items()},
                "max_length": MAX_LENGTH,
                "seed": SEED,
                "samples": SAMPLES,
            },
            "quotas": mix.quotas,
            "first_samples": [f"{sources[source].name}:{window}" for source, window in located[:SHOWN_SAMPLES]],
            "located_digest": hashlib.blake2b(json.dumps(located).encode(), digest_size=16).hexdigest(),
            "rows_digest": rows.hexdigest(),
        }
    return json.dumps(record, indent=1) + "\n"


if __name__ == "__main__":
    regen_main(GOLDEN_PATH, render)
