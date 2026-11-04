# Add-a-model — what to run

Templates worth cribbing: `families/bailing.py` (separate-GLU), `families/gpt_oss.py` (in-layer routing,
interleaved weights), `families/deepseek_v4.py` (most class attributes in one place).

## Verify

Lint on the host; everything else inside the Docker image (`halo:blackwell` / `:hopper`), where the
tools are on `PATH`. Never pin `--master_port` — the harness allocates one per run.

```bash
make lint
python tests/cpu/models/test_<name>_support.py            # registration, gather/merge, balancing
python tests/cpu/model_loading/test_liger_family_coverage.py    # every spec class exists and is exact
python tests/cpu/checkpoint/test_merge_ep_shards.py       # merged-from-shards == gathered for the family
python tests/cpu/conventions/test_tiny_family_roster.py   # the family's tiny model and manifest rows
python tests/cpu/parallelism/test_docs_limitation_tables.py  # the published support tables against the registries
torchrun --nproc_per_node=2 tests/gpu/parallelism/ep/test_ep_family_vs_reference.py --family <name>
torchrun --nproc_per_node=1 tests/gpu/kernels/test_liger_family_kernels.py   # each Liger role vs the family's own module
```

The family counts as supported once the tests of step 7 of
[Add EP support](../../agent-docs/models/adding-a-model.md#add-ep-support) and the Liger cases of
[Declare Liger coverage](../../agent-docs/models/adding-a-model.md#declare-liger-coverage) pass and its
docs rows exist (step 8); where you wired `bias_update`, also `tests/gpu/parallelism/ep/test_gptoss_bias_balancing.py`.
