# AI Tooling

The training images ship with skills that teach coding agents this codebase's
configs, limits, and footguns. [Claude Code](https://github.com/anthropics/claude-code)
itself is not installed by default — build with
`--build-arg INSTALL_CLAUDE_CODE=1` to bake it in, or install it in a running
container with `curl -fsSL https://claude.ai/install.sh | bash`.

## Using it

From a shell inside the container (bring your own subscription or API key):

```bash
cd /workspace && claude
```

Describe the task. Four skills — `parallelism`, `optimize`, `debug`, `docs` — fire on
their own when relevant. The other six are user-invoked only: name them with
`/launch`, `/checkpoints`, `/data`, `/add-model`, `/rl-setup`, `/write-tests`.

| Skill | What it does |
| --- | --- |
| `launch` | assembles and runs a training/test job — image, mounts, GPU count, detach |
| `parallelism` | produces a *valid* EP/CP/TP/ETP layout for your model and hardware, and rejects invalid ones before they cost a run |
| `optimize` | recommends throughput/memory levers for a config — and tells you which famous levers don't help at Halo's MoE shapes |
| `debug` | routes a hang, OOM, NaN, or NCCL/DeepEP fault to the right helper and fix |
| `checkpoints` | merging shards and adapters, converting, resuming, loading into vLLM |
| `data` | dataset format per method, and offline tokenize/pack/shard when worth it |
| `rl-setup` | wires up online / environmental GRPO: vLLM container, NCCL weight sync, environments |
| `add-model` | integrates a new model family (EP wrapper, registries, tests) |
| `write-tests` | writes CPU/GPU tests that actually fail when behavior breaks |
| `docs` | maintains both doc trees — `agent-docs/` and this guide — under the style charter |

## What the agent reads

Every skill is grounded in [`agent-docs/`](../agent-docs/README.md) ↗, the same
reference this guide points to. The skills themselves live in
[`skills/`](../skills/) ↗ at the repo root as plain Markdown in the
[Agent Skills](https://agentskills.io) format, readable by any tool. The
directory is tracked in the repo and copied into the images, so a local checkout
and the in-image agent both find it with no extra setup.

The rules the agent follows live in `CLAUDE.md`, with `AGENTS.md` symlinked to
it for other scaffolds.
