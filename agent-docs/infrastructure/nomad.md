# Nomad Deployment

Batch job specs for running Halo training on an existing HashiCorp Nomad cluster. Specs live at `launcher-configs/nomad/*.nomad.hcl` — three jobs, tabled below. Adapt to other models by overriding `config_path` and `model` — and edit the task's `OUTPUT_DIR` / `WANDB_PROJECT` env, which are fixed per spec rather than variables.

Unlike [SkyPilot](skypilot.md), Nomad provisions nothing: it places containers on clients you already run. The GPU hosts, their drivers, and the scratch disk are yours to supply.

## Prerequisites

- **Nomad 1.4+** for the specs' core (the `device` block landed in 0.9, native service discovery in 1.3, Nomad Variables in 1.4). The two-node spec additionally uses group `max_run_duration`, **which needs Nomad 2.0.3+** — delete that line to run it on older clusters.
- **The `nomad-device-nvidia` device plugin on every GPU client.** It is not bundled with Nomad; download it from [releases.hashicorp.com/nomad-device-nvidia](https://releases.hashicorp.com/nomad-device-nvidia/), drop the binary in the client's `plugin_dir` (default `<data_dir>/plugins`), and enable it:

    ```hcl
    plugin "nomad-device-nvidia" {
      config {
        enabled = true
      }
    }
    ```

    The block label must equal the plugin binary's filename. Confirm the clients fingerprint their GPUs with `nomad node status -verbose <node>` — each GPU appears as `nvidia/gpu/<model>`.

- **The docker task driver**, plus the NVIDIA Container Toolkit registered with dockerd. The driver applies its `docker.nvidia_runtime` client option (default `nvidia`) when a task requests a GPU device, so the job specs need no `runtime` of their own.
- **`docker.volumes.enabled = true`** on the clients, for the scratch bind mount. It defaults to `false` and is the one client option the specs need that is restrictive out of the box; [Scratch volume](#scratch-volume) gives the host-volume alternative for clusters that keep it off.

Two docker plugin options look like blockers and are not. `allow_privileged` (default `false`) gates `privileged = true`, which none of these specs set. `allowed_modes` gates the `pid_mode`, `ipc_mode`, `userns_mode` and `uts_mode` task fields, and its `ipc` default — `["", "none", "host", "container", "private", "sharable"]` — already admits the `ipc_mode = "host"` the EP specs use. Only an operator who has narrowed `allowed_modes` has to widen it again.

The docker driver has **no** `gpus` field; `resources { device "nvidia/gpu" { count = N } }` is the only way to request GPUs.

## Submit

```bash
nomad job run launcher-configs/nomad/qwen3.5-35b-a3b-8gpu-ep.nomad.hcl

# retarget without editing — every variable takes -var (or NOMAD_VAR_<name>);
# scratch_host_path must be a verified large volume (see Scratch volume below)
nomad job run \
  -var 'scratch_host_path=/mnt/scratch' \
  -var 'model=/mnt/scratch/checkpoints/<your-stage-1-run>' \
  -var 'image=public.ecr.aws/whitecircle/halo:hopper' \
  launcher-configs/nomad/qwen3.5-35b-a3b-8gpu-ep.nomad.hcl
```

Check a spec before submitting. `nomad fmt -check` and `nomad job validate` both work **without a cluster** — validate falls back to in-process validation and says so. Treat it as a schema check either way: it does not meaningfully verify driver config even against a live server, so a nonsense `shm_size` still passes. `nomad job plan` needs a server and reports whether the placement is actually feasible.

```bash
nomad fmt -check launcher-configs/nomad/          # HCL syntax
nomad job validate <spec>                         # jobspec schema
nomad job plan <spec>                             # scheduler dry-run, no allocation
```

## Available jobs

| Spec | Layout | GPUs | Training config |
|---|---|---|---|
| `qwen3-4b-1gpu-lora.nomad.hcl` | single process, `python` | 1 | `examples/sft/qwen3/qwen3-4b-ultrachat-lora.yaml` |
| `qwen3.5-35b-a3b-8gpu-ep.nomad.hcl` | 1 node, EP=8 node-local | 8 | `examples/sft/qwen3_5/qwen3.5-35b-a3b-ultrachat-ep.yaml` |
| `qwen3.5-122b-a10b-2node-ep.nomad.hcl` | 2 nodes, EP=8 node-local + cross-node DP | 2 × 8 | `examples/sft/qwen3_5/qwen3.5-122b-a10b-ep.yaml` |

The LoRA job is the entry point: one GPU, one process, driven by a config whose own header calls for `python` rather than `torchrun`.

Shared variables: `image`, `scratch_host_path`, `config_path`, `model`, `memory_mb`. The 8-GPU job adds `gpus`; the two-node job adds `gpus_per_node`, `shared_filesystem`, `max_run_duration` and `rendezvous_timeout_s`.

**All three take EP size and scope from the training config**, and none pass `--expert_parallel_size` or `--ep_scope`. The 122B YAML pins `expert_parallel_size: 8`, and `ep_scope` resolves node-local on its own because the 8-rank EP group fits one NVLink domain. Forcing either on the command line would silently override a retargeted `config_path`.

## Multi-node rendezvous

The two-node spec is two groups — `rank0` and `rank1` — with a job-level `distinct_hosts` constraint so they land on different clients. `rank0` reserves a dynamic port, registers it as a Nomad-native service, and uses its own `NOMAD_IP_rendezvous` / `NOMAD_PORT_rendezvous` as the torchrun master address. `rank1` resolves that service through a `template`, keyed on the spec's `local.master_service` (`halo-qwen35-122b-master`):

```hcl
template {
  data        = <<-EOT
    {{- with nomadService "${local.master_service}" }}
    {{- with index . 0 }}
    MASTER_ADDR={{ .Address }}
    MASTER_PORT={{ .Port }}
    {{- end }}
    {{- end }}
  EOT
  destination = "local/rendezvous.env"
  change_mode = "noop"
}
```

Native service discovery (`provider = "nomad"`) means no Consul is required. `change_mode = "noop"` keeps a mid-run change to the service entry from restarting a training task.

**Instance 0 is selected explicitly.** `nomadService` returns a list, and a bare `range` emits one `MASTER_ADDR`/`MASTER_PORT` pair per registered instance — the shell sourcing that file keeps the *last* one. During a redeploy the outgoing and incoming rank 0 both hold registrations for the `shutdown_delay` window, so a `range` can point rank 1 at the master that is shutting down.

**The template is deliberately not `env = true`.** A `nomadService` query against an unregistered service renders an *empty file successfully* — it does not block — so reading it straight into the environment would start rank 1 with `MASTER_ADDR` unset instead of making it wait. Nomad keeps re-rendering the file as the service appears, so the entrypoint polls it and fails loud on a bounded timeout (`rendezvous_timeout_s`, default 1800 s) rather than idling on 8 GPUs. It guards `MASTER_PORT` alongside `MASTER_ADDR`, so a half-rendered file produces the same friendly failure rather than an `unbound variable` abort.

Shell variables in that entrypoint are written brace-less (`$MASTER_ADDR`, not `${MASTER_ADDR:-}`). Nomad applies its own `${…}` interpolation to task `args`, and a name it does not recognise is replaced with the empty string — which would make the guard always true and rank 1 unable to ever rendezvous. The script pre-initialises both variables instead, so `set -u` is satisfied without braces.

Both ranks run one shared script, branching on `NODE_RANK`. Beyond that variable and the rendezvous template, only rank 0 carries the port reservation, the service registration and a `shutdown_delay`.

### Gang scheduling

**Nomad has no gang / all-or-nothing scheduling — not in OSS, not in Enterprise, not in 2.0** ([hashicorp/nomad#18773](https://github.com/hashicorp/nomad/issues/18773)).

Placement is per-allocation and best-effort: whichever group fits starts immediately, and the other goes to a blocked evaluation. A two-node training job can therefore half-place, with rank 0 holding 8 GPUs while rank 1 waits for capacity. The spec bounds the damage rather than preventing it:

- `max_run_duration` on both groups caps the wall-clock a half-placed job can burn (Nomad 2.0.3+).
- `rendezvous_timeout_s` makes rank 1 exit loudly instead of waiting forever.
- `restart { attempts = 0 }` and `reschedule { attempts = 0 }` keep a failed run dead — a training job that silently restarts from step 0 wastes the node.

Run `nomad job plan` first on a busy cluster: it reports up front whether both groups can place.

## Scratch volume

`scratch_host_path` (default `/mnt`) is bind-mounted at the *same path* inside the container and holds the HF caches, `TMPDIR`, the toolkit scratch root and the checkpoints:

```hcl
volumes = ["${var.scratch_host_path}:${var.scratch_host_path}"]
```

`/mnt` is a convention, **not a guarantee of capacity** — on many hosts it shares the small root device. Verify with `findmnt` / `df -h` on the target clients and point the variable at the real large volume before submitting.

The repo ships the source at `/workspace` inside the image, so no code bind-mount is needed; `work_dir = "/workspace"` is enough.

Bind mounts outside the alloc directory need `docker.volumes.enabled = true` on the client. Where that is not allowed, declare a host volume on each client instead and swap the bind for a volume mount — no client-side docker change required:

```hcl
# client agent
client {
  host_volume "halo-scratch" {
    path = "/mnt/scratch"
  }
}

# group
volume "scratch" {
  type   = "host"
  source = "halo-scratch"
}

# task
volume_mount {
  volume      = "scratch"
  destination = "/mnt/scratch"
}
```

On a cluster without a shared filesystem the scratch path is per-client local disk, so the two-node spec sets `DIST_SHARED_FILESYSTEM=0` — each node saves and resumes its own checkpoint copy. Set `shared_filesystem=1` only when that path is one NFS/Lustre mount on both clients ([Filesystem handling](../data/filesystem-handling.md)).

## Credentials

`WANDB_API_KEY` and `HF_TOKEN` come from **Nomad Variables**, read by a `template` into the task environment. Workload identity grants every task read on `nomad/jobs/<job id>` by default, so no ACL policy is needed:

```bash
nomad var put nomad/jobs/halo-qwen35-35b-ep WANDB_API_KEY=... HF_TOKEN=...
```

**`nomadVar` blocks forever on a path that does not exist**, which would leave the alloc pending while it holds its GPU reservation. Every spec therefore gates on `nomadVarExists` first, and guards each key on its own — with `error_on_missing_key` at its `false` default, an unset key renders the literal string `<no value>`, so an unguarded line would export `HF_TOKEN=<no value>`:

```hcl
{{- if nomadVarExists "nomad/jobs/halo-qwen35-35b-ep" }}
{{- with nomadVar "nomad/jobs/halo-qwen35-35b-ep" }}
{{- if .WANDB_API_KEY }}
WANDB_API_KEY={{ .WANDB_API_KEY }}
{{- end }}
{{- if .HF_TOKEN }}
HF_TOKEN={{ .HF_TOKEN }}
{{- end }}
{{- end }}
{{- end }}
```

So a missing variable, or one holding only some of the keys, renders empty for the rest and the task still starts. `HF_TOKEN` is only needed for gated or private repos; `WANDB_API_KEY` is not optional in practice, because all three training configs pin `report_to: wandb` — without a key, add `--report_to=none` to the entrypoint or override the field.

## Mapping onto `docker run`

The specs carry the same container flags every other Halo surface passes (`CLAUDE.md`, the Makefile, the SkyPilot tasks):

| `docker run` | Nomad docker driver |
|---|---|
| `--gpus all` | `resources { device "nvidia/gpu" { count = N } }` |
| `--ipc=host` | `ipc_mode = "host"` |
| `--shm-size=128g` | `shm_size = 137438953472` (**bytes**) |
| `--ulimit memlock=-1 --ulimit stack=67108864` | `ulimit { memlock = "-1" stack = "67108864" }` (values are **strings**) |
| `--network host` | `network_mode = "host"` + group `network { mode = "host" }` |
| `-v $D:$D` | `volumes = ["$D:$D"]` |
| `-w /workspace` | `work_dir = "/workspace"` |
| `--env-file .env` | `template { … env = true }` over Nomad Variables |
| `-e HF_HOME=…` | `env { HF_HOME = … }` |
| *(no equivalent)* | `resources { cpu = … }` — reserved MHz; the specs hardcode 40000 (8-GPU) / 8000 (1-GPU). Lower it on a smaller client or the group will not place |
| *(no equivalent)* | `resources { memory = … }` — **required, and a hard cap** |

`--cap-add=SYS_PTRACE` (for [py-spy](../reference/debugging.md)) has no counterpart in the shipped specs: it needs `sys_ptrace` in the client's `allow_caps`, which is not in the default list. Add `cap_add = ["sys_ptrace"]` to the task config once the client permits it.

That last row is the one real divergence from `docker run`, which caps nothing by default. `resources.memory` is a **hard cgroup limit**, and the `shm_size` allocation counts against it — a 128 GiB `/dev/shm` inside a 256 GiB cap leaves only 128 GiB for the trainer. Overshooting is an OOM-kill, not swapping, and with `restart { attempts = 0 }` that kill ends the job. The specs default to 512 GiB for the 8-GPU jobs and 128 GiB for the single-GPU one, tunable via `memory_mb`; raise it rather than trimming `shm_size` on a large host.

The single-GPU job drops `ipc_mode` and `network_mode` — one process needs neither — but keeps a 16 GiB `shm_size`, because docker's 64 MB default is what breaks PyTorch dataloader workers, and keeps `memlock=-1` because those workers pin the memory they hand over.

## Job management

```bash
nomad job status halo-qwen35-35b-ep      # allocations, blocked evaluations
nomad alloc logs -f <alloc-id> sft       # stream training output
nomad alloc exec -task sft <alloc-id> bash
nomad job stop -purge halo-qwen35-35b-ep
```

Training output goes to Nomad's alloc logs; checkpoints go to `output_dir` on the scratch volume.

## Troubleshooting

- **`device "nvidia/gpu"` never places**: the device plugin is missing or disabled on the clients. `nomad node status -verbose <node>` must list `nvidia/gpu/…` devices.
- **Container starts but sees no GPU**: the NVIDIA Container Toolkit is not registered with dockerd, or the client's `allow_runtimes` excludes `nvidia`.
- **Task fails on the bind mount**: `docker.volumes.enabled` is `false` on the client — use the host-volume form above.
- **`ipc_mode = "host"` rejected**: an operator has narrowed the docker plugin's `allowed_modes.ipc` to exclude `host` (its default admits it). This is not the `allow_privileged` gate — the specs set no `privileged`.
- **Task OOM-killed early, or killed mid-checkpoint**: `resources.memory` is a hard cap that includes `shm_size`. Raise `memory_mb`.
- **Rank 1 times out at the rendezvous**: rank 0 never placed. `nomad job status` shows the blocked evaluation; see [Gang scheduling](#gang-scheduling).
- **NCCL or DeepEP faults once training starts**: not a Nomad problem — [Troubleshooting](../reference/troubleshooting.md), [DeepEP](deepep.md).

## Resources

- [Nomad job specification](https://developer.hashicorp.com/nomad/docs/job-specification) · [`device`](https://developer.hashicorp.com/nomad/docs/job-specification/device) · [`template`](https://developer.hashicorp.com/nomad/docs/job-specification/template) · [docker driver](https://developer.hashicorp.com/nomad/docs/job-declare/task-driver/docker)
- [NVIDIA device plugin](https://developer.hashicorp.com/nomad/plugins/devices/nvidia) · [Nomad Variables in jobs](https://developer.hashicorp.com/nomad/docs/job-declare/nomad-variables)
- [SkyPilot](skypilot.md) · [Multi-Node](../parallelism/multi-node.md) · [Expert Parallelism](../parallelism/expert-parallelism.md)
