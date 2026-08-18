# Qwen3.5-122B-A10B — two-node topology: node-local EP=8 within each client, data-parallel across the
# pair. Experts stay on NVLink inside a node, so only the DP gradient sync crosses the fabric — no
# cross-node all-to-all, no GIN, no gdrdrv. EP size and scope come from the training config.
# Submit: nomad job run launcher-configs/nomad/qwen3.5-122b-a10b-2node-ep.nomad.hcl
# Creds:  nomad var put nomad/jobs/halo-qwen35-122b-ep WANDB_API_KEY=... HF_TOKEN=...
#
# Nomad has no gang scheduling, so the two groups are placed independently and a partial placement is
# possible: rank 0 can start while rank 1 waits for a free node. Rank 1's wait is bounded and both
# groups carry max_run_duration, so a half-placed job fails on a clock instead of holding GPUs
# forever. See agent-docs/infrastructure/nomad.md#gang-scheduling before submitting to a busy cluster.

variable "image" {
  type        = string
  default     = "public.ecr.aws/whitecircle/halo:blackwell"
  description = "Halo training image. Use halo:hopper on H100/H200 clients."
}

variable "scratch_host_path" {
  type        = string
  default     = "/mnt"
  description = "Host dir bind-mounted at the same path in-container (HF caches, TMPDIR, scratch root, checkpoints). /mnt is a convention, not a capacity guarantee — see agent-docs/infrastructure/nomad.md#scratch-volume."
}

variable "config_path" {
  type    = string
  default = "examples/sft/qwen3_5/qwen3.5-122b-a10b-ep.yaml"
}

variable "model" {
  type    = string
  default = "Qwen/Qwen3.5-122B-A10B"
}

variable "gpus_per_node" {
  type    = number
  default = 8
}

variable "memory_mb" {
  type        = number
  default     = 524288
  description = "Hard cgroup cap per task, shm_size included. An overshoot is an OOM-kill, not swapping."
}

variable "max_run_duration" {
  type        = string
  default     = "24h"
  description = "Wall-clock cap per group. Bounds the GPU-hours a half-placed job can burn."
}

variable "rendezvous_timeout_s" {
  type        = string
  default     = "1800"
  description = "How long rank 1 waits for rank 0 to register before failing. Raise on a busy cluster."
}

variable "shared_filesystem" {
  type        = string
  default     = "0"
  description = <<-EOT
    "0" because scratch_host_path is per-client local disk, so each node saves and resumes its own
    checkpoint copy. Set "1" only when that path is one shared NFS/Lustre mount on both clients.
  EOT
}

locals {
  master_service = "halo-qwen35-122b-master"

  # One script for both ranks; NODE_RANK selects the branch. Rank 1 polls the rendezvous file rather
  # than reading it as env, because a nomadService template renders empty instead of blocking —
  # agent-docs/infrastructure/nomad.md#multi-node-rendezvous has the full rationale.
  #
  # Shell variables are written brace-less ($VAR, not $${VAR}) on purpose: Nomad applies its own
  # ${...} interpolation to task args, and an unknown name there is replaced with the empty string.
  run_script = <<-EOT
    set -euo pipefail
    MASTER_ADDR=""
    MASTER_PORT=""
    if [ "$NODE_RANK" = "0" ]; then
      MASTER_ADDR="$NOMAD_IP_rendezvous"
      MASTER_PORT="$NOMAD_PORT_rendezvous"
    else
      deadline=$((SECONDS + RENDEZVOUS_TIMEOUT_S))
      while [ -z "$MASTER_ADDR" ] && [ "$SECONDS" -lt "$deadline" ]; do
        . "$NOMAD_TASK_DIR/rendezvous.env" 2>/dev/null || true
        [ -n "$MASTER_ADDR" ] || sleep 5
      done
      if [ -z "$MASTER_ADDR" ] || [ -z "$MASTER_PORT" ]; then
        echo "ERROR: rank 0 did not register ${local.master_service} within $RENDEZVOUS_TIMEOUT_S s." >&2
        echo "Nomad places the two groups independently; check 'nomad job status $NOMAD_JOB_NAME' for a blocked evaluation." >&2
        exit 1
      fi
    fi
    export MASTER_ADDR MASTER_PORT
    mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TMPDIR" "$OUTPUT_DIR"
    echo "node_rank=$NODE_RANK master=$MASTER_ADDR:$MASTER_PORT gpus=$GPUS_PER_NODE"
    torchrun --nnodes=2 --node_rank="$NODE_RANK" --nproc_per_node="$GPUS_PER_NODE" \
      --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
      scripts/training/sft.py "$CONFIG_PATH" \
      --model_name_or_path="$MODEL" \
      --output_dir="$OUTPUT_DIR" \
      --project_name="$WANDB_PROJECT"
  EOT

  common_env = {
    CONFIG_PATH          = var.config_path
    MODEL                = var.model
    GPUS_PER_NODE        = var.gpus_per_node
    OUTPUT_DIR           = "${var.scratch_host_path}/checkpoints/qwen35-122b-a10b-ep"
    WANDB_PROJECT        = "halo-qwen35-122b-ep"
    RENDEZVOUS_TIMEOUT_S = var.rendezvous_timeout_s

    HF_HOME           = "${var.scratch_host_path}/hf"
    HF_DATASETS_CACHE = "${var.scratch_host_path}/hf/datasets"
    TMPDIR            = "${var.scratch_host_path}/tmp"
    HALO_DATA_ROOT    = var.scratch_host_path

    DIST_SHARED_FILESYSTEM = var.shared_filesystem
    # A 122B load across two nodes outruns the default collective timeout.
    DIST_NCCL_TIMEOUT_MINUTES = "60"
  }

  # nomadVarExists first: a bare nomadVar BLOCKS forever on a missing path, which would strand the
  # alloc holding its GPUs. Each key is guarded too — an unset one renders the literal "<no value>".
  creds_template = <<-EOT
    {{- if nomadVarExists "nomad/jobs/halo-qwen35-122b-ep" }}
    {{- with nomadVar "nomad/jobs/halo-qwen35-122b-ep" }}
    {{- if .WANDB_API_KEY }}
    WANDB_API_KEY={{ .WANDB_API_KEY }}
    {{- end }}
    {{- if .HF_TOKEN }}
    HF_TOKEN={{ .HF_TOKEN }}
    {{- end }}
    {{- end }}
    {{- end }}
  EOT
}

job "halo-qwen35-122b-ep" {
  type = "batch"

  # Job level, so it spreads the two groups: each wants all 8 GPUs of one client.
  constraint {
    distinct_hosts = true
  }

  group "rank0" {
    count            = 1
    max_run_duration = var.max_run_duration

    # Let the service registration drain before the alloc disappears.
    shutdown_delay = "5s"

    # A training run that dies has consumed its GPUs and its dataset position; restarting it silently
    # from step 0 wastes the node. Fail the alloc and leave it dead for an operator to look at.
    restart {
      attempts = 0
      mode     = "fail"
    }

    reschedule {
      attempts  = 0
      unlimited = false
    }

    network {
      mode = "host"
      port "rendezvous" {}
    }

    # How rank 1 finds rank 0 — Nomad-native discovery, so no Consul is required.
    service {
      provider = "nomad"
      name     = local.master_service
      port     = "rendezvous"
    }

    task "sft" {
      driver = "docker"

      config {
        image    = var.image
        work_dir = "/workspace"

        # /workspace ships in the image; `volumes` needs client docker.volumes.enabled = true.
        volumes = ["${var.scratch_host_path}:${var.scratch_host_path}"]

        network_mode = "host"
        ipc_mode     = "host"
        shm_size     = 137438953472 # bytes = 128 GiB, and it counts against resources.memory

        ulimit {
          memlock = "-1"
          stack   = "67108864"
        }

        command = "bash"
        args    = ["-lc", local.run_script]
      }

      env = merge(local.common_env, { NODE_RANK = "0" })

      template {
        data        = local.creds_template
        destination = "secrets/creds.env"
        env         = true
        change_mode = "noop"
      }

      resources {
        cpu    = 40000
        memory = var.memory_mb

        device "nvidia/gpu" {
          count = var.gpus_per_node
        }
      }
    }
  }

  group "rank1" {
    count            = 1
    max_run_duration = var.max_run_duration

    restart {
      attempts = 0
      mode     = "fail"
    }

    reschedule {
      attempts  = 0
      unlimited = false
    }

    network {
      mode = "host"
    }

    task "sft" {
      driver = "docker"

      config {
        image        = var.image
        work_dir     = "/workspace"
        volumes      = ["${var.scratch_host_path}:${var.scratch_host_path}"]
        network_mode = "host"
        ipc_mode     = "host"
        shm_size     = 137438953472 # bytes = 128 GiB, and it counts against resources.memory

        ulimit {
          memlock = "-1"
          stack   = "67108864"
        }

        command = "bash"
        args    = ["-lc", local.run_script]
      }

      env = merge(local.common_env, { NODE_RANK = "1" })

      # Re-rendered as rank 0 registers; the entrypoint polls it. Instance 0 explicitly: a bare range
      # emits one pair per registered instance and the shell would keep the LAST, which during a
      # redeploy overlap is the outgoing master.
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

      template {
        data        = local.creds_template
        destination = "secrets/creds.env"
        env         = true
        change_mode = "noop"
      }

      resources {
        cpu    = 40000
        memory = var.memory_mb

        device "nvidia/gpu" {
          count = var.gpus_per_node
        }
      }
    }
  }
}
