# Qwen3.5-35B-A3B — node-local EP=8 SFT on UltraChat 200K, one 8-GPU Blackwell client.
# EP=8 (256 routed experts -> 32/rank) comes from the training config; nothing here overrides it.
# Submit: nomad job run launcher-configs/nomad/qwen3.5-35b-a3b-8gpu-ep.nomad.hcl
# Creds:  nomad var put nomad/jobs/halo-qwen35-35b-ep WANDB_API_KEY=... HF_TOKEN=...

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
  default = "examples/sft/qwen3_5/qwen3.5-35b-a3b-ultrachat-ep.yaml"
}

variable "model" {
  type        = string
  default     = "Qwen/Qwen3.5-35B-A3B"
  description = "Hub id, a local mirror, or a previous run's output_dir to chain stages."
}

variable "gpus" {
  type    = number
  default = 8
}

variable "memory_mb" {
  type        = number
  default     = 524288
  description = "Hard cgroup cap per task, shm_size included. An overshoot is an OOM-kill, not swapping."
}

job "halo-qwen35-35b-ep" {
  type = "batch"

  group "train" {
    count = 1

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

    # torchrun's rendezvous port, allocated by Nomad rather than hardcoded, so a second job on the
    # same client cannot collide with it. Host mode: the container shares the host netns for NCCL.
    network {
      mode = "host"
      port "rendezvous" {}
    }

    task "sft" {
      driver = "docker"

      config {
        image    = var.image
        work_dir = "/workspace"

        # /workspace ships in the image; `volumes` needs client docker.volumes.enabled = true.
        volumes = ["${var.scratch_host_path}:${var.scratch_host_path}"]

        # The container flags every other Halo surface passes (Makefile, dev container, SkyPilot).
        network_mode = "host"
        ipc_mode     = "host"
        shm_size     = 137438953472 # bytes = 128 GiB, and it counts against resources.memory

        ulimit {
          memlock = "-1"
          stack   = "67108864"
        }

        command = "bash"
        args = ["-lc", <<-EOT
          set -euo pipefail
          mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TMPDIR" "$OUTPUT_DIR"
          torchrun --nnodes=1 --nproc_per_node="$GPUS" --master_port="$NOMAD_PORT_rendezvous" \
            scripts/training/sft.py "$CONFIG_PATH" \
            --model_name_or_path="$MODEL" \
            --output_dir="$OUTPUT_DIR" \
            --project_name="$WANDB_PROJECT"
        EOT
        ]
      }

      env {
        CONFIG_PATH       = var.config_path
        MODEL             = var.model
        GPUS              = var.gpus
        OUTPUT_DIR        = "${var.scratch_host_path}/checkpoints/qwen35-35b-a3b-ep"
        WANDB_PROJECT     = "halo-qwen35-35b-ep"
        HF_HOME           = "${var.scratch_host_path}/hf"
        HF_DATASETS_CACHE = "${var.scratch_host_path}/hf/datasets"
        TMPDIR            = "${var.scratch_host_path}/tmp"
        HALO_DATA_ROOT    = var.scratch_host_path
      }

      # nomadVarExists first: a bare nomadVar BLOCKS forever on a missing path, which would strand the
      # alloc holding its GPUs. Each key is guarded too — an unset one renders the literal "<no value>".
      template {
        data        = <<-EOT
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
        EOT
        destination = "secrets/creds.env"
        env         = true
        change_mode = "noop"
      }

      resources {
        cpu    = 40000
        memory = var.memory_mb

        device "nvidia/gpu" {
          count = var.gpus
        }
      }
    }
  }
}
