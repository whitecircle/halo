# Qwen3-4B LoRA on UltraChat 200K — single GPU, the smallest job that exercises the whole path.
# The config's own header calls for `python scripts/training/sft.py`, not torchrun: one process,
# one GPU, no distributed init. Adapters are mergeable afterwards (agent-docs/reference/model-merging.md).
# Qwen3 rather than Qwen3.5 because examples/sft/qwen3_5/ holds no single-GPU or LoRA config.
# Submit: nomad job run launcher-configs/nomad/qwen3-4b-1gpu-lora.nomad.hcl
# Creds:  nomad var put nomad/jobs/halo-qwen3-4b-lora WANDB_API_KEY=... HF_TOKEN=...

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
  default = "examples/sft/qwen3/qwen3-4b-ultrachat-lora.yaml"
}

variable "model" {
  type    = string
  default = "Qwen/Qwen3-4B-Instruct-2507"
}

variable "memory_mb" {
  type        = number
  default     = 131072
  description = "Hard cgroup cap per task, shm_size included. An overshoot is an OOM-kill, not swapping."
}

job "halo-qwen3-4b-lora" {
  type = "batch"

  group "train" {
    count = 1

    # A training run that dies has consumed its GPU and its dataset position; restarting it silently
    # from step 0 wastes the slot. Fail the alloc and leave it dead for an operator to look at.
    restart {
      attempts = 0
      mode     = "fail"
    }

    reschedule {
      attempts  = 0
      unlimited = false
    }

    task "sft" {
      driver = "docker"

      config {
        image    = var.image
        work_dir = "/workspace"

        # /workspace ships in the image; `volumes` needs client docker.volumes.enabled = true.
        volumes = ["${var.scratch_host_path}:${var.scratch_host_path}"]

        # Single-process, so host IPC and host networking buy nothing. /dev/shm still has to hold the
        # dataloader workers' tensors — docker's 64 MB default is what breaks them — and the loader
        # pins that memory, so the memlock ceiling still has to come off.
        shm_size = 17179869184 # bytes = 16 GiB, and it counts against resources.memory

        ulimit {
          memlock = "-1"
        }

        command = "bash"
        args = ["-lc", <<-EOT
          set -euo pipefail
          mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TMPDIR" "$OUTPUT_DIR"
          python scripts/training/sft.py "$CONFIG_PATH" \
            --model_name_or_path="$MODEL" \
            --output_dir="$OUTPUT_DIR" \
            --project_name="$WANDB_PROJECT"
        EOT
        ]
      }

      env {
        CONFIG_PATH       = var.config_path
        MODEL             = var.model
        OUTPUT_DIR        = "${var.scratch_host_path}/checkpoints/qwen3-4b-ultrachat-lora"
        WANDB_PROJECT     = "halo-qwen3-4b-lora"
        HF_HOME           = "${var.scratch_host_path}/hf"
        HF_DATASETS_CACHE = "${var.scratch_host_path}/hf/datasets"
        TMPDIR            = "${var.scratch_host_path}/tmp"
        HALO_DATA_ROOT    = var.scratch_host_path
      }

      # nomadVarExists first: a bare nomadVar BLOCKS forever on a missing path, which would strand the
      # alloc holding its GPU. Each key is guarded too — an unset one renders the literal "<no value>".
      template {
        data        = <<-EOT
          {{- if nomadVarExists "nomad/jobs/halo-qwen3-4b-lora" }}
          {{- with nomadVar "nomad/jobs/halo-qwen3-4b-lora" }}
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
        cpu    = 8000
        memory = var.memory_mb

        device "nvidia/gpu" {
          count = 1
        }
      }
    }
  }
}
