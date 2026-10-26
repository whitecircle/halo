"""``CommonScriptArguments`` — the dataset / logging / callback knobs every training script shares."""

from dataclasses import dataclass, field
from typing import ClassVar

from src.config_validation import (
    RangeValidatedConfig,
    present,
    require_finite,
    require_int,
    require_positive,
    require_positive_int,
)
from src.env import torch_trace_dir
from src.model_loading.tokenizer_backend import TokenizerBackend
from src.models.moe_balancing import BalancingMode
from src.training.parser import is_null_string

_DEFAULT_PROJECT = "default-project"


@dataclass
class CommonScriptArguments(RangeValidatedConfig):
    # The tracking project a run lands in when the config names none, declared per script class so the
    # ``project_name`` field and its help stay declared once rather than redeclared in every subclass.
    PROJECT_NAME: ClassVar[str] = _DEFAULT_PROJECT

    dataset: str | list[str] = field(
        default="/path/to/dataset",
        metadata={
            "help": "Dataset source: an `org/name[:config][@split]` hub id, an `s3://` URI, a "
            "directory written by `save_to_disk`, or a single .jsonl/.json/.parquet/.arrow/.csv "
            "file. Can be a list of any of these."
        },
    )
    # Annotated `float`, not `float | list[float]`, though a per-dataset list is supported:
    # HfArgumentParser allows only `Optional[X]` in a union and refuses to build otherwise. YAML goes
    # through `parse_dict`, which assigns without argparse coercion, so a list reaches
    # `load_datasets` intact; it just cannot be spelled on the command line.
    dataset_ratio: float | None = field(
        default=None,
        metadata={
            "help": "How much of each dataset to take, between 0 and 1. A single value applies to "
            "every dataset; a YAML list gives one ratio per entry in `dataset`."
        },
    )
    test_size: float | None = field(
        default=None,
        metadata={
            "help": "Test split proportion, like 0.05; unset keeps the dataset's own split. Set on a dataset that "
            "already has a test split, it concatenates the two and re-splits, and it is ignored (warned) on a "
            "sharded or pre-processed dataset."
        },
    )
    project_name: str = field(
        default=_DEFAULT_PROJECT,
        metadata={
            "help": "Name of logging project (wandb or clearml). Becomes WANDB_PROJECT / "
            "CLEARML_PROJECT; leave it out to take the per-script default rather than nulling it."
        },
    )
    pad_token: str | None = field(default=None, metadata={"help": "Override for the tokenizer's pad token."})
    bos_token: str | None = field(default=None, metadata={"help": "Override for the tokenizer's bos token."})
    eos_token: str | None = field(default=None, metadata={"help": "Override for the tokenizer's eos token."})
    chat_template: str | None = field(
        default=None,
        metadata={
            "help": "Chat template: a path to a .jinja/.jinja2/.j2 file, or the template string itself. Unset uses "
            "the tokenizer's own template."
        },
    )
    force_chat_template: bool = field(
        default=False,
        metadata={"help": "Use chat_template even when the tokenizer ships its own."},
    )
    added_special_tokens: list[str] | None = field(
        default=None, metadata={"help": "Special tokens added to the tokenizer's existing extra-special set."}
    )
    tokenizer_backend: TokenizerBackend = field(
        default="hf",
        metadata={
            "help": "Text→ids backend: 'hf' or 'gigatoken' (optional extra; token IDs verified identical at startup)."
        },
    )
    tools_field: str | None = field(
        default=None,
        metadata={
            "help": "Dataset column holding tool definitions (a list of dicts or a JSON string) passed to "
            "apply_chat_template. Scripts that don't render chat templates from rows reject it up front."
        },
    )
    unfreeze_layers_patterns: list[str] | None = field(
        default=None,
        metadata={
            "help": "Freeze the whole model, then unfreeze the modules whose full names match these fnmatch patterns."
        },
    )
    freeze_layers_patterns: list[str] | None = field(
        default=None,
        metadata={
            "help": "Freeze the parameters whose full names match these fnmatch patterns, applied after unfreezing."
        },
    )
    enable_efficiency_metrics: bool = field(
        default=False,
        metadata={
            "help": "Enable EfficiencyCallback (step time, tokens/s, MFU/S-MFU, peak memory), which sets "
            "include_num_input_tokens_seen='all'. Off by default: MFU is misleading on trainers that forward "
            "several sequences per step (DPO, SMPO, reward, distillation). "
            "See agent-docs/training-methods/callbacks.md#efficiencycallback."
        },
    )
    enable_moe_metrics: bool = field(
        default=True,
        metadata={
            "help": "Enable MoEMetricsCallback (per-expert load, dead-expert fraction, first/last-layer skew); a "
            "no-op on dense models. It turns on output_router_logits only under moe_balancing: aux_loss, and is not "
            "wired under a bias balancing mode (whose callback logs the load) or pipeline parallelism. "
            "See agent-docs/training-methods/callbacks.md#moemetricscallback."
        },
    )
    num_full_model_params: float | None = field(
        default=None,
        metadata={
            "help": "Total parameter count across all EP/TP ranks; EfficiencyCallback then reports distributed "
            "efficiency = params_ratio × MFU. Leave unset on data-parallel runs, whose local count is the whole "
            "model."
        },
    )
    report_mfu_diagnostics: bool = field(
        default=False,
        metadata={
            "help": "Add MFU, S-MFU and achieved TFLOPS to the headline training log; they are computed every step "
            "either way. Off by default: tokens/s/GPU is the reported throughput metric, and MFU is misleading "
            "across GPUs, dtypes and MoE models. "
            "See agent-docs/reference/configuration-reference.md#commonscriptarguments."
        },
    )
    log_decoded_samples: bool = field(
        default=False,
        metadata={
            "help": "Write the first few decoded training/eval samples (input_ids decoded with "
            "skip_special_tokens=False) to <output_dir>/log/{train,eval,...}_sample.txt. Off by "
            "default; useful for eyeballing exactly what the model sees after templating/packing."
        },
    )
    save_completions: bool = field(
        default=True,
        metadata={
            "help": "GRPO family: write each step's rollout completions and trajectories to "
            "<output_dir>/completions/completions_<step>.parquet, plus a wandb completions table. Independent of "
            "TRL's console-only log_completions; ignored by trainers that do not generate. "
            "See agent-docs/training-methods/async-grpo/monitoring.md#saved-trajectories."
        },
    )
    moe_balancing: BalancingMode = field(
        default="auto",
        metadata={
            "help": "MoE router balancing: 'auto' (resolved per model), 'bias_update' (DeepSeek-V3 sign update into "
            "the family's own checkpoint slot), 'bias_update_transient' (the same update in a side-buffer no export "
            "carries), 'aux_loss' (the model's aux loss at router_aux_loss_coef) or 'none'. Both bias modes are "
            "downgraded to 'none' on the weight-sync RL scripts. "
            "See agent-docs/training-methods/callbacks.md#moe-balancing-modes."
        },
    )
    router_balancing_rate: float = field(
        default=1.0e-3,
        metadata={
            "help": "Sign-step magnitude (gamma) of RouterBiasBalancingCallback under a bias_update mode; the "
            "DeepSeek-V3 paper used 1e-3. Ignored under every other resolved mode, so also on the weight-sync RL "
            "scripts."
        },
    )
    enable_torch_profiler: bool = field(
        default=False,
        metadata={
            "help": "Enable TorchProfilerCallback: per-rank Chrome traces, flame-graph stacks, an optional memory "
            "timeline and a top-ops table over a short window of steps. Off by default for the overhead of its "
            "active window. See agent-docs/reference/debugging.md#1a-torchprofiler--gpuoperator-trace."
        },
    )
    profiler_output_dir: str = field(
        default_factory=torch_trace_dir,
        metadata={"help": "Output directory for torch.profiler artifacts (defaults under HALO_DATA_ROOT)."},
    )
    profiler_wait: int = field(
        default=5,
        metadata={"help": "torch.profiler schedule: steps to skip before profiling."},
    )
    profiler_warmup: int = field(
        default=1,
        metadata={"help": "torch.profiler schedule: warmup steps before recording."},
    )
    profiler_active: int = field(
        default=3,
        metadata={"help": "torch.profiler schedule: number of steps recorded into the trace."},
    )
    profiler_ranks: str = field(
        default="0",
        metadata={
            "help": "Which global ranks profile: '0', 'all' or a comma list like '0,8'. "
            "Use 'all' to capture every rank when diagnosing rank skew or stragglers."
        },
    )
    profiler_record_memory_snapshot: bool = field(
        default=False,
        metadata={
            "help": "Also record CUDA allocation history over the profiler's active window and dump a "
            ".pickle for https://pytorch.org/memory_viz (allocation timeline + flame graph)."
        },
    )

    def __post_init__(self):
        if self.project_name == _DEFAULT_PROJECT:
            self.project_name = self.PROJECT_NAME
        self._validate_ranges()

    def _validate_ranges(self) -> None:
        super()._validate_ranges()
        # ``project_name`` becomes WANDB_PROJECT / CLEARML_PROJECT verbatim: ``--project_name=`` or
        # ``--project_name=None`` would name the tracking project "" or "None", and a YAML null reaches
        # os.environ as None, where the assignment raises TypeError later in the run. Annotated ``str``,
        # so the parser leaves a null spelling as literal text on the CLI.
        project = self.project_name
        if not isinstance(project, str) or not project.strip() or is_null_string(project):
            raise ValueError(
                f"project_name must be a non-empty string and not a null spelling (it becomes "
                f"WANDB_PROJECT / CLEARML_PROJECT), got {project!r}. Omit the key — from the YAML "
                f"and the command line alike — to take this script's own default."
            )
        owner = type(self).__name__
        # One ratio per dataset when a list: each keeps that share of its rows, so 0 empties it and
        # above 1 cannot be drawn without replacement.
        if self.dataset_ratio is not None:
            for ratio in self.dataset_ratio if isinstance(self.dataset_ratio, list) else [self.dataset_ratio]:
                require_finite(owner, dataset_ratio=ratio)
                if not 0 < ratio <= 1:
                    raise ValueError(f"{owner}: dataset_ratio must be in (0, 1], got {ratio!r}")
        # The proportion train_test_split re-splits at; an int there would be a row count.
        if self.test_size is not None:
            require_finite(owner, test_size=self.test_size)
            if not 0 < self.test_size < 1:
                raise ValueError(f"{owner}: test_size must be a proportion in (0, 1), got {self.test_size!r}")
        require_positive(owner, **present(num_full_model_params=self.num_full_model_params))
        require_positive(owner, router_balancing_rate=self.router_balancing_rate)
        # torch.profiler.schedule's own bounds.
        require_int(owner, minimum=0, profiler_wait=self.profiler_wait, profiler_warmup=self.profiler_warmup)
        require_positive_int(owner, profiler_active=self.profiler_active)
