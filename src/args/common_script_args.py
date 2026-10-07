"""``CommonScriptArguments`` — the dataset / logging / callback knobs every training script shares."""

from dataclasses import dataclass, field
from typing import ClassVar

from src.args.validation import RangeValidatedConfig
from src.data.pipeline.tokenizer_backend import TokenizerBackend
from src.env import torch_trace_dir
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
            "help": "Test split proportion (like 0.05). Leave empty to keep the dataset's own split: "
            "set on a dataset that already carries a test split, the two splits are concatenated and "
            "re-split at this proportion. Ignored (warned) on a sharded or pre-processed dataset, "
            "whose split was fixed at preparation time."
        },
    )
    project_name: str = field(
        default=_DEFAULT_PROJECT,
        metadata={
            "help": "Name of logging project (wandb or clearml). Becomes WANDB_PROJECT / "
            "CLEARML_PROJECT; leave it out to take the per-script default rather than nulling it."
        },
    )
    pad_token: str | None = field(default=None, metadata={"help": "Special pad token"})
    bos_token: str | None = field(default=None, metadata={"help": "Special bos token"})
    eos_token: str | None = field(default=None, metadata={"help": "Special eos token"})
    chat_template: str | None = field(
        default=None,
        metadata={
            "help": "Chat template for the model. Can be either: "
            "(1) a path to a .jinja/.jinja2/.j2 file, or "
            "(2) the template string directly. "
            "If not provided, uses the tokenizer's default template."
        },
    )
    force_chat_template: bool = field(
        default=False,
        metadata={"help": "Force custom chat template even if tokenizer already has one"},
    )
    added_special_tokens: list[str] | None = field(default=None, metadata={"help": "Additional special tokens"})
    tokenizer_backend: TokenizerBackend = field(
        default="hf",
        metadata={
            "help": "Text→ids backend: 'hf' or 'gigatoken' (optional extra; token IDs verified identical at startup)."
        },
    )
    tools_field: str | None = field(
        default=None,
        metadata={
            "help": "Field in dataset with tool definitions (list of dicts or "
            "JSON string) to pass to apply_chat_template. Scripts that "
            "don't render chat templates from rows reject it up front."
        },
    )
    unfreeze_layers_patterns: list[str] | None = field(
        default=None,
        metadata={"help": "Patterns of layer names needed to be unfreeze for learning"},
    )
    freeze_layers_patterns: list[str] | None = field(
        default=None,
        metadata={"help": "Patterns of layer names to freeze (applied after unfreeze)"},
    )
    enable_efficiency_metrics: bool = field(
        default=False,
        metadata={
            "help": "Enable EfficiencyCallback (per-step time, TPS, MFU/S-MFU, peak memory). "
            "Off by default — MFU/S-MFU values can be misleading for trainers that "
            "consume multiple sequences per step (DPO/SMPO/Reward forward both chosen "
            "and rejected; Distillation also forwards a teacher). Set true in any "
            "YAML to enable. Auto-sets include_num_input_tokens_seen='all' when on."
        },
    )
    enable_moe_metrics: bool = field(
        default=True,
        metadata={
            "help": "Enable MoEMetricsCallback (per-expert load distribution, dead-expert fraction, "
            "first/last layer skew). Auto-no-op for non-MoE models. Needs output_router_logits, which "
            "it enables only under moe_balancing=aux_loss; otherwise it reports only if already on. "
            "Not wired at all under a bias_update balancing mode (RouterBiasBalancingCallback carries "
            "the load metrics there) nor under pipeline parallelism, whose stage forwards return bare "
            "tensors without router_logits — moe_balancing=bias_update is the route to load metrics under PP."
        },
    )
    num_full_model_params: float | None = field(
        default=None,
        metadata={
            "help": "Total parameters in the full model (across all EP/TP ranks). When set, "
            "EfficiencyCallback computes distributed efficiency = params_ratio * mfu. "
            "Auto-detected from local params when EP/TP not used; leave None for DP-only."
        },
    )
    report_mfu_diagnostics: bool = field(
        default=False,
        metadata={
            "help": "Add MFU / S-MFU / achieved-TFLOPS to the headline training log. Off by "
            "default: tokens/s/GPU (with cluster tokens/s, peak memory, step time) is the "
            "reported and gated throughput metric — MFU's denominator (peak device FLOPS) "
            "shifts with GPU/dtype/kernel and the MoE numerator is ambiguous, so it is a "
            "misleading headline. The values are still computed every step (available on "
            "EfficiencyCallback for diagnostics); this only controls headline visibility."
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
            "help": "GRPO family (online/env): persist per-step rollout completions/trajectories to "
            "<output_dir>/completions/completions_<step>.parquet (prompt, rendered completion, reward, "
            "advantage) and, under wandb, a `completions` table — DECOUPLED from "
            "console logging. This is the durable generation record; keep it on to inspect what the "
            "policy produced. The rich per-sample CONSOLE table stays gated by TRL's `log_completions` "
            "(set that True only to also spam the console). Text is rendered from detokenized message "
            "content, unaffected by train_on_sampled_tokens (raw ids feed the loss only). Ignored by "
            "non-generating trainers (SFT, offline GRPO). Default on."
        },
    )
    moe_balancing: BalancingMode = field(
        default="auto",
        metadata={
            "help": "MoE router balancing: 'auto' (default; resolved per model), 'bias_update' "
            "(DeepSeek-V3 sign update into the family's own checkpoint slot; raises where the family "
            "has none), 'bias_update_transient' (the same update in a trainer-only side-buffer no "
            "export carries), 'aux_loss' (the model's switch-style aux loss at "
            "router_aux_loss_coef) or 'none'. Both bias modes are downgraded to 'none' on the "
            "weight-sync RL scripts. Per-family resolution and slots: "
            "agent-docs/training-methods/callbacks.md#moe-balancing-modes."
        },
    )
    router_balancing_rate: float = field(
        default=1.0e-3,
        metadata={
            "help": "Sign-step magnitude (gamma) for RouterBiasBalancingCallback when "
            "moe_balancing resolves to bias_update or bias_update_transient. DeepSeek-V3 paper "
            "used 1e-3. Ignored under every other resolved mode — including on the on-policy "
            "weight-sync RL scripts, where both bias modes are downgraded to 'none' and this knob "
            "is therefore unreachable."
        },
    )
    enable_torch_profiler: bool = field(
        default=False,
        metadata={
            "help": "Enable TorchProfilerCallback — captures a short window of training steps "
            "with torch.profiler and writes per-rank Chrome traces, flame-graph stacks, "
            "an optional memory timeline, and a top-ops table. Off by default (adds "
            "overhead during the active window). See agent-docs/reference/debugging.md."
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
            "help": "Which global ranks profile: '0' (default), 'all', or a comma list like '0,8'. "
            "Use 'all' to capture every rank when diagnosing rank skew / stragglers."
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
        """Guard ``project_name``, which becomes ``WANDB_PROJECT`` / ``CLEARML_PROJECT`` verbatim.

        Checked here rather than where ``__post_init__`` applies :data:`PROJECT_NAME`, so the CLI path
        is held to the same rule: ``--project_name=`` and ``--project_name=None`` land by ``setattr``, skipping
        ``__post_init__``, and would otherwise name the tracking project ``""`` or ``"None"``. A YAML
        ``project_name: null`` reaches ``os.environ`` as ``None``, where the assignment raises
        TypeError later in the run.
        """
        super()._validate_ranges()
        project = self.project_name
        # Annotated ``str``, so the parser leaves a null spelling as literal text on the CLI.
        if not isinstance(project, str) or not project.strip() or is_null_string(project):
            raise ValueError(
                f"project_name must be a non-empty string and not a null spelling (it becomes "
                f"WANDB_PROJECT / CLEARML_PROJECT), got {project!r}. Omit the key — from the YAML "
                f"and the command line alike — to take this script's own default."
            )
