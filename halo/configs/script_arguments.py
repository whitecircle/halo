"""The arguments training scripts share: ``ModelArguments`` (TRL's model config plus remote-code trust),
``CommonScriptArguments`` (the dataset / logging / callback knobs every script parses) and the field bundles
several script-argument and trainer-config dataclasses mix in."""

from dataclasses import dataclass, field
from typing import Any, ClassVar

from trl import ModelConfig

from halo.checkpoint.format import DEFAULT_MAX_SHARD_SIZE
from halo.config_validation import ValidatedConfig, present, require_finite, require_int
from halo.configs.parser import is_null_string
from halo.env import torch_trace_dir
from halo.model_loading.tokenizer_backend import TokenizerBackend
from halo.models.moe.balancing import BalancingMode

_DEFAULT_PROJECT = "default-project"

# The dataset column a ground-truth answer is read from unless a config renames it.
DEFAULT_ANSWER_FIELD = "answer"


@dataclass
class CheckpointArguments:
    """How a run writes and resumes its checkpoints."""

    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": "Allow starting a run in a non-empty output_dir. Off by default so a fresh run "
            "cannot silently interleave its checkpoints with an existing one's."
        },
    )
    save_sharded_ep: bool = field(
        default=False,
        metadata={
            "help": "Save EP checkpoints as per-rank expert shards written in parallel instead of the gathered HF "
            "checkpoint from_pretrained loads. Nothing loads the shards until "
            "scripts/after_training/merge_ep_shards.py merges them; the merged directory resumes exactly, "
            "optimizer included, on the save's topology. "
            "See agent-docs/reference/checkpoints.md#expert-parallelism-ep-eptp-epcp."
        },
    )
    save_max_shard_size: str = field(
        default=DEFAULT_MAX_SHARD_SIZE,
        metadata={
            "help": "Largest safetensors shard the gathered EP/TP/FSDP2/CP saves write, like '5GB'; HF's own "
            "save_pretrained keeps its default. A sharded EP save writes one file per rank regardless, and "
            "merge_ep_shards.py sizes its output by its own --max_shard_size."
        },
    )
    allow_optimizer_warm_restart: bool = field(
        default=False,
        metadata={
            "help": "On resume, continue with a freshly initialized optimizer (weights, step and LR schedule still "
            "resume) when restoring this topology's optimizer shards fails on a rank, or the checkpoint holds an "
            "interrupted save's optimizer state. Off, both raise on every rank, since a warm restart resets the "
            "optimizer moments mid-run. "
            "See agent-docs/reference/checkpoints.md#warm-restart-vs-exact-resume-torchrun."
        },
    )


@dataclass
class ModelLoadArguments:
    """How the policy model is built from its checkpoint."""

    text_only_model: bool = field(
        default=False,
        metadata={
            "help": "Load a multimodal checkpoint through its text-only CausalLM class (Qwen3.5/3.6: "
            "Qwen3_5MoeForCausalLM), dropping the vision tower and any MTP tail from the build and the export. For "
            "text-only datasets. "
            "See agent-docs/reference/configuration-reference.md#model-checkpoint-and-from-scratch-flags."
        },
    )
    reset_sinks: bool = field(
        default=True,
        metadata={
            "help": "GptOss only: true neutralizes the attention sinks for fine-tuning; false keeps the pretrained "
            "sinks live and frozen, the on-policy RL setting, which needs a sink-carrying attention implementation "
            "and no context parallelism. See agent-docs/models/gpt-oss.md#attention-sinks."
        },
    )
    train_sinks: bool = field(
        default=False,
        metadata={
            "help": "GptOss only: train the live sinks (requires reset_sinks: false) under full fine-tuning with "
            "flash_attention_4 or eager. Refused with flex_attention, FA3, adapter runs and weight-sync RL. "
            "See agent-docs/models/gpt-oss.md#attention-sinks."
        },
    )


@dataclass
class ModelArguments(ModelConfig):
    """TRL's ``ModelConfig`` with ``trust_remote_code``, which TRL moved onto its own trainer configs.

    The toolkit's loaders read it here for every method, including those whose training config is not a TRL
    one; a YAML key shared with a TRL config reaches both dataclasses.
    """

    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Trust the checkpoint's remote modeling and tokenizer code when loading it."},
    )


@dataclass
class TorchProfilerArguments(ValidatedConfig):
    """The opt-in ``TorchProfilerCallback`` and its ``torch.profiler`` schedule."""

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
        default=5, metadata={"help": "torch.profiler schedule: steps to skip before profiling."}
    )
    profiler_warmup: int = field(
        default=1, metadata={"help": "torch.profiler schedule: warmup steps before recording."}
    )
    profiler_active: int = field(
        default=3, metadata={"help": "torch.profiler schedule: number of steps recorded into the trace."}
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

    def _validate_fields(self) -> None:
        super()._validate_fields()
        owner = type(self).__name__
        # torch.profiler.schedule's own bounds.
        require_int(owner, ge=0, profiler_wait=self.profiler_wait, profiler_warmup=self.profiler_warmup)
        require_int(owner, ge=1, profiler_active=self.profiler_active)


@dataclass
class CommonScriptArguments(TorchProfilerArguments, CheckpointArguments, ModelLoadArguments):
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
    # Annotated `float` though a per-dataset YAML list is supported: HfArgumentParser builds only `Optional[X]`
    # unions, and `parse_dict` assigns a YAML list uncoerced (it just cannot be spelled on the command line).
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
        default=False, metadata={"help": "Use chat_template even when the tokenizer ships its own."}
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
            "help": "GRPO family: write each step's rollout completions and trajectories under "
            "<output_dir>/completions/ (completions_<step>.parquet online, one part per rank in completions_<step>/ "
            "for async GRPO), plus a wandb completions table. Independent of "
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
            "help": "Sign-step magnitude (gamma) of BalancingBiasCallback under a bias_update mode; the "
            "DeepSeek-V3 paper used 1e-3. Ignored under every other resolved mode, so also on the weight-sync RL "
            "scripts."
        },
    )

    def __post_init__(self):
        if self.project_name == _DEFAULT_PROJECT:
            self.project_name = self.PROJECT_NAME
        super().__post_init__()

    def _validate_fields(self) -> None:
        super()._validate_fields()
        # A YAML null would reach os.environ as None and raise later in the run; annotated ``str``, so the
        # parser leaves a CLI null spelling as literal text.
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
                require_finite(owner, gt=0, le=1, dataset_ratio=ratio)
        # The proportion train_test_split re-splits at; an int there would be a row count.
        require_finite(owner, gt=0, lt=1, why=" (a proportion)", **present(test_size=self.test_size))
        require_finite(owner, gt=0, **present(num_full_model_params=self.num_full_model_params))
        require_finite(owner, gt=0, router_balancing_rate=self.router_balancing_rate)


@dataclass
class GenerationEvalArguments(ValidatedConfig):
    """Eval-time example-generation knobs (SFT, DPO, SMPO, offline GRPO)."""

    generate_eval_examples: bool = field(default=True, metadata={"help": "Generate text examples during evaluation."})
    num_eval_examples: int = field(default=50, metadata={"help": "Examples to generate per evaluation."})

    def _validate_fields(self) -> None:
        super()._validate_fields()
        require_int(type(self).__name__, ge=1, num_eval_examples=self.num_eval_examples)


@dataclass
class DatasetNumProcArguments(ValidatedConfig):
    """Dataset-preprocessing worker count of the trainer configs that map their own data, read through
    :func:`~halo.data.pipeline.coordinated_map.resolve_map_num_proc`."""

    dataset_num_proc: int | None = field(
        default=None,
        metadata={
            "help": "Worker processes for dataset preprocessing (map/filter). Unset means the toolkit "
            "default (HALO_DATASET_NUM_PROC, else three quarters of the usable CPUs, at most 64), not one worker."
        },
    )

    def _validate_fields(self) -> None:
        super()._validate_fields()
        require_int(type(self).__name__, ge=1, **present(dataset_num_proc=self.dataset_num_proc))


@dataclass
class ModelInitKwargsArguments:
    """``model_init_kwargs`` of the trainer configs whose trainer also takes the model as a path string."""

    model_init_kwargs: dict[str, Any] | None = field(
        default=None,
        metadata={
            "help": "Model-config overrides on every entry-script path: written onto the loaded "
            "config's fields before the load, raising on a key that config does not declare and "
            "on dtype/torch_dtype. Model-loading kwargs only where a trainer is constructed "
            "programmatically with the model as a path string."
        },
    )


@dataclass
class CrossDocumentLinearAttentionArguments:
    """The opt-in of the configs whose multi-document rows (``packing``, ``padding_free``) are refused on
    linear-attention layers no document boundary reaches
    (:func:`~halo.models.segment_markers.reject_unmarked_linear_attention_rows`)."""

    allow_cross_document_linear_attention: bool = field(
        default=False,
        metadata={
            "help": "Train multi-document rows (packing, padding_free) on a model whose linear-attention layers "
            "receive no document boundary (Ring's Lightning Attention-2): their recurrent state then carries from "
            "each document of a row into the next. Off, such a run is refused."
        },
    )
