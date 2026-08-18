"""``GenerateExamplesCallback`` — sample generations at eval time, parallelism-aware."""

import logging
import random

import pandas as pd
import torch
import torch.distributed as dist
import wandb
from clearml import Task
from datasets import Dataset
from tabulate import tabulate
from torch.distributed.tensor import DTensor
from tqdm import tqdm
from transformers import (
    PreTrainedTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)

from src.distributed.context_parallel.wrapper import UlyssesCPModelWrapper
from src.distributed.pipeline_parallel.stage import PipelineStageModule
from src.distributed.runtime import barrier, get_global_rank, get_global_world_size, is_global_main_process
from src.models.structure import unwrap_framework_wrappers

logger = logging.getLogger(__name__)

# One fixed seed for every sampling this callback does — which prompts to show, and which rows to
# print. It must be a constant, not a run seed: the sample is drawn independently on every rank and
# a per-rank value would make the ranks generate from different prompts and gather mismatched rows.
_SAMPLE_SEED = 42

# Prompts are tail-truncated to this many characters in the log table; the completion is what the
# sample is read for, and a full long-context prompt would bury it.
_MAX_PRINTED_PROMPT_CHARS = 200


def _pretty_print_dataframe(df):
    formatted_df = df.copy()
    formatted_df["Prompt"] = formatted_df["Prompt"].apply(
        lambda x: ("..." + x[-_MAX_PRINTED_PROMPT_CHARS:]) if len(x) > _MAX_PRINTED_PROMPT_CHARS else x
    )
    logger.info("\n" + tabulate(formatted_df, headers="keys", tablefmt="simple", showindex=False))


def _is_cp_wrapped(model) -> bool:
    """Whether a CP wrapper sits anywhere in the tree, not just at its root.

    Under ``use_peft`` the wrapper sits inside ``PeftModel.base_model.model`` and PEFT delegates
    ``generate`` to it, so a root ``isinstance`` would let the callback generate sequence-sharded.
    """
    return any(isinstance(module, UlyssesCPModelWrapper) for module in model.modules())


def _has_tp_dtensor_params(model) -> bool:
    """Whether the model has TP-sharded parameters (incompatible with generate()).

    A TP shard is a DTensor on a mesh naming a ``tp`` dim, covering HF ``tp_plan`` and toolkit
    attention-only TP alike; ``_tp_plan`` is populated on every load, TP or not, so it proves nothing.
    """
    for param in model.parameters():
        if isinstance(param.data, DTensor):
            mesh = param.data.device_mesh
            dim_names = getattr(mesh, "mesh_dim_names", None) or ()
            if "tp" in dim_names:
                return True
    return False


def _has_any_dtensor_params(model) -> bool:
    """Check if the model has any DTensor parameters (TP or FSDP2)."""
    return any(isinstance(param.data, DTensor) for param in model.parameters())


def _is_fsdp2_model(model) -> bool:
    """Whether the model is wrapped with FSDP2 (fully_shard).

    ``fully_shard`` stamps ``_is_fsdp_managed_module`` on every module it manages, so the flag answers
    it for any module; a DTensor check would not — ``reshard_after_forward=False`` leaves plain tensors.
    """
    return hasattr(model, "_is_fsdp_managed_module")


def _is_distributed_parallel_model(model) -> bool:
    """Whether the model uses parallelism (EP, FSDP2, TP DTensors) requiring all ranks in forward."""
    if _is_fsdp2_model(model):
        return True
    for module in model.modules():
        if hasattr(module, "ep_config") or hasattr(module, "dispatcher"):
            return True
    return _has_any_dtensor_params(model)


def _disable_gradient_checkpointing(model) -> list[torch.nn.Module]:
    """Disable gradient checkpointing everywhere it is on; return the modules that had it on."""
    was_enabled = [module for module in model.modules() if getattr(module, "gradient_checkpointing", False)]
    for module in was_enabled:
        module.gradient_checkpointing = False
    return was_enabled


def _restore_gradient_checkpointing(modules: list[torch.nn.Module]) -> None:
    """Re-enable gradient checkpointing on exactly the modules that had it.

    Module for module: switching it back on everywhere would enable it on layers a partially
    checkpointed model deliberately left off, changing memory and step time for the rest of training.
    """
    for module in modules:
        module.gradient_checkpointing = True


class GenerateExamplesCallback(TrainerCallback):
    """Generate text examples during evaluation and log them.

    DDP splits across ranks (gathered on rank 0); FSDP2/EP generate on all ranks together; TP and CP are
    skipped (incompatible with generate).
    """

    def __init__(
        self,
        preprocessed_dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        num_examples=5,
        max_new_tokens=256,
        logger_backend="wandb",
    ):
        """
        Args:
            preprocessed_dataset: tokenized ``datasets.Dataset``.
            logger_backend: ``'clearml'`` or ``'wandb'``.
        """
        self.dataset = preprocessed_dataset
        self.tokenizer = tokenizer
        self.num_examples = min(num_examples, len(self.dataset))
        self.max_new_tokens = max_new_tokens
        self.logger_backend = logger_backend

        rng = random.Random(_SAMPLE_SEED)
        sample_indices = rng.choices(range(len(self.dataset)), k=self.num_examples)
        self.samples = self.dataset.select(sample_indices)

    @classmethod
    def from_config(cls, args, training_config, generate_dataset, tokenizer) -> "GenerateExamplesCallback | None":
        """Build the callback from script args, or ``None`` when it cannot run.

        ``None`` unless ``args.generate_eval_examples`` is set AND ``generate_dataset`` carries
        tokenized ``input_ids`` (generation replays the tokenized prompt; an untokenized or absent
        generate split — e.g. the VLM path — cannot be generated from).
        """
        if not getattr(args, "generate_eval_examples", False):
            return None
        if generate_dataset is None or "input_ids" not in generate_dataset.column_names:
            return None
        return cls(
            preprocessed_dataset=generate_dataset,
            tokenizer=tokenizer,
            num_examples=args.num_eval_examples,
            logger_backend=training_config.report_to[0] if training_config.report_to else "none",
        )

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs["model"]
        model.eval()

        # unwrap_framework_wrappers, not a DDP-only peel: it also drops torch.compile's
        # OptimizedModule, whose _orig_mod indirection makes the PP-stage and CP-wrapper isinstance
        # guards below silently miss and let generate() run on a partial or sequence-sharded model.
        unwrapped = unwrap_framework_wrappers(model)

        # A PP stage is a partial model with no generate(): calling it AttributeErrors at the first eval.
        if isinstance(unwrapped, PipelineStageModule):
            if is_global_main_process():
                logger.warning(
                    "[GenerateExamplesCallback] Skipping: a pipeline-parallel stage holds only part "
                    "of the model and cannot generate. Generate from a saved checkpoint instead."
                )
            return

        if _is_cp_wrapped(unwrapped):
            if is_global_main_process():
                logger.warning(
                    "[GenerateExamplesCallback] Skipping: "
                    "Context Parallelism is incompatible with autoregressive generation."
                )
            return

        if _has_tp_dtensor_params(unwrapped):
            if is_global_main_process():
                logger.warning(
                    "[GenerateExamplesCallback] Skipping: "
                    "Tensor Parallelism (DTensor) is incompatible with autoregressive generation."
                )
            return

        if _is_distributed_parallel_model(unwrapped):
            # FSDP2/EP generation is collective: swallowing a per-rank error would desync the next one.
            self._generate_standard(unwrapped, state)
            return
        try:
            # DDP / single GPU: generation is rank-local, so a failure is safe to swallow.
            self._generate_standard(unwrapped, state)
        except RuntimeError as e:
            if is_global_main_process():
                logger.warning(
                    f"[GenerateExamplesCallback] Generation failed at step "
                    f"{state.global_step}: {e}. Training will continue."
                )

    def _generate_standard(self, gen_model, state: TrainerState):
        """Generate examples using model.generate().

        Distributed parallel models (FSDP2, EP): all ranks participate in forward. DDP: work split.
        """
        is_parallel = _is_distributed_parallel_model(gen_model)
        world_size = get_global_world_size()

        gc_enabled_modules = _disable_gradient_checkpointing(gen_model)
        try:
            if is_parallel or world_size <= 1:
                records = self._generate_all_ranks(gen_model)
            else:
                records = self._generate_split_ranks(gen_model)
        finally:
            _restore_gradient_checkpointing(gc_enabled_modules)

        if is_global_main_process() and records:
            combined_df = pd.DataFrame(records)
            self._log_results(combined_df, state)
            n_print = min(3, len(combined_df))
            _pretty_print_dataframe(combined_df.sample(n_print, random_state=_SAMPLE_SEED))

    def _generate_all_ranks(self, model) -> list:
        """Generate on all ranks together (required for EP/FSDP2 or single-GPU).

        Forced greedy: the collective forward requires every rank to run the SAME number of decode steps;
        sampling would diverge ranks on EOS timing and hang the job.
        """
        records = []

        barrier()

        for example in tqdm(
            self.samples,
            desc="Generating examples",
            disable=not is_global_main_process(),
        ):
            record = self._generate_single(model, example, deterministic=True)
            records.append(record)

        barrier()

        return records

    def _generate_split_ranks(self, model) -> list:
        """Split generation across DDP ranks, gather results on rank 0."""
        rank = get_global_rank()
        world_size = get_global_world_size()

        all_indices = list(range(len(self.samples)))
        my_indices = [i for i in all_indices if i % world_size == rank]

        local_records = []
        for idx in my_indices:
            example = self.samples[idx]
            # A per-example failure must not skip the collective all_gather_object below (would hang peers).
            try:
                record = self._generate_single(model, example)
            except Exception as e:
                record = self._build_record("", f"<generation failed: {e}>", example)
            local_records.append(record)

        all_records = [None] * world_size
        dist.all_gather_object(all_records, local_records)

        if is_global_main_process():
            records = []
            for rank_records in all_records:
                records.extend(rank_records)
            return records
        return []

    def _generate_single(self, model, example, deterministic: bool = False) -> dict:
        """Generate a single example. ``deterministic=True`` forces greedy so FSDP2/EP stay rank-synced."""
        input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to(model.device)
        attention_mask = torch.tensor(example["attention_mask"]).unsqueeze(0).to(model.device)

        generate_kwargs = {"max_new_tokens": self.max_new_tokens}
        if deterministic:
            generate_kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                **generate_kwargs,
            )

        prompt_text = self.tokenizer.decode(input_ids.squeeze(), skip_special_tokens=False)
        completion_text = self.tokenizer.decode(outputs.squeeze(), skip_special_tokens=False)
        completion_text = completion_text[len(prompt_text) :]

        return self._build_record(prompt_text, completion_text, example)

    @staticmethod
    def _build_record(prompt_text: str, completion_text: str, example: dict) -> dict:
        """Build output record with optional chosen/rejected fields."""
        record = {"Prompt": prompt_text, "Completion": completion_text}
        if "chosen" in example:
            record["Chosen"] = example["chosen"][0]["content"]
        if "rejected" in example:
            record["Rejected"] = example["rejected"][0]["content"]
        return record

    def _log_results(self, df: pd.DataFrame, state: TrainerState):
        """Log results to the configured backend."""
        if self.logger_backend == "clearml":
            task = Task.current_task()
            if task:
                task_logger = task.get_logger()
                task_logger.report_table(
                    "Generated Text Samples",
                    "DataFrame",
                    iteration=state.global_step,
                    table_plot=df,
                )
        elif self.logger_backend == "wandb":
            wandb.log({f"eval/generated_text_{state.global_step}": wandb.Table(dataframe=df)})
