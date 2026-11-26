"""Sample generations at eval time, parallelism-aware."""

import logging
import random
from contextlib import contextmanager

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

from halo.distributed.context_parallel.ulysses_model import UlyssesModel
from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.mesh import has_tp_dim
from halo.distributed.pipeline_parallel.stage import PipelineStageModule
from halo.models.introspection.unwrapping import unwrap_framework_wrappers
from halo.runtime.collectives import barrier
from halo.runtime.ranks import get_global_rank, get_global_world_size, is_global_main_process

logger = logging.getLogger(__name__)

# A constant: every rank draws the sample independently, and differing seeds would gather mismatched rows.
_SAMPLE_SEED = 42
# Prompts are tail-truncated in the log table, so a long-context prompt does not crowd out the completion.
_MAX_PRINTED_PROMPT_CHARS = 200


def _pretty_print_dataframe(df):
    formatted_df = df.copy()
    formatted_df["Prompt"] = formatted_df["Prompt"].apply(
        lambda x: ("..." + x[-_MAX_PRINTED_PROMPT_CHARS:]) if len(x) > _MAX_PRINTED_PROMPT_CHARS else x
    )
    logger.info("\n" + tabulate(formatted_df, headers="keys", tablefmt="simple", showindex=False))


def _generation_skip_reason(model) -> str | None:
    """Why ``model`` cannot ``generate()`` here, or None. The CP wrapper is searched in the whole tree, since PEFT
    delegates ``generate`` to it. A TP shard is a DTensor on a mesh naming a ``tp`` dim; ``_tp_plan`` is no signal."""
    if isinstance(model, PipelineStageModule):
        return (
            "a pipeline-parallel stage holds only part of the model and cannot generate. "
            "Generate from a saved checkpoint instead."
        )
    if any(isinstance(module, UlyssesModel) for module in model.modules()):
        return "Context Parallelism is incompatible with autoregressive generation."
    if any(isinstance(param.data, DTensor) and has_tp_dim(param.data.device_mesh) for param in model.parameters()):
        return "Tensor Parallelism (DTensor) is incompatible with autoregressive generation."
    return None


def _uses_collective_forward(model) -> bool:
    """Whether every rank must join the forward. FSDP is read off its ``_is_fsdp_managed_module`` stamp, since a
    DTensor check would miss FSDP2 under ``reshard_after_forward=False``, which leaves plain tensors."""
    return (
        hasattr(model, "_is_fsdp_managed_module")
        or bool(named_ep_layers(model))
        or any(isinstance(param.data, DTensor) for param in model.parameters())
    )


@contextmanager
def _gradient_checkpointing_disabled(model):
    """Gradient checkpointing off for the block, then back on only for the modules that had it: a partially
    checkpointed model left the others off deliberately."""
    enabled = [module for module in model.modules() if getattr(module, "gradient_checkpointing", False)]
    for module in enabled:
        module.gradient_checkpointing = False
    try:
        yield
    finally:
        for module in enabled:
            module.gradient_checkpointing = True


class GenerateExamplesCallback(TrainerCallback):
    """Log generations at evaluation: ``logger_backend`` (``'clearml'``, ``'wandb'``) receives the table, else print
    only. DDP splits samples across ranks; FSDP and EP generate on all ranks together; PP, CP and TP are skipped."""

    def __init__(
        self,
        preprocessed_dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        num_examples: int,
        max_new_tokens=256,
        logger_backend="wandb",
    ):
        self.dataset = preprocessed_dataset
        self.tokenizer = tokenizer
        self.num_examples = min(num_examples, len(self.dataset))
        self.max_new_tokens = max_new_tokens
        self.logger_backend = logger_backend

        rng = random.Random(_SAMPLE_SEED)
        sample_indices = rng.sample(range(len(self.dataset)), self.num_examples)
        self.samples = self.dataset.select(sample_indices)

    @classmethod
    def from_config(cls, args, training_config, generate_dataset, tokenizer) -> "GenerateExamplesCallback | None":
        """``None`` unless ``args.generate_eval_examples`` is set and ``generate_dataset`` carries tokenized
        ``input_ids``, since generation replays the tokenized prompt."""
        if not args.generate_eval_examples:
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

        # Must also drop torch.compile's OptimizedModule: a compiled root hides a PP stage, generating on part of it.
        unwrapped = unwrap_framework_wrappers(model)

        reason = _generation_skip_reason(unwrapped)
        if reason is not None:
            if is_global_main_process():
                logger.warning(f"[GenerateExamplesCallback] Skipping: {reason}")
            return

        if _uses_collective_forward(unwrapped):
            # Collective generation, so swallowing a per-rank error would desync the next collective.
            self._generate_and_log(unwrapped, state, collective=True)
            return
        try:
            # Rank-local generation, so a failure can be swallowed.
            self._generate_and_log(unwrapped, state, collective=False)
        except RuntimeError as e:
            if is_global_main_process():
                logger.warning(
                    f"[GenerateExamplesCallback] Generation failed at step "
                    f"{state.global_step}: {e}. Training will continue."
                )

    def _generate_and_log(self, model, state: TrainerState, *, collective: bool):
        with _gradient_checkpointing_disabled(model):
            if collective or get_global_world_size() <= 1:
                records = self._generate_all_ranks(model)
            else:
                records = self._generate_split_ranks(model)

        if is_global_main_process() and records:
            combined_df = pd.DataFrame(records)
            self._log_results(combined_df, state)
            n_print = min(3, len(combined_df))
            _pretty_print_dataframe(combined_df.sample(n_print, random_state=_SAMPLE_SEED))

    def _generate_all_ranks(self, model) -> list[dict]:
        """Forced greedy, between barriers: sampling would diverge ranks on EOS timing, and a collective forward needs
        every rank to run the same number of decode steps."""
        barrier()
        records = [
            self._generate_single(model, example, greedy=True)
            for example in tqdm(self.samples, desc="Generating examples", disable=not is_global_main_process())
        ]
        barrier()
        return records

    def _generate_split_ranks(self, model) -> list[dict]:
        """Generate every ``world_size``-th sample on this rank; rank 0 returns all ranks' records."""
        rank = get_global_rank()
        world_size = get_global_world_size()

        local_records = []
        for index in range(rank, len(self.samples), world_size):
            example = self.samples[index]
            # A per-example failure must not skip the collective all_gather_object below and hang the peers.
            try:
                record = self._generate_single(model, example)
            except Exception as e:
                record = self._build_record("", f"<generation failed: {e}>", example)
            local_records.append(record)

        gathered = [None] * world_size
        dist.all_gather_object(gathered, local_records)
        if not is_global_main_process():
            return []
        return [record for rank_records in gathered for record in rank_records]

    def _generate_single(self, model, example, greedy: bool = False) -> dict:
        input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to(model.device)
        attention_mask = torch.tensor(example["attention_mask"]).unsqueeze(0).to(model.device)

        generate_kwargs = {"max_new_tokens": self.max_new_tokens}
        if greedy:
            generate_kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)

        with torch.no_grad():
            outputs = model.generate(input_ids, attention_mask=attention_mask, **generate_kwargs)

        prompt_text = self.tokenizer.decode(input_ids.squeeze(), skip_special_tokens=False)
        completion_text = self.tokenizer.decode(outputs.squeeze(), skip_special_tokens=False)
        completion_text = completion_text[len(prompt_text) :]

        return self._build_record(prompt_text, completion_text, example)

    @staticmethod
    def _build_record(prompt_text: str, completion_text: str, example: dict) -> dict:
        """The logged row; preference examples add their chosen and rejected replies."""
        record = {"Prompt": prompt_text, "Completion": completion_text}
        if "chosen" in example:
            record["Chosen"] = example["chosen"][0]["content"]
        if "rejected" in example:
            record["Rejected"] = example["rejected"][0]["content"]
        return record

    def _log_results(self, df: pd.DataFrame, state: TrainerState):
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
