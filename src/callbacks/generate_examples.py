"""``GenerateExamplesCallback`` — sample generations at eval time, parallelism-aware."""

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

from src.distributed.context_parallel.wrapper import UlyssesCPModelWrapper
from src.distributed.expert_parallel.base_layer import find_ep_layers
from src.distributed.mesh import has_tp_dim
from src.distributed.pipeline_parallel.stage import PipelineStageModule
from src.distributed.runtime import barrier, get_global_rank, get_global_world_size, is_global_main_process
from src.models.structure import unwrap_framework_wrappers

logger = logging.getLogger(__name__)

# Fixed seed for every sampling this callback does (which prompts to show, which rows to print). It
# must be a constant rather than a run seed: the sample is drawn independently on every rank, and a
# per-rank value would make ranks generate from different prompts and gather mismatched rows.
_SAMPLE_SEED = 42

# Prompts are tail-truncated to this many characters in the log table, so a long-context prompt does
# not crowd out the completion.
_MAX_PRINTED_PROMPT_CHARS = 200


def _pretty_print_dataframe(df):
    formatted_df = df.copy()
    formatted_df["Prompt"] = formatted_df["Prompt"].apply(
        lambda x: ("..." + x[-_MAX_PRINTED_PROMPT_CHARS:]) if len(x) > _MAX_PRINTED_PROMPT_CHARS else x
    )
    logger.info("\n" + tabulate(formatted_df, headers="keys", tablefmt="simple", showindex=False))


def _generation_skip_reason(model) -> str | None:
    """Why ``model`` cannot ``generate()`` here, or None.

    The CP wrapper is searched for in the whole tree: under ``use_peft`` it sits inside
    ``PeftModel.base_model.model`` and PEFT delegates ``generate`` to it. A TP shard is a DTensor on a
    mesh naming a ``tp`` dim (HF ``tp_plan`` and toolkit attention-only TP alike); ``_tp_plan`` is
    populated on every load, TP or not, so it is no signal.
    """
    if isinstance(model, PipelineStageModule):
        return (
            "a pipeline-parallel stage holds only part of the model and cannot generate. "
            "Generate from a saved checkpoint instead."
        )
    if any(isinstance(module, UlyssesCPModelWrapper) for module in model.modules()):
        return "Context Parallelism is incompatible with autoregressive generation."
    if any(isinstance(param.data, DTensor) and has_tp_dim(param.data.device_mesh) for param in model.parameters()):
        return "Tensor Parallelism (DTensor) is incompatible with autoregressive generation."
    return None


def _uses_collective_forward(model) -> bool:
    """Whether every rank must join the forward: FSDP, EP layers or DTensor parameters.

    FSDP is read off the ``_is_fsdp_managed_module`` stamp it puts on every module it manages; a
    DTensor check alone would miss FSDP2 under ``reshard_after_forward=False``, which leaves plain
    tensors.
    """
    return (
        hasattr(model, "_is_fsdp_managed_module")
        or bool(find_ep_layers(model))
        or any(isinstance(param.data, DTensor) for param in model.parameters())
    )


@contextmanager
def _gradient_checkpointing_disabled(model):
    """Gradient checkpointing off for the block, then back on for exactly the modules that had it.

    Switching it back on everywhere would enable it on layers a partially checkpointed model
    deliberately left off, changing memory and step time for the rest of training.
    """
    enabled = [module for module in model.modules() if getattr(module, "gradient_checkpointing", False)]
    for module in enabled:
        module.gradient_checkpointing = False
    try:
        yield
    finally:
        for module in enabled:
            module.gradient_checkpointing = True


class GenerateExamplesCallback(TrainerCallback):
    """Generate text examples from a tokenized dataset during evaluation and log them.

    ``logger_backend`` (``'clearml'`` or ``'wandb'``) receives the table; any other value only prints
    the sample to the log. DDP splits the samples across ranks (gathered on rank 0); FSDP and EP
    generate on all ranks together; PP, CP and TP are skipped (incompatible with generate).
    """

    def __init__(
        self,
        preprocessed_dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        num_examples=5,
        max_new_tokens=256,
        logger_backend="wandb",
    ):
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

        ``None`` unless ``args.generate_eval_examples`` is set and ``generate_dataset`` carries
        tokenized ``input_ids``: generation replays the tokenized prompt, so an untokenized or absent
        generate split (the VLM path, for instance) cannot be generated from.
        """
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

        # The peel must drop torch.compile's OptimizedModule too: a compiled root would hide a PP stage
        # from the root isinstance check and run generate() on a partial model.
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
        """Generate every sample on every rank, between barriers.

        Forced greedy: a collective forward requires every rank to run the same number of decode
        steps, and sampling would diverge ranks on EOS timing and hang the job.
        """
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
            # A per-example failure must not skip the collective all_gather_object below, which
            # would hang the peers.
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
