#!/usr/bin/env python
"""LoRA / QLoRA on the preference + classification-head trainers (dense FSDP2, + adapter checkpoint).

One file, five trainers (``--trainer``), two modes (``--mode``): confirms every non-GRPO/non-distill
trainer trains adapters with a frozen base and writes a round-trippable adapter checkpoint, under both
plain LoRA (FSDP2 grad sync) and 4-bit QLoRA (AllReduce-hook grad sync). Dense Qwen3-0.6B throughout.

    dpo / smpo            — preference {prompt, chosen, rejected}, CausalLM, CAUSAL_LM adapters
    kto                   — unpaired {prompt, completion, label}, CausalLM, CAUSAL_LM adapters
    reward                — preference, SequenceClassification(num_labels=1), SEQ_CLS adapters + score head
    classification        — {input_ids, labels}, SequenceClassification(num_labels=2), SEQ_CLS + score head

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/lora/test_lora_pref_heads.py --trainer dpo --mode lora
"""

import argparse
import math
import random

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import DPOConfig, KTOConfig, RewardConfig

from src.args.common_script_args import CommonScriptArguments
from src.configs.classification_config import ClassificationConfig
from src.configs.smpo_config import SmoothMarginPOConfig
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from src.models.loading.tokenizer_setup import setup_model_and_tokenizer
from src.trainers.preference.dpo import DistributedDPOTrainer
from src.trainers.preference.kto import DistributedKTOTrainer
from src.trainers.preference.smpo import SmoothMarginPOTrainer
from src.trainers.reward.bradley_terry import DistributedRewardTrainer
from src.trainers.reward.classification import ClassificationTrainer
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.peft_helpers import (
    assert_adapter_checkpoint,
    assert_adapters_moved,
    assert_only_adapters_trainable,
    snapshot_adapters,
    unwrap,
)
from tests.common.utils import log

MODEL = QWEN3_0_6B
MAX_STEPS = 3
N = 32
MAXLEN = 256
SEED = 42
_IS_SEQ_CLS = {"reward", "classification"}


def _pref_rows(n):
    rng = random.Random(SEED)
    rows = []
    for _ in range(n):
        a, b = rng.randint(1, 50), rng.randint(1, 50)
        rows.append(
            {
                "prompt": [{"role": "user", "content": f"What is {a} + {b}?"}],
                "chosen": [{"role": "assistant", "content": f"The answer is {a + b}."}],
                "rejected": [{"role": "assistant", "content": f"The answer is {a + b + rng.randint(1, 9)}."}],
            }
        )
    return Dataset.from_list(rows)


def _pref_rows_str(tokenizer, n):
    """SMPO tokenizes pre-templated STRING prompt/chosen/rejected (it concatenates prompt+completion),
    unlike DPO/KTO which accept conversational message lists and template internally."""
    rng = random.Random(SEED)
    rows = []
    for _ in range(n):
        a, b = rng.randint(1, 50), rng.randint(1, 50)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"What is {a} + {b}?"}], tokenize=False, add_generation_prompt=True
        )
        rows.append(
            {
                "prompt": prompt,
                "chosen": f"The answer is {a + b}.",
                "rejected": f"The answer is {a + b + rng.randint(1, 9)}.",
            }
        )
    return Dataset.from_list(rows)


def _kto_rows(n):
    rng = random.Random(SEED)
    rows = []
    for _ in range(n):
        a, b = rng.randint(1, 50), rng.randint(1, 50)
        correct = rng.random() < 0.5
        val = a + b if correct else a + b + rng.randint(1, 9)
        rows.append(
            {
                "prompt": [{"role": "user", "content": f"What is {a} + {b}?"}],
                "completion": [{"role": "assistant", "content": f"The answer is {val}."}],
                "label": correct,
            }
        )
    return Dataset.from_list(rows)


def _reward_rows(n, tokenizer):
    rng = random.Random(SEED)
    rows = []
    for _ in range(n):
        a, b = rng.randint(1, 50), rng.randint(1, 50)
        chosen = [
            {"role": "user", "content": f"What is {a} + {b}?"},
            {"role": "assistant", "content": f"The answer is {a + b}."},
        ]
        rejected = [
            {"role": "user", "content": f"What is {a} + {b}?"},
            {"role": "assistant", "content": f"It is {a + b + rng.randint(1, 9)}."},
        ]
        rows.append(
            {
                "chosen": tokenizer.apply_chat_template(chosen, tokenize=False),
                "rejected": tokenizer.apply_chat_template(rejected, tokenize=False),
            }
        )
    return Dataset.from_list(rows)


def _cls_rows(n, tokenizer):
    rng = random.Random(SEED)
    rows = []
    for _ in range(n):
        label = rng.randint(0, 1)
        text = "This is great, I loved it." if label == 1 else "This was terrible and boring."
        enc = tokenizer(text, truncation=True, max_length=MAXLEN)
        rows.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": label})
    return Dataset.from_list(rows)


def _load_model(trainer: str, mode: str):
    quant = None
    if mode == "qlora":
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    common = {
        "dtype": torch.bfloat16,
        "trust_remote_code": True,
        "attn_implementation": "sdpa",
        "quantization_config": quant,
    }
    if trainer == "reward":
        return AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1, **common)
    if trainer == "classification":
        return AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2, **common)
    return AutoModelForCausalLM.from_pretrained(MODEL, **common)


def _peft_config(trainer: str) -> LoraConfig:
    seq_cls = trainer in _IS_SEQ_CLS
    return LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.SEQ_CLS if seq_cls else TaskType.CAUSAL_LM,
        modules_to_save=["score"] if seq_cls else None,
    )


def _build_trainer(trainer, model, tokenizer, peft_config, output_dir, parallelism_config):
    """Construct the requested trainer with a tiny config + dataset. Returns the trainer."""
    common = {
        "output_dir": output_dir,
        "max_steps": MAX_STEPS,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-3,
        "bf16": True,
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": "none",
        "dataloader_drop_last": True,
        "gradient_checkpointing": False,
        "use_liger_kernel": False,
        "fsdp": "",
    }
    if trainer == "dpo":
        cfg = DPOConfig(beta=0.1, max_length=MAXLEN, **common)
        return DistributedDPOTrainer(
            model=model,
            args=cfg,
            train_dataset=_pref_rows(N),
            processing_class=tokenizer,
            peft_config=peft_config,
            parallelism_config=parallelism_config,
        )
    if trainer == "smpo":
        cfg = SmoothMarginPOConfig(beta=0.1, max_length=MAXLEN, max_prompt_length=64, **common)
        return SmoothMarginPOTrainer(
            model=model,
            args=cfg,
            train_dataset=_pref_rows_str(tokenizer, N),
            processing_class=tokenizer,
            peft_config=peft_config,
            parallelism_config=parallelism_config,
        )
    if trainer == "kto":
        cfg = KTOConfig(beta=0.1, max_length=MAXLEN, **common)
        return DistributedKTOTrainer(
            model=model,
            args=cfg,
            train_dataset=_kto_rows(N),
            processing_class=tokenizer,
            peft_config=peft_config,
            parallelism_config=parallelism_config,
        )
    if trainer == "reward":
        cfg = RewardConfig(max_length=MAXLEN, **common)
        return DistributedRewardTrainer(
            model=model,
            args=cfg,
            train_dataset=_reward_rows(N, tokenizer),
            processing_class=tokenizer,
            peft_config=peft_config,
            parallelism_config=parallelism_config,
        )
    if trainer == "classification":
        cfg = ClassificationConfig(remove_unused_columns=False, **common)
        return ClassificationTrainer(
            model=model,
            args=cfg,
            train_dataset=_cls_rows(N, tokenizer),
            processing_class=tokenizer,
            peft_config=peft_config,
            parallelism_config=parallelism_config,
        )
    raise ValueError(trainer)


def run(ctx) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", required=True, choices=["dpo", "smpo", "kto", "reward", "classification"])
    parser.add_argument("--mode", default="lora", choices=["lora", "qlora"])
    args = parser.parse_args()
    trainer_name, mode = args.trainer, args.mode

    log(f"\n{'=' * 70}\n  {trainer_name.upper()} adapter test — mode={mode}, world={ctx.world_size}\n{'=' * 70}")
    checks: dict[str, bool] = {}

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = _load_model(trainer_name, mode)
    # Through the seam the training scripts run (apply_max_length → setup_model_and_tokenizer),
    # never a hand-written config.pad_token_id: the two seq-cls heads pool against that id, so
    # recording it here by hand would keep this test green through a regression in the sync.
    tokenizer = setup_model_and_tokenizer(CommonScriptArguments(), model, tokenizer, MAXLEN)

    parallelism_config = ParallelismConfig()
    trainer = _build_trainer(
        trainer_name, model, tokenizer, _peft_config(trainer_name), ctx.output_dir, parallelism_config
    )

    wrapped = unwrap(trainer.model)
    ok, detail = assert_only_adapters_trainable(wrapped)
    checks["only_adapters_trainable"] = ok
    log(f"  [trainable] {detail}")
    before = snapshot_adapters(wrapped, expert_lora=False)

    barrier()
    result = trainer.train()
    checks["loss_finite"] = math.isfinite(result.training_loss)
    checks["steps_done"] = result.global_step == MAX_STEPS
    log(f"  training_loss={result.training_loss:.5f}, steps={result.global_step}")

    after = snapshot_adapters(unwrap(trainer.model), expert_lora=False)
    ok, detail = assert_adapters_moved(before, after)
    checks["adapters_moved"] = ok
    log(f"  [moved] {detail}")

    ok, detail = assert_adapter_checkpoint(trainer, ctx.output_dir, ctx.rank, expert_lora=False)
    checks["adapter_checkpoint"] = ok
    log(f"  [checkpoint] {detail}")

    return {"checks": checks}


main = gpu_test_main(min_world_size=2, prefix="lora_pref_heads")(run)

if __name__ == "__main__":
    main()
