"""End-to-end runs of the training entry scripts through ``halo launch``, each method's tiny recipe under
``tests/common/recipes/``."""

import argparse
import json
import random
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from datasets import Dataset, DatasetDict
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification

from halo.cli import TRAINING_ROOT, training_methods
from tests.common.models import QWEN3_0_6B
from tests.common.repo import REPO_ROOT
from tests.common.tiny_models import TINY_DENSE_FAMILY, TINY_MOE_FAMILIES, build_tiny_family_checkpoint
from tests.common.tokenizers import load_cached_tokenizer

RECIPES_DIR = REPO_ROOT / "tests" / "common" / "recipes"
FIRST_STEPS = 2
RESUMED_STEPS = 3
TRAIN_ROWS, TEST_ROWS = 32, 8
SEED = 0
# Logged on global rank 0 when a run resumes.
RESUME_LOG = "Resuming from checkpoint: "
# Rank 0 logs the first once per model resolved to flash_attention_4 (the warm-up runs at world >= 2); every rank logs
# the second on failure.
FA4_WARMUP_LOG = "FA4 kernel warmup done"
FA4_WARMUP_SKIPPED_LOG = "FA4 kernel warmup skipped"
ATTENTION_SETTINGS = ("recipe", "unset")
# At the Qwen3 tokenizer's vocabulary.
TINY_LAYERS = 2
# A MoE takes the grouped-GEMM EP layers at the scripts' defaults.
TINY_FAMILIES = {"qwen3": TINY_DENSE_FAMILY, "gpt_oss": TINY_MOE_FAMILIES["gpt_oss"]}
DEFAULT_FAMILY = "qwen3"
# The checkpoint the dense vLLM tier serves, which a method generating through it must train.
SERVED_MODEL = QWEN3_0_6B


def _turn(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def _sums(low: int = 1, high: int = 50) -> list[tuple[str, str]]:
    rng = random.Random(SEED)
    pairs = []
    for _ in range(TRAIN_ROWS + TEST_ROWS):
        a, b = rng.randint(low, high), rng.randint(low, high)
        pairs.append((f"What is {a} + {b}?", str(a + b)))
    return pairs


def _split(rows: list[dict]) -> DatasetDict:
    return DatasetDict({"train": Dataset.from_list(rows[:TRAIN_ROWS]), "test": Dataset.from_list(rows[TRAIN_ROWS:])})


def _conversations(field: str, *, answer_field: str | None = None) -> DatasetDict:
    rows = []
    for question, answer in _sums():
        row = {field: [_turn("user", question), _turn("assistant", f"The answer is {answer}.")]}
        if answer_field is not None:
            row[answer_field] = answer
        rows.append(row)
    return _split(rows)


def _preference_pairs() -> DatasetDict:
    return _split(
        [
            {
                "prompt": [_turn("user", question)],
                "chosen": [_turn("assistant", f"The answer is {answer}.")],
                "rejected": [_turn("assistant", f"The answer is {int(answer) + 7}.")],
            }
            for question, answer in _sums()
        ]
    )


def _unpaired_preferences() -> DatasetDict:
    rows = []
    for i, (question, answer) in enumerate(_sums()):
        desirable = i % 2 == 0
        shown = answer if desirable else str(int(answer) + 7)
        rows.append(
            {
                "prompt": [_turn("user", question)],
                "completion": [_turn("assistant", f"The answer is {shown}.")],
                "label": desirable,
            }
        )
    return _split(rows)


def _labelled_prompts() -> DatasetDict:
    return _split(
        [
            {"prompt": [_turn("user", question)], "label": "even" if int(answer) % 2 == 0 else "odd"}
            for question, answer in _sums()
        ]
    )


def _graded_completions() -> DatasetDict:
    rows = []
    for question, answer in _sums():
        offsets = (0, 1, 5, 20)
        rows.append(
            {
                "prompt": [_turn("user", question)],
                "completions": [[_turn("assistant", f"The answer is {int(answer) + d}.")] for d in offsets],
                "rewards": [1.0, 0.7, 0.3, 0.0],
            }
        )
    return _split(rows)


def _answered_prompts() -> DatasetDict:
    """Three-digit sums asking for a boxed answer; with thinking off the served model answers some outright and works
    others through the calculator."""
    return _split(
        [
            {"prompt": [_turn("user", f"{question} Put the final answer in \\boxed{{}}.")], "answer": answer}
            for question, answer in _sums(low=100, high=999)
        ]
    )


def _sentence_pairs() -> DatasetDict:
    return _split([{"anchor": question, "positive": f"The sum is {answer}."} for question, answer in _sums()])


@dataclass(frozen=True)
class EntryScriptCase:
    """How one method's run is fed and its export read back. ``frozen_models`` counts models a launch loads beside the
    policy (DPO/KTO reference, distillation teacher). ``server_fields`` names a live-engine method's URL and port
    fields."""

    rows: Callable[[], DatasetDict]
    export: type = AutoModelForCausalLM
    model_fields: tuple[str, ...] = ("model_name_or_path",)
    server_fields: tuple[str, str] | None = None
    frozen_models: int = 0


CASES: dict[str, EntryScriptCase] = {
    "classification": EntryScriptCase(_labelled_prompts, export=AutoModelForSequenceClassification),
    "distillation/self-distill": EntryScriptCase(lambda: _conversations("prompt", answer_field="answer")),
    "distillation/teacher-distill": EntryScriptCase(
        lambda: _conversations("messages"), model_fields=("model_name_or_path", "teacher_model"), frozen_models=1
    ),
    "embedding": EntryScriptCase(_sentence_pairs, export=SentenceTransformer),
    "grpo/environmental": EntryScriptCase(
        _answered_prompts, server_fields=("rollout_server_url", "rollout_group_port")
    ),
    "grpo/offline": EntryScriptCase(_graded_completions),
    "grpo/online": EntryScriptCase(_answered_prompts, server_fields=("vllm_server_base_url", "vllm_group_port")),
    "preference/dpo": EntryScriptCase(_preference_pairs, frozen_models=1),
    "preference/kto": EntryScriptCase(_unpaired_preferences, frozen_models=1),
    "preference/smpo": EntryScriptCase(_preference_pairs),
    "reward/reward-model": EntryScriptCase(_preference_pairs, export=AutoModelForSequenceClassification),
    "sft": EntryScriptCase(lambda: _conversations("prompt")),
}


def row_parser(description: str | None = None) -> argparse.ArgumentParser:
    """An entry-script row's arguments: method, attention and tiny family."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--method", choices=sorted(CASES), required=True)
    parser.add_argument("--family", choices=sorted(TINY_FAMILIES), default=DEFAULT_FAMILY)
    parser.add_argument(
        "--attn",
        choices=ATTENTION_SETTINGS,
        default="recipe",
        help="'unset' drops the recipe's attn_implementation, so the run trains on the script's own default",
    )
    return parser


def method_scripts() -> dict[str, Path]:
    """Every training method by its ``halo launch`` name: the path-derived names, not the stem aliases."""
    return {
        name: script
        for name, script in training_methods(REPO_ROOT).items()
        if name == "/".join(script.relative_to(REPO_ROOT / TRAINING_ROOT).with_suffix("").parts).replace("_", "-")
    }


def recipe_path(method: str) -> Path:
    script = method_scripts()[method]
    return RECIPES_DIR / script.relative_to(REPO_ROOT / TRAINING_ROOT).with_suffix(".yaml")


def write_run_config(
    method: str,
    scratch: Path,
    *,
    model: str,
    dataset_dir: str,
    server_url: str | None = None,
    attention_unset: bool = False,
) -> Path:
    """``method``'s recipe plus the run's settings, written to ``scratch``; ``dataset`` cannot be a CLI override, so
    the config carries them all."""
    case = CASES[method]
    config = yaml.safe_load(recipe_path(method).read_text())
    if attention_unset:
        del config["attn_implementation"]
    config.update(dict.fromkeys(case.model_fields, model))
    config.update(dataset=dataset_dir, output_dir=str(scratch / "output"))
    if case.server_fields is not None:
        config[case.server_fields[0]] = server_url
    path = scratch / "run.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def launch_command(
    method: str,
    config: Path,
    *,
    nproc: int,
    port: int | None,
    overrides: list[str],
    accelerate_config: Path | None = None,
) -> list[str]:
    """``halo launch <method> <config> -n <nproc> [--port <port>] [--accelerate <config>] <overrides>`` as an argv."""
    launcher = ["-n", str(nproc)] + ([] if port is None else ["--port", str(port)])
    if accelerate_config is not None:
        launcher += ["--accelerate", str(accelerate_config)]
    return [sys.executable, "-m", "halo.cli", "launch", method, str(config), *launcher, *overrides]


def run_launch(argv: list[str]) -> subprocess.CompletedProcess:
    """One launch, streamed to stdout and returned whole. Unbounded: the GPU launcher's row timeout kills a hung
    launch."""
    lines = []
    with subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as launch:
        for line in launch.stdout:
            print(line, end="", flush=True)
            lines.append(line)
    return subprocess.CompletedProcess(argv, launch.returncode, stdout="".join(lines))


def first_launch_overrides() -> list[str]:
    return [f"--max_steps={FIRST_STEPS}", "--save_steps=1"]


def resume_overrides() -> list[str]:
    return [f"--max_steps={RESUMED_STEPS}", "--save_steps=1", "--resume_from_checkpoint=true"]


def checkpoint_dir(output_dir: Path, step: int) -> Path:
    return output_dir / f"checkpoint-{step}"


def state_stamps(output_dir: Path, steps: range) -> dict[int, int]:
    """Each step's ``trainer_state.json`` mtime: a resume leaves earlier checkpoints alone, while a restart from step 0
    rewrites them."""
    return {step: (checkpoint_dir(output_dir, step) / "trainer_state.json").stat().st_mtime_ns for step in steps}


def _log_history(checkpoint: Path) -> list[dict]:
    return json.loads((checkpoint / "trainer_state.json").read_text())["log_history"]


def logged_steps(checkpoint: Path) -> list[dict]:
    return [entry for entry in _log_history(checkpoint) if "loss" in entry]


def trained_steps(checkpoint: Path) -> list[int]:
    return [entry["step"] for entry in logged_steps(checkpoint)]


def had_a_learning_signal(checkpoint: Path) -> bool:
    """Whether any step logged a nonzero gradient norm; a run of tied GRPO groups has none and leaves its export on the
    base weights."""
    return any(entry.get("grad_norm", 0.0) > 0.0 for entry in _log_history(checkpoint))


def build_tiny_model(path: Path, family: str) -> Path:
    """Save the tiny model of ``family`` a run trains, with the Qwen3 tokenizer and chat template, at ``path``."""
    build_tiny_family_checkpoint(
        TINY_FAMILIES[family],
        str(path),
        load_cached_tokenizer(QWEN3_0_6B),
        overrides={"num_hidden_layers": TINY_LAYERS},
    )
    return path


def export_checks(export: type, checkpoint: Path, base: str, *, learned: bool) -> dict[str, bool]:
    """Whether ``checkpoint`` loads as ``export`` with every weight and, if ``learned``, holds weights moved off
    ``base``."""
    base_state = AutoModelForCausalLM.from_pretrained(base).state_dict()
    if export is SentenceTransformer:
        backbone = SentenceTransformer(str(checkpoint), device="cpu")[0].auto_model
        trained = {f"model.{name}": tensor for name, tensor in backbone.state_dict().items()}
        decoder = {name for name in base_state if name.startswith("model.")}
        # The pipeline builds its backbone whatever the files hold; the files must hold every key of it.
        _, info = AutoModel.from_pretrained(checkpoint, output_loading_info=True)
        complete = set(trained) == decoder and not (
            info["missing_keys"] or info["unexpected_keys"] or info["mismatched_keys"]
        )
    else:
        model, info = export.from_pretrained(checkpoint, output_loading_info=True)
        trained = model.state_dict()
        complete = not (info["missing_keys"] or info["unexpected_keys"] or info["mismatched_keys"])
    checks = {"export_loads_every_weight": complete}
    if learned:
        shared = [
            name for name, tensor in trained.items() if name in base_state and base_state[name].shape == tensor.shape
        ]
        moved = any(not torch.equal(trained[name].float(), base_state[name].float()) for name in shared)
        checks["export_holds_trained_weights"] = bool(shared) and moved
    return checks
