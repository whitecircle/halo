#!/usr/bin/env python
"""Tree packing against flat packing end to end: Qwen3-0.6B under FSDP2 data parallelism, each arm launched as a
user launches it (``halo launch sft <config> -n N``), so ``scripts/training/sft.py`` builds the rows, the collator and
the tree attention.

The corpus puts long documents of numbered records in the user turn, which is never trained. Each document has five
rows: two open with the same question, two ask one question answered in different words (one prompt, shared up to its
last token), one asks its own; a sixth of the rows share nothing. No trained token is shared, so both arms train the
same tokens.

Every optimizer step is one whole epoch in both arms: per-device batch 1 and gradient accumulation = packed rows /
ranks, at the largest ``max_length`` from :data:`MAX_LENGTH` down in :data:`MAX_LENGTH_STEP` steps at which both arms
pack a row count the ranks divide (a padded duplicate row would weigh one example twice). The counts come from the
script's own data path over each arm's config, before either launch. With ``average_tokens_across_devices`` each
logged loss is the token-weighted mean over the same trained tokens, so the tree arm must reproduce the flat arm's: at
step 1, on identical weights, loss and gradient norm within :data:`TOLERANCE` (the tree kernels' bf16 rounding); at
step 2 the loss within the same and the gradient norm within :data:`UPDATED_GRAD_NORM_TOLERANCE`, since after one
update each arm also carries the run-to-run drift of a nondeterministic bf16 backward. A tree row read as one causal
sequence, its siblings attending each other, misses the step-1 loss by far more.

Both arms log the same trained tokens at every step and one epoch per step, and the tree arm stores at most
:data:`MAX_TREE_STORED_FRACTION` of the flat arm's tokens. FA4 only: the ``split`` form needs its log-sum-exp
gradient. The row runs this file as one process (``launcher="python"``), which starts the ranks.

    HALO_TEST_NPROC=2 python tests/gpu/data/test_tree_packing_e2e.py
"""

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from datasets import Dataset, DatasetDict
from transformers import AutoConfig
from trl import SFTConfig

from halo.configs.parallelism_arguments import ParallelismArguments
from halo.configs.parser import HaloArgumentParser
from halo.configs.script_arguments import ModelArguments
from halo.configs.sft import SFTScriptArguments
from halo.data.sources.loading import load_datasets_auto
from halo.entrypoints.model_setup import apply_max_length, apply_text_path_padding_side
from halo.env import env_int
from scripts.training.sft import _prepare_text_data
from tests.common.entry_scripts import checkpoint_dir, launch_command, logged_steps, run_launch
from tests.common.harness import gpu_test_main, skip_unless_blackwell
from tests.common.models import QWEN3_0_6B
from tests.common.ports import free_port
from tests.common.rank_log import log
from tests.common.repo import REPO_ROOT
from tests.common.tokenizers import load_cached_tokenizer

STEPS = 2
# Step-1 loss parts by 6.4e-4 and grad norm by ~5e-3 (the tree kernels' bf16 rounding); after one update each arm
# drifts against itself, flat by 7e-3, so step 2's norm parts by up to 1.4e-2. A tree row read as one causal sequence
# misses step 1's loss by 1.37.
TOLERANCE = 1e-2
UPDATED_GRAD_NORM_TOLERANCE = 5e-2
MAX_TREE_STORED_FRACTION = 0.5
MAX_LENGTH = 8192
MAX_LENGTH_STEP = 256
# Above the longest conversation, so no candidate length drops a row from either arm.
MIN_MAX_LENGTH = 4096
# The tree builder groups rows within one map worker's shard, so the probe and both launches pin one count.
DATASET_NUM_PROC = 4
ATTENTION = "flash_attention_4"
SHARE_MARKER = "\nQuestions:\n"
SHARE_REQUIRES = "\nDocument:\n"
ASSISTANT_TEMPLATE = "<|im_start|>assistant\n"
CHAT_TEMPLATE = REPO_ROOT / "jinja-templates" / "qwen3" / "qwen3-multiturn.jinja"
CONVERSATION_FIELD = "prompt"
SEED = 0
# Documents with their five rows each, of DOCUMENT_RECORDS records; rows sharing nothing, of UNSHARED_RECORDS.
DOCUMENTS, DOCUMENT_RECORDS = 16, 40
UNSHARED_ROWS, UNSHARED_RECORDS = 16, 20
TEST_ROWS, TEST_RECORDS = 4, 10
HEADER = "You are given a document of numbered records and questions about it."

_MONTHS = "January, February, March, April, May, June, July, August, September, October".split(", ")
_NAMES = (
    "Alma Reyes, Boris Lind, Chen Wei, Dara Okafor, Elif Demir, Felix Brandt, Greta Holm, Hugo Marin, "
    "Ines Duarte, Jonas Berg, Kaito Mori, Lena Novak, Marek Hale, Nadia Farah, Oskar Vidal, Priya Rao, "
    "Quinn Avery, Rosa Klein, Samir Haddad, Tove Ek, Umar Siddiq, Vera Costa, Wim Peeters, Xenia Lutz"
).split(", ")
_CITIES = (
    "Arden, Bellmoor, Calder, Dunmore, Elsworth, Fairhaven, Glenrock, Harlow, Ivybridge, Jarrow, "
    "Kestrel Bay, Linwood, Marston, Northfield, Oakhurst, Pembury, Quarry Hill, Redcliff, Stanmore, "
    "Thornbury, Upton, Valemont, Westbrook, Yarrow"
).split(", ")
_ITEMS = (
    "barrels of cider, wool bales, copper kettles, crates of pears, fishing nets, oak beams, salt sacks, "
    "lanterns, bolts of linen, clay jars, iron hinges, bushels of rye, glass bottles, spools of rope"
).split(", ")
_PLACES = "harbour, town hall, old mill, market square, river dock, granary, north gate, chapel yard".split(", ")
_ADJECTIVES = "narrow, flooded, crowded, silent, restored, frozen, dusty, newly paved, windswept".split(", ")
_NOUNS = "bridge, road, warehouse, canal, orchard, quay, tannery, school, watchtower, ferry".split(", ")
_VERBS = "reopened, closed for repairs, was rebuilt, flooded again, stayed busy, lost its roof".split(", ")
_EVENTS = "spring fair, first frost, long drought, harvest feast, great storm, census, toll dispute".split(", ")

PARSER = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
PARSER.parse_args()
NPROC = env_int("HALO_TEST_NPROC", 2)
if NPROC < 2:
    PARSER.error("the arms train under FSDP2 data parallelism, which needs two ranks or more")
skip_unless_blackwell("tree packing's split form")


@dataclass(frozen=True)
class Record:
    index: int
    month: str
    year: int
    name: str
    city: str
    qty: int
    item: str
    place: str
    adjective: str
    noun: str
    verb: str
    event: str

    def sentence(self) -> str:
        return (
            f"Record {self.index}: In {self.month} {self.year}, {self.name} of {self.city} counted {self.qty} "
            f"{self.item} at the {self.place}, and noted that the {self.adjective} {self.noun} {self.verb} after "
            f"the {self.event}."
        )

    def restated(self, phrasing: int) -> str:
        if phrasing:
            return (
                f"According to record {self.index}, {self.name} of {self.city} counted {self.qty} {self.item} at "
                f"the {self.place} in {self.month} {self.year}; the {self.adjective} {self.noun} {self.verb}."
            )
        return (
            f"Record {self.index} says that in {self.month} {self.year}, {self.name} of {self.city} counted "
            f"{self.qty} {self.item} at the {self.place}, and that the {self.adjective} {self.noun} {self.verb} "
            f"after the {self.event}."
        )


def _records(rng: random.Random, count: int) -> list[Record]:
    return [
        Record(
            index=i + 1,
            month=rng.choice(_MONTHS),
            year=rng.randint(1801, 1899),
            name=rng.choice(_NAMES),
            city=rng.choice(_CITIES),
            qty=rng.randint(3, 990),
            item=rng.choice(_ITEMS),
            place=rng.choice(_PLACES),
            adjective=rng.choice(_ADJECTIVES),
            noun=rng.choice(_NOUNS),
            verb=rng.choice(_VERBS),
            event=rng.choice(_EVENTS),
        )
        for i in range(count)
    ]


def _pairs(rng: random.Random, records: list[Record], count: int) -> list[tuple[Record, Record]]:
    starts = rng.sample(range(len(records) - 1), count)
    return [(records[start], records[start + 1]) for start in starts]


def _question(pair: tuple[Record, Record]) -> str:
    first, second = pair
    return (
        f"Using only the document above, restate what records {first.index} and {second.index} say about "
        f"{first.name} of {first.city} and {second.name} of {second.city}, including each month, year and quantity."
    )


def _answer(pairs: list[tuple[Record, Record]], phrasing: int) -> str:
    parts = [f"{first.restated(phrasing)} {second.restated(phrasing)}" for first, second in pairs]
    if len(parts) == 1:
        return parts[0]
    return "\n".join(f"{n}. {part}" for n, part in enumerate(parts, start=1))


def _document_row(document: str, pairs: list[tuple[Record, Record]], phrasing: int = 0) -> list[dict]:
    questions = ",\n".join(f'  "{n}": "{_question(pair)}"' for n, pair in enumerate(pairs, start=1))
    user = f"{HEADER}\n{SHARE_REQUIRES}{document}{SHARE_MARKER}{{\n{questions}\n}}\nAnswer in plain sentences."
    return [{"role": "user", "content": user}, {"role": "assistant", "content": _answer(pairs, phrasing)}]


def _document_rows(rng: random.Random) -> list[list[dict]]:
    """Five rows over one document: two share a question, two ask one question in two phrasings, the last its own."""
    records = _records(rng, DOCUMENT_RECORDS)
    document = "\n".join(record.sentence() for record in records)
    pairs = _pairs(rng, records, 5)
    return [
        _document_row(document, [pairs[0]]),
        _document_row(document, [pairs[0], pairs[1]]),
        _document_row(document, [pairs[2]]),
        _document_row(document, [pairs[2]], phrasing=1),
        _document_row(document, [pairs[4]]),
    ]


def _unshared_row(rng: random.Random, records_count: int, *, with_marker: bool) -> list[dict]:
    records = _records(rng, records_count)
    if with_marker:
        return _document_row("\n".join(record.sentence() for record in records), _pairs(rng, records, 1))
    target = rng.choice(records)
    user = (
        "Here are my notes:\n"
        + "\n".join(record.sentence() for record in records)
        + f"\nWhich record mentions {target.name} of {target.city}? Quote it and restate it."
    )
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": f"{target.sentence()} {target.restated(0)}"},
    ]


def _conversations(seed: int, *, documents: int, unshared: int, unshared_records: int) -> list[list[dict]]:
    """Shuffled, as a dataset arrives: the tree path orders rows by their shared text itself."""
    rng = random.Random(seed)
    rows = [row for _ in range(documents) for row in _document_rows(rng)]
    rows += [_unshared_row(rng, unshared_records, with_marker=i % 2 == 0) for i in range(unshared)]
    rng.shuffle(rows)
    return rows


def _write_corpus(path: Path) -> int:
    train = _conversations(SEED, documents=DOCUMENTS, unshared=UNSHARED_ROWS, unshared_records=UNSHARED_RECORDS)
    test = _conversations(SEED + 1, documents=0, unshared=TEST_ROWS, unshared_records=TEST_RECORDS)
    DatasetDict(
        {
            "train": Dataset.from_dict({CONVERSATION_FIELD: train}),
            "test": Dataset.from_dict({CONVERSATION_FIELD: test}),
        }
    ).save_to_disk(str(path))
    return len(train)


def _config(*, tree: bool, dataset: Path, output_dir: Path, max_length: int, grad_accum: int) -> dict:
    values = {
        "model_name_or_path": QWEN3_0_6B,
        "attn_implementation": ATTENTION,
        "dataset": str(dataset),
        "conversation_field": CONVERSATION_FIELD,
        "chat_template": str(CHAT_TEMPLATE),
        "force_chat_template": True,
        "assistant_message_template": ASSISTANT_TEMPLATE,
        "train_on_completions_only": True,
        "packing": True,
        "packing_strategy": "bfd",
        "max_length": max_length,
        "dataset_num_proc": DATASET_NUM_PROC,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": grad_accum,
        "average_tokens_across_devices": True,
        "gradient_checkpointing": True,
        "max_steps": STEPS,
        "learning_rate": 1.0e-5,
        "lr_scheduler_type": "constant",
        "optim": "adamw_torch_fused",
        "max_grad_norm": 1.0,
        "logging_steps": 1,
        "eval_strategy": "no",
        "save_strategy": "steps",
        "save_steps": STEPS,
        "save_only_model": True,
        "report_to": "none",
        "output_dir": str(output_dir),
        "dataloader_num_workers": 0,
    }
    if tree:
        values |= {
            "tree_packing": True,
            "tree_share_markers": [SHARE_MARKER],
            "tree_share_requires": SHARE_REQUIRES,
            "tree_full_attention": "split",
        }
    return values


def _packed_train_rows(values: dict, hf_config) -> int:
    """The packed training rows a launch would train, from the SFT script's own data path on the CPU."""
    parser = HaloArgumentParser((SFTScriptArguments, SFTConfig, ModelArguments, ParallelismArguments))
    args, sft_config, model_args, _ = parser.parse_dict({**values, "bf16": False, "use_cpu": True})
    splits, _ = load_datasets_auto(
        args.dataset,
        args.test_size,
        args.dataset_ratio,
        conversation_field=args.conversation_field,
        tools_field=args.tools_field,
    )
    tokenizer = apply_max_length(sft_config, args, None, load_cached_tokenizer(QWEN3_0_6B))
    apply_text_path_padding_side(tokenizer, vlm_run=False)
    train, *_ = _prepare_text_data(
        splits,
        None,
        args,
        sft_config,
        model_args,
        tokenizer,
        tokenizer,
        SimpleNamespace(is_cp_mode=False, cp_size=1, pp_size=1),
        hf_config,
    )
    return len(train)


def _even_max_length(dataset: Path, scratch: Path) -> tuple[int, dict[str, int]]:
    """The largest candidate ``max_length`` at which both arms pack a row count the ranks divide, and those counts."""
    hf_config = AutoConfig.from_pretrained(QWEN3_0_6B)
    hf_config._attn_implementation = ATTENTION
    for max_length in range(MAX_LENGTH, MIN_MAX_LENGTH - 1, -MAX_LENGTH_STEP):
        rows = {
            arm: _packed_train_rows(
                _config(tree=arm == "tree", dataset=dataset, output_dir=scratch, max_length=max_length, grad_accum=1),
                hf_config,
            )
            for arm in ("flat", "tree")
        }
        log(f"max_length {max_length}: packed training rows {rows}")
        if all(count % NPROC == 0 for count in rows.values()):
            return max_length, rows
    raise RuntimeError(
        f"no max_length from {MAX_LENGTH} down to {MIN_MAX_LENGTH} packs a row count {NPROC} ranks divide in both arms"
    )


def _relative(actual: float, expected: float) -> float:
    return abs(actual - expected) / abs(expected)


def _per_step(steps: list[dict], counter: str) -> list[float]:
    totals = [float(entry[counter]) for entry in steps]
    return [total - previous for total, previous in zip(totals, [0.0, *totals[:-1]], strict=True)]


@gpu_test_main(prefix="tree_packing_e2e", partial_state=False, process_group=False)
def run(ctx):
    scratch = Path(ctx.output_dir)
    dataset = scratch / "corpus"
    conversations = _write_corpus(dataset)
    max_length, rows = _even_max_length(dataset, scratch / "probe")
    log(f"{conversations} conversations; max_length {max_length} packs {rows} training rows")

    checks: dict[str, bool] = {}
    steps: dict[str, list[dict]] = {}
    for arm, count in rows.items():
        output = scratch / arm
        config = scratch / f"{arm}.yaml"
        values = _config(
            tree=arm == "tree", dataset=dataset, output_dir=output, max_length=max_length, grad_accum=count // NPROC
        )
        # JSON is YAML 1.2, and its escapes keep the newline markers exact.
        config.write_text(json.dumps(values, indent=2) + "\n")
        log(f"{arm}: {STEPS} steps of {count // NPROC} micro-batches on {NPROC} ranks")
        launch = run_launch(launch_command("sft", config, nproc=NPROC, port=free_port(), overrides=[]))
        last = checkpoint_dir(output, STEPS)
        checks[f"{arm}_launch_exits_cleanly"] = launch.returncode == 0
        checks[f"{arm}_logs_every_step"] = (last / "trainer_state.json").is_file() and len(logged_steps(last)) == STEPS
        if not checks[f"{arm}_logs_every_step"]:
            return {"checks": checks}
        steps[arm] = logged_steps(last)

    flat, tree = steps["flat"], steps["tree"]
    trained = {arm: _per_step(logged, "num_unmasked_output_tokens_seen") for arm, logged in steps.items()}
    stored = {arm: _per_step(logged, "num_attended_tokens_seen") for arm, logged in steps.items()}
    loss_deviation = [_relative(t["loss"], f["loss"]) for t, f in zip(tree, flat, strict=True)]
    grad_norm_deviation = [_relative(t["grad_norm"], f["grad_norm"]) for t, f in zip(tree, flat, strict=True)]
    log(f"loss per step: flat {[e['loss'] for e in flat]}, tree {[e['loss'] for e in tree]}")
    log(f"grad norm per step: flat {[e['grad_norm'] for e in flat]}, tree {[e['grad_norm'] for e in tree]}")
    log(f"trained tokens per step {trained}; stored tokens per step {stored}")
    checks |= {
        "every_step_logs_a_finite_loss_and_grad_norm": all(
            math.isfinite(entry["loss"]) and math.isfinite(entry["grad_norm"]) for entry in flat + tree
        ),
        "every_step_is_one_whole_epoch": all(
            math.isclose(entry["epoch"], entry["step"], abs_tol=1e-6) for entry in flat + tree
        ),
        "both_arms_train_the_same_tokens_every_step": trained["flat"] == trained["tree"]
        and len(set(trained["flat"])) == 1
        and trained["flat"][0] > 0,
        "tree_stores_a_fraction_of_the_flat_tokens": all(
            t <= MAX_TREE_STORED_FRACTION * f for t, f in zip(stored["tree"], stored["flat"], strict=True)
        ),
        "tree_loss_matches_flat_every_step": all(deviation <= TOLERANCE for deviation in loss_deviation),
        "tree_grad_norm_matches_flat_on_identical_weights": grad_norm_deviation[0] <= TOLERANCE,
        "tree_grad_norm_matches_flat_after_an_update": all(
            deviation <= UPDATED_GRAD_NORM_TOLERANCE for deviation in grad_norm_deviation[1:]
        ),
    }
    metrics = {
        "max_length": max_length,
        "flat_packed_rows": rows["flat"],
        "tree_packed_rows": rows["tree"],
        "tree_stored_token_fraction": stored["tree"][0] / stored["flat"][0],
        **{f"loss_rel_deviation_step{n}": value for n, value in enumerate(loss_deviation, start=1)},
        **{f"grad_norm_rel_deviation_step{n}": value for n, value in enumerate(grad_norm_deviation, start=1)},
    }
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
