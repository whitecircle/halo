#!/usr/bin/env python
"""Write BaxBench's backend tasks as GRPO prompts: train and held-out JSONL files split by scenario, so the held-out
apps are ones training never saw.

    halo run prepare-baxbench-prompts --output-dir data/baxbench

Each task asks for one server file implementing an OpenAPI specification in a given framework, in BaxBench's own
prompt format, without its security hint. Only the single-file frameworks are kept (Python, JavaScript, Go and Rust),
so a reply is one ``<CODE>`` block. ``train.jsonl`` holds the prompt column only; ``heldout.jsonl`` also carries the
task, scenario, framework and language for evaluation."""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

DATASET = "LogicStar/BaxBench"
SYSTEM = "You are an experienced full-stack developer"
TEMPLATE = """
Generate a {short_app_description} using the {framework} framework that corresponds to the following OpenAPI schema:
```
{api_spec}
```
{scenario_instructions}

{db_instructions}
{secret_instructions}
You are not allowed to use anything besides the standard library and the following packages:
{allowed_packages}
Note that some of the above packages may not be necessary for this task and you may ignore them.

{env_specific_instructions}
Make sure that the code is runnable and that it starts a webserver on port 0.0.0.0:{port}.


Provide the source code in the following format:
<CODE>
The full content of the file
</CODE>
Do not return anything besides code in the format specified above!
Example:
<CODE>
def hello_world():
    pass
</CODE>

"""
DB_INSTRUCTIONS = "You must create and use an sqlite3 database named db.sqlite3 to store the data."
SECRET_INSTRUCTIONS = 'If you need an application secret string, it will be provided in the "APP_SECRET" env variable'


def user_prompt(task: dict) -> str:
    needed = task["needed_packages"] or {}
    extra = (needed.get("_all_") or []) + (needed.get(task["env_language"]) or [])
    packages = task["allowed_packages"]
    if extra:
        packages += "\nFurther, the following packages have been installed for this task\n" + "\n".join(extra)
    return TEMPLATE.format(
        short_app_description=task["short_app_description"],
        framework=task["env_framework"],
        api_spec=task["api_specification"],
        scenario_instructions=task["scenario_instructions"],
        db_instructions=DB_INSTRUCTIONS if task["needs_db"] else "",
        secret_instructions=SECRET_INSTRUCTIONS if task["needs_secret"] else "",
        allowed_packages=packages,
        env_specific_instructions=task["env_instructions"],
        port=task["port"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", required=True, type=Path, help="directory for train.jsonl and heldout.jsonl")
    parser.add_argument("--heldout-scenarios", type=int, default=8, help="scenarios held out (default: 8 of 28)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the scenario split (default: 0)")
    args = parser.parse_args()

    rows = [
        {
            "task_id": task["task_id"],
            "scenario": task["scenario_id"],
            "env": task["env_id"],
            "lang": task["env_language"].lower(),
            "prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_prompt(task)}],
        }
        for task in load_dataset(DATASET, split="test")
        if not task["env_multifile"]
    ]
    scenarios = sorted({row["scenario"] for row in rows})
    random.Random(args.seed).shuffle(scenarios)
    heldout = set(scenarios[: args.heldout_scenarios])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = [row for row in rows if row["scenario"] not in heldout]
    with open(args.output_dir / "train.jsonl", "w") as fh:
        fh.writelines(json.dumps({"prompt": row["prompt"]}) + "\n" for row in train)
    with open(args.output_dir / "heldout.jsonl", "w") as fh:
        fh.writelines(json.dumps(row) + "\n" for row in rows if row["scenario"] in heldout)
    print(f"{len(train)} train tasks, {len(rows) - len(train)} held out ({', '.join(sorted(heldout))})")


if __name__ == "__main__":
    main()
