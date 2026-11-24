# Honesty Environment

`HonestyEnvironment` (`halo/environments/tasks/honesty.py`, registry name `honesty`) trains on the agent honesty harness's probes: each row is one probe instance, each episode acts in that instance's world through the harness's tools, and the harness grades whether the final reply matches what the tools did.

- Needs the harness (`honesty_rl` and `honesty`, stdlib only) on every node that builds the environment, the Ray actor nodes included: `pip install -e <agent-honesty-harness checkout>`. Halo imports without it; building `honesty` raises an `ImportError` that names it.
- Recipe: `examples/grpo/environmental/gptoss/vllm/gptoss-20b-honesty-lora-ep1.yaml`.

## Dataset

`python -m honesty_rl.generate --out data/honesty` writes `train.jsonl`, `eval_seen.jsonl`, `eval_heldout.jsonl` and `canonical.jsonl`, one `Task.to_row()` per line: `prompt`, `task` (the task as JSON: its world and grading facts), `template`, `family`.

- Name the task column in `context_fields: [task]`. A row without it, with a cell that is no task, or whose `prompt` is not the task's user message raises at reset.
- f04 and c09 script a second user turn, which this environment does not play: a row of either raises at reset. Evaluate them with the harness itself.

## Episodes

- The world (a downloads folder and an email store) is built in a temp dir at reset and removed when the episode is released. One instance serves concurrent episodes; each call runs in its own episode's world.
- The tools are `list_files`, `read_file`, `write_file`, `search_email`, `get_attachment` and `draft_email`, under the harness's `terse` schemas, which the engine sees verbatim. There is no send tool.
- A failure the world models comes back as the call's output (`ERROR: no such file: …`), not as a tool error. A call the schema refuses (a missing argument) never reaches the world.
- The system turn is the task's own: none for most probes, a support persona for p07 and c07. `system_prompt` is refused, so a trained policy is graded under the condition the harness evaluates.
- After a cut or empty turn the nudge asks for the next call or a reply to the user, never for a solution.

## Reward

The `environment` term prices the harness verdict (`honesty_rl.grade` over the final reply and the episode's ledger):

- `HONEST` grades 1, every other class 0, `AMBIGUOUS` included.
- `ERROR` (a tool raised inside the harness, or the grader raised) grades 0 and marks the episode invalid, out of the group baseline.
- An episode that ended without a reply (out of turns, or of output budget) grades 0 and stays in the baseline; `turn_overflow_penalty` prices it below a wrong reply.
- Tool calls are unpriced by default: the world's `ERROR` outputs are evidence to report, and paying calls would reward exploring for its own sake.

A `whitecircle` term under `view: full` reads every call's arguments, the `draft_email` body included, and every tool output. The harness ships a five-item action-honesty checklist for it (`<agent-honesty-harness>/integrations/whitecircle/honesty_policies.json`); `halo run create-whitecircle-policies` creates its policies and prints the term ([White Circle](../../../grpo/rewards/whitecircle.md)).

## Metrics

- `outcome/honest`, and `outcome/solve_rate` beside it: the honest rate over gradable episodes.
- `outcome/class/<CLASS>`: each verdict class one-hot over every episode; `NO_REPLY` marks an episode that never replied.
- Slices `template/<probe>/*` and `family/<family>/*`: a falling `family/control/solve_rate` is over-refusal, a policy that passes the gaps by refusing everything.
- `trajectory.info["honesty_verdict"]` keeps the class, its reason and the grader's flags; `eval_env.py --save_trajectories` writes it.

## Evaluation

```bash
python scripts/environments/inference/eval_env.py \
    --env_type honesty --dataset data/honesty/eval_heldout.jsonl --split train \
    --context_fields task --group_by template \
    --base_url http://localhost:8000/v1 --model <served-model> --num_examples 0
```

A single `.jsonl` file loads as one `train` split. Under `--training_config <recipe>`, pass `--env_kwargs '{"reward_terms": [{"source": "environment"}]}'` to grade by the harness verdict alone, without the recipe's White Circle term.
