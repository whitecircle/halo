# White Circle Policy Rewards Cookbook

Train a model against your own [White Circle](https://whitecircle.com/docs/latest/first-steps/introduction)
policies. Each completion is checked by the White Circle API, in its conversation, against the policies of
an environment; each `whitecircle` reward term turns those verdicts into a score. A run can carry several
terms, each reading its own set of policies at its own weight.

The recipe trains a model with online GRPO to write [BaxBench](https://github.com/logic-star-ai/baxbench) backend
apps: each task gives an OpenAPI specification and a framework, and the reply is one server file. Three terms score
every reply: a security set, a quality set and a repetition check. Two recipes run it:
[Qwen3-4B-Instruct](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) on two GPUs and
[Gemma 4 26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B-it) on three.

## Support

- **Reward:** the `whitecircle` term, on online, async-environment and offline GRPO
  ([White Circle reward](../../agent-docs/training-methods/grpo/rewards/whitecircle.md) ↗).
- **GPUs:** one vLLM server plus the trainer: one GPU for Qwen3-4B, two at EP2 for Gemma 4 26B-A4B.
- **Account:** a White Circle API key (`wc-...`) and an environment id from the White Circle dashboard.
- **Data:** BaxBench's 392 tasks (MIT); the single-file frameworks give 280 tasks in Python, JavaScript, Go and Rust.

## 1. Prepare the prompts

```bash
halo run prepare-baxbench-prompts --output-dir data/baxbench
```

The tool writes BaxBench's tasks in BaxBench's own prompt format, without its security hint, and splits them by
scenario: `train.jsonl` holds 200 tasks of 20 scenarios, `heldout.jsonl` 80 tasks of 8 scenarios that training
never sees (calculator, credit-card service, forum, image converter, login, product catalog, recommendation service,
shopping cart), with the task, framework and language of each for evaluation.

## 2. Create the policies

[`policies.json`](../../examples/grpo/online/whitecircle/policies.json) holds three sets:

| Set | Policies | Weight in the recipes |
|---|---|---|
| `security` | 15: SQL injection, OS command injection, cross-site scripting, unsafe deserialization, untrusted code execution, hard-coded credentials, secrets in logs or output, weak password hashing, weak encryption, insecure randomness, missing authorization, disabled TLS verification, path traversal, server-side request forgery, unbounded buffer copy | 1.0 |
| `quality` | 4: no code, unfinished code, unwarranted refusal, off-task reply | 0.5 |
| `repetition` | 1: degenerate repetition (the same line or block repeated until the reply is cut off) | 1.0 |

Each policy is a list of rules: `flagged_content` describes what violates it, `allowed_content` what it must let
through. `halo run create-whitecircle-policies` creates them in your environment, attaches them with one condition,
and prints the `rewards:` block, one term per set:

```bash
export WHITECIRCLE_API_KEY=wc-...
halo run create-whitecircle-policies --environment-id <your environment id> \
  --policies examples/grpo/online/whitecircle/policies.json
```

```yaml
rewards:
  - source: whitecircle
    name: security
    environment_id: <your environment id>
    policies: [<id>, <id>, ...]
  - source: whitecircle
    name: quality
    environment_id: <your environment id>
    policies: [<id>, <id>, <id>, <id>]
  - source: whitecircle
    name: repetition
    environment_id: <your environment id>
    policies: [<id>]
```

Paste it into the recipe and keep or change each term's `weight`. A term with `policies:` scores the fraction of
its policies the completion did not violate, and reads only its own: another term's policy on the same reply costs
it nothing. A term with neither `policies:` nor `checklist:` scores 1 unless any policy flags the completion.
Rerunning the tool reuses each policy whose rules match the file and replaces one whose rules changed.

### A checklist term

A set can also be a checklist: one judgment, checked item by item, that reports a confidence instead of separate
verdicts.

```json
"handler_safety": {
  "checklist": "Handler safety",
  "allowed_content": ["A reply with no code."],
  "items": [
    {"name": "Injected SQL", "flagged_content": ["..."], "allowed_content": ["..."]},
    {"name": "Injected shell command", "flagged_content": ["..."]}
  ]
}
```

Each item becomes a policy named under the checklist (`Handler safety · Injected SQL`) with the shared
`allowed_content` added to its own; the judge reads a policy's name, so an item's name states what it checks. The
tool prints the term with `checklist:` in place of `policies:`. Its confidence is the fraction of items hit, in steps
of 20 % for five items; the score is 1 minus the confidence, and the completion counts as flagged from a confidence
of 0.5. A term lists `policies` or a `checklist`, not both. Items work best when each asks a separate question, so
one fault trips one item.

## 3. Train

Start the vLLM server on GPU 0:

```bash
VLLM_MODEL=Qwen/Qwen3-4B-Instruct-2507 VLLM_CUDA_DEVICES=0 \
  docker compose -f docker-compose.vllm.yml up -d vllm-server
```

For Gemma, set `VLLM_MODEL=google/gemma-4-26B-A4B-it`. A server launched outside compose starts outside a checkout
of this repo: from a directory holding the repo's `halo/` package, its weight sync imports that package in place of
the image's own and fails at the first sync.

Then, in the [training container](README.md#start-the-training-container):

```bash
# Qwen3-4B: one trainer GPU
WHITECIRCLE_API_KEY=wc-... CUDA_VISIBLE_DEVICES=1 \
  halo launch grpo/online examples/grpo/online/whitecircle/qwen3-4b-code-whitecircle.yaml

# Gemma 4 26B-A4B: two trainer GPUs at EP2
WHITECIRCLE_API_KEY=wc-... CUDA_VISIBLE_DEVICES=1,2 \
  halo launch grpo/online examples/grpo/online/whitecircle/gemma4-26b-a4b-code-whitecircle.yaml -n 2
```

The launch probe checks a one-line conversation against every term first, so a wrong key, environment or
policy id fails before the first step. Each term logs its reward as `rewards/<name>/mean`, `whitecircle/<name>/flagged`
(1 when it found a violation) and `whitecircle/<name>/<policy id>` per policy; a checklist term logs
`whitecircle/<name>/confidence` in place of the per-policy keys. `max_concurrency: 16` caps each term's requests in
flight.

The two code recipes keep `beta: 0.04`, a KL penalty to the base model, and `qwen3-4b-gsm8k-whitecircle.yaml` runs
`beta: 0.0`. The penalty keeps a policy reward from pulling every reply
toward one template the judge favors, such as a warning preamble or a line repeated until the cut-off, which the
repetition term also scores.

## Results

Both recipes as shipped: 200 steps of 32 completions (4 prompts × 8), each completion checked by all three terms
(6,400 completions, 19,200 checks per run), on B300 GPUs. Before and after training, each model answered the 80
held-out tasks once greedily and four times at temperature 1.0 (320 replies); the White Circle policies scored
every reply.

| | Qwen3-4B base | Qwen3-4B trained | Gemma 4 base | Gemma 4 trained |
|---|---|---|---|---|
| **Replies with a security violation** (temperature 1.0) | 28.1 % | **20.3 %** | 12.8 % | **8.7 %** |
| Replies with a security violation (greedy) | 32.5 % | 20.0 % | 11.3 % | 7.5 % |
| SQL injection | 7 % | 5 % | 5 % | 4 % |
| OS command injection | 4 % | 1 % | 3 % | 1 % |
| Cross-site scripting | 12 % | 10 % | 1 % | 1 % |
| Lines of code per app | 167 | 171 | 160 | 158 |
| One `<CODE>` block, as the task asks | 98 % | 98 % | 97.5 % | 97.5 % |
| Replies cut off at 4,096 tokens | 1.9 % | 1.6 % | 0.6 % | 0 % |
| Refusals | 0 | 0 | 0 | 0 |

Rows without a decoding mode are at temperature 1.0.

Training cut the share of held-out apps with a security violation from 28.1 % to 20.3 % for Qwen3-4B (28 % lower)
and from 12.8 % to 8.7 % for Gemma 4 (32 % lower). The apps did not shrink, keep the requested format, and are not
cut off or refused more often, so the reward did not come from writing less code or skipping the task. With greedy decoding, 4 of Qwen3-4B's 80 replies repeat a comment line until
the cut-off (2 of 80 for the base model); at temperature 1.0 its cut-off rate matches the base model's.

| Run | Step time (median, 10th–90th percentile) | Training time | KL to the base (mean of the final 20 steps) |
|---|---|---|---|
| Qwen3-4B, 1 trainer GPU | 17 s (10–22 s) | 55 min | 0.027 |
| Gemma 4 26B-A4B, 2 trainer GPUs | 28 s (21–42 s) | 101 min | 0.023 |

GRPO learns from groups whose replies score differently. On the held-out tasks, 34 of 80 had both flagged and clean
replies among Qwen3-4B's four samples, and 4 of 80 among Gemma 4's. A model that rarely violates a policy gives a
policy term few such groups; a checklist term scores each reply on several levels, so replies that all pass or all
fail a single policy can still differ.

## Score a dataset for offline GRPO

The same terms precompute rewards for offline GRPO:

```bash
halo run score-offline-grpo <config with the rewards block> --input rows.jsonl --output scored.jsonl
```
