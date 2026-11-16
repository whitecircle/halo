# White Circle Policy Rewards Cookbook

Train a model against your own [White Circle](https://whitecircle.com/docs/latest/first-steps/introduction)
policies. Each completion is checked by the White Circle API, in its conversation, against the policies of
an environment; each `whitecircle` reward term turns those verdicts into a score. A run can carry several
terms, each reading its own set of policies at its own weight.

The recipe trains [Gemma 4 26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B-it) with online GRPO on
[`CyberNative/Code_Vulnerability_Security_DPO`](https://huggingface.co/datasets/CyberNative/Code_Vulnerability_Security_DPO),
requests for code that often invite an insecure implementation, under two terms: a security set and a
quality set.

## Support

- **Reward:** the `whitecircle` term, on online, async-environment and offline GRPO
  ([White Circle reward](../../agent-docs/training-methods/grpo/rewards/whitecircle.md) ↗).
- **GPUs:** three: one vLLM server, two trainer ranks at EP2.
- **Account:** a White Circle API key (`wc-...`) and an environment id from the White Circle dashboard.

## 1. Create the policies

[`policies.json`](../../examples/grpo/online/whitecircle/policies.json) holds two sets, `security` and
`quality`; edit or extend them. `halo run create-whitecircle-policies` creates them in your environment, attaches them with one
condition, and prints the `rewards:` block, one term per set:

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
    policies: [<id>, <id>, <id>, <id>]
  - source: whitecircle
    name: quality
    environment_id: <your environment id>
    policies: [<id>, <id>]
```

Paste it into [the recipe](../../examples/grpo/online/whitecircle/gemma4-26b-a4b-code-whitecircle.yaml)
and set each term's `weight` (the recipe weighs security 1.0, quality 0.5). Rerunning it reuses the policies it
finds by name. A term with `policies:` scores the fraction of its policies the completion did
not violate; a term without it scores 1 unless any policy flags the completion.

## 2. Train

Start the vLLM server on GPU 0:

```bash
VLLM_MODEL=google/gemma-4-26B-A4B-it VLLM_CUDA_DEVICES=0 \
  docker compose -f docker-compose.vllm.yml up -d vllm-server
```

A server launched outside compose starts outside a checkout of this repo: from a directory holding the
repo's `halo/` package, its weight sync imports that package in place of the image's own and fails at the
first sync.

Then, in the [training container](README.md#start-the-training-container):

```bash
WHITECIRCLE_API_KEY=wc-... CUDA_VISIBLE_DEVICES=1,2 \
  halo launch grpo/online examples/grpo/online/whitecircle/gemma4-26b-a4b-code-whitecircle.yaml -n 2
```

The launch probe checks a one-line conversation against every term first, so a wrong key, environment or
policy id fails before the first step. Each term logs `whitecircle/<name>/<policy id>` (1 when the policy
flagged the completion) and its reward as `rewards/<name>/mean`.

## Results

The recipe's settings on 384 of the dataset's Python requests, with a 9-policy security set (weight 1.0) and
a 4-policy quality set (weight 0.5) in one environment: 30 steps of 32 completions (4 prompts × 8), one B300
serving and two training at EP2. Every completion is checked by both terms.

| Steps | `security` | `quality` | Mean length | Step time |
|---|---|---|---|---|
| 1–10 | 0.943 | 0.913 | 582 | 12.7 s |
| 11–20 | 0.959 | 0.932 | 444 | 10.7 s |
| 21–30 | 0.963 | 0.890 | 398 | 10.5 s |

At `learning_rate: 1e-6` thirty steps move the security reward by 0.02 while completions shorten; the run
shows the signal reaching the trainer, not a converged policy. A step takes 10–13 s with both terms checking all 32
completions.

## Score a dataset for offline GRPO

The same terms precompute rewards for offline GRPO:

```bash
halo run score-offline-grpo <config with the rewards block> --input rows.jsonl --output scored.jsonl
```
