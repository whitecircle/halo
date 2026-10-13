# Served Reward Model

A `reward_model` term scores with a `*ForSequenceClassification` or `*ForRewardModel` checkpoint served by the rollout engines' pooling routes, on both arms.

The scorer renders the conversation with the `tokenizer`'s chat template, adds no special tokens, and maps the head's logit to `sigmoid((logit − logit_shift) / logit_scale)`. On the online arm it renders the original conversation, not TRL's generation prompt.

## Serving

| Backend | Serve | Route |
|---|---|---|
| `vllm` | `vllm serve <model> --runner pooling` (`--convert classify` for a decoder checkpoint vLLM does not map to a classification head) | `POST /classify` with the rendered text, `add_special_tokens: false`, `use_activation: false`; reads `probs[label_index]` |
| `sglang` | `python -m sglang.launch_server --model-path <model> --is-embedding` | `POST /classify` with the token ids; reads `embedding[label_index]` |

TRL's in-process path stays reachable through the online trainer's API (not the YAML): a model id in `reward_funcs` loads a classification head on every rank, in trainer memory.

## Options

| Option | Default | Effect |
|---|---|---|
| `url` | — | The server root, without `/v1` |
| `model` | — | The served model; also the default `tokenizer` |
| `backend` | `vllm` | `vllm` or `sglang` ([Serving](#serving)) |
| `tokenizer` | `model` | Whose chat template renders the conversation |
| `label_index` | `0` | The head output read as the logit |
| `logit_shift`, `logit_scale` | `0.0`, `1.0` | The sigmoid mapping above |
| `view` | `final` | `final`: the prompt plus one assistant message holding the final answer; `full`: every turn; `digest` is refused |
| `batch_size` | `8` | Rows per request |
| `max_concurrency` | `4` | Requests in flight per scorer |
| `request_timeout` | `60` s | Per request |
| `on_error` | `invalid` | [Outages](README.md#outages) |

The raw logit logs as `reward_model/<name>/logit`, the verdict rate as `reward_model/<name>/scored`.
