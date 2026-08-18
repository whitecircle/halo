# Model Merging

Combine several fine-tuned checkpoints of the **same architecture** into one model in weight space — no extra training, no inference-time ensembling. `scripts/after_training/merge_models.py` is the self-contained merger (no mergekit dependency) and supports four methods:

| Method | What it does | Models | Needs `--base_model` |
|---|---|---|---|
| `linear` | Weighted average of the weights (always divided by the weight sum, so per-model `:weight` values are relative) | ≥2 | No |
| `slerp` | Spherical interpolation between two checkpoints | exactly 2 | No |
| `task_arithmetic` | `base + Σ wᵢ·(modelᵢ − base)` — sum task vectors onto a base | ≥2 | Yes |
| `ties` | Trim each task vector to its top-`density` magnitudes, elect a per-parameter sign, average sign-agreeing deltas, add `λ`× onto the base ([arXiv:2306.01708](https://arxiv.org/abs/2306.01708)) | ≥2 | Yes |

All checkpoints must share the architecture: every model has to provide each parameter key at the same shape, or the merge raises — as does a model carrying a key the others do not.

Floating-point tensors are merged in float32 and cast to `--dtype` (default `bfloat16`), except router-balancing tensors, which keep their source dtype (a bf16 cast quantizes away the fp32 sign steps the balancing update writes). Non-float tensors (routing tables, expert-count buffers) are copied through unchanged, and a disagreement between models raises rather than inventing an average.

## Usage

```bash
# Weighted linear average (weights are normalized by their sum)
python scripts/after_training/merge_models.py --method linear \
    --models /ckpt/run-a:0.3 /ckpt/run-b:0.7 --output_dir /mnt/merged-linear

# SLERP halfway between two checkpoints
python scripts/after_training/merge_models.py --method slerp --t 0.5 \
    --models /ckpt/run-a /ckpt/run-b --output_dir /mnt/merged-slerp

# TIES onto a base
python scripts/after_training/merge_models.py --method ties --density 0.6 --lambda 1.0 \
    --base_model /ckpt/base --models /ckpt/run-a:0.5 /ckpt/run-b:0.5 --output_dir /mnt/merged-ties
```

Each `--models` entry is a **local directory**, optionally suffixed `:weight` (split on the last colon, so absolute paths are safe); the default weight is `1.0`. Weights are read off disk, so a Hub id there raises `FileNotFoundError` — `--tokenizer_source` is the one flag that resolves one.

| Flag | Default | Applies to | Meaning |
|---|---|---|---|
| `--models` | — | all | Local model dirs, each optionally `path:weight` (the suffix is rejected under `slerp`, which reads no weights) |
| `--output_dir` | — | all | Where to write the merged HF checkpoint. Must be a **new** directory — it is refused if it matches any source it reads from (`--models`, `--base_model`, `--tokenizer_source`), since the write deletes the `model*.safetensors` it does not own |
| `--method` | `linear` | all | `linear` / `slerp` / `task_arithmetic` / `ties` |
| `--base_model` | — | task_arithmetic, ties | Model the task vectors are relative to (required) |
| `--dtype` | `bfloat16` | all | Output dtype (`float16` / `bfloat16` / `float32`; short aliases `fp16`/`bf16`/`fp32` accepted) |
| `--density` | `0.6` | ties | Fraction of each task vector's deltas kept |
| `--lambda` | `1.0` | ties | Scale of the merged delta onto the base |
| `--t` | `0.5` | slerp | Interpolation factor in `[0, 1]` |
| `--tokenizer_source` | base, else first model | all | Where to copy config + tokenizer/processor from |
| `--max_shard_size` | `5GB` | all | Per-file cap for the output safetensors shards |
| `--allow_missing_tokenizer` | off | all | Accept a merged checkpoint whose `--tokenizer_source` ships no tokenizer files (refused by default) |
| `--trust_remote_code` | on | all | Trust remote code when re-reading the merged config (`--no-trust_remote_code` to disable); needed for Bailing/Ling |
| `--quiet` | off | all | Suppress per-key progress logging |

The *Applies to* column is enforced, not advisory: a knob passed explicitly for a method that does not consume it is **rejected** before any I/O, so a refused merge leaves no output directory behind. Defaults resolve after that gate, so naming a knob at its documented default still counts as passing it.

## Memory and output

Merging runs on CPU and streams the inputs **one tensor at a time** — each key is read from every model, merged, then released — so peak host memory is one fp32 copy of the largest tensor per contributing model plus the writer's pending output shard, never the merged model. Peak disk is the merged artifact, one input's size.

The output is a standard HF checkpoint: sharded safetensors plus an index above `--max_shard_size` (5 GB by default), or a single `model.safetensors` with no index when it fits. Every non-weight file in `--tokenizer_source` is copied verbatim — config, tokenizer, the multimodal processor files, and any remote-code modules the config's `auto_map` names — and the saved config's `dtype` is set to the merged dtype. A Hub id resolves to its local snapshot first and is copied the same way.

The resume sidecars are the exception: `scheduler.pt`, `router_balancing_biases.pt` and `rng_state_*` describe one training run, which an N-way merge has none of, so they are dropped. The merged model therefore ships **no** balancing sidecar — a family whose balancing bias lives in a checkpoint slot keeps it through the merged weights, while a `bias_update_transient` run's bias is gone (see [Checkpoints](checkpoints.md#resume-by-parallelism-mode)).

The result loads with `from_pretrained` and serves on vLLM with no conversion. A `--tokenizer_source` that ships no tokenizer files **raises** once the weights are written, since no `from_pretrained`-based consumer could build a tokenizer from the output — re-point it at a directory or Hub id that carries one, or pass `--allow_missing_tokenizer` to accept the tokenizer-less artifact.

`slerp` falls back to linear interpolation when the two tensors are near-(anti-)colinear (`|cosine| > 0.9995`), where spherical interpolation is numerically unstable and ≈ equal anyway.

## Other merges

Weight-space merging is distinct from the reassembly/adapter merges:

- **EP sharded checkpoints** → reassemble with `scripts/after_training/merge_ep_shards.py`. See [Checkpoints & Resume](checkpoints.md).
- **LoRA / PEFT adapters** → fold into the base with `scripts/after_training/merge_peft_adapters.py`. It refuses a directory carrying native EP expert-LoRA (`peft_type: EXPERT_LORA` or `LORA_WITH_EP_EXPERT_LORA`, or `*.experts.<attr>.lora_{A,B}` keys), which `merge_and_unload` would silently drop; those fold only at save time, with `merge_expert_lora_on_save: true`. See [PEFT (LoRA)](../optimization/peft.md).
