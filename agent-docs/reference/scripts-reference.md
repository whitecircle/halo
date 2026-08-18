# Scripts Reference

Catalog of toolkit scripts by category. For config fields see [Configuration](configuration-reference.md); for parallelism flags see [Expert Parallelism](../parallelism/expert-parallelism.md).

Every entry script here answers `python <script> --help` with its full flag list — the fastest way to see a script's real surface, including the CLI overrides for any YAML field. The `halo` CLI (`src/cli.py`) runs them by name: `halo launch <method> <config>` for `scripts/training/`, `halo run <tool>` for every other `scripts/` subtree except `diagrams/` (`--list` on either enumerates them). The launcher follows the flags, never the config: `accelerate launch` with `--accelerate <cfg>`, `torchrun` when `--nproc/-n > 1`, plain `python` otherwise — so `halo launch sft <ep-config>.yaml` without `-n` runs single-process.

Run scripts from the repo root with the repo root importable: the wheel installs `src` only (`packages = ["src"]`), and `python scripts/a/b.py` puts the *script's* directory on `sys.path`, not the repo root — so every CLI that imports a `scripts.*` helper needs `scripts` importable as well as `src`. Those helpers are the shared flag surfaces: `scripts/_common.py` (the shard cap, the Hub source block and `--trust_remote_code`, taken by the checkpoint tools across `after_training/`, `before_training/` and `inference/reward_model/`), `scripts/inference/_common.py` (the OpenAI endpoint, resume and Gradio blocks), `scripts/inference/reward_model/_common.py` (the reward-model scoring block, on top of the previous two) and `scripts/environments/_common.py` (the env-eval flags and output writer). Both training images set `PYTHONPATH=/workspace` and the Makefile passes it into every container it starts, so in-image runs need nothing extra; a host run needs `PYTHONPATH=.` from the repo root.

Flag spelling is per script and stable: the Gradio apps, `scripts/profiling/**`, `before_training/prepare_dataset.py` and `before_training/s3_datasets.py` spell their own multi-word flags with dashes (`--api-key`); every other script uses underscores (`--api_key`), matching the YAML field names a training flag overrides. The shared `--trust_remote_code` keeps its one spelling everywhere, `prepare_dataset.py` included.

The checkpoint tools under `after_training/` and `before_training/` take one source/destination pair: `--input_dir` → `--output_dir` for a local checkpoint directory, `--model_id` → `--output_dir` where the source may also be a Hub repo (`patch_vocab.py`, `convert_deepseek_v4_bf16.py`, `convert_mistral4_bf16.py`, `convert_glm5_bf16.py`; `reattach_vision_tower.py` takes both — `--input_dir` for the text-only export, `--model_id` for the multimodal base). Three tools keep a differently-shaped source because it is a different thing: `merge_models.py --models` (N inputs), `merge_peft_adapters.py --adapter_dir` (an adapter, not a checkpoint), and `prepare_dataset.py --input`/`--output` (dataset URIs — `s3://`, `hf://` or a local path).

`merge_peft_adapters.py` selects the head with `--task {causal_lm,classification}`.

Launcher: `torchrun` for all multi-GPU work — every parallel axis **and** plain FSDP2 data parallelism; `python` for single-GPU and LoRA. `accelerate launch` with the `launcher-configs/accelerate/*.yaml` configs stays supported for plain data parallelism only.

## Training scripts

Every training entry script runs the shared skeleton in `src/training/script_runner.py`: distributed init → parallelism config → resume resolution, DP-shard-aware dataset loading, trainer config, then `run_trainer` — which emits the canonical start log (mode plus EP/CP/TP/ETP/DP sizes) and reorders integration callbacks last so `moe/*` and efficiency keys reach W&B.

### Supervised fine-tuning

| Script | Description |
|--------|-------------|
| `scripts/training/sft.py` | SFT (text or VLM, auto-detected); every available axis — EP/CP/TP/ETP — plus `init_from_scratch` (dense data-parallel only) |

```bash
python scripts/training/sft.py examples/sft/qwen3/qwen3-4b-ultrachat.yaml
torchrun --nproc_per_node=8 scripts/training/sft.py \
    examples/sft/gptoss/gptoss-20b-multinode-ep.yaml --expert_parallel_size=8
```

### Preference optimization

| Script | Description |
|--------|-------------|
| `scripts/training/preference/dpo.py` | DPO (text or VLM, auto-detected); EP/TP/ETP; TRL `loss_type` variants (`sigmoid`, `hinge`, `ipo`, …) |
| `scripts/training/preference/kto.py` | KTO (text or VLM, auto-detected; unpaired binary feedback); EP/TP/ETP |
| `scripts/training/preference/smpo.py` | Smooth Margin PO (text or VLM, auto-detected; reference-model-free); EP/CP/TP/ETP |
| `scripts/training/preference/rewards.py` | Bradley-Terry reward model; EP/TP/ETP |

### Online GRPO

| Script | Description |
|--------|-------------|
| `scripts/training/online_grpo/rlvr.py` | RLVR: online GRPO with verifiable rewards (math accuracy, format checking); EP/TP/ETP, no CP or PP |

Run a vLLM server on a dedicated GPU, then launch training on the rest:

```bash
# vLLM runs in its own container (the training image has no vllm)
VLLM_CUDA_DEVICES=7 docker compose -f docker-compose.vllm.yml up vllm-server
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun --nproc_per_node=7 \
    scripts/training/online_grpo/rlvr.py \
    examples/grpo/online/rlvr-online-grpo-template.yaml
```

### Offline GRPO

| Script | Description |
|--------|-------------|
| `scripts/training/offline_grpo.py` | Group-relative PO with pre-computed rewards; EP/TP/ETP |

```bash
torchrun --nproc_per_node=8 scripts/training/offline_grpo.py \
    examples/grpo/offline/gptoss/offline-grpo-gptoss-20b-gsm8k.yaml --expert_parallel_size=8
```

### Environmental GRPO (multi-turn RL)

| Script | Description |
|--------|-------------|
| `scripts/training/environmental_grpo.py` | Multi-turn environmental GRPO — resolves the environment from `environment_type` in YAML; EP/TP/ETP via `torchrun`, no CP or PP (register custom environments with `register_environment`) |

Benchmark environments available via registry (no dedicated script):

| Registry Name | Environment | Compatible Datasets |
|---------------|-------------|-------------------|
| `qa_search` | `NativeToolUseEnvironment` (factory) | SimpleQA, GAIA, TriviaQA, PopQA |
| `code_contests` / `codeforces` | `CodeContestsEnvironment` | Codeforces, DeepCoder (RL pools); LiveCodeBench, ICPC-Eval, HLCE (benchmarks) |
| `exam_qa` | `ExamQAEnvironment` | MMLU-Pro, GPQA, MMLU, ARC |

```bash
# Start vLLM: docker compose -f docker-compose.vllm.yml up vllm-server
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node=7 \
    scripts/training/environmental_grpo.py \
    examples/grpo/environmental/environmental-grpo-template.yaml
```

### Other methods

| Script | Description |
|--------|-------------|
| `scripts/training/classification.py` | Multi-class/multi-label sequence classification; EP/TP/ETP |
| `scripts/training/embedding.py` | Embedding fine-tuning with SBERT losses; EP/TP/ETP (no CP, no PP) |
| `scripts/training/distillation/teacher_distill.py` | Off-policy teacher→student distillation (text or VLM, auto-detected), multiple loss types; EP/TP/ETP (no CP, no PP) |
| `scripts/training/distillation/self_distill.py` | Offline SDPG privileged-context self-distillation (text or VLM, auto-detected); faithful online SDPG runs via `online_grpo/rlvr.py --use_sdpg=true` |

## Inference & generation scripts

| Script | Description |
|--------|-------------|
| `scripts/inference/generation/openai_batched_generation.py` | Async batched generation with OpenAI-compatible API |
| `scripts/inference/generation/dataset_deduplication.py` | FAISS-based semantic deduplication. Keyed on `--text_field` alone: two rows with the same text collapse to one however they differ elsewhere, so an images column does **not** make them distinct — deduplicate a VLM dataset before pairing its images, or on a field that carries the difference |
| `scripts/inference/reward_model/rm_rejection_sampling.py` | vLLM + reward model preference dataset generation. A hypothesis the endpoint cut at `--max_gen_tokens` is dropped rather than scored as if it had finished, and a row left with fewer than two usable hypotheses is skipped; both counts are in the run summary |
| `scripts/inference/reward_model/rm_scoring.py` | Score datasets using reward models. A response cut at `--max_gen_tokens` is dropped and counted (`truncated=`) rather than scored as if it had finished |
| `scripts/inference/playground/gradio_openai_chatbot.py` | Chatbot UI (Gradio + OpenAI API). `--api-key` defaults to `$VLLM_API_KEY`, else `$OPENAI_API_KEY`, else vLLM's `EMPTY` placeholder |
| `scripts/inference/playground/gradio_environment_playground.py` | Environment playground (Gradio) for testing GRPO environments against a rollout server. Episodes run through the shared eval driver (`run_episode` in `src/environments/eval_runner.py`), so a turn the engine cut off at its token cap is recovered here exactly as in training. `--vllm-url` prefills the rollout server base URL; `--api-key` defaults to `$VLLM_API_KEY`, else `$OPENAI_API_KEY`, else vLLM's `EMPTY` placeholder, and stays server-side; `--host` binds loopback (`127.0.0.1`), so publishing the UI — and that key's spend — takes an explicit `--host 0.0.0.0` |
| `scripts/environments/inference/run_code_contests.py` | Evaluate on competitive programming: dataset adapter (`codeforces`, `deepcoder`, `livecodebench`, `icpc`, `hlce`) + language (python/cpp/c) + success@k bucketed by adapter group field (rating/difficulty/contest), against vLLM or OpenRouter. `--reasoning_effort` (low/medium/high, default medium) sets template effort and, unless `--max_tokens` is given, the generation budget: the effort's thinking budget (4096/8192/16384) plus 4096 tokens of solution headroom → 8192/12288/20480; `--max_turns` default 15; `--env_kwargs` for the env/grading knobs without a flag; `--save_trajectories <path>` / `--trajectory_dir <folder>` record JSONL |
| `scripts/environments/inference/run_env.py` | Generic eval runner for other envs (QA, exam, SWE, MCP) over an OpenAI endpoint; reads prompt/answer columns, reports reward / success@k; `--save_trajectories` / `--trajectory_dir` record JSONL |
| `scripts/environments/inference/regrade_trajectories.py` | Offline re-grader: replays saved JSONL (`<jsonl...> --workers --output`) through grading only, decoupled from generation. Needs the code-contest meta `run_code_contests.py` stamps (`adapter`/`language` on top of the generic eval meta); a `run_env.py` trajectory is rejected up front |

In `openai_batched_generation.py`, `--input_path` / `--output_path` are S3 **keys**, not URIs:
`build_s3_uri` joins them under `HALO_S3_DEFAULT_BUCKET` (default `my-bucket` — set it to your own
bucket) and `--subfolder` (default `datasets`, `None` to skip). The same flag names on
`dataset_deduplication.py` are ordinary local paths.

The three async CLIs — `openai_batched_generation.py`, `rm_rejection_sampling.py`, `rm_scoring.py`
— run under a shared SIGINT/SIGTERM handler
(`run_async_cli`) that exits `128 + signo`, the shell's own convention for a signalled process.
Progress is checkpointed, so re-running resumes; the non-zero exit is what stops a wrapper script or
`&&` chain from consuming a partial output dataset as a finished one. A run that produced no usable
row raises rather than writing an empty result (`reject_empty_results`), so a dead endpoint or a
wrong `--model_name` cannot republish the resumed rows as a finished job. Each CLI's summary line
names its per-reason drop counts (first-response failures, degenerate skips).

Every endpoint these CLIs talk to is OpenAI-compatible (`--openai_base_url` / `--openai_api_key`),
reached through the one shared client (`create_openai_client`) with the toolkit's retry policy. There
is no separate Azure mode — point `--openai_base_url` at the deployment's OpenAI-compatible route like
any other endpoint.

```bash
HALO_S3_DEFAULT_BUCKET=your-bucket \
python scripts/inference/generation/openai_batched_generation.py \
    --model_name my-model --input_path prompts --output_path responses

python scripts/inference/reward_model/rm_scoring.py \
    --model_name my-model --prompts_source data/prompts.jsonl --rm_model_path path/to/reward-model
```

## Post-training scripts

| Script | Description |
|--------|-------------|
| `scripts/after_training/merge_peft_adapters.py` | Merge LoRA/PEFT adapters into the base model; refuses a native EP expert-LoRA directory |
| `scripts/after_training/merge_models.py` | Merge same-architecture checkpoints in weight space (linear / slerp / task_arithmetic / ties). Streams one tensor at a time across the inputs (each key loaded from every model, merged, written), so peak host memory is a single layer plus one pending output shard — never the merged model |
| `scripts/after_training/merge_ep_shards.py` | Merge EP sharded checkpoints into a single model (`--max_shard_size` caps the output shards, default `5GB`). `--delete_input_shards` frees the per-rank inputs once the merged checkpoint is complete — until then peak disk holds both. Refuses an input holding a PEFT adapter beside the shards: the aux copy carries `adapter_config.json` but no weight file, so the merged directory would claim an adapter it does not hold |
| `scripts/after_training/convert_to_bf16.py` | Convert model weights to BF16, norm leaves kept fp32. Re-applies the source's training sidecars to a full model before saving (neutralized GptOss sinks, balancing tensors re-read from the source shards at their trained fp32) and carries `router_balancing_biases.pt` / `training_provenance.json` into the output either way, so an unmerged adapter conversion still hands them to the later merge; `--merge_adapter` without `--peft` is refused. `--model_type` (`causal_lm` default, `classifier`, `base`) picks the class the checkpoint loads with and gates the PEFT refusal. `--verify` asserts the STORED dtypes read from the saved safetensors headers — never a `from_pretrained` reload, which casts on the way in and so can never fail — weighted by parameter count, not tensor count. An unmerged PEFT save is exempt: PEFT restores LoRA A/B to fp32 as it writes them. `--check_inference` is the separate diagnostic: it reloads the saved checkpoint and prints what it generates, returning no verdict — `--verify` is the gate that raises |
| `scripts/after_training/quantize_to_lowp.py` | Quantize a bf16/fp32 checkpoint to block-scaled mxfp8/mxfp4/nvfp4 (pairs with QAT — see [Mixed-Precision Training](../optimization/low-precision-moe-kernels.md)). `--format {mxfp8,mxfp4,nvfp4}` is required — the checkpoint names no target. The training run's lowp **scope** is not recorded in the checkpoint either, so four more flags restate it under the `ParallelismConfig` names and defaults — a config's values transfer verbatim: `--lowp_apply_dense_mlp` / `--lowp_apply_moe_experts` (both on; `--no-` prefix to disable) and `--lowp_keep_first_blocks` / `--lowp_keep_last_blocks` (both `0`) |
| `scripts/after_training/reset_sinks.py` | Reset attention sink tokens. `--dry_run` prints every sink tensor without writing. `--output_dir` is required for a write unless `--in_place` is given (the two are mutually exclusive); a `--dry_run` only reads and needs neither. Both branches stage the write — a sibling temp file for a single-file checkpoint, a sibling staging directory for a sharded one — verify every sink tensor sits at its dtype min, and only then replace the target, so a write that kept live sinks raises with the target untouched. It writes no `training_provenance.json` — both branches carry the source's record over verbatim, so a checkpoint whose sinks it just neutralized can still claim `live` to the merge tools that trust that record. Only `PeftAdapterSaver` writes the file |
| `scripts/after_training/unfuse_moe_experts.py` | Unfuse contiguous fused-GLU MoE experts (`experts.gate_up_proj [E,2I,H]`/`down_proj`), `--input_dir` → `--output_dir` (`--max_shard_size` caps the output shards, default `5GB`), into the per-expert layout the checkpoint's own family stores (`EPMoELayerBase.hub_per_expert_keys`): GLM-4 MoE Lite / Laguna / Qwen3 MoE / Qwen3.5-3.6 / Bailing-Ling / Cohere2 MoE / GLM-5 Next `experts.{i}.{gate,up,down}_proj.weight` — what vLLM's `glm4_moe_lite` loader expects — and LFM-2 / DeepSeek-V4 `experts.{i}.w{1,3,2}.weight`. A `model_type` whose checkpoint does not store one tensor per expert is refused: Gemma4 and Mistral4 register no per-expert conversion with transformers, GptOss's fused tensor is interleaved, Inkling's hub layout is the interleaved TM `w13_weight` one, Zaya's hub layout is fused already, and Step-3.7 Flash's is per-layer stacked (`moe.{gate,up}_proj [E, M, H]`, `moe.down_proj [E, H, M]`) |
| `scripts/after_training/reattach_vision_tower.py` | Rebuild the multimodal wrapper layout from a `text_only_model` export: the trained text weights re-prefix to `model.language_model.*`, the base's untrained vision tower and MTP tail stream back in (`--input_dir` = the export, `--model_id` = the multimodal base, hub or local), and the composite config is regrafted with the trained text config. Needed to serve such an export on the pinned vLLM/SGLang images — their registries carry only the wrapper class. A wrapper-layout input is refused |

```bash
python scripts/after_training/merge_peft_adapters.py \
    --adapter_dir /models/sft-llama-lora --output_dir /models/sft-llama-merged

python scripts/after_training/merge_ep_shards.py \
    --input_dir /models/ep-checkpoint --output_dir /models/merged-model
```

### Input guards

Every tool here refuses an input it cannot express, rather than writing a plausible-looking result:

- **Per-rank EP-sharded input** (`metadata.format` = `ep_sharded`) is rejected by all of them except
  `merge_ep_shards.py`, which exists to consume it. Such a save reuses the ordinary index filename
  while each expert tensor is one rank's partial slice under a `.shard_N` key, so a
  `from_pretrained`-based tool would see the real expert keys as **missing** — which transformers
  resolves by randomly initializing them, warning only — and then save that over the source. Merge
  first, or re-save gathered. The refusal does not need the index: a directory whose shards carry
  `.shard_N` keys with no index (a save killed before the index write) is caught by a header-only
  peek. The `scripts/before_training/` bf16 converters refuse it on the same terms, since all of
  them also accept a local directory. The pipeline-parallel save layout is not one of these: it uses
  global parameter names under a standard HF index with no `format` marker, so every tool would take
  it directly.
- **In-place conversion** — `--output_dir` equal to any input directory — is rejected by every tool
  that streams from the source while writing, because the write deletes the weight files it does not
  overwrite and so destroys the source mid-read. That is `merge_ep_shards.py`,
  `unfuse_moe_experts.py`, `quantize_to_lowp.py`, `merge_models.py`, `merge_peft_adapters.py`,
  `convert_to_bf16.py`, `reattach_vision_tower.py` (on both of its sources), `convert_deepseek_v4_bf16.py`,
  `convert_mistral4_bf16.py`, `convert_glm5_bf16.py`, and `patch_vocab.py`, whose pre-save
  sweep would otherwise leave nothing behind on a failed save. A refused conversion creates no output
  directory. `prepare_dataset.py` is not one of them: it materializes into a temp directory and
  publishes with a staged swap, so its `--output` is guarded by `--overwrite`, not by this refusal.

    `reset_sinks.py` is the exception: it holds the whole checkpoint in memory and stages the write,
    so in-place is safe there — but never implicit. It takes `--in_place`, an `--output_dir` aimed at
    the input is refused like the rest, and omitting both is an error on a write (a `--dry_run`
    only reads) rather than a silent rewrite of the only copy. A fresh-dir run sweeps leftovers
    like the others.
- **Remote code.** Every checkpoint tool that loads a checkpoint's own modeling/config/tokenizer code
  takes one `--trust_remote_code`, spelled once (`add_trust_remote_code_arg` in
  `scripts/_common.py`): `merge_peft_adapters.py`, `merge_models.py`,
  `convert_to_bf16.py`, `reset_sinks.py`, `reattach_vision_tower.py`, `patch_vocab.py`,
  `prepare_dataset.py`, `convert_deepseek_v4_bf16.py`, and the reward-model scoring CLIs
  (`rm_scoring.py`, `rm_rejection_sampling.py`) through `scripts/inference/reward_model/_common.py`.

    **The default follows the input source.** A local checkpoint or adapter (`--input_dir`,
    `--adapter_dir`, `--models`, `--rm_model_path`, or the tokenizer of the run being prepared) defaults **on**: the
    remote-code families in the roster (Bailing/Ling, Laguna) do not load without it, and the operator
    already produced that artifact. A Hub-capable `--model_id` source (`patch_vocab.py`,
    `convert_deepseek_v4_bf16.py`, `reattach_vision_tower.py`) defaults **off**, because a freshly downloaded third-party repo
    must not execute its own code merely because a tool was pointed at it. Either way the opposite is
    one flag away (`--trust_remote_code` / `--no-trust_remote_code`).

    The tools that never enable remote code expose no such flag — `merge_ep_shards.py`,
    `unfuse_moe_experts.py`, `quantize_to_lowp.py`, `convert_mistral4_bf16.py` and
    `convert_glm5_bf16.py` stream safetensors and rewrite `config.json` as JSON, so no checkpoint
    code is ever imported.
- **Uninitialized buffers.** A tool whose loaded model is handed to a forward — reward scoring, the
  dedup embeddings, `convert_to_bf16.py --check_inference` — calls `finalize_loaded_model` after
  placement, since transformers 5 re-materializes non-persistent buffers as uninitialized memory
  ([buffers](../models/adding-a-model.md)). The load-only conversion tools do not: `state_dict` omits
  those buffers, so an unrepaired one reaches neither a number nor the written checkpoint.
  `tests/cpu/models/test_load_finalization.py` pins each new `scripts/` loader into one of the two
  groups.
- **Shard cap and Hub source.** Two more flags are spelled once each, beside
  `add_trust_remote_code_arg`: `add_max_shard_size_arg` writes `--max_shard_size` (default
  `DEFAULT_MAX_SHARD_SIZE`, `5GB`) for every tool that re-shards its safetensors output —
  `quantize_to_lowp.py` mirrors the input's shard layout one-for-one and takes no cap — and `add_hub_source_args`
  writes `--model_id` plus the `--revision` pin for every tool whose source may be a Hub repo.
  `patch_vocab.py` takes `--model_id` without `--revision` — it threads none, and an advertised pin a
  tool ignores would silently convert whatever the Hub's default branch holds that day. Pin such a
  source by downloading it first and pointing `--model_id` at the directory.
- **Wrong model family.** `unfuse_moe_experts.py` resolves the family from the checkpoint's
  `model_type` and emits the projection names it declares
  (LFM-2 reads `w1`/`w3`/`w2`, not GLM-4's `gate_proj`/`up_proj`/`down_proj`); a family whose checkpoint
  is not per-expert at all is refused rather than written under a guessed triple. Both checks run after
  the sharded-input refusal and after the already-per-expert copy-through, so those keep their own
  diagnosis.
- **Asymmetric key sets.** `merge_models.py` refuses models whose tensor key sets differ — a key present
  in only one model would otherwise be dropped from the merge (typically one checkpoint saved untied,
  carrying `lm_head.weight`, and another tied).
- **Tokenizer-less source.** `merge_models.py` refuses a `--tokenizer_source` (default: the base model,
  else the first input) that ships no tokenizer files — every `from_pretrained`-based consumer of the
  merged checkpoint would fail to build a tokenizer. This one raises *after* the merged weights are
  written (the source is only read at the aux-file copy), so re-point `--tokenizer_source` at a directory
  or Hub id that carries one, or pass `--allow_missing_tokenizer` if a tokenizer-less artifact is intended.

## Preparation scripts

| Script | Description |
|--------|-------------|
| `scripts/before_training/prepare_dataset.py` | Pre-process SFT or raw-text datasets: tokenization, packing, sharding. `--mode chat` (default) applies the chat template; `--mode text` tokenizes `--text-field` directly and appends EOS per document — the (continued) pre-training path |
| `scripts/before_training/s3_datasets.py` | CLI over the S3 transport: `push`, `download`, `list`, `exists`, `delete` a folder under `--bucket` (default `HALO_S3_DEFAULT_BUCKET`). `delete` removes ONE object; `--recursive/-r` is opt-in for a whole prefix. See [S3 Utilities](../data/s3-utilities.md) |
| `scripts/before_training/patch_vocab.py` | Patch vocabulary with new tokens, reset attention sinks |
| `scripts/before_training/convert_deepseek_v4_bf16.py` | Dequantize a FineGrained-FP8 DeepSeek-V4 checkpoint to uniform bf16 for training; `--max_shard_size` sizes the output shards (default `5GB`; see [DeepSeek-V4](../models/deepseek-v4.md)) |
| `scripts/before_training/convert_mistral4_bf16.py` | Dequantize a public FP8 Mistral Small 4 checkpoint to bf16, streaming shard-by-shard (no full-model RAM footprint); `--max_shard_size` caps the output shards (default `5GB`; see [Mistral4](../models/mistral4.md)) |
| `scripts/before_training/convert_glm5_bf16.py` | Dequantize the fp8 block-quantized GLM-5.3-Flash release to bf16 (block-wise `weight * scale_inv`, unquantized tensors keep their stored dtype), streaming shard-by-shard; `--model_id` may be a Hub repo; `--max_shard_size` caps the output shards (default `5GB`; see [GLM-5 Next](../models/glm5-next.md)) |
| `scripts/environments/preparation/prepare_code_dataset.py` | Prepare a competitive-programming dataset (Codeforces, DeepCoder) for the coding RL env — composes prompts, packs tests/checker/time-limit. `--min_rating`/`--max_rating` also drop unrated problems (no-op without a `rating` column). `--num_proc` defaults to the toolkit's own dataset-processing default rather than a fixed count; a `--push_to_hub` repo is created private unless `--no-private` |

```bash
python scripts/before_training/prepare_dataset.py \
    --input "s3://bucket/raw/dataset" --output "s3://bucket/preprocessed/dataset" \
    --model-name "Qwen/Qwen3-8B" --max-length 8192 --num-shards 64 --pack-sequences \
    --assistant-message-template $'<|im_start|>assistant\n'

python scripts/before_training/patch_vocab.py \
    --model_id Zyphra/ZAYA1-8B --output_dir /mnt/models/ZAYA1-8B-patched \
    --patterns '["<tool_call>", "<tool_result>"]'
# GptOss sink reset: --reset_sinks
# remote-code family (Bailing/Ling, Laguna): add --trust_remote_code — a Hub source is opt-in here
```

`patch_vocab.py` only **grows** the embedding (added tokens reuse existing padding rows), never shrinks it: models like GPT-OSS ship a vocab padded past `len(tokenizer)`, and shrinking would drop the high special tokens (harmony `<|return|>`/EOS, …) and break generation. A vocab-patched checkpoint is a **training base** — serve the original full-vocab model, not the patched one.

## Profiling & benchmarks

| Script | Description |
|--------|-------------|
| `scripts/profiling/trace_report.py` | TraceLens analysis workbooks from torch.profiler traces — per-rank perf report + multi-rank NCCL collective/skew report (see [Debugging](debugging.md#1b-tracelens-automated-trace-analysis)). Trace sets are independent, so one unreportable capture does not cost the others' workbooks, but any failure exits non-zero; `--keep-going` accepts a partial run |
| `scripts/profiling/py_spy_diag.py` | Attach py-spy to a running job. Takes a subcommand: `dump` (instantaneous all-rank stack dump, hang triage) or `record` (per-rank CPU flame graphs over `--duration` seconds). Both default to this node's torchrun ranks; `--pid` (repeatable) targets others |
| `scripts/profiling/nvlink_health.py` | Torch-free NVLink preflight from `nvidia-smi`: per-link hard errors, FEC correction depth, active bandwidth. Exits non-zero only when a link is unhealthy, separating real faults from benign `Xid 145` churn (see [Debugging](debugging.md)) |

Throughput benchmarks live under `tests/gpu/profiling/`, grouped by what they measure:

| Group | Scripts |
|---|---|
| Trainer × parallelism throughput | `benchmark_sft_dense`, `benchmark_sft_ep`, `benchmark_sft_ep_cp`, `benchmark_sft_ep_tp`, `benchmark_smpo_ep`, `benchmark_smpo_ep_cp`, `benchmark_offline_grpo_ep` — each takes `--model` from the shared roster (`tests/common/models.py`), so a per-model number is a flag, not a script |
| Kernel / mechanism A/B | `benchmark_attention_implementations` (FA4/FA2 vs flex/sdpa/eager, fwd + bwd), `benchmark_grouped_mm` (loop vs `grouped_mm`), `benchmark_torch_compile` (Liger × compile 2×2 on EP), `benchmark_collators` (packing vs padding-free), `benchmark_roofline`, `bench_ep_buffer_backends` (DeepEP V1 Buffer vs V2 ElasticBuffer) |
| Stock-TRL reference | `benchmark_trl_baseline` (same `EfficiencyCallback` metrics), `benchmark_convergence` (loss curves, same seed + data order) |

Shell runners in the same directory: `run_all_benchmarks.sh`, `run_mfu_benchmarks.sh`,
`run_ep_tp_benchmarks.sh`.

```bash
./tests/gpu/profiling/run_ep_tp_benchmarks.sh --quick
torchrun --nproc_per_node=8 tests/gpu/profiling/benchmark_smpo_ep.py
```

The docs' own figures are generated by the standalone matplotlib scripts in `scripts/diagrams/`
(`gen_*.py`); they are linted with the rest of the tree (their per-file exemptions are the
shared-style star imports and matplotlib's `use()`-before-`pyplot` import order) and are not imported
by library code. Editing a generator means re-running
`make diagrams` and committing the PNGs — the `docs` workflow's `diagrams` job byte-compares them and
fails the PR otherwise ([CI](../infrastructure/ci.md#hosted-tier)).

## Related pages

- [Checkpoints & Resume](checkpoints.md)
- [SFT Dataset Pre-Processing](../data/dataset-preparation.md) — guide to `prepare_dataset.py`
- [Configuration Reference](configuration-reference.md) — YAML options for training scripts
