# Scripts Reference

Catalog of toolkit scripts by category. For config fields see [Configuration](configuration-reference.md); for parallelism flags see [ParallelismConfig](configuration-reference.md#parallelismconfig).

Every entry script answers `python <script> --help` with its full flag list, including the CLI overrides for any YAML field.

The `halo` CLI (`halo/cli.py`) runs them by name: `halo launch <method> <config>` for `scripts/training/`, `halo run <tool>` for every other `scripts/` subtree except `diagrams/` (`--list` on either enumerates them). `halo launch` follows its flags, never the config: `accelerate launch` with `--accelerate <cfg>`, `torchrun` when `--nproc/-n > 1`, plain `python` otherwise, so `halo launch sft <ep-config>.yaml` without `-n` runs single-process. `--numa-binding <mode>` (torchrun's modes) goes on the `torchrun` line ahead of the script and is refused for an `accelerate launch` or `python` launch, which have no NUMA binding. `halo run` always runs one `python` process (every tool is single-process) and owns only `--list`, `--dry_run` and `--root`, so `-n`, `-p` and a tool's own `--port` reach the tool. Both exec the script with `--root` first on `PYTHONPATH`, so a second checkout runs on its own `halo` and `scripts`.

Run scripts from the repo root with the repo root importable. The wheel installs `halo` only (`packages = ["halo"]`), and `python scripts/<subtree>/<tool>.py` puts the *script's* directory on `sys.path`, not the repo root. So every CLI that imports a `scripts.*` helper needs `scripts` importable as well as `halo`.

Both training images set `PYTHONPATH=/workspace` and the Makefile passes it into every container it starts. A host run needs `PYTHONPATH=.` from the repo root.

A tool script parses its flags and calls its driver in `halo/`, which holds the logic; the tests exercise the driver and reach a script only through its parser or `main()`. The checkpoint tools' drivers are in `halo/checkpoint/tools/`: `ep_shard_merge.py`, `pp_stage_merge.py`, `model_merge.py`, `adapter_merge.py`, `bf16_conversion.py`, `lowp_export.py`, `sink_reset.py`, `expert_unfuse.py`, `vision_reattach.py`, `vocab_patch.py`, `fp8_dequant.py` (the GLM-5 and Mistral 3/4 converters) and `deepseek_v4_dequant.py`. The dataset tools' drivers are in `halo/data/`, the generation, scoring and environment tools' in `halo/entrypoints/`, and the code-contests test compaction is `halo/environments/tasks/coding/suite_compaction.py`. Four kinds of script keep their logic. A training script assembles its run around the trainer, its driver. The two Gradio playgrounds are the app. `nvlink_health.py`, the `scripts/docs/` checks and the `scripts/diagrams/` generators import no `halo` module, so `nvlink_health.py` and the docs checks run on the bare host. `trace_report.py` hands its traces to TraceLens's report generators. Each entry point configures CLI logging in `main()`, so importing a script changes no logging state.

The `scripts.*` flag helpers are the shared flag surfaces:

| Helper | Flags |
|---|---|
| `scripts/cli_arguments.py` | The shard cap, the Hub source block, `--dtype`, `--device_map`, `--task` (`convert_to_bf16.py` and `merge_peft_adapters.py`) and `--trust_remote_code`, each taken by the `before_training/` and `after_training/` tools that need it (`quantize_to_lowp.py` and `s3.py` take none, `prepare_dataset.py` only `--trust_remote_code`); `--trust_remote_code` and the dtype flag, spelled `--rm_dtype`, by the reward-model scorers `score.py` and `rejection_sample.py` (their device flag is their own `--rm_device`); the Hub source block (`--model_id` without `--revision`) and `--trust_remote_code` by `before_training/deduplicate_dataset.py`; the OpenAI-compatible endpoint block (`--base_url`, `--api_key`), taken by every CLI that drives a served model — `openai_batched_generation.py`, the two reward-model scorers, `eval_env.py`, `eval_code_contests.py` and the two Gradio playgrounds — with a required `--model` on all but the playgrounds (the chatbot declares its own optional `--model`; the environment playground takes the name in its UI) |
| `scripts/inference/inference_arguments.py` | The generation block (`--max_workers`, `--temperature`, `--max_tokens`) and the prompt-row fields (`--id_field`, `--prompt_field`, `--local_system_prompt_field`, `--global_system_prompt`), shared by the S3 generation CLI and the reward-model scorers; the S3 dataset block (`--dataset`, `--output_path`, `--subfolder`) and `--checkpoint_interval` of the S3 generation CLI; the Gradio server block (`--host`, `--port`, `--share`) of the two playgrounds |
| `scripts/inference/reward_model/reward_model_arguments.py` | The reward-model scoring block, on top of the previous two |
| `scripts/environments/eval_arguments.py` | The env-eval dataset/endpoint/training-contract/sampling/trajectory flags; their readers, the output writer and `TrainingContract` (the `--training_config` YAML read with the training script's config classes) live in `halo/entrypoints/eval_contract.py` |

The generation CLIs' runs live in `halo/entrypoints/`: `resumable_generation.py` (the S3 prompt dataset and its resume, the merged push, the empty-run refusal, the batched generation run and `run_async_cli`), `reward_scoring.py` (the reward-model load, built on a CUDA device through `apply_family_patches` as a training load is and on CPU with the backend as given, and refused when the backend drops the live sinks a GptOss checkpoint holds, plus the batched scorer and the `score.py` run) and `rejection_sampling.py` (the `rejection_sample.py` run).

Multi-word flags use underscores (`--base_url`), matching the YAML field names a training flag overrides; a training script also takes the dashed spelling of an override ([CLI overrides](configuration-reference.md#cli-overrides)). A tool's on/off flag is negated by argparse's `--no-` prefix (`--no-trust_remote_code`), and a tool flag that is itself a negative keeps underscores (`--no_overwrite`). A training script's flags-only command line also takes transformers' `--no_<field>` for a field that defaults to true (`--no_gradient_checkpointing`); with a YAML, override it as `--<field>=false`. Beyond argparse's `-h`, short options exist only on `s3.py` (`-b`, `-s`, `-v`, `-q`, `-r`, `-n`, `-y`), `gradio_openai_chatbot.py` (`-m`) and `halo launch` (`-n`, `-a`, `-p`). One concept has one spelling across the tools: `--output_dir` for an output directory, `--output_path` for an output file, `--dataset` / `--split` / `--config` for a dataset source, `--model_id` for a model that may be a Hub repo (the reward-model scorers prefix it, `--rm_model_id`, beside the served `--model`; `prepare_dataset.py`'s flags are its config's fields, so it spells `--model_name_or_path`), `--max_tokens` for a generation cap and `--max_workers` for request or episode concurrency.

The checkpoint tools under `after_training/` and `before_training/` take one source/destination pair: `--input_dir` → `--output_dir` for a local checkpoint directory, `--model_id` → `--output_dir` where the source may also be a Hub repo (`patch_vocab.py`, `convert_deepseek_v4_bf16.py`, `convert_mistral4_bf16.py`, `convert_glm5_bf16.py`, `reset_sinks.py`). `reattach_vision_tower.py` takes both: `--input_dir` for the text-only export, `--model_id` for the multimodal base. A Hub `--model_id` is fetched without its vendor weight dumps (gpt-oss's `original/` and `metal/`), which no tool reads or carries into its output.

Three tools keep a differently-shaped source because it is a different thing: `merge_models.py --models` (N inputs), `merge_peft_adapters.py --adapter_dir` (an adapter, not a checkpoint), and `prepare_dataset.py --input`/`--output` (dataset URIs — `s3://`, `hf://` or a local path).

`merge_peft_adapters.py` and `convert_to_bf16.py` select the head with one `--task` (`add_task_arg`): `causal_lm`, `classification`, and `base` on the conversion alone, since an adapter has no bare-backbone task.

Launcher choice: [Two launchers](../parallelism/data-parallelism.md#two-launchers).

## Training scripts

Every training entry script runs the shared skeleton in `halo/entrypoints/`: `parse_script_args` (distributed init, the YAML/CLI parse, the `output_dir` expansion, the `run.log` tee and the `full_determinism` attention switch) → `init_training_script` (parallelism config, run setup, resume resolution, returning the run's `ScriptContext`) → DP-shard-aware dataset loading and the modality verdict (`data_setup.py`) → the policy load and window pins (`model_setup.py`) → `configure_balancing_and_build_callbacks` → `run_trainer`, which emits the canonical start log (mode, EP/CP/TP/ETP/PP/DP sizes and the attention backend the policy was built with) and reorders integration callbacks last so `moe/*` and efficiency keys reach W&B. `with_distributed_teardown(main)()` tears the process group down after a finished or early-stopped run.

Which parallelism axes each trainer takes: [Trainer compatibility](trainer-architecture.md#trainer-compatibility). Launch commands are on each method's page.

| Script | Method |
|--------|--------|
| `scripts/training/sft.py` | [SFT](../training-methods/sft.md), text or VLM (auto-detected); `init_from_scratch` for [pre-training](../training-methods/pretraining.md) |
| `scripts/training/preference/dpo.py` | [DPO](../training-methods/preference/dpo.md), text or VLM; TRL `loss_type` variants |
| `scripts/training/preference/kto.py` | [KTO](../training-methods/preference/kto.md), text or VLM; unpaired binary feedback |
| `scripts/training/preference/smpo.py` | [SMPO](../training-methods/preference/smpo.md), text or VLM; reference-free |
| `scripts/training/reward/reward_model.py` | [Bradley-Terry reward model](../training-methods/reward-modeling.md) |
| `scripts/training/grpo/online.py` | [Online GRPO](../training-methods/grpo/online.md) scored by the config's reward terms; [online SDPG](../training-methods/distillation/online-sdpg.md) with `--use_sdpg=true` |
| `scripts/training/grpo/offline.py` | [Offline GRPO](../training-methods/grpo/offline.md) on pre-scored completions |
| `scripts/training/grpo/environmental.py` | [Async GRPO with Environments](../training-methods/async-grpo/README.md); `environment_type` picks the [environment](../training-methods/async-grpo/environments/README.md#registry) |
| `scripts/training/classification.py` | [Sequence classification](../training-methods/classification.md), multi-class or multi-label |
| `scripts/training/embedding.py` | [Embedding fine-tuning](../training-methods/embedding.md) with SBERT losses |
| `scripts/training/distillation/teacher_distill.py` | [Teacher distillation](../training-methods/distillation/teacher-distillation.md), off-policy, text or VLM |
| `scripts/training/distillation/self_distill.py` | [Self-distillation](../training-methods/distillation/self-distillation.md): offline privileged-context SDPG, text or VLM |

## Inference & generation scripts

| Script | Description |
|--------|-------------|
| `scripts/inference/generation/openai_batched_generation.py` | Async batched generation with OpenAI-compatible API. Every request goes out as plain text, so rows carrying a structured `response_format` (any `type` but `text`) are refused before generation; the reward-model scripts honor one |
| `scripts/inference/reward_model/rejection_sample.py` | Rejection-sampled preference (or offline-GRPO) dataset generation: hypotheses from any OpenAI-compatible endpoint, scored by a local reward model. A hypothesis the endpoint cut at `--max_tokens` is dropped rather than scored as if it had finished, and a row left with fewer than two usable hypotheses is skipped; both counts are in the run summary. Each generated turn is written as `{role, content}` plus the reply's non-null extras (an engine's `reasoning_content`); null reply fields are dropped |
| `scripts/inference/reward_model/score.py` | Score datasets using reward models. A response cut at `--max_tokens` is dropped and counted (`truncated=`) rather than scored as if it had finished. The scored turn is `{role, content}` plus the reply's non-null extras (an engine's `reasoning_content`) |
| `scripts/inference/playground/gradio_openai_chatbot.py` | Chatbot UI (Gradio + OpenAI API). `--api_key` defaults to `$VLLM_API_KEY`, else `$OPENAI_API_KEY`, else vLLM's `EMPTY` placeholder |
| `scripts/inference/playground/gradio_environment_playground.py` | Environment playground (Gradio) for testing GRPO environments against a rollout server. Episodes run through the shared eval driver (`run_episode` in `halo/environments/eval_runner.py`), so a turn the engine cut off at its token cap, or one the model ended on nothing, is recovered here exactly as in training. `--base_url` prefills the rollout server base URL; `--api_key` defaults to `$VLLM_API_KEY`, else `$OPENAI_API_KEY`, else vLLM's `EMPTY` placeholder, and stays server-side; `--host` binds loopback (`127.0.0.1`), so publishing the UI — and that key's spend — takes an explicit `--host 0.0.0.0` |
| `scripts/environments/inference/eval_code_contests.py` | Evaluate a served model on competitive programming, success@k per rating, difficulty or contest: [Code Contests → Evaluation](../training-methods/async-grpo/environments/tasks/code-contests/evaluation.md) |
| `scripts/environments/inference/eval_env.py` | Evaluate a served model on any other environment (`qa_search`, `exam_qa`, `swe`, `mcp`, …): [Evaluation](../training-methods/async-grpo/evaluation.md#running-an-evaluation) |
| `scripts/environments/inference/regrade_trajectories.py` | Re-grade saved `eval_code_contests.py` trajectories in one bounded process, apart from generation: [Re-grading](../training-methods/async-grpo/evaluation.md#re-grading-recorded-trajectories) |

In `openai_batched_generation.py`, `--dataset` / `--output_path` name S3 datasets. A full
`s3://<bucket>/<key>` URI is used as given; a bare key is placed by `build_s3_uri` under `--subfolder`
(default `datasets`, `None` to skip) in `HALO_S3_DEFAULT_BUCKET`, and raises naming its flag while that
variable is unset.

`score.py` and `rejection_sample.py` read `--dataset` (a local JSONL) and append to `--output_dir`. `--follow_up_prompt_field`
names follow-up turns sent after the first answer, and `--correct_answer_field` the ground truth shown to the reward
model. That model loads from `--rm_model_id` with `--rm_attn_implementation` (default `sdpa`) and drops, counted, a
conversation over `--rm_max_seq_len` tokens (`16000`). `rejection_sample.py` adds `--n_hypos` hypotheses per prompt (`5`,
at least 2) scored `--rm_max_batch_size` per forward (`8`). `gradio_openai_chatbot.py --stop_token_ids` takes a
comma-separated list passed to vLLM.

The three async CLIs — `openai_batched_generation.py`, `rejection_sample.py`, `score.py`
— run under a shared SIGINT/SIGTERM handler
(`run_async_cli`) that exits `128 + signo`, the shell's own convention for a signalled process.
Progress is stored for resume (generation through `parallel_openai_requests` and its JSONL resume store under `$HALO_DATA_ROOT/inference/resume`, `halo/inference/batch_requests.py`; the two reward-model CLIs in their own JSONL output), so re-running resumes; the non-zero exit is what stops a wrapper script or
`&&` chain from consuming a partial output dataset as a finished one.

A run that produced no usable row raises rather than writing an empty result (`reject_empty_results`), so a dead endpoint or a
wrong `--model` cannot republish the resumed rows as a finished job. Each CLI's summary line
names its per-reason drop counts (first-response failures, degenerate skips).

Every endpoint these CLIs talk to is OpenAI-compatible (`--base_url` / `--api_key`),
reached through the one shared client (`build_openai_client`, `halo/inference/openai_client.py`) with the toolkit's retry policy;
`--base_url` defaults to `http://localhost:8000/v1` and `--api_key` to `VLLM_API_KEY`, then `OPENAI_API_KEY`, then the
`EMPTY` placeholder (`halo/inference/endpoints.py`). There
is no separate Azure mode — point `--base_url` at the deployment's OpenAI-compatible route like
any other endpoint.

```bash
python scripts/inference/generation/openai_batched_generation.py \
    --model my-model --dataset s3://your-bucket/datasets/prompts --output_path s3://your-bucket/datasets/responses

python scripts/inference/reward_model/score.py \
    --model my-model --dataset data/prompts.jsonl --rm_model_id path/to/reward-model
```

## Post-training scripts

| Script | Description |
|--------|-------------|
| `scripts/after_training/merge_peft_adapters.py` | Merge LoRA/PEFT adapters into the base model, loaded through the class the adapter's keys address (the text-only class after a `text_only_model` run, whose merged checkpoint keeps the text-only `model.*` keys its `*_text` config declares and ships the run's tokenizer with no processor files); refuses a native EP expert-LoRA directory, a per-rank-sharded adapter or base, and an adapter whose keys name no module of the base — PEFT would merge nothing there and the tool would write the bare base. The base path comes from `adapter_config.json`; `--task classification` (with `--num_labels`, default 1, refused under any other task) merges a reward/classification head, `--dtype` (default `bfloat16`) is the load dtype and `--device_map cpu` fits a model larger than the GPUs. The processing class is the adapter directory's when it saved a full processor, else the base's (a tokenizer-only save would leave an unloadable VLM); a text-only merge ships the run's tokenizer. A base with fewer embedding rows than the adapter's tokenizer has ids, whose own tokenizer fits its rows, grows to it first (the run's own grow); added rows an adapter does not hold (one saved before adapter saves carried them) are rebuilt, with a warning, as the mean of the checkpoint's rows, the frozen init the run gave them. Re-applies the training sidecars (`apply_training_sidecars`), since the merge rebuilds the base from the hub. Norms stay at `--dtype` except modules a family pins through transformers' `_keep_in_fp32_modules_strict` (GPT-OSS's non-strict `_keep_in_fp32_modules` holds only under `--dtype float16`); `--fp32_norms` writes every normalization leaf in fp32 (`convert_to_bf16.py`'s layout) |
| `scripts/after_training/merge_models.py` | Merge same-architecture checkpoints in weight space (linear / slerp / task_arithmetic / ties). Streams one tensor at a time across the inputs (each key loaded from every model, merged, written), so peak host memory scales with the largest tensor, never the merged model ([Model Merging](model-merging.md#memory-and-output)) |
| `scripts/after_training/merge_ep_shards.py` | Merge EP sharded checkpoints into a single model, writing each family's `ExpertLayout` through the same `to_hub_expert_tensors` the gathered save applies (GptOss gate/up re-interleaved) (`--max_shard_size` caps the output shards, default `5GB`). It casts nothing — the sharded writer already applied the save dtype — and needs `ep_size` in the per-rank index metadata, so a gathered or already-merged directory is declined. It carries the resume state, every rank's optimizer shard and `optimizer_meta.pt` included, so the merged directory resumes exactly on the save's topology; a meta naming a rank with no shard is refused. The per-rank files and the meta are hard-linked when `--output_dir` is on the input's filesystem and copied with their mode otherwise, and the disk preflight counts only those on a filesystem other than the output's (an estimate: two bind mounts of one volume, a protected hard link or a project-quota boundary refuse a link on one filesystem, and that file is then copied uncounted). A later training save into either directory replaces the linked files rather than writing through them. `--delete_input_shards` frees the per-rank weight shards once the merged checkpoint is complete — until then peak disk holds both — and leaves the input's resume state in place. Refuses an input holding a PEFT adapter, or only its `adapter_config.json`, beside the shards: the aux copy would carry the config but no adapter weights. Which runs may save sharded: [Checkpoints](checkpoints.md#expert-parallelism-ep-eptp-epcp) |
| `scripts/after_training/merge_pp_stages.py` | Merge the per-node directories of a pipeline-parallel save (`--input_dir` one directory per node, `--output_dir`) into one `from_pretrained`-loadable checkpoint that resumes exactly on the save's topology: the inputs must carry one index and one `optimizer_meta.pt`, each part the index names is copied from the first input holding it, every input's optimizer shards and RNG states are hard-linked when `--output_dir` is on their filesystem (copied with their mode otherwise; a later training save into either directory replaces them rather than writing through the link), and a part or a rank's shard no input holds is refused ([Per-node export](../parallelism/pipeline-parallelism.md#per-node-export)) |
| `scripts/after_training/convert_to_bf16.py` | Convert model weights to BF16, every normalization leaf (a torch norm base, or a class named `*Norm*`) forced back to fp32. Re-applies the source's training sidecars to a full model before saving (neutralized GptOss sinks, balancing tensors re-read from the source shards at their trained fp32) and carries `router_balancing_biases.pt` / `training_provenance.json` into the output either way, so an adapter conversion still hands them to the later merge (`merge_peft_adapters.py`, the one merge tool). `--peft` rewrites every floating tensor of `adapter_model.safetensors` at bf16 without loading the base (a bf16 adapter keeps its weights bit for bit) and carries the adapter directory's other files across, so the later merge resolves the processing class off the same directory. `--task` (`causal_lm` default, `classification`, `base`; the same flag as the merge's) picks the class a full checkpoint, or `--check_inference` on an adapter, loads with. `--peft` refuses `--task base`, which no adapter is trained for, and `--device_map`, since it loads no model, both before any write. `--verify` asserts the STORED dtypes read from the saved safetensors headers — never a `from_pretrained` reload, which casts on the way in and so can never fail — weighted by parameter count, not tensor count. A `--peft` save verifies its adapter file the same way. `--check_inference` is the separate diagnostic: it reloads the saved checkpoint and prints what it generates (an adapter with no tokenizer takes its base's; with none at all it skips), returning no verdict — `--verify` is the gate that raises — and `--task base`, with no head to run, refuses it |
| `scripts/after_training/quantize_to_lowp.py` | Quantize a bf16/fp32 checkpoint to block-scaled mxfp8/mxfp4/nvfp4 (pairs with QAT — see [Mixed-Precision Training](../optimization/low-precision-moe-kernels.md)). `--format {mxfp8,mxfp4,nvfp4}` is required — the checkpoint names no target. The training run's lowp **scope** is not recorded in the checkpoint either, so four more flags restate it under the `ParallelismConfig` names and defaults — a config's values transfer verbatim: `--lowp_apply_dense_mlp` / `--lowp_apply_moe_experts` (both on; `--no-` prefix to disable) and `--lowp_keep_first_blocks` / `--lowp_keep_last_blocks` (both `0`). It quantizes the dense MLP projections and the declared expert weights, fused 3-D banks included (`experts.gate_up_proj` / `down_proj`), that match `--include` and not `--exclude` (regexes). A 2-D `*.weight` matrix blocks along its last axis, as the QAT forward does; a fused 3-D bank uses the `fused_contraction_axis` its family spec declares, and an unresolvable family raises. A sharded input gets a rebuilt index and every non-weight file copied. A scope that selects no block-divisible weight is refused before the output exists. The output is no `from_pretrained` checkpoint: its `quant_method: block_scaled` is refused by vLLM and SGLang, while plain transformers only warns and random-initializes every quantized weight |
| `scripts/after_training/reset_sinks.py` | Reset attention sink tokens in `--model_id` (a local checkpoint directory or a Hub repo id). `--dry_run` prints every sink tensor without writing. `--output_dir` is required for a write unless `--in_place` is given (the two are mutually exclusive; `--in_place` takes a local directory only); a `--dry_run` only reads and needs neither. It works shard by shard and loads no model code (no `--trust_remote_code`): only the shards holding a sink are rewritten, one resident at a time, and the rest are hard-linked (or copied) across, so the output keeps the source's shard layout and index. Every rewritten shard is staged as a sibling, verified at its dtype min, and only once all of them verified renamed over its target, so a write that kept live sinks raises with the target untouched; a failed rename names the verified shards still to place. `--revision` pins a Hub source. A GPT-OSS checkpoint with no key named like a sink raises rather than copying through, and a family without sinks is copied through unchanged. It writes no `training_provenance.json` — the source's record is carried over verbatim, so a checkpoint whose sinks it just neutralized can still claim `live` to the merge tools that trust that record. Only `save_peft_adapters` writes the file |
| `scripts/after_training/unfuse_moe_experts.py` | Unfuse contiguous fused-GLU MoE experts (`experts.gate_up_proj [E,2I,H]`/`down_proj`), `--input_dir` → `--output_dir` (`--max_shard_size` caps the output shards, default `5GB`), into the per-expert layout the checkpoint's own family declares (`MoEFamilySpec.expert_layout.per_expert_keys`): GLM-4 MoE Lite / Laguna / Qwen3 MoE / Qwen3.5-3.6 / Qwen3.8-Flash-Next / Bailing-Ling / Cohere2 MoE / GLM-5 Next `experts.{i}.{gate,up,down}_proj.weight` — what vLLM's `glm4_moe_lite` loader expects — and LFM-2 / DeepSeek-V4 `experts.{i}.w{1,3,2}.weight`. For Qwen3.5/3.6 and Qwen3.8-Flash-Next that is the spelling transformers' converter fuses on load, not the Hub's: their Hub checkpoints store the language-model experts fused, and only Qwen3.5's MTP layer per expert. A `model_type` whose checkpoint does not store one tensor per expert is refused: Gemma4 and Mistral4 register no per-expert conversion with transformers, GptOss's fused tensor is interleaved, Inkling's hub layout is the interleaved TM `w13_weight` one, Zaya's hub layout is fused already, and Step-3.7 Flash's is per-layer stacked (`moe.{gate,up}_proj [E, M, H]`, `moe.down_proj [E, H, M]`). Weights are unchanged, and a checkpoint with no fused expert keys is copied through |
| `scripts/after_training/reattach_vision_tower.py` | Rebuild the multimodal wrapper layout from a `text_only_model` export: the trained text weights re-prefix to `model.language_model.*`, the base's untrained vision tower and MTP tail stream back in (`--input_dir` = the export, `--model_id` = the multimodal base, hub or local), and the composite config is regrafted with the trained text config. Both pinned engines (vLLM 0.32.0, SGLang 0.5.22) serve a Qwen3.5/3.6 text-only export as-is, and vLLM a Qwen3.8-Flash-Next one; SGLang registers only Flash-Next's multimodal class, so that export runs through this tool first ([Qwen3.8-Flash-Next](../models/qwen4_exp.md#checkpoints-and-serving)). It restores the vision tower, the MTP head and the processor files. A wrapper-layout input is refused, and so is an export whose keys are not the text-only layout its config declares (text keys already under `model.language_model.*`, or a module the base's text tower lacks) |

```bash
python scripts/after_training/merge_peft_adapters.py \
    --adapter_dir /models/sft-llama-lora --output_dir /models/sft-llama-merged

python scripts/after_training/merge_ep_shards.py \
    --input_dir /models/ep-checkpoint --output_dir /models/merged-model
```

### Input guards

Every tool here refuses an input it cannot express, rather than writing a plausible-looking result:

- **Per-rank EP-sharded input** (`metadata.format` = `ep_sharded`) is rejected by all of them except
  `merge_ep_shards.py`, which exists to consume it. Merge first, or re-save gathered.

    Such a save reuses the ordinary index filename while each expert tensor is one rank's partial
    slice under a `.shard_N` key. A `from_pretrained`-based tool would see the real expert keys as
    **missing**, which transformers resolves by randomly initializing them (warning only), and then
    save that over the source.

    The refusal does not need the index: a directory whose shards carry `.shard_N` keys with no
    index (a save killed before the index write) is caught by a header-only peek. The
    `scripts/before_training/` bf16 converters refuse it on the same terms, since all of them also
    accept a local directory.

- **In-place conversion** (`--output_dir` equal to any input directory) is rejected by every tool
  that streams from the source while writing: the write deletes the weight files it does not
  overwrite and so destroys the source mid-read. A refused conversion creates no output directory.

    That is `merge_ep_shards.py`, `merge_pp_stages.py` (on every node directory), `unfuse_moe_experts.py`,
    `quantize_to_lowp.py`, `merge_models.py`,
    `merge_peft_adapters.py`, `convert_to_bf16.py`, `reattach_vision_tower.py` (on both of its
    sources), `convert_deepseek_v4_bf16.py`, `convert_mistral4_bf16.py`, `convert_glm5_bf16.py`, and
    `patch_vocab.py`, whose save would overwrite the source's shards in place with no undo, and
    whose post-save sweep then deletes every source shard the save did not rewrite.

    `prepare_dataset.py` is not one of them: it bakes into a staging sibling of its `--output` (a temp
    directory for S3 or the Hub) and publishes with a swap once the bake is done, so its `--output` is
    guarded by `--overwrite`, not by this refusal.

    `reset_sinks.py` is the exception: it stages and verifies every shard it rewrites before any
    replaces its target, so in-place is safe there, but never implicit. It takes `--in_place`; an `--output_dir` aimed at
    the input is refused like the rest.

    Omitting both is an error on a write (a `--dry_run` only reads) rather than a silent rewrite of
    the only copy. A fresh-dir run sweeps leftovers like the others.

- **Remote code.** Every checkpoint tool that loads a checkpoint's own modeling/config/tokenizer code
  takes one `--trust_remote_code`, spelled once (`add_trust_remote_code_arg` in
  `scripts/cli_arguments.py`): `merge_peft_adapters.py`, `merge_models.py`,
  `convert_to_bf16.py`, `reattach_vision_tower.py`, `patch_vocab.py`,
  `prepare_dataset.py`, `convert_deepseek_v4_bf16.py`, the reward-model scoring CLIs
  (`score.py`, `rejection_sample.py`) through `scripts/inference/reward_model/reward_model_arguments.py`,
  and `deduplicate_dataset.py`.

    **The default follows the input source.** A local checkpoint or merge input (`--input_dir`,
    `merge_models.py`'s `--models`, `--base_model` and `--tokenizer_source`, all local directories, a Hub
    id refused) defaults **on**: the remote-code families in the roster (Bailing/Ling, Laguna) do not load
    without it, and the operator already produced that artifact.

    A source that may be a Hub repo (the `--model_id` of `patch_vocab.py`, `convert_deepseek_v4_bf16.py`,
    `reattach_vision_tower.py` and `deduplicate_dataset.py`, the scorers' `--rm_model_id`, and
    `prepare_dataset.py`'s `--model_name_or_path`) defaults **off**: a freshly downloaded
    third-party repo must not execute its own code merely because a tool was pointed at it.

    An adapter (`merge_peft_adapters.py`'s `--adapter_dir`, `convert_to_bf16.py --peft`) follows the base its
    `base_model_name_or_path` names: **on** for a local directory, **off** for a Hub id, so an adapter on a
    Hub base of a remote-code family needs `--trust_remote_code`. Everywhere the opposite is one flag away
    (`--trust_remote_code` / `--no-trust_remote_code`).

    The tools that never enable remote code expose no such flag — `merge_ep_shards.py`,
    `merge_pp_stages.py` (it copies files), `unfuse_moe_experts.py`, `quantize_to_lowp.py`, `reset_sinks.py`,
    `convert_mistral4_bf16.py` and `convert_glm5_bf16.py` stream safetensors and rewrite `config.json` as
    JSON, so no checkpoint code is ever imported.

- **Uninitialized buffers.** A tool whose loaded model is handed to a forward (reward scoring, the
  dedup embeddings, `convert_to_bf16.py --check_inference`) calls `finalize_loaded_model` after
  placement, since transformers 5 re-materializes non-persistent buffers as uninitialized memory
  ([buffers](../models/adding-a-model.md)).

    The load-only conversion tools do not: `state_dict` omits those buffers, so an unrepaired one
    reaches neither a number nor the written checkpoint.
    `tests/cpu/model_loading/test_load_finalization.py` pins each new tool loader, in a script or its
    `halo/` driver, into one of the two groups.

- **Shard cap and Hub source.** Two more flags are spelled once each, beside
  `add_trust_remote_code_arg`. `add_max_shard_size_arg` writes `--max_shard_size` (default
  `DEFAULT_MAX_SHARD_SIZE`, `5GB`) for every tool that re-shards its safetensors output;
  `quantize_to_lowp.py` and `reset_sinks.py` mirror the input's shard layout one-for-one and take no cap.

    `add_hub_source_args` writes `--model_id` plus the `--revision` pin for every tool whose source
    may be a Hub repo. `patch_vocab.py` takes `--model_id` without `--revision`:
    it threads none, and an advertised pin a tool ignores would silently convert whatever the Hub's
    default branch holds that day. Pin such a source by downloading it first and pointing
    `--model_id` at the directory.

- **Wrong model family.** `unfuse_moe_experts.py` resolves the family from the checkpoint's
  `model_type` and emits the projection names it declares
  (LFM-2 reads `w1`/`w3`/`w2`, not GLM-4's `gate_proj`/`up_proj`/`down_proj`). A family whose checkpoint
  is not per-expert at all is refused rather than written under a guessed triple.

    Both checks run after the sharded-input refusal and after the already-per-expert copy-through,
    so those keep their own diagnosis.

- **`merge_models.py` inputs.** Asymmetric key sets (past a Hub base's keys the shipped model never
  reads), a key stored at different shapes, a `--tokenizer_source` that is not a local directory and a tokenizer-less one are refused
  ([Model Merging](model-merging.md#memory-and-output)).

## Preparation scripts

| Script | Description |
|--------|-------------|
| `scripts/before_training/prepare_dataset.py` | Pre-process SFT or raw-text datasets: tokenization, packing, sharding. `--mode chat` (default) applies the chat template; `--mode text` tokenizes `--text_field` directly and appends EOS per document — the (continued) pre-training path. The output holds `train` and `test` (an input with no `test` bakes its `validation` split as `test`); any other input split is left out with a warning. A `--conversation_field` / `--text_field` / `--tools_field` / `--images_field` the run reads but a baked split does not carry is refused before tokenization |
| `scripts/before_training/s3.py` | CLI over the S3 transport: `push`, `download`, `list`, `exists`, `delete` a folder at a bare key under `--bucket` (default `HALO_S3_DEFAULT_BUCKET`) or at a full `s3://bucket/key` URI. `delete` removes ONE object; `--recursive/-r` is opt-in for a whole prefix. See [S3 Utilities](../data/s3-utilities.md) |
| `scripts/before_training/deduplicate_dataset.py` | FAISS-based semantic deduplication. Keyed on `--text_field` alone: two rows with the same text collapse to one however they differ elsewhere, so an images column does **not** make them distinct — deduplicate a VLM dataset before pairing its images, or on a field that carries the difference. `--dataset` is a Hub dataset id, an `s3://` URI, a local data file or a `save_to_disk` directory, read at `--config`/`--split` like the eval CLIs' (`load_dataset_split`), and `--output_path` a local path |
| `scripts/before_training/patch_vocab.py` | Add `--patterns` (a JSON list, inline or in a file) to `--model_id`'s vocabulary and resize its embeddings, new rows the mean of the existing ones, into a fresh `--output_dir`. Growth only; a pattern already one token is skipped. `--chat_template` sets the tokenizer's template, `--reset_sinks` resets attention sinks to their minimum before the save. A source with no tokenizer file is refused |
| `scripts/before_training/convert_deepseek_v4_bf16.py` | Dequantize a FineGrained-FP8 DeepSeek-V4 checkpoint to uniform bf16 for training; `--max_shard_size` sizes the output shards (default `5GB`; see [DeepSeek-V4](../models/deepseek-v4.md)) |
| `scripts/before_training/convert_mistral4_bf16.py` | Dequantize a public FP8 Mistral Small 4 checkpoint to bf16, streaming shard-by-shard (no full-model RAM footprint); `--max_shard_size` caps the output shards (default `5GB`; see [Mistral4](../models/mistral4.md)) |
| `scripts/before_training/convert_glm5_bf16.py` | Dequantize the fp8 block-quantized GLM-5.3-Flash release to bf16 (block-wise `weight * scale_inv`, unquantized tensors keep their stored dtype), streaming shard-by-shard; `--model_id` may be a Hub repo; `--max_shard_size` caps the output shards (default `5GB`; see [GLM-5 Next](../models/glm5-next.md)) |
| `scripts/environments/preparation/compact_code_tests.py` | Reduce a bulky test corpus to one capped row per problem for `prepare_code_dataset.py --tests_table`: `--source hardtests` (base64-zlib-pickled suites) or `codeforces_generated` (open-r1's one-row-per-test parquet), `--input_dir` shards → `--output_dir` parquet parts (`key`/`tests`/`checker`), one per source shard, stamped with the caps that built it (a part with the same caps is kept, so an interrupted run resumes; other caps rebuild it). At most `--max_tests` 40 tests within `--max_test_bytes` 262144 bytes each, of which `--max_large_tests` 2 may reach `--max_large_bytes` 4000000; source order preserved; `--num_proc` shard workers (default: the toolkit's dataset-processing default); undecodable and emptied suites counted in the final log line |
| `scripts/environments/preparation/prepare_code_dataset.py` | Prepare a competitive-programming dataset (`--adapter codeforces`/`deepcoder`/`hardtests`) for the coding RL env — composes prompts, packs tests/checker/time-limit. `--tests_table <dir>` joins a compacted suite onto the rows by problem id (a table matching no row of any split exits); `--exclude_keys <file>` drops listed problem ids (a trailing `*` excludes a prefix); `--holdout_per_band N` carves a deterministic `test` split (N rows per rating band plus N below the first band) from a source that ships only `train`; `--push_bands` pushes the `full` config plus one per rating band over the shared `test` split, `--config_name` a single config (each needs `--push_to_hub`, and the two exclude each other); `--verify_checkers` (default on, `--no-verify_checkers` off) drops a problem whose special judge rejects its own reference output or accepts garbage on the first test, running the judge in a `SandboxExecutor`. `--min_rating`/`--max_rating` also drop unrated problems, and a source whose staged rows have no `rating` column (`deepcoder`) refuses them. A problem whose only tests are its statement's examples is dropped and counted per split unless `--include_examples_only`, which an adapter marking no such problem refuses. `--num_proc` defaults to the toolkit's own dataset-processing default; a `--push_to_hub` repo is created private unless `--no-private`. See [Code Contests](../training-methods/async-grpo/environments/tasks/code-contests/datasets.md) |

```bash
python scripts/before_training/prepare_dataset.py \
    --input "s3://bucket/raw/dataset" --output "s3://bucket/preprocessed/dataset" \
    --model_name_or_path "Qwen/Qwen3-8B" --max_length 8192 --test_size 0.01 --num_shards 64 \
    --pack_sequences --assistant_message_template $'<|im_start|>assistant\n'

python scripts/before_training/patch_vocab.py \
    --model_id Zyphra/ZAYA1-8B --output_dir <large-volume>/models/ZAYA1-8B-patched \
    --patterns '["<tool_call>", "<tool_result>"]'
# GptOss sink reset: --reset_sinks
# remote-code family (Bailing/Ling, Laguna): add --trust_remote_code — a Hub source is opt-in here
```

`patch_vocab.py` only **grows** the embedding (added tokens reuse existing padding rows), never shrinks it: models like GPT-OSS ship a vocab padded past `len(tokenizer)`, and shrinking would drop the high special tokens (harmony `<|return|>`/EOS, …) and break generation. A patched checkpoint therefore serves like its base, except a GPT-OSS patched with `--reset_sinks`: served sinks-off, the pretrained model degenerates ([GPT-OSS](../models/gpt-oss.md#serving-for-grpo-vllm)). The added tokens start at the mean embedding and mean nothing until trained.

`deduplicate_dataset.py` embeds `--text_field` with `--model_id` (default `sentence-transformers/all-MiniLM-L6-v2`,
capped at `--model_max_length` tokens, `--batch_size` per forward on `--device`), or reads `--load_embeddings <file.npy>`
instead; `--save_embeddings [PATH]` keeps what it computed (default `<output_path>.embeddings.npy`) and is refused
beside `--load_embeddings`. A row whose cosine similarity to an earlier one exceeds `--similarity_threshold` (`0.95`)
is dropped, over `--dedup_steps` reshuffled passes (`3`) of `--dedup_batch_size` rows per search batch (`100000`) on
`--max_workers` threads. `--sample_size N` runs on a seeded sample and `--keep_fields` whitelists the output columns.

## Profiling & benchmarks

| Script | Description |
|--------|-------------|
| `scripts/profiling/trace_report.py` | TraceLens analysis workbooks from torch.profiler traces — per-rank perf report + multi-rank NCCL collective/skew report (see [Debugging](debugging.md#1b-tracelens--automated-trace-analysis)). Trace sets are independent, so one unreportable capture does not cost the others' workbooks, but any failure exits non-zero; `--keep_going` accepts a partial run |
| `scripts/profiling/py_spy_diag.py` | Attach py-spy to a running job. Takes a subcommand: `dump` (instantaneous all-rank stack dump, hang triage) or `record` (per-rank CPU flame graphs over `--duration` seconds). Both default to this node's torchrun ranks, attach to every one at once, and exit 1 naming each pid they could not attach to; `--pid` (repeatable) targets others |
| `scripts/profiling/nvlink_health.py` | Torch-free NVLink preflight from `nvidia-smi`: per-link hard errors, FEC correction depth, active bandwidth. Exits non-zero when a link is unhealthy (hard errors) or down (inactive, or no speed reported — an unwired link nvidia-smi lists included, so the check is for fully wired NVLink nodes), or when a GPU has fewer active links than its peers; correctable FEC churn, the benign `Xid 145` source, is reported as marginal and passes. `--per_link` prints every link, `--json` the raw report (see [Debugging](debugging.md)) |
| `scripts/profiling/weight_sync_transport.py` | Weight-sync transport preflight against a live rollout server (driver `halo/weight_sync/transport_check.py`): forms the group with the toolkit's own client, pushes one parameter of the served checkpoint unchanged and names the transport NCCL formed on. `--server_url` (required), `--backend vllm\|sglang`, `--group_port` (`0` = ephemeral), `--group_host` (auto), `--device` (`cuda:0`), `--param` (the input embedding), `--model_id` / `--revision` (the checkpoint the server loaded; default the id it advertises), `--rounds` (3), `--connection_timeout` (120 s), `--expect efa\|ib\|socket\|p2p\|shm` (exit 1 on a mismatch or an altered served model), `--json`. Procedure and transport legend: [Rollout Servers](../infrastructure/rollout-servers.md#servers-on-other-nodes-efa) |

Throughput benchmarks live under `tests/gpu/profiling/`, grouped by what they measure:

| Group | Scripts |
|---|---|
| Trainer × parallelism throughput | `benchmark_sft_dense`, `benchmark_sft_ep`, `benchmark_smpo_ep` — each takes `--model` from the shared roster (`tests/common/models.py`) and its parallel layout as flags (`--ep`, `--cp`, `--tp`, …), so a per-model or per-layout number is a flag, not a script |
| Kernel / mechanism A/B | `benchmark_attention_implementations` (FA4/FA2 vs flex/sdpa/eager, fwd + bwd), `benchmark_gemma4_attention` (Gemma 4 attention backends per layer type), `benchmark_grouped_mm` (loop vs `grouped_mm`), `benchmark_moe_block` (the Gemma 4 26B-A4B routed-expert block: transformers eager / `grouped_mm` vs Halo's grouped path), `benchmark_torch_compile` (Liger × compile 2×2 on EP), `benchmark_collators` (packing vs padding-free), `benchmark_roofline`, `bench_ep_buffer_backends` (DeepEP V1 Buffer vs V2 ElasticBuffer) |
| Stock-TRL reference | `benchmark_trl_baseline` (same `EfficiencyCallback` metrics), `benchmark_convergence` (loss curves, same seed + data order) |

`run_baseline_benchmarks.sh` in the same directory runs the rows `make bench` holds to `tests/baselines/bench/`
([Golden performance baselines](../contributing/README.md#golden-performance-baselines)).

```bash
make bench BAND=1
torchrun --nproc_per_node=8 tests/gpu/profiling/benchmark_smpo_ep.py --ep 8 --cp 2
```

## Docs scripts

| Script | Description |
|--------|-------------|
| `scripts/docs/check_links.py` | The docs link gate (`scripts/docs/check_links.sh`, `make docs`): every relative link, heading anchor and code reference in the given markdown files or directories (default: `agent-docs`, `human-docs`, `skills` and the root markdown) must resolve; exits 1 naming each one that does not |

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
