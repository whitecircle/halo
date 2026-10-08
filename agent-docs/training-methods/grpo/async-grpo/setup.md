# Servers and Launch

## GPU split

Trainer GPUs must not overlap the server's: one process cannot NCCL-broadcast to itself. Nothing checks it up front — `CUDA_VISIBLE_DEVICES` is the enforcement, and the compose files put the server on GPU 7, the trainer on 0–6. An overlap fails group formation in `init_communicator()`, which names the trainer device; a server that never joins hits the same 120 s deadline.

## Rollout backend

`rollout_backend: vllm` (default) or `sglang`. Both serve rollouts over `/v1/chat/completions` and take weights over NCCL. A pair the engine cannot update online is refused at construction with its loader reason ([which families each serves](../../../infrastructure/rollout-servers.md#which-families-each-engine-serves)). SGLang also refuses `rollout_max_thinking_tokens` and [`carry_reasoning`](rollouts.md#carried-reasoning).

TRL's `top_p`, `top_k`, `min_p`, `repetition_penalty` and `generation_kwargs` reach no sampler here, so one set away from its TRL default is refused at trainer construction; the first four have `rollout_*` equivalents (`rollout_top_p`, `rollout_top_k`, `rollout_min_p`, `rollout_repetition_penalty`). `temperature` is force-set to `rollout_temperature`, so log-probs are scored at the sampling temperature.

Rank 0 probes each server at startup and broadcasts its verdict. The context check **raises** when a turn cannot fit (`max_prompt_length` + the environment's measured prompt overhead + `rollout_max_tokens`), and warns on the multi-turn worst case (that prompt budget plus the smaller of `max_turns × rollout_max_tokens` and `rollout_max_episode_tokens`). `verify_backend()` fails the launch on an unreachable external judge.

## Starting a server

```bash
make build-vllm
VLLM_MODEL=<hub-id-or-path> VLLM_CUDA_DEVICES=7 \
    docker compose -f docker-compose.vllm.yml up -d vllm-server

make build-sglang
SGLANG_MODEL=<hub-id-or-path> SGLANG_CUDA_DEVICES=7 \
    docker compose -f docker-compose.sglang.yml up -d sglang-server
```

Confirm `/health` (`:8000` vLLM, `:30000` SGLang) first, or client init hangs up to `rollout_connection_timeout` (default 120 s). Parsers, `--moe-backend triton` and the rest: [Rollout Servers](../../../infrastructure/rollout-servers.md#vllm).

## Launching the trainer

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 DIST_NCCL_TIMEOUT_MINUTES=60 torchrun --nproc_per_node=4 \
    scripts/training/environmental_grpo.py <config>
```

Add `--expert_parallel_size=4` for MoE expert distribution. `halo launch environmental-grpo <config> --nproc 4` builds the same line; `accelerate launch` works only for plain data parallelism on a dense model: on a MoE it refuses the grouped-GEMM expert wrappers that weight sync needs. Context and Pipeline Parallelism are rejected at config time.

## Expert parallelism group size

Dense models and `ep1` configs skip this. The trainer ranks must form **one** DeepEP dispatch group: set `expert_parallel_size` to the trainer-GPU count. Above `ep_size: 2`, a group narrower than the NVLink domain is rejected at config time — `ep4` on 4 trainer GPUs is fine, on 8 it is not ([DeepEP](../../../infrastructure/deepep.md#ep-grouping-what-is-reliable)).

- `ep_size: 8` needs 9 GPUs or a second node, since the server holds one.
- ZeRO-3 (`fsdp_reshard_after_forward: true`) is rejected wherever an expert-distribution group exists: its backward all-gather races the DeepEP combine. Available at `ep_group_size: 1` — `ep_size: 1` with no expert TP ([ZeRO-2 vs ZeRO-3](../../../parallelism/data-parallelism.md#zero-2-vs-zero-3-reshard_after_forward)).
- `beta > 0` without PEFT holds a frozen reference: the script loads it like the policy (run dtype, `model_revision`, attention, `reset_sinks`), from `model_name_or_path` and never a resume checkpoint, and hands it to TRL in place of TRL's fp32 copy ([Online GRPO](../online-grpo.md#grpo-objective-for-verifiable-rewards)). It is an unsharded dense replica per rank, warned about under EP, ETP and TP. Use `beta: 0`, or `use_peft: true` off TP (which refuses adapters), to hold none.

## Weight synchronization

A forced sync runs at train-begin, after the resume restore and before the first rollout or `eval_on_start`; the prefetch thread starts there too, so no rollout comes from pre-restore weights. Afterwards it runs at the **start** of a step, before that round's generation.

`sync_weights_every_n_steps` (default `1`) trades freshness against the pause window; on slow environments sync every 2–4 steps. A declined step pushes nothing.

`vllm_group_port` (TRL `GRPOConfig`, default `51216`) is bound on the **trainer** host, one listener per server, so two servers sharing a port collide even on distinct hosts. In multi-server mode it is the base: an entry with no `group_port` binds `vllm_group_port + index`.

Component start and every push are fail-fast: an error any rank raises at start (its rollout manager or re-score clients, the main process's sync client) or out of a push (the forwarding rank's sends defer theirs to a reject every rank joins) raises on all ranks together. Each EP layer's expert gather transiently materializes the **full expert set** on every rank, so its cost scales with total expert count, not `ep_size`.

## Multiple servers and prefetch

With a **single** server — `rollout_server_url` (default `http://localhost:8000`), or a one-entry `rollout_server_configs` — prefetch is auto-disabled and the sync blocks, since that engine is paused for the push. Two or more enable it.

```yaml
rollout_server_configs:
  - {url: "http://localhost:8000", group_port: 51216}
  - {url: "http://localhost:8001", group_port: 51217}
enable_prefetch: true
```

Every push pauses all servers together: the gather streams each chunk to every server, whatever the model's shape or the training world.

Prefetch runs **one round deep**: a round pops what the previous one submitted, then submits its own. `num_rollout_workers` and an explicit `max_concurrent_rollouts` are refused below `1`; turn prefetch off with `enable_prefetch: false`.

A checkpoint carries each rank's submitted-but-untrained round (`prefetch_pending-<rank>-of-<world>.pt`, with the group size, round size and DP slice it was drawn under), and a resume submits it again before its first round. A save on a generation boundary therefore resumes onto the batches an uninterrupted run trains, and by default every save is on one, since `steps_per_generation` equals the accumulation steps. A save between generation boundaries (a round that feeds several optimizer steps, with `save_steps` landing inside it) resumes into the middle of that round and trains the round twice. A checkpoint with no file for this world size, or one drawn under another layout, resumes with a cold round, which skips that batch and trains the next one twice. A file present on some ranks and missing on others (a non-shared filesystem resumed under another rank-to-node placement, or a torn copy) raises: resume with the saving placement, or delete the `prefetch_pending-*` files to open with a cold round.

`async/prefetch_hit_rate` says which phase bounds the step, not whether the servers are healthy. On a short single-turn environment it should climb toward 1; below ~0.8, add servers or raise `max_concurrent_rollouts` / `num_rollout_workers`. A multi-turn round outlasts the update, so it sits near 0 by construction.

![One rollout server, the compose default: trainer ranks on GPUs 0–6, the engine on GPU 7 joining the NCCL group whose store the trainer binds on :51216, actors generating over POST /v1/chat/completions; the push pauses the engine (POST /pause?mode=keep), prefetch is off, and step time is sync + round + update](../../../assets/diagrams/environmental_grpo_single_server.png)

![Two rollout servers, the code-contests shape: six trainer ranks on GPUs 0–5, engines on GPUs 6 and 7, one NCCL group per server with store ports 51216 and 51217 bound on the trainer host; the streamed sync pauses both servers for the whole push, prefetch keeps the pipeline one round deep, and step time is sync + max(round, update)](../../../assets/diagrams/environmental_grpo_multi_server.png)

## Multi-node

Training, engines and Ray actors can sit on separate nodes. Point `ray_address` at the cluster head ([Ray Cluster](../../../infrastructure/ray.md#multi-node)) and give each inference node its own `rollout_server_configs` entry, `group_port` and, where the trainer's routable address differs per server, `group_host`. Use resolvable host names: a loopback URL reaches actor nodes with no engine, and those episodes error into masked rows outside their group's baseline, shrinking the batch; a round where none survives trips the all-invalid halt ([Batch construction](objective.md#batch-construction)).

`num_rollout_workers` actors are created per training rank, soft-pinned to that rank's node; on a shared cluster the budget divides by world size ([pool sizing](../../../infrastructure/ray.md#pool-sizing)).

The trainer's NCCL address must be routable from the serving nodes: set the process-wide `VLLM_GROUP_HOST` or `SGLANG_GROUP_HOST` when the default-route NIC is wrong ([multi-homed nodes](../online-grpo.md#multi-homed-nodes-vllm_group_host)). The sync runs over EFA when both sides use their EFA overlays ([Servers on other nodes](../../../infrastructure/rollout-servers.md#servers-on-other-nodes-efa)), over sockets otherwise. Co-located on one host, run the trainer container with `--network=host` too, or `localhost:8000` resolves inside its own namespace.

![Two-node async GRPO topology: node 1 holds the trainer ranks (model, optimizer, InferenceClientManager binding one weight-sync store per server) and the Ray actors beside them (num_rollout_workers per rank, each holding the environment), node 2 holds the rollout server on GPUs no trainer rank uses, its workers joining the sync group and dialing the trainer's store; weights cross on NCCL, generation on HTTP, and a network strip lists the ports, the group port plus one per extra server, Ray's head port and the EFA overlay for the cross-node sync](../../../assets/diagrams/multi_node_separate_inference.png)

![Async GRPO across four node roles: one training node, two inference nodes with one rollout_server_configs entry each (its own url and group_port 51216 / 51217, each its own NCCL group with the trainer as rank 0), and a GPU-less actor tier joined through ray_address whose actors post /v1/chat/completions round-robin across the servers; a config card shows the matching YAML](../../../assets/diagrams/multi_node_dedicated_rollout.png)

## Shipped recipes

`examples/grpo/environmental/<family>/<backend>/` — `gemma4`, `gptoss` and `qwen3_5`, each with `vllm/` and `sglang/`. In a filename, `-lora-` / `-full-` is the adapter and `-ep1` / `-ep4` the expert distribution (undistributed experts, or a 4-rank DeepEP group). Each header carries its own launch line and server flags. One template sits at the root: `environmental-grpo-template.yaml` (`react_math`).

Start from `gptoss/vllm/gptoss-20b-code-contests-lora-ep1.yaml` for a tool-heavy graded environment, or `qwen3_5/vllm/qwen3.6-35b-a3b-react-math-full-ep4.yaml` for a light two-tool one. The `sglang/` directories ship ep1 files only.

Every family recipe runs `beta: 0`; the root template runs `0.01`. The code-contests ones drive a two-server pool at `episode_timeout: 2700`, so launch them with `DIST_NCCL_TIMEOUT_MINUTES=60` ([timeout bounds](performance.md#sizing-a-run)). The rest are single-server, prefetch off.
