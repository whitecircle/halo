# Troubleshooting

Symptom → cause → fix for Async GRPO runs. Distributed-training failures (NCCL hangs, OOM) and the weight-sync errors both GRPO arms share (group address, port bind, transport mismatch, SGLang cuMem) are on the toolkit's [Troubleshooting](../../reference/troubleshooting.md) page.

## Training dynamics

| Symptom | Cause | Fix |
|---|---|---|
| `entropy` climbing for tens of steps, reasoning shrinking, `sampling/logratio_mean` widening until `sampling/update_skipped` holds at 1 | A KL-free run (`beta: 0`) whose failures outlast its solves (`advantage/net_token_mass` stays negative) | `balance_token_mass: true` ([Token-mass balance](objective.md#token-mass-balance)); arm the [early stop](monitoring.md#early-stop); resume from the last checkpoint before the drift |

## Servers and weight sync

| Symptom | Cause | Fix |
|---|---|---|
| `Rollout server … cannot serve this run: …` | The server is off the toolkit's image or lacks a launch flag the run reads; the message names each | Serve as it says ([Startup preflight](../../infrastructure/rollout-servers.md#startup-preflight)) |
| `server unreachable at …`, or a stall in `init_communicator()` | The server is down, or never joins the NCCL group (an unroutable advertised master, or trainer and server on one GPU) | Verify `/health`; set `VLLM_GROUP_HOST` / `SGLANG_GROUP_HOST` to a routable NIC; give them different `CUDA_VISIBLE_DEVICES` ([Rollout Servers](../../infrastructure/rollout-servers.md#troubleshooting)) |
| `GENERATION is paused or wedged (no response … to a 1-token probe)` | A trainer was killed while attached to the weight-sync routes; SGLang, which has no pause route, shows a pause this way | `POST /resume` (vLLM) or `/continue_generation` (SGLang), or restart the server if that sync had streamed part of the model |
| `vLLM server … is PAUSED for a weight update` | A trainer died without lifting its `/pause` | `POST /resume`, or restart the server if that sync had opened its update session, which unloads the model: `/resume` then takes the engine down |
| `Prefetch off: 1 rollout server …` (info) | The single-server default: each round rolls out, then trains | `enable_prefetch: true` overlaps the next round with the update, one round stale ([Multiple servers and prefetch](setup.md#multiple-servers-and-prefetch)) |
| `prefetch_depth=N sets how many rounds prefetch holds, but …` | A depth above 1 on a run that does not prefetch: `enable_prefetch: false`, or unset on one server | `enable_prefetch: true` (one server included), or drop `prefetch_depth` ([Multiple servers and prefetch](setup.md#multiple-servers-and-prefetch)) |
| `404 … "The model <path> does not exist"` on every rollout | `model_name` is unset, so the checkpoint path (`model_name_or_path`) is sent as the id | Set `model_name` to the id at `/v1/models`, or serve with `--served-model-name` |

## Rollouts

| Symptom | Cause | Fix |
|---|---|---|
| `DeadlineExpired: request exceeded its …s deadline` | Server overloaded or wedged (sync pauses are excluded); actors retry with backoff (`request_timeout` 120 s, `max_retries` 3, `retry_base_wait` 1 s) | Check server health first; 4xx are terminal except 408, 429 and vLLM's `not JSON compliant` 400 |
| Episodes booked into `episode/error_rate` near `episode_timeout` | The episode (generation, tools, grading) passed its deadline (default 1200 s) | Raise `DIST_NCCL_TIMEOUT_MINUTES` first, then `episode_timeout` ([bounds](performance.md#sizing-a-run)) |
| `500 "Already borrowed"` | A vLLM fast-tokenizer race that scales with concurrency | Harmless unless `gave up after N tries` appears; then raise `max_retries` or `retry_base_wait` |
| `consecutive steps produced no valid episode anywhere in the world` | Two steps in a row had nothing to train on; the message ends with the cause. `reward term '<name>' scored nothing` is a judge or reward-model outage under `on_error: invalid`; a rollout error points at the server or the sandbox | Fix the scorer's key or endpoint, or the server, and resume ([Batch construction](objective.md#batch-construction)) |
| `… rounds hold …, how this model writes a tool call, in their text, and the server parsed no call` | The server's tool parser does not read this model's format, so calls stay text and every episode ties | Serve the family's parser ([Tool calls](episodes.md#tool-calls)). If the parser is right and the calls are malformed or prose, raise `tool_parser_watch_rounds` or set it to `null` |
| `… of this train round's episodes errored or timed out, over episode_error_rate_limit=…` | Most episodes of three train rounds in a row failed, so the batch is the fast survivors or the rollout path is down. The message gives one rank's error; a timeout means turns outran `request_timeout` at `rollout_max_tokens` | Fix the server or sandbox, or raise `request_timeout` / lower `rollout_max_tokens`; `episode_error_rate_limit: null` trains through it ([Halts](monitoring.md#halts)) |
| `No token carried a policy gradient in … train round(s) in a row` | Every group tied or dropped; the message gives the shares and this rank's tool-call count. Tied with no tool calls usually means a parser problem, tied with calls an environment that scores every attempt alike | Fix that. It stops a launch that has not trained yet at `zero_gradient_round_limit` rounds in a row: raise that for a very sparse reward, or set it to `null` ([`zero_gradient_round`](monitoring.md#group-and-entropy-diagnostics)) |
| `ray.exceptions.RayActorError` | The episode returns as an errored, masked-out row and the actor restarts (`max_restarts=-1`) | Check `ray logs` and per-node memory ([Ray Cluster](../../infrastructure/ray.md#monitoring)) |

## Sandboxes and environments

All five are raised at environment construction.

| Symptom | Cause | Fix |
|---|---|---|
| `SweEnvironment has nothing to grade against` | `requires_answer: false` with no `test_function`, while the reward prices the grade | Grade against an `answer` column, pass a `test_function`, or reward with external terms only ([SWE → Reward](environments/tasks/swe.md#reward)) |
| `bubblewrap sandbox runs the jail's root as a subordinate uid` | The image lacks `uidmap`, `unshare` or a subordinate range for root, or the trainer does not run as root | Run the training image, which ships them, as root ([Sandboxes](environments/sandbox.md#choosing-a-backend)) |
| `bubblewrap's jail cannot reach a path it binds` | `TMPDIR` or a directory above it is private, and the jail's root is not this process's uid | Point `TMPDIR` at a directory every user can search, as `/tmp` is ([Sandboxes](environments/sandbox.md#choosing-a-backend)) |
| `bubblewrap cannot mount the jail's /proc` | Docker (and the NVIDIA runtime) mask parts of the container's `/proc`, so the kernel refuses the jail a fresh one | Before the trainer starts: `mkdir -p /run/fullproc && mount -t proc proc /run/fullproc` ([Sandboxes](environments/sandbox.md#choosing-a-backend)) |
| `the local sandbox waits on its programs through pidfd_open, which this environment refuses` | A kernel before 5.3 or the container's seccomp profile refuses `pidfd_open` | Run on Linux 5.3+ with `pidfd_open` allowed, or set `sandbox_backend: remote` with its `sandbox_url` ([Sandboxes](environments/sandbox.md#choosing-a-backend)) |
