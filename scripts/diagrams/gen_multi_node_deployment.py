"""Generate the two multi-node topologies for training-methods/grpo/async-grpo/setup.md.

`multi_node_separate_inference.png` — trainer and Ray actors on one node, the engine on another,
with the ports and env knobs the cross-node paths need.
`multi_node_dedicated_rollout.png` — one training node, one inference node per
`rollout_server_configs` entry, a GPU-less actor tier, and the config that wires them together.
"""

import matplotlib.pyplot as plt
from _pipeline_style import *

# ── Scenario 1: one training node, one inference node ────────────────────────────────────────
STRIP_Y, STRIP_H = 0.28, card_height(4)
FRAME_Y = STRIP_Y + STRIP_H + 0.54
FRAME_H = 4.35
FRAME_TOP = FRAME_Y + FRAME_H
W1, H1 = 13.2, FRAME_TOP + 0.95
M1 = 0.33

N1_X, N1_W = M1, 6.2
N2_X = N1_X + N1_W + 1.3
N2_W = W1 - M1 - N2_X
N2_H = FRAME_LABEL + card_height(6) + FRAME_PAD
N2_Y = FRAME_Y + (FRAME_H - N2_H) / 2
CARD_X, CARD_W = N1_X + FRAME_PAD, N1_W - 2 * FRAME_PAD
ENGINE_X, ENGINE_W = N2_X + FRAME_PAD, N2_W - 2 * FRAME_PAD

fig, ax = plt.subplots(figsize=(W1, H1))
ax.set_xlim(0, W1)
ax.set_ylim(0, H1)
ax.axis("off")

title(ax, "Separate inference node", "one server · one NCCL group · actors beside the trainer")

frame(ax, N1_X, FRAME_Y, N1_W, FRAME_H, "NODE 1 — training + actors")
frame(ax, N2_X, N2_Y, N2_W, N2_H, "NODE 2 — inference")

trainer_y = FRAME_TOP - FRAME_LABEL - card_height(4)
card(
    ax,
    CARD_X,
    trainer_y,
    CARD_W,
    card_height(4),
    "Trainer — torchrun ranks",
    [
        "model + optimizer, FSDP2 / EP / TP",
        "InferenceClientManager on the main process",
        "rank 0 binds the sync store, one per server",
        "a push pauses the server, then resumes it",
    ],
    color=BLUE,
)

actors_y = trainer_y - 0.30 - card_height(4)
card(
    ax,
    CARD_X,
    actors_y,
    CARD_W,
    card_height(4),
    "Ray actors — same node, CPU",
    [
        "num_rollout_workers actors per training rank",
        "each holds the environment: tools, sandbox, grader",
        "no tokenizer — the engine renders the template",
        "max_concurrent_rollouts bounds in-flight episodes",
    ],
    color=SLATE,
)

engine_y = N2_Y + FRAME_PAD
card(
    ax,
    ENGINE_X,
    engine_y,
    ENGINE_W,
    card_height(6),
    "Rollout server",
    [
        "vLLM :8000 · SGLang :30000",
        "rollout_backend picks the client",
        "its workers join the sync group",
        "and dial the trainer's store",
        "--moe-backend triton (vLLM)",
        "GPUs no trainer rank uses",
    ],
    color=BLUE,
)

arrow(ax, N1_X + N1_W, trainer_y + 0.85, N2_X, engine_y + 1.64, "NCCL :51216", color=VIOLET)
arrow(ax, N1_X + N1_W, actors_y + 0.85, N2_X, engine_y + 0.49, "HTTP :8000", color=TEAL)
arrow(
    ax,
    CARD_X + 0.5 * CARD_W,
    trainer_y,
    CARD_X + 0.5 * CARD_W,
    actors_y + card_height(4),
    "Ray — local, ray_address null",
    color=SLATE,
    side="right",
)

section(ax, M1, FRAME_Y - 0.18, "NETWORK")

STRIP = [
    (
        "HTTP",
        [
            "vLLM :8000 · SGLang :30000",
            "actors → engine, per turn",
            "round-robin over the pool",
            "resolves from every node",
        ],
    ),
    (
        "NCCL group",
        ["vllm_group_port 51216", "bound on the trainer host", "+1 per extra server", "VLLM_GROUP_HOST, multi-homed"],
    ),
    (
        "Ray",
        [
            "ray_address: null — local",
            "each rank's ray.init starts Ray",
            "dashboard off — no port 8265",
            "actors inherit the trainer's env",
        ],
    ),
    (
        "EFA",
        [
            "docker-compose.vllm.efa.yml",
            "+ make EFA=1 on the trainer",
            "else the sync runs on sockets",
            "weight_sync_transport.py",
        ],
    ),
]
CELL_GAP = 0.15
cell_xs, cell_w = columns(W1, len(STRIP), M1, CELL_GAP)
for x, (name, lines) in zip(cell_xs, STRIP, strict=True):
    card(ax, x, STRIP_Y, cell_w, STRIP_H, name, lines, color=SLATE)

save(plt.gcf(), "multi_node_separate_inference")
plt.close()
print("✓ multi_node_separate_inference.png")


# ── Scenario 2: dedicated rollout nodes ──────────────────────────────────────────────────────
W2, H2 = 13.0, 7.07
M2 = 0.325

L_X, L_W = M2, 5.6
R_X = L_X + L_W + 1.3
R_W = W2 - M2 - R_X
TOP2 = H2 - 0.95
L_CARD_X, L_CARD_W = L_X + FRAME_PAD, L_W - 2 * FRAME_PAD
R_CARD_X, R_CARD_W = R_X + FRAME_PAD, R_W - 2 * FRAME_PAD

fig, ax = plt.subplots(figsize=(W2, H2))
ax.set_xlim(0, W2)
ax.set_ylim(0, H2)
ax.axis("off")

title(ax, "Dedicated rollout nodes", "rollout_server_configs: one entry per inference node")

frame(ax, L_X, TOP2 - 2.61, L_W, 2.61, "TRAINING NODE")
frame(ax, R_X, TOP2 - 3.31, R_W, 3.31, "INFERENCE NODES")
frame(ax, L_X, TOP2 - 5.82, L_W, 2.61, "ACTOR TIER — GPU-less CPU nodes")

card(
    ax,
    L_CARD_X,
    TOP2 - 2.41,
    L_CARD_W,
    card_height(5),
    "Trainer — torchrun ranks",
    [
        "model + optimizer, FSDP2 / EP / TP",
        "InferenceClientManager on the main process",
        "one NCCL group per server, trainer is rank 0",
        "each group port is bound on this host",
        "a push pauses every server, then resumes",
    ],
    color=BLUE,
)

card(
    ax,
    R_CARD_X,
    TOP2 - 1.63,
    R_CARD_W,
    card_height(2),
    "Inference node 1 — vLLM or SGLang",
    ["url        http://inf1:8000", "group_port 51216"],
    color=BLUE,
    mono_lines=True,
)

card(
    ax,
    R_CARD_X,
    TOP2 - 3.11,
    R_CARD_W,
    card_height(2),
    "Inference node 2 — vLLM or SGLang",
    ["url        http://inf2:8000", "group_port 51217"],
    color=BLUE,
    mono_lines=True,
)

card(
    ax,
    L_CARD_X,
    TOP2 - 5.62,
    L_CARD_W,
    card_height(5),
    "Ray actors — no GPU needed",
    [
        "num_rollout_workers ÷ world_size per rank",
        "soft-pinned to the rank's node, then spills",
        "each holds the environment and its tools",
        "POST /v1/chat/completions, round-robin",
        "inherit the ray start env, not the trainer's",
    ],
    color=SLATE,
)

card(
    ax,
    R_X,
    TOP2 - 5.82,
    R_W,
    card_height(6),
    "Config",
    [
        'ray_address: "ray-head:6379"',
        "num_rollout_workers: 64",
        "rollout_server_configs:",
        '  - {url: "http://inf1:8000", group_port: 51216}',
        '  - {url: "http://inf2:8000", group_port: 51217}',
        "enable_prefetch: true",
    ],
    color=SLATE,
    mono_lines=True,
)

arrow(ax, L_X + L_W, TOP2 - 1.15, R_X, TOP2 - 1.04, "NCCL 51216", color=VIOLET)
arrow(ax, L_X + L_W, TOP2 - 2.30, R_X, TOP2 - 2.52, "NCCL 51217", color=VIOLET)
arrow(
    ax,
    L_X + 0.5 * L_W,
    TOP2 - 2.61,
    L_X + 0.5 * L_W,
    TOP2 - 3.21,
    "Ray · ray_address: ray-head:6379",
    color=SLATE,
    side="right",
)
arrow(ax, L_X + L_W, TOP2 - 3.26, R_X, TOP2 - 3.26, "HTTP per turn", color=TEAL)

save(plt.gcf(), "multi_node_dedicated_rollout")
plt.close()
print("✓ multi_node_dedicated_rollout.png")
