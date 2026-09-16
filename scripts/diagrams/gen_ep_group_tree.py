"""Generate the EP group-hierarchy and multi-node EP+TP figures.

`ep_group_hierarchy` — the world splits into EP dispatch groups; each rank owns
`num_experts / ep_size` experts and still reads its own batch, and ranks holding the same expert
slice are DP replicas averaged after the backward.

`ep_multi_node_layout` — the only EP+TP shape `ParallelismConfig` accepts across NVLink domains:
node-local TP groups under one global EP group spanning the job.
"""

import matplotlib.pyplot as plt
from _pipeline_style import *

# The shapes the two figures depict, replayed through ParallelismConfig by
# tests/cpu/parallelism/test_shipped_diagram_topology.py so no figure shows a rejected topology.
TOPOLOGIES = {
    "ep_group_hierarchy": {"nodes": 1, "gpus": 4, "ep": 2, "tp": 1, "scope": "node", "dp": 4},
    "ep_multi_node_layout": {"nodes": 2, "gpus": 8, "ep": 16, "tp": 8, "scope": "global", "dp": 2},
}

# ── EP group hierarchy ────────────────────────────────────────────────────────

W, H = 11.6, 5.75

WORLD_Y, WORLD_H = 4.80, 0.55
GROUP_Y, GROUP_H, GROUP_W, GROUP_XS = 2.30, 2.00, 5.4, (0.3, 5.9)
GPU_Y, GPU_W, GPU_DXS = 2.65, 2.35, (0.25, 2.80)
LINK_YS = (1.90, 1.55)

GROUPS = [
    ("EP group 0 — ranks 0, 1", [("rank 0", "experts 0–15", "batch b0"), ("rank 1", "experts 16–31", "batch b1")]),
    ("EP group 1 — ranks 2, 3", [("rank 2", "experts 0–15", "batch b2"), ("rank 3", "experts 16–31", "batch b3")]),
]


def gpu_center(group_idx, gpu_idx):
    """Center x of one rank card — the anchor the replica links join."""
    return GROUP_XS[group_idx] + GPU_DXS[gpu_idx] + GPU_W / 2


fig, ax = plt.subplots(figsize=(W, H))
fig.patch.set_facecolor(BG)
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

T = TOPOLOGIES["ep_group_hierarchy"]
title(ax, "EP groups", f"world {T['gpus'] * T['nodes']} · ep {T['ep']} · 32 experts → 16 per rank · dp {T['dp']}")

world = FancyBboxPatch(
    (0.3, WORLD_Y), 11.0, WORLD_H, boxstyle="round,pad=0.03", facecolor=tint(SLATE), edgecolor=SLATE, lw=1.3
)
ax.add_patch(world)
ax.text(
    5.8,
    WORLD_Y + WORLD_H / 2,
    "WORLD — 4 ranks, one NVLink domain",
    ha="center",
    va="center",
    fontsize=LABEL,
    fontweight="bold",
    color=INK,
)

for gi, (group_head, ranks) in enumerate(GROUPS):
    gx = GROUP_XS[gi]
    frame(ax, gx, GROUP_Y, GROUP_W, GROUP_H, color=VIOLET)
    ax.text(gx + 0.25, GROUP_Y + GROUP_H - 0.12, group_head, ha="left", va="top", fontsize=LABEL, fontweight="bold")
    ax.text(
        gx + GROUP_W / 2,
        GROUP_Y + 0.22,
        "DeepEP all-to-all inside the group",
        ha="center",
        va="center",
        fontsize=SMALL,
        color=VIOLET,
    )
    for ri, (rank_head, experts, batch) in enumerate(ranks):
        card(
            ax,
            gx + GPU_DXS[ri],
            GPU_Y,
            GPU_W,
            card_height(2),
            rank_head,
            [experts, batch],
            color=TEAL,
            mono_lines=True,
        )
    arrow(ax, gx + GROUP_W / 2, WORLD_Y, gx + GROUP_W / 2, GROUP_Y + GROUP_H)

replica_link(ax, gpu_center(0, 0), gpu_center(1, 0), LINK_YS[0])
replica_link(ax, gpu_center(0, 1), gpu_center(1, 1), LINK_YS[1])
ax.text(
    5.8,
    LINK_YS[1] - 0.24,
    "DP replicas — the ranks holding one expert slice; their expert grads are averaged after the backward",
    ha="center",
    va="top",
    fontsize=SMALL,
    color=INK2,
)

footnote(
    ax,
    0.3,
    0.3,
    11.0,
    "One dispatch group per NVLink domain: ep_size > 2 with ep_group_size below the domain"
    " (ep4 on 8) is rejected at config time.",
)

save(fig, "ep_group_hierarchy")
plt.close(fig)
print("✓ ep_group_hierarchy.png")


# ── Multi-node EP + TP ────────────────────────────────────────────────────────

MW, MH = 13.0, 6.8

NODE_Y, NODE_H, NODE_W, NODE_XS = 4.35, 2.05, 6.05, (0.3, 6.65)
CELL_Y, CELL_H, CELL_GAP = NODE_Y + 1.02, 0.5, 0.08
CELL_W = (NODE_W - 0.5 - 7 * CELL_GAP) / 8
TP_Y, TP_H = NODE_Y + 0.22, card_height(1, has_title=False)
EP_Y, EP_H = 2.55, card_height(3)
DP_Y, DP_H = 1.10, card_height(2)
BAND_W = 12.4

fig, ax = plt.subplots(figsize=(MW, MH))
fig.patch.set_facecolor(BG)
ax.set_xlim(0, MW)
ax.set_ylim(0, MH)
ax.axis("off")

M = TOPOLOGIES["ep_multi_node_layout"]
title(
    ax,
    "EP + TP across two nodes",
    f"{M['nodes']} × {M['gpus']} · tp {M['tp']} node-local · ep {M['ep']} {M['scope']} · dp {M['dp']}",
)

for ni, (nx, head) in enumerate(zip(NODE_XS, ("Node 0 — ranks 0–7", "Node 1 — ranks 8–15"), strict=True)):
    frame(ax, nx, NODE_Y, NODE_W, NODE_H, dashed=False)
    ax.text(nx + 0.25, NODE_Y + NODE_H - 0.12, head, ha="left", va="top", fontsize=LABEL, fontweight="bold")
    for i in range(8):
        cx = nx + 0.25 + i * (CELL_W + CELL_GAP)
        chip(ax, cx, CELL_Y, CELL_W, CELL_H, f"r{8 * ni + i}", color=TEAL, fontsize=TINY, mono=True)
    card(
        ax,
        nx + 0.25,
        TP_Y,
        NODE_W - 0.5,
        TP_H,
        "",
        ["TP group, 8 ranks — attention sharded as DTensor over NVLink"],
        color=BLUE,
    )
    arrow(ax, nx + NODE_W / 2, NODE_Y, nx + NODE_W / 2, EP_Y + EP_H, "tokens", side="right")

card(
    ax,
    0.3,
    EP_Y,
    BAND_W,
    EP_H,
    "One global EP group — all 16 ranks (ep_scope='global')",
    [
        "128 experts / ep 16 = 8 per rank",
        "rank r owns experts 8r … 8r+7",
        "DeepEP all-to-all crosses RDMA between the nodes",
    ],
    color=VIOLET,
    mono_lines=True,
)

card(
    ax,
    0.3,
    DP_Y,
    BAND_W,
    DP_H,
    "FSDP2 mesh (dp 2, tp 8)",
    [
        "ranks sharing a TP position form a DP pair: (0,8) (1,9) … (7,15)",
        "non-expert params shard over the pair: RDMA all-gather / reduce-scatter",
    ],
    color=SLATE,
    mono_lines=True,
    dashed=True,
)

footnote(
    ax,
    0.3,
    0.3,
    BAND_W,
    "Above one NVLink domain, EP under TP must be a single group spanning the job —"
    " ep8/tp2 on 2×8 forms two groups and is rejected at config time.",
)

save(fig, "ep_multi_node_layout")
plt.close(fig)
print("✓ ep_multi_node_layout.png")
