"""Generate the two async-GRPO server-topology figures.

- environmental_grpo_single_server.png — the compose default: one engine, prefetch off, a serial step.
- environmental_grpo_multi_server.png — the code-contests shape: two engines, prefetch on.

Both figures use one layout so the modes can be compared square on.
"""

import matplotlib.pyplot as plt
from _pipeline_style import *

W, H = 10.0, 7.15
LX, LW = 0.3, 3.45  # left column
RX, RW = 5.85, 3.85  # right column
BRACKET_X = 5.70  # where the multi-server arrows land
TOP_Y, MID_Y, BOT_Y = 5.27, 3.83, 2.51  # tops of the two card bands, and the lower band's floor
NCCL_Y, HTTP_Y = 4.55, 3.10

BAR_X0, BAR_X1, BAR_H = 1.25, 9.70, 0.38
TRAINER_BAR_Y, ENGINE_BAR_Y = 1.58, 1.10
SYNC, ROUND, UPDATE = 0.65, 4.90, 2.60  # one step, drawn to scale in bar inches


def draw(mode):
    """Render one of the two topologies; `mode` is "single" or "multi"."""
    multi = mode == "multi"
    n_trainer = 6 if multi else 7

    fig, ax = plt.subplots(figsize=(W, H))
    fig.patch.set_facecolor(BG)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    title(
        ax,
        "Two rollout servers" if multi else "One rollout server",
        sub_mono=(
            "rollout_server_configs: 2 · enable_prefetch: true"
            if multi
            else "rollout_server_url · enable_prefetch → off"
        ),
    )

    # --- GPUs on the host ---
    section(ax, LX, 6.55, "GPU SPLIT")
    cw, cgap = 0.58, 0.08
    for g in range(8):
        chip(ax, LX + g * (cw + cgap), 5.86, cw, 0.42, str(g), color=BLUE if g < n_trainer else VIOLET, bold=True)
    for lx, color, text in ((5.90, BLUE, "trainer rank"), (7.75, VIOLET, "rollout engine")):
        chip(ax, lx, 5.93, 0.28, 0.28, color=color)
        ax.text(lx + 0.40, 6.07, text, ha="left", va="center", fontsize=SMALL, color=INK2)

    # --- Who talks to whom ---
    section(ax, LX, 5.64, "TOPOLOGY")
    card(
        ax,
        LX,
        MID_Y,
        LW,
        card_height(3),
        f"Trainer — GPU 0–{n_trainer - 1}",
        [
            f"torchrun --nproc_per_node={n_trainer}",
            "rank 0 of both sync groups" if multi else "rank 0 of the sync group",
            "binds :51216 and :51217" if multi else "binds the store on :51216",
        ],
        color=BLUE,
    )
    card(
        ax,
        LX,
        BOT_Y,
        LW,
        card_height(2),
        "Ray environment actors",
        ["num_rollout_workers per rank", "round-robin over the server URLs"],
        color=TEAL,
    )

    if multi:
        card(
            ax,
            RX,
            TOP_Y - card_height(2),
            RW,
            card_height(2),
            "Server 1 — GPU 6",
            [":8000 · NCCL group port 51216", "ranks 1+; dials the trainer host"],
            color=VIOLET,
        )
        card(
            ax,
            RX,
            BOT_Y,
            RW,
            card_height(2),
            "Server 2 — GPU 7",
            [":8001 · NCCL group port 51217", "group port = vllm_group_port + index"],
            color=VIOLET,
        )
        ax.annotate(
            "",
            xy=(BRACKET_X, TOP_Y),
            xytext=(BRACKET_X, BOT_Y),
            arrowprops={"arrowstyle": "|-|, widthA=0.5, widthB=0.5", "color": VIOLET, "lw": 1.4},
        )
        target = BRACKET_X
    else:
        card(
            ax,
            RX,
            TOP_Y - card_height(8),
            RW,
            card_height(8),
            "Rollout server — GPU 7",
            [
                "vLLM :8000 · SGLang :30000 · TP 1",
                "serves POST /v1/chat/completions",
                "NCCL ranks 1+; the trainer is rank 0",
                "one group; store :51216 on the trainer",
                "for the push: POST /pause?mode=keep",
                "in-flight requests resume after it",
                "nothing is served until /resume",
                "one engine: nothing to prefetch against",
            ],
            color=VIOLET,
        )
        target = RX

    arrow(ax, LX + LW, NCCL_Y, target, NCCL_Y, "NCCL weight sync", color=VIOLET)
    arrow(ax, LX + LW, HTTP_Y, target, HTTP_Y, "POST\n/v1/chat/completions", color=TEAL, side="below")

    # --- One step, to scale ---
    section(ax, LX, 2.28, "ONE STEP ON THE CLOCK")
    for y, text in ((TRAINER_BAR_Y, "trainer"), (ENGINE_BAR_Y, "engines" if multi else "engine")):
        ax.text(BAR_X0 - 0.12, y + BAR_H / 2, text, ha="right", va="center", fontsize=SMALL, color=INK2)

    chip(ax, BAR_X0, TRAINER_BAR_Y, SYNC, BAR_H, "sync", color=VIOLET)
    chip(ax, BAR_X0, ENGINE_BAR_Y, SYNC, BAR_H, "paused", color=VIOLET)
    if multi:
        chip(ax, BAR_X0 + SYNC, TRAINER_BAR_Y, UPDATE, BAR_H, "update on the round before", color=BLUE)
        chip(ax, BAR_X0 + SYNC + UPDATE, TRAINER_BAR_Y, ROUND - UPDATE, BAR_H, "waiting", color=ROSE)
        chip(ax, BAR_X0 + SYNC, ENGINE_BAR_Y, ROUND, BAR_H, "generating this round", color=TEAL)
        chip(ax, BAR_X0 + SYNC + ROUND, ENGINE_BAR_Y, UPDATE, BAR_H, "what prefetch removes", color=ROSE, dashed=True)
        formula = "step = sync + max(round, update)"
    else:
        chip(ax, BAR_X0 + SYNC, TRAINER_BAR_Y, ROUND, BAR_H, "waiting for the round", color=ROSE)
        chip(ax, BAR_X0 + SYNC + ROUND, TRAINER_BAR_Y, UPDATE, BAR_H, "update", color=BLUE)
        chip(ax, BAR_X0 + SYNC, ENGINE_BAR_Y, ROUND, BAR_H, "generating this round", color=TEAL)
        chip(ax, BAR_X0 + SYNC + ROUND, ENGINE_BAR_Y, UPDATE, BAR_H, "idle", color=ROSE)
        formula = "step = sync + round + update"
    ax.text(BAR_X0, 0.93, formula, fontsize=SMALL, fontweight="bold", color=INK, va="top", fontfamily=MONO)

    footnote(
        ax,
        LX,
        0.20,
        BAR_X1 - LX,
        (
            "The streamed sync pauses both servers for the whole push; prefetch then keeps the pipeline one round deep."
            if multi
            else "One server: the sync pauses the only engine, and the round cannot overlap the update — prefetch is off."
        ),
    )

    name = f"environmental_grpo_{mode}_server"
    save(fig, name)
    plt.close(fig)
    print(f"✓ {name}.png")


draw("single")
draw("multi")
