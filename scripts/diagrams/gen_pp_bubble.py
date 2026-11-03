"""Generate the pipeline-bubble diagram for gpu-training-theory.md §9.

Both schedules as grids (stages x time slots) at p=4, m=8: every stage runs 2m busy slots inside a step of
2(m + p - 1), so the idle share is the bubble (p - 1) / (m + p - 1) under either. What differs is how many
microbatches a stage holds live, forward done and backward not. The schedules are simulated from torch's
per-stage op orders and their dependencies rather than hand-placed, so the figure follows the formula.
"""

import matplotlib.pyplot as plt
from _theory_style import *
from matplotlib.patches import Rectangle

P, M = 4, 8  # stages, microbatches; m = 2p, under the trainer's 4*pp warn threshold
SHIPPED_P, SHIPPED_M = 2, 8  # the two-node recipes
ROW_GAP = 2.05  # data units between the 1F1B grid's floor and the GPipe grid's top
LIVE_X = 1.15  # past the last slot, the centre of the live-microbatch column


def one_f_one_b(p, m):
    """Schedule1F1B's order per stage: `p-1-s` warmup forwards, one forward per backward, then the cooldown."""
    order = []
    for s in range(p):
        warmup = p - 1 - s
        ops = [("F", k) for k in range(warmup)]
        for k in range(warmup, m):
            ops += [("F", k), ("B", k - warmup)]
        ops += [("B", k) for k in range(m - warmup, m)]
        order.append(ops)
    return order


def gpipe(p, m):
    """ScheduleGPipe's order on every stage: all forwards, then all backwards."""
    return [[("F", k) for k in range(m)] + [("B", k) for k in range(m)] for _ in range(p)]


def dependency(s, kind, k, p):
    """The op this one waits for: the stage above for a forward, the stage below for a backward."""
    if kind == "F":
        return (s - 1, "F", k) if s > 0 else None
    return (s + 1, "B", k) if s < p - 1 else (s, "F", k)


def simulate(order, p):
    """Place every op in the first slot where its stage is free and its dependency has finished."""
    nxt, free, done, placed = [0] * p, [0] * p, {}, []
    while any(nxt[s] < len(order[s]) for s in range(p)):
        progressed = False
        for s in range(p):
            if nxt[s] >= len(order[s]):
                continue
            kind, k = order[s][nxt[s]]
            dep = dependency(s, kind, k, p)
            if dep is not None and dep not in done:
                continue
            start = max(free[s], done[dep] if dep is not None else 0)
            done[(s, kind, k)] = free[s] = start + 1
            placed.append((s, kind, k, start))
            nxt[s] += 1
            progressed = True
        if not progressed:
            raise RuntimeError("pipeline schedule deadlocked — dependency bug")
    return placed, max(free)


def most_live(ops):
    """The most microbatches one stage holds at once: forward run, backward not yet."""
    live = peak = 0
    for kind, _ in ops:
        live += 1 if kind == "F" else -1
        peak = max(peak, live)
    return peak


def draw_grid(ax, floor, order, placed, slots):
    """One schedule, stage 0 on top; the rose ground shows through wherever a stage has no op."""
    for s in range(P):
        y = floor + P - 1 - s
        ax.add_patch(Rectangle((0, y + 0.06), slots, 0.88, facecolor=ROSE_T, edgecolor="none", zorder=1))
        ax.text(-0.3, y + 0.5, f"stage {s}", ha="right", va="center", fontsize=SMALL, color=INK2)
        ax.text(
            slots + LIVE_X,
            y + 0.5,
            str(most_live(order[s])),
            ha="center",
            va="center",
            fontsize=LABEL,
            fontweight="bold",
            color=AMBER,
            fontfamily=MONO,
        )
    for s, kind, k, start in placed:
        y = floor + P - 1 - s
        fill, edge = (BLUE_T, BLUE) if kind == "F" else (VIOLET_T, VIOLET)
        ax.add_patch(Rectangle((start, y + 0.06), 1, 0.88, facecolor=fill, edgecolor=edge, lw=1.0, zorder=2))
        ax.text(
            start + 0.5,
            y + 0.5,
            f"{kind}{k + 1}",
            ha="center",
            va="center",
            fontsize=TINY,
            color=INK,
            fontfamily=MONO,
            zorder=3,
        )


ORDER_1F1B, ORDER_GPIPE = one_f_one_b(P, M), gpipe(P, M)
PLACED_1F1B, SLOTS = simulate(ORDER_1F1B, P)
PLACED_GPIPE, SLOTS_GPIPE = simulate(ORDER_GPIPE, P)
IDLE = SLOTS - 2 * M  # idle slots per stage, identical on every stage
if SLOTS_GPIPE != SLOTS or IDLE * (M + P - 1) != (P - 1) * SLOTS:
    raise RuntimeError("simulated bubble disagrees with (p-1)/(m+p-1)")

FLOOR_GPIPE = 0
FLOOR_1F1B = P + ROW_GAP
TOP = FLOOR_1F1B + P

fig, ax = plt.subplots(figsize=(12.0, 7.7))
ax.set_xlim(-1.75, SLOTS + LIVE_X + 0.95)
ax.set_ylim(-1.95, TOP + 2.1)
ax.axis("off")

ax.text(-1.75, TOP + 2.0, "The pipeline bubble", fontsize=TITLE, fontweight="bold", va="top")
ax.text(
    -1.75,
    TOP + 1.28,
    f"p = {P} stages · m = {M} microbatches · one slot = one microbatch forward or backward",
    fontsize=SUB,
    color=INK2,
    va="top",
)

ax.text(-1.75, TOP + 0.5, "1F1B (default)", fontsize=SECTION, fontweight="bold", va="center")
ax.annotate(
    "",
    xy=(SLOTS, TOP + 0.12),
    xytext=(0, TOP + 0.12),
    arrowprops={"arrowstyle": "|-|, widthA=0.4, widthB=0.4", "color": INK2, "lw": 1.1},
)
ax.text(
    SLOTS / 2,
    TOP + 0.36,
    f"one optimizer step = {SLOTS} slots",
    ha="center",
    va="bottom",
    fontsize=SMALL,
    color=INK2,
    fontfamily=MONO,
)
ax.text(
    SLOTS + LIVE_X,
    TOP + 0.12,
    "live\nmicrobatches",
    ha="center",
    va="bottom",
    fontsize=TINY,
    color=AMBER,
    fontweight="bold",
)

draw_grid(ax, FLOOR_1F1B, ORDER_1F1B, PLACED_1F1B, SLOTS)

# Fill and drain are the same p-1 slots, once at each end of the step.
for x0, label in ((0, "fill"), (SLOTS - (P - 1), "drain")):
    ax.add_patch(Rectangle((x0, FLOOR_1F1B - 0.2), P - 1, 0.14, facecolor=ROSE, edgecolor="none", zorder=3))
    ax.text(
        x0 + (P - 1) / 2, FLOOR_1F1B - 0.38, label, ha="center", va="top", fontsize=TINY, color=ROSE, fontweight="bold"
    )
ax.text(
    SLOTS / 2,
    FLOOR_1F1B - 0.38,
    "steady state — one forward and one backward per stage, per pair of slots",
    ha="center",
    va="top",
    fontsize=TINY,
    color=INK3,
)

ax.text(-1.75, P + 0.5, "GPipe", fontsize=SECTION, fontweight="bold", va="center")
ax.text(
    SLOTS / 2,
    P + 0.36,
    f"all forwards, then all backwards — the same {SLOTS} slots",
    ha="center",
    va="bottom",
    fontsize=SMALL,
    color=INK2,
    fontfamily=MONO,
)

draw_grid(ax, FLOOR_GPIPE, ORDER_GPIPE, PLACED_GPIPE, SLOTS)

lx = 0.0
for fill, edge, label in ((BLUE_T, BLUE, "forward"), (VIOLET_T, VIOLET, "backward"), (ROSE_T, ROSE_T, "idle")):
    ax.add_patch(Rectangle((lx, -0.95), 0.62, 0.46, facecolor=fill, edgecolor=edge, lw=1.0))
    ax.text(lx + 0.85, -0.72, label, ha="left", va="center", fontsize=SMALL, color=INK2)
    lx += 1.05 + len(label) * 0.3

ax.text(
    SLOTS,
    -0.72,
    f"idle = {IDLE} of {SLOTS} slots per stage  =  (p−1)/(m+p−1)  =  {IDLE / SLOTS:.0%}",
    ha="right",
    va="center",
    fontsize=LABEL,
    fontweight="bold",
    color=ROSE,
)
ax.text(
    -1.75,
    -1.55,
    "More microbatches shrink the fraction and thin every GEMM's M (§4); more stages widen fill and drain. "
    f"The shipped p = {SHIPPED_P}, m = {SHIPPED_M} recipes idle {SHIPPED_P - 1}/{SHIPPED_M + SHIPPED_P - 1} of a step.",
    fontsize=TINY,
    color=INK3,
    va="center",
)

plt.tight_layout()
save(plt.gcf(), "pp_bubble")
