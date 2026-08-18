"""Generate the pipeline-bubble diagram for gpu-training-theory.md §9.

A 1F1B schedule grid (stages x time slots) at p=4, m=8: every stage runs 2m
busy slots inside a step of 2(m + p - 1), so the idle share is the bubble
(p - 1) / (m + p - 1). The schedule is simulated from its dependencies rather
than hand-placed, so the figure follows the formula.
"""

import matplotlib.pyplot as plt
from _theory_style import *
from matplotlib.patches import Rectangle

P, M = 4, 8  # stages, microbatches; m = 2p, under the trainer's 4*pp warn threshold


def op_order(p, m):
    """Per-stage 1F1B order: `p-1-s` warmup forwards, then one forward per backward, then the cooldown."""
    order = []
    for s in range(p):
        warmup = p - 1 - s
        ops = [("F", k) for k in range(warmup)]
        for k in range(warmup, m):
            ops += [("F", k), ("B", k - warmup)]
        ops += [("B", k) for k in range(m - warmup, m)]
        order.append(ops)
    return order


def dependency(s, kind, k, p):
    """The op this one waits for: the stage above for a forward, the stage below for a backward."""
    if kind == "F":
        return (s - 1, "F", k) if s > 0 else None
    return (s + 1, "B", k) if s < p - 1 else (s, "F", k)


def simulate(p, m):
    """Place every op in the first slot where its stage is free and its dependency has finished."""
    order = op_order(p, m)
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
            raise RuntimeError("1F1B schedule deadlocked — dependency bug")
    return placed, max(free)


PLACED, SLOTS = simulate(P, M)
IDLE = SLOTS - 2 * M  # idle slots per stage, identical on every stage
if IDLE * (M + P - 1) != (P - 1) * SLOTS:
    raise RuntimeError("simulated bubble disagrees with (p-1)/(m+p-1)")

fig, ax = plt.subplots(figsize=(12.0, 4.9))
fig.patch.set_facecolor(BG)
ax.set_xlim(-1.75, SLOTS + 0.15)
ax.set_ylim(-1.95, P + 1.8)
ax.axis("off")

ax.text(-1.75, P + 1.66, "The pipeline bubble", fontsize=TITLE, fontweight="bold", va="top")
ax.text(
    -1.75,
    P + 0.92,
    f"1F1B · p = {P} stages · m = {M} microbatches · one slot = one microbatch forward or backward",
    fontsize=SUB,
    color=INK2,
    va="top",
)

ax.annotate(
    "",
    xy=(SLOTS, P + 0.12),
    xytext=(0, P + 0.12),
    arrowprops={"arrowstyle": "|-|, widthA=0.4, widthB=0.4", "color": INK2, "lw": 1.1},
)
ax.text(
    SLOTS / 2,
    P + 0.4,
    f"one optimizer step = {SLOTS} slots",
    ha="center",
    va="bottom",
    fontsize=SMALL,
    color=INK2,
    fontfamily=MONO,
)

# One row per stage, stage 0 on top; the rose ground shows through wherever a stage has no op.
for s in range(P):
    y = P - 1 - s
    ax.add_patch(Rectangle((0, y + 0.06), SLOTS, 0.88, facecolor=ROSE_T, edgecolor="none", zorder=1))
    ax.text(-0.3, y + 0.5, f"stage {s}", ha="right", va="center", fontsize=SMALL, color=INK2)

for s, kind, k, start in PLACED:
    y = P - 1 - s
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

# Fill and drain are the same p-1 slots, once at each end of the step.
for x0, label in ((0, "fill"), (SLOTS - (P - 1), "drain")):
    ax.add_patch(Rectangle((x0, -0.2), P - 1, 0.14, facecolor=ROSE, edgecolor="none", zorder=3))
    ax.text(x0 + (P - 1) / 2, -0.38, label, ha="center", va="top", fontsize=TINY, color=ROSE, fontweight="bold")
ax.text(
    SLOTS / 2,
    -0.38,
    "steady state — one forward and one backward per stage, per pair of slots",
    ha="center",
    va="top",
    fontsize=TINY,
    color=INK3,
)

lx = 0.0
for fill, edge, label in ((BLUE_T, BLUE, "forward"), (VIOLET_T, VIOLET, "backward"), (ROSE_T, ROSE_T, "idle")):
    ax.add_patch(Rectangle((lx, -1.3), 0.62, 0.46, facecolor=fill, edgecolor=edge, lw=1.0))
    ax.text(lx + 0.85, -1.07, label, ha="left", va="center", fontsize=SMALL, color=INK2)
    lx += 1.05 + len(label) * 0.3

ax.text(
    SLOTS,
    -1.07,
    f"idle = {IDLE} of {SLOTS} slots per stage  =  (p−1)/(m+p−1)  =  {IDLE / SLOTS:.0%}",
    ha="right",
    va="center",
    fontsize=LABEL,
    fontweight="bold",
    color=ROSE,
)
ax.text(
    -1.75,
    -1.72,
    "More microbatches stretch the steady state and shrink the fraction — and thin every GEMM's M (§2). "
    "More stages widen fill and drain, so the bubble grows.",
    fontsize=TINY,
    color=INK3,
    va="center",
)

plt.tight_layout()
save(plt.gcf(), "pp_bubble")
plt.close()
print("✓ pp_bubble.png")
