"""Generate the two MTP head figures for optimization/mtp-heads.md.

- mtp_acceptance.png: held-out acceptance while a head fits, per start, loss and learning rate. Color is the loss, the
  line style the rate, and a cross marks a run the gate stopped. The page's table holds every reading; the figure names
  no winner, as only `soft` fits have been served.
- mtp_step_cost.png: seconds per step and peak allocated memory of the arms `test_sft_mtp_head_cost.py` measured.
"""

import matplotlib.pyplot as plt
from _chart_style import *

STEPS = (0, 30, 60, 90, 120, 150)
HUES = {"hard": VIOLET, "soft": TEAL}
RATE_STYLES = {"1e-5": ("-", 2.2), "3.33e-6": ((0, (5, 2)), 1.7), "1e-6": ((0, (1, 1.5)), 1.7)}

# (loss, learning rate, acceptance at each of STEPS, stopped by the gate)
STALE = (
    ("soft", "1e-5", (0.7760, 0.8154, 0.8240, 0.8238, 0.8247, 0.8252), False),
    ("hard", "1e-5", (0.7760, 0.8009, 0.7985, 0.7896, 0.7845, 0.7643), True),
    ("hard", "3.33e-6", (0.7760, 0.7992, 0.8060, 0.8054, 0.8038, 0.7983), False),
    ("hard", "1e-6", (0.7760, 0.8051, 0.8056, 0.8071, 0.8072, 0.8070), False),
)
OWN = (
    ("soft", "1e-5", (0.8187, 0.8223, 0.8232, 0.8238, 0.8248, 0.8245), False),
    ("soft", "3.33e-6", (0.8187, 0.8210, 0.8230, 0.8255, 0.8251, 0.8260), False),
    ("soft", "1e-6", (0.8187, 0.8206, 0.8214, 0.8226, 0.8232, 0.8241), False),
    ("hard", "1e-5", (0.8187, 0.8015, 0.8005, 0.7905, 0.7869, 0.7638), True),
    ("hard", "3.33e-6", (0.8187, 0.8094, 0.8082, 0.8075, 0.8057, 0.8004), True),
    ("hard", "1e-6", (0.8187, 0.8172, 0.8129, 0.8119, 0.8109, 0.8106), True),
)
# Line-end labels per panel: (acceptance the label sits at, text, hue key).
STALE_ENDS = (
    (0.8252, "soft 1e-5", "soft"),
    (0.8075, "hard 1e-6", "hard"),
    (0.7983, "hard 3.33e-6", "hard"),
    (0.7643, "hard 1e-5", "hard"),
)
OWN_ENDS = (
    (0.8250, "soft, all three", "soft"),
    (0.8106, "hard 1e-6", "hard"),
    (0.8004, "hard 3.33e-6", "hard"),
    (0.7638, "hard 1e-5", "hard"),
)

# (arm, seconds per step, lower quartile, upper quartile, peak allocated GiB, step ratio label)
COST = (
    ("No head", 2.062, 1.962, 2.093, 85.8, None),
    ("joint, hard", 2.189, 2.126, 2.263, 86.2, "1.06×"),
    ("joint, soft", 2.209, 2.143, 2.260, 87.0, "1.07×"),
    ("head_only, soft", 0.762, 0.761, 0.763, 25.3, "0.37×"),
)


def fits_panel(ax, caption, fits, ends):
    for loss, rate, readings, stopped in fits:
        style, width = RATE_STYLES[rate]
        ax.plot(STEPS, readings, color=HUES[loss], ls=style, lw=width, zorder=3)
        if stopped:
            ax.plot(STEPS[-1], readings[-1], "x", ms=8, mew=2.0, color=HUES[loss], zorder=4)
    for level, text, loss in ends:
        ax.text(
            153 + 4,
            level,
            text,
            va="center",
            fontsize=TINY,
            color=INK2,
            fontweight="bold" if loss == "soft" else "normal",
        )
    ax.set_title(caption, loc="left", fontsize=SMALL, color=INK2, pad=6)
    ax.set_xticks(STEPS)
    ax.set_xlim(0, 153)
    ax.set_ylim(0.755, 0.832)
    ax.set_yticks((0.76, 0.78, 0.80, 0.82))
    ax.set_xlabel("step", fontsize=SMALL, color=INK2)
    tidy(ax, "y")


fig, (stale, own) = plt.subplots(1, 2, figsize=(11.0, 4.5), sharey=True)
fig.subplots_adjust(top=0.78, left=0.06, right=0.93, wspace=0.27, bottom=0.2)
heading(
    fig,
    "Held-out acceptance while a head fits",
    "Qwen3.6-35B at EP8 · 24 first-turn rows · a cross marks a run the gate stopped · only soft fits have been served",
)
fits_panel(stale, "Stale start: Qwen3.5-35B's head on the Qwen3.6 trunk", STALE, STALE_ENDS)
fits_panel(own, "Own start: the release's head", OWN, OWN_ENDS)
stale.set_ylabel("eval_mtp_acceptance_tv", fontsize=SMALL, color=INK2)
handles = [
    plt.Line2D([], [], color=INK2, ls=RATE_STYLES[rate][0], lw=RATE_STYLES[rate][1], label=f"learning_rate {rate}")
    for rate in RATE_STYLES
]
fig.legend(
    handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=TINY, handlelength=2.6, columnspacing=1.6
)

save(fig, "mtp_acceptance", BENCHMARKS)


fig, (seconds, memory) = plt.subplots(1, 2, figsize=(9.6, 2.9), sharey=True, gridspec_kw={"wspace": 0.06})
fig.subplots_adjust(top=0.68, left=0.145, right=0.985, bottom=0.2)
heading(
    fig,
    "What a head costs a step",
    "Qwen3.6-35B at EP8 on FA2 · packed random tokens, 32,992 per rank · weight 0.3 · median of 50 steps",
)
fig.texts[1].set_y(0.86)
ys = list(range(len(COST)))[::-1]
for y, (_, median, low, high, peak, ratio) in zip(ys, COST, strict=True):
    seconds.barh(y, median, height=0.58, color=BLUE, zorder=2)
    seconds.plot([low, high], [y, y], color=INK, lw=1.3, zorder=3, solid_capstyle="butt")
    if ratio:
        seconds.text(high + 0.05, y, ratio, va="center", fontsize=SMALL, fontweight="bold", color=INK)
    memory.barh(y, peak, height=0.58, color=AMBER, zorder=2)
seconds.set_yticks(ys)
seconds.set_yticklabels([row[0] for row in COST], fontsize=SMALL)
seconds.set_xlim(0, 2.7)
seconds.set_xlabel("seconds per step, line = quartiles", fontsize=SMALL, color=INK2)
tidy(seconds, "x")
memory.set_xlim(0, 100)
memory.set_xlabel("peak allocated GiB", fontsize=SMALL, color=INK2)
tidy(memory, "x")

save(fig, "mtp_step_cost", BENCHMARKS)
