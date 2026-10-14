import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
})
suites = ["LIBERO\n10", "LIBERO\nGoal", "LIBERO\nSpatial", "LIBERO\nObject"]
full = np.array([87.2, 97.4, 97.8, 98.8])
few  = np.array([58.0, 63.2, 74.8, 90.6])
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdcd8"

fig, ax = plt.subplots(figsize=(3.3, 2.3))
x = np.arange(len(suites)); w = 0.30; gap = 0.04
xb = x - w/2 - gap/2; xo = x + w/2 + gap/2
ax.bar(xb, full, w, color=BLUE, label="All demonstrations (${\\sim}$40 / task)", zorder=3, linewidth=0)
ax.bar(xo, few,  w, color=ORANGE, label="3 demonstrations / task", zorder=3, linewidth=0)
for xi, v in zip(xb, full):
    ax.text(xi, v + 1.2, f"{v:.1f}", ha="center", va="bottom", fontsize=7, color=INK)
for xi, v in zip(xo, few):
    ax.text(xi, v + 1.2, f"{v:.1f}", ha="center", va="bottom", fontsize=7, color=INK)
# drop annotations: dashed reference line at the full-data level, delta above it over the few-shot bar,
# arrow down to the few-shot value when there is room
for xbi, xoi, f, s in zip(xb, xo, full, few):
    ax.plot([xbi, xoi + w/2 + 0.02], [f, f], color=INK2, lw=0.5, ls=(0, (2, 1.5)), zorder=4)
    ax.text(xoi, f + 1.2, f"$-${f - s:.1f}", ha="center", va="bottom", fontsize=6.5, color=INK2)
    if f - s > 12:
        ax.annotate("", xy=(xoi, s + 9.0), xytext=(xoi, f - 0.8),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.6, mutation_scale=5, shrinkA=0, shrinkB=0), zorder=4)
ax.set_xticks(x); ax.set_xticklabels(suites)
ax.set_ylim(0, 112); ax.set_yticks([0, 25, 50, 75, 100])
ax.set_ylabel("Success rate (%)", color=INK)
ax.yaxis.grid(True, color=GRID, linewidth=0.5, zorder=0); ax.set_axisbelow(True)
for s_ in ["top", "right"]: ax.spines[s_].set_visible(False)
for s_ in ["left", "bottom"]: ax.spines[s_].set_color(GRID); ax.spines[s_].set_linewidth(0.6)
ax.tick_params(axis="both", colors=INK2, length=2, width=0.5)
for lab in ax.get_xticklabels(): lab.set_color(INK)
ax.set_xlim(-0.55, len(suites) - 0.35)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, ncol=2, handlelength=1.0, handletextpad=0.5, columnspacing=1.2, borderaxespad=0)
fig.tight_layout(pad=0.3)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig1_data_scarcity")
fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=220)
print("saved", out)
