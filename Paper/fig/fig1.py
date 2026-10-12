import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

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
few  = np.array([58.0, 48.8, 72.0, 82.8])
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdcd8"

fig, ax = plt.subplots(figsize=(3.3, 2.3))
x = np.arange(len(suites)); w = 0.30; gap = 0.04
b1 = ax.bar(x - w/2 - gap/2, full, w, color=BLUE, label="All demonstrations (${\\sim}$40 / task)", zorder=3, linewidth=0)
b2 = ax.bar(x + w/2 + gap/2, few,  w, color=ORANGE, label="3 demonstrations / task", zorder=3, linewidth=0)
for xi, v in zip(x - w/2 - gap/2, full):
    ax.text(xi, v + 1.2, f"{v:.1f}", ha="center", va="bottom", fontsize=7, color=INK)
for xi, v in zip(x + w/2 + gap/2, few):
    ax.text(xi, v + 1.2, f"{v:.1f}", ha="center", va="bottom", fontsize=7, color=INK)
# drop annotations: dashed reference line at the full-data level over the few-shot bar, delta text in the gap
for xi, f, s_ in zip(x, full, few):
    x0 = xi - w/2 - gap/2 + w/2; x1 = xi + w/2 + gap/2 + w/2 + 0.02
    ax.plot([x0, x1], [f, f], color=INK2, lw=0.5, ls=(0, (2, 1.5)), zorder=4)
    ax.annotate("", xy=(xi + w/2 + gap/2, s_ + 5.5), xytext=(xi + w/2 + gap/2, f - 0.8),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.6, mutation_scale=5, shrinkA=0, shrinkB=0), zorder=4)
    ax.text(xi + w/2 + gap/2 + 0.06, (f + s_ + 5.5)/2, f"$-${f - s_:.1f}", ha="left", va="center", fontsize=6.3, color=INK2)
ax.set_xticks(x); ax.set_xticklabels(suites)
ax.set_ylim(0, 108); ax.set_yticks([0, 25, 50, 75, 100])
ax.set_ylabel("Success rate (%)", color=INK)
ax.yaxis.grid(True, color=GRID, linewidth=0.5, zorder=0); ax.set_axisbelow(True)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
for s in ["left", "bottom"]: ax.spines[s].set_color(GRID); ax.spines[s].set_linewidth(0.6)
ax.tick_params(axis="both", colors=INK2, length=2, width=0.5)
for lab in ax.get_xticklabels(): lab.set_color(INK)
ax.set_xlim(-0.55, len(suites) - 0.35)
leg = ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, ncol=2, handlelength=1.0, handletextpad=0.5, columnspacing=1.2, borderaxespad=0)
fig.tight_layout(pad=0.3)
import os; out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig1_data_scarcity")
fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=220)
print("saved", out)
