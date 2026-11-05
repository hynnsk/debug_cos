
import numpy as np
import matplotlib.pyplot as plt

# Font settings
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 11

# Data
data = {
    "Long":    [93.0, 58.4, 75.0],
    "Goal":    [97.4, 63.2, 75.0],
    "Spatial": [97.8, 74.8, 81.2],
    "Object":  [98.8, 80.2, 90.4],
}

colors = ["#E5E5E5", "#999999", "#404040"]

# Plot
fig, axes = plt.subplots(
    1, 4,
    figsize=(12, 2.2),
    sharex=True,
    gridspec_kw={"wspace": 0.08}
)

for ax, (task, values) in zip(axes, data.items()):
    y = np.arange(len(values))

    ax.barh(
        y,
        np.array(values) - 50,
        left=50,
        color=colors,
        height=1.0,
        edgecolor="none"
    )

    ax.set_title(task, fontsize=14, fontweight="bold", pad=10)
    ax.invert_yaxis()

    # Remove method names and Y-axis ticks
    ax.set_yticks([])
    ax.tick_params(axis="y", left=False, labelleft=False)

    # X-axis settings
    ax.set_xlim(50, 100)
    ax.set_xticks([50, 60, 70, 80, 90, 100])
    ax.tick_params(axis="x", length=0, labelsize=10)

    # Styling
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.grid(axis="x", linestyle="--", alpha=0.2)
    ax.set_axisbelow(True)

plt.savefig(
    "./utils/figure/plot_figure1.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.03
)
plt.close()
