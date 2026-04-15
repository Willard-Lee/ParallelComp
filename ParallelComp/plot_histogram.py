"""
plot_histogram.py
Brazil Weather Analysis Tool – CAT3024N Parallel Computing
----------------------------------------------------------
Reads histogram_data.txt written by the C++ OpenCL program
and produces a clean matplotlib bar chart saved as
histogram_plot.png, then opens it automatically.

histogram_data.txt format:
  # comment lines starting with #
  bin_low  bin_high  count       (one row per bin)
"""

import sys
import os
import subprocess

# ── Try importing matplotlib ──────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend – works everywhere
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    print("matplotlib not found. Install it with:  pip install matplotlib")
    sys.exit(1)

# ── Read the data file ────────────────────────────────────
DATA_FILE = "histogram_data.txt"
PLOT_FILE = "histogram_plot.png"

if not os.path.exists(DATA_FILE):
    print(f"ERROR: {DATA_FILE} not found. Run the C++ program first.")
    sys.exit(1)

bin_lows   = []
bin_highs  = []
counts     = []
params     = {}

with open(DATA_FILE, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            # Parse header: # bins=10 min=1.00 max=40.00 binwidth=3.90
            for token in line.split():
                if "=" in token:
                    k, v = token.split("=", 1)
                    try:
                        params[k] = float(v)
                    except ValueError:
                        pass
            continue
        parts = line.split()
        if len(parts) >= 3:
            bin_lows.append(float(parts[0]))
            bin_highs.append(float(parts[1]))
            counts.append(int(parts[2]))

if not counts:
    print("ERROR: No data found in histogram_data.txt")
    sys.exit(1)

num_bins   = int(params.get("bins",     len(counts)))
min_val    = params.get("min",      bin_lows[0])
max_val    = params.get("max",      bin_highs[-1])
bin_width  = params.get("binwidth", bin_highs[0] - bin_lows[0])
total      = sum(counts)

# ── Build bar positions and labels ────────────────────────
bar_centres = [(lo + hi) / 2 for lo, hi in zip(bin_lows, bin_highs)]
bar_width   = bin_width * 0.85          # slight gap between bars
x_labels    = [f"{lo:.1f}" for lo in bin_lows] + [f"{bin_highs[-1]:.1f}"]

# ── Colour bars by temperature zone ──────────────────────
def bar_colour(mid):
    if   mid < 10:  return "#85B7EB"   # cool – blue
    elif mid < 20:  return "#9FE1CB"   # mild – teal
    elif mid < 30:  return "#EF9F27"   # warm – amber
    else:           return "#D85A30"   # hot  – coral

colours = [bar_colour(c) for c in bar_centres]

# ── Plot ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("#FAFAFA")
ax.set_facecolor("#F5F5F5")

bars = ax.bar(bar_centres, counts, width=bar_width,
              color=colours, edgecolor="white", linewidth=0.6, zorder=3)

# Value labels on top of each bar
for bar, count in zip(bars, counts):
    if count > 0:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.01,
                str(count),
                ha="center", va="bottom",
                fontsize=7.5, color="#444441")

# Grid
ax.yaxis.grid(True, color="white", linewidth=1.2, zorder=2)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#D3D1C7")
ax.spines["bottom"].set_color("#D3D1C7")

# Axis labels and title
ax.set_xlabel("Temperature (°C)", fontsize=11, color="#444441", labelpad=8)
ax.set_ylabel("Number of readings", fontsize=11, color="#444441", labelpad=8)
ax.set_title(
    f"Brazil Air Temperature Distribution  |  {num_bins} bins  |  "
    f"{total:,} readings  |  bin width = {bin_width:.2f}°C",
    fontsize=11, color="#2C2C2A", pad=12
)

# X ticks at bin edges
ax.set_xticks([lo for lo in bin_lows] + [bin_highs[-1]])
ax.set_xticklabels(x_labels, fontsize=8, color="#5F5E5A", rotation=45, ha="right")
ax.tick_params(axis="y", labelcolor="#5F5E5A", labelsize=9)

# Legend for colour coding
legend_patches = [
    mpatches.Patch(color="#85B7EB", label="< 10°C  (cool)"),
    mpatches.Patch(color="#9FE1CB", label="10–20°C (mild)"),
    mpatches.Patch(color="#EF9F27", label="20–30°C (warm)"),
    mpatches.Patch(color="#D85A30", label="> 30°C  (hot)"),
]
ax.legend(handles=legend_patches, fontsize=8.5,
          framealpha=0.9, edgecolor="#D3D1C7",
          loc="upper right")

# Stats annotation box
stats_text = (
    f"Min: {min_val:.1f}°C\n"
    f"Max: {max_val:.1f}°C\n"
    f"Range: {max_val - min_val:.1f}°C"
)
ax.text(0.02, 0.97, stats_text,
        transform=ax.transAxes,
        fontsize=8.5, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor="white", edgecolor="#D3D1C7", alpha=0.9))

plt.tight_layout()

# ── Save and open ─────────────────────────────────────────
plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
print(f"Plot saved to {PLOT_FILE}")

# Open the image automatically (Windows, macOS, Linux)
try:
    if sys.platform.startswith("win"):
        os.startfile(PLOT_FILE)
    elif sys.platform == "darwin":
        subprocess.run(["open", PLOT_FILE])
    else:
        subprocess.run(["xdg-open", PLOT_FILE])
except Exception:
    print(f"Could not open image automatically. Open {PLOT_FILE} manually.")