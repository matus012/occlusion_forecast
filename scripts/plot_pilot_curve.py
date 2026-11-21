"""Plot the README figure from results/pilot_summary.json (pure matplotlib, no AV2 pixels).

    python scripts/plot_pilot_curve.py   ->  reports/figures/pilot_degradation.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "pilot_summary.json"
OUT = ROOT / "reports" / "figures" / "pilot_degradation.png"

STYLE = {
    "C1": ("clean-trained model, gaps left as-is", "#1f6fd1", "o", "-"),
    "C2": ("same model, gaps filled with constant velocity", "#2a9d5c", "s", "--"),
    "C3": ("occlusion-trained model (collapsed: ignores history)", "#b5b5b5", "^", ":"),
}


def main() -> None:
    data = json.loads(SRC.read_text(encoding="utf-8"))
    share = data["severity_hidden_share"]
    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(figsize=(8.0, 4.6), dpi=110)
    fig.patch.set_facecolor("white")
    for arm, (label, color, marker, ls) in STYLE.items():
        rows = data["arms"][arm]["by_severity"]
        xs = [100 * share[s] for s in rows]
        ys = [rows[s]["minfde6"] for s in rows]
        faint = arm == "C3"
        ax.plot(xs, ys, color=color, marker=marker, ls=ls, lw=1.5 if faint else 2.5, ms=5 if faint else 7,
                alpha=0.8 if faint else 1.0, label=label)
    c1 = data["arms"]["C1"]["by_severity"]
    ax.annotate(
        f"+{100 * (c1['S4']['minfde6'] / c1['S0']['minfde6'] - 1):.0f} %",
        xy=(80, c1["S4"]["minfde6"]), xytext=(70.5, c1["S4"]["minfde6"] - 0.05),
        color=STYLE["C1"][1], fontsize=13, fontweight="bold", ha="right",
    )
    ax.set_xlabel("share of the 5 s observed history hidden (%)")
    ax.set_ylabel("final-position error, minFDE$_6$ (m)")
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10.5, loc="upper left", bbox_to_anchor=(0.01, 0.9), frameon=True)
    ax.set_title("Local pilot: 1 seed, 6 of 20 epochs, 20k training scenarios", fontsize=12)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
