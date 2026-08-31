#!/usr/bin/env python3
"""Create the compact poster figure from cached pseudo50 take3 results."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from Generator.MBD_systems.visualization.results import load_metrics


SETTING_ROOT = REPO_ROOT / "Simulation-Test" / "week9_pseudo50-test"
OUTPUT_PATH = Path(__file__).resolve().parent / "pseudo50_take3.png"
SVG_PATH = OUTPUT_PATH.with_suffix(".svg")

DETECTORS = [
    (2, "CAM Only"),
    (3, "Kalman [1]"),
    (6, "PRV"),
]
ATTACKS = [
    ("randomPositionOffset", "Random Position Offset"),
    ("constantPositionOffset", "Constant Position Offset"),
]
METRICS = [
    ("false_positive_rate", "False Positive Rate"),
    ("true_positive_rate", "True Positive Rate"),
    ("f1_score", "F1 Score"),
]

# Dark and medium colors from the ColorBrewer Blues family.
COLORS = ["#08519C", "#6BAED6"]


def add_labels(axis, bars, horizontal_offset):
    for bar in bars:
        value = bar.get_height()
        axis.annotate(
            f"{value:.1f}",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(horizontal_offset, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )


def main():
    results = {}
    for attack, _ in ATTACKS:
        input_folder = SETTING_ROOT / f"json_{attack}"
        for detector_type, _ in DETECTORS:
            results[(detector_type, attack)] = load_metrics(
                SETTING_ROOT, input_folder, detector_type, no_catch=True
            )

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.labelcolor": "#243447",
        "axes.titlecolor": "#17324D",
        "xtick.color": "#243447",
        "ytick.color": "#243447",
    })

    figure, axes = plt.subplots(1, 3, figsize=(8.4, 3.65))
    x = np.arange(len(DETECTORS))
    width = 0.34

    for axis, (metric, heading) in zip(axes, METRICS):
        values_for_scale = []
        for attack_index, (attack, attack_label) in enumerate(ATTACKS):
            values = [
                100 * results[(detector_type, attack)][metric]
                for detector_type, _ in DETECTORS
            ]
            values_for_scale.extend(values)
            offset = (attack_index - 0.5) * width
            bars = axis.bar(
                x + offset,
                values,
                width,
                label=attack_label,
                color=COLORS[attack_index],
                edgecolor="white",
                linewidth=0.7,
            )
            add_labels(axis, bars, -2 if attack_index == 0 else 2)

        axis.set_title(heading, fontsize=11, fontweight="bold", pad=7)
        axis.set_xticks(x, [label for _, label in DETECTORS])
        axis.tick_params(axis="x", labelsize=9.5, length=0, pad=5)
        axis.tick_params(axis="y", labelsize=8.5, length=3)
        axis.set_ylabel("Rate (%)", fontsize=9, labelpad=3)
        axis.grid(axis="y", color="#DCE4EC", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#9AA9B8")
        axis.spines["bottom"].set_color("#9AA9B8")

        if metric == "false_positive_rate":
            axis.set_ylim(0, max(20, np.ceil((max(values_for_scale) + 3) / 5) * 5))
        else:
            axis.set_ylim(0, 105)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.065, 0.025),
        ncol=1,
        frameon=False,
        fontsize=9.5,
        handlelength=1.4,
        labelspacing=0.55,
    )
    figure.text(
        0.07,
        0.009,
        "*Results produced using our implementation of methods",
        ha="left",
        va="bottom",
        fontsize=9.5,
        fontweight="normal",
        fontstyle="italic",
        fontstretch="condensed",
        color="#243447",
        linespacing=1.1,
    )
    captions = [
        ("CAM Only:", "Identify implausible trajectories of attackers"),
        ("Kalman:", "Previous literature uses CPM to decrease FPs"),
        ("PRV:", "Proposed framework to detect plausible trajectories"),
    ]
    for y, (label, description) in zip((0.112, 0.072, 0.032), captions):
        figure.text(
            0.565,
            y,
            label,
            ha="right",
            va="baseline",
            fontsize=9.5,
            fontweight="bold",
            color="#243447",
        )
        figure.text(
            0.572,
            y,
            description,
            ha="left",
            va="baseline",
            fontsize=9.5,
            color="#243447",
        )
    figure.subplots_adjust(left=0.07, right=0.995, top=0.92, bottom=0.23, wspace=0.20)
    figure.savefig(OUTPUT_PATH, dpi=600, bbox_inches="tight", facecolor="white")
    figure.savefig(SVG_PATH, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Saved {OUTPUT_PATH}")
    print(f"Saved {SVG_PATH}")


if __name__ == "__main__":
    main()
