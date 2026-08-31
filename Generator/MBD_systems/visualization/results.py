import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
MAIN_SCRIPT = REPO_ROOT / "Generator" / "MBD_systems" / "main.py"

DETECTORS = {
    2: ("CAM Only", "kalman_cam_only"),
    3: ("Kalman", "kalman_cam_cpm"),
    6: ("PRV", "kalman_cam_cpm_prv"),
}

ATTACK_LABELS = {
    "randomPositionOffset": "Random Position Offset",
    "constantPositionOffset": "Constant Position Offset",
    "randomSpeedOffset": "Random Speed Offset",
    "constantSpeedOffset": "Constant Speed Offset",
    "suddenStop": "Sudden Stop",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run and plot Kalman detector comparisons.")
    parser.add_argument(
        "filename",
        help="Output filename suffix; the setting is added as a prefix",
    )
    parser.add_argument("--setting", default="urban", help="Simulation setting, such as urban or highway")
    parser.add_argument(
        "--setting-root",
        type=Path,
        help="Explicit Simulation-Test folder (overrides the folder derived from --setting)",
    )
    parser.add_argument(
        "--types", nargs="+", type=int, default=[2, 3, 6],
        help="Plotted detector types (2, 3, or 6)",
    )
    parser.add_argument(
        "--attacks",
        nargs="+",
        default=["randomPositionOffset", "constantPositionOffset"],
        help="Attack folder suffixes",
    )
    parser.add_argument(
        "--include-f1",
        action="store_true",
        help="Add an F1 Score subplot to the generated figure",
    )
    parser.add_argument(
        "--no-pos-check",
        action="store_true",
        help="Disable only CaTCH's road-edge position plausibility check",
    )
    parser.add_argument(
        "--no-catch",
        action="store_true",
        help="Run Kalman detector types without the CaTCH gate",
    )
    parser.add_argument(
        "--skip-runs",
        action="store_true",
        help="Plot existing detector outputs without rerunning the detectors",
    )
    return parser.parse_args()


def validate_args(args):
    unsupported = [detector_type for detector_type in args.types if detector_type not in DETECTORS]
    if unsupported:
        raise ValueError(f"Unsupported detector types: {unsupported}; choose from 2, 3, or 6")
    if len(set(args.types)) != len(args.types):
        raise ValueError("Detector types must not contain duplicates")
    if len(set(args.attacks)) != len(args.attacks):
        raise ValueError("Attacks must not contain duplicates")


def run_detector(input_folder, detector_type, no_pos_check=False, no_catch=False):
    print(f"Running type {detector_type} on {input_folder.name}...", flush=True)
    command = [
        sys.executable,
        str(MAIN_SCRIPT),
        "--input_folder",
        str(input_folder),
        "--type",
        str(detector_type),
        "--train",
        "0",
    ]
    if no_pos_check:
        command.append("--no-pos-check")
    if no_catch:
        command.append("--no-catch")
    subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
    )


def load_metrics(setting_root, input_folder, detector_type, no_catch=False):
    _, result_folder = DETECTORS[detector_type]
    if no_catch:
        result_folder += "_no_catch"
    metrics_path = setting_root / "results" / input_folder.name / result_folder / "predicted.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Detector did not create {metrics_path}")

    with metrics_path.open("r", encoding="utf-8") as file:
        metrics = json.load(file)

    window_path = setting_root / "evaluation_window.json"
    if window_path.is_file():
        with window_path.open("r", encoding="utf-8") as file:
            window = json.load(file)
        debug_path = metrics_path.with_name("debug.json")
        with debug_path.open("r", encoding="utf-8") as file:
            decisions = json.load(file)
        start_ns = int(float(window["start_time_s"]) * 1_000_000_000)
        end_ns = int(float(window["end_time_s"]) * 1_000_000_000)
        decisions = [
            decision for decision in decisions
            if decision.get("message_type", "CAM") == "CAM"
            and start_ns <= int(decision["rcvTime"]) < end_ns
        ]
        metrics["tp"] = sum(
            decision["attacker"] == 1 and decision["prediction"] == 1
            for decision in decisions
        )
        metrics["tn"] = sum(
            decision["attacker"] == 0 and decision["prediction"] == 0
            for decision in decisions
        )
        metrics["fp"] = sum(
            decision["attacker"] == 0 and decision["prediction"] == 1
            for decision in decisions
        )
        metrics["fn"] = sum(
            decision["attacker"] == 1 and decision["prediction"] == 0
            for decision in decisions
        )
        metrics["total_messages"] = len(decisions)
        metrics["evaluation_start_time_s"] = window["start_time_s"]
        metrics["evaluation_end_time_s"] = window["end_time_s"]

    tp, tn = metrics["tp"], metrics["tn"]
    fp, fn = metrics["fp"], metrics["fn"]
    metrics["true_positive_rate"] = tp / (tp + fn) if tp + fn else 0.0
    metrics["false_positive_rate"] = fp / (fp + tn) if fp + tn else 0.0
    metrics["f1_score"] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return metrics


def attack_label(attack):
    return ATTACK_LABELS.get(attack, attack.replace("_", " ").title())


def add_value_labels(axis, bars):
    for bar in bars:
        height = bar.get_height()
        axis.annotate(
            f"{height:.1f}%",
            (bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_results(args, results, output_path):
    detector_labels = [DETECTORS[detector_type][0] for detector_type in args.types]
    x_positions = np.arange(len(args.types))
    bar_width = 0.78 / len(args.attacks)
    colors = ["#4C78A8", "#F28E2B", "#59A14F", "#B07AA1"]

    plot_specs = [
        ("false_positive_rate", "False Positive Rate"),
        ("true_positive_rate", "True Positive Rate"),
    ]
    if args.include_f1:
        plot_specs.append(("f1_score", "F1 Score"))

    figure, axes = plt.subplots(1, len(plot_specs), figsize=(6.5 * len(plot_specs), 5.5))
    axes = np.atleast_1d(axes)
    plot_specs = [
        (metric_name, title, axes[index])
        for index, (metric_name, title) in enumerate(plot_specs)
    ]

    for metric_name, title, axis in plot_specs:
        all_values = []
        for attack_index, attack in enumerate(args.attacks):
            values = [results[(detector_type, attack)][metric_name] * 100 for detector_type in args.types]
            all_values.extend(values)
            offset = (attack_index - (len(args.attacks) - 1) / 2) * bar_width
            bars = axis.bar(
                x_positions + offset,
                values,
                bar_width,
                label=attack_label(attack),
                color=colors[attack_index % len(colors)],
            )
            add_value_labels(axis, bars)

        axis.set_title(title, fontsize=13, weight="bold")
        axis.set_ylabel("Rate (%)")
        axis.set_xticks(
            x_positions, detector_labels, rotation=15, ha="right", fontsize=9
        )
        axis.grid(axis="y", alpha=0.25, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

        if metric_name == "false_positive_rate":
            upper_limit = min(100, max(10, math.ceil((max(all_values, default=0) + 8) / 10) * 10))
            axis.set_ylim(0, upper_limit)
        else:
            axis.set_ylim(0, 105)

    handles, labels = axes[0].get_legend_handles_labels()
    setting_label = (
        args.setting if any(character.isupper() for character in args.setting)
        else args.setting.title()
    )
    if args.no_pos_check:
        setting_label += " — No Road-Edge Position Check"
    if args.no_catch:
        setting_label += " — No CaTCH Gate"
    figure.suptitle(f"{setting_label} Detection Performance", fontsize=15, weight="bold", y=0.98)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=len(labels),
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.84))
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def print_metrics(args, results):
    print("\nDetector metrics")
    f1_header = f" {'F1':>8}" if args.include_f1 else ""
    print(f"{'Detector':<24} {'Attack':<26} {'FPR':>8} {'TPR':>8}{f1_header}")
    print("-" * (79 if args.include_f1 else 70))
    for detector_type in args.types:
        for attack in args.attacks:
            metrics = results[(detector_type, attack)]
            f1_value = f" {metrics['f1_score'] * 100:>7.2f}%" if args.include_f1 else ""
            print(
                f"{DETECTORS[detector_type][0]:<24} "
                f"{attack_label(attack):<26} "
                f"{metrics['false_positive_rate'] * 100:>7.2f}% "
                f"{metrics['true_positive_rate'] * 100:>7.2f}%"
                f"{f1_value}"
            )


def main():
    args = parse_args()
    try:
        validate_args(args)
        setting_root = (
            args.setting_root.resolve()
            if args.setting_root is not None
            else REPO_ROOT / "Simulation-Test" / f"{args.setting}-test"
        )
        if not setting_root.is_dir():
            raise FileNotFoundError(f"Setting folder does not exist: {setting_root}")

        results = {}
        for attack in args.attacks:
            input_folder = setting_root / f"json_{attack}"
            if not input_folder.is_dir():
                raise FileNotFoundError(f"Attack input folder does not exist: {input_folder}")
            for detector_type in args.types:
                if not args.skip_runs:
                    run_detector(
                        input_folder, detector_type, args.no_pos_check, args.no_catch
                    )
                results[(detector_type, attack)] = load_metrics(
                    setting_root, input_folder, detector_type, args.no_catch
                )

        output_path = (
            Path(__file__).resolve().parent
            / "created"
            / f"{args.setting}_{args.filename}"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plot_results(args, results, output_path)
        print_metrics(args, results)
        print(f"\nSaved figure in {output_path}")
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
