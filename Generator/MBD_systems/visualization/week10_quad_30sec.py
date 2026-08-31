#!/usr/bin/env python3
"""Slice Week-10 attacks, run the paper detectors, and make 2x2 figures.

Evaluation begins at each communication-window boundary. Three settings use
30 seconds; sparse Highway 2 AM uses its complete 300-second window.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
WEEK10_ROOT = REPO_ROOT / "Simulations-week10"
ATTACK_ROOT = WEEK10_ROOT / "attacks"
WORK_ROOT = WEEK10_ROOT / ".work" / "visualization-corrected"
OUTPUT_ROOT = Path(__file__).resolve().parent / "created"
MAIN_SCRIPT = REPO_ROOT / "Generator" / "MBD_systems" / "main.py"
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"

SETTINGS = (
    ("InTAS_urban_2AM_300sec", "Urban", "2 AM", 7_200, 30, "urban-low"),
    ("InTAS_urban_7AM_300sec", "Urban", "7 AM", 25_200, 30, "urban-low"),
    ("InTAS_highway_2AM_300sec", "Highway", "2 AM", 7_200, 300, "urban-low"),
    ("InTAS_highway_7AM_300sec", "Highway", "7 AM", 25_200, 30, "urban-low"),
)
ATTACKS = (
    ("randomPositionOffset", "Random Position Offset", "#4C78A8"),
    ("constantPositionOffset", "Constant Position Offset", "#F28E2B"),
)
DETECTORS = (
    (2, "CAM Only", "kalman_cam_only_no_catch"),
    (3, "Kalman", "kalman_cam_cpm_no_catch"),
    (6, "PRV", "kalman_cam_cpm_prv_no_catch"),
)
ATTACK_SEEDS = (1, 2, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "run", "plot", "all"), nargs="?", default="all"
    )
    parser.add_argument("--slice-workers", type=int, default=4)
    parser.add_argument("--detector-jobs", type=int, default=2)
    parser.add_argument(
        "--areas", nargs="+", choices=("urban", "highway"),
        help="Limit detector execution to these areas (plot still requires all results)",
    )
    parser.add_argument(
        "--attacks", nargs="+", choices=tuple(name for name, _label, _color in ATTACKS),
        help="Limit detector execution to these attacks",
    )
    parser.add_argument(
        "--attack-seeds", nargs="+", type=int, choices=ATTACK_SEEDS,
        help="Limit detector execution to these attack seeds",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def iter_json_objects(path: Path) -> Iterator[dict]:
    """Yield objects from a top-level JSON array without loading the file."""
    depth = 0
    in_string = False
    escaped = False
    current: list[str] = []
    with path.open("r", encoding="utf-8") as source:
        while chunk := source.read(1024 * 1024):
            for character in chunk:
                if depth == 0:
                    if character != "{":
                        continue
                    depth = 1
                    current = [character]
                    in_string = False
                    escaped = False
                    continue

                current.append(character)
                if in_string:
                    if escaped:
                        escaped = False
                    elif character == "\\":
                        escaped = True
                    elif character == '"':
                        in_string = False
                elif character == '"':
                    in_string = True
                elif character == "{":
                    depth += 1
                elif character == "}":
                    depth -= 1
                    if depth == 0:
                        yield json.loads("".join(current))
                        current = []
    if depth:
        raise ValueError(f"Incomplete JSON object in {path}")


def slice_file(job: tuple[str, str, int, int]) -> tuple[str, int, int]:
    source_text, destination_text, start_ns, end_ns = job
    source = Path(source_text)
    destination = Path(destination_text)
    selected: list[dict] = []
    scanned = 0
    for record in iter_json_objects(source):
        scanned += 1
        event_time = record.get("rcvTime", record.get("sendTime"))
        if event_time is None:
            raise ValueError(f"Missing rcvTime/sendTime in {source}")
        event_time = int(event_time)
        if event_time >= end_ns:
            # Week-10 receiver files are ordered by receive time.
            break
        if event_time >= start_ns:
            selected.append(record)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(selected, output, separators=(",", ":"))
    os.replace(temporary, destination)
    return str(destination.parent.parent), scanned, len(selected)


def dataset_root(setting: str, attack: str, seed: int) -> Path:
    return WORK_ROOT / setting / f"attack-seed-{seed}" / f"json_{attack}"


def source_root(setting: str, attack: str, seed: int) -> Path:
    return ATTACK_ROOT / attack / str(seed) / "simulation-seed-1" / setting


def prepare(force: bool, workers: int) -> None:
    if workers < 1:
        raise ValueError("--slice-workers must be at least 1")
    jobs: list[tuple[str, str, int, int]] = []
    dataset_info: dict[str, dict] = {}

    for setting, area, time_label, start_s, duration_s, _profile in SETTINGS:
        start_ns = start_s * 1_000_000_000
        end_ns = (start_s + duration_s) * 1_000_000_000
        for attack, _label, _color in ATTACKS:
            for seed in ATTACK_SEEDS:
                source = source_root(setting, attack, seed)
                destination = dataset_root(setting, attack, seed)
                metadata_path = destination / "slice_metadata.json"
                if not source.is_dir():
                    raise FileNotFoundError(source)
                if metadata_path.is_file() and not force:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    if metadata.get("status") == "complete":
                        print(f"Prepared (cached): {destination.relative_to(REPO_ROOT)}")
                        continue
                key = str(destination)
                dataset_info[key] = {
                    "status": "complete",
                    "setting": setting,
                    "area": area,
                    "time_label": time_label,
                    "attack": attack,
                    "attack_seed": seed,
                    "simulation_seed": 1,
                    "window_start_s": start_s,
                    "window_end_s": start_s + duration_s,
                    "window_duration_s": duration_s,
                    "window_semantics": "start-inclusive, end-exclusive receive time; ego falls back to send time",
                    "source": str(source.relative_to(REPO_ROOT)),
                    "files": {},
                    "records": {},
                }
                for message_type in ("cam", "cpm", "ego"):
                    for source_file in sorted((source / message_type).glob("*.json")):
                        destination_file = destination / message_type / source_file.name
                        jobs.append((str(source_file), str(destination_file), start_ns, end_ns))

    print(f"Slicing {len(jobs):,} receiver files with {workers} workers...", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for _result in executor.map(slice_file, jobs, chunksize=8):
            pass

    for key, metadata in dataset_info.items():
        destination = Path(key)
        counts = {}
        file_counts = {}
        for message_type in ("cam", "cpm", "ego"):
            paths = list((destination / message_type).glob("*.json"))
            file_counts[message_type] = len(paths)
            count = 0
            for path in paths:
                count += sum(1 for _ in iter_json_objects(path))
            counts[message_type] = count
        metadata["files"] = file_counts
        metadata["records"] = counts
        metadata_path = destination / "slice_metadata.json"
        temporary = metadata_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, metadata_path)
        print(
            f"Prepared {destination.relative_to(REPO_ROOT)}: "
            f"CAM={counts['cam']:,}, CPM={counts['cpm']:,}, ego={counts['ego']:,}"
        )


def detector_result_path(setting: str, attack: str, seed: int, result_dir: str) -> Path:
    input_root = dataset_root(setting, attack, seed)
    return input_root.parent / "results" / input_root.name / result_dir / "predicted.json"


def run_one_detector(job: tuple[str, str, int, int, str, str, bool]) -> str:
    setting, attack, seed, detector_type, result_dir, profile, force = job
    input_root = dataset_root(setting, attack, seed)
    result_path = detector_result_path(setting, attack, seed, result_dir)
    if result_path.is_file() and not force:
        return f"Detector cached: {setting} / {attack} / seed {seed} / type {detector_type}"

    log_dir = WORK_ROOT / "logs" / setting / f"attack-seed-{seed}" / attack
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"detector-{detector_type}.log"
    command = [
        str(PYTHON if PYTHON.is_file() else Path(sys.executable)),
        str(MAIN_SCRIPT),
        "--input_folder", str(input_root),
        "--type", str(detector_type),
        "--train", "0",
        "--no-catch",
        "--catch-profile", profile,
        "--workers", "1",
    ]
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)
    if not result_path.is_file():
        raise FileNotFoundError(f"Detector completed without {result_path}")
    return f"Detector complete: {setting} / {attack} / seed {seed} / type {detector_type}"


def run_detectors(
    force: bool, jobs: int, areas: list[str] | None = None,
    attacks: list[str] | None = None, attack_seeds: list[int] | None = None,
) -> None:
    if jobs < 1:
        raise ValueError("--detector-jobs must be at least 1")
    work = []
    for setting, _area, _time_label, _start_s, _duration_s, profile in SETTINGS:
        if areas and _area.lower() not in areas:
            continue
        for attack, _label, _color in ATTACKS:
            if attacks and attack not in attacks:
                continue
            for seed in ATTACK_SEEDS:
                if attack_seeds and seed not in attack_seeds:
                    continue
                metadata = dataset_root(setting, attack, seed) / "slice_metadata.json"
                if not metadata.is_file():
                    raise FileNotFoundError(f"Run prepare first; missing {metadata}")
                for detector_type, _label, result_dir in DETECTORS:
                    work.append((setting, attack, seed, detector_type, result_dir, profile, force))

    print(f"Running {len(work)} detector jobs with concurrency {jobs}...", flush=True)
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [executor.submit(run_one_detector, job) for job in work]
        for future in as_completed(futures):
            print(future.result(), flush=True)


def rates(counts: dict[str, int]) -> dict[str, float]:
    tp, tn, fp, fn = (counts[name] for name in ("tp", "tn", "fp", "fn"))
    return {
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "true_positive_rate": tp / (tp + fn) if tp + fn else 0.0,
        "f1_score": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def collect_metrics() -> dict:
    report: dict = {
        "description": "Week-10 detector comparison; attack seeds are micro-aggregated",
        "simulation_seed": 1,
        "attack_seeds": list(ATTACK_SEEDS),
        "detectors": {str(number): label for number, label, _folder in DETECTORS},
        "settings": {},
    }
    for setting, area, time_label, start_s, duration_s, profile in SETTINGS:
        setting_report = {
            "area": area,
            "time_label": time_label,
            "window_start_s": start_s,
            "window_end_s": start_s + duration_s,
            "window_duration_s": duration_s,
            "catch_profile": profile,
            "attacks": {},
        }
        for attack, attack_label, _color in ATTACKS:
            attack_report = {"label": attack_label, "detectors": {}}
            for detector_type, detector_label, result_dir in DETECTORS:
                per_seed = {}
                pooled = {name: 0 for name in ("tp", "tn", "fp", "fn")}
                for seed in ATTACK_SEEDS:
                    path = detector_result_path(setting, attack, seed, result_dir)
                    if not path.is_file():
                        raise FileNotFoundError(f"Run detectors first; missing {path}")
                    raw = json.loads(path.read_text(encoding="utf-8"))
                    counts = {name: int(raw[name]) for name in pooled}
                    per_seed[str(seed)] = {"counts": counts, "rates": rates(counts)}
                    for name in pooled:
                        pooled[name] += counts[name]
                attack_report["detectors"][str(detector_type)] = {
                    "label": detector_label,
                    "per_seed": per_seed,
                    "pooled_counts": pooled,
                    "pooled_rates": rates(pooled),
                }
            setting_report["attacks"][attack] = attack_report
        report["settings"][setting] = setting_report
    return report


def add_value_labels(axis, bars) -> None:
    for bar in bars:
        value = bar.get_height()
        axis.annotate(
            f"{value:.1f}%", (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8,
        )


def plot_metric(report: dict, metric: str, heading: str, filename: str) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    x_positions = np.arange(len(DETECTORS))
    bar_width = 0.36
    all_values = []

    for axis, (setting, area, time_label, _start_s, duration_s, _profile) in zip(axes.flat, SETTINGS):
        for attack_index, (attack, attack_label, color) in enumerate(ATTACKS):
            values = [
                report["settings"][setting]["attacks"][attack]["detectors"][str(detector_type)]
                ["pooled_rates"][metric] * 100
                for detector_type, _label, _folder in DETECTORS
            ]
            all_values.extend(values)
            offset = (attack_index - 0.5) * bar_width
            bars = axis.bar(
                x_positions + offset, values, bar_width, label=attack_label, color=color
            )
            add_value_labels(axis, bars)

        axis.set_title(f"{area} — {time_label} ({duration_s} s)", fontsize=13, weight="bold")
        axis.set_ylabel("Rate (%)")
        axis.set_xticks(x_positions, [label for _number, label, _folder in DETECTORS], rotation=15, ha="right")
        axis.grid(axis="y", alpha=0.25, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    if metric == "false_positive_rate":
        upper = min(100, max(10, math.ceil((max(all_values, default=0) + 8) / 10) * 10))
    else:
        upper = 105
    for axis in axes.flat:
        axis.set_ylim(0, upper)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.suptitle(f"Week-10 {heading}", fontsize=16, weight="bold", y=0.985)
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=2, frameon=False)
    figure.text(
        0.5, 0.015,
        "Attack seeds 1–3 pooled • Simulation seed 1 • No CaTCH gate",
        ha="center", fontsize=9, color="#444444",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.90))
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_ROOT / filename
    save_options = {"bbox_inches": "tight", "facecolor": "white"}
    if output_path.suffix.lower() == ".png":
        save_options["dpi"] = 200
    figure.savefig(output_path, **save_options)
    plt.close(figure)
    print(f"Saved {output_path.relative_to(REPO_ROOT)}")


def plot() -> None:
    report = collect_metrics()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = WORK_ROOT / "week10_quad_metrics.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, report_path)
    plot_metric(report, "false_positive_rate", "False Positive Rate", "week10_false_positive_rate_quad.png")
    plot_metric(report, "true_positive_rate", "True Positive Rate", "week10_true_positive_rate_quad.png")
    plot_metric(report, "f1_score", "F1 Score", "week10_f1_score_quad.png")
    print(f"Saved {report_path.relative_to(REPO_ROOT)}")


def main() -> None:
    args = parse_args()
    if args.command in ("prepare", "all"):
        prepare(args.force, args.slice_workers)
    if args.command in ("run", "all"):
        run_detectors(
            args.force, args.detector_jobs, args.areas, args.attacks, args.attack_seeds
        )
    if args.command in ("plot", "all"):
        plot()


if __name__ == "__main__":
    main()
