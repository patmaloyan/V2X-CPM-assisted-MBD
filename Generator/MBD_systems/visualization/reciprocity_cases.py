#!/usr/bin/env python3
"""Summarize the directional reciprocity cases used by PRV (Type 6).

This is deliberately a diagnostic beside the detector: it reuses PRV's Kalman
association and one-second buckets, but does not alter its scoring or decisions.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys

import ijson
import matplotlib.pyplot as plt


MBD_DIR = Path(__file__).resolve().parents[1]
if str(MBD_DIR) not in sys.path:
    sys.path.insert(0, str(MBD_DIR))

from data_structures import Parameters  # noqa: E402
from kalman_detector import evaluation_receiver_ids, load_json_list  # noqa: E402
from cpm_detector import (  # noqa: E402
    PrvDetector,
)


CASES = ("mutual", "b_to_a_only", "a_to_b_only")
CASE_LABELS = {
    "mutual": "Mutual\n$B \\leftrightarrow A$",
    "b_to_a_only": "$B \\rightarrow A$\nonly",
    "a_to_b_only": "$A \\rightarrow B$\nonly",
}
COLORS = ("#4C78A8", "#59A14F", "#E15759")


class ReciprocityCaseCollector(PrvDetector):
    """Collect evidence direction cases immediately before PRV scores a bucket."""

    def __init__(self, identity_by_alias):
        super().__init__(Parameters(), catch_enabled=False)
        self.identity_by_alias = identity_by_alias
        self.subject_by_track = {}
        self.case_counts = defaultdict(Counter)

    def track_id(self, track, create_trust=False):
        track_id = super().track_id(track, create_trust)
        identity = self.identity_by_alias.get(str(track.station_alias))
        if identity is not None:
            self.subject_by_track[track_id] = identity["sender_id"]
        return track_id

    def close_current_bucket(self):
        accepted = {
            track_id: state.accepted for track_id, state in self.trust.items()
        }
        for subject_id in self.trust:
            subject = self.subject_by_track.get(subject_id)
            if subject is None:
                continue
            for counterpart_id, counterpart_accepted in accepted.items():
                if counterpart_id == subject_id or not counterpart_accepted:
                    continue
                # The subject being judged is B; the other local track is A.
                b_to_a = self.bucket_edges.get((subject_id, counterpart_id))
                a_to_b = self.bucket_edges.get((counterpart_id, subject_id))
                if b_to_a is not None and a_to_b is not None:
                    case = "mutual"
                elif b_to_a is not None:
                    case = "b_to_a_only"
                elif a_to_b is not None:
                    case = "a_to_b_only"
                else:
                    continue
                self.case_counts[subject][case] += 1
        return super().close_current_bucket()


def identity_map(cam_paths):
    identities = {}
    for path in cam_paths:
        for message in load_json_list(path):
            alias = str(message.get("sender_alias", 0))
            identities[alias] = {
                "sender_id": str(message.get("sender_id", alias)),
                "attacker": int(message.get("attacker", 0)),
            }
    return identities


def collect_receiver(args):
    receiver_id, cam_path, cpm_path, ego_path, identities = args
    detector = ReciprocityCaseCollector(identities)
    detector.process_receiver(cam_path, cpm_path, ego_path)
    return receiver_id, {
        subject: dict(counts) for subject, counts in detector.case_counts.items()
    }


def collect_attack(input_folder, workers):
    cam_dir = input_folder / "cam"
    cpm_dir = input_folder / "cpm"
    ego_dir = input_folder / "ego"
    cam_paths = {path.stem: path for path in cam_dir.glob("*.json")}
    cpm_paths = {path.stem: path for path in cpm_dir.glob("*.json")}
    receiver_ids = evaluation_receiver_ids(
        input_folder, sorted(set(cam_paths) | set(cpm_paths))
    )
    identities = identity_map(cam_paths.values())
    jobs = [
        (
            receiver_id,
            cam_paths.get(receiver_id),
            cpm_paths.get(receiver_id),
            ego_dir / f"{receiver_id}.json",
            identities,
        )
        for receiver_id in receiver_ids
    ]

    by_subject = defaultdict(Counter)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(collect_receiver, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            _receiver_id, receiver_counts = future.result()
            for subject, counts in receiver_counts.items():
                by_subject[subject].update(counts)
            if index % 10 == 0 or index == len(futures):
                print(f"  processed {index}/{len(futures)} receivers", flush=True)

    attacker_by_subject = {
        value["sender_id"]: value["attacker"] for value in identities.values()
    }
    grouped = {}
    for attacker, group_name in ((0, "benign"), (1, "attacker")):
        subjects = [
            subject for subject in by_subject
            if attacker_by_subject.get(subject) == attacker
        ]
        totals = Counter()
        for subject in subjects:
            totals.update(by_subject[subject])
        total_cases = sum(totals.values())
        grouped[group_name] = {
            "vehicles": len(subjects),
            "total_cases": total_cases,
            "cases": {
                case: {
                    "count": totals[case],
                    "average_per_vehicle": (
                        totals[case] / len(subjects) if subjects else 0.0
                    ),
                    "percentage": (
                        100.0 * totals[case] / total_cases if total_cases else 0.0
                    ),
                }
                for case in CASES
            },
        }
    return grouped


def collect_debug(debug_path):
    """Reconstruct cases from an existing PRV run without rerunning Kalman.

    The debug JSON includes non-standard ``NaN`` values, so sed replaces those
    numeric placeholders while ijson reads the multi-gigabyte array as a stream.
    """
    counts_by_subject = defaultdict(Counter)
    attacker_by_subject = {}
    subjects = {"benign": set(), "attacker": set()}
    current_key = None
    states = {}
    edges = set()

    def finish_bucket():
        if current_key is None:
            return
        for subject, attacker in states.items():
            subjects["attacker" if attacker else "benign"].add(subject)
        pairs = {
            tuple(sorted((source, target)))
            for source, target in edges
            if source != target and source in states and target in states
        }
        for first, second in pairs:
            for subject, counterpart in ((first, second), (second, first)):
                # PRV scores B only when its counterpart A was accepted at the
                # start of this interval. CPM sources are themselves accepted.
                if states.get(counterpart) is None:
                    continue
                outbound = (subject, counterpart) in edges
                inbound = (counterpart, subject) in edges
                if outbound and inbound:
                    case = "mutual"
                elif outbound:
                    case = "b_to_a_only"
                elif inbound:
                    case = "a_to_b_only"
                else:
                    continue
                counts_by_subject[subject][case] += 1

    process = subprocess.Popen(
        ["sed", "s/NaN/0.0/g", str(debug_path)], stdout=subprocess.PIPE
    )
    assert process.stdout is not None
    try:
        for record_index, record in enumerate(
            ijson.items(process.stdout, "item", use_float=True), 1
        ):
            if str(record.get("message_type", "")).upper() != "CPM":
                continue
            key = (str(record.get("receiver_id")), int(record["reciprocity_bucket"]))
            if current_key is not None and key != current_key:
                finish_bucket()
                states = {}
                edges = set()
            current_key = key
            source = str(record.get("sender_id"))
            attacker = int(record.get("attacker", 0))
            attacker_by_subject[source] = attacker
            # None marks a quarantined counterpart; 0/1 are benign/attacker and accepted.
            states[source] = (
                attacker if record.get("reciprocity_state") == "accepted" else None
            )
            if record.get("accepted"):
                for event in record.get("cpm_object_events", []):
                    if event.get("edge_added") is True:
                        edges.add((source, str(event.get("object_id"))))
            if record_index % 250_000 == 0:
                print(f"  streamed {record_index:,} debug records", flush=True)
        finish_bucket()
    finally:
        process.stdout.close()
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, process.args)

    grouped = {}
    for attacker, group_name in ((0, "benign"), (1, "attacker")):
        group_subjects = {
            subject for subject in subjects[group_name]
            if attacker_by_subject.get(subject) == attacker
        }
        totals = Counter()
        for subject in group_subjects:
            totals.update(counts_by_subject[subject])
        total_cases = sum(totals.values())
        grouped[group_name] = {
            "vehicles": len(group_subjects),
            "total_cases": total_cases,
            "cases": {
                case: {
                    "count": totals[case],
                    "average_per_vehicle": (
                        totals[case] / len(group_subjects) if group_subjects else 0.0
                    ),
                    "percentage": (
                        100.0 * totals[case] / total_cases if total_cases else 0.0
                    ),
                }
                for case in CASES
            },
        }
    return grouped


def plot(grouped, attack_label, output_path):
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 5.5), sharey=True)
    for axis, group_name, panel_title in zip(
        axes, ("benign", "attacker"), ("Benign vehicles", "Attacker vehicles")
    ):
        group = grouped[group_name]
        percentages = [group["cases"][case]["percentage"] for case in CASES]
        bars = axis.bar(range(len(CASES)), percentages, color=COLORS, width=0.68)
        axis.set_title(
            f"{panel_title} (n={group['vehicles']})", fontsize=13, fontweight="bold"
        )
        axis.set_xticks(range(len(CASES)), [CASE_LABELS[case] for case in CASES])
        axis.set_ylim(0, max(100, max(percentages, default=0) * 1.18))
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        for bar, case in zip(bars, CASES):
            stats = group["cases"][case]
            axis.annotate(
                f"{stats['percentage']:.1f}%\n{stats['average_per_vehicle']:.1f} avg",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 5), textcoords="offset points",
                ha="center", va="bottom", fontsize=10,
            )
    axes[0].set_ylabel("Share of reciprocity evidence cases (%)")
    figure.suptitle(
        f"Week9 Urban7p50 — {attack_label} — Reciprocity Cases (Take 1)",
        fontsize=15, fontweight="bold",
    )
    figure.text(
        0.5, 0.015,
        "Bar labels show percentage and average case count per physical vehicle over 300 seconds; no-edge pairs are excluded.",
        ha="center", fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path,
        default=Path("Simulation-Test/week9-urban7p50-test"),
    )
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    output_dir = MBD_DIR / "visualization" / "cases_graphs"
    attacks = {
        "randomPositionOffset": "Random Position Offset",
        "constantPositionOffset": "Constant Position Offset",
    }
    for attack, label in attacks.items():
        print(f"Collecting {attack}...", flush=True)
        debug_path = (
            args.root / "results" / f"json_{attack}"
            / "kalman_cam_cpm_prv_no_catch"
            / "debug.json"
        )
        grouped = collect_debug(debug_path)
        stem = f"week9-urban7p50_{attack}_reciprocity_cases_take1"
        plot(grouped, label, output_dir / f"{stem}.png")
        with (output_dir / f"{stem}.json").open("w") as handle:
            json.dump(grouped, handle, indent=2)
        print(f"  wrote {output_dir / (stem + '.png')}", flush=True)


if __name__ == "__main__":
    main()
