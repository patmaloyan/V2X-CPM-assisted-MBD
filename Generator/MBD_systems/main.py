import argparse
import json
from pathlib import Path

import pandas as pd

from data_structures import Parameters
from catch_profiles import load_catch_profile
from cpm_detector import (
    MIN_TRUST_UPDATES,
    PrvDetector,
    RECIPROCITY_NIS_THRESHOLD,
    TwoEdgeReciprocityDetector,
)
from kalman_detector import (
    CamCpmKalmanDetector,
    process_cam_cpm_kalman_folder,
    process_kalman_folder,
)


BASE_KALMAN_METRICS = (
    "wireless_range_m",
    "range_margin_m",
    "nis_threshold",
    "known_alias_nis_threshold",
    "max_kalman_prediction_gap_s",
    "max_association_prediction_gap_s",
    "process_noise_intensity",
    "total_messages",
    "evaluated_receivers",
    "excluded_attacker_receivers",
)
CPM_KALMAN_METRICS = BASE_KALMAN_METRICS + (
    "cpm_sensor_range_m",
    "cam_messages",
    "cpm_messages",
)
RECIPROCITY_METRICS = CPM_KALMAN_METRICS + ("reciprocity_nis_threshold",)

# A single table defines every Kalman-family CLI type. ``detector=None`` is the
# CAM-only path; all other entries share the CAM+CPM folder processor.
KALMAN_DETECTORS = {
    2: {
        "detector": None,
        "result_name": "kalman_cam_only",
        "metric_keys": BASE_KALMAN_METRICS,
    },
    3: {
        "detector": CamCpmKalmanDetector,
        "result_name": "kalman_cam_cpm",
        "metric_keys": CPM_KALMAN_METRICS,
    },
    4: {
        "detector": TwoEdgeReciprocityDetector,
        "result_name": "kalman_cam_cpm_enhanced_two_edges",
        "metric_keys": CPM_KALMAN_METRICS,
    },
    6: {
        "detector": PrvDetector,
        "result_name": "kalman_cam_cpm_prv",
        "metric_keys": RECIPROCITY_METRICS + ("minimum_trust_updates",),
        "extra_metrics": {
            "reciprocity_nis_threshold": RECIPROCITY_NIS_THRESHOLD,
            "minimum_trust_updates": MIN_TRUST_UPDATES,
        },
    },
}
CATCH_PARAMETER_FIELDS = {
    "mpr": "MAX_PLAUSIBLE_RANGE",
    "msar": "MAX_SA_RANGE",
    "mpdn": "MAX_PLAUSIBLE_DIST_NEGATIVE",
    "mps": "MAX_PLAUSIBLE_SPEED",
    "mpa": "MAX_PLAUSIBLE_ACCEL",
    "mpd": "MAX_PLAUSIBLE_DECEL",
    "mhc": "MAX_HEADING_CHANGE",
    "mdi": "MAX_DELTA_INTERSECTION",
    "mtd": "MAX_TIME_DELTA",
    "pht": "POS_HEADING_TIME",
    "mmru": "MAX_MGT_RNG_UP",
    "mmrd": "MAX_MGT_RNG_DOWN",
    "msat": "MAX_SA_TIME",
    "mnrs": "MAX_NON_ROUTE_SPEED",
}


def evaluate_predictions(scenario_stats):
    total_tp = sum(s.get('tp', 0) for s in scenario_stats)
    total_tn = sum(s.get('tn', 0) for s in scenario_stats)
    total_fp = sum(s.get('fp', 0) for s in scenario_stats)
    total_fn = sum(s.get('fn', 0) for s in scenario_stats)

    total_messages = total_tp + total_tn + total_fp + total_fn

    aggregated_metrics = {
        'tp': total_tp,
        'tn': total_tn,
        'fp': total_fp,
        'fn': total_fn,
        'accuracy': (total_tp + total_tn) / total_messages if total_messages > 0 else 0,
        'precision': total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0,
        'recall': total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0,
        'false_positive_rate': total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0,
    }
    aggregated_metrics['f1'] = (2 * aggregated_metrics['precision'] * aggregated_metrics['recall'] /
                                (aggregated_metrics['precision'] + aggregated_metrics['recall'])
                                if (aggregated_metrics['precision'] + aggregated_metrics['recall']) > 0 else 0)
    return aggregated_metrics


def add_catch_output(aggregated_metrics, metrics, profile, params):
    aggregated_metrics["catch_enabled"] = metrics["catch_enabled"]
    aggregated_metrics["initial_covariance_diag"] = metrics["initial_covariance_diag"]
    aggregated_metrics["measurement_noise_diag"] = metrics["measurement_noise_diag"]
    aggregated_metrics["cpm_association_noise_diag"] = metrics[
        "cpm_association_noise_diag"
    ]
    aggregated_metrics["process_noise_model"] = metrics["process_noise_model"]
    aggregated_metrics["catch_profile"] = profile if metrics["catch_enabled"] else None
    aggregated_metrics["catch_metrics"] = metrics["catch_metrics"]
    aggregated_metrics["catch_check_activations"] = metrics["catch_check_activations"]
    aggregated_metrics["kalman_skipped"] = metrics["kalman_skipped"]
    aggregated_metrics["parameters"] = vars(params) if metrics["catch_enabled"] else None
    aggregated_metrics["position_plausibility_check_enabled"] = (
        params.POSITION_PLAUSIBILITY_ENABLED if metrics["catch_enabled"] else None
    )


def result_directory(input_folder: Path, detection_type: str):
    """Store results as results/<attack_type>/<detection_type>/."""
    output_dir = input_folder.parent / "results" / input_folder.name / detection_type
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_kalman_detector(
    input_folder, detection_type, params, catch_enabled, catch_profile
):
    """Run one configured Kalman-family detector and prepare its CLI output."""
    config = KALMAN_DETECTORS[detection_type]
    detector = config["detector"]
    if detector is None:
        metrics, debug_results = process_kalman_folder(
            input_folder, params, catch_enabled
        )
    else:
        metrics, debug_results = process_cam_cpm_kalman_folder(
            input_folder, params, detector, catch_enabled
        )

    metrics.update(config.get("extra_metrics", {}))
    aggregated_metrics = evaluate_predictions([metrics])
    for key in config["metric_keys"]:
        aggregated_metrics[key] = metrics.get(key)
    add_catch_output(aggregated_metrics, metrics, catch_profile, params)
    return config, aggregated_metrics, debug_results


def save_kalman_results(
    input_folder, config, aggregated_metrics, debug_results, catch_enabled
):
    result_name = config["result_name"] + ("" if catch_enabled else "_no_catch")
    output_dir = result_directory(input_folder, result_name)
    output_file = output_dir / "predicted.json"
    debug_file = output_dir / "debug.json"
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(aggregated_metrics, file, indent=4)
    with open(debug_file, "w", encoding="utf-8") as file:
        json.dump(
            debug_results.where(pd.notnull(debug_results), None).to_dict(
                orient="records"
            ),
            file,
            indent=4,
        )
    print(f"Saved in {output_file}")
    print(f"Saved debug in {debug_file}")


def load_parameters(args):
    if args.train == 1:
        values = {
            name: getattr(args, option)
            for option, name in CATCH_PARAMETER_FIELDS.items()
        }
        return Parameters(**values)
    if args.parameter:
        with open(args.parameter, "r", encoding="utf-8") as file:
            values = json.load(file)["parameters"]
        return Parameters(**{
            name: values[option]
            for option, name in CATCH_PARAMETER_FIELDS.items()
        })
    return load_catch_profile(args.catch_profile)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", help="Pfad zu den Eingabedateien", required=True)
    parser.add_argument(
        "--type",
        type=int,
        choices=sorted(KALMAN_DETECTORS),
        help=(
            "2 = CAM-only Kalman, 3 = CAM+CPM Kalman, "
            "4 = two-edge reciprocal CPM Kalman, 6 = PRV"
        ),
        required=True,
    )
    parser.add_argument(
        "--catch-profile",
        choices=["urban-low", "urban-high", "highway-low", "highway-high"],
        default="urban-low",
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
    parser.add_argument("--train", type=float)
    parser.add_argument("--parameter")
    for option in CATCH_PARAMETER_FIELDS:
        parser.add_argument(f"--{option}", type=float)
    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    catch_enabled = not args.no_catch

    params = load_parameters(args)
    params.POSITION_PLAUSIBILITY_ENABLED = not args.no_pos_check

    detection_type = int(args.type)
    config, aggregated_metrics, debug_results = run_kalman_detector(
        input_folder, detection_type, params, catch_enabled, args.catch_profile
    )
    print(aggregated_metrics["f1"])
    if args.train == 0:
        save_kalman_results(
            input_folder, config, aggregated_metrics, debug_results, catch_enabled
        )

if __name__ == "__main__":
    main()
