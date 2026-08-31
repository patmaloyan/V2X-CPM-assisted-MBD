"""Implements detection logic for known IDs, pseudonym changes, and new vehicles."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from data_processing import apply_catch_gate
from data_structures import Parameters
from kalman_filter import (
    CPM_ASSOCIATION_COVARIANCE,
    INITIAL_STATE_COVARIANCE,
    KalmanTrack,
    PROCESS_NOISE_INTENSITY,
    SENSOR_MEASUREMENT_COVARIANCE,
    parse_position,
    velocity_from_cam,
)


EGO_OBJECT_POSITION_THRESHOLD_M = 10
EGO_OBJECT_SPEED_THRESHOLD_MPS = 5
# 99% chi-square threshold for the four measured values [x, y, vx, vy].
NIS_THRESHOLD = 13.28
# 97.5% chi-square threshold for established four-dimensional tracks.
KNOWN_ALIAS_NIS_THRESHOLD = 11.14
MAX_KALMAN_PREDICTION_GAP_S = 4.1
MAX_KALMAN_PREDICTION_GAP_NS = int(
    MAX_KALMAN_PREDICTION_GAP_S * 1_000_000_000
)
MAX_ASSOCIATION_PREDICTION_GAP_S = 3.1
MAX_ASSOCIATION_PREDICTION_GAP_NS = int(
    MAX_ASSOCIATION_PREDICTION_GAP_S * 1_000_000_000
)
RANGE_MARGIN_M = 25.0
EGO_LOOKBACK_NS = 2_000_000_000
CPM_SENSOR_RANGE_M = 80.0
PSEUDONYM_INTERVAL_S = 50
TRACK_EXPIRY_NS = PSEUDONYM_INTERVAL_S * 1_000_000_000


def evaluation_receiver_ids(input_folder: Path, receiver_ids):
    """Return honest receivers; attacker ego files are omitted during generation."""
    ego_dir = input_folder / "ego"
    if not ego_dir.is_dir():
        return list(receiver_ids)
    return [
        receiver_id for receiver_id in receiver_ids
        if (ego_dir / f"{receiver_id}.json").is_file()
    ]


class CamOnlyKalmanDetector:
    def __init__(self, catch_params=None, catch_enabled=True):
        self.catch_params = catch_params or Parameters()
        self.catch_enabled = catch_enabled
        self.wireless_range_m = self.catch_params.MAX_PLAUSIBLE_RANGE
        self.tracks = []
        self.tracks_by_station_alias = {}
        self.initial_covariance = INITIAL_STATE_COVARIANCE.copy()
        self.measurement_noise = SENSOR_MEASUREMENT_COVARIANCE.copy()
        self.cpm_association_noise = CPM_ASSOCIATION_COVARIANCE.copy()

    def catch_messages(self, messages):
        if self.catch_enabled:
            message_types = messages.get("message_type")
            if message_types is not None and (message_types == "CPM").any():
                cams = messages[message_types == "CAM"].copy()
                checked_cams = apply_catch_gate(cams, self.catch_params)
                catch_by_pair = {
                    sender_pair_key(row): (
                        int(row["catch_prediction"]), row["catch_failed_checks"]
                    )
                    for row in checked_cams.to_dict(orient="records")
                }
                checked = messages.reset_index(drop=True).copy()
                predictions = []
                failed_checks = []
                for row in checked.to_dict(orient="records"):
                    catch_result = catch_by_pair.get(sender_pair_key(row))
                    if str(row.get("message_type", "CAM")).upper() == "CPM":
                        catch_result = catch_result or (1, ["missing_paired_cam"])
                    predictions.append(catch_result[0])
                    failed_checks.append(catch_result[1])
                checked["catch_prediction"] = predictions
                checked["catch_failed_checks"] = failed_checks
                return checked
            return apply_catch_gate(messages, self.catch_params)
        messages = messages.reset_index(drop=True).copy()
        messages["catch_prediction"] = 0
        messages["catch_failed_checks"] = [[] for _ in range(len(messages))]
        return messages

    def catch_rejection(self):
        return catch_rejection()

    def catch_debug(self, message, source_decision):
        return gate_debug(message, source_decision)

    def process_receiver(self, cam_path: Path, ego_path: Path | None):
        messages = self.catch_messages(combined_message_frame(cam_path, None))
        ego_snapshots = sorted(load_json_list(ego_path), key=lambda msg: int(msg["sendTime"]))

        rows = []
        for cam in messages.to_dict(orient="records"):
            catch_prediction = int(cam["catch_prediction"])
            source_decision = (
                self.catch_rejection()
                if catch_prediction
                else self.process_cam(cam, ego_snapshots)
            )
            rows.append({
                "receiver_id": cam_path.stem,
                "message_type": "CAM",
                "messageID": cam.get("messageID"),
                "sender_id": cam.get("sender_id"),
                "sender_alias": cam.get("sender_alias", 0),
                "sender_just_entered_communication_zone": sender_just_entered(cam),
                "receiver_just_entered_communication_zone": receiver_just_entered(cam),
                "sendTime": int(cam.get("sendTime", 0)),
                "rcvTime": int(cam.get("rcvTime", 0)),
                "attacker": int(cam.get("attacker", 0)),
                "prediction": 0 if source_decision["accepted"] else 1,
                **source_decision,
                **self.catch_debug(cam, source_decision),
            })

        return pd.DataFrame(rows)

    def process_cam(
        self, cam: dict, ego_snapshots: list[dict], commit: bool = True,
    ):
        self.prepare_tracks_for_message(cam)

        # Flowchart order: known alias, pseudonym change, then new vehicle.
        known_result = self.known_station_alias_check(cam, ego_snapshots, commit)
        if known_result is not None:
            return known_result

        pseudonym_result = self.pseudonym_change_check(cam, commit)
        if pseudonym_result is not None:
            return pseudonym_result

        return self.new_vehicle_check(cam, ego_snapshots, commit)

    def prepare_tracks_for_message(self, message):
        current_time = int(message["rcvTime"])
        active_tracks = []
        for track in self.tracks:
            if current_time - track.last_accepted_time <= TRACK_EXPIRY_NS:
                active_tracks.append(track)
            elif self.tracks_by_station_alias.get(track.station_alias) is track:
                self.tracks_by_station_alias.pop(track.station_alias)
        self.tracks = active_tracks

    def known_station_alias_check(
        self, cam: dict, ego_snapshots: list[dict], commit: bool = True,
    ):
        track = self.tracks_by_station_alias.get(int(cam["sender_alias"]))
        if track is None:
            return None

        deviation = track.deviation_against_cam(cam, self.measurement_noise)
        if (
            int(cam["rcvTime"]) - track.last_update_time
            >= MAX_KALMAN_PREDICTION_GAP_NS
        ):
            return decision(
                False, "known_alias_stale_reject", deviation.position_error,
                deviation.speed_error, track.station_alias, deviation.nis,
                KNOWN_ALIAS_NIS_THRESHOLD,
            )

        if nis_within_threshold(deviation.nis, KNOWN_ALIAS_NIS_THRESHOLD):
            if commit:
                track.update_from_cam(cam, self.measurement_noise)
            return decision(
                True, "known_alias_accept", deviation.position_error,
                deviation.speed_error, track.station_alias, deviation.nis,
                KNOWN_ALIAS_NIS_THRESHOLD,
            )

        return decision(
            False, "known_alias_reject", deviation.position_error,
            deviation.speed_error, track.station_alias, deviation.nis,
            KNOWN_ALIAS_NIS_THRESHOLD,
        )

    def pseudonym_change_check(self, cam: dict, commit: bool = True):
        best_track = None
        best_pos_error = None
        best_speed_error = None
        best_nis = None
        best_score = float("inf")

        for track in self.tracks:
            if not association_prediction_is_fresh(track, int(cam["rcvTime"])):
                continue
            deviation = track.deviation_against_cam(cam, self.measurement_noise)
            score = deviation.nis
            if score < best_score:
                best_track = track
                best_pos_error = deviation.position_error
                best_speed_error = deviation.speed_error
                best_nis = deviation.nis
                best_score = score

        if best_track is None or not nis_within_threshold(
            best_nis, KNOWN_ALIAS_NIS_THRESHOLD
        ):
            return None

        old_station_alias = best_track.station_alias
        if commit:
            self.tracks_by_station_alias.pop(old_station_alias, None)
            best_track.station_alias = int(cam["sender_alias"])
            self.tracks_by_station_alias[best_track.station_alias] = best_track
            best_track.update_from_cam(cam, self.measurement_noise)
        return decision(
            True, "pseudonym_accept", best_pos_error, best_speed_error,
            old_station_alias, best_nis, KNOWN_ALIAS_NIS_THRESHOLD,
        )

    def new_vehicle_check(
        self, cam: dict, ego_snapshots: list[dict], commit: bool = True,
    ):
        # A new ID is accepted if it just entered, appears near range edge, or is seen by the receiver.
        if sender_just_entered(cam) == 1:
            if commit:
                self.add_track(cam)
            return decision(True, "new_vehicle_sender_zone_entry_accept", None, None, None)

        if receiver_just_entered(cam) == 1:
            if commit:
                self.add_track(cam)
            return decision(True, "new_vehicle_receiver_zone_entry_accept", None, None, None)

        if self.within_wireless_margin(cam):
            if commit:
                self.add_track(cam)
            return decision(True, "new_vehicle_margin_accept", None, None, None)

        ego_match = self.find_ego_sensor_match(cam, ego_snapshots)
        if ego_match is not None:
            if commit:
                self.add_track(cam)
            return decision( True, "new_vehicle_ego_accept", ego_match["pos_error"],
                ego_match["speed_error"], ego_match["object_id"])

        return decision(False, "new_vehicle_reject", None, None, None)

    def add_track(self, cam: dict):
        track = KalmanTrack.from_cam(cam, self.initial_covariance)
        self.tracks.append(track)
        self.tracks_by_station_alias[track.station_alias] = track

    def within_wireless_margin(self, cam: dict):
        receiver_pos = parse_position(cam["receiver"]["pos"])
        sender_pos = parse_position(cam["sender"]["pos"])
        distance = float(np.linalg.norm(receiver_pos[0:2] - sender_pos[0:2]))
        return (
            self.wireless_range_m - RANGE_MARGIN_M
            <= distance
            <= self.wireless_range_m + RANGE_MARGIN_M
        )

    def find_ego_sensor_match(self, cam: dict, ego_snapshots: list[dict]):
        snapshot = latest_ego_snapshot(ego_snapshots, int(cam["rcvTime"]))
        if snapshot is None:
            return None

        cam_pos = parse_position(cam["sender"]["pos"])
        cam_speed = float(cam["sender"]["spd"])
        best_match = None
        best_score = float("inf")

        for obj in snapshot.get("perceivedObjects", []):
            object_state = relative_perceived_object_state(obj, snapshot)
            if object_state is None:
                continue

            obj_pos, obj_velocity = object_state
            pos_error = float(np.linalg.norm(cam_pos[0:2] - obj_pos[0:2]))
            speed_error = abs(cam_speed - float(np.linalg.norm(obj_velocity[0:2])))
            score = (
                pos_error / EGO_OBJECT_POSITION_THRESHOLD_M
                + speed_error / EGO_OBJECT_SPEED_THRESHOLD_MPS
            )
            if score < best_score:
                best_score = score
                best_match = {"object_id": obj.get("object_id"), "pos_error": pos_error, "speed_error": speed_error}

        if best_match and ego_errors_within_threshold(
            best_match["pos_error"], best_match["speed_error"]
        ):
            return best_match
        return None


class CamCpmKalmanDetector(CamOnlyKalmanDetector):
    """Kalman detector using one time-ordered CAM/CPM stream per receiver."""

    def __init__(self, catch_params=None, catch_enabled=True):
        super().__init__(catch_params, catch_enabled)

    def pre_source_decision(self, message: dict):
        return None

    def message_debug(self):
        return {}

    def on_perceived_object_match(
        self, cpm: dict, matched_track: KalmanTrack,
        deviation=None, perceived_object=None,
    ):
        return {}

    def perceived_object_matches(self, measurement):
        best_track, deviation = self.closest_track(measurement)
        if best_track is not None and nis_within_threshold(
            deviation.nis, self.perceived_object_nis_threshold()
        ):
            return [(best_track, deviation)]
        return []

    def perceived_object_nis_threshold(self):
        return NIS_THRESHOLD

    def process_receiver(self, cam_path: Path | None, cpm_path: Path | None, ego_path: Path | None):
        # CAMs and CPMs received by this vehicle share one chronological track history.
        messages = self.catch_messages(combined_message_frame(cam_path, cpm_path))
        ego_snapshots = sorted(load_json_list(ego_path), key=lambda msg: int(msg["sendTime"]))
        receiver_id = (cam_path or cpm_path).stem

        rows = []
        paired_source_decisions = {}
        for message in messages.to_dict(orient="records"):
            message_type = str(message["message_type"]).upper()
            pair_key = sender_pair_key(message)
            if message_type == "CPM":
                source_decision = paired_source_decisions.get(pair_key)
                if source_decision is None:
                    source_decision = decision(
                        False, "missing_paired_cam", None, None, None
                    )
                else:
                    source_decision = dict(source_decision)
            else:
                catch_prediction = int(message["catch_prediction"])
                if catch_prediction:
                    source_decision = self.catch_rejection()
                else:
                    self.prepare_tracks_for_message(message)
                    # Type 4 can reject before Kalman state is changed; type 3 returns None here.
                    source_decision = self.pre_source_decision(message)
                    if source_decision is None:
                        source_decision = self.process_cam(message, ego_snapshots)
                paired_source_decisions[pair_key] = dict(source_decision)
            object_counts = empty_object_counts()
            if message_type == "CPM":
                perceived_objects = message.get("perceivedObjects", [])
                # The orange flowchart branch runs only after the CPM source is accepted.
                if source_decision["accepted"]:
                    object_counts = self.process_perceived_objects(
                        cpm_object_message(message)
                    )
                elif isinstance(perceived_objects, list):
                    # Count skipped objects so debug totals still reconcile with raw CPM data.
                    object_counts["cpm_objects_observed"] = len(perceived_objects)
                    object_counts["cpm_objects_source_rejected"] = len(perceived_objects)
                else:
                    object_counts["cpm_objects_malformed"] = 1

            gate_fields = self.catch_debug(message, source_decision)
            if message_type == "CPM":
                gate_fields["kalman_skipped"] = True
                gate_fields["kalman_prediction"] = None
            rows.append({
                "receiver_id": receiver_id,
                "message_type": message_type,
                "messageID": message.get("messageID"),
                "sender_id": message.get("sender_id"),
                "sender_alias": message.get("sender_alias", 0),
                "sender_just_entered_communication_zone": sender_just_entered(message),
                "receiver_just_entered_communication_zone": receiver_just_entered(message),
                "sendTime": int(message.get("sendTime", 0)),
                "rcvTime": int(message.get("rcvTime", 0)),
                "attacker": int(message.get("attacker", 0)),
                "prediction": 0 if source_decision["accepted"] else 1,
                **source_decision,
                **gate_fields,
                **object_counts,
                **self.message_debug(),
            })

        return pd.DataFrame(rows)

    def process_perceived_objects(self, cpm: dict):
        counts = empty_object_counts()
        objects = cpm.get("perceivedObjects", [])
        if not isinstance(objects, list):
            counts["cpm_objects_malformed"] = 1
            return counts

        counts["cpm_objects_observed"] = len(objects)
        for perceived_object in objects:
            object_id = perceived_object.get("object_id") if isinstance(perceived_object, dict) else None
            if not perceived_object_within_sensor_range(perceived_object):
                counts["cpm_objects_out_of_range"] += 1
                counts["cpm_object_events"].append({"object_id": object_id, "action": "out_of_range"})
                continue

            measurement = perceived_object_as_cam(perceived_object, cpm)
            if measurement is None:
                counts["cpm_objects_malformed"] += 1
                counts["cpm_object_events"].append({"object_id": object_id, "action": "malformed"})
                continue

            matched_tracks = self.perceived_object_matches(measurement)
            # object_id is simulation ground truth; association uses only Kalman deviation.
            if matched_tracks:
                # CPM object data is indirect: it may confirm a track, but must not update it.
                edge_results = []
                deviations = []
                for matched_track, deviation in matched_tracks:
                    matched_track.last_accepted_time = int(cpm["rcvTime"])
                    deviations.append(deviation)
                    edge_results.append(self.on_perceived_object_match(
                        cpm, matched_track, deviation, perceived_object
                    ))
                counts["cpm_objects_matched"] += 1
                best_deviation = min(deviations, key=lambda item: item.nis)
                event = {
                    "object_id": object_id,
                    "action": "matched",
                    "pos_error": best_deviation.position_error,
                    "speed_error": best_deviation.speed_error,
                    "nis": best_deviation.nis,
                    "nis_threshold": self.perceived_object_nis_threshold(),
                    "normalized_nis": (
                        best_deviation.nis
                        / self.perceived_object_nis_threshold()
                    ),
                }
                if len(edge_results) == 1:
                    event.update(edge_results[0])
                else:
                    event["edge_added"] = any(result.get("edge_added", False) for result in edge_results)
                    event["edges_added"] = sum(
                        result.get("edge_added", False) for result in edge_results
                    )
                counts["cpm_object_events"].append(event)
                continue

            if self.add_anonymous_object_track(measurement):
                counts["cpm_objects_initialized"] += 1
                action = "initialized"
            else:
                counts["cpm_objects_untracked"] += 1
                action = "untracked"
            counts["cpm_object_events"].append({
                "object_id": object_id, "action": action
            })

        return counts

    def closest_track(self, measurement: dict):
        best_track = None
        best_deviation = None
        best_score = float("inf")
        for track in self.tracks:
            if not association_prediction_is_fresh(
                track, int(measurement["rcvTime"])
            ):
                continue
            deviation = track.deviation_against_cam(
                measurement, self.cpm_association_noise
            )
            score = deviation.nis
            if score < best_score:
                best_track = track
                best_deviation = deviation
                best_score = score
        return best_track, best_deviation

    def add_anonymous_object_track(self, measurement: dict):
        # Do not expose the CPM ground-truth object_id as an observable station identity.
        track = KalmanTrack.from_cam(measurement, self.initial_covariance)
        self.tracks.append(track)
        return True

def process_kalman_folder(
    input_folder: Path, catch_params: Parameters, catch_enabled=True,
):
    cam_dir = input_folder / "cam"
    ego_dir = input_folder / "ego"

    receiver_results = []
    cam_paths = {path.stem: path for path in cam_dir.glob("*.json")}
    receiver_ids = evaluation_receiver_ids(input_folder, sorted(cam_paths))
    for receiver_id in receiver_ids:
        cam_path = cam_paths[receiver_id]
        ego_path = ego_dir / cam_path.name if ego_dir.is_dir() else None
        detector = CamOnlyKalmanDetector(catch_params, catch_enabled)
        receiver_results.append(detector.process_receiver(cam_path, ego_path))

    if not receiver_results:
        raise ValueError(f"No CAM JSON files found in {cam_dir}")

    results = pd.concat(receiver_results, ignore_index=True)
    metrics = calculate_metrics(results)
    add_catch_metrics(metrics, results)
    metrics["wireless_range_m"] = catch_params.MAX_PLAUSIBLE_RANGE
    metrics["range_margin_m"] = RANGE_MARGIN_M
    metrics["nis_threshold"] = NIS_THRESHOLD
    metrics["known_alias_nis_threshold"] = KNOWN_ALIAS_NIS_THRESHOLD
    metrics["max_kalman_prediction_gap_s"] = MAX_KALMAN_PREDICTION_GAP_S
    metrics["max_association_prediction_gap_s"] = MAX_ASSOCIATION_PREDICTION_GAP_S
    metrics["process_noise_intensity"] = PROCESS_NOISE_INTENSITY
    metrics["initial_covariance_diag"] = np.diag(INITIAL_STATE_COVARIANCE).tolist()
    metrics["measurement_noise_diag"] = np.diag(SENSOR_MEASUREMENT_COVARIANCE).tolist()
    metrics["cpm_association_noise_diag"] = np.diag(CPM_ASSOCIATION_COVARIANCE).tolist()
    metrics["process_noise_model"] = "continuous_white_acceleration"
    metrics["total_messages"] = int(len(results))
    metrics["evaluated_receivers"] = len(receiver_ids)
    metrics["excluded_attacker_receivers"] = len(cam_paths) - len(receiver_ids)
    metrics["catch_enabled"] = catch_enabled
    return metrics, results


def process_cam_cpm_kalman_folder(
    input_folder: Path, catch_params: Parameters,
    detector_factory=CamCpmKalmanDetector, catch_enabled=True,
):
    cam_dir = input_folder / "cam"
    cpm_dir = input_folder / "cpm"
    ego_dir = input_folder / "ego"
    if not cam_dir.is_dir() or not cpm_dir.is_dir():
        raise ValueError(f"Type 3 expects CAM and CPM folders at {cam_dir} and {cpm_dir}")

    cam_paths = {path.stem: path for path in cam_dir.glob("*.json")}
    cpm_paths = {path.stem: path for path in cpm_dir.glob("*.json")}
    all_receiver_ids = sorted(set(cam_paths) | set(cpm_paths))
    if not all_receiver_ids:
        raise ValueError(f"No CAM or CPM JSON files found in {input_folder}")
    receiver_ids = evaluation_receiver_ids(input_folder, all_receiver_ids)
    if not receiver_ids:
        raise ValueError(f"No non-attacker receivers found in {input_folder}")

    receiver_results = []
    for receiver_id in receiver_ids:
        ego_path = ego_dir / f"{receiver_id}.json" if ego_dir.is_dir() else None
        # Kalman state is local to a receiver and must never leak between vehicles.
        detector = detector_factory(catch_params, catch_enabled)
        receiver_results.append(detector.process_receiver(
            cam_paths.get(receiver_id), cpm_paths.get(receiver_id), ego_path
        ))

    results = pd.concat(receiver_results, ignore_index=True)
    # CPMs still influence the shared tracks, but detector accuracy is measured
    # only on CAM decisions so types 2-5 use the same evaluation population.
    evaluated_results = results[results["message_type"] == "CAM"]
    metrics = calculate_metrics(evaluated_results)
    add_catch_metrics(metrics, evaluated_results)
    metrics["wireless_range_m"] = catch_params.MAX_PLAUSIBLE_RANGE
    metrics["range_margin_m"] = RANGE_MARGIN_M
    metrics["cpm_sensor_range_m"] = CPM_SENSOR_RANGE_M
    metrics["nis_threshold"] = NIS_THRESHOLD
    metrics["known_alias_nis_threshold"] = KNOWN_ALIAS_NIS_THRESHOLD
    metrics["max_kalman_prediction_gap_s"] = MAX_KALMAN_PREDICTION_GAP_S
    metrics["max_association_prediction_gap_s"] = MAX_ASSOCIATION_PREDICTION_GAP_S
    metrics["process_noise_intensity"] = PROCESS_NOISE_INTENSITY
    metrics["initial_covariance_diag"] = np.diag(INITIAL_STATE_COVARIANCE).tolist()
    metrics["measurement_noise_diag"] = np.diag(SENSOR_MEASUREMENT_COVARIANCE).tolist()
    metrics["cpm_association_noise_diag"] = np.diag(CPM_ASSOCIATION_COVARIANCE).tolist()
    metrics["process_noise_model"] = "continuous_white_acceleration"
    metrics["total_messages"] = int(len(evaluated_results))
    metrics["cam_messages"] = int((results["message_type"] == "CAM").sum())
    metrics["cpm_messages"] = int((results["message_type"] == "CPM").sum())
    metrics["evaluated_receivers"] = len(receiver_ids)
    metrics["excluded_attacker_receivers"] = len(all_receiver_ids) - len(receiver_ids)
    metrics["catch_enabled"] = catch_enabled
    return metrics, results


def combined_message_frame(cam_path: Path | None, cpm_path: Path | None):
    records = []
    for message_type, path in (("CAM", cam_path), ("CPM", cpm_path)):
        for message in load_json_list(path):
            row = dict(message)
            row["message_type"] = message_type
            records.append(row)
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    frame["rcvTime"] = frame["rcvTime"].astype("int64")
    frame["sendTime"] = frame["sendTime"].astype("int64")
    cam_receive_times = {
        sender_pair_key(row): int(row["rcvTime"])
        for row in frame[frame["message_type"] == "CAM"].to_dict(orient="records")
    }
    effective_times = []
    paired_cams = []
    for row in frame.to_dict(orient="records"):
        paired_cam_time = cam_receive_times.get(sender_pair_key(row))
        is_paired_cpm = row["message_type"] == "CPM" and paired_cam_time is not None
        paired_cams.append(is_paired_cpm)
        effective_times.append(
            max(int(row["rcvTime"]), paired_cam_time)
            if is_paired_cpm else int(row["rcvTime"])
        )
    frame["paired_cam"] = paired_cams
    frame["effectiveRcvTime"] = effective_times
    frame["message_type_order"] = frame["message_type"].map({"CAM": 0, "CPM": 1})
    # Buffer an early CPM until its matching CAM arrives, then process CAM first.
    return frame.sort_values(
        ["effectiveRcvTime", "sendTime", "message_type_order", "messageID"],
        kind="mergesort", ignore_index=True,
    ).drop(columns=["message_type_order"])


def sender_pair_key(message: dict):
    return int(message.get("sender_alias", 0)), int(message.get("sendTime", 0))


def cpm_object_message(message: dict):
    effective = dict(message)
    effective["rcvTime"] = int(
        message.get("effectiveRcvTime", message.get("rcvTime", 0))
    )
    return effective


def relative_perceived_object_state(perceived_object: dict, reference: dict):
    if not isinstance(perceived_object, dict):
        return None
    try:
        reference_position = parse_position(reference["sender"]["pos"])
        relative_position = parse_position(perceived_object["rel_pos"])
        object_position = reference_position + relative_position

        if "rel_vel" in perceived_object:
            relative_velocity = parse_position(perceived_object["rel_vel"])[0:2]
            object_velocity = velocity_from_cam(reference) + relative_velocity
        else:
            # Transitional support for existing simulations where object speed and
            # heading are ground-referenced but global position is not required.
            object_velocity = velocity_from_cam({"sender": perceived_object})
    except (KeyError, TypeError, ValueError, IndexError):
        return None

    return object_position, object_velocity


def perceived_object_as_cam(perceived_object: dict, cpm: dict):
    object_state = relative_perceived_object_state(perceived_object, cpm)
    if object_state is None:
        return None
    object_position, object_velocity = object_state
    speed = float(np.linalg.norm(object_velocity[0:2]))
    heading = (
        math.degrees(math.atan2(object_velocity[0], object_velocity[1])) % 360.0
        if speed > 0.0 else 0.0
    )
    return {
        "sender_alias": 0,
        "rcvTime": int(cpm["rcvTime"]),
        "sender": {
            "pos": ",".join(str(value) for value in object_position),
            "spd": speed,
            "hed": heading,
        },
    }


def perceived_object_within_sensor_range(perceived_object: dict):
    try:
        relative_position = parse_position(perceived_object["rel_pos"])
    except (KeyError, TypeError, ValueError, IndexError):
        return False
    return float(np.linalg.norm(relative_position[0:2])) <= CPM_SENSOR_RANGE_M


def empty_object_counts():
    return {
        "cpm_objects_observed": 0,
        "cpm_objects_matched": 0,
        "cpm_objects_initialized": 0,
        "cpm_objects_untracked": 0,
        "cpm_objects_out_of_range": 0,
        "cpm_objects_source_rejected": 0,
        "cpm_objects_malformed": 0,
        "cpm_object_events": [],
    }


def calculate_metrics(results: pd.DataFrame, prediction_column="prediction"):
    prediction = results[prediction_column]
    tp = ((results["attacker"] == 1) & (prediction == 1)).sum()
    tn = ((results["attacker"] == 0) & (prediction == 0)).sum()
    fp = ((results["attacker"] == 0) & (prediction == 1)).sum()
    fn = ((results["attacker"] == 1) & (prediction == 0)).sum()
    return {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}


def add_catch_metrics(metrics, results):
    catch_metrics = calculate_metrics(results, "catch_prediction")
    denominator = catch_metrics["fp"] + catch_metrics["tn"]
    catch_metrics["false_positive_rate"] = (
        catch_metrics["fp"] / denominator if denominator else 0.0
    )
    failed_checks = results["catch_failed_checks"].explode().dropna()
    metrics["catch_metrics"] = catch_metrics
    metrics["catch_check_activations"] = {
        str(name): int(count)
        for name, count in failed_checks.value_counts().items()
    }
    metrics["kalman_skipped"] = int(results["kalman_skipped"].sum())


def catch_rejection():
    return decision(False, "catch_reject", None, None, None)


def gate_debug(message, source_decision):
    skipped = bool(message["catch_prediction"])
    return {
        "catch_prediction": int(message["catch_prediction"]),
        "catch_failed_checks": message["catch_failed_checks"],
        "kalman_skipped": skipped,
        "kalman_prediction": None if skipped else int(not source_decision["accepted"]),
    }


def load_json_list(path: Path | None):
    if path is None or not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def latest_ego_snapshot(ego_snapshots: list[dict], rcv_time: int):
    latest = None
    for snapshot in ego_snapshots:
        send_time = int(snapshot.get("sendTime", 0))
        if send_time > rcv_time:
            break
        if rcv_time - send_time <= EGO_LOOKBACK_NS:
            latest = snapshot
    return latest


def nis_within_threshold(nis, threshold=NIS_THRESHOLD):
    return nis <= threshold


def association_prediction_is_fresh(track, time_ns):
    return int(time_ns) - track.last_update_time < MAX_ASSOCIATION_PREDICTION_GAP_NS


def ego_errors_within_threshold(pos_error, speed_error):
    return (
        pos_error <= EGO_OBJECT_POSITION_THRESHOLD_M
        and speed_error <= EGO_OBJECT_SPEED_THRESHOLD_MPS
    )


def sender_just_entered(cam: dict):
    return int(cam.get(
        "just_entered_communication_zone",
        cam.get("just_entered_communication_zone_cpm", cam.get("just_entered_communication_zone_cam", 0)),
    ))


def receiver_just_entered(cam: dict):
    return int(cam.get("receiver", {}).get("just_entered_communication_zone", 0))


def decision(
    accepted: bool, reason: str, pos_error, speed_error, matched_id, nis=None,
    nis_threshold=NIS_THRESHOLD,
):
    return {
        "accepted": accepted,
        "reason": reason,
        "pos_error": pos_error,
        "speed_error": speed_error,
        "nis": nis,
        "nis_threshold": nis_threshold if nis is not None else None,
        "normalized_nis": nis / nis_threshold if nis is not None else None,
        "matched_id": matched_id,
    }
