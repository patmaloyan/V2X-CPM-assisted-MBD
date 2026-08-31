import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import numpy as np

from cpm_detector import TwoEdgeReciprocityDetector
from data_structures import Parameters
from kalman_detector import (
    KNOWN_ALIAS_NIS_THRESHOLD,
    MAX_ASSOCIATION_PREDICTION_GAP_NS,
    MAX_ASSOCIATION_PREDICTION_GAP_S,
    MAX_KALMAN_PREDICTION_GAP_NS,
    MAX_KALMAN_PREDICTION_GAP_S,
    NIS_THRESHOLD,
    PSEUDONYM_INTERVAL_S,
    TRACK_EXPIRY_NS,
    CamCpmKalmanDetector,
    CamOnlyKalmanDetector,
    association_prediction_is_fresh,
    decision,
    evaluation_receiver_ids,
)


def message(message_type="CAM", sender_x=0.0):
    return {
        "type": message_type,
        "messageID": f"{message_type.lower()}_1",
        "rcvTime": 1,
        "sendTime": 1,
        "sender_id": "veh_1",
        "sender_alias": 10,
        "attacker": 0,
        "receiver": {"pos": "0,0,0"},
        "sender": {
            "pos": f"{sender_x},0,0", "spd": 0.0, "hed": 0.0, "acl": 0.0,
        },
        "perceivedObjects": [{}],
    }


class StubCamDetector(CamOnlyKalmanDetector):
    def __init__(self, catch_prediction):
        super().__init__(Parameters(MAX_PLAUSIBLE_RANGE=100.0))
        self.catch_prediction = catch_prediction
        self.kalman_calls = 0

    def catch_messages(self, messages):
        messages = messages.copy()
        messages["catch_prediction"] = self.catch_prediction
        messages["catch_failed_checks"] = [
            ["range_plausibility"] if self.catch_prediction else []
            for _ in range(len(messages))
        ]
        return messages

    def process_cam(self, cam, ego_snapshots):
        self.kalman_calls += 1
        return decision(True, "stub_accept", None, None, None)


class StubCpmDetector(CamCpmKalmanDetector):
    def __init__(self):
        super().__init__(Parameters())
        self.object_calls = 0

    def catch_messages(self, messages):
        messages = messages.copy()
        messages["catch_prediction"] = 1
        messages["catch_failed_checks"] = [["range_plausibility"]]
        return messages

    def process_perceived_objects(self, cpm):
        self.object_calls += 1
        return super().process_perceived_objects(cpm)


class StubPairedDetector(CamCpmKalmanDetector):
    def __init__(self):
        super().__init__(Parameters(), catch_enabled=False)
        self.kalman_calls = 0
        self.object_calls = 0

    def process_cam(self, cam, ego_snapshots):
        self.kalman_calls += 1
        return decision(True, "stub_accept", None, None, None)

    def process_perceived_objects(self, cpm):
        self.object_calls += 1
        return super().process_perceived_objects(cpm)


class CatchKalmanGateTests(unittest.TestCase):
    def test_cpm_association_includes_two_vehicle_gnss_biases(self):
        detector = CamCpmKalmanDetector(Parameters(), catch_enabled=False)
        detector.add_track(message(sender_x=0.0))
        track = detector.tracks[0]
        track.filter.P = np.zeros((4, 4))

        measurement = message("CPM", sender_x=8.0)
        measurement["rcvTime"] = 2
        direct_nis = track.deviation_against_cam(
            measurement, detector.measurement_noise
        ).nis
        cpm_nis = track.deviation_against_cam(
            measurement, detector.cpm_association_noise
        ).nis
        matches = detector.perceived_object_matches(measurement)

        self.assertGreater(direct_nis, KNOWN_ALIAS_NIS_THRESHOLD)
        self.assertLess(cpm_nis, NIS_THRESHOLD)
        self.assertIs(matches[0][0], track)

    def test_pseudonym_association_uses_known_alias_threshold(self):
        detector = CamOnlyKalmanDetector(Parameters(), catch_enabled=False)
        detector.add_track(message(sender_x=0.0))

        outside_known_gate = message(sender_x=10.0)
        outside_known_gate["sender_alias"] = 11
        outside_known_gate["rcvTime"] = 2
        deviation = detector.tracks[0].deviation_against_cam(
            outside_known_gate, detector.measurement_noise
        )

        self.assertGreater(deviation.nis, KNOWN_ALIAS_NIS_THRESHOLD)
        self.assertLess(deviation.nis, NIS_THRESHOLD)
        self.assertIsNone(detector.pseudonym_change_check(outside_known_gate))

        inside_known_gate = message(sender_x=8.0)
        inside_known_gate["sender_alias"] = 11
        inside_known_gate["rcvTime"] = 2
        result = detector.pseudonym_change_check(inside_known_gate)

        self.assertEqual(result["reason"], "pseudonym_accept")
        self.assertEqual(result["nis_threshold"], KNOWN_ALIAS_NIS_THRESHOLD)

    def test_no_catch_bypass_marks_every_message_as_passed(self):
        detector = CamOnlyKalmanDetector(Parameters(), catch_enabled=False)
        result = detector.catch_messages(pd.DataFrame([message()])).iloc[0]

        self.assertEqual(result["catch_prediction"], 0)
        self.assertEqual(result["catch_failed_checks"], [])

    def test_no_catch_still_uses_selected_profile_mpr(self):
        low_profile = CamOnlyKalmanDetector(
            Parameters(MAX_PLAUSIBLE_RANGE=10.0), catch_enabled=False
        )
        high_profile = CamOnlyKalmanDetector(
            Parameters(MAX_PLAUSIBLE_RANGE=1000.0), catch_enabled=False
        )

        self.assertEqual(low_profile.wireless_range_m, 10.0)
        self.assertEqual(high_profile.wireless_range_m, 1000.0)

    def test_type_four_requires_two_unreciprocated_edges(self):
        detector = TwoEdgeReciprocityDetector(Parameters(), catch_enabled=False)
        detector.edges = {10: {20: [0]}}
        source = message()
        source["rcvTime"] = 2_000_000_000

        self.assertIsNone(detector.pre_source_decision(source))

        detector.edges[10][30] = [0]
        result = detector.pre_source_decision(source)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "reciprocity_reject")

    def test_evaluation_uses_only_receivers_with_ego_files(self):
        with tempfile.TemporaryDirectory() as directory:
            input_folder = Path(directory)
            ego_dir = input_folder / "ego"
            ego_dir.mkdir()
            (ego_dir / "veh_honest.json").write_text("[]", encoding="utf-8")

            result = evaluation_receiver_ids(
                input_folder, ["veh_attacker", "veh_honest"]
            )

        self.assertEqual(result, ["veh_honest"])

    def run_cam(self, detector):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "veh_1.json"
            path.write_text(json.dumps([message()]), encoding="utf-8")
            return detector.process_receiver(path, None)

    def test_catch_rejection_skips_kalman(self):
        detector = StubCamDetector(1)
        result = self.run_cam(detector).iloc[0]

        self.assertEqual(detector.kalman_calls, 0)
        self.assertTrue(result["kalman_skipped"])
        self.assertIsNone(result["kalman_prediction"])
        self.assertEqual(result["reason"], "catch_reject")

    def test_catch_pass_runs_existing_kalman_path(self):
        detector = StubCamDetector(0)
        result = self.run_cam(detector).iloc[0]

        self.assertEqual(detector.kalman_calls, 1)
        self.assertFalse(result["kalman_skipped"])
        self.assertEqual(result["kalman_prediction"], 0)
        self.assertEqual(result["reason"], "stub_accept")

    def test_rejected_cpm_skips_perceived_objects(self):
        detector = StubCpmDetector()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "veh_1.json"
            path.write_text(json.dumps([message("CPM")]), encoding="utf-8")
            result = detector.process_receiver(None, path, None).iloc[0]

        self.assertEqual(detector.object_calls, 0)
        self.assertEqual(result["cpm_objects_source_rejected"], 1)

    def test_paired_cpm_uses_cam_decision_without_second_kalman_update(self):
        detector = StubPairedDetector()
        cam = message("CAM")
        cam["rcvTime"] = 2
        cpm = message("CPM")
        cpm["rcvTime"] = 1
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            cam_path = directory / "cam.json"
            cpm_path = directory / "cpm.json"
            cam_path.write_text(json.dumps([cam]), encoding="utf-8")
            cpm_path.write_text(json.dumps([cpm]), encoding="utf-8")
            result = detector.process_receiver(cam_path, cpm_path, None)

        self.assertEqual(result["message_type"].tolist(), ["CAM", "CPM"])
        self.assertEqual(detector.kalman_calls, 1)
        self.assertEqual(detector.object_calls, 1)
        self.assertEqual(result.iloc[1]["reason"], "stub_accept")
        self.assertTrue(result.iloc[1]["kalman_skipped"])

    def test_wireless_margin_uses_catch_range(self):
        detector = CamOnlyKalmanDetector(
            Parameters(MAX_PLAUSIBLE_RANGE=100.0)
        )
        self.assertTrue(detector.within_wireless_margin(message(sender_x=80)))
        self.assertFalse(detector.within_wireless_margin(message(sender_x=74)))
        self.assertFalse(detector.within_wireless_margin(message(sender_x=126)))

    def test_stale_known_alias_is_rejected_without_update(self):
        detector = CamOnlyKalmanDetector(Parameters())
        first = message(sender_x=0.0)
        first["rcvTime"] = 0
        detector.add_track(first)
        track = detector.tracks[0]

        current = message(sender_x=0.0)
        current["rcvTime"] = MAX_KALMAN_PREDICTION_GAP_NS
        result = detector.known_station_alias_check(current, [])

        self.assertEqual(MAX_KALMAN_PREDICTION_GAP_S, 4.1)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "known_alias_stale_reject")
        self.assertEqual(track.last_update_time, 0)

    def test_long_gap_track_is_not_used_for_pseudonym_matching(self):
        detector = CamOnlyKalmanDetector(Parameters())
        first = message(sender_x=0.0)
        first["rcvTime"] = 0
        detector.add_track(first)
        track = detector.tracks[0]

        changed = message(sender_x=0.0)
        changed["sender_alias"] = 11
        changed["rcvTime"] = MAX_ASSOCIATION_PREDICTION_GAP_NS

        self.assertEqual(MAX_ASSOCIATION_PREDICTION_GAP_S, 3.1)
        self.assertFalse(
            association_prediction_is_fresh(track, changed["rcvTime"])
        )
        self.assertIsNone(detector.pseudonym_change_check(changed))
        self.assertEqual(track.station_alias, first["sender_alias"])

    def test_long_gap_track_is_not_used_for_cpm_association(self):
        measurement = message(sender_x=0.0)
        measurement["rcvTime"] = MAX_ASSOCIATION_PREDICTION_GAP_NS

        detector = CamCpmKalmanDetector(Parameters())
        first = message(sender_x=0.0)
        first["rcvTime"] = 0
        detector.add_track(first)
        self.assertEqual(detector.closest_track(measurement), (None, None))

        two_edge = TwoEdgeReciprocityDetector(Parameters())
        two_edge.add_track(first)
        self.assertEqual(two_edge.perceived_object_matches(measurement), [])

    def test_track_expiry_matches_pseudonym_interval(self):
        detector = CamOnlyKalmanDetector(Parameters())
        first = message(sender_x=0.0)
        first["rcvTime"] = 0
        detector.add_track(first)

        self.assertEqual(PSEUDONYM_INTERVAL_S, 50)
        self.assertEqual(TRACK_EXPIRY_NS, 50_000_000_000)

        at_limit = message(sender_x=0.0)
        at_limit["rcvTime"] = TRACK_EXPIRY_NS
        detector.prepare_tracks_for_message(at_limit)
        self.assertEqual(len(detector.tracks), 1)

        past_limit = message(sender_x=0.0)
        past_limit["rcvTime"] = TRACK_EXPIRY_NS + 1
        detector.prepare_tracks_for_message(past_limit)
        self.assertEqual(detector.tracks, [])

    def test_known_alias_uses_standard_nis_threshold(self):
        detector = CamOnlyKalmanDetector(Parameters())
        first = message(sender_x=0.0)
        detector.add_track(first)

        current = message(sender_x=25.0)
        current["rcvTime"] = 2
        result = detector.known_station_alias_check(current, [])

        self.assertEqual(KNOWN_ALIAS_NIS_THRESHOLD, 11.14)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "known_alias_reject")
        self.assertEqual(result["nis_threshold"], KNOWN_ALIAS_NIS_THRESHOLD)
        self.assertAlmostEqual(
            result["normalized_nis"], result["nis"] / KNOWN_ALIAS_NIS_THRESHOLD
        )

        detector = CamOnlyKalmanDetector(Parameters())
        detector.add_track(first)
        current["sender_alias"] = 11
        self.assertIsNone(detector.pseudonym_change_check(current))

    def test_second_just_entered_message_uses_known_alias_check(self):
        detector = CamOnlyKalmanDetector(Parameters())
        first = message(sender_x=0.0)
        first["just_entered_communication_zone"] = 1
        first_result = detector.process_cam(first, [])
        self.assertEqual(first_result["reason"], "new_vehicle_sender_zone_entry_accept")

        second = message(sender_x=100.0)
        second["rcvTime"] = 1_000_000_001
        second["just_entered_communication_zone"] = 1
        second_result = detector.process_cam(second, [])

        self.assertEqual(second_result["reason"], "known_alias_reject")
        self.assertFalse(second_result["accepted"])


if __name__ == "__main__":
    unittest.main()
