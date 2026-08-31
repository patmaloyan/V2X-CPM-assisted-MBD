import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from cpm_detector import (
    CPM_SENSOR_RANGE_M,
    MIN_TRUST_UPDATES,
    RECIPROCITY_NIS_THRESHOLD,
    EdgeEvidence,
    PrvDetector,
    PrvTrust,
    distance_opportunity,
    edge_evidence,
    nis_confidence,
    prv_pair_score,
)


def source_message(message_type="CAM", alias=10, position=0.0, rcv_time=1):
    return {
        "messageID": f"{message_type.lower()}_{rcv_time}",
        "rcvTime": rcv_time,
        "sendTime": rcv_time,
        "sender_id": "veh_1",
        "sender_alias": alias,
        "attacker": 0,
        "receiver": {"pos": "0,0,0"},
        "sender": {
            "pos": f"{position},0,0",
            "spd": 0.0,
            "hed": 0.0,
            "acl": 0.0,
        },
        "perceivedObjects": [],
    }


class PassCatchPrvDetector(PrvDetector):
    def __init__(self):
        super().__init__()
        self.object_calls = 0

    def catch_messages(self, messages):
        messages = messages.copy()
        messages["catch_prediction"] = 0
        messages["catch_failed_checks"] = [[] for _ in range(len(messages))]
        return messages

    def process_perceived_objects(self, cpm):
        self.object_calls += 1
        return super().process_perceived_objects(cpm)


def process_source(detector, message_type, message):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "veh_receiver.json"
        path.write_text(json.dumps([message]), encoding="utf-8")
        if message_type == "CAM":
            return detector.process_receiver(path, None, None).iloc[0]
        return detector.process_receiver(None, path, None).iloc[0]


class PrvMathTests(unittest.TestCase):
    def test_nis_confidence_uses_gate_domain_without_clamping(self):
        self.assertEqual(nis_confidence(0.0), 1.0)
        self.assertEqual(nis_confidence(RECIPROCITY_NIS_THRESHOLD), 0.0)

    def test_distance_opportunity_is_clamped(self):
        self.assertEqual(distance_opportunity(-1.0), 1.0)
        self.assertEqual(distance_opportunity(0.0), 1.0)
        self.assertEqual(distance_opportunity(CPM_SENSOR_RANGE_M), 0.0)
        self.assertEqual(distance_opportunity(CPM_SENSOR_RANGE_M + 1.0), 0.0)

    def test_prv_pair_score_cases(self):
        inbound = EdgeEvidence(0.0, 0.0, 1.0, 1.0, 0.81)
        outbound = EdgeEvidence(0.0, 0.0, 1.0, 0.5, 0.25)

        self.assertAlmostEqual(prv_pair_score(inbound, outbound), 0.9)
        self.assertEqual(prv_pair_score(inbound, None), 0.81)
        self.assertEqual(prv_pair_score(None, outbound), -0.25)
        self.assertEqual(prv_pair_score(None, None), 0.0)

    def test_edge_weight_combines_distance_and_confidence(self):
        evidence = edge_evidence(
            RECIPROCITY_NIS_THRESHOLD / 2.0,
            CPM_SENSOR_RANGE_M / 2.0,
        )
        self.assertEqual(evidence.confidence, 0.5)
        self.assertEqual(evidence.opportunity, 0.5)
        self.assertEqual(evidence.weight, 0.25)

    def test_reciprocity_matching_uses_its_own_threshold(self):
        detector = PrvDetector()
        self.assertEqual(
            detector.perceived_object_nis_threshold(), RECIPROCITY_NIS_THRESHOLD
        )
        track = object()
        detector.closest_track = lambda measurement: (
            track,
            SimpleNamespace(nis=RECIPROCITY_NIS_THRESHOLD),
        )
        self.assertEqual(len(detector.perceived_object_matches({})), 1)

        detector.closest_track = lambda measurement: (
            track,
            SimpleNamespace(nis=RECIPROCITY_NIS_THRESHOLD + 0.01),
        )
        self.assertEqual(detector.perceived_object_matches({}), [])


class FinalDecisionCommitTests(unittest.TestCase):
    def quarantined_detector(self):
        detector = PassCatchPrvDetector()
        first = source_message(rcv_time=0)
        detector.add_track(first)
        track = detector.tracks[0]
        track_id = detector.track_id(track, create_trust=True)
        detector.trust[track_id].accepted = False
        detector.trust[track_id].score = -0.25
        return detector, track

    def test_quarantine_does_not_update_known_track(self):
        detector, track = self.quarantined_detector()
        initial_state = track.filter.x.copy()

        result = process_source(
            detector, "CAM", source_message(position=5.0, rcv_time=1),
        )

        self.assertEqual(result["reason"], "prv_quarantine")
        self.assertEqual(result["kalman_prediction"], 0)
        self.assertEqual(track.last_update_time, 0)
        self.assertTrue((track.filter.x == initial_state).all())

    def test_quarantine_does_not_update_long_gap_track(self):
        detector, track = self.quarantined_detector()

        result = process_source(
            detector, "CAM", source_message(position=10.0, rcv_time=3_100_000_000),
        )

        self.assertEqual(result["reason"], "prv_quarantine")
        self.assertEqual(track.last_update_time, 0)

    def test_quarantine_does_not_commit_pseudonym_change(self):
        detector, track = self.quarantined_detector()

        result = process_source(
            detector, "CAM", source_message(alias=11, rcv_time=1),
        )

        self.assertEqual(result["reason"], "prv_quarantine")
        self.assertEqual(track.station_alias, 10)
        self.assertIs(detector.tracks_by_station_alias[10], track)
        self.assertNotIn(11, detector.tracks_by_station_alias)

    def test_quarantine_skips_cpm_objects(self):
        detector, track = self.quarantined_detector()

        result = process_source(
            detector, "CPM", source_message("CPM", rcv_time=1),
        )

        self.assertEqual(result["reason"], "missing_paired_cam")
        self.assertEqual(detector.object_calls, 0)
        self.assertEqual(track.last_update_time, 0)

    def test_final_acceptance_commits_update(self):
        detector, track = self.quarantined_detector()
        detector.trust[detector.track_ids[id(track)]].accepted = True

        result = process_source(
            detector, "CAM", source_message(position=5.0, rcv_time=1),
        )

        self.assertTrue(result["accepted"])
        self.assertEqual(track.last_update_time, 1)
        self.assertNotEqual(track.filter.x[0], 0.0)


class PrvTrustStateTests(unittest.TestCase):
    def setUp(self):
        self.detector = PrvDetector()
        self.detector.current_bucket = 10
        self.detector.trust = {
            1: PrvTrust(),
            2: PrvTrust(),
        }

    def test_updates_trust_with_normalized_interval_score(self):
        self.detector.bucket_edges[(1, 2)] = edge_evidence(0.0, 0.0)

        result = self.detector.close_current_bucket()

        self.assertEqual(result["evidence_counts"][1], 1)
        self.assertEqual(result["normalized_scores"][1], -0.5)
        self.assertAlmostEqual(self.detector.trust[1].score, -1.0 / 6.0)
        self.assertEqual(self.detector.trust[1].evidence_updates, 1)
        self.assertTrue(self.detector.trust[1].accepted)

    def test_requires_three_evidence_updates_before_quarantine(self):
        self.detector.bucket_edges[(1, 2)] = edge_evidence(0.0, 0.0)

        for expected_updates in range(1, MIN_TRUST_UPDATES):
            self.detector.close_current_bucket()
            self.assertEqual(
                self.detector.trust[1].evidence_updates, expected_updates
            )
            self.assertTrue(self.detector.trust[1].accepted)

        self.detector.close_current_bucket()
        self.assertEqual(
            self.detector.trust[1].evidence_updates, MIN_TRUST_UPDATES
        )
        self.assertFalse(self.detector.trust[1].accepted)

    def test_no_evidence_does_not_change_trust_or_state(self):
        self.detector.trust[1] = PrvTrust(
            accepted=False, score=-0.25, evidence_updates=MIN_TRUST_UPDATES
        )

        result = self.detector.close_current_bucket()

        self.assertIsNone(result["normalized_scores"][1])
        self.assertEqual(self.detector.trust[1].score, -0.25)
        self.assertFalse(self.detector.trust[1].accepted)

    def test_positive_evidence_allows_reentry_at_zero(self):
        self.detector.trust[1] = PrvTrust(
            accepted=False, score=-0.25, evidence_updates=MIN_TRUST_UPDATES
        )
        self.detector.bucket_edges[(2, 1)] = edge_evidence(0.0, 0.0)

        self.detector.close_current_bucket()

        self.assertEqual(self.detector.trust[1].score, 0.0)
        self.assertTrue(self.detector.trust[1].accepted)


if __name__ == "__main__":
    unittest.main()
