import copy
import json
import unittest

import numpy as np

from tools.densevideo import highmotion_v2_scoring as scoring


class HighMotionV2ScoringTests(unittest.TestCase):
    def test_uniform_indices_match_original_numpy_rule(self):
        for count in range(1, 5238):
            expected = np.linspace(0, count - 1, min(8, count), dtype=int).tolist()
            self.assertEqual(scoring.sample_indices(count), expected)

    def test_parser_preserves_empty_unknown_and_missing_positions(self):
        self.assertEqual(scoring.parse_prediction("top,unknown,,bottom"),
                         ["top", None, None, "bottom"])
        self.assertEqual(scoring.parse_prediction('["r1c1", null, "bottom-right"]'),
                         ["topleft", None, "bottomright"])
        self.assertEqual(scoring.parse_prediction("top and bottom"), [None])
        self.assertEqual(scoring.parse_prediction(""), [])

    def test_perfect_all_valid_sequence(self):
        labels = list(scoring.NAMES[:8])
        result = scoring.score_sample(",".join(labels), labels, [True] * 8)
        self.assertEqual(result["metrics"], {"grid_acc": 1, "grid_ade": 0, "grid_fde": 0,
                                             "grid_transition_acc": 1, "token_f1": 1})

    def test_mask_never_compresses_or_shifts_predictions(self):
        result = scoring.score_sample("top,left,bottom", ["top", None, "bottom"],
                                      [True, False, True])
        self.assertEqual(result["metrics"]["grid_acc"], 1)
        self.assertIsNone(result["metrics"]["grid_transition_acc"])
        result = scoring.score_sample("top,bottom", ["top", None, "bottom"],
                                      [True, False, True])
        self.assertEqual(result["metrics"]["grid_acc"], 0.5)

    def test_missing_final_ground_truth_is_not_last_valid_fde(self):
        result = scoring.score_sample("top,top", ["top", None], [True, False])
        self.assertIsNone(result["metrics"]["grid_fde"])
        self.assertEqual(result["denominators"]["grid_fde"], 0)

    def test_single_valid_slot_has_no_automatic_perfect_transition(self):
        result = scoring.score_sample("top", ["top"], [True])
        self.assertIsNone(result["metrics"]["grid_transition_acc"])

    def test_all_invalid_is_null_not_zero_or_perfect(self):
        result = scoring.score_sample("middle,middle", [None, None], [False, False])
        self.assertTrue(all(value is None for value in result["metrics"].values()))
        summary = scoring.aggregate_scores([result])
        self.assertEqual(summary["records_without_scored_slots"], 1)
        self.assertTrue(all(value is None for value in summary["metrics"].values()))

    def test_model_uncertainty_does_not_change_denominator(self):
        result = scoring.score_sample("unknown,", ["top", "bottom"], [True, True])
        self.assertEqual(result["valid_slots"], 2)
        self.assertEqual(result["metrics"]["grid_acc"], 0)
        self.assertEqual(result["metrics"]["token_f1"], 0)
        self.assertEqual(result["metrics"]["grid_ade"], scoring.MAX_DISTANCE)

    def test_surplus_labels_cannot_improve_f1(self):
        normal = scoring.score_sample("top", ["top"], [True])
        surplus = scoring.score_sample("top,top", ["top"], [True])
        self.assertLess(surplus["metrics"]["token_f1"], normal["metrics"]["token_f1"])

    def test_bad_reference_mask_pairs_are_rejected(self):
        for labels, masks in (([None], [True]), (["top"], [False]), (["top"], [1]),
                              (["top"], []), ([], []), (["unknown"], [True])):
            with self.subTest(labels=labels, masks=masks), self.assertRaises(ValueError):
                scoring.score_sample("top", labels, masks)

    def test_aggregation_reports_metric_specific_denominators(self):
        first = scoring.score_sample("top,bottom", ["top", "bottom"], [True, True])
        second = scoring.score_sample("top,left", ["top", None], [True, False])
        summary = scoring.aggregate_scores([first, second])
        self.assertEqual(summary["metric_scored_records"]["grid_acc"], 2)
        self.assertEqual(summary["metric_scored_records"]["grid_fde"], 1)
        self.assertEqual(summary["metric_scored_records"]["grid_transition_acc"], 1)
        self.assertEqual(summary["metric_scored_slots_or_edges"]["grid_acc"], 3)

    @staticmethod
    def fixture():
        original = {"qid": "0", "video_path": "egodex/synthetic/0.mp4", "frame_count": 3,
                    "question": "Track the right-hand palm center.",
                    "answer": json.dumps(["left"] * 3)}
        reference = {**original, "benchmark_version": scoring.VERSION,
                     "target_joint": scoring.TARGET_JOINT, "reference_policy": scoring.MASK_POLICY,
                     "legacy_question_sha256": scoring.text_sha256(original["question"]),
                     "legacy_answer_sha256": scoring.text_sha256(original["answer"]),
                     "answer": json.dumps(["top", None, "bottom"]),
                     "reference_valid": json.dumps([True, False, True])}
        prediction = {"doc_id": 0, "doc": original, "input": "Original effective prompt.",
                      "filtered_resps": ["top,left,bottom"]}
        return reference, prediction

    def test_rescoring_retains_inputs_and_reports_no_inference_or_win(self):
        reference, prediction = self.fixture()
        original = copy.deepcopy(prediction)
        report = scoring.rescore_records([reference], [prediction], expected_count=1,
                                         expected_inputs=[prediction["input"]])
        self.assertEqual(report["metrics"]["grid_acc"], 1)
        self.assertFalse(report["inference_performed"])
        self.assertFalse(report["automatic_publication"])
        self.assertFalse(report["grt_superiority_verified"])
        self.assertEqual(prediction, original)

    def test_wrong_version_question_target_or_original_answer_rejected(self):
        for key, value in (("benchmark_version", "old"), ("target_joint", "leftHand"),
                           ("reference_policy", "other"), ("question", "Changed prompt"),
                           ("legacy_answer_sha256", "0" * 64), ("frame_count", 4)):
            with self.subTest(key=key):
                reference, prediction = self.fixture()
                reference[key] = value
                with self.assertRaises(ValueError):
                    scoring.rescore_records([reference], [prediction], expected_count=1)

    def test_partial_duplicate_reordered_ambiguous_and_changed_inputs_rejected(self):
        for case in ("missing", "wrong-id", "empty-output", "extra-output", "changed-prompt"):
            with self.subTest(case=case):
                reference, prediction = self.fixture()
                records = [prediction]
                prompts = [prediction["input"]]
                if case == "missing":
                    records = []
                elif case == "wrong-id":
                    prediction["doc_id"] = 1
                elif case == "empty-output":
                    prediction["filtered_resps"] = [""]
                elif case == "extra-output":
                    prediction["filtered_resps"].append("top,top,top")
                else:
                    prediction["input"] = "Changed effective prompt"
                with self.assertRaises(ValueError):
                    scoring.rescore_records([reference], records, expected_count=1,
                                            expected_inputs=prompts)


if __name__ == "__main__":
    unittest.main()
