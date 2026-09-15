import json
import tempfile
import unittest
from pathlib import Path

from tools.densevideo.compare_lmms_sample_jsonl import SampleFormatError, compare_sample_files, main


class CompareLMMSSampleJsonlTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    @staticmethod
    def _row(doc_id, question, reference, prediction):
        return {
            "doc_id": doc_id,
            "input": question,
            "target": reference,
            "filtered_resps": [prediction],
        }

    @staticmethod
    def _write(path: Path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_aligns_by_doc_id_and_accepts_only_order_difference(self):
        left = self.root / "left.jsonl"
        right = self.root / "right.jsonl"
        rows = [
            self._row(0, "question zero", "reference zero", "prediction zero"),
            self._row(1, "question one", "reference one", "prediction one"),
        ]
        self._write(left, rows)
        self._write(right, list(reversed(rows)))

        report = compare_sample_files(left, right)

        self.assertTrue(report["identical"])
        self.assertEqual(report["aligned_rows"], 2)
        self.assertEqual(report["mismatch_counts"], {"question": 0, "reference": 0, "prediction": 0})
        self.assertEqual(main([str(left), str(right)]), 0)

    def test_reports_question_reference_prediction_and_doc_set_mismatches(self):
        left = self.root / "left.jsonl"
        right = self.root / "right.jsonl"
        self._write(
            left,
            [
                self._row(0, "same question", "left reference", "left prediction"),
                self._row(1, "left-only", "reference", "prediction"),
            ],
        )
        self._write(
            right,
            [
                self._row(0, "different question", "right reference", "right prediction"),
                self._row(2, "right-only", "reference", "prediction"),
            ],
        )

        report = compare_sample_files(left, right)

        self.assertFalse(report["identical"])
        self.assertEqual(report["only_left_doc_ids"], [1])
        self.assertEqual(report["only_right_doc_ids"], [2])
        self.assertEqual(report["mismatch_counts"], {"question": 1, "reference": 1, "prediction": 1})
        self.assertEqual(main([str(left), str(right), "--max-details", "0"]), 1)

    def test_rejects_duplicate_doc_ids_and_missing_required_fields(self):
        duplicate = self.root / "duplicate.jsonl"
        valid = self.root / "valid.jsonl"
        missing = self.root / "missing.jsonl"
        row = self._row(0, "question", "reference", "prediction")
        self._write(duplicate, [row, row])
        self._write(valid, [row])
        self._write(missing, [{"doc_id": 0, "input": "question", "target": "reference"}])

        with self.assertRaisesRegex(SampleFormatError, "duplicate doc_id"):
            compare_sample_files(duplicate, valid)
        with self.assertRaisesRegex(SampleFormatError, "missing required field 'filtered_resps'"):
            compare_sample_files(valid, missing)


if __name__ == "__main__":
    unittest.main()
