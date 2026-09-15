import csv
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.densevideo import score_open_mos_matrix  # noqa: E402
from tools.densevideo.score_open_mos import JudgeResponse  # noqa: E402


def _write_samples(root: Path, method: str):
    path = root / "runs" / method / "densevideo" / "checkpoint" / "test_samples_densevideo.jsonl"
    path.parent.mkdir(parents=True)
    rows = [
        {"doc_id": 0, "target": "alpha", "filtered_resps": ["alpha"]},
        {"doc_id": 1, "target": "beta", "filtered_resps": ["wrong"]},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_matrix_scores_multiple_methods_with_unique_resume_ids(tmp_path, monkeypatch):
    input_root = tmp_path / "input"
    output_dir = tmp_path / "mos"
    for method in ("base", "grt"):
        _write_samples(input_root, method)

    judged_requests = 0
    original_score_batch = score_open_mos_matrix.score_batch_resilient

    def count_unique_requests(judge, requests):
        nonlocal judged_requests
        judged_requests += len(requests)
        return original_score_batch(judge, requests)

    monkeypatch.setattr(score_open_mos_matrix, "score_batch_resilient", count_unique_requests)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score_open_mos_matrix.py",
            "--method",
            "base",
            "--method",
            "grt",
            "--input-root",
            str(input_root),
            "--output-dir",
            str(output_dir),
            "--expected-samples",
            "2",
            "--backend",
            "dry-run",
            "--judge-revision",
            "judge-commit",
        ],
    )
    score_open_mos_matrix.main()

    with (output_dir / "matrix.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert {row["method"] for row in rows} == {"base", "grt"}
    assert len({row["sample_id"] for row in rows}) == 4
    assert not any(row["error"] for row in rows)
    assert all(len(row["request_fingerprint"]) == 64 for row in rows)
    assert all(len(row["judge_fingerprint"]) == 64 for row in rows)
    assert len({row["judge_fingerprint"] for row in rows}) == 1
    assert len({row["request_fingerprint"] for row in rows}) == 2
    assert {row["mos_judge_revision"] for row in rows} == {"judge-commit"}
    # Both methods produced exactly the same two judge inputs.  Score each
    # unique input once and replicate the result, rather than injecting
    # method-dependent judge noise.
    assert judged_requests == 2


def test_matrix_resume_reuses_identical_completed_judgment(tmp_path, monkeypatch):
    input_root = tmp_path / "input"
    output_dir = tmp_path / "mos"
    for method in ("base", "grt"):
        _write_samples(input_root, method)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score_open_mos_matrix.py",
            "--method",
            "base",
            "--input-root",
            str(input_root),
            "--output-dir",
            str(output_dir),
            "--expected-samples",
            "2",
            "--backend",
            "dry-run",
        ],
    )
    score_open_mos_matrix.main()

    def unexpected_judge_call(judge, requests):
        raise AssertionError("identical resumed inputs should be served from the MOS cache")

    monkeypatch.setattr(score_open_mos_matrix, "score_batch_resilient", unexpected_judge_call)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score_open_mos_matrix.py",
            "--method",
            "base",
            "--method",
            "grt",
            "--input-root",
            str(input_root),
            "--output-dir",
            str(output_dir),
            "--expected-samples",
            "2",
            "--backend",
            "dry-run",
            "--resume",
        ],
    )
    score_open_mos_matrix.main()

    with (output_dir / "matrix.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert {row["method"] for row in rows} == {"base", "grt"}


def test_matrix_resume_rejudges_legacy_rows_without_fingerprints(tmp_path, monkeypatch):
    input_root = tmp_path / "input"
    output_dir = tmp_path / "mos"
    for method in ("base", "grt"):
        _write_samples(input_root, method)
    output_dir.mkdir(parents=True)

    legacy_rows = []
    values = ((0, "alpha", "alpha"), (1, "beta", "wrong"))
    for method, score in (("base", 1), ("grt", 5)):
        for doc_id, answer, pred in values:
            legacy_rows.append(
                {
                    "sample_id": f"{method}::doc_{doc_id}",
                    "doc_id": doc_id,
                    "run_name": f"{method}_densevideo_open_mos",
                    "method": method,
                    "model": method,
                    "video_name": "",
                    "question_id": doc_id,
                    "type": "",
                    "question": "",
                    "answer": answer,
                    "pred": pred,
                    "open_mos_correctness": "yes",
                    "open_mos_score": score,
                    "open_mos_review": json.dumps({"pred": "yes", "score": score}),
                    "mos_judge_model": "legacy-judge",
                    "error": "",
                }
            )
    (output_dir / "matrix.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in legacy_rows),
        encoding="utf-8",
    )

    judged_requests = 0
    original_score_batch = score_open_mos_matrix.score_batch_resilient

    def count_unique_requests(judge, requests):
        nonlocal judged_requests
        judged_requests += len(requests)
        return original_score_batch(judge, requests)

    monkeypatch.setattr(score_open_mos_matrix, "score_batch_resilient", count_unique_requests)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score_open_mos_matrix.py",
            "--method",
            "base",
            "--method",
            "grt",
            "--input-root",
            str(input_root),
            "--output-dir",
            str(output_dir),
            "--expected-samples",
            "2",
            "--backend",
            "dry-run",
            "--resume",
        ],
    )
    score_open_mos_matrix.main()

    with (output_dir / "matrix.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_doc = {}
    for row in rows:
        by_doc.setdefault(row["doc_id"], set()).add(row["open_mos_score"])
    assert by_doc == {"0": {"5"}, "1": {"0"}}
    assert judged_requests == 2
    assert all(len(row["request_fingerprint"]) == 64 for row in rows)


def test_matrix_resume_rejudges_changed_prediction_with_same_sample_id(tmp_path, monkeypatch):
    input_root = tmp_path / "input"
    output_dir = tmp_path / "mos"
    _write_samples(input_root, "base")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score_open_mos_matrix.py",
            "--method",
            "base",
            "--input-root",
            str(input_root),
            "--output-dir",
            str(output_dir),
            "--expected-samples",
            "2",
            "--backend",
            "dry-run",
        ],
    )
    score_open_mos_matrix.main()

    sample_path = input_root / "runs/base/densevideo/checkpoint/test_samples_densevideo.jsonl"
    changed_rows = [
        {"doc_id": 0, "target": "alpha", "filtered_resps": ["changed"]},
        {"doc_id": 1, "target": "beta", "filtered_resps": ["wrong"]},
    ]
    sample_path.write_text("".join(json.dumps(row) + "\n" for row in changed_rows), encoding="utf-8")

    judged_requests = 0
    original_score_batch = score_open_mos_matrix.score_batch_resilient

    def count_unique_requests(judge, requests):
        nonlocal judged_requests
        judged_requests += len(requests)
        return original_score_batch(judge, requests)

    monkeypatch.setattr(score_open_mos_matrix, "score_batch_resilient", count_unique_requests)
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--resume"])
    score_open_mos_matrix.main()

    rows = list(csv.DictReader((output_dir / "matrix.csv").open(encoding="utf-8", newline="")))
    assert len(rows) == 2
    assert judged_requests == 1
    assert {row["doc_id"]: row["open_mos_score"] for row in rows} == {"0": "0", "1": "0"}


def test_matrix_resume_rejudges_when_judge_fingerprint_changes(tmp_path, monkeypatch):
    input_root = tmp_path / "input"
    output_dir = tmp_path / "mos"
    _write_samples(input_root, "base")
    base_argv = [
        "score_open_mos_matrix.py",
        "--method",
        "base",
        "--input-root",
        str(input_root),
        "--output-dir",
        str(output_dir),
        "--expected-samples",
        "2",
        "--backend",
        "dry-run",
    ]
    monkeypatch.setattr(sys, "argv", [*base_argv, "--max-new-tokens", "64"])
    score_open_mos_matrix.main()

    judged_requests = 0
    original_score_batch = score_open_mos_matrix.score_batch_resilient

    def count_unique_requests(judge, requests):
        nonlocal judged_requests
        judged_requests += len(requests)
        return original_score_batch(judge, requests)

    monkeypatch.setattr(score_open_mos_matrix, "score_batch_resilient", count_unique_requests)
    monkeypatch.setattr(sys, "argv", [*base_argv, "--max-new-tokens", "65", "--resume"])
    score_open_mos_matrix.main()

    assert judged_requests == 2


def test_matrix_resume_rejudges_when_judge_revision_changes(tmp_path, monkeypatch):
    input_root = tmp_path / "input"
    output_dir = tmp_path / "mos"
    _write_samples(input_root, "base")
    base_argv = [
        "score_open_mos_matrix.py",
        "--method",
        "base",
        "--input-root",
        str(input_root),
        "--output-dir",
        str(output_dir),
        "--expected-samples",
        "2",
        "--backend",
        "dry-run",
    ]
    monkeypatch.setattr(sys, "argv", [*base_argv, "--judge-revision", "commit-a"])
    score_open_mos_matrix.main()

    judged_requests = 0
    original_score_batch = score_open_mos_matrix.score_batch_resilient

    def count_unique_requests(judge, requests):
        nonlocal judged_requests
        judged_requests += len(requests)
        return original_score_batch(judge, requests)

    monkeypatch.setattr(score_open_mos_matrix, "score_batch_resilient", count_unique_requests)
    monkeypatch.setattr(
        sys,
        "argv",
        [*base_argv, "--judge-revision", "commit-b", "--resume"],
    )
    score_open_mos_matrix.main()

    assert judged_requests == 2
    rows = list(csv.DictReader((output_dir / "matrix.csv").open(encoding="utf-8", newline="")))
    assert {row["mos_judge_revision"] for row in rows} == {"commit-b"}


def test_matrix_malformed_rows_are_errors_and_resume_retries_them(tmp_path, monkeypatch):
    input_root = tmp_path / "input"
    output_dir = tmp_path / "mos"
    _write_samples(input_root, "base")
    argv = [
        "score_open_mos_matrix.py",
        "--method",
        "base",
        "--input-root",
        str(input_root),
        "--output-dir",
        str(output_dir),
        "--expected-samples",
        "2",
        "--backend",
        "dry-run",
    ]

    original_score_batch = score_open_mos_matrix.score_batch_resilient

    def malformed_responses(judge, requests):
        return [(JudgeResponse(text="not json", model="judge"), "") for _ in requests]

    monkeypatch.setattr(score_open_mos_matrix, "score_batch_resilient", malformed_responses)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(RuntimeError, match="errors=2"):
        score_open_mos_matrix.main()

    failed_rows = list(csv.DictReader((output_dir / "matrix.csv").open(encoding="utf-8", newline="")))
    assert all(row["error"].startswith("invalid_mos_response:") for row in failed_rows)
    assert all(row["open_mos_score"] == "" for row in failed_rows)

    judged_requests = 0

    def count_unique_requests(judge, requests):
        nonlocal judged_requests
        judged_requests += len(requests)
        return original_score_batch(judge, requests)

    monkeypatch.setattr(score_open_mos_matrix, "score_batch_resilient", count_unique_requests)
    monkeypatch.setattr(sys, "argv", [*argv, "--resume"])
    score_open_mos_matrix.main()

    retried_rows = list(csv.DictReader((output_dir / "matrix.csv").open(encoding="utf-8", newline="")))
    assert judged_requests == 2
    assert not any(row["error"] for row in retried_rows)


def test_matrix_searches_repeated_input_roots(tmp_path, monkeypatch):
    archived_root = tmp_path / "archived"
    fresh_root = tmp_path / "fresh"
    output_dir = tmp_path / "mos"
    _write_samples(archived_root, "base")
    _write_samples(fresh_root, "grt")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score_open_mos_matrix.py",
            "--method",
            "base",
            "--method",
            "grt",
            "--input-root",
            str(archived_root),
            "--input-root",
            str(fresh_root),
            "--output-dir",
            str(output_dir),
            "--expected-samples",
            "2",
            "--backend",
            "dry-run",
        ],
    )
    score_open_mos_matrix.main()

    rows = list(csv.DictReader((output_dir / "matrix.csv").open(encoding="utf-8", newline="")))
    assert len(rows) == 4
    assert {row["method"] for row in rows} == {"base", "grt"}
