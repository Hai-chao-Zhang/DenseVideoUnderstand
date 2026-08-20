from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import sync_leaderboard as sync  # noqa: E402


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def descriptor(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sync.sha256_file(path)}


def synthetic_site_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, object]:
    project = tmp_path / "main"
    driver = project / "tools/densevideo/qwen7_floor_cap48_full.py"
    driver.parent.mkdir(parents=True)
    driver.write_text("# authenticated synthetic native verifier\n", encoding="utf-8")
    root = tmp_path / "campaign"
    snapshot_path = root / "full_provenance/source_snapshot.sha256"
    snapshot_path.parent.mkdir(parents=True)
    snapshot_path.write_text(
        f"{sync.sha256_file(driver)}  {driver.resolve()}\n", encoding="utf-8"
    )
    candidate = "grt_qwen2_5_vl_7b_dual_floor_s080_o055_cap48"

    def write_samples(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps({"doc_id": index}) + "\n" for index in range(634)),
            encoding="utf-8",
        )

    summary_fields = (
        "run_name",
        "method",
        "task",
        "run_status",
        "samples",
        "token_f1",
        "result_json",
        "mean_recompute_ratio",
        "reference_patch_compute_ratio",
    )

    def write_summary(
        path: Path, *, method: str, run_name: str, result: Path, samples: object = 634
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=summary_fields)
            writer.writeheader()
            writer.writerow(
                {
                    "run_name": run_name,
                    "method": method,
                    "task": "densevideo",
                    "run_status": "success",
                    "samples": samples,
                    "token_f1": 0.4,
                    "result_json": str(result),
                    "mean_recompute_ratio": 0.8,
                    "reference_patch_compute_ratio": 0.8,
                }
            )

    archive_sample = root / "archive/samples.jsonl"
    archive_summary = root / "archive/summary.csv"
    archive_result = root / "archive/results.json"
    archive_run_name = "archived-qwen7-full"
    write_samples(archive_sample)
    write_json(archive_result, {"method": sync.QWEN7_FLOOR_FULL_ARCHIVE})
    write_summary(
        archive_summary,
        method=sync.QWEN7_FLOOR_FULL_ARCHIVE,
        run_name=archive_run_name,
        result=archive_result,
    )
    markers: dict[str, object] = {}
    for method in sync.QWEN7_FLOOR_FULL_METHODS:
        sample = root / "full_models" / method / "samples.jsonl"
        summary = root / "full_summaries" / f"{method}.csv"
        result = root / "full_models" / method / "results.json"
        write_samples(sample)
        write_json(result, {"method": method})
        write_summary(
            summary,
            method=method,
            run_name=f"{method}-full",
            result=result,
        )
        markers[method] = {
            "sample": descriptor(sample),
            "summary": descriptor(summary),
            "result": descriptor(result),
        }

    contract_path = root / "full_provenance/campaign_contract.json"
    contract: dict[str, object] = {
        "schema_version": 1,
        "campaign_kind": "qwen7_route_floor_cap48_full_v1",
        "project_root": str(project.resolve()),
        "benchmark": "DIVE-Bench",
        "dataset_revision": "b" * 40,
        "expected_full_samples": 634,
        "generation": {"max_new_tokens": 48, "temperature": 0},
        "output_root": str(root.resolve()),
        "checkpoint": {
            "name": "Qwen/Qwen2.5-VL-7B-Instruct",
            "revision": "c" * 40,
        },
        "judge": {
            "model": "Qwen/Qwen3-VL-32B-Instruct",
            "revision": "d" * 40,
            "judge_fingerprint": "e" * 64,
        },
        "source_snapshot": {
            **descriptor(snapshot_path),
            "files": [descriptor(driver)],
        },
        "archived_full_control": {
            "method": sync.QWEN7_FLOOR_FULL_ARCHIVE,
            "sample": descriptor(archive_sample),
            "summary": descriptor(archive_summary),
            "result": descriptor(archive_result),
            "samples": 634,
            "summary_run_name": archive_run_name,
        },
    }
    contract["campaign_fingerprint"] = sync._canonical_fingerprint(contract)
    fingerprint = str(contract["campaign_fingerprint"])
    write_json(contract_path, contract)
    monkeypatch.setattr(sync, "QWEN7_FLOOR_FULL_CAMPAIGN_FINGERPRINT", fingerprint)
    monkeypatch.setattr(
        sync, "QWEN7_FLOOR_FULL_CONTRACT_SHA256", sync.sha256_file(contract_path)
    )

    attempt = root / ".full_attempts" / f"attempt-{fingerprint[:16]}"
    selection = attempt / "combined.json"
    matrix = root / "full_mos/matrix.csv"
    matrix_jsonl = root / "full_mos/matrix.jsonl"
    summary = root / "full_summaries" / f"{candidate}.csv"
    diagnostic = attempt / "screen128_remainder506_diagnostics.json"
    selection.parent.mkdir(parents=True, exist_ok=True)
    selection.write_text("synthetic\n", encoding="utf-8")
    matrix.parent.mkdir(parents=True, exist_ok=True)
    scores = {
        sync.QWEN7_FLOOR_FULL_ARCHIVE: 1,
        sync.QWEN7_FLOOR_FULL_METHODS[0]: 2,
        sync.QWEN7_FLOOR_FULL_METHODS[1]: 3,
        candidate: 4,
    }
    mos_rows: list[dict[str, object]] = []
    for method in sync.QWEN7_FLOOR_FULL_MOS_METHODS:
        for index in range(634):
            score = scores[method]
            mos_rows.append(
                {
                    "sample_id": f"{method}::{index}",
                    "doc_id": index,
                    "run_name": f"{method}-full",
                    "method": method,
                    "model": method,
                    "video_name": f"video-{index}",
                    "question_id": f"question-{index}",
                    "type": "QA",
                    "open_mos_correctness": "OK",
                    "open_mos_score": score,
                    "mos_judge_model": "Qwen/Qwen3-VL-32B-Instruct",
                    "mos_judge_revision": "d" * 40,
                    "request_fingerprint": "f" * 64,
                    "judge_fingerprint": "e" * 64,
                    "error": "",
                    "question": f"question text {index}",
                    "answer": f"answer text {index}",
                    "pred": f"prediction text {method} {index}",
                    "open_mos_review": f"pred: OK; score: {score}",
                }
            )
    with matrix.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=sync.QWEN7_FLOOR_FULL_MOS_FIELDS
        )
        writer.writeheader()
        writer.writerows(mos_rows)
    matrix_jsonl.write_text(
        "".join(json.dumps(row) + "\n" for row in mos_rows), encoding="utf-8"
    )
    write_json(
        diagnostic,
        {
            "eligible_as_quality_gate_input": False,
            "full_634_is_only_hard_gate": True,
            "split_sizes": {
                "screen_selected_128": 128,
                "screen_unselected_506": 506,
            },
        },
    )
    completion_path = attempt / "full_completed.json"
    completed: dict[str, object] = {
        "schema": sync.QWEN7_FLOOR_FULL_COMPLETION_SCHEMA,
        "status": "complete",
        "stage": "full",
        "publishable": True,
        "expected_samples": 634,
        "campaign_fingerprint": fingerprint,
        "selected_method": candidate,
        "contract": descriptor(contract_path),
        "driver": descriptor(driver),
        "gates": {
            "public": descriptor(selection),
            "fresh": descriptor(selection),
            "combined": descriptor(selection),
        },
        "mos_matrix_csv": descriptor(matrix),
        "mos_matrix_jsonl": descriptor(matrix_jsonl),
        "markers": markers,
        "split_diagnostics": descriptor(diagnostic),
    }
    write_json(completion_path, completed)
    entry = {
        "family": "qwen7",
        "completion_schema": sync.QWEN7_FLOOR_FULL_COMPLETION_SCHEMA,
        "campaign_fingerprint": fingerprint,
        "selected_method": candidate,
        "revisions": {
            "model": "c" * 40,
            "dataset": "b" * 40,
            "judge": "d" * 40,
        },
    }
    judge = {
        "model": "Qwen/Qwen3-VL-32B-Instruct",
        "revision": "d" * 40,
        "fingerprint": "e" * 64,
    }
    return {
        "driver": driver,
        "completion_path": completion_path,
        "completed": completed,
        "contract": contract,
        "entry": entry,
        "judge": judge,
        "selection_pair": (selection.resolve(), sync.sha256_file(selection)),
        "matrix_pair": (matrix.resolve(), sync.sha256_file(matrix)),
        "summary_pair": (summary.resolve(), sync.sha256_file(summary)),
    }


def native_stub(evidence: dict[str, object], calls: dict[str, int]):
    contract = evidence["contract"]
    completion_path = evidence["completion_path"]

    def verified_contract(path: Path, sha256: str, fingerprint: str):
        calls["contract"] += 1
        assert descriptor(path)["sha256"] == sha256
        assert fingerprint == contract["campaign_fingerprint"]
        return path, contract

    def verify_completion_file(path: Path, **kwargs):
        calls["completion"] += 1
        assert path == completion_path
        return json.loads(path.read_text(encoding="utf-8"))

    return SimpleNamespace(
        PROJECT_ROOT=Path(evidence["driver"]).parents[2],
        FULL_SAMPLES=634,
        CANDIDATE=evidence["entry"]["selected_method"],
        METHODS=(
            "qwen2_5_vl_7b_floor_quality_base",
            "qwen2_5_vl_7b_floor_grt_all",
            evidence["entry"]["selected_method"],
        ),
        MOS_METHODS=(
            "qwen2_5_vl_7b",
            "qwen2_5_vl_7b_floor_quality_base",
            "qwen2_5_vl_7b_floor_grt_all",
            evidence["entry"]["selected_method"],
        ),
        verified_contract=verified_contract,
        verify_completion_file=verify_completion_file,
    )


def validate_raw(evidence: dict[str, object]) -> dict[str, object]:
    completion_path = Path(evidence["completion_path"])
    contract_path = Path(evidence["completed"]["contract"]["path"])  # type: ignore[index]
    return sync._validate_qwen7_floor_full_raw_artifacts(
        contract=evidence["contract"],  # type: ignore[arg-type]
        completed=evidence["completed"],  # type: ignore[arg-type]
        contract_base=contract_path.parent,
        completion_base=completion_path.parent,
    )


def rewrite_jsonl_doc_id(path: Path, *, row_index: int, value: object) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[row_index]["doc_id"] = value
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def add_jsonl_field(path: Path, *, field: str, value: object) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0][field] = value
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def prepend_duplicate_json_key(
    path: Path, *, row_index: int, field: str, hidden_value: object
) -> None:
    rows = path.read_text(encoding="utf-8").splitlines()
    assert rows[row_index].startswith("{")
    rows[row_index] = (
        "{"
        + json.dumps(field)
        + ":"
        + json.dumps(hidden_value)
        + ","
        + rows[row_index][1:]
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def rewrite_csv_field(path: Path, *, field: str, value: object) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        rows = list(reader)
    rows[0][field] = str(value)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rewrite_first_csv_row_shape(path: Path, *, add_cell: bool) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if add_cell:
        rows[1].append("UNDECLARED")
    else:
        rows[1].pop()
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)


def rewrite_summary_closed_schema(path: Path, *, mode: str) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if mode == "duplicate_samples":
        index = rows[0].index("samples")
        rows[0].insert(index, "samples")
        for row in rows[1:]:
            row.insert(index, "634.9")
    elif mode == "missing_required":
        index = rows[0].index("samples")
        for row in rows:
            row.pop(index)
    elif mode == "extra_cell":
        rows[1].append("UNDECLARED")
    elif mode == "missing_cell":
        rows[1].pop()
    else:  # pragma: no cover - test helper misuse
        raise AssertionError(mode)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)


def raw_sample_descriptor(evidence: dict[str, object], method: str) -> dict[str, str]:
    if method == sync.QWEN7_FLOOR_FULL_ARCHIVE:
        return evidence["contract"]["archived_full_control"]["sample"]  # type: ignore[index,return-value]
    return evidence["completed"]["markers"][method]["sample"]  # type: ignore[index,return-value]


def raw_summary_descriptor(evidence: dict[str, object], method: str) -> dict[str, str]:
    if method == sync.QWEN7_FLOOR_FULL_ARCHIVE:
        return evidence["contract"]["archived_full_control"]["summary"]  # type: ignore[index,return-value]
    return evidence["completed"]["markers"][method]["summary"]  # type: ignore[index,return-value]


def replace_raw_descriptor(
    evidence: dict[str, object], *, method: str, field: str, path: Path
) -> None:
    if method == sync.QWEN7_FLOOR_FULL_ARCHIVE:
        evidence["contract"]["archived_full_control"][field] = descriptor(path)  # type: ignore[index]
    else:
        evidence["completed"]["markers"][method][field] = descriptor(path)  # type: ignore[index]


def test_site_sync_uses_native_full_verifier_and_exact_release_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = synthetic_site_evidence(tmp_path, monkeypatch)
    calls = {"contract": 0, "completion": 0}
    monkeypatch.setattr(
        sync,
        "_load_qwen7_floor_native_verifier",
        lambda *args: native_stub(evidence, calls),
    )
    sync._validate_multifamily_completion(
        evidence["completion_path"],
        entry=evidence["entry"],
        selection={},
        selection_pair=evidence["selection_pair"],
        matrix_pair=evidence["matrix_pair"],
        summary_pair=evidence["summary_pair"],
        judge=evidence["judge"],
    )
    assert calls == {"contract": 1, "completion": 1}


def test_site_sync_rejects_source_tamper_before_loading_native_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = synthetic_site_evidence(tmp_path, monkeypatch)
    loaded = {"value": False}

    def should_not_load(*args):
        loaded["value"] = True
        return native_stub(evidence, {"contract": 0, "completion": 0})

    monkeypatch.setattr(sync, "_load_qwen7_floor_native_verifier", should_not_load)
    Path(evidence["driver"]).write_text("tampered\n", encoding="utf-8")
    with pytest.raises(sync.SyncError, match="SHA-256"):
        sync._validate_qwen7_floor_full_completion(
            evidence["completion_path"],
            completed=evidence["completed"],
            entry=evidence["entry"],
            selection_pair=evidence["selection_pair"],
            matrix_pair=evidence["matrix_pair"],
            summary_pair=evidence["summary_pair"],
            judge=evidence["judge"],
        )
    assert loaded["value"] is False


def test_site_sync_propagates_native_telemetry_tamper_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = synthetic_site_evidence(tmp_path, monkeypatch)
    native = native_stub(evidence, {"contract": 0, "completion": 0})

    def reject(*args, **kwargs):
        raise ValueError("telemetry formula/count mismatch")

    native.verify_completion_file = reject
    monkeypatch.setattr(
        sync, "_load_qwen7_floor_native_verifier", lambda *args: native
    )
    with pytest.raises(sync.SyncError, match="telemetry formula/count mismatch"):
        sync._validate_qwen7_floor_full_completion(
            evidence["completion_path"],
            completed=evidence["completed"],
            entry=evidence["entry"],
            selection_pair=evidence["selection_pair"],
            matrix_pair=evidence["matrix_pair"],
            summary_pair=evidence["summary_pair"],
            judge=evidence["judge"],
        )


def test_site_sync_rejects_nonterminal_full_contract_explicitly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = synthetic_site_evidence(tmp_path, monkeypatch)
    contract_path = Path(evidence["completed"]["contract"]["path"])
    with pytest.raises(sync.SyncError, match="FullCompletionRequired"):
        sync._validate_multifamily_completion(
            contract_path,
            entry=evidence["entry"],
            selection={},
            selection_pair=evidence["selection_pair"],
            matrix_pair=evidence["matrix_pair"],
            summary_pair=evidence["summary_pair"],
            judge=evidence["judge"],
        )


@pytest.mark.parametrize(
    ("method", "bad_doc_id"),
    (
        (sync.QWEN7_FLOOR_FULL_ARCHIVE, "0"),
        (sync.QWEN7_FLOOR_FULL_METHODS[0], False),
        (sync.QWEN7_FLOOR_FULL_METHODS[1], 0.5),
        (sync.QWEN7_FLOOR_FULL_METHODS[2], 9999),
    ),
)
def test_site_raw_sample_doc_id_type_and_domain_tamper_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    bad_doc_id: object,
) -> None:
    evidence = synthetic_site_evidence(tmp_path, monkeypatch)
    sample_path = Path(raw_sample_descriptor(evidence, method)["path"])
    rewrite_jsonl_doc_id(sample_path, row_index=0, value=bad_doc_id)
    replace_raw_descriptor(evidence, method=method, field="sample", path=sample_path)
    with pytest.raises(sync.SyncError, match="doc_id"):
        validate_raw(evidence)


@pytest.mark.parametrize(
    ("method_index", "bad_doc_id"),
    ((0, "0"), (1, False), (2, 0.5), (3, 9999)),
)
def test_site_raw_mos_jsonl_doc_id_type_and_domain_tamper_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method_index: int,
    bad_doc_id: object,
) -> None:
    evidence = synthetic_site_evidence(tmp_path, monkeypatch)
    matrix_descriptor = evidence["completed"]["mos_matrix_jsonl"]  # type: ignore[index]
    matrix_jsonl = Path(matrix_descriptor["path"])
    rewrite_jsonl_doc_id(
        matrix_jsonl, row_index=method_index * 634, value=bad_doc_id
    )
    evidence["completed"]["mos_matrix_jsonl"] = descriptor(matrix_jsonl)  # type: ignore[index]
    with pytest.raises(sync.SyncError, match="doc_id"):
        validate_raw(evidence)


def test_site_raw_mos_csv_doc_id_must_be_canonical_integer_string(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = synthetic_site_evidence(tmp_path, monkeypatch)
    matrix_descriptor = evidence["completed"]["mos_matrix_csv"]  # type: ignore[index]
    matrix_csv = Path(matrix_descriptor["path"])
    rewrite_csv_field(matrix_csv, field="doc_id", value="00")
    evidence["completed"]["mos_matrix_csv"] = descriptor(matrix_csv)  # type: ignore[index]
    with pytest.raises(sync.SyncError, match="canonical integer string"):
        validate_raw(evidence)


def test_site_raw_mos_jsonl_rejects_undeclared_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = synthetic_site_evidence(tmp_path, monkeypatch)
    matrix_descriptor = evidence["completed"]["mos_matrix_jsonl"]  # type: ignore[index]
    matrix_jsonl = Path(matrix_descriptor["path"])
    add_jsonl_field(matrix_jsonl, field="undeclared", value="tampered")
    evidence["completed"]["mos_matrix_jsonl"] = descriptor(matrix_jsonl)  # type: ignore[index]
    with pytest.raises(sync.SyncError, match="exact MOS fields"):
        validate_raw(evidence)


def test_site_raw_sample_jsonl_rejects_duplicate_doc_id_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = synthetic_site_evidence(tmp_path, monkeypatch)
    method = sync.QWEN7_FLOOR_FULL_METHODS[-1]
    sample_path = Path(raw_sample_descriptor(evidence, method)["path"])
    prepend_duplicate_json_key(
        sample_path,
        row_index=0,
        field="doc_id",
        hidden_value="HIDDEN-TAMPER",
    )
    replace_raw_descriptor(evidence, method=method, field="sample", path=sample_path)
    with pytest.raises(sync.SyncError, match="duplicate JSON key 'doc_id'"):
        validate_raw(evidence)


def test_site_raw_mos_jsonl_rejects_duplicate_question_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = synthetic_site_evidence(tmp_path, monkeypatch)
    matrix_descriptor = evidence["completed"]["mos_matrix_jsonl"]  # type: ignore[index]
    matrix_jsonl = Path(matrix_descriptor["path"])
    prepend_duplicate_json_key(
        matrix_jsonl,
        row_index=0,
        field="question",
        hidden_value="HIDDEN-TAMPER",
    )
    evidence["completed"]["mos_matrix_jsonl"] = descriptor(matrix_jsonl)  # type: ignore[index]
    with pytest.raises(sync.SyncError, match="duplicate JSON key 'question'"):
        validate_raw(evidence)


@pytest.mark.parametrize("add_cell", (True, False))
def test_site_raw_mos_csv_rejects_extra_or_missing_cell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    add_cell: bool,
) -> None:
    evidence = synthetic_site_evidence(tmp_path, monkeypatch)
    matrix_descriptor = evidence["completed"]["mos_matrix_csv"]  # type: ignore[index]
    matrix_csv = Path(matrix_descriptor["path"])
    rewrite_first_csv_row_shape(matrix_csv, add_cell=add_cell)
    evidence["completed"]["mos_matrix_csv"] = descriptor(matrix_csv)  # type: ignore[index]
    with pytest.raises(sync.SyncError, match="exact MOS cells"):
        validate_raw(evidence)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("question", "tampered-question"),
        ("pred", "tampered-prediction"),
        ("open_mos_review", "tampered-review"),
    ),
)
def test_site_raw_mos_csv_jsonl_text_pred_review_tamper_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    evidence = synthetic_site_evidence(tmp_path, monkeypatch)
    matrix_descriptor = evidence["completed"]["mos_matrix_csv"]  # type: ignore[index]
    matrix_csv = Path(matrix_descriptor["path"])
    rewrite_csv_field(matrix_csv, field=field, value=value)
    evidence["completed"]["mos_matrix_csv"] = descriptor(matrix_csv)  # type: ignore[index]
    with pytest.raises(sync.SyncError, match="CSV/JSONL"):
        validate_raw(evidence)


@pytest.mark.parametrize("method", sync.QWEN7_FLOOR_FULL_MOS_METHODS)
def test_site_each_raw_summary_rejects_fractional_634_9(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    evidence = synthetic_site_evidence(tmp_path, monkeypatch)
    summary_path = Path(raw_summary_descriptor(evidence, method)["path"])
    rewrite_csv_field(summary_path, field="samples", value="634.9")
    replace_raw_descriptor(
        evidence, method=method, field="summary", path=summary_path
    )
    with pytest.raises(sync.SyncError, match="must be an integer"):
        validate_raw(evidence)


def test_site_raw_summary_rejects_boolean_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = synthetic_site_evidence(tmp_path, monkeypatch)
    method = sync.QWEN7_FLOOR_FULL_METHODS[-1]
    summary_path = Path(raw_summary_descriptor(evidence, method)["path"])
    rewrite_csv_field(summary_path, field="samples", value="True")
    replace_raw_descriptor(
        evidence, method=method, field="summary", path=summary_path
    )
    with pytest.raises(sync.SyncError, match="invalid numeric|must be an integer"):
        validate_raw(evidence)


@pytest.mark.parametrize("token", ("634.0", "0634", "6.34e2"))
def test_site_raw_summary_rejects_noncanonical_integer_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    token: str,
) -> None:
    evidence = synthetic_site_evidence(tmp_path, monkeypatch)
    method = sync.QWEN7_FLOOR_FULL_METHODS[-1]
    summary_path = Path(raw_summary_descriptor(evidence, method)["path"])
    rewrite_csv_field(summary_path, field="samples", value=token)
    replace_raw_descriptor(
        evidence, method=method, field="summary", path=summary_path
    )
    with pytest.raises(sync.SyncError, match="canonical nonnegative"):
        validate_raw(evidence)


@pytest.mark.parametrize(
    "mode", ("duplicate_samples", "missing_required", "extra_cell", "missing_cell")
)
def test_site_raw_summary_rejects_closed_schema_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    evidence = synthetic_site_evidence(tmp_path, monkeypatch)
    method = sync.QWEN7_FLOOR_FULL_METHODS[-1]
    summary_path = Path(raw_summary_descriptor(evidence, method)["path"])
    rewrite_summary_closed_schema(summary_path, mode=mode)
    replace_raw_descriptor(
        evidence, method=method, field="summary", path=summary_path
    )
    match = (
        "summary header"
        if mode in {"duplicate_samples", "missing_required"}
        else "does not match"
    )
    with pytest.raises(sync.SyncError, match=match):
        validate_raw(evidence)


def test_qwen7_floor_schema_has_dedicated_selection_contract() -> None:
    assert sync.QWEN7_FLOOR_FULL_COMPLETION_SCHEMA in (
        sync.MULTIFAMILY_CONTRACTS["qwen7"]["completion_schema"]
    )
    contract = sync.MULTIFAMILY_QWEN7_FLOOR_SELECTION_CONTRACT
    assert contract["baselines"] == {
        "qwen2_5_vl_7b",
        "qwen2_5_vl_7b_floor_quality_base",
        "qwen2_5_vl_7b_floor_grt_all",
    }
    assert contract["candidates"] == {
        "grt_qwen2_5_vl_7b_dual_floor_s080_o055_cap48"
    }
