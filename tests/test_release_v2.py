import csv
import hashlib
import json
import sys
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from qwen_dual_release_fixture import write_dual_bundle  # noqa: E402
import test_sync_leaderboard as sync_test_fixture  # noqa: E402
import test_qwen7_floor_full_release as qwen7_floor_fixture  # noqa: E402
from test_sync_leaderboard import (  # noqa: E402
    WINNER_VALUES,
    _legacy_open_mos_fixture,
    legacy_site_payload,
)

from scripts.sync_leaderboard import (  # noqa: E402
    MULTIFAMILY_JUDGE,
    QWEN7_FLOOR_FULL_MOS_FIELDS,
    SyncError,
    _canonical_fingerprint,
    _inspect_failed_full_mos_pair,
    _load_hashed_json,
    _load_json,
    _open_mos_request_fingerprint,
    _require_pinned_pair,
    _site_row,
    _validate_qwen7_floor_full_raw_artifacts,
    _validate_failed_selection,
    _validate_llava7_failed_evidence,
    build_multifamily_release_payload,
    load_site_payload,
    rerank,
    sha256_file,
    validate_multifamily_release_binding,
)
from scripts import qwen_dual_release as dual_release  # noqa: E402


REAL_LLAVA7_FAILURE = Path(
    "/work/nvme/bdqf/william/charles/dive_grt_quality_outputs/"
    "quality_20260819_062149/llava7_dual_route"
)


def descriptor(path: Path) -> dict[str, str]:
    path = path.resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def real_llava_entry() -> dict:
    root = REAL_LLAVA7_FAILURE
    contract = json.loads(
        (root / "provenance/campaign_contract.json").read_text(encoding="utf-8")
    )
    return {
        "family": "llava7",
        "status": "not_promoted",
        "reason_code": "strict_quality_gate_failed",
        "base_method": "llava_onevision_original",
        "replaces_method": None,
        "candidate_methods": ["grt_llava_onevision_7b_hf_dual_s002_o005"],
        "selected_method": None,
        "selection_schema": "strict_full_gate_v1",
        "completion_schema": None,
        "campaign_fingerprint": contract["campaign_fingerprint"],
        "revisions": {
            "model": contract["checkpoint"]["revision"],
            "dataset": contract["dataset_revision"],
            "judge": MULTIFAMILY_JUDGE["revision"],
        },
        "evidence": {
            "kind": "failed_full_selection",
            "contract": descriptor(root / "provenance/campaign_contract.json"),
            "source_snapshot": descriptor(root / "provenance/source_snapshot.sha256"),
            "validation": descriptor(root / "full_validation.json"),
            "selection": descriptor(root / "full_selection.json"),
            "open_mos_matrix_csv": descriptor(root / "full_mos/matrix.csv"),
            "open_mos_matrix_jsonl": descriptor(root / "full_mos/matrix.jsonl"),
        },
    }


def raw_candidate(family: str, method: str, open_mos: str) -> dict[str, str]:
    return {
        "task": "densevideo",
        "rank": "0",
        "display_name": f"GRT {family} verified",
        "method": method,
        "samples": "634",
        "open_mos": open_mos,
        "token_f1": "0.9",
        "cer": "0.1",
        "wer": "0.1",
        "exact_match": "0.5",
        "mean_recompute_ratio": "0.75",
        "reference_patch_compute_ratio": "0.70",
        "mean_effective_fps": "0.02",
        "mean_throughput_fps": "4.5",
        "source": "open",
    }


def csv_row_from_site(row: dict) -> dict[str, str]:
    def value(field: str) -> str:
        item = row.get(field)
        return "" if item is None else str(item)

    return {
        "task": "densevideo",
        "rank": str(row["rank"]),
        "display_name": str(row["model"]),
        "method": str(row["method"]),
        "samples": str(row["samples"]),
        "source": str(row.get("source", "open")),
        "open_mos": value("open_mos"),
        "token_f1": value("token_f1"),
        "cer": value("cer"),
        "wer": value("wer"),
        "exact_match": value("exact_match"),
        "mean_recompute_ratio": value("recompute_ratio"),
        "reference_patch_compute_ratio": value("reference_recompute_ratio"),
        "mean_effective_fps": value("effective_fps"),
        "mean_throughput_fps": value("throughput_fps"),
    }


def release_rows(candidates: list[dict[str, str]]) -> list[dict[str, str]]:
    original = legacy_site_payload()
    candidate_methods = {row["method"] for row in candidates}
    retained = [
        dict(row)
        for row in original["tracks"]["lpm"]
        if row["method"] != "grt_llava_ov_0_5b"
        and row["method"] not in candidate_methods
    ]
    candidate_site = [
        _site_row(row, row["method"], 634, strict_telemetry=False) for row in candidates
    ]
    ranked = rerank([*retained, *candidate_site], "open_mos")
    raw_by_method = {row["method"]: dict(row) for row in candidates}
    rows = []
    for row in ranked:
        method = str(row["method"])
        if method in raw_by_method:
            raw = raw_by_method[method]
            raw["rank"] = str(row["rank"])
            rows.append(raw)
        else:
            rows.append(csv_row_from_site(row))
    return rows


def verified_family(
    family: str,
    *,
    base: str,
    replacement: str | None,
    candidate_row: dict[str, str] | None,
) -> dict:
    promoted = candidate_row is not None
    selected = candidate_row["method"] if promoted else None
    return {
        "family": family,
        "status": "promoted" if promoted else "not_promoted",
        "reason_code": (
            "strict_quality_gate_passed" if promoted else "strict_quality_gate_failed"
        ),
        "base_method": base,
        "replaces_method": replacement,
        "candidate_methods": [selected or f"grt_{family}_failed"],
        "selected_method": selected,
        "campaign_fingerprint": hashlib.sha256(family.encode()).hexdigest(),
        "candidate": (
            _site_row(candidate_row, selected, 634)
            if candidate_row is not None
            else None
        ),
    }


def dual_candidate_row(evidence: dict) -> dict[str, str]:
    selected = evidence["selected_method"]
    selection = evidence["selection_payload"]
    return {
        "task": "densevideo",
        "rank": "0",
        "display_name": f"Verified {evidence['family']} dual GRT",
        "method": selected,
        "samples": "634",
        "open_mos": str(selection["selected_score"]),
        "token_f1": str(selection["selected_secondary_score"]),
        "cer": "0.1",
        "wer": "0.1",
        "exact_match": "0.5",
        "mean_recompute_ratio": str(selection["selected_recompute_ratio"]),
        "reference_patch_compute_ratio": str(
            selection["selected_reference_patch_compute_ratio"]
        ),
        "mean_effective_fps": "0.02",
        "mean_throughput_fps": "4.5",
        "source": "open",
    }


@pytest.mark.parametrize("qwen7_promoted,expected_rows", [(True, 29), (False, 28)])
def test_v2_payload_publishes_only_promoted_and_preserves_highmotion(
    qwen7_promoted: bool, expected_rows: int
) -> None:
    original = legacy_site_payload()
    candidates = {
        "route31": raw_candidate("route31", "grt_route31_v2", "4.7"),
        "qwen3": raw_candidate("qwen3", "grt_qwen3_v2", "4.8"),
    }
    if qwen7_promoted:
        candidates["qwen7"] = raw_candidate("qwen7", "grt_qwen7_v2", "4.9")
    verified = [
        verified_family(
            "route31",
            base="llava_onevision_0_5b",
            replacement="grt_llava_ov_0_5b",
            candidate_row=candidates["route31"],
        ),
        verified_family(
            "llava7",
            base="llava_onevision_original",
            replacement=None,
            candidate_row=None,
        ),
        verified_family(
            "qwen3",
            base="qwen2_5_vl_3b",
            replacement=None,
            candidate_row=candidates["qwen3"],
        ),
        verified_family(
            "qwen7",
            base="qwen2_5_vl_7b",
            replacement=None,
            candidate_row=candidates.get("qwen7"),
        ),
    ]
    rows = release_rows(list(candidates.values()))
    payload = build_multifamily_release_payload(
        deepcopy(original),
        rows,
        verified,
        metadata={},
        expected_lpm_count=expected_rows,
        expected_highmotion_count=3,
        generated_at="2026-08-20T12:00:00",
        source_artifacts=12,
        open_mos_judge=MULTIFAMILY_JUDGE["model"],
    )

    assert len(payload["tracks"]["lpm"]) == expected_rows
    assert payload["tracks"]["highmotion"] == original["tracks"]["highmotion"]
    assert payload["grtMethods"] == [
        item["selected_method"] for item in verified if item["status"] == "promoted"
    ]
    family_rows = {row["family"]: row for row in payload["grtFamilies"]}
    assert family_rows["llava7"] == {
        "family": "llava7",
        "status": "not_promoted",
        "reasonCode": "strict_quality_gate_failed",
        "method": None,
        "baseMethod": "llava_onevision_original",
        "campaignFingerprint": verified[1]["campaign_fingerprint"],
    }
    methods = {row["method"] for row in payload["tracks"]["lpm"]}
    assert "grt_llava7_failed" not in methods
    assert "grt_qwen7_failed" not in methods
    assert "llava_onevision_original" in methods
    assert "qwen2_5_vl_7b" in methods


@pytest.mark.skipif(
    not (REAL_LLAVA7_FAILURE / "full_validation.json").is_file(),
    reason="real prepared LLaVA7 full failure evidence is unavailable",
)
def test_v2_full_binding_e2e_with_real_failure_and_synthetic_promotions(
    tmp_path: Path,
) -> None:
    route_fixture = sync_test_fixture.LeaderboardSyncTests()
    route_fixture.setUp()
    try:
        route_selection = route_fixture.make_route31_selection()
        evaluation_descriptor = route_selection["artifact_provenance"]["artifacts"][
            "evaluation_contract"
        ]
        evaluation_path = Path(evaluation_descriptor["path"])
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        evaluation["dataset_revision"] = real_llava_entry()["revisions"]["dataset"]
        evaluation["judge_revision"] = MULTIFAMILY_JUDGE["revision"]
        evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
        route_selection["artifact_provenance"]["artifacts"]["evaluation_contract"] = (
            descriptor(evaluation_path)
        )
        route_artifacts = route_selection["artifact_provenance"]["artifacts"]
        route_scores = {
            **route_selection["baseline_scores"],
            **route_selection["candidate_scores"],
        }
        route_matrix_path = Path(route_artifacts["open_mos_matrix"]["path"])
        with route_matrix_path.open("w", encoding="utf-8", newline="") as handle:
            fields = (
                "sample_id",
                "doc_id",
                "method",
                "open_mos_score",
                "mos_judge_model",
                "mos_judge_revision",
                "judge_fingerprint",
                "error",
                "question",
                "answer",
            )
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for method, score in route_scores.items():
                for doc_id in range(634):
                    writer.writerow(
                        {
                            "sample_id": f"{method}::{doc_id}",
                            "doc_id": doc_id,
                            "method": method,
                            "open_mos_score": score,
                            "mos_judge_model": MULTIFAMILY_JUDGE["model"],
                            "mos_judge_revision": MULTIFAMILY_JUDGE["revision"],
                            "judge_fingerprint": MULTIFAMILY_JUDGE["fingerprint"],
                            "error": "",
                            "question": f"question {doc_id}",
                            "answer": f"answer {doc_id}",
                        }
                    )
        route_artifacts["open_mos_matrix"] = descriptor(route_matrix_path)
        route_summary_methods = {
            "summary_archived_public": "llava_onevision_0_5b",
            "summary_route31_base": "llava_onevision_0_5b_hf_route31_base",
            "summary_route31_exact": "llava_onevision_0_5b_hf_route31_exact",
            "summary_route31_candidate": ("grt_llava_onevision_0_5b_hf_route31_t0001"),
        }
        secondary_scores = {
            **route_selection["baseline_secondary_scores"],
            **route_selection["candidate_secondary_scores"],
        }
        for artifact_name, method in route_summary_methods.items():
            summary_path = Path(route_artifacts[artifact_name]["path"])
            with summary_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "method",
                        "task",
                        "run_status",
                        "samples",
                        "result_json",
                        "token_f1",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "method": method,
                        "task": "densevideo",
                        "run_status": "success",
                        "samples": 634,
                        "result_json": str(tmp_path / f"{method}_results.json"),
                        "token_f1": secondary_scores[method],
                    }
                )
            route_artifacts[artifact_name] = descriptor(summary_path)
        route_selection_path = tmp_path / "route31_selection.json"
        route_selection_path.write_text(json.dumps(route_selection), encoding="utf-8")

        dual_bundle = write_dual_bundle(
            tmp_path / "dual",
            judge_model=MULTIFAMILY_JUDGE["model"],
            judge_revision=MULTIFAMILY_JUDGE["revision"],
            judge_fingerprint=MULTIFAMILY_JUDGE["fingerprint"],
        )
        qwen_evidence = {
            family: dual_release.validate_completion(
                dual_bundle["completion"],
                family=family,
                judge_model=MULTIFAMILY_JUDGE["model"],
                judge_revision=MULTIFAMILY_JUDGE["revision"],
                judge_fingerprint=MULTIFAMILY_JUDGE["fingerprint"],
            )
            for family in ("qwen3", "qwen7")
        }
        route_entry = {
            "family": "route31",
            "status": "promoted",
            "reason_code": "strict_quality_gate_passed",
            "base_method": "llava_onevision_0_5b",
            "replaces_method": "grt_llava_ov_0_5b",
            "candidate_methods": ["grt_llava_onevision_0_5b_hf_route31_t0001"],
            "selected_method": "grt_llava_onevision_0_5b_hf_route31_t0001",
            "selection_schema": "hf_route31_full_campaign_v1",
            "completion_schema": None,
            "campaign_fingerprint": route_selection["artifact_provenance"][
                "campaign_fingerprint"
            ],
            "revisions": {
                "model": evaluation["model_revision"],
                "dataset": evaluation["dataset_revision"],
                "judge": MULTIFAMILY_JUDGE["revision"],
            },
            "evidence": {
                "kind": "passed_full_selection",
                "selection": descriptor(route_selection_path),
                "completion": None,
                "open_mos_matrix": route_artifacts["open_mos_matrix"],
                "summary_csv": route_artifacts["summary_route31_candidate"],
            },
        }
        entries = [route_entry, real_llava_entry()]
        for family, base in (
            ("qwen3", "qwen2_5_vl_3b"),
            ("qwen7", "qwen2_5_vl_7b"),
        ):
            evidence = qwen_evidence[family]
            entries.append(
                {
                    "family": family,
                    "status": "promoted",
                    "reason_code": "strict_quality_gate_passed",
                    "base_method": base,
                    "replaces_method": None,
                    "candidate_methods": [evidence["selected_method"]],
                    "selected_method": evidence["selected_method"],
                    "selection_schema": "strict_full_gate_v1",
                    "completion_schema": dual_release.COMPLETION_SCHEMA,
                    "campaign_fingerprint": evidence["campaign_fingerprint"],
                    "revisions": {
                        "model": evidence["model_revision"],
                        "dataset": evidence["dataset_revision"],
                        "judge": MULTIFAMILY_JUDGE["revision"],
                    },
                    "evidence": {
                        "kind": "passed_full_selection",
                        "selection": evidence["selection"],
                        "completion": evidence["completion"],
                        "open_mos_matrix": evidence["open_mos_matrix"],
                        "summary_csv": evidence["summary_csv"],
                    },
                }
            )
        manifest = {
            "schema_version": 2,
            "schema": "dive_grt_multifamily_release_v2",
            "benchmark": "DIVE-Bench",
            "task": "densevideo",
            "expected_samples": 634,
            "baseline_lpm_rows": 27,
            "expected_lpm_rows": 29,
            "expected_highmotion_rows": 3,
            "judge": MULTIFAMILY_JUDGE,
            "promotion_summary": {
                "promoted_families": ["route31", "qwen3", "qwen7"],
                "not_promoted_families": ["llava7"],
                "published_grt_methods": [
                    entries[0]["selected_method"],
                    entries[2]["selected_method"],
                    entries[3]["selected_method"],
                ],
                "replacement_methods": ["grt_llava_ov_0_5b"],
                "promoted_count": 3,
                "replacement_count": 1,
            },
            "families": entries,
        }
        manifest_path = tmp_path / "release_v2.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        route_raw = {
            "task": "densevideo",
            "rank": "0",
            "display_name": "GRT Route31 verified",
            "method": entries[0]["selected_method"],
            "samples": "634",
            **WINNER_VALUES,
            "source": "open",
        }
        candidate_rows = [
            route_raw,
            dual_candidate_row(qwen_evidence["qwen3"]),
            dual_candidate_row(qwen_evidence["qwen7"]),
        ]
        rows = release_rows(candidate_rows)
        leaderboard_csv = tmp_path / "leaderboard.csv"
        with leaderboard_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        leaderboard_markdown = tmp_path / "leaderboard.md"
        leaderboard_markdown.write_text("# synthetic v2\n", encoding="utf-8")

        old_revision = sync_test_fixture.JUDGE_REVISION
        old_fingerprint = sync_test_fixture.JUDGE_FINGERPRINT
        sync_test_fixture.JUDGE_REVISION = MULTIFAMILY_JUDGE["revision"]
        sync_test_fixture.JUDGE_FINGERPRINT = MULTIFAMILY_JUDGE["fingerprint"]
        try:
            legacy_binding, _, legacy_by_method, _ = _legacy_open_mos_fixture(
                tmp_path,
                {
                    "selection": route_entry["evidence"]["selection"],
                    "open_mos_matrix": route_entry["evidence"]["open_mos_matrix"],
                },
            )
        finally:
            sync_test_fixture.JUDGE_REVISION = old_revision
            sync_test_fixture.JUDGE_FINGERPRINT = old_fingerprint
        summary_descriptors = [
            route_entry["evidence"]["summary_csv"],
            entries[2]["evidence"]["summary_csv"],
            entries[3]["evidence"]["summary_csv"],
        ]
        matrix_descriptors_by_path = {
            item["path"]: item
            for item in (
                route_entry["evidence"]["open_mos_matrix"],
                entries[2]["evidence"]["open_mos_matrix"],
                entries[3]["evidence"]["open_mos_matrix"],
                *(
                    {"path": row["path"], "sha256": row["sha256"]}
                    for row in legacy_binding["methods"].values()
                ),
            )
        }
        matrix_descriptors = list(matrix_descriptors_by_path.values())
        metadata = {
            "generated_at": "2026-08-20T12:00:00",
            "open_mos_judge": MULTIFAMILY_JUDGE["model"],
            "open_mos_judge_provenance": {
                "model": MULTIFAMILY_JUDGE["model"],
                "revision": MULTIFAMILY_JUDGE["revision"],
                "judge_fingerprint": MULTIFAMILY_JUDGE["fingerprint"],
            },
            "open_mos": legacy_by_method,
            "summary_csv": [item["path"] for item in summary_descriptors],
            "open_mos_artifacts": [item["path"] for item in matrix_descriptors],
            "api_summaries": [],
            "artifact_provenance": {
                "schema_version": 3,
                "leaderboard_csv": descriptor(leaderboard_csv),
                "leaderboard_markdown": descriptor(leaderboard_markdown),
                "source_artifacts": {
                    "summary_csv": summary_descriptors,
                    "open_mos_artifacts": matrix_descriptors,
                    "api_summaries": [],
                },
                "verified_grt_selection": None,
                "grt_release_manifest": descriptor(manifest_path),
                "legacy_open_mos_completion": legacy_binding,
                "grt_family_audits": entries,
                "verified_grt_selections": [entries[0], entries[2], entries[3]],
                "not_promoted_grt_families": [entries[1]],
            },
        }
        metadata_path = tmp_path / "leaderboard_sources.json"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

        verified = validate_multifamily_release_binding(
            manifest,
            manifest_path=manifest_path,
            leaderboard_csv=leaderboard_csv,
            leaderboard_rows=rows,
            metadata=metadata,
            metadata_path=metadata_path,
        )
        assert [item["status"] for item in verified] == [
            "promoted",
            "not_promoted",
            "promoted",
            "promoted",
        ]
        tampered_metadata = deepcopy(metadata)
        tampered_metadata["artifact_provenance"]["not_promoted_grt_families"] = []
        with pytest.raises(SyncError, match="audit partition"):
            validate_multifamily_release_binding(
                manifest,
                manifest_path=manifest_path,
                leaderboard_csv=leaderboard_csv,
                leaderboard_rows=rows,
                metadata=tampered_metadata,
                metadata_path=metadata_path,
            )
        leaked_failure = deepcopy(metadata)
        leaked_descriptor = entries[1]["evidence"]["open_mos_matrix_csv"]
        leaked_failure["artifact_provenance"]["source_artifacts"][
            "open_mos_artifacts"
        ].append(leaked_descriptor)
        leaked_failure["open_mos_artifacts"].append(leaked_descriptor["path"])
        with pytest.raises(SyncError, match="outside MOS source groups"):
            validate_multifamily_release_binding(
                manifest,
                manifest_path=manifest_path,
                leaderboard_csv=leaderboard_csv,
                leaderboard_rows=rows,
                metadata=leaked_failure,
                metadata_path=metadata_path,
            )
    finally:
        route_fixture.tearDown()


def write_failure_matrix(
    csv_path: Path,
    jsonl_path: Path,
    *,
    identity_tamper: bool = False,
) -> tuple[str, ...]:
    methods = ("public", "fresh", "all", "candidate")
    rows = []
    for method in methods:
        for doc_id in range(634):
            question = f"question {doc_id}"
            if identity_tamper and method == "candidate" and doc_id == 0:
                question = "hidden tampered question"
            answer = f"answer {doc_id}"
            pred = f"prediction {doc_id}"
            score = 1 if method == "candidate" else 2
            correctness = "no"
            rows.append(
                {
                    "sample_id": f"{method}::doc_{doc_id}",
                    "doc_id": doc_id,
                    "run_name": f"{method}_densevideo_open_mos",
                    "method": method,
                    "model": method,
                    "video_name": f"video-{doc_id}",
                    "question_id": f"q-{doc_id}",
                    "type": "QA",
                    "open_mos_correctness": correctness,
                    "open_mos_score": score,
                    "mos_judge_model": MULTIFAMILY_JUDGE["model"],
                    "mos_judge_revision": MULTIFAMILY_JUDGE["revision"],
                    "request_fingerprint": _open_mos_request_fingerprint(
                        question, answer, pred
                    ),
                    "judge_fingerprint": MULTIFAMILY_JUDGE["fingerprint"],
                    "error": "",
                    "question": question,
                    "answer": answer,
                    "pred": pred,
                    "open_mos_review": json.dumps(
                        {"pred": correctness, "score": score},
                        separators=(",", ":"),
                    ),
                }
            )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=QWEN7_FLOOR_FULL_MOS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    return methods


def test_failed_matrix_pair_is_closed_typed_paired_and_recomputed(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "matrix.csv"
    jsonl_path = tmp_path / "matrix.jsonl"
    methods = write_failure_matrix(csv_path, jsonl_path)
    metrics = _inspect_failed_full_mos_pair(
        csv_path,
        jsonl_path,
        family="fixture",
        expected_methods=set(methods),
        judge=MULTIFAMILY_JUDGE,
    )
    assert metrics["candidate"]["open_mos"] == 1
    assert metrics["public"]["open_mos"] == 2

    tampered_csv = tmp_path / "extra.csv"
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    lines[1] += ",UNDECLARED"
    tampered_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(SyncError, match="CSV row schema"):
        _inspect_failed_full_mos_pair(
            tampered_csv,
            jsonl_path,
            family="fixture",
            expected_methods=set(methods),
            judge=MULTIFAMILY_JUDGE,
        )

    tampered_jsonl = tmp_path / "duplicate.jsonl"
    json_lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    json_lines[0] = json_lines[0].replace(
        '"question":', '"question":"HIDDEN","question":', 1
    )
    tampered_jsonl.write_text("\n".join(json_lines) + "\n", encoding="utf-8")
    with pytest.raises(SyncError, match="duplicate JSON key"):
        _inspect_failed_full_mos_pair(
            csv_path,
            tampered_jsonl,
            family="fixture",
            expected_methods=set(methods),
            judge=MULTIFAMILY_JUDGE,
        )


@pytest.mark.parametrize("bad_score", [1.5, True, 2.0])
def test_failed_matrix_json_score_must_be_native_integer(
    tmp_path: Path, bad_score: object
) -> None:
    csv_path = tmp_path / "matrix.csv"
    jsonl_path = tmp_path / "matrix.jsonl"
    methods = write_failure_matrix(csv_path, jsonl_path)
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["open_mos_score"] = bad_score
    lines[0] = json.dumps(row, separators=(",", ":"))
    jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(SyncError, match="typed identity"):
        _inspect_failed_full_mos_pair(
            csv_path,
            jsonl_path,
            family="fixture",
            expected_methods=set(methods),
            judge=MULTIFAMILY_JUDGE,
        )


def test_failed_matrix_csv_score_must_be_canonical_integer_string(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "matrix.csv"
    jsonl_path = tmp_path / "matrix.jsonl"
    methods = write_failure_matrix(csv_path, jsonl_path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        rows = list(reader)
    rows[0]["open_mos_score"] = "2.0"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(SyncError, match="judge/score"):
        _inspect_failed_full_mos_pair(
            csv_path,
            jsonl_path,
            family="fixture",
            expected_methods=set(methods),
            judge=MULTIFAMILY_JUDGE,
        )


@pytest.mark.parametrize("bad_score", [1.5, True, 2.0])
def test_promoted_qwen7_preflight_requires_native_integer_json_score(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_score: object,
) -> None:
    evidence = qwen7_floor_fixture.synthetic_site_evidence(tmp_path, monkeypatch)
    completed = evidence["completed"]
    assert isinstance(completed, dict)
    matrix_jsonl = Path(completed["mos_matrix_jsonl"]["path"])
    lines = matrix_jsonl.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["open_mos_score"] = bad_score
    lines[0] = json.dumps(row, separators=(",", ":"))
    matrix_jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")
    completed["mos_matrix_jsonl"] = descriptor(matrix_jsonl)
    completion_path = Path(evidence["completion_path"])
    contract_path = Path(completed["contract"]["path"])
    with pytest.raises(SyncError, match="score must be a JSON integer"):
        _validate_qwen7_floor_full_raw_artifacts(
            contract=evidence["contract"],
            completed=completed,
            contract_base=contract_path.parent,
            completion_base=completion_path.parent,
        )


def test_promoted_qwen7_preflight_requires_canonical_integer_csv_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = qwen7_floor_fixture.synthetic_site_evidence(tmp_path, monkeypatch)
    completed = evidence["completed"]
    assert isinstance(completed, dict)
    matrix_csv = Path(completed["mos_matrix_csv"]["path"])
    qwen7_floor_fixture.rewrite_csv_field(
        matrix_csv, field="open_mos_score", value="2.0"
    )
    completed["mos_matrix_csv"] = descriptor(matrix_csv)
    completion_path = Path(evidence["completion_path"])
    contract_path = Path(completed["contract"]["path"])
    with pytest.raises(SyncError, match="canonical integer string"):
        _validate_qwen7_floor_full_raw_artifacts(
            contract=evidence["contract"],
            completed=completed,
            contract_base=contract_path.parent,
            completion_base=completion_path.parent,
        )


def test_failed_matrix_rejects_cross_method_text_tamper(tmp_path: Path) -> None:
    csv_path = tmp_path / "matrix.csv"
    jsonl_path = tmp_path / "matrix.jsonl"
    methods = write_failure_matrix(csv_path, jsonl_path, identity_tamper=True)
    with pytest.raises(SyncError, match="paired identities"):
        _inspect_failed_full_mos_pair(
            csv_path,
            jsonl_path,
            family="fixture",
            expected_methods=set(methods),
            judge=MULTIFAMILY_JUDGE,
        )


def test_failed_selection_reason_is_not_a_trusted_gate_result() -> None:
    selection = {
        "status": "failed",
        "reason": "candidate definitely passed (untrusted prose)",
        "metric": "open_mos",
        "selected_method": None,
        "max_recompute_ratio": 0.98,
        "max_reference_recompute_ratio": 0.98,
        "reference_ratio_overrides": [],
    }
    _validate_failed_selection(selection, family="fixture")
    selection["selected_method"] = "forged-winner"
    with pytest.raises(SyncError, match="disposition"):
        _validate_failed_selection(selection, family="fixture")


@pytest.mark.parametrize("name", ["manifest", "metadata"])
def test_manifest_and_metadata_loaders_reject_nested_duplicate_keys(
    tmp_path: Path, name: str
) -> None:
    path = tmp_path / f"{name}.json"
    path.write_text(
        '{"artifact_provenance":{"schema_version":3,"schema_version":2}}\n',
        encoding="utf-8",
    )
    with pytest.raises(SyncError, match="duplicate JSON key"):
        _load_json(path)


@pytest.mark.parametrize("disposition", ["promoted", "failed", "contract"])
def test_hashed_json_loaders_reject_nested_duplicate_keys(
    tmp_path: Path, disposition: str
) -> None:
    path = tmp_path / f"{disposition}-selection.json"
    path.write_text(
        '{"candidate_scores":{"candidate":1,"candidate":2}}\n',
        encoding="utf-8",
    )
    with pytest.raises(SyncError, match="duplicate JSON key"):
        _load_hashed_json(path, f"{disposition} selection")


def test_site_payload_loader_rejects_duplicates_and_preserves_decimal(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.js"
    duplicate.write_text(
        'window.DIVE_LEADERBOARD = {"tracks":{"lpm":[],"lpm":[]}};\n',
        encoding="utf-8",
    )
    with pytest.raises(SyncError, match="duplicate JSON key"):
        load_site_payload(duplicate)

    precise = tmp_path / "precise.js"
    precise.write_text(
        'window.DIVE_LEADERBOARD = {"metric":0.10000000000000000001};\n',
        encoding="utf-8",
    )
    payload = load_site_payload(precise)
    assert payload["metric"] == Decimal("0.10000000000000000001")
    assert isinstance(payload["metric"], Decimal)


@pytest.mark.skipif(
    not (REAL_LLAVA7_FAILURE / "full_validation.json").is_file(),
    reason="real prepared LLaVA7 full failure evidence is unavailable",
)
def test_real_llava7_failure_preflight_and_descriptor_tamper(
    tmp_path: Path,
) -> None:
    entry = real_llava_entry()
    contract = json.loads(
        Path(entry["evidence"]["contract"]["path"]).read_text(encoding="utf-8")
    )
    verified = _validate_llava7_failed_evidence(
        entry,
        manifest_path=Path("/tmp/synthetic-v2-manifest.json"),
        judge=MULTIFAMILY_JUDGE,
    )
    assert verified["candidate"] is None
    assert len(verified["methods"]) == 4
    assert entry["evidence"]["validation"]["sha256"] == (
        "1be08185463674e44933eeb18c45a471f1b53555f5a99fe608aec832a356ac2e"
    )
    assert entry["evidence"]["selection"]["sha256"] == (
        "692e97e50eeca81686798d1d2a9917b5651df5b7e511c6fb67ddcca99d747500"
    )
    assert entry["evidence"]["open_mos_matrix_csv"]["sha256"] == (
        "76ac7d6c365130713a6c5570ac1db042a33138265cf97ef4fa8d2ea2a8200840"
    )
    assert entry["evidence"]["open_mos_matrix_jsonl"]["sha256"] == (
        "3ef325b812ad0184482112e0713341de255b3e1760153d01dab4789cb4027710"
    )

    tampered = deepcopy(entry)
    tampered["evidence"]["validation"]["sha256"] = "0" * 64
    with pytest.raises(SyncError, match="SHA-256"):
        _validate_llava7_failed_evidence(
            tampered,
            manifest_path=Path("/tmp/synthetic-v2-manifest.json"),
            judge=MULTIFAMILY_JUDGE,
        )

    self_signed = deepcopy(contract)
    self_signed["campaign_kind"] = "attacker-resigned-copy"
    self_signed["campaign_fingerprint"] = _canonical_fingerprint(self_signed)
    self_signed_path = tmp_path / "campaign_contract.json"
    self_signed_path.write_text(json.dumps(self_signed), encoding="utf-8")
    tampered = deepcopy(entry)
    tampered["campaign_fingerprint"] = self_signed["campaign_fingerprint"]
    tampered["evidence"]["contract"] = descriptor(self_signed_path)
    with pytest.raises(SyncError, match="identity/gate"):
        _validate_llava7_failed_evidence(
            tampered,
            manifest_path=Path("/tmp/synthetic-v2-manifest.json"),
            judge=MULTIFAMILY_JUDGE,
        )

    resigned_validation = tmp_path / "full_validation.json"
    resigned_validation.write_text(
        json.dumps({"status": "passed", "attacker": "re-signed"}) + "\n",
        encoding="utf-8",
    )
    resigned_pair = (resigned_validation.resolve(), sha256_file(resigned_validation))
    with pytest.raises(SyncError, match="not the pinned artifact"):
        _require_pinned_pair(
            resigned_pair,
            entry["evidence"]["validation"]["sha256"],
            "llava7 failed validation",
        )
