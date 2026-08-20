from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

from scripts import qwen_dual_release as dual


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_samples(
    path: Path,
    count: int = dual.EXPECTED_SAMPLES,
    *,
    token_f1: float = 0.01,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index in range(count):
            subtitle = index < 317
            route = "sub" if subtitle else "ocr"
            handle.write(
                json.dumps(
                    {
                        "doc_id": index,
                        "input": (
                            "What subtitles appear in the entire video?"
                            if subtitle
                            else "What text is extracted by OCR in the entire video?"
                        ),
                        "target": f"answer {index}",
                        "filtered_resps": [f"prediction {index}"],
                        "token_f1": token_f1,
                        "doc": {
                            "qid": f"fixture_{index}_{route}",
                            "video": f"video_{index}.mp4",
                            "question": (
                                "What subtitles appear in the entire video?"
                                if subtitle
                                else "What text is extracted by OCR in the entire video?"
                            ),
                            "answer": f"answer {index}",
                            "type": "QA",
                        },
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def write_summary(
    path: Path,
    *,
    method: str,
    role: str,
    result: Path,
    run_log: Path,
    token_f1: float,
) -> None:
    recomputed = 10 if role in {"base", "exact"} else 8
    total_recomputed = recomputed * dual.EXPECTED_SAMPLES
    total_orig = 10 * dual.EXPECTED_SAMPLES
    policy = {"base": "disabled", "exact": "all", "candidate": "motion"}[role]
    fields = (
        "method",
        "task",
        "run_status",
        "samples",
        "result_json",
        "run_log",
        "token_f1",
        "gate_policy",
        "gate_policy_counts",
        "mean_requested_frames",
        "mean_reference_frames",
        "total_orig_patches",
        "total_reference_orig_patches",
        "total_recomputed_patches",
        "mean_recompute_ratio",
        "reference_patch_compute_ratio",
        "mean_effective_fps",
        "mean_throughput_fps",
        "mean_wall_time_s",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "method": method,
                "task": "densevideo",
                "run_status": "success",
                "samples": dual.EXPECTED_SAMPLES,
                "result_json": str(result.resolve()),
                "run_log": str(run_log.resolve()),
                "token_f1": token_f1,
                "gate_policy": policy,
                "gate_policy_counts": json.dumps({policy: dual.EXPECTED_SAMPLES}),
                "mean_requested_frames": 8,
                "mean_reference_frames": 8,
                "total_orig_patches": total_orig,
                "total_reference_orig_patches": total_orig,
                "total_recomputed_patches": total_recomputed,
                "mean_recompute_ratio": recomputed / 10,
                "reference_patch_compute_ratio": recomputed / 10,
                "mean_effective_fps": 0.02,
                "mean_throughput_fps": 4.0,
                "mean_wall_time_s": 2.0,
            }
        )


def write_telemetry_log(path: Path, family: str, role: str) -> None:
    recomputed = 10 if role in {"base", "exact"} else 8
    ratio = recomputed / 10
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index in range(dual.EXPECTED_SAMPLES):
            handle.write(
                "[FPS_STATS] duration_s=400 orig_fps=30 total_frames=12000 "
                "capped_frames=8 sampled_frames=8 effective_fps=0.02\n"
            )
            handle.write(
                "[DENSE_METRICS] sampled_frames=8 requested_frames=8 "
                "reference_frames=8 effective_fps=0.02 wall_time_s=2 "
                "throughput_fps=4 pre_tokens=20 post_tokens=10 "
                f"recomputed_patches={recomputed} "
                "orig_patches=10 reference_orig_patches=10 "
                f"recompute_ratio={ratio} patch_projection_recompute_ratio={ratio} "
                f"patch_projection_compute_ratio_vs_reference={ratio} "
                "retention_ratio=0.5 gate_keep_ratio=0.5 "
                "post_tokens_before_prune=20 post_tokens_after_prune=10 "
                "prune_keep_ratio_actual=0.5 merge_ratio=0.5 "
                "tokenization_time_s=0.1\n"
            )
            if role == "candidate":
                route = "subtitle" if index < 317 else "ocr"
                threshold = dual.FAMILY_SPECS[family][f"{route}_threshold"]
                handle.write(
                    "[DENSE_ROUTE_GATE] "
                    f"question_route={route} "
                    "prompt_router=densevideo_lpm_first_sentence_v1 "
                    "gate_policy=motion "
                    f"gate_diff_threshold={threshold}\n"
                )


def selection_payload(family: str) -> dict[str, Any]:
    spec = dual.FAMILY_SPECS[family]
    scores = {
        str(spec["archive"]): 1.0,
        str(spec["base"]): 1.1,
        str(spec["exact"]): 1.2,
        str(spec["candidate"]): 2.0,
    }
    secondary = {
        str(spec["archive"]): 0.01,
        str(spec["base"]): 0.02,
        str(spec["exact"]): 0.03,
        str(spec["candidate"]): 0.04,
    }
    return {
        "status": "passed",
        "metric": "open_mos",
        "secondary_metric": "token_f1",
        "selected_method": spec["candidate"],
        "selected_score": 2.0,
        "selected_secondary_score": 0.04,
        "selected_recompute_ratio": 0.8,
        "selected_reference_patch_compute_ratio": 0.8,
        "required_score": 1.2,
        "secondary_required_score": 0.03,
        "max_recompute_ratio": 0.98,
        "max_reference_recompute_ratio": 0.98,
        "passing_candidates": [spec["candidate"]],
        "candidate_scores": {spec["candidate"]: scores[str(spec["candidate"])]},
        "candidate_secondary_scores": {
            spec["candidate"]: secondary[str(spec["candidate"])]
        },
        "baseline_scores": {
            method: scores[method]
            for method in (spec["archive"], spec["base"], spec["exact"])
        },
        "baseline_secondary_scores": {
            method: secondary[method]
            for method in (spec["archive"], spec["base"], spec["exact"])
        },
        "sample_counts": {
            method: dual.EXPECTED_SAMPLES
            for method in (
                spec["archive"],
                spec["base"],
                spec["exact"],
                spec["candidate"],
            )
        },
    }


def signed(path: Path, unsigned: dict[str, Any]) -> dict[str, Any]:
    payload = {**unsigned, "campaign_fingerprint": dual.canonical_fingerprint(unsigned)}
    write_json(path, payload)
    return payload


def write_dual_bundle(
    root: Path,
    *,
    judge_model: str,
    judge_revision: str,
    judge_fingerprint: str,
) -> dict[str, Any]:
    root.mkdir(parents=True)
    project_root = root / "project"
    source_paths = []
    for relative in dual.SOURCE_RELATIVE_PATHS:
        source = project_root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"fixture source: {relative}\n", encoding="utf-8")
        source_paths.append(source.resolve())
    config = source_paths[0]
    snapshot = root / "provenance" / "source_snapshot.sha256"
    snapshot.parent.mkdir(parents=True)
    source_descriptors = [dual.descriptor(path) for path in source_paths]
    snapshot.write_text(
        "".join(f"{row['sha256']}  {row['path']}\n" for row in source_descriptors),
        encoding="utf-8",
    )
    snapshot_desc = {**dual.descriptor(snapshot), "files": source_descriptors}

    archive_samples: dict[str, Path] = {}
    archive_results: dict[str, Path] = {}
    archive_summary = root / "archived_summary.csv"
    with archive_summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "run_name",
                "task",
                "method",
                "samples",
                "token_f1",
                "run_status",
                "result_json",
            ),
        )
        writer.writeheader()
        for family, spec in dual.FAMILY_SPECS.items():
            sample = root / "archives" / f"{spec['archive']}_samples_densevideo.jsonl"
            result = root / "archives" / f"{spec['archive']}_results.json"
            write_samples(sample)
            write_json(result, {"config": {"limit": dual.EXPECTED_SAMPLES}})
            archive_samples[family] = sample
            archive_results[family] = result
            writer.writerow(
                {
                    "run_name": f"{family}_archived_run",
                    "task": "densevideo",
                    "method": spec["archive"],
                    "samples": dual.EXPECTED_SAMPLES,
                    "token_f1": 0.01,
                    "run_status": "success",
                    "result_json": str(result.resolve()),
                }
            )
    archived_controls = {
        family: {
            "method": spec["archive"],
            "samples": dual.EXPECTED_SAMPLES,
            "summary_run_name": f"{family}_archived_run",
            "sample": dual.descriptor(archive_samples[family]),
            "result": dual.descriptor(archive_results[family]),
            "summary": dual.descriptor(archive_summary),
        }
        for family, spec in dual.FAMILY_SPECS.items()
    }
    parent_archives = {
        family: {
            "method": spec["archive"],
            "samples": dual.EXPECTED_SAMPLES,
            "summary_run_name": f"{family}_archived_run",
            "sample": str(archive_samples[family].resolve()),
            "sample_sha256": dual.sha256_file(archive_samples[family]),
            "result_json": str(archive_results[family].resolve()),
            "result_json_sha256": dual.sha256_file(archive_results[family]),
            "summary": str(archive_summary.resolve()),
            "summary_sha256": dual.sha256_file(archive_summary),
            "root": str(root.resolve()),
        }
        for family, spec in dual.FAMILY_SPECS.items()
    }
    parent_archives["llava7"] = {"method": "llava_onevision_original"}
    parent_path = root / "parent" / "campaign_contract.json"
    parent = signed(
        parent_path,
        {
            "schema_version": 1,
            "benchmark": "DIVE-Bench",
            "task": "densevideo",
            "dataset_revision": dual.DATASET_REVISION,
            "expected_screen_samples": dual.SCREEN_SAMPLES,
            "expected_full_samples": dual.EXPECTED_SAMPLES,
            "generation": "max_new_tokens=128,temperature=0",
            "judge_revision": judge_revision,
            "model_revisions": {
                spec["checkpoint"]: spec["revision"]
                for spec in dual.FAMILY_SPECS.values()
            },
            "source_snapshot": snapshot_desc,
            "archived_artifacts": parent_archives,
        },
    )
    oracle_dir = root / "oracles"
    oracle_dir.mkdir(parents=True)
    oracles: dict[str, dict[str, Any]] = {}
    for spec in dual.FAMILY_SPECS.values():
        for key in ("subtitle_oracle", "ocr_oracle"):
            method = str(spec[key])
            sample = oracle_dir / f"{method}_samples.jsonl"
            result = oracle_dir / f"{method}_results.json"
            summary = oracle_dir / f"{method}_summary.csv"
            sample.write_text("{}\n", encoding="utf-8")
            write_json(result, {"method": method})
            summary.write_text(f"method,samples\n{method},128\n", encoding="utf-8")
            artifact = oracle_dir / f"{method}.json"
            write_json(
                artifact,
                {
                    "schema_version": 2,
                    "status": "complete",
                    "method": method,
                    "samples": dual.SCREEN_SAMPLES,
                    "campaign_fingerprint": parent["campaign_fingerprint"],
                    "sample": dual.descriptor(sample),
                    "summary": dual.descriptor(summary),
                    "result": dual.descriptor(result),
                },
            )
            oracles[method] = {
                "method": method,
                "purpose": "threshold_design_evidence_only_not_quality_gate_input",
                "eligible_as_quality_gate_artifact": False,
                "marker": dual.descriptor(artifact),
                "sample": dual.descriptor(sample),
                "summary": dual.descriptor(summary),
                "result": dual.descriptor(result),
            }

    contract_path = root / "provenance" / "campaign_contract.json"
    unsigned_contract = {
        "schema_version": 2,
        "campaign_kind": "qwen2_5_vl_query_aware_dual_threshold",
        "project_root": str(project_root.resolve()),
        "benchmark": "DIVE-Bench",
        "task": "densevideo",
        "dataset_revision": dual.DATASET_REVISION,
        "expected_screen_samples": dual.SCREEN_SAMPLES,
        "expected_full_samples": dual.EXPECTED_SAMPLES,
        "expected_unselected_complement_samples": dual.COMPLEMENT_SAMPLES,
        "generation": {"max_new_tokens": 128, "temperature": 0},
        "plugin": {
            "environment_variable": "LMMS_EVAL_PLUGINS",
            "package": "densevideo_qwen_dual_plugin",
            "model": "qwen2_5_vl_dual_route",
        },
        "judge": {
            "model": judge_model,
            "revision": judge_revision,
            "backend": "transformers",
            "dtype": "bfloat16",
            "max_new_tokens": 64,
            "batch_size": 8,
            "trim_char_limit": 6000,
            "trust_remote_code": True,
            "temperature": 0,
            "prompt_version": "densevideo-open-mos-v1",
            "response_schema_version": "strict-json-pred-score-v1",
            "prompt_contract_sha256": dual.PROMPT_CONTRACT_SHA256,
            "judge_fingerprint": judge_fingerprint,
        },
        "families": {
            family: dual.expected_family_contract(family)
            for family in dual.FAMILY_SPECS
        },
        "full_reporting": {
            "all_634": "sole_hard_quality_gate",
            "screen_selected_128": "promotion_only_not_publishable",
            "unselected_complement_506": "reporting_only_never_a_gate",
        },
        "v2_provenance": {
            "contract": dual.descriptor(parent_path),
            "campaign_fingerprint": parent["campaign_fingerprint"],
            "source_snapshot": snapshot_desc,
            "route_oracles": oracles,
            "archived_controls": archived_controls,
        },
        "config": dual.descriptor(config),
        "output_root": str(root.resolve()),
        "source_snapshot": snapshot_desc,
    }
    contract = signed(contract_path, unsigned_contract)
    fingerprint = contract["campaign_fingerprint"]
    contract_desc = dual.descriptor(contract_path)

    selections: dict[str, dict[str, Any]] = {}
    selection_paths: dict[str, Path] = {}
    for family in dual.FAMILY_SPECS:
        payload = selection_payload(family)
        path = root / f"full_{family}_selection.json"
        write_json(path, payload)
        selections[family] = payload
        selection_paths[family] = path

    markers: dict[str, dict[str, dict[str, str]]] = {}
    sample_paths: dict[str, Path] = {}
    summary_paths: dict[str, Path] = {}
    for family, spec in dual.FAMILY_SPECS.items():
        for role in ("base", "exact", "candidate"):
            method = str(spec[role])
            model_root = (
                root
                / "full_models"
                / method
                / "runs"
                / method
                / "densevideo"
                / "fixture"
            )
            result = model_root / "fixture_results.json"
            sample = model_root / "fixture_samples_densevideo.jsonl"
            run_log = model_root / "fixture.log"
            summary = root / "full_summaries" / f"{method}.csv"
            token_f1 = {"base": 0.02, "exact": 0.03, "candidate": 0.04}[role]
            write_samples(sample, token_f1=token_f1)
            write_telemetry_log(run_log, family, role)
            model, model_args_map = dual.expected_model_args(family, role)
            model_args = ",".join(
                f"{key}={value}" for key, value in model_args_map.items()
            )
            write_json(
                result,
                {
                    "config": {
                        "model": model,
                        "model_args": model_args,
                        "gen_kwargs": {"max_new_tokens": 128, "temperature": 0},
                        "limit": dual.EXPECTED_SAMPLES,
                        "batch_size": 1,
                    }
                },
            )
            write_summary(
                summary,
                method=method,
                role=role,
                result=result,
                run_log=run_log,
                token_f1=token_f1,
            )
            recomputed = 10 if role in {"base", "exact"} else 8
            marker_path = root / "full_provenance" / f"{method}.json"
            marker_payload = {
                "schema_version": 1,
                "status": "complete",
                "stage": "full",
                "method": method,
                "samples": dual.EXPECTED_SAMPLES,
                "campaign_fingerprint": fingerprint,
                "sample": dual.descriptor(sample),
                "summary": {**dual.descriptor(summary), "row_index": 0},
                "result": dual.descriptor(result),
                "run_log": dual.descriptor(run_log),
                "resolved_contract": {
                    "model": model,
                    "model_args": model_args,
                    "generation": {"max_new_tokens": 128, "temperature": 0},
                    "limit": dual.EXPECTED_SAMPLES,
                    "batch_size": 1,
                    "telemetry": {
                        "fps_stats_rows": dual.EXPECTED_SAMPLES,
                        "dense_metrics_rows": dual.EXPECTED_SAMPLES,
                        "total_recomputed_patches": recomputed * dual.EXPECTED_SAMPLES,
                        "total_orig_patches": 10 * dual.EXPECTED_SAMPLES,
                        "total_reference_orig_patches": 10 * dual.EXPECTED_SAMPLES,
                        "recompute_ratio": recomputed / 10,
                        "reference_patch_compute_ratio": recomputed / 10,
                        "effective_fps_formula": "sampled_frames/duration_s",
                        "throughput_fps_formula": "sampled_frames/wall_time_s",
                        "recompute_ratio_formula": "sum(recomputed_patches)/sum(orig_patches)",
                        "reference_ratio_formula": "sum(recomputed_patches)/sum(reference_orig_patches)",
                    },
                },
            }
            write_json(marker_path, marker_payload)
            markers[method] = {
                "marker": dual.descriptor(marker_path),
                "sample": dual.descriptor(sample),
                "summary": dual.descriptor(summary),
                "result": dual.descriptor(result),
                "run_log": dual.descriptor(run_log),
            }
            sample_paths[method] = sample
            summary_paths[method] = summary

    score_by_method = {
        str(spec[role]): {"archive": 1.0, "base": 1.1, "exact": 1.2, "candidate": 2.0}[
            role
        ]
        for spec in dual.FAMILY_SPECS.values()
        for role in ("archive", "base", "exact", "candidate")
    }
    matrix_csv = root / "full_mos" / "matrix.csv"
    matrix_jsonl = root / "full_mos" / "matrix.jsonl"
    matrix_csv.parent.mkdir(parents=True)
    fields = (
        "sample_id",
        "doc_id",
        "method",
        "question_id",
        "video_name",
        "open_mos_score",
        "mos_judge_model",
        "mos_judge_revision",
        "judge_fingerprint",
        "error",
    )
    with (
        matrix_csv.open("w", encoding="utf-8", newline="") as csv_handle,
        matrix_jsonl.open("w", encoding="utf-8") as jsonl_handle,
    ):
        writer = csv.DictWriter(csv_handle, fieldnames=fields)
        writer.writeheader()
        for method in dual.mos_methods():
            for index in range(dual.EXPECTED_SAMPLES):
                route = "sub" if index < 317 else "ocr"
                row = {
                    "sample_id": f"{method}::{index}",
                    "doc_id": str(index),
                    "method": method,
                    "question_id": f"fixture_{index}_{route}",
                    "video_name": f"video_{index}.mp4",
                    "open_mos_score": score_by_method[method],
                    "mos_judge_model": judge_model,
                    "mos_judge_revision": judge_revision,
                    "judge_fingerprint": judge_fingerprint,
                    "error": "",
                }
                writer.writerow(row)
                jsonl_handle.write(json.dumps(row, sort_keys=True) + "\n")

    mos_root = root / "full_mos_inputs"
    mos_samples: list[dict[str, Any]] = []
    archive_summaries: dict[str, dict[str, str]] = {}
    for family, spec in dual.FAMILY_SPECS.items():
        staged_summary = root / "full_summaries" / f"{spec['archive']}.csv"
        with staged_summary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("method", "task", "run_status", "samples", "token_f1"),
            )
            writer.writeheader()
            writer.writerow(
                {
                    "method": spec["archive"],
                    "task": "densevideo",
                    "run_status": "success",
                    "samples": dual.EXPECTED_SAMPLES,
                    "token_f1": 0.01,
                }
            )
        archive_summaries[family] = {
            "method": str(spec["archive"]),
            **dual.descriptor(staged_summary),
        }
        for role in ("archive", "base", "exact", "candidate"):
            method = str(spec[role])
            source = (
                archive_samples[family] if role == "archive" else sample_paths[method]
            )
            relative = (
                Path("runs")
                / method
                / "densevideo"
                / "authorized"
                / "authorized_samples_densevideo.jsonl"
            )
            staged = mos_root / relative
            staged.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(source.resolve(), staged)
            row: dict[str, Any] = {
                "method": method,
                "role": "archived_control" if role == "archive" else role,
                "family": family,
                "source": dual.descriptor(source),
                "source_rows": dual.EXPECTED_SAMPLES,
                "rows": dual.EXPECTED_SAMPLES,
                "stage_mode": "immutable_symlink",
                "staged_sha256": dual.sha256_file(source),
                "staged_relative_path": str(relative),
            }
            if role == "archive":
                row["fresh_identity_source"] = markers[str(spec["base"])]["sample"]
            else:
                row["marker"] = markers[method]["marker"]
            mos_samples.append(row)
    mos_manifest = root / "full_summaries" / "full_sample_sources.json"
    write_json(
        mos_manifest,
        {
            "schema_version": 2,
            "status": "complete",
            "stage": "full",
            "campaign_fingerprint": fingerprint,
            "expected_samples": dual.EXPECTED_SAMPLES,
            "contract": contract_desc,
            "methods": list(dual.mos_methods()),
            "fresh_methods": list(dual.fresh_methods()),
            "archived_methods": list(dual.archive_methods()),
            "archive_summaries": archive_summaries,
            "samples": mos_samples,
        },
    )

    validation_path = root / "full_validation.json"
    write_json(
        validation_path,
        {
            "schema_version": 1,
            "status": "passed",
            "stage": "full",
            "campaign_fingerprint": fingerprint,
            "expected_samples": dual.EXPECTED_SAMPLES,
            "contract": contract_desc,
            "quality_gate_inputs": {
                family: {
                    "controls": [spec["archive"], spec["base"], spec["exact"]],
                    "candidate": spec["candidate"],
                    "route_oracles_included": False,
                }
                for family, spec in dual.FAMILY_SPECS.items()
            },
            "artifacts": markers,
            "families": {
                family: {
                    "methods": {
                        role: spec[role]
                        for role in ("archive", "base", "exact", "candidate")
                    },
                    "archived_control": archived_controls[family],
                    "typed_identity": "exact",
                    "route_counts": {"subtitle": 317, "ocr": 317, "unknown": 0},
                    "candidate_route_log_counts": {
                        "subtitle": 317,
                        "ocr": 317,
                        "unknown": 0,
                    },
                    "candidate_recompute_ratio": 0.8,
                    "candidate_reference_patch_compute_ratio": 0.8,
                    "prediction_diagnostics_only": {
                        "archive_equals_fresh": dual.EXPECTED_SAMPLES,
                        "base_equals_exact": dual.EXPECTED_SAMPLES,
                        "candidate_equals_base": dual.EXPECTED_SAMPLES,
                        "samples": dual.EXPECTED_SAMPLES,
                        "eligible_as_quality_gate_input": False,
                    },
                }
                for family, spec in dual.FAMILY_SPECS.items()
            },
        },
    )

    screen_markers: dict[str, dict[str, dict[str, str]]] = {}
    for method in dual.fresh_methods():
        marker_path = root / "screen_provenance" / f"{method}.json"
        write_json(
            marker_path,
            {
                "schema_version": 1,
                "status": "complete",
                "stage": "screen",
                "method": method,
                "samples": dual.SCREEN_SAMPLES,
                "campaign_fingerprint": fingerprint,
            },
        )
        screen_markers[method] = {"marker": dual.descriptor(marker_path)}
    screen_descriptors: dict[str, dict[str, str]] = {}
    for name in (
        "screen_validation.json",
        "screen_sample_sources.json",
        "screen_matrix.csv",
        "screen_matrix.jsonl",
    ):
        artifact = root / "screen_fixture" / name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("{}\n", encoding="utf-8")
        screen_descriptors[name] = dual.descriptor(artifact)
    screen_selections = {}
    for family in dual.FAMILY_SPECS:
        artifact = root / f"screen_{family}_selection.json"
        write_json(artifact, {"status": "passed"})
        screen_selections[family] = dual.descriptor(artifact)
    screen_completed = root / "screen_completed.json"
    write_json(
        screen_completed,
        {
            "schema_version": 1,
            "status": "complete",
            "stage": "screen",
            "campaign_fingerprint": fingerprint,
            "expected_samples": dual.SCREEN_SAMPLES,
            "contract": contract_desc,
            "markers": screen_markers,
            "validation": screen_descriptors["screen_validation.json"],
            "mos_manifest": screen_descriptors["screen_sample_sources.json"],
            "mos_matrix_csv": screen_descriptors["screen_matrix.csv"],
            "mos_matrix_jsonl": screen_descriptors["screen_matrix.jsonl"],
            "selections": screen_selections,
        },
    )

    complement_families: dict[str, Any] = {}
    subset_counts = {
        "all_634": dual.EXPECTED_SAMPLES,
        "selected_screen_128": dual.SCREEN_SAMPLES,
        "unselected_complement_506": dual.COMPLEMENT_SAMPLES,
    }
    for family, spec in dual.FAMILY_SPECS.items():
        role_values = {
            "archive": (1.0, 0.01),
            "base": (1.1, 0.02),
            "exact": (1.2, 0.03),
            "candidate": (2.0, 0.04),
        }
        metrics = {
            role: {
                "method": spec[role],
                **{
                    subset: {
                        "samples": count,
                        "open_mos": values[0],
                        "token_f1": values[1],
                    }
                    for subset, count in subset_counts.items()
                },
            }
            for role, values in role_values.items()
        }
        comparisons = {}
        for subset in subset_counts:
            comparisons[subset] = {
                metric: {
                    "candidate": metrics["candidate"][subset][metric],
                    "best_control": max(
                        metrics[role][subset][metric]
                        for role in ("archive", "base", "exact")
                    ),
                    "delta": metrics["candidate"][subset][metric]
                    - max(
                        metrics[role][subset][metric]
                        for role in ("archive", "base", "exact")
                    ),
                    "strictly_greater": True,
                }
                for metric in ("open_mos", "token_f1")
            }
            comparisons[subset]["eligible_as_quality_gate_input"] = subset == "all_634"
        complement_families[family] = {
            "metrics": metrics,
            "comparisons": comparisons,
            "identity_sets": {
                "all_634_sha256": dual.canonical_digest(
                    sorted(
                        (
                            f"fixture_{index}_{'sub' if index < 317 else 'ocr'}",
                            f"video_{index}.mp4",
                        )
                        for index in range(dual.EXPECTED_SAMPLES)
                    )
                ),
                "selected_screen_128_sha256": "b" * 64,
                "unselected_complement_506_sha256": "c" * 64,
            },
            "quality_gate_scope": "all_634",
            "all_634_is_sole_hard_quality_gate": True,
            "selected_screen_scope": "promotion_only_not_publishable",
            "unselected_complement_scope": "reporting_only",
        }
    complement = root / "full_complement_report.json"
    matrix_desc = dual.descriptor(matrix_csv)
    write_json(
        complement,
        {
            "schema_version": 1,
            "status": "complete",
            "campaign_fingerprint": fingerprint,
            "contract": contract_desc,
            "mos_matrix": matrix_desc,
            "subset_contract": {
                "full": dual.EXPECTED_SAMPLES,
                "screen_selected": dual.SCREEN_SAMPLES,
                "unselected_complement": dual.COMPLEMENT_SAMPLES,
            },
            "families": complement_families,
        },
    )

    completion = root / "full_completed.json"
    write_json(
        completion,
        {
            "schema_version": 1,
            "status": "complete",
            "stage": "full",
            "campaign_fingerprint": fingerprint,
            "expected_samples": dual.EXPECTED_SAMPLES,
            "contract": contract_desc,
            "markers": markers,
            "validation": dual.descriptor(validation_path),
            "mos_manifest": dual.descriptor(mos_manifest),
            "mos_matrix_csv": matrix_desc,
            "mos_matrix_jsonl": dual.descriptor(matrix_jsonl),
            "selections": {
                family: dual.descriptor(path)
                for family, path in selection_paths.items()
            },
            "screen_completion": dual.descriptor(screen_completed),
            "complement_report": dual.descriptor(complement),
        },
    )
    return {
        "completion": completion,
        "contract": contract_path,
        "selection_paths": selection_paths,
        "matrix": matrix_csv,
        "summaries": summary_paths,
        "fingerprint": fingerprint,
    }
