#!/usr/bin/env python3
"""Independent fail-closed verifier for Qwen dual-route release evidence.

The dual-route runner emits one atomic ``full_completed.json`` for both Qwen
families.  This module deliberately revalidates that graph without importing
the runner, so a release consumer does not have to trust the code that wrote
the completion marker.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence


COMPLETION_SCHEMA = "qwen_dual_completed_v1"
EXPECTED_SAMPLES = 634
SCREEN_SAMPLES = 128
COMPLEMENT_SAMPLES = 506
DATASET_REVISION = "5cc61a045c8e5e95d1d9c87e22ccd0f699575aea"
PROMPT_CONTRACT_SHA256 = (
    "3de3c4cd6839e56de8cec077215ad43d0ae634a605db27cffc79109e859d881d"
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")
TAGGED_FIELD_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")

FAMILY_SPECS: dict[str, dict[str, Any]] = {
    "qwen3": {
        "checkpoint": "Qwen/Qwen2.5-VL-3B-Instruct",
        "revision": "66285546d2b821cf421d4f5eb2576359d3770cd3",
        "archive": "qwen2_5_vl_3b",
        "base": "qwen2_5_vl_3b_dual_quality_base",
        "exact": "qwen2_5_vl_3b_dual_grt_all",
        "candidate": "grt_qwen2_5_vl_3b_dual_s100_o03",
        "subtitle_threshold": 10.0,
        "ocr_threshold": 0.3,
        "subtitle_oracle": "grt_qwen2_5_vl_3b_t100",
        "ocr_oracle": "grt_qwen2_5_vl_3b_t03",
    },
    "qwen7": {
        "checkpoint": "Qwen/Qwen2.5-VL-7B-Instruct",
        "revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
        "archive": "qwen2_5_vl_7b",
        "base": "qwen2_5_vl_7b_dual_quality_base",
        "exact": "qwen2_5_vl_7b_dual_grt_all",
        "candidate": "grt_qwen2_5_vl_7b_dual_s30_o300",
        "subtitle_threshold": 3.0,
        "ocr_threshold": 30.0,
        "subtitle_oracle": "grt_qwen2_5_vl_7b_t30",
        "ocr_oracle": "grt_qwen2_5_vl_7b_t300",
    },
}

SOURCE_RELATIVE_PATHS = (
    "configs/densevideo/grt_qwen_dual_route_models.yaml",
    "densevideo_qwen_dual_plugin/__init__.py",
    "densevideo_qwen_dual_plugin/models/__init__.py",
    "densevideo_qwen_dual_plugin/models/qwen2_5_vl_dual_route.py",
    "densevideo_qwen_dual_plugin/tasks/__init__.py",
    "lmms_eval/__main__.py",
    "lmms_eval/api/registry.py",
    "lmms_eval/evaluator.py",
    "lmms_eval/models/__init__.py",
    "lmms_eval/models/model_utils/load_video.py",
    "lmms_eval/models/qwen2_5_vl.py",
    "lmms_eval/tasks/densevideo/_default_template_yaml",
    "lmms_eval/tasks/densevideo/lpm_generation.yaml",
    "lmms_eval/tasks/densevideo/utils.py",
    "tools/densevideo/check_grt_quality_gate.py",
    "tools/densevideo/collect_run_metrics.py",
    "tools/densevideo/prepare_qwen_dual_route_campaign.sh",
    "tools/densevideo/public_leaderboard_env.sh",
    "tools/densevideo/qwen_dual_route_campaign.py",
    "tools/densevideo/qwen_dual_route_manifest.sh",
    "tools/densevideo/run_open_leaderboard.py",
    "tools/densevideo/run_open_mos_backfill.py",
    "tools/densevideo/score_open_mos.py",
    "tools/densevideo/score_open_mos_matrix.py",
    "tools/densevideo/slurm_grt_qwen_dual_route_full.sbatch",
    "tools/densevideo/slurm_grt_qwen_dual_route_screen.sbatch",
)

SUBTITLE_PREFIX = "what subtitles appear in the entire video"
OCR_PREFIX = "what text is extracted by ocr in the entire video"

COMPLETION_FIELDS = {
    "schema_version",
    "status",
    "stage",
    "campaign_fingerprint",
    "expected_samples",
    "contract",
    "markers",
    "validation",
    "mos_manifest",
    "mos_matrix_csv",
    "mos_matrix_jsonl",
    "selections",
    "screen_completion",
    "complement_report",
}
SCREEN_COMPLETION_FIELDS = COMPLETION_FIELDS - {
    "screen_completion",
    "complement_report",
}
CONTRACT_FIELDS = {
    "schema_version",
    "campaign_kind",
    "project_root",
    "benchmark",
    "task",
    "dataset_revision",
    "expected_screen_samples",
    "expected_full_samples",
    "expected_unselected_complement_samples",
    "generation",
    "plugin",
    "judge",
    "families",
    "full_reporting",
    "v2_provenance",
    "config",
    "output_root",
    "source_snapshot",
    "campaign_fingerprint",
}


class QwenDualReleaseError(ValueError):
    """The dual-route evidence graph is incomplete, inconsistent, or changed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_fingerprint(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("campaign_fingerprint", None)
    encoded = json.dumps(
        unsigned,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_file(raw: object, label: str) -> Path:
    raw_text = str(raw or "").strip()
    path = Path(raw_text).expanduser()
    if not path.is_absolute():
        raise QwenDualReleaseError(f"{label} path is not absolute")
    path = path.resolve()
    if not path.is_file():
        raise QwenDualReleaseError(f"missing {label}: {path}")
    return path


def descriptor(path: Path) -> dict[str, str]:
    resolved = require_file(path, "artifact")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def require_descriptor(value: object, label: str) -> tuple[Path, dict[str, str]]:
    if not isinstance(value, Mapping) or "path" not in value or "sha256" not in value:
        raise QwenDualReleaseError(f"{label} has no path/SHA descriptor")
    digest = str(value.get("sha256", ""))
    if SHA256_RE.fullmatch(digest) is None:
        raise QwenDualReleaseError(f"{label} has an invalid SHA-256")
    path = require_file(value.get("path"), label)
    if sha256_file(path) != digest:
        raise QwenDualReleaseError(f"{label} checksum mismatch: {path}")
    return path, {"path": str(path), "sha256": digest}


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QwenDualReleaseError(f"could not read {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise QwenDualReleaseError(f"{label} is not a JSON object")
    return payload


def verify_descriptor_tree(value: object, label: str) -> int:
    verified = 0
    if isinstance(value, Mapping):
        has_path = "path" in value
        has_sha = "sha256" in value
        if has_path != has_sha:
            raise QwenDualReleaseError(f"{label} has a partial descriptor")
        if has_path:
            require_descriptor(value, label)
            verified += 1
        for key, child in value.items():
            if key not in {"path", "sha256"}:
                verified += verify_descriptor_tree(child, f"{label}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            verified += verify_descriptor_tree(child, f"{label}[{index}]")
    return verified


def finite(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise QwenDualReleaseError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise QwenDualReleaseError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise QwenDualReleaseError(f"{label} must be finite")
    return result


def strict_integer(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise QwenDualReleaseError(f"{label} must be an integer")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise QwenDualReleaseError(f"{label} must be an integer") from exc
    if not number.is_finite() or number != number.to_integral_value():
        raise QwenDualReleaseError(f"{label} must be an integer")
    return int(number)


def fresh_methods() -> tuple[str, ...]:
    return tuple(
        str(spec[role])
        for spec in FAMILY_SPECS.values()
        for role in ("base", "exact", "candidate")
    )


def mos_methods() -> tuple[str, ...]:
    return tuple(
        str(spec[role])
        for spec in FAMILY_SPECS.values()
        for role in ("archive", "base", "exact", "candidate")
    )


def archive_methods() -> tuple[str, ...]:
    return tuple(str(spec["archive"]) for spec in FAMILY_SPECS.values())


def expected_family_contract(family: str) -> dict[str, Any]:
    spec = FAMILY_SPECS[family]
    return {
        "checkpoint": spec["checkpoint"],
        "revision": spec["revision"],
        "methods": {
            role: spec[role] for role in ("archive", "base", "exact", "candidate")
        },
        "quality_controls": [spec["archive"], spec["base"], spec["exact"]],
        "route_contract": {
            "classifier": "densevideo_lpm_first_sentence_v1",
            "subtitle": {
                "gate_policy": "motion",
                "l2_threshold": spec["subtitle_threshold"],
                "oracle": spec["subtitle_oracle"],
            },
            "ocr": {
                "gate_policy": "motion",
                "l2_threshold": spec["ocr_threshold"],
                "oracle": spec["ocr_oracle"],
            },
            "unknown": {"gate_policy": "all"},
        },
        "quality_gate": {
            "scope": "all_634_only",
            "primary_metric": "open_mos",
            "secondary_metric": "token_f1",
            "strictly_greater_than_all_three_controls": True,
            "max_recompute_ratio": 0.98,
            "max_reference_patch_compute_ratio": 0.98,
        },
    }


def validate_checksum_snapshot(
    value: object,
    label: str,
    *,
    expected_sources: Sequence[Path] | None = None,
) -> tuple[Path, ...]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256", "files"}:
        raise QwenDualReleaseError(f"{label} has an inexact schema")
    path, _ = require_descriptor(value, label)
    rows = value.get("files")
    if not isinstance(rows, list) or not rows:
        raise QwenDualReleaseError(f"{label} has no source inventory")
    lines: list[str] = []
    seen: set[Path] = set()
    ordered_sources: list[Path] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {"path", "sha256"}:
            raise QwenDualReleaseError(f"{label}.files[{index}] is not exact")
        source, exact = require_descriptor(row, f"{label}.files[{index}]")
        if source in seen:
            raise QwenDualReleaseError(f"{label} contains a duplicate source")
        seen.add(source)
        ordered_sources.append(source)
        lines.append(f"{exact['sha256']}  {source}\n")
    if expected_sources is not None and tuple(
        Path(str(row["path"])).expanduser().resolve() for row in rows
    ) != tuple(expected_sources):
        raise QwenDualReleaseError(f"{label} source inventory changed")
    if path.read_text(encoding="utf-8") != "".join(lines):
        raise QwenDualReleaseError(f"{label} content/inventory mismatch")
    return tuple(ordered_sources)


def validate_contract(
    path: Path,
    *,
    fingerprint: str,
    output_root: Path,
    judge_model: str,
    judge_revision: str,
    judge_fingerprint: str,
) -> dict[str, Any]:
    contract = load_json(path, "Qwen dual campaign contract")
    expected_judge = {
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
        "prompt_contract_sha256": PROMPT_CONTRACT_SHA256,
        "judge_fingerprint": judge_fingerprint,
    }
    if (
        set(contract) != CONTRACT_FIELDS
        or contract.get("campaign_fingerprint") != fingerprint
        or canonical_fingerprint(contract) != fingerprint
        or contract.get("schema_version") != 2
        or contract.get("campaign_kind") != "qwen2_5_vl_query_aware_dual_threshold"
        or contract.get("benchmark") != "DIVE-Bench"
        or contract.get("task") != "densevideo"
        or contract.get("dataset_revision") != DATASET_REVISION
        or contract.get("expected_screen_samples") != SCREEN_SAMPLES
        or contract.get("expected_full_samples") != EXPECTED_SAMPLES
        or contract.get("expected_unselected_complement_samples") != COMPLEMENT_SAMPLES
        or contract.get("generation") != {"max_new_tokens": 128, "temperature": 0}
        or contract.get("plugin")
        != {
            "environment_variable": "LMMS_EVAL_PLUGINS",
            "package": "densevideo_qwen_dual_plugin",
            "model": "qwen2_5_vl_dual_route",
        }
        or contract.get("judge") != expected_judge
        or contract.get("full_reporting")
        != {
            "all_634": "sole_hard_quality_gate",
            "screen_selected_128": "promotion_only_not_publishable",
            "unselected_complement_506": "reporting_only_never_a_gate",
        }
        or Path(str(contract.get("output_root", ""))).expanduser().resolve()
        != output_root
    ):
        raise QwenDualReleaseError("Qwen dual campaign contract semantics changed")
    project_root = Path(str(contract.get("project_root", ""))).expanduser().resolve()
    if not project_root.is_dir():
        raise QwenDualReleaseError("Qwen dual project root is not a directory")
    config_path, _ = require_descriptor(contract.get("config"), "Qwen dual config")
    expected_sources = tuple(
        (project_root / relative).resolve() for relative in SOURCE_RELATIVE_PATHS
    )
    if config_path != expected_sources[0]:
        raise QwenDualReleaseError("Qwen dual config path changed")
    validate_checksum_snapshot(
        contract.get("source_snapshot"),
        "Qwen dual snapshot",
        expected_sources=expected_sources,
    )
    if verify_descriptor_tree(contract, "Qwen dual contract") == 0:
        raise QwenDualReleaseError("Qwen dual contract binds no artifacts")

    families = contract.get("families")
    if not isinstance(families, Mapping) or set(families) != set(FAMILY_SPECS):
        raise QwenDualReleaseError("Qwen dual contract family set changed")
    for family, row in families.items():
        if not isinstance(row, Mapping):
            raise QwenDualReleaseError(f"{family} dual contract is not an object")
        if row != expected_family_contract(family):
            raise QwenDualReleaseError(f"{family} dual method/route contract changed")

    v2 = contract.get("v2_provenance")
    if not isinstance(v2, Mapping) or set(v2) != {
        "contract",
        "campaign_fingerprint",
        "source_snapshot",
        "route_oracles",
        "archived_controls",
    }:
        raise QwenDualReleaseError("Qwen dual v2 provenance schema changed")
    parent_path, _ = require_descriptor(v2.get("contract"), "Qwen dual parent contract")
    parent = load_json(parent_path, "Qwen dual parent contract")
    parent_fingerprint = str(v2.get("campaign_fingerprint", ""))
    if (
        SHA256_RE.fullmatch(parent_fingerprint) is None
        or parent.get("campaign_fingerprint") != parent_fingerprint
        or canonical_fingerprint(parent) != parent_fingerprint
        or parent.get("benchmark") != "DIVE-Bench"
        or parent.get("task") != "densevideo"
        or parent.get("dataset_revision") != DATASET_REVISION
        or parent.get("expected_screen_samples") != SCREEN_SAMPLES
        or parent.get("expected_full_samples") != EXPECTED_SAMPLES
        or parent.get("generation") != "max_new_tokens=128,temperature=0"
        or parent.get("judge_revision") != judge_revision
        or not isinstance(parent.get("model_revisions"), Mapping)
        or any(
            parent["model_revisions"].get(str(spec["checkpoint"]))
            != str(spec["revision"])
            for spec in FAMILY_SPECS.values()
        )
        or parent.get("source_snapshot") != v2.get("source_snapshot")
    ):
        raise QwenDualReleaseError("Qwen dual parent contract binding changed")
    validate_checksum_snapshot(v2.get("source_snapshot"), "Qwen dual parent snapshot")
    archives = v2.get("archived_controls")
    parent_archives = parent.get("archived_artifacts")
    if (
        not isinstance(archives, Mapping)
        or set(archives) != set(FAMILY_SPECS)
        or not isinstance(parent_archives, Mapping)
        or set(parent_archives) != {"llava7", "qwen3", "qwen7"}
    ):
        raise QwenDualReleaseError("Qwen dual archived control set changed")
    for family, spec in FAMILY_SPECS.items():
        archive = archives[family]
        if not isinstance(archive, Mapping) or set(archive) != {
            "method",
            "samples",
            "summary_run_name",
            "sample",
            "result",
            "summary",
        }:
            raise QwenDualReleaseError(f"{family} archived control schema changed")
        if (
            archive.get("method") != spec["archive"]
            or archive.get("samples") != EXPECTED_SAMPLES
            or not str(archive.get("summary_run_name", ""))
            or not isinstance(parent_archives.get(family), Mapping)
            or parent_archives[family].get("method") != spec["archive"]
            or parent_archives[family].get("samples") != EXPECTED_SAMPLES
            or parent_archives[family].get("summary_run_name")
            != archive.get("summary_run_name")
        ):
            raise QwenDualReleaseError(f"{family} archived control identity changed")
        parent_archive = parent_archives[family]
        parent_fields = {
            "sample": ("sample", "sample_sha256"),
            "result": ("result_json", "result_json_sha256"),
            "summary": ("summary", "summary_sha256"),
        }
        bound_paths: dict[str, Path] = {}
        for field, (path_field, sha_field) in parent_fields.items():
            path_value, exact = require_descriptor(
                archive.get(field), f"{family} archive {field}"
            )
            parent_path, parent_exact = require_descriptor(
                {
                    "path": parent_archive.get(path_field),
                    "sha256": parent_archive.get(sha_field),
                },
                f"{family} parent archive {field}",
            )
            if exact != parent_exact or path_value != parent_path:
                raise QwenDualReleaseError(
                    f"{family} dual/parent archive {field} differs"
                )
            bound_paths[field] = path_value
        result_path = bound_paths["result"]
        expected_sample = (
            result_path.with_name(
                f"{result_path.name[: -len('_results.json')]}_samples_densevideo.jsonl"
            )
            if result_path.name.endswith("_results.json")
            else None
        )
        if expected_sample is None or bound_paths["sample"] != expected_sample:
            raise QwenDualReleaseError(
                f"{family} archived sample/result pairing changed"
            )
        with bound_paths["summary"].open("r", encoding="utf-8", newline="") as handle:
            summary_rows = [
                row
                for row in csv.DictReader(handle)
                if row.get("method") == spec["archive"]
                and row.get("run_name") == archive["summary_run_name"]
            ]
        if (
            len(summary_rows) != 1
            or summary_rows[0].get("run_status") != "success"
            or strict_integer(
                summary_rows[0].get("samples"), f"{family}.parent_archive.samples"
            )
            != EXPECTED_SAMPLES
            or Path(str(summary_rows[0].get("result_json", ""))).expanduser().resolve()
            != result_path
        ):
            raise QwenDualReleaseError(f"{family} archived summary binding changed")
    expected_oracles = {
        str(spec[key])
        for spec in FAMILY_SPECS.values()
        for key in ("subtitle_oracle", "ocr_oracle")
    }
    oracles = v2.get("route_oracles")
    if not isinstance(oracles, Mapping) or set(oracles) != expected_oracles:
        raise QwenDualReleaseError("Qwen dual oracle set changed")
    for method, oracle in oracles.items():
        if (
            not isinstance(oracle, Mapping)
            or set(oracle)
            != {
                "method",
                "purpose",
                "eligible_as_quality_gate_artifact",
                "marker",
                "sample",
                "summary",
                "result",
            }
            or oracle.get("method") != method
            or oracle.get("eligible_as_quality_gate_artifact") is not False
            or oracle.get("purpose")
            != "threshold_design_evidence_only_not_quality_gate_input"
        ):
            raise QwenDualReleaseError(f"Qwen dual oracle policy changed for {method}")
        marker_path, _ = require_descriptor(
            oracle.get("marker"), f"Qwen dual oracle marker {method}"
        )
        marker = load_json(marker_path, f"Qwen dual oracle marker {method}")
        if (
            marker.get("schema_version") != 2
            or marker.get("status") != "complete"
            or marker.get("method") != method
            or marker.get("samples") != SCREEN_SAMPLES
            or marker.get("campaign_fingerprint") != parent_fingerprint
        ):
            raise QwenDualReleaseError(f"Qwen dual oracle marker changed for {method}")
        for field in ("sample", "summary", "result"):
            _, oracle_desc = require_descriptor(
                oracle.get(field), f"Qwen dual oracle {method}.{field}"
            )
            _, marker_desc = require_descriptor(
                marker.get(field), f"Qwen dual oracle marker {method}.{field}"
            )
            if oracle_desc != marker_desc:
                raise QwenDualReleaseError(
                    f"Qwen dual oracle marker binding changed for {method}.{field}"
                )
    return contract


def selection_score(selection: Mapping[str, Any], method: str) -> float:
    for field in ("baseline_scores", "candidate_scores"):
        values = selection.get(field)
        if isinstance(values, Mapping) and method in values:
            return finite(values[method], f"selection.{method}.open_mos")
    raise QwenDualReleaseError(f"selection has no OpenMOS score for {method}")


def selection_secondary(selection: Mapping[str, Any], method: str) -> float:
    for field in ("baseline_secondary_scores", "candidate_secondary_scores"):
        values = selection.get(field)
        if isinstance(values, Mapping) and method in values:
            return finite(values[method], f"selection.{method}.token_f1")
    raise QwenDualReleaseError(f"selection has no Token-F1 score for {method}")


def validate_selection(
    payload: Mapping[str, Any], family: str
) -> tuple[str, frozenset[str]]:
    spec = FAMILY_SPECS[family]
    baselines = {str(spec["archive"]), str(spec["base"]), str(spec["exact"])}
    candidates = {str(spec["candidate"])}
    baseline_scores = payload.get("baseline_scores")
    baseline_secondary = payload.get("baseline_secondary_scores")
    candidate_scores = payload.get("candidate_scores")
    candidate_secondary = payload.get("candidate_secondary_scores")
    selected = str(payload.get("selected_method", ""))
    if (
        payload.get("status") != "passed"
        or payload.get("metric") != "open_mos"
        or payload.get("secondary_metric") != "token_f1"
        or not isinstance(baseline_scores, Mapping)
        or set(baseline_scores) != baselines
        or not isinstance(baseline_secondary, Mapping)
        or set(baseline_secondary) != baselines
        or not isinstance(candidate_scores, Mapping)
        or set(candidate_scores) != candidates
        or not isinstance(candidate_secondary, Mapping)
        or set(candidate_secondary) != candidates
        or selected != spec["candidate"]
        or payload.get("passing_candidates") != [selected]
    ):
        raise QwenDualReleaseError(
            f"{family} dual selection method/status contract changed"
        )
    selected_score = finite(payload.get("selected_score"), f"{family}.selected_score")
    selected_secondary = finite(
        payload.get("selected_secondary_score"), f"{family}.selected_secondary"
    )
    required = finite(payload.get("required_score"), f"{family}.required_score")
    secondary_required = finite(
        payload.get("secondary_required_score"), f"{family}.secondary_required"
    )
    baseline_values = [
        finite(value, f"{family}.baseline") for value in baseline_scores.values()
    ]
    secondary_values = [
        finite(value, f"{family}.baseline_secondary")
        for value in baseline_secondary.values()
    ]
    ratio = finite(payload.get("selected_recompute_ratio"), f"{family}.ratio")
    reference = finite(
        payload.get("selected_reference_patch_compute_ratio"),
        f"{family}.reference_ratio",
    )
    if (
        selected_score
        != finite(candidate_scores[selected], f"{family}.candidate_score")
        or selected_secondary
        != finite(candidate_secondary[selected], f"{family}.candidate_secondary")
        or required != max(baseline_values)
        or secondary_required != max(secondary_values)
        or selected_score <= required
        or selected_secondary <= secondary_required
        or finite(payload.get("max_recompute_ratio"), f"{family}.max_ratio") != 0.98
        or finite(
            payload.get("max_reference_recompute_ratio"),
            f"{family}.max_reference_ratio",
        )
        != 0.98
        or not 0 <= ratio <= 0.98
        or not 0 <= reference <= 0.98
    ):
        raise QwenDualReleaseError(f"{family} dual strict full gate did not pass")
    methods = frozenset({*baselines, *candidates})
    counts = payload.get("sample_counts")
    if (
        not isinstance(counts, Mapping)
        or set(counts) != methods
        or any(
            strict_integer(value, f"{family}.sample_counts.{method}")
            != EXPECTED_SAMPLES
            for method, value in counts.items()
        )
    ):
        raise QwenDualReleaseError(f"{family} dual gate is not exactly all-634")
    return selected, methods


def expected_model_args(family: str, role: str) -> tuple[str, dict[str, str]]:
    spec = FAMILY_SPECS[family]
    common = {
        "pretrained": str(spec["checkpoint"]),
        "revision": str(spec["revision"]),
        "device_map": "auto",
        "max_num_frames": "8",
        "max_image_size": "384",
        "use_custom_video_loader": "True",
        "profiling": "True",
    }
    if role == "base":
        return "qwen2_5_vl", {**common, "use_gated_tok": "False"}
    if role == "exact":
        return "qwen2_5_vl", {
            **common,
            "use_gated_tok": "True",
            "gate_policy": "all",
            "gate_diff_threshold": f"{float(spec['ocr_threshold']):.1f}",
        }
    if role == "candidate":
        return "qwen2_5_vl_dual_route", {
            **common,
            "use_gated_tok": "True",
            "gate_policy": "motion",
            "gate_diff_threshold": f"{float(spec['ocr_threshold']):.1f}",
            "prompt_router": "densevideo_lpm_first_sentence_v1",
            "subtitle_gate_diff_threshold": (
                f"{float(spec['subtitle_threshold']):.1f}"
            ),
            "ocr_gate_diff_threshold": f"{float(spec['ocr_threshold']):.1f}",
        }
    raise QwenDualReleaseError(f"unsupported dual run role: {role}")


def parse_model_args(value: object, label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in str(value or "").split(","):
        if not item or item.count("=") != 1:
            raise QwenDualReleaseError(f"{label} contains a malformed model arg")
        key, raw = item.split("=", 1)
        if not key or key in result:
            raise QwenDualReleaseError(f"{label} contains a duplicate model arg")
        result[key] = raw
    return result


def public_route(value: object) -> str:
    first = re.split(r"[.!?]", str(value or ""), maxsplit=1)[0]
    normalized = " ".join(first.strip().lower().split())
    if normalized == SUBTITLE_PREFIX or normalized.startswith(f"{SUBTITLE_PREFIX} "):
        return "subtitle"
    if normalized == OCR_PREFIX or normalized.startswith(f"{OCR_PREFIX} "):
        return "ocr"
    return "unknown"


@dataclass(frozen=True)
class SampleEvidence:
    identities: dict[int, tuple[int, str, str, str, str, str]]
    mos_keys: frozenset[tuple[str, str]]
    route_counts: dict[str, int]
    token_f1: tuple[float, ...]
    predictions: dict[int, str]


def load_sample_identity(path: Path, label: str) -> SampleEvidence:
    identities: dict[int, tuple[int, str, str, str, str, str]] = {}
    mos_keys: set[tuple[str, str]] = set()
    routes = {"subtitle": 0, "ocr": 0, "unknown": 0}
    token_f1: list[float] = []
    predictions: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise QwenDualReleaseError(
                    f"{label}:{line_number} is not JSON"
                ) from exc
            if not isinstance(row, Mapping) or not all(
                key in row for key in ("doc_id", "input", "target", "doc")
            ):
                raise QwenDualReleaseError(
                    f"{label}:{line_number} has no typed identity"
                )
            doc = row.get("doc")
            if not isinstance(doc, Mapping):
                raise QwenDualReleaseError(
                    f"{label}:{line_number} has no typed document"
                )
            doc_id = strict_integer(row["doc_id"], f"{label}:{line_number}.doc_id")
            route = public_route(row.get("input"))
            question_route = public_route(doc.get("question"))
            qid = str(doc.get("qid", ""))
            suffix = "_sub" if route == "subtitle" else "_ocr"
            responses = row.get("filtered_resps")
            if (
                route not in {"subtitle", "ocr"}
                or question_route != route
                or not qid.endswith(suffix)
                or doc.get("type") != "QA"
                or not str(doc.get("video", "")).strip()
                or not isinstance(responses, list)
                or len(responses) != 1
                or not isinstance(responses[0], str)
            ):
                raise QwenDualReleaseError(
                    f"{label}:{line_number} typed route/prediction changed"
                )
            identity = (
                doc_id,
                qid,
                str(doc["video"]),
                str(doc.get("question", "")),
                str(doc.get("answer", "")),
                route,
            )
            mos_key = (str(doc.get("qid", "")), str(doc.get("video", "")))
            if doc_id in identities or not all(mos_key) or mos_key in mos_keys:
                raise QwenDualReleaseError(
                    f"{label}:{line_number} identity is duplicate/blank"
                )
            identities[doc_id] = identity
            mos_keys.add(mos_key)
            routes[route] += 1
            token_f1.append(
                finite(row.get("token_f1"), f"{label}:{line_number}.token_f1")
            )
            predictions[doc_id] = responses[0]
    if len(identities) != EXPECTED_SAMPLES:
        raise QwenDualReleaseError(f"{label} is not exactly 634 typed samples")
    if routes != {"subtitle": 317, "ocr": 317, "unknown": 0}:
        raise QwenDualReleaseError(f"{label} typed route counts changed")
    return SampleEvidence(
        identities=identities,
        mos_keys=frozenset(mos_keys),
        route_counts=routes,
        token_f1=tuple(token_f1),
        predictions=predictions,
    )


def inspect_matrix(
    csv_path: Path,
    jsonl_path: Path,
    *,
    selections: Mapping[str, Mapping[str, Any]],
    expected_keys: set[tuple[str, str]],
    judge_model: str,
    judge_revision: str,
    judge_fingerprint: str,
) -> dict[str, float]:
    expected_methods = set(mos_methods())
    required = {
        "sample_id",
        "method",
        "question_id",
        "video_name",
        "open_mos_score",
        "mos_judge_model",
        "mos_judge_revision",
        "judge_fingerprint",
        "error",
    }
    csv_rows: dict[str, tuple[str, tuple[str, str], float, str, str, str]] = {}
    values: dict[str, list[float]] = defaultdict(list)
    keys: dict[str, set[tuple[str, str]]] = defaultdict(set)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise QwenDualReleaseError("Qwen dual MOS CSV is missing release columns")
        for row_number, row in enumerate(reader, start=2):
            method = str(row.get("method", ""))
            sample_id = str(row.get("sample_id", ""))
            key = (str(row.get("question_id", "")), str(row.get("video_name", "")))
            score = finite(row.get("open_mos_score"), f"matrix[{row_number}].score")
            normalized = (
                method,
                key,
                score,
                str(row.get("mos_judge_model", "")),
                str(row.get("mos_judge_revision", "")),
                str(row.get("judge_fingerprint", "")),
            )
            if (
                method not in expected_methods
                or not sample_id.startswith(f"{method}::")
                or sample_id in csv_rows
                or not all(key)
                or key in keys[method]
                or str(row.get("error", ""))
                or not 0 <= score <= 5
                or normalized[3:] != (judge_model, judge_revision, judge_fingerprint)
            ):
                raise QwenDualReleaseError(
                    f"Qwen dual MOS CSV row {row_number} is invalid"
                )
            csv_rows[sample_id] = normalized
            keys[method].add(key)
            values[method].append(score)
    if (
        set(values) != expected_methods
        or any(len(values[method]) != EXPECTED_SAMPLES for method in expected_methods)
        or any(keys[method] != expected_keys for method in expected_methods)
    ):
        raise QwenDualReleaseError("Qwen dual MOS CSV is not the exact 8 x 634 matrix")

    jsonl_rows: dict[str, tuple[str, tuple[str, str], float, str, str, str]] = {}
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise QwenDualReleaseError(
                    f"Qwen dual MOS JSONL line {line_number} is invalid"
                ) from exc
            sample_id = str(row.get("sample_id", ""))
            normalized = (
                str(row.get("method", "")),
                (str(row.get("question_id", "")), str(row.get("video_name", ""))),
                finite(row.get("open_mos_score"), f"matrix_jsonl[{line_number}].score"),
                str(row.get("mos_judge_model", "")),
                str(row.get("mos_judge_revision", "")),
                str(row.get("judge_fingerprint", "")),
            )
            if sample_id in jsonl_rows or str(row.get("error", "")):
                raise QwenDualReleaseError(
                    f"Qwen dual MOS JSONL line {line_number} is duplicate/failed"
                )
            jsonl_rows[sample_id] = normalized
    if jsonl_rows != csv_rows:
        raise QwenDualReleaseError("Qwen dual MOS CSV/JSONL rows differ")
    means: dict[str, float] = {}
    for family, selection in selections.items():
        spec = FAMILY_SPECS[family]
        for method in (spec["archive"], spec["base"], spec["exact"], spec["candidate"]):
            mean = math.fsum(sorted(values[str(method)])) / EXPECTED_SAMPLES
            means[str(method)] = mean
            if not math.isclose(
                mean, selection_score(selection, str(method)), rel_tol=0, abs_tol=1e-12
            ):
                raise QwenDualReleaseError(
                    f"{family} selection OpenMOS does not reproduce the dual matrix"
                )
    return means


def summary_row(
    path: Path,
    *,
    method: str,
    result_path: Path,
    row_index: int,
) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if row_index < 0 or row_index >= len(rows):
        raise QwenDualReleaseError(f"{method} summary row index is invalid")
    row = rows[row_index]
    if (
        row.get("method") != method
        or row.get("task") != "densevideo"
        or str(row.get("run_status", "")).lower() != "success"
        or strict_integer(row.get("samples"), f"{method}.summary.samples")
        != EXPECTED_SAMPLES
        or Path(str(row.get("result_json", ""))).expanduser().resolve() != result_path
    ):
        raise QwenDualReleaseError(f"{method} summary identity/result binding changed")
    return row


def tagged_rows(path: Path, tag: str, method: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if f"[{tag}]" not in line:
                continue
            pairs = TAGGED_FIELD_RE.findall(line)
            keys = [key for key, _ in pairs]
            if len(keys) != len(set(keys)):
                raise QwenDualReleaseError(f"{method} {tag} has duplicate fields")
            rows.append(dict(pairs))
    if len(rows) != EXPECTED_SAMPLES:
        raise QwenDualReleaseError(
            f"{method} raw telemetry coverage changed: {tag}={len(rows)}"
        )
    return rows


def positive(row: Mapping[str, str], field: str, label: str) -> float:
    value = finite(row.get(field), f"{label}.{field}")
    if value <= 0:
        raise QwenDualReleaseError(f"{label}.{field} must be positive")
    return value


def validate_telemetry(
    *,
    method: str,
    role: str,
    summary: Mapping[str, str],
    run_log: Path,
    resolved: Mapping[str, Any],
) -> None:
    telemetry = resolved.get("telemetry")
    if not isinstance(telemetry, Mapping) or set(telemetry) != {
        "fps_stats_rows",
        "dense_metrics_rows",
        "total_recomputed_patches",
        "total_orig_patches",
        "total_reference_orig_patches",
        "recompute_ratio",
        "reference_patch_compute_ratio",
        "effective_fps_formula",
        "throughput_fps_formula",
        "recompute_ratio_formula",
        "reference_ratio_formula",
    }:
        raise QwenDualReleaseError(f"{method} resolved telemetry schema changed")
    if (
        telemetry.get("fps_stats_rows") != EXPECTED_SAMPLES
        or telemetry.get("dense_metrics_rows") != EXPECTED_SAMPLES
        or telemetry.get("effective_fps_formula") != "sampled_frames/duration_s"
        or telemetry.get("throughput_fps_formula") != "sampled_frames/wall_time_s"
        or telemetry.get("recompute_ratio_formula")
        != "sum(recomputed_patches)/sum(orig_patches)"
        or telemetry.get("reference_ratio_formula")
        != "sum(recomputed_patches)/sum(reference_orig_patches)"
    ):
        raise QwenDualReleaseError(f"{method} telemetry formula contract changed")
    fps_rows = tagged_rows(run_log, "FPS_STATS", method)
    dense_rows = tagged_rows(run_log, "DENSE_METRICS", method)
    recomputed_values: list[float] = []
    orig_values: list[float] = []
    reference_values: list[float] = []
    effective_values: list[float] = []
    wall_values: list[float] = []
    throughput_values: list[float] = []
    for index, (fps, dense) in enumerate(zip(fps_rows, dense_rows)):
        label = f"{method}.request[{index}]"
        duration = positive(fps, "duration_s", label)
        positive(fps, "orig_fps", label)
        total_frames = positive(fps, "total_frames", label)
        capped_frames = positive(fps, "capped_frames", label)
        sampled_fps = positive(fps, "sampled_frames", label)
        effective_fps = positive(fps, "effective_fps", label)
        sampled = positive(dense, "sampled_frames", label)
        effective = positive(dense, "effective_fps", label)
        wall = positive(dense, "wall_time_s", label)
        throughput = positive(dense, "throughput_fps", label)
        requested = positive(dense, "requested_frames", label)
        reference_frames = positive(dense, "reference_frames", label)
        pre_tokens = positive(dense, "pre_tokens", label)
        post_tokens = positive(dense, "post_tokens", label)
        recomputed = positive(dense, "recomputed_patches", label)
        orig = positive(dense, "orig_patches", label)
        reference_orig = positive(dense, "reference_orig_patches", label)
        recompute_ratio = positive(dense, "recompute_ratio", label)
        patch_ratio = positive(dense, "patch_projection_recompute_ratio", label)
        reference_ratio = positive(
            dense, "patch_projection_compute_ratio_vs_reference", label
        )
        for field in (
            "retention_ratio",
            "gate_keep_ratio",
            "post_tokens_before_prune",
            "post_tokens_after_prune",
            "prune_keep_ratio_actual",
            "merge_ratio",
            "tokenization_time_s",
        ):
            positive(dense, field, label)
        if not (
            total_frames >= capped_frames >= sampled
            and sampled == sampled_fps
            and sampled <= requested
            and sampled <= reference_frames
            and pre_tokens >= post_tokens
            and recomputed <= orig
            and recomputed <= reference_orig
            and math.isclose(
                effective_fps, sampled / duration, rel_tol=2e-5, abs_tol=2e-6
            )
            and math.isclose(effective, effective_fps, rel_tol=2e-5, abs_tol=2e-6)
            and math.isclose(throughput, sampled / wall, rel_tol=2e-5, abs_tol=2e-6)
            and math.isclose(
                recompute_ratio, recomputed / orig, rel_tol=2e-5, abs_tol=2e-6
            )
            and math.isclose(patch_ratio, recomputed / orig, rel_tol=2e-5, abs_tol=2e-6)
            and math.isclose(
                reference_ratio,
                recomputed / reference_orig,
                rel_tol=2e-5,
                abs_tol=2e-6,
            )
        ):
            raise QwenDualReleaseError(f"{label} telemetry formula/ordering changed")
        recomputed_values.append(recomputed)
        orig_values.append(orig)
        reference_values.append(reference_orig)
        effective_values.append(effective)
        wall_values.append(wall)
        throughput_values.append(throughput)
    recomputed_total = math.fsum(sorted(recomputed_values))
    orig_total = math.fsum(sorted(orig_values))
    reference_total = math.fsum(sorted(reference_values))
    recompute_ratio = recomputed_total / orig_total
    reference_ratio = recomputed_total / reference_total
    summary_expected = {
        "total_recomputed_patches": recomputed_total,
        "total_orig_patches": orig_total,
        "total_reference_orig_patches": reference_total,
        "mean_recompute_ratio": recompute_ratio,
        "reference_patch_compute_ratio": reference_ratio,
        "mean_effective_fps": math.fsum(sorted(effective_values)) / EXPECTED_SAMPLES,
        "mean_wall_time_s": math.fsum(sorted(wall_values)) / EXPECTED_SAMPLES,
        "mean_throughput_fps": math.fsum(sorted(throughput_values)) / EXPECTED_SAMPLES,
    }
    for field, expected in summary_expected.items():
        if not math.isclose(
            finite(summary.get(field), f"{method}.summary.{field}"),
            expected,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise QwenDualReleaseError(
                f"{method} summary/raw telemetry differs for {field}"
            )
    marker_expected = {
        "total_recomputed_patches": recomputed_total,
        "total_orig_patches": orig_total,
        "total_reference_orig_patches": reference_total,
        "recompute_ratio": recompute_ratio,
        "reference_patch_compute_ratio": reference_ratio,
    }
    for field, expected in marker_expected.items():
        if not math.isclose(
            finite(telemetry.get(field), f"{method}.marker.{field}"),
            expected,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise QwenDualReleaseError(
                f"{method} marker/raw telemetry differs for {field}"
            )
    policy = {"base": "disabled", "exact": "all", "candidate": "motion"}[role]
    try:
        policy_counts = json.loads(str(summary.get("gate_policy_counts", "")))
    except json.JSONDecodeError as exc:
        raise QwenDualReleaseError(f"{method} gate_policy_counts is invalid") from exc
    if (
        summary.get("gate_policy") != policy
        or policy_counts != {policy: EXPECTED_SAMPLES}
        or finite(summary.get("mean_requested_frames"), f"{method}.requested") != 8
        or finite(summary.get("mean_reference_frames"), f"{method}.reference") != 8
    ):
        raise QwenDualReleaseError(f"{method} routing/frame telemetry changed")
    if role in {"base", "exact"}:
        if not (
            math.isclose(recompute_ratio, 1.0, rel_tol=0, abs_tol=1e-12)
            and math.isclose(reference_ratio, 1.0, rel_tol=0, abs_tol=1e-12)
        ):
            raise QwenDualReleaseError(
                f"{method} control telemetry is not exact compute"
            )
    elif not (0 <= recompute_ratio <= 0.98 and 0 <= reference_ratio <= 0.98):
        raise QwenDualReleaseError(f"{method} candidate telemetry exceeds compute caps")


def validate_candidate_route_log(
    path: Path,
    *,
    method: str,
    spec: Mapping[str, Any],
    expected_routes: Mapping[str, int],
) -> None:
    counts = {"subtitle": 0, "ocr": 0, "unknown": 0}
    for row_number, fields in enumerate(
        tagged_rows(path, "DENSE_ROUTE_GATE", method), start=1
    ):
        route = str(fields.get("question_route", ""))
        if route not in counts:
            raise QwenDualReleaseError(
                f"{method} route log {row_number} has unknown route"
            )
        policy = "all" if route == "unknown" else "motion"
        expected_threshold = (
            None if route == "unknown" else float(spec[f"{route}_threshold"])
        )
        raw_threshold = fields.get("gate_diff_threshold")
        threshold_matches = (
            raw_threshold == "none"
            if expected_threshold is None
            else raw_threshold is not None
            and math.isclose(
                finite(raw_threshold, f"{method}.route[{row_number}].threshold"),
                expected_threshold,
                rel_tol=0,
                abs_tol=1e-12,
            )
        )
        if (
            fields.get("prompt_router") != "densevideo_lpm_first_sentence_v1"
            or fields.get("gate_policy") != policy
            or not threshold_matches
        ):
            raise QwenDualReleaseError(
                f"{method} route log {row_number} contract changed"
            )
        counts[route] += 1
    if counts != dict(expected_routes):
        raise QwenDualReleaseError(f"{method} route log/sample counts differ")


def validate_marker_graph(
    completed: Mapping[str, Any],
    *,
    output_root: Path,
    fingerprint: str,
) -> tuple[
    dict[str, SampleEvidence],
    dict[str, tuple[Path, dict[str, str]]],
]:
    markers = completed.get("markers")
    if not isinstance(markers, Mapping) or set(markers) != set(fresh_methods()):
        raise QwenDualReleaseError("Qwen dual marker set is not exactly six methods")
    sample_identities: dict[str, SampleEvidence] = {}
    summaries: dict[str, tuple[Path, dict[str, str]]] = {}
    for family, spec in FAMILY_SPECS.items():
        for role in ("base", "exact", "candidate"):
            method = str(spec[role])
            evidence = markers[method]
            if not isinstance(evidence, Mapping) or set(evidence) != {
                "marker",
                "sample",
                "summary",
                "result",
                "run_log",
            }:
                raise QwenDualReleaseError(f"completion evidence changed for {method}")
            paths: dict[str, Path] = {}
            exact: dict[str, dict[str, str]] = {}
            for field, value in evidence.items():
                paths[field], exact[field] = require_descriptor(
                    value, f"{method}.{field}"
                )
            if paths["marker"] != output_root / "full_provenance" / f"{method}.json":
                raise QwenDualReleaseError(f"{method} marker path changed")
            if paths["summary"] != output_root / "full_summaries" / f"{method}.csv":
                raise QwenDualReleaseError(f"{method} summary path changed")
            model_root = output_root / "full_models" / method
            if any(
                model_root not in paths[field].parents
                for field in ("sample", "result", "run_log")
            ):
                raise QwenDualReleaseError(f"{method} model artifact path changed")
            result_name = paths["result"].name
            if not result_name.endswith("_results.json") or paths["sample"] != paths[
                "result"
            ].with_name(
                f"{result_name[: -len('_results.json')]}_samples_densevideo.jsonl"
            ):
                raise QwenDualReleaseError(f"{method} result/sample pairing changed")
            marker = load_json(paths["marker"], f"{method} marker")
            if (
                set(marker)
                != {
                    "schema_version",
                    "status",
                    "stage",
                    "method",
                    "samples",
                    "campaign_fingerprint",
                    "sample",
                    "summary",
                    "result",
                    "run_log",
                    "resolved_contract",
                }
                or marker.get("schema_version") != 1
                or marker.get("status") != "complete"
                or marker.get("stage") != "full"
                or marker.get("method") != method
                or marker.get("samples") != EXPECTED_SAMPLES
                or marker.get("campaign_fingerprint") != fingerprint
            ):
                raise QwenDualReleaseError(f"{method} marker identity/schema changed")
            for field in ("sample", "summary", "result", "run_log"):
                _, marker_desc = require_descriptor(
                    marker.get(field), f"{method}.{field}"
                )
                if marker_desc != exact[field]:
                    raise QwenDualReleaseError(
                        f"{method} marker/completion hash differs"
                    )
            resolved = marker.get("resolved_contract")
            result = load_json(paths["result"], f"{method} result")
            result_config = result.get("config")
            expected_model, expected_args = expected_model_args(family, role)
            if (
                not isinstance(resolved, Mapping)
                or set(resolved)
                != {
                    "model",
                    "model_args",
                    "generation",
                    "limit",
                    "batch_size",
                    "telemetry",
                }
                or not isinstance(result_config, Mapping)
                or resolved.get("model") != result_config.get("model")
                or resolved.get("model_args") != result_config.get("model_args")
                or resolved.get("model") != expected_model
                or parse_model_args(
                    resolved.get("model_args"), f"{method}.resolved.model_args"
                )
                != expected_args
                or resolved.get("generation")
                != {"max_new_tokens": 128, "temperature": 0}
                or result_config.get("gen_kwargs") != resolved.get("generation")
                or strict_integer(resolved.get("limit"), f"{method}.limit")
                != EXPECTED_SAMPLES
                or strict_integer(result_config.get("limit"), f"{method}.result.limit")
                != EXPECTED_SAMPLES
                or str(resolved.get("batch_size")) != "1"
                or str(result_config.get("batch_size")) != "1"
            ):
                raise QwenDualReleaseError(
                    f"{method} resolved inference contract changed"
                )
            row_index = strict_integer(
                marker.get("summary", {}).get("row_index"), f"{method}.row_index"
            )
            row = summary_row(
                paths["summary"],
                method=method,
                result_path=paths["result"],
                row_index=row_index,
            )
            validate_telemetry(
                method=method,
                role=role,
                summary=row,
                run_log=paths["run_log"],
                resolved=resolved,
            )
            sample_evidence = load_sample_identity(
                paths["sample"], f"{family}.{method}.sample"
            )
            if role == "candidate":
                validate_candidate_route_log(
                    paths["run_log"],
                    method=method,
                    spec=spec,
                    expected_routes=sample_evidence.route_counts,
                )
            sample_identities[method] = sample_evidence
            summaries[method] = (paths["summary"], row)
    return sample_identities, summaries


def validate_mos_manifest(
    path: Path,
    *,
    output_root: Path,
    fingerprint: str,
    contract_descriptor: Mapping[str, str],
    markers: Mapping[str, Any],
    contract: Mapping[str, Any],
    selections: Mapping[str, Mapping[str, Any]],
) -> None:
    payload = load_json(path, "Qwen dual MOS manifest")
    samples = payload.get("samples")
    summaries = payload.get("archive_summaries")
    if (
        set(payload)
        != {
            "schema_version",
            "status",
            "stage",
            "campaign_fingerprint",
            "expected_samples",
            "contract",
            "methods",
            "fresh_methods",
            "archived_methods",
            "archive_summaries",
            "samples",
        }
        or payload.get("schema_version") != 2
        or payload.get("status") != "complete"
        or payload.get("stage") != "full"
        or payload.get("campaign_fingerprint") != fingerprint
        or payload.get("expected_samples") != EXPECTED_SAMPLES
        or payload.get("contract") != contract_descriptor
        or payload.get("methods") != list(mos_methods())
        or payload.get("fresh_methods") != list(fresh_methods())
        or payload.get("archived_methods") != list(archive_methods())
        or not isinstance(summaries, Mapping)
        or set(summaries) != set(FAMILY_SPECS)
        or not isinstance(samples, list)
        or [row.get("method") for row in samples if isinstance(row, Mapping)]
        != list(mos_methods())
    ):
        raise QwenDualReleaseError("Qwen dual MOS manifest contract changed")
    by_method = {
        str(row.get("method", "")): row for row in samples if isinstance(row, Mapping)
    }
    archives = contract["v2_provenance"]["archived_controls"]
    for family, spec in FAMILY_SPECS.items():
        archive_summary = summaries[family]
        if (
            not isinstance(archive_summary, Mapping)
            or set(archive_summary)
            != {
                "method",
                "path",
                "sha256",
            }
            or archive_summary.get("method") != spec["archive"]
        ):
            raise QwenDualReleaseError(f"{family} archive summary descriptor changed")
        summary_path, _ = require_descriptor(
            archive_summary, f"{family} staged archive summary"
        )
        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 1:
            raise QwenDualReleaseError(
                f"{family} staged archive summary is not one row"
            )
        row = rows[0]
        if (
            row.get("method") != spec["archive"]
            or row.get("task") != "densevideo"
            or row.get("run_status") != "success"
            or strict_integer(row.get("samples"), f"{family}.archive.samples")
            != EXPECTED_SAMPLES
            or not math.isclose(
                finite(row.get("token_f1"), f"{family}.archive.token_f1"),
                selection_secondary(selections[family], str(spec["archive"])),
                rel_tol=0,
                abs_tol=1e-12,
            )
        ):
            raise QwenDualReleaseError(
                f"{family} archive summary differs from selection"
            )
        for role in ("archive", "base", "exact", "candidate"):
            method = str(spec[role])
            sample = by_method[method]
            completion_role = "archived_control" if role == "archive" else role
            expected_fields = {
                "method",
                "role",
                "family",
                "source",
                "source_rows",
                "rows",
                "stage_mode",
                "staged_sha256",
                "staged_relative_path",
                "fresh_identity_source" if role == "archive" else "marker",
            }
            if (
                not isinstance(sample, Mapping)
                or set(sample) != expected_fields
                or sample.get("role") != completion_role
                or sample.get("family") != family
                or sample.get("source_rows") != EXPECTED_SAMPLES
                or sample.get("rows") != EXPECTED_SAMPLES
                or sample.get("stage_mode") != "immutable_symlink"
                or SHA256_RE.fullmatch(str(sample.get("staged_sha256", ""))) is None
            ):
                raise QwenDualReleaseError(
                    f"{family} MOS source entry changed for {method}"
                )
            source_path, source_desc = require_descriptor(
                sample.get("source"), f"{family} MOS source {method}"
            )
            expected_source = (
                archives[family]["sample"]
                if role == "archive"
                else markers[method]["sample"]
            )
            _, expected_source_desc = require_descriptor(
                expected_source, f"{family} expected MOS source {method}"
            )
            if source_desc != expected_source_desc:
                raise QwenDualReleaseError(
                    f"{family} MOS source binding changed for {method}"
                )
            relative = Path(str(sample.get("staged_relative_path", "")))
            if relative.is_absolute() or ".." in relative.parts:
                raise QwenDualReleaseError(f"{family} MOS path escapes staging root")
            staged = output_root / "full_mos_inputs" / relative
            if (
                not staged.is_symlink()
                or not staged.is_file()
                or staged.resolve() != source_path
                or sha256_file(staged) != sample.get("staged_sha256")
            ):
                raise QwenDualReleaseError(
                    f"{family} staged MOS sample changed for {method}"
                )
            if role == "archive":
                _, fresh_desc = require_descriptor(
                    sample.get("fresh_identity_source"),
                    f"{family} fresh identity source",
                )
                if fresh_desc != markers[str(spec["base"])]["sample"]:
                    raise QwenDualReleaseError(
                        f"{family} archive/fresh MOS binding changed"
                    )
            else:
                _, marker_desc = require_descriptor(
                    sample.get("marker"), f"{family} MOS marker {method}"
                )
                if marker_desc != markers[method]["marker"]:
                    raise QwenDualReleaseError(f"{family} MOS marker binding changed")


def validate_full_validation(
    path: Path,
    *,
    fingerprint: str,
    contract_descriptor: Mapping[str, str],
    markers: Mapping[str, Any],
    contract: Mapping[str, Any],
    selections: Mapping[str, Mapping[str, Any]],
    sample_evidence: Mapping[str, Mapping[str, SampleEvidence]],
) -> None:
    payload = load_json(path, "Qwen dual full validation")
    expected_inputs = {
        family: {
            "controls": [spec["archive"], spec["base"], spec["exact"]],
            "candidate": spec["candidate"],
            "route_oracles_included": False,
        }
        for family, spec in FAMILY_SPECS.items()
    }
    reports = payload.get("families")
    if (
        set(payload)
        != {
            "schema_version",
            "status",
            "stage",
            "campaign_fingerprint",
            "expected_samples",
            "contract",
            "quality_gate_inputs",
            "artifacts",
            "families",
        }
        or payload.get("schema_version") != 1
        or payload.get("status") != "passed"
        or payload.get("stage") != "full"
        or payload.get("campaign_fingerprint") != fingerprint
        or payload.get("expected_samples") != EXPECTED_SAMPLES
        or payload.get("contract") != contract_descriptor
        or payload.get("quality_gate_inputs") != expected_inputs
        or payload.get("artifacts") != markers
        or not isinstance(reports, Mapping)
        or set(reports) != set(FAMILY_SPECS)
    ):
        raise QwenDualReleaseError("Qwen dual full validation contract changed")
    archives = contract["v2_provenance"]["archived_controls"]
    for family, spec in FAMILY_SPECS.items():
        report = reports[family]
        if not isinstance(report, Mapping) or set(report) != {
            "methods",
            "archived_control",
            "typed_identity",
            "route_counts",
            "candidate_route_log_counts",
            "candidate_recompute_ratio",
            "candidate_reference_patch_compute_ratio",
            "prediction_diagnostics_only",
        }:
            raise QwenDualReleaseError(f"{family} full validation schema changed")
        routes = {"subtitle": 317, "ocr": 317, "unknown": 0}
        diagnostics = report.get("prediction_diagnostics_only")
        family_evidence = sample_evidence[family]
        actual_diagnostics = {
            "archive_equals_fresh": sum(
                family_evidence["archive"].predictions[doc_id]
                == family_evidence["base"].predictions[doc_id]
                for doc_id in sorted(family_evidence["base"].predictions)
            ),
            "base_equals_exact": sum(
                family_evidence["base"].predictions[doc_id]
                == family_evidence["exact"].predictions[doc_id]
                for doc_id in sorted(family_evidence["base"].predictions)
            ),
            "candidate_equals_base": sum(
                family_evidence["candidate"].predictions[doc_id]
                == family_evidence["base"].predictions[doc_id]
                for doc_id in sorted(family_evidence["base"].predictions)
            ),
            "samples": EXPECTED_SAMPLES,
            "eligible_as_quality_gate_input": False,
        }
        if (
            report.get("methods")
            != {role: spec[role] for role in ("archive", "base", "exact", "candidate")}
            or report.get("archived_control") != archives[family]
            or report.get("typed_identity") != "exact"
            or report.get("route_counts") != routes
            or report.get("candidate_route_log_counts") != routes
            or family_evidence["base"].route_counts != routes
            or not isinstance(diagnostics, Mapping)
            or set(diagnostics)
            != {
                "archive_equals_fresh",
                "base_equals_exact",
                "candidate_equals_base",
                "samples",
                "eligible_as_quality_gate_input",
            }
            or diagnostics.get("samples") != EXPECTED_SAMPLES
            or diagnostics.get("eligible_as_quality_gate_input") is not False
            or diagnostics != actual_diagnostics
            or any(
                strict_integer(diagnostics.get(field), f"{family}.{field}")
                not in range(EXPECTED_SAMPLES + 1)
                for field in (
                    "archive_equals_fresh",
                    "base_equals_exact",
                    "candidate_equals_base",
                )
            )
            or not math.isclose(
                finite(report.get("candidate_recompute_ratio"), f"{family}.ratio"),
                finite(
                    selections[family].get("selected_recompute_ratio"),
                    f"{family}.selected_ratio",
                ),
                rel_tol=0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                finite(
                    report.get("candidate_reference_patch_compute_ratio"),
                    f"{family}.reference_ratio",
                ),
                finite(
                    selections[family].get("selected_reference_patch_compute_ratio"),
                    f"{family}.selected_reference",
                ),
                rel_tol=0,
                abs_tol=1e-12,
            )
        ):
            raise QwenDualReleaseError(f"{family} typed/route validation changed")


def validate_complement(
    path: Path,
    *,
    fingerprint: str,
    contract_descriptor: Mapping[str, str],
    matrix_descriptor: Mapping[str, str],
    selections: Mapping[str, Mapping[str, Any]],
    matrix_means: Mapping[str, float],
    sample_evidence: Mapping[str, Mapping[str, SampleEvidence]],
) -> None:
    payload = load_json(path, "Qwen dual complement report")
    reports = payload.get("families")
    if (
        set(payload)
        != {
            "schema_version",
            "status",
            "campaign_fingerprint",
            "contract",
            "mos_matrix",
            "subset_contract",
            "families",
        }
        or payload.get("schema_version") != 1
        or payload.get("status") != "complete"
        or payload.get("campaign_fingerprint") != fingerprint
        or payload.get("contract") != contract_descriptor
        or payload.get("mos_matrix") != matrix_descriptor
        or payload.get("subset_contract")
        != {
            "full": EXPECTED_SAMPLES,
            "screen_selected": SCREEN_SAMPLES,
            "unselected_complement": COMPLEMENT_SAMPLES,
        }
        or not isinstance(reports, Mapping)
        or set(reports) != set(FAMILY_SPECS)
    ):
        raise QwenDualReleaseError("Qwen dual complement contract changed")
    subset_counts = {
        "all_634": EXPECTED_SAMPLES,
        "selected_screen_128": SCREEN_SAMPLES,
        "unselected_complement_506": COMPLEMENT_SAMPLES,
    }
    for family, spec in FAMILY_SPECS.items():
        report = reports[family]
        if not isinstance(report, Mapping) or set(report) != {
            "metrics",
            "comparisons",
            "identity_sets",
            "quality_gate_scope",
            "all_634_is_sole_hard_quality_gate",
            "selected_screen_scope",
            "unselected_complement_scope",
        }:
            raise QwenDualReleaseError(f"{family} complement schema changed")
        metrics = report.get("metrics")
        comparisons = report.get("comparisons")
        identities = report.get("identity_sets")
        if (
            report.get("quality_gate_scope") != "all_634"
            or report.get("all_634_is_sole_hard_quality_gate") is not True
            or report.get("selected_screen_scope") != "promotion_only_not_publishable"
            or report.get("unselected_complement_scope") != "reporting_only"
            or not isinstance(metrics, Mapping)
            or set(metrics) != {"archive", "base", "exact", "candidate"}
            or not isinstance(comparisons, Mapping)
            or set(comparisons) != set(subset_counts)
            or not isinstance(identities, Mapping)
            or set(identities)
            != {
                "all_634_sha256",
                "selected_screen_128_sha256",
                "unselected_complement_506_sha256",
            }
            or any(
                SHA256_RE.fullmatch(str(value)) is None for value in identities.values()
            )
        ):
            raise QwenDualReleaseError(
                f"{family} complement hard-gate semantics changed"
            )
        role_methods = {
            role: str(spec[role]) for role in ("archive", "base", "exact", "candidate")
        }
        family_evidence = sample_evidence[family]
        expected_all_identity = canonical_digest(
            sorted(family_evidence["base"].mos_keys)
        )
        if identities.get("all_634_sha256") != expected_all_identity:
            raise QwenDualReleaseError(f"{family} all-634 identity fingerprint changed")
        for role, method in role_methods.items():
            row = metrics[role]
            if (
                not isinstance(row, Mapping)
                or set(row) != {"method", *subset_counts}
                or row.get("method") != method
            ):
                raise QwenDualReleaseError(f"{family} complement role method changed")
            for subset, count in subset_counts.items():
                values = row[subset]
                if (
                    not isinstance(values, Mapping)
                    or set(values) != {"samples", "token_f1", "open_mos"}
                    or values.get("samples") != count
                    or not 0
                    <= finite(values.get("open_mos"), f"{family}.{role}.{subset}.mos")
                    <= 5
                ):
                    raise QwenDualReleaseError(
                        f"{family} complement metric coverage changed"
                    )
                finite(values.get("token_f1"), f"{family}.{role}.{subset}.f1")
            full_values = row["all_634"]
            actual_f1 = (
                math.fsum(sorted(family_evidence[role].token_f1)) / EXPECTED_SAMPLES
            )
            if not math.isclose(
                finite(full_values.get("token_f1"), f"{family}.{role}.all_f1"),
                actual_f1,
                rel_tol=0,
                abs_tol=1e-12,
            ) or not math.isclose(
                finite(full_values.get("open_mos"), f"{family}.{role}.all_mos"),
                matrix_means[method],
                rel_tol=0,
                abs_tol=1e-12,
            ):
                raise QwenDualReleaseError(
                    f"{family} all-634 report does not reproduce raw evidence"
                )
        for subset in subset_counts:
            comparison = comparisons[subset]
            if (
                not isinstance(comparison, Mapping)
                or set(comparison)
                != {
                    "open_mos",
                    "token_f1",
                    "eligible_as_quality_gate_input",
                }
                or comparison.get("eligible_as_quality_gate_input")
                is not (subset == "all_634")
            ):
                raise QwenDualReleaseError(f"{family} complement eligibility changed")
            for metric in ("open_mos", "token_f1"):
                row = comparison[metric]
                if not isinstance(row, Mapping) or set(row) != {
                    "candidate",
                    "best_control",
                    "delta",
                    "strictly_greater",
                }:
                    raise QwenDualReleaseError(
                        f"{family} complement comparison changed"
                    )
                candidate = finite(
                    row.get("candidate"), f"{family}.{subset}.{metric}.candidate"
                )
                controls = [
                    finite(
                        metrics[role][subset][metric],
                        f"{family}.{role}.{subset}.{metric}",
                    )
                    for role in ("archive", "base", "exact")
                ]
                best = finite(
                    row.get("best_control"), f"{family}.{subset}.{metric}.best"
                )
                delta = finite(row.get("delta"), f"{family}.{subset}.{metric}.delta")
                if (
                    candidate
                    != finite(
                        metrics["candidate"][subset][metric],
                        f"{family}.candidate.{metric}",
                    )
                    or best != max(controls)
                    or not math.isclose(
                        delta, candidate - best, rel_tol=0, abs_tol=1e-12
                    )
                    or row.get("strictly_greater") is not (candidate > best)
                    or (subset == "all_634" and row.get("strictly_greater") is not True)
                ):
                    raise QwenDualReleaseError(
                        f"{family} complement arithmetic/gate changed"
                    )
        selection = selections[family]
        if not all(
            math.isclose(left, right, rel_tol=0, abs_tol=1e-12)
            for left, right in (
                (
                    finite(
                        metrics["candidate"]["all_634"]["open_mos"], "candidate MOS"
                    ),
                    finite(selection.get("selected_score"), "selected MOS"),
                ),
                (
                    finite(metrics["candidate"]["all_634"]["token_f1"], "candidate F1"),
                    finite(selection.get("selected_secondary_score"), "selected F1"),
                ),
                (
                    finite(
                        comparisons["all_634"]["open_mos"]["best_control"], "MOS floor"
                    ),
                    finite(selection.get("required_score"), "selection MOS floor"),
                ),
                (
                    finite(
                        comparisons["all_634"]["token_f1"]["best_control"], "F1 floor"
                    ),
                    finite(
                        selection.get("secondary_required_score"), "selection F1 floor"
                    ),
                ),
            )
        ):
            raise QwenDualReleaseError(
                f"{family} all-634 report differs from selection"
            )


def validate_screen_completion(
    path: Path,
    *,
    fingerprint: str,
    contract_descriptor: Mapping[str, str],
) -> None:
    payload = load_json(path, "Qwen dual screen completion")
    markers = payload.get("markers")
    if (
        set(payload) != SCREEN_COMPLETION_FIELDS
        or payload.get("schema_version") != 1
        or payload.get("status") != "complete"
        or payload.get("stage") != "screen"
        or payload.get("campaign_fingerprint") != fingerprint
        or payload.get("expected_samples") != SCREEN_SAMPLES
        or payload.get("contract") != contract_descriptor
        or not isinstance(markers, Mapping)
        or set(markers) != set(fresh_methods())
        or set((payload.get("selections") or {})) != set(FAMILY_SPECS)
    ):
        raise QwenDualReleaseError(
            "Qwen dual screen completion identity/schema changed"
        )
    verify_descriptor_tree(payload, "Qwen dual screen completion")
    for method, evidence in markers.items():
        marker_path, _ = require_descriptor(
            evidence.get("marker") if isinstance(evidence, Mapping) else None,
            f"screen marker {method}",
        )
        marker = load_json(marker_path, f"screen marker {method}")
        if (
            marker.get("schema_version") != 1
            or marker.get("status") != "complete"
            or marker.get("stage") != "screen"
            or marker.get("method") != method
            or marker.get("samples") != SCREEN_SAMPLES
            or marker.get("campaign_fingerprint") != fingerprint
        ):
            raise QwenDualReleaseError(f"screen marker identity changed for {method}")


def validate_completion(
    path: Path,
    *,
    family: str,
    judge_model: str,
    judge_revision: str,
    judge_fingerprint: str,
) -> dict[str, Any]:
    """Validate one family view of the shared dual completion.

    The returned descriptors are safe to place directly in the immutable
    multi-family release manifest.  The other family's full gate is also
    checked because the producer's completion is atomic across both families.
    """

    if family not in FAMILY_SPECS:
        raise QwenDualReleaseError(f"unsupported Qwen dual family: {family!r}")
    completed_path = require_file(path, f"{family} Qwen dual completion")
    completed = load_json(completed_path, f"{family} Qwen dual completion")
    fingerprint = str(completed.get("campaign_fingerprint", ""))
    if (
        set(completed) != COMPLETION_FIELDS
        or completed.get("schema_version") != 1
        or completed.get("status") != "complete"
        or completed.get("stage") != "full"
        or completed.get("expected_samples") != EXPECTED_SAMPLES
        or SHA256_RE.fullmatch(fingerprint) is None
        or verify_descriptor_tree(completed, "Qwen dual completion") == 0
    ):
        raise QwenDualReleaseError("Qwen dual completion identity/schema changed")
    contract_path, contract_descriptor = require_descriptor(
        completed.get("contract"), "Qwen dual campaign contract"
    )
    preliminary = load_json(contract_path, "Qwen dual campaign contract")
    output_root = Path(str(preliminary.get("output_root", ""))).expanduser().resolve()
    if completed_path != output_root / "full_completed.json":
        raise QwenDualReleaseError("Qwen dual completion is outside its output root")
    contract = validate_contract(
        contract_path,
        fingerprint=fingerprint,
        output_root=output_root,
        judge_model=judge_model,
        judge_revision=judge_revision,
        judge_fingerprint=judge_fingerprint,
    )

    marker_identities, marker_summaries = validate_marker_graph(
        completed,
        output_root=output_root,
        fingerprint=fingerprint,
    )
    archives = contract["v2_provenance"]["archived_controls"]
    common_mos_keys: set[tuple[str, str]] | None = None
    family_sample_evidence: dict[str, dict[str, SampleEvidence]] = {}
    for current_family, spec in FAMILY_SPECS.items():
        archive_path, _ = require_descriptor(
            archives[current_family]["sample"], f"{current_family} archived sample"
        )
        archive_identity = load_sample_identity(
            archive_path, f"{current_family}.archived_sample"
        )
        by_role = {
            "archive": archive_identity,
            "base": marker_identities[str(spec["base"])],
            "exact": marker_identities[str(spec["exact"])],
            "candidate": marker_identities[str(spec["candidate"])],
        }
        if any(
            item.identities != archive_identity.identities for item in by_role.values()
        ):
            raise QwenDualReleaseError(
                f"{current_family} archive/fresh/all/candidate typed identities differ"
            )
        family_sample_evidence[current_family] = by_role
        if common_mos_keys is None:
            common_mos_keys = set(archive_identity.mos_keys)
        elif common_mos_keys != set(archive_identity.mos_keys):
            raise QwenDualReleaseError("Qwen dual family identity sets differ")
    assert common_mos_keys is not None

    raw_selections = completed.get("selections")
    if not isinstance(raw_selections, Mapping) or set(raw_selections) != set(
        FAMILY_SPECS
    ):
        raise QwenDualReleaseError("Qwen dual completion selection set changed")
    selections: dict[str, dict[str, Any]] = {}
    selection_descriptors: dict[str, dict[str, str]] = {}
    selected_methods: dict[str, str] = {}
    method_sets: dict[str, frozenset[str]] = {}
    for current_family in FAMILY_SPECS:
        selection_path, selection_descriptor = require_descriptor(
            raw_selections[current_family], f"{current_family} dual selection"
        )
        if selection_path != output_root / f"full_{current_family}_selection.json":
            raise QwenDualReleaseError(f"{current_family} dual selection path changed")
        selection = load_json(selection_path, f"{current_family} dual selection")
        selected, methods = validate_selection(selection, current_family)
        selections[current_family] = selection
        selection_descriptors[current_family] = selection_descriptor
        selected_methods[current_family] = selected
        method_sets[current_family] = methods

    for current_family, spec in FAMILY_SPECS.items():
        selection = selections[current_family]
        for method in (spec["base"], spec["exact"], spec["candidate"]):
            row = marker_summaries[str(method)][1]
            if not math.isclose(
                finite(row.get("token_f1"), f"{current_family}.{method}.token_f1"),
                selection_secondary(selection, str(method)),
                rel_tol=0,
                abs_tol=1e-12,
            ):
                raise QwenDualReleaseError(
                    f"{current_family} summary Token-F1 differs for {method}"
                )
        archived_summary_path, _ = require_descriptor(
            archives[current_family]["summary"], f"{current_family} archived summary"
        )
        with archived_summary_path.open("r", encoding="utf-8", newline="") as handle:
            archived_rows = [
                row
                for row in csv.DictReader(handle)
                if row.get("run_name") == archives[current_family]["summary_run_name"]
                and row.get("task") == "densevideo"
                and row.get("method") == spec["archive"]
            ]
        if (
            len(archived_rows) != 1
            or archived_rows[0].get("run_status") != "success"
            or strict_integer(
                archived_rows[0].get("samples"), f"{current_family}.archive.samples"
            )
            != EXPECTED_SAMPLES
            or not math.isclose(
                finite(
                    archived_rows[0].get("token_f1"),
                    f"{current_family}.archive.token_f1",
                ),
                selection_secondary(selection, str(spec["archive"])),
                rel_tol=0,
                abs_tol=1e-12,
            )
        ):
            raise QwenDualReleaseError(
                f"{current_family} archived summary differs from selection"
            )

    matrix_path, matrix_descriptor = require_descriptor(
        completed.get("mos_matrix_csv"), "Qwen dual MOS CSV"
    )
    jsonl_path, jsonl_descriptor = require_descriptor(
        completed.get("mos_matrix_jsonl"), "Qwen dual MOS JSONL"
    )
    if (
        matrix_path != output_root / "full_mos" / "matrix.csv"
        or jsonl_path != output_root / "full_mos" / "matrix.jsonl"
    ):
        raise QwenDualReleaseError("Qwen dual MOS paths changed")
    matrix_means = inspect_matrix(
        matrix_path,
        jsonl_path,
        selections=selections,
        expected_keys=common_mos_keys,
        judge_model=judge_model,
        judge_revision=judge_revision,
        judge_fingerprint=judge_fingerprint,
    )

    mos_manifest_path, _ = require_descriptor(
        completed.get("mos_manifest"), "Qwen dual MOS manifest"
    )
    if mos_manifest_path != output_root / "full_summaries" / "full_sample_sources.json":
        raise QwenDualReleaseError("Qwen dual MOS manifest path changed")
    markers = completed["markers"]
    validate_mos_manifest(
        mos_manifest_path,
        output_root=output_root,
        fingerprint=fingerprint,
        contract_descriptor=contract_descriptor,
        markers=markers,
        contract=contract,
        selections=selections,
    )
    validation_path, _ = require_descriptor(
        completed.get("validation"), "Qwen dual full validation"
    )
    if validation_path != output_root / "full_validation.json":
        raise QwenDualReleaseError("Qwen dual validation path changed")
    validate_full_validation(
        validation_path,
        fingerprint=fingerprint,
        contract_descriptor=contract_descriptor,
        markers=markers,
        contract=contract,
        selections=selections,
        sample_evidence=family_sample_evidence,
    )
    screen_path, _ = require_descriptor(
        completed.get("screen_completion"), "Qwen dual screen completion"
    )
    if screen_path != output_root / "screen_completed.json":
        raise QwenDualReleaseError("Qwen dual screen completion path changed")
    validate_screen_completion(
        screen_path,
        fingerprint=fingerprint,
        contract_descriptor=contract_descriptor,
    )
    complement_path, _ = require_descriptor(
        completed.get("complement_report"), "Qwen dual complement report"
    )
    if complement_path != output_root / "full_complement_report.json":
        raise QwenDualReleaseError("Qwen dual complement path changed")
    validate_complement(
        complement_path,
        fingerprint=fingerprint,
        contract_descriptor=contract_descriptor,
        matrix_descriptor=matrix_descriptor,
        selections=selections,
        matrix_means=matrix_means,
        sample_evidence=family_sample_evidence,
    )

    selected = selected_methods[family]
    summary_path = marker_summaries[selected][0]
    return {
        "schema": COMPLETION_SCHEMA,
        "family": family,
        "campaign_fingerprint": fingerprint,
        "model_revision": str(contract["families"][family]["revision"]),
        "dataset_revision": str(contract["dataset_revision"]),
        "selected_method": selected,
        "methods": method_sets[family],
        "selection_payload": selections[family],
        "selection": selection_descriptors[family],
        "completion": descriptor(completed_path),
        "open_mos_matrix": matrix_descriptor,
        "open_mos_jsonl": jsonl_descriptor,
        "summary_csv": descriptor(summary_path),
        "contract": contract_descriptor,
    }


def is_dual_completion(payload: Mapping[str, Any]) -> bool:
    """Return true only for the exact shared full-completion envelope."""

    return (
        set(payload) == COMPLETION_FIELDS
        and payload.get("stage") == "full"
        and payload.get("expected_samples") == EXPECTED_SAMPLES
    )
