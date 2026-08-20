#!/usr/bin/env python3
"""Publish verified GRT LPM results into the static DIVE leaderboard.

The site intentionally retains its legacy High-Motion snapshot.  This tool
supports both the original one-selection release and a four-family manifest.
It validates all inputs before atomically writing output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import tempfile
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from .qwen_dual_release import (
        COMPLETION_SCHEMA as QWEN_DUAL_COMPLETION_SCHEMA,
        QwenDualReleaseError,
        mos_methods as qwen_dual_mos_methods,
        validate_completion as validate_qwen_dual_completion,
    )
except ImportError:  # Executed directly from scripts/.
    script_directory = str(Path(__file__).resolve().parent)
    if script_directory not in sys.path:
        sys.path.insert(0, script_directory)
    from qwen_dual_release import (  # type: ignore[no-redef]
        COMPLETION_SCHEMA as QWEN_DUAL_COMPLETION_SCHEMA,
        QwenDualReleaseError,
        mos_methods as qwen_dual_mos_methods,
        validate_completion as validate_qwen_dual_completion,
    )


ASSIGNMENT = "window.DIVE_LEADERBOARD ="
LPM_TASK = "densevideo"
DEFAULT_BASE_METHOD = "llava_onevision_0_5b"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
IMMUTABLE_REVISION_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
LEGACY_SELECTION_SCHEMA = "legacy_run_fingerprint_v1"
ROUTE31_SELECTION_SCHEMA = "hf_route31_full_campaign_v1"
ROUTE31_PUBLIC_METHOD = "llava_onevision_0_5b"
ROUTE31_BASE_METHOD = "llava_onevision_0_5b_hf_route31_base"
ROUTE31_EXACT_METHOD = "llava_onevision_0_5b_hf_route31_exact"
ROUTE31_WINNER_METHOD = "grt_llava_onevision_0_5b_hf_route31_t0001"
ROUTE31_METHODS = (
    ROUTE31_PUBLIC_METHOD,
    ROUTE31_BASE_METHOD,
    ROUTE31_EXACT_METHOD,
    ROUTE31_WINNER_METHOD,
)
ROUTE31_ARTIFACT_NAMES = {
    "actual_vs_virtual_diagnostic",
    "archived_public_floor",
    "archived_public_result",
    "archived_public_sample",
    "base_exact_control",
    "candidate_route_invariants",
    "diagnostic_mos_matrix",
    "diagnostic_sample",
    "driver_snapshot",
    "evaluation_contract",
    "mos_integrity",
    "offline_evidence",
    "open_mos_matrix",
    "resolved_run_checksums",
    "resolved_runs",
    "runtime",
    "sample_source_checksums",
    "sample_sources",
    "source_snapshot",
    "summary_archived_public",
    "summary_route31_base",
    "summary_route31_candidate",
    "summary_route31_exact",
    "telemetry_contract",
}
MULTIFAMILY_RELEASE_SCHEMA = "dive_grt_multifamily_release_v1"
MULTIFAMILY_RELEASE_V2_SCHEMA = "dive_grt_multifamily_release_v2"
MULTIFAMILY_RELEASE_V2_FAMILIES = ("route31", "llava7", "qwen3", "qwen7")
MULTIFAMILY_RELEASE_V2_PROMOTED = "promoted"
MULTIFAMILY_RELEASE_V2_NOT_PROMOTED = "not_promoted"
MULTIFAMILY_RELEASE_V2_PASSED_REASON = "strict_quality_gate_passed"
MULTIFAMILY_RELEASE_V2_FAILED_REASON = "strict_quality_gate_failed"
MULTIFAMILY_RELEASE_V2_PASSED_EVIDENCE = "passed_full_selection"
MULTIFAMILY_RELEASE_V2_FAILED_EVIDENCE = "failed_full_selection"
STRICT_FULL_GATE_SCHEMA = "strict_full_gate_v1"
MULTIFAMILY_EXPECTED_SAMPLES = 634
MULTIFAMILY_EXPECTED_LPM_ROWS = 30
MULTIFAMILY_BASELINE_LPM_ROWS = 27
MULTIFAMILY_EXPECTED_HIGHMOTION_ROWS = 3
MULTIFAMILY_EXPECTED_HIGHMOTION_SAMPLES = 1000
MULTIFAMILY_JUDGE = {
    "model": "Qwen/Qwen3-VL-32B-Instruct",
    "revision": "0cfaf48183f594c314753d30a4c4974bc75f3ccb",
    "fingerprint": "64c644b2bd2696677afd3507a6bd6efa03e902d712c0e55e26ab9ab144c0a76d",
}
LLAVA7_FAILED_CAMPAIGN_FINGERPRINT = (
    "c50676095ba30a724e9722c2553acbb6a58b32b534652a0c90610919791df63e"
)
LLAVA7_FAILED_CONTRACT_SHA256 = (
    "7db2fa85aefc5c255f52e7ea7c42b9717e648c9d0e2a4f0098f98b21ed4621b4"
)
LLAVA7_FAILED_SOURCE_SNAPSHOT_SHA256 = (
    "53df24d6980722e0ac230ae5d6d1a5002e0e52258d62775ab29aa6201063d6b0"
)
LLAVA7_FAILED_VALIDATION_SHA256 = (
    "1be08185463674e44933eeb18c45a471f1b53555f5a99fe608aec832a356ac2e"
)
LLAVA7_FAILED_SELECTION_SHA256 = (
    "692e97e50eeca81686798d1d2a9917b5651df5b7e511c6fb67ddcca99d747500"
)
LLAVA7_FAILED_MATRIX_CSV_SHA256 = (
    "76ac7d6c365130713a6c5570ac1db042a33138265cf97ef4fa8d2ea2a8200840"
)
LLAVA7_FAILED_MATRIX_JSONL_SHA256 = (
    "3ef325b812ad0184482112e0713341de255b3e1760153d01dab4789cb4027710"
)
LEGACY_GRT_METHOD = "grt_llava_ov_0_5b"
LEGACY_OPEN_MOS_CAMPAIGN = "legacy_open_mos_rejudge_v1"
LEGACY_OPEN_MOS_METHODS = (
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "qwen2_5_vl_32b",
    "qwen2_5_vl_72b",
    "qwen2_vl_2b",
    "qwen3_vl_2b",
    "qwen3_vl_32b",
    "qwen3_vl_4b",
    "qwen3_vl_8b",
)
LEGACY_OPEN_MOS_INVENTORY_METHODS = (
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "grt_llava_ov_0_5b",
    "llava_onevision_0_5b",
    "llava_onevision_original",
    "qwen2_5_vl_32b",
    "qwen2_5_vl_3b",
    "qwen2_5_vl_72b",
    "qwen2_5_vl_7b",
    "qwen2_vl_2b",
    "qwen3_vl_2b",
    "qwen3_vl_32b",
    "qwen3_vl_4b",
    "qwen3_vl_8b",
)
LEGACY_OPEN_MOS_OMITTED_METHODS = (
    "grt_llava_ov_0_5b",
    "llava_onevision_0_5b",
    "llava_onevision_original",
    "qwen2_5_vl_3b",
    "qwen2_5_vl_7b",
)
LEGACY_OPEN_MOS_EXPECTED_ROWS = (
    len(LEGACY_OPEN_MOS_METHODS) * MULTIFAMILY_EXPECTED_SAMPLES
)
QWEN3_RECOVERY_COMPLETION_SCHEMA = "qwen3_full_recovery_completed_v2"
QWEN7_FLOOR_FULL_COMPLETION_SCHEMA = "qwen7_cap48_next_full_completed_v1"
QWEN7_FLOOR_FULL_CAMPAIGN_FINGERPRINT = (
    "5c391e7b789070c0000ecca43ab4fd626fb4984d28d2a35753009bf1aef0f04f"
)
QWEN7_FLOOR_FULL_CONTRACT_SHA256 = (
    "da98dda1e211a00d773243348cd9c200aac11dbe9964df61b0eedde959f497b7"
)
QWEN7_FLOOR_FULL_ARCHIVE = "qwen2_5_vl_7b"
QWEN7_FLOOR_FULL_METHODS = (
    "qwen2_5_vl_7b_floor_quality_base",
    "qwen2_5_vl_7b_floor_grt_all",
    "grt_qwen2_5_vl_7b_dual_floor_s080_o055_cap48",
)
QWEN7_FLOOR_FULL_MOS_METHODS = (
    QWEN7_FLOOR_FULL_ARCHIVE,
    *QWEN7_FLOOR_FULL_METHODS,
)
QWEN7_FLOOR_FULL_MOS_FIELDS = (
    "sample_id",
    "doc_id",
    "run_name",
    "method",
    "model",
    "video_name",
    "question_id",
    "type",
    "open_mos_correctness",
    "open_mos_score",
    "mos_judge_model",
    "mos_judge_revision",
    "request_fingerprint",
    "judge_fingerprint",
    "error",
    "question",
    "answer",
    "pred",
    "open_mos_review",
)
QWEN7_FLOOR_FULL_MOS_TEXT_FIELDS = tuple(
    field
    for field in QWEN7_FLOOR_FULL_MOS_FIELDS
    if field not in {"doc_id", "open_mos_score"}
)
QWEN7_FLOOR_FULL_SUMMARY_REQUIRED_FIELDS = frozenset(
    {"run_name", "method", "task", "run_status", "samples"}
)
QWEN3_RECOVERY_PROTOCOL = "dive-grt-followup-v2-qwen3-full-recovery-v2"
QWEN3_RECOVERY_ORIGINAL_DRIVER_SHA256 = (
    "9e18c9a68e7fe931176aa830dfb9ccc7268cc3d813ccf08580e668645dde6fd1"
)
QWEN3_RECOVERY_FIXED_SHA256 = {
    "run_contract": "965b83370039f140f7aa04111a95560e1d8328cdcd4b0353faeff24b0057b737",
    "driver_manifest": "9035b9e0513327552fc3095481e84865355d733d6286937bf21b3d3f5d161d3e",
    "input_manifest": "df7aebc5b7d2cfa807534e27601ef0c293ceadf4254a7cdf1d7b6be62cf965bd",
    "broken_screen_report": "ac0ea1668239bc9ec8b7810327f606e9c23e37a60919335514510998634fb838",
    "campaign_contract": "30e721955ce20191ea8601f41a9b7962324f66a6e732b9deba46ca96193a5a9c",
    "campaign_source_snapshot": "44382135a968142bddcbe71ab2b3110ec9a35dead671b43dcda1536b31b1dedd",
    "screen_selection": "eb4090b5bc646f39490b7a03d769f141df837b234ba14e669f148e5ca7f66199",
    "screen_mos_matrix": "33e974f5279a1f0b3ae4f6871a56e3229d98cb36b866c36caa753165d43fe584",
    "config": "86b51b36c46b0186825f8a475777690dc1992062eefffce540a9492f3c4c08bd",
    "original_driver": QWEN3_RECOVERY_ORIGINAL_DRIVER_SHA256,
}
QWEN3_RECOVERY_OLD_MARKER_SHA256 = {
    "qwen2_5_vl_3b_quality_base": "ea70d6c8dcaaefb7f70ea676707a8c7a87238b2d4d4ef8a198ea5e678cc6c7da",
    "qwen2_5_vl_3b_grt_all": "92eeb0da09d9ef0b17da017a19329b304def294564ffe4a879d7c723209750ef",
    "grt_qwen2_5_vl_3b_t03": "32f18cad315eaaae1b96bf648dddcf8793537ab11d455b2036f880c34e7dba92",
    "grt_qwen2_5_vl_3b_t01": "d5c8c38d3a0301ee856dad5b71f8e3803da0d0e592b9fd321f331607f999281b",
}
QWEN3_RECOVERY_PUBLIC = "qwen2_5_vl_3b"
QWEN3_RECOVERY_REUSED = (
    "qwen2_5_vl_3b_quality_base",
    "qwen2_5_vl_3b_grt_all",
    "grt_qwen2_5_vl_3b_t03",
    "grt_qwen2_5_vl_3b_t01",
)
QWEN3_RECOVERY_NATIVE = (
    "grt_qwen2_5_vl_3b_t300",
    "grt_qwen2_5_vl_3b_t100",
)
QWEN3_RECOVERY_METHODS = QWEN3_RECOVERY_REUSED + QWEN3_RECOVERY_NATIVE
QWEN3_RECOVERY_FINALISTS = (
    "grt_qwen2_5_vl_3b_t03",
    "grt_qwen2_5_vl_3b_t01",
    "grt_qwen2_5_vl_3b_t300",
    "grt_qwen2_5_vl_3b_t100",
)
MULTIFAMILY_CONTRACTS: Mapping[str, Mapping[str, Any]] = {
    "route31": {
        "base_method": ROUTE31_PUBLIC_METHOD,
        "replaces_method": "grt_llava_ov_0_5b",
        "selection_schema": ROUTE31_SELECTION_SCHEMA,
        "completion_schema": None,
    },
    "llava7": {
        "base_method": "llava_onevision_original",
        "replaces_method": None,
        "selection_schema": STRICT_FULL_GATE_SCHEMA,
        "completion_schema": "llava7_dual_completed_v1",
    },
    "qwen3": {
        "base_method": "qwen2_5_vl_3b",
        "replaces_method": None,
        "selection_schema": STRICT_FULL_GATE_SCHEMA,
        "completion_schema": frozenset(
            {
                "followup_parallel_completed_v1",
                QWEN_DUAL_COMPLETION_SCHEMA,
                QWEN3_RECOVERY_COMPLETION_SCHEMA,
            }
        ),
    },
    "qwen7": {
        "base_method": "qwen2_5_vl_7b",
        "replaces_method": None,
        "selection_schema": STRICT_FULL_GATE_SCHEMA,
        "completion_schema": frozenset(
            {
                "followup_parallel_completed_v1",
                QWEN_DUAL_COMPLETION_SCHEMA,
                QWEN7_FLOOR_FULL_COMPLETION_SCHEMA,
            }
        ),
    },
}
MULTIFAMILY_SELECTION_CONTRACTS: Mapping[str, Mapping[str, Any]] = {
    "route31": {
        "baselines": {
            ROUTE31_PUBLIC_METHOD,
            ROUTE31_BASE_METHOD,
            ROUTE31_EXACT_METHOD,
        },
        "candidates": {ROUTE31_WINNER_METHOD},
    },
    "llava7": {
        "baselines": {
            "llava_onevision_original",
            "llava_onevision_7b_dual_quality_base",
            "llava_onevision_7b_dual_grt_all",
        },
        "candidates": {"grt_llava_onevision_7b_hf_dual_s002_o005"},
    },
    "qwen3": {
        "baselines": {
            "qwen2_5_vl_3b",
            "qwen2_5_vl_3b_quality_base",
            "qwen2_5_vl_3b_grt_all",
        },
        "candidates": None,
        "allowed_candidates": {
            "grt_qwen2_5_vl_3b_t01",
            "grt_qwen2_5_vl_3b_t03",
            "grt_qwen2_5_vl_3b_t10",
            "grt_qwen2_5_vl_3b_t30",
            "grt_qwen2_5_vl_3b_t100",
            "grt_qwen2_5_vl_3b_t300",
        },
    },
    "qwen7": {
        "baselines": {
            "qwen2_5_vl_7b",
            "qwen2_5_vl_7b_quality_base",
            "qwen2_5_vl_7b_grt_all",
        },
        "candidates": None,
        "allowed_candidates": {
            "grt_qwen2_5_vl_7b_t01",
            "grt_qwen2_5_vl_7b_t03",
            "grt_qwen2_5_vl_7b_t10",
            "grt_qwen2_5_vl_7b_t30",
            "grt_qwen2_5_vl_7b_t100",
            "grt_qwen2_5_vl_7b_t300",
        },
    },
}
MULTIFAMILY_DUAL_SELECTION_CONTRACTS: Mapping[str, Mapping[str, Any]] = {
    "qwen3": {
        "baselines": {
            "qwen2_5_vl_3b",
            "qwen2_5_vl_3b_dual_quality_base",
            "qwen2_5_vl_3b_dual_grt_all",
        },
        "candidates": {"grt_qwen2_5_vl_3b_dual_s100_o03"},
    },
    "qwen7": {
        "baselines": {
            "qwen2_5_vl_7b",
            "qwen2_5_vl_7b_dual_quality_base",
            "qwen2_5_vl_7b_dual_grt_all",
        },
        "candidates": {"grt_qwen2_5_vl_7b_dual_s30_o300"},
    },
}
MULTIFAMILY_QWEN7_FLOOR_SELECTION_CONTRACT: Mapping[str, Any] = {
    "baselines": {
        "qwen2_5_vl_7b",
        "qwen2_5_vl_7b_floor_quality_base",
        "qwen2_5_vl_7b_floor_grt_all",
    },
    "candidates": {"grt_qwen2_5_vl_7b_dual_floor_s080_o055_cap48"},
}

LPM_NUMERIC_FIELDS: Mapping[str, tuple[str, ...]] = {
    "open_mos": ("open_mos",),
    "token_f1": ("token_f1",),
    "cer": ("cer",),
    "wer": ("wer",),
    "exact_match": ("exact_match",),
    "recompute_ratio": (
        "mean_recompute_ratio",
        "mean_patch_projection_recompute_ratio",
        "recompute_ratio",
    ),
    "reference_recompute_ratio": (
        "reference_patch_compute_ratio",
        "mean_patch_projection_compute_ratio_vs_reference",
        "selected_reference_patch_compute_ratio",
        "reference_recompute_ratio",
    ),
    "effective_fps": ("mean_effective_fps", "effective_fps"),
    "throughput_fps": ("mean_throughput_fps", "throughput_fps"),
}


class SyncError(ValueError):
    """Raised when a release artifact violates the publication contract."""


def _strict_json_loads(source: str, label: str) -> Any:
    """Parse JSON with Decimal floats and duplicate-key rejection at every depth."""

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SyncError(f"{label} has duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            source,
            parse_float=Decimal,
            object_pairs_hook=unique_object,
        )
    except json.JSONDecodeError as exc:
        raise SyncError(f"{label} is invalid JSON: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise SyncError(f"could not hash release artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _artifact_path(raw_path: Any, base_directory: Path, label: str) -> Path:
    text = str(raw_path or "").strip()
    if not text:
        raise SyncError(f"{label} has no path")
    path = Path(text)
    if not path.is_absolute():
        path = base_directory / path
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise SyncError(f"{label} path is not a readable artifact: {path}") from exc


def validate_artifact_descriptor(
    descriptor: Any,
    *,
    base_directory: Path,
    label: str,
    expected_path: Path | None = None,
) -> tuple[Path, str]:
    if not isinstance(descriptor, dict):
        raise SyncError(f"{label} must be a path/SHA object")
    path = _artifact_path(descriptor.get("path"), base_directory, label)
    expected_sha = str(descriptor.get("sha256") or "").strip()
    if SHA256_PATTERN.fullmatch(expected_sha) is None:
        raise SyncError(f"{label} has no valid lowercase SHA-256 digest")
    if expected_path is not None and path != expected_path.resolve(strict=True):
        raise SyncError(f"{label} path does not match the supplied artifact")
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise SyncError(f"{label} SHA-256 does not match the artifact bytes")
    return path, actual_sha


def _quote_javascript_keys(source: str) -> str:
    """Turn the restricted generated JS object literal into JSON.

    Existing site snapshots use unquoted identifier keys but otherwise contain
    JSON values.  A tiny lexer avoids changing identifier-like text in strings.
    Newly generated snapshots use quoted keys and also pass through unchanged.
    """

    output: list[str] = []
    index = 0
    in_string = False
    quote = ""
    escaped = False
    while index < len(source):
        character = source[index]
        if in_string:
            output.append(character)
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                in_string = False
            index += 1
            continue
        if character in {'"', "'"}:
            in_string = True
            quote = character
            output.append(character)
            index += 1
            continue
        if character.isalpha() or character in "_$":
            end = index + 1
            while end < len(source) and (source[end].isalnum() or source[end] in "_$"):
                end += 1
            previous = next(
                (item for item in reversed(output) if not item.isspace()), ""
            )
            lookahead = end
            while lookahead < len(source) and source[lookahead].isspace():
                lookahead += 1
            identifier = source[index:end]
            if (
                previous in "{,"
                and lookahead < len(source)
                and source[lookahead] == ":"
            ):
                output.append(json.dumps(identifier))
            else:
                output.append(identifier)
            index = end
            continue
        output.append(character)
        index += 1
    return "".join(output)


def load_site_payload(path: Path) -> dict[str, Any]:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SyncError(f"could not read site data {path}: {exc}") from exc
    marker = source.find(ASSIGNMENT)
    if marker < 0:
        raise SyncError(f"site data does not contain {ASSIGNMENT!r}: {path}")
    object_start = source.find("{", marker + len(ASSIGNMENT))
    object_end = source.rfind("}")
    if object_start < 0 or object_end < object_start:
        raise SyncError(f"site data has no complete leaderboard object: {path}")
    literal = _quote_javascript_keys(source[object_start : object_end + 1])
    payload = _strict_json_loads(literal, "site leaderboard object")
    if not isinstance(payload, dict):
        raise SyncError("site leaderboard payload must be an object")
    return payload


def _json_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise SyncError(f"cannot serialize non-finite decimal {value}")
        rendered = format(value, "f")
        if "." in rendered:
            rendered = rendered.rstrip("0").rstrip(".")
        return rendered or "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SyncError(f"cannot serialize non-finite float {value}")
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    raise SyncError(f"unsupported leaderboard value type: {type(value).__name__}")


def _render_json(value: Any, level: int = 0) -> str:
    indent = "  " * level
    child_indent = "  " * (level + 1)
    if isinstance(value, dict):
        if not value:
            return "{}"
        scalar_only = all(not isinstance(item, (dict, list)) for item in value.values())
        if scalar_only:
            return (
                "{ "
                + ", ".join(
                    f"{json.dumps(str(key), ensure_ascii=False)}: {_json_scalar(item)}"
                    for key, item in value.items()
                )
                + " }"
            )
        parts = [
            f"{child_indent}{json.dumps(str(key), ensure_ascii=False)}: {_render_json(item, level + 1)}"
            for key, item in value.items()
        ]
        return "{\n" + ",\n".join(parts) + f"\n{indent}}}"
    if isinstance(value, list):
        if not value:
            return "[]"
        parts = [f"{child_indent}{_render_json(item, level + 1)}" for item in value]
        return "[\n" + ",\n".join(parts) + f"\n{indent}]"
    return _json_scalar(value)


def render_site_payload(payload: Mapping[str, Any]) -> str:
    generated_at = payload.get("generatedAt", "unknown")
    source_artifacts = payload.get("sourceArtifacts", "unknown")
    return (
        "/* Generated by scripts/sync_leaderboard.py from verified release artifacts.\n"
        f" * Snapshot: {generated_at}; source artifacts: {source_artifacts}.\n"
        " * Keep numeric values unrounded here; formatting belongs in app.js.\n"
        " */\n"
        f"{ASSIGNMENT} {_render_json(dict(payload))};\n"
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = _strict_json_loads(
            path.read_text(encoding="utf-8"), f"JSON artifact {path}"
        )
    except OSError as exc:
        raise SyncError(f"could not read JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SyncError(f"JSON artifact must contain an object: {path}")
    return payload


def _read_csv(path: Path) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fields = tuple(reader.fieldnames or ())
    except (OSError, csv.Error) as exc:
        raise SyncError(f"could not read leaderboard CSV {path}: {exc}") from exc
    if not fields or not rows:
        raise SyncError(f"leaderboard CSV is empty or has no header: {path}")
    return rows, fields


def _decimal(value: Any, label: str, *, required: bool = True) -> Decimal | None:
    if value is None or str(value).strip() == "":
        if required:
            raise SyncError(f"missing numeric value for {label}")
        return None
    try:
        number = value if isinstance(value, Decimal) else Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as exc:
        raise SyncError(f"invalid numeric value for {label}: {value!r}") from exc
    if not number.is_finite():
        raise SyncError(f"non-finite numeric value for {label}: {value!r}")
    return number


def _integer(value: Any, label: str) -> int:
    number = _decimal(value, label)
    assert number is not None
    integral = number.to_integral_value()
    if number != integral:
        raise SyncError(f"{label} must be an integer, got {value!r}")
    return int(integral)


def _densevideo_token_f1(prediction: Any, reference: Any) -> float:
    def tokens(value: Any) -> list[str]:
        return " ".join(str(value or "").strip().lower().split()).split()

    pred_tokens = tokens(prediction)
    ref_tokens = tokens(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts: dict[str, int] = {}
    ref_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    overlap = sum(
        min(count, ref_counts.get(token, 0))
        for token, count in pred_counts.items()
    )
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _open_mos_request_fingerprint(
    question: str, answer: str, pred: str, *, trim_limit: int = 6000
) -> str:
    def trimmed(value: str) -> str:
        return value[:trim_limit] + "..." if len(value) > trim_limit else value

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert evaluator for DIVE-Bench dense video "
                "understanding answers. Compare a model prediction with the "
                "reference answer and assign a MOS-style semantic match score.\n"
                "Use score 0 for no useful match and score 5 for a near-complete "
                "meaningful match. Consider paraphrases valid, but penalize "
                "omissions, hallucinations, wrong temporal order, and generic "
                "refusals."
            ),
        },
        {
            "role": "user",
            "content": (
                "Evaluate this video question-answer pair.\n\n"
                f"Question: {question}\n"
                f"Reference answer: {trimmed(answer)}\n"
                f"Predicted answer: {trimmed(pred)}\n\n"
                "Return only strict JSON with keys: "
                '{"pred": "yes" or "no", "score": integer 0-5}. '
                "Do not include explanation or markdown."
            ),
        },
    ]
    payload = {
        "messages": messages,
        "answer": answer,
        "pred": pred,
        "prompt_version": "densevideo-open-mos-v1",
        "response_schema_version": "strict-json-pred-score-v1",
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _first(row: Mapping[str, Any], aliases: Iterable[str]) -> Any:
    for alias in aliases:
        value = row.get(alias)
        if value is not None and str(value).strip() != "":
            return value
    return None


def _candidate_row(rows: Sequence[Mapping[str, str]], method: str) -> Mapping[str, str]:
    matches = [
        row
        for row in rows
        if str(row.get("method") or row.get("run_name") or "").strip() == method
        and str(row.get("task", LPM_TASK)).strip() == LPM_TASK
    ]
    if len(matches) != 1:
        raise SyncError(
            f"expected exactly one {LPM_TASK} row for selected method {method!r}; found {len(matches)}"
        )
    return matches[0]


def _site_row(
    source: Mapping[str, str],
    method: str,
    expected_samples: int,
    *,
    strict_telemetry: bool = True,
) -> dict[str, Any]:
    samples = _integer(source.get("samples"), f"{method}.samples")
    if samples != expected_samples:
        raise SyncError(
            f"selected method has {samples} samples; expected {expected_samples}"
        )
    model = str(source.get("display_name") or source.get("model") or "").strip()
    if not model:
        raise SyncError("selected leaderboard row has no display_name/model")
    row: dict[str, Any] = {
        "rank": 0,
        "model": model,
        "method": method,
        "samples": samples,
    }
    for output_field, aliases in LPM_NUMERIC_FIELDS.items():
        # Only promoted winners have a complete metric/telemetry contract.
        # Ordinary release rows may legitimately omit MOS, error metrics, or
        # profiling columns; Token-F1 remains the universal ranking fallback.
        required = strict_telemetry or output_field == "token_f1"
        row[output_field] = _decimal(
            _first(source, aliases),
            f"{method}.{output_field}",
            required=required,
        )
    row["source"] = str(source.get("source") or "open").strip() or "open"

    for ratio_field in ("recompute_ratio", "reference_recompute_ratio"):
        ratio = row[ratio_field]
        if ratio is None:
            if strict_telemetry:
                raise SyncError(f"{method}.{ratio_field} is required")
            continue
        upper_bound = Decimal(1)
        if (
            ratio < 0
            or ratio > upper_bound
            or (strict_telemetry and ratio == upper_bound)
        ):
            interval = "[0, 1)" if strict_telemetry else "[0, 1]"
            raise SyncError(
                f"{method}.{ratio_field} must be in {interval}, got {ratio}"
            )
    for rate_field in ("effective_fps", "throughput_fps"):
        rate = row[rate_field]
        if rate is None:
            if strict_telemetry:
                raise SyncError(f"{method}.{rate_field} is required")
            continue
        if rate <= 0:
            raise SyncError(f"{method}.{rate_field} must be positive, got {rate}")
    return row


def _release_lpm_rows(
    leaderboard_rows: Sequence[Mapping[str, str]],
    *,
    expected_count: int,
    expected_samples: int,
    strict_methods: Iterable[str],
) -> list[dict[str, Any]]:
    """Build the entire published LPM track from the verified release CSV.

    The existing site LPM snapshot is deliberately not an input. Ordinary
    models may lack Open-MOS and GRT telemetry, while promoted winners retain
    the strict numeric/telemetry contract enforced by ``_site_row``.
    """

    sources = [
        row
        for row in leaderboard_rows
        if str(row.get("task") or "").strip() == LPM_TASK
    ]
    if len(sources) != expected_count:
        raise SyncError(
            f"release CSV contains {len(sources)} LPM rows; expected {expected_count}"
        )
    strict = {str(method).strip() for method in strict_methods if str(method).strip()}
    converted: list[dict[str, Any]] = []
    source_ranks: dict[str, int] = {}
    for source in sources:
        method = str(source.get("method") or source.get("run_name") or "").strip()
        if not method:
            raise SyncError("release CSV contains an LPM row with no method")
        if method in source_ranks:
            raise SyncError(f"release CSV contains duplicate LPM method {method!r}")
        source_ranks[method] = _integer(source.get("rank"), f"{method}.source_rank")
        converted.append(
            _site_row(
                source,
                method,
                expected_samples,
                strict_telemetry=method in strict,
            )
        )
    missing_strict = strict.difference(source_ranks)
    if missing_strict:
        raise SyncError(
            "release CSV is missing strict winner row(s): "
            + ", ".join(sorted(missing_strict))
        )

    ranked = rerank(converted, "open_mos")
    for row in ranked:
        method = str(row["method"])
        if source_ranks[method] != row["rank"]:
            raise SyncError(
                f"source rank {source_ranks[method]} disagrees with release rank "
                f"{row['rank']} for {method}"
            )
    if set(source_ranks.values()) != set(range(1, expected_count + 1)):
        raise SyncError("release CSV LPM ranks are not one contiguous rank set")
    _validate_track(
        ranked,
        track="lpm",
        expected_count=expected_count,
        expected_samples=expected_samples,
        primary="open_mos",
    )
    return ranked


def _published_matches(selection_value: Any, published_value: Any) -> bool:
    """Accept exact parity or the release builder's documented six-digit MOS rendering."""

    selected = _decimal(selection_value, "selection metric")
    published = _decimal(published_value, "published metric")
    assert selected is not None and published is not None
    return selected == published or Decimal(format(selected, ".6g")) == published


def _selection_from_metadata(metadata: Mapping[str, Any]) -> dict[str, Any] | None:
    for key in ("full_selection", "selection", "grt_selection"):
        value = metadata.get(key)
        if isinstance(value, dict):
            return dict(value)
    return None


def selection_identity(provenance: Any) -> tuple[str, str, str]:
    if not isinstance(provenance, dict):
        raise SyncError("full selection has no artifact_provenance object")
    has_run = "run_fingerprint" in provenance
    has_campaign = "campaign_fingerprint" in provenance
    if has_run == has_campaign:
        raise SyncError(
            "selection must declare exactly one of run_fingerprint or campaign_fingerprint"
        )
    if has_run:
        schema = LEGACY_SELECTION_SCHEMA
        field = "run_fingerprint"
    else:
        schema = ROUTE31_SELECTION_SCHEMA
        field = "campaign_fingerprint"
    fingerprint = str(provenance.get(field) or "").strip()
    if SHA256_PATTERN.fullmatch(fingerprint) is None:
        raise SyncError(f"selection {field} must be a lowercase SHA-256 digest")
    return schema, field, fingerprint


def _load_hashed_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = _strict_json_loads(path.read_text(encoding="utf-8"), label)
    except OSError as exc:
        raise SyncError(f"could not read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SyncError(f"{label} must contain a JSON object")
    return payload


def validate_selection_artifacts(
    selection: Mapping[str, Any],
    *,
    selection_path: Path,
) -> dict[str, tuple[Path, str]]:
    provenance = selection.get("artifact_provenance")
    if not isinstance(provenance, dict):
        raise SyncError("full selection has no artifact_provenance object")
    schema, _, fingerprint = selection_identity(provenance)
    artifacts = provenance.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise SyncError("selection artifact_provenance.artifacts must be non-empty")
    base_directory = selection_path.parent
    validated: dict[str, tuple[Path, str]] = {}
    for name, descriptor in artifacts.items():
        normalized_name = str(name).strip()
        if not normalized_name or normalized_name in validated:
            raise SyncError("selection contains an empty or duplicate artifact name")
        if schema == ROUTE31_SELECTION_SCHEMA and (
            not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}
        ):
            raise SyncError(
                f"Route31 selection artifact {normalized_name!r} must be an exact path/SHA object"
            )
        validated[normalized_name] = validate_artifact_descriptor(
            descriptor,
            base_directory=base_directory,
            label=f"selection artifact {normalized_name!r}",
        )

    method = str(selection.get("selected_method") or "").strip()
    if schema == LEGACY_SELECTION_SCHEMA:
        required_names = ("sample_manifest", "open_mos_matrix", f"summary_{method}")
        missing_names = [name for name in required_names if name not in validated]
        if missing_names:
            raise SyncError(
                "selection artifact provenance is missing required evidence: "
                + ", ".join(missing_names)
            )
        top_level_descriptors = {
            "sample_manifest": {
                "path": provenance.get("sample_manifest"),
                "sha256": provenance.get("sample_manifest_sha256"),
            },
            "open_mos_matrix": {
                "path": provenance.get("open_mos_matrix"),
                "sha256": provenance.get("open_mos_matrix_sha256"),
            },
        }
        for name, descriptor in top_level_descriptors.items():
            top_level = validate_artifact_descriptor(
                descriptor,
                base_directory=base_directory,
                label=f"selection top-level {name}",
            )
            if top_level != validated[name]:
                raise SyncError(
                    f"selection top-level {name} does not match artifacts.{name} path/SHA"
                )
        return validated

    if method != ROUTE31_WINNER_METHOD:
        raise SyncError("Route31 campaign selection has an unexpected winner method")
    if set(validated) != ROUTE31_ARTIFACT_NAMES:
        raise SyncError(
            "Route31 campaign selection has an incomplete or expanded artifact set"
        )
    attempt_id = str(provenance.get("attempt_id") or "").strip()
    if (
        set(provenance)
        != {
            "schema_version",
            "attempt_id",
            "campaign_fingerprint",
            "same_snapshot_fresh_methods",
            "archived_floor_is_fresh",
            "default_max_new_tokens",
            "subtitle_max_new_tokens",
            "samples",
            "artifacts",
        }
        or provenance.get("schema_version") != 1
        or not attempt_id
        or selection.get("attempt_id") != attempt_id
        or provenance.get("same_snapshot_fresh_methods") is not True
        or provenance.get("archived_floor_is_fresh") is not False
        or provenance.get("default_max_new_tokens") != 128
        or provenance.get("subtitle_max_new_tokens") != 31
        or provenance.get("samples") != 634
    ):
        raise SyncError("Route31 campaign selection provenance contract is invalid")
    if selection.get("sample_counts") != {method: 634 for method in ROUTE31_METHODS}:
        raise SyncError(
            "Route31 campaign selection does not bind all matched 634-sample methods"
        )

    dual_gates = selection.get("dual_gate_provenance")
    if not isinstance(dual_gates, dict) or set(dual_gates) != {
        "archived_public",
        "fresh_controls",
    }:
        raise SyncError("Route31 campaign selection has invalid dual-gate provenance")
    for name, descriptor in dual_gates.items():
        if not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}:
            raise SyncError(f"Route31 {name} gate must be an exact path/SHA object")
        gate_path, _ = validate_artifact_descriptor(
            descriptor,
            base_directory=base_directory,
            label=f"Route31 {name} gate",
        )
        gate = _load_hashed_json(gate_path, f"Route31 {name} gate")
        baselines = (
            {ROUTE31_PUBLIC_METHOD}
            if name == "archived_public"
            else {ROUTE31_BASE_METHOD, ROUTE31_EXACT_METHOD}
        )
        if (
            gate.get("status") != "passed"
            or gate.get("selected_method") != ROUTE31_WINNER_METHOD
            or gate.get("metric") != "open_mos"
            or gate.get("secondary_metric") != "token_f1"
            or gate.get("passing_candidates") != [ROUTE31_WINNER_METHOD]
            or gate.get("candidate_scores") != selection.get("candidate_scores")
            or gate.get("candidate_secondary_scores")
            != selection.get("candidate_secondary_scores")
            or gate.get("sample_counts")
            != {method: 634 for method in baselines | {ROUTE31_WINNER_METHOD}}
        ):
            raise SyncError(f"Route31 {name} gate does not match the selected campaign")

    driver_manifest = validated["driver_snapshot"][0]
    driver_lines = driver_manifest.read_text(encoding="utf-8").splitlines()
    if len(driver_lines) != 1:
        raise SyncError("Route31 driver snapshot must contain exactly one checksum row")
    match = re.fullmatch(r"([0-9a-f]{64})  (.+)", driver_lines[0])
    if match is None:
        raise SyncError("Route31 driver snapshot row is malformed")
    driver_path = _artifact_path(match.group(2), base_directory, "Route31 driver")
    driver_sha = match.group(1)
    if sha256_file(driver_path) != driver_sha:
        raise SyncError("Route31 driver no longer matches its authenticated snapshot")
    source_snapshot_sha = validated["source_snapshot"][1]
    expected_campaign = hashlib.sha256(
        f"{source_snapshot_sha}\n{driver_sha}\n".encode("utf-8")
    ).hexdigest()
    if fingerprint != expected_campaign:
        raise SyncError("Route31 campaign fingerprint is not bound to source + driver")

    evaluation = _load_hashed_json(
        validated["evaluation_contract"][0], "Route31 contract"
    )
    expected_methods = {
        ROUTE31_BASE_METHOD: {"role": "base", "threshold": None},
        ROUTE31_EXACT_METHOD: {"role": "exact", "threshold": None},
        ROUTE31_WINNER_METHOD: {"role": "candidate", "threshold": "0.001"},
    }
    if (
        evaluation.get("schema_version") != 1
        or evaluation.get("campaign_fingerprint") != fingerprint
        or evaluation.get("expected_samples") != 634
        or evaluation.get("expected_question_routes") != {"ocr": 317, "subtitle": 317}
        or evaluation.get("generation")
        != {
            "default_max_new_tokens": 128,
            "subtitle_max_new_tokens": 31,
            "temperature": 0,
        }
        or evaluation.get("video_decode_backend") != "pyav_seek"
        or evaluation.get("methods") != expected_methods
        or evaluation.get("driver") != {"path": str(driver_path), "sha256": driver_sha}
        or any(
            IMMUTABLE_REVISION_PATTERN.fullmatch(str(evaluation.get(field) or ""))
            is None
            for field in ("model_revision", "dataset_revision", "judge_revision")
        )
        or evaluation.get("evidence_sha256") != validated["offline_evidence"][1]
        or evaluation.get("archived_public_result_sha256")
        != validated["archived_public_result"][1]
        or evaluation.get("quality_contract")
        != {
            "gates": ["archived_public", "fresh_controls", "combined"],
            "metrics": ["open_mos", "token_f1"],
            "comparison": "strictly_greater",
            "max_recompute_ratio": Decimal("0.98"),
            "max_reference_patch_compute_ratio": Decimal("0.98"),
            "mos_batch_size": 8,
        }
    ):
        raise SyncError(
            "Route31 evaluation contract does not match the selected campaign"
        )
    return validated


def validate_release_binding(
    selection: Mapping[str, Any],
    *,
    selection_path: Path | None,
    leaderboard_csv: Path,
    metadata: Mapping[str, Any],
    metadata_path: Path,
) -> Path:
    release_provenance = metadata.get("artifact_provenance")
    if not isinstance(release_provenance, dict):
        raise SyncError("release metadata has no artifact_provenance object")
    schema_version = _integer(
        release_provenance.get("schema_version"),
        "release artifact_provenance.schema_version",
    )
    if schema_version < 1:
        raise SyncError("release artifact provenance schema must be versioned")
    metadata_directory = metadata_path.parent
    validate_artifact_descriptor(
        release_provenance.get("leaderboard_csv"),
        base_directory=metadata_directory,
        label="release leaderboard_csv",
        expected_path=leaderboard_csv,
    )
    verified_selection = release_provenance.get("verified_grt_selection")
    verified_path, _ = validate_artifact_descriptor(
        verified_selection,
        base_directory=metadata_directory,
        label="release verified_grt_selection",
        expected_path=selection_path,
    )
    try:
        on_disk_selection = _strict_json_loads(
            verified_path.read_text(encoding="utf-8"),
            "metadata-bound selection",
        )
    except OSError as exc:
        raise SyncError(
            f"could not read metadata-bound selection {verified_path}: {exc}"
        ) from exc
    if on_disk_selection != selection:
        raise SyncError(
            "metadata-bound selection bytes do not describe the supplied selection"
        )

    provenance = selection.get("artifact_provenance")
    schema, fingerprint_field, fingerprint = selection_identity(provenance)
    method = str(selection.get("selected_method") or "").strip()
    if not isinstance(verified_selection, dict):
        raise SyncError("release verified_grt_selection must be an artifact descriptor")
    descriptor_schema = str(verified_selection.get("selection_schema") or "").strip()
    if schema == ROUTE31_SELECTION_SCHEMA and descriptor_schema != schema:
        raise SyncError(
            "release metadata does not identify the Route31 selection schema"
        )
    if schema == ROUTE31_SELECTION_SCHEMA and set(verified_selection) != {
        "path",
        "sha256",
        "selection_schema",
        "campaign_fingerprint",
        "selected_method",
    }:
        raise SyncError("release Route31 selection descriptor has unexpected fields")
    if descriptor_schema and descriptor_schema != schema:
        raise SyncError(
            "release metadata selection_schema does not match full selection"
        )
    opposite_field = (
        "campaign_fingerprint"
        if fingerprint_field == "run_fingerprint"
        else "run_fingerprint"
    )
    if opposite_field in verified_selection:
        raise SyncError(
            "release metadata mixes incompatible selection fingerprint schemas"
        )
    if str(verified_selection.get(fingerprint_field) or "").strip() != fingerprint:
        raise SyncError(
            f"release metadata {fingerprint_field} does not match full selection"
        )
    if str(verified_selection.get("selected_method") or "").strip() != method:
        raise SyncError(
            "release metadata selected_method does not match full selection"
        )

    selection_artifacts = validate_selection_artifacts(
        selection,
        selection_path=verified_path,
    )
    source_groups = release_provenance.get("source_artifacts")
    if not isinstance(source_groups, dict):
        raise SyncError("release metadata has no source_artifacts object")
    validated_groups: dict[str, list[tuple[Path, str]]] = {}
    for group in ("summary_csv", "open_mos_artifacts", "api_summaries"):
        descriptors = source_groups.get(group)
        if not isinstance(descriptors, list):
            raise SyncError(f"release source_artifacts.{group} must be a list")
        validated_groups[group] = [
            validate_artifact_descriptor(
                descriptor,
                base_directory=metadata_directory,
                label=f"release source_artifacts.{group}[{index}]",
            )
            for index, descriptor in enumerate(descriptors)
        ]
        top_level_paths = metadata.get(group)
        if not isinstance(top_level_paths, list):
            raise SyncError(f"release metadata {group} must be a path list")
        resolved_top_level = {
            _artifact_path(path, metadata_directory, f"release {group} path")
            for path in top_level_paths
        }
        descriptor_paths = {path for path, _ in validated_groups[group]}
        if resolved_top_level != descriptor_paths:
            raise SyncError(
                f"release metadata {group} paths do not match hashed source descriptors"
            )

    matrix_evidence = selection_artifacts["open_mos_matrix"]
    if matrix_evidence not in validated_groups["open_mos_artifacts"]:
        raise SyncError("selected Open-MOS matrix is not bound into release metadata")
    summary_name = (
        "summary_route31_candidate"
        if schema == ROUTE31_SELECTION_SCHEMA
        else f"summary_{method}"
    )
    summary_evidence = selection_artifacts[summary_name]
    if summary_evidence not in validated_groups["summary_csv"]:
        raise SyncError("selected summary CSV is not bound into release metadata")

    if schema == ROUTE31_SELECTION_SCHEMA:
        telemetry = _load_hashed_json(
            selection_artifacts["telemetry_contract"][0], "Route31 telemetry"
        )
        telemetry_methods = telemetry.get("methods")
        if (
            telemetry.get("schema_version") != 1
            or telemetry.get("status") != "passed"
            or telemetry.get("limits")
            != {
                "max_recompute_ratio": Decimal("0.98"),
                "max_reference_patch_compute_ratio": Decimal("0.98"),
            }
            or not isinstance(telemetry_methods, dict)
            or set(telemetry_methods) != set(ROUTE31_METHODS[1:])
        ):
            raise SyncError("Route31 telemetry artifact does not match the campaign")
        winner_telemetry = telemetry_methods[ROUTE31_WINNER_METHOD]
        if (
            winner_telemetry.get("role") != "candidate"
            or winner_telemetry.get("passes_recompute_caps") is not True
            or winner_telemetry.get("effective_max_new_tokens_counts")
            != {"128": 317, "31": 317}
            or winner_telemetry.get("question_route_counts")
            != {"ocr": 317, "subtitle": 317}
        ):
            raise SyncError("Route31 winner telemetry route/cap contract is invalid")
        release_rows, _ = _read_csv(leaderboard_csv)
        release_winner = _candidate_row(release_rows, ROUTE31_WINNER_METHOD)
        telemetry_bindings = {
            "mean_recompute_ratio": (
                winner_telemetry.get("mean_recompute_ratio"),
                _first(release_winner, LPM_NUMERIC_FIELDS["recompute_ratio"]),
            ),
            "reference_patch_compute_ratio": (
                winner_telemetry.get("reference_patch_compute_ratio"),
                _first(release_winner, LPM_NUMERIC_FIELDS["reference_recompute_ratio"]),
            ),
            "mean_effective_fps": (
                winner_telemetry.get("mean_effective_fps"),
                _first(release_winner, LPM_NUMERIC_FIELDS["effective_fps"]),
            ),
            "mean_throughput_fps": (
                winner_telemetry.get("mean_throughput_fps"),
                _first(release_winner, LPM_NUMERIC_FIELDS["throughput_fps"]),
            ),
        }
        for label, (telemetry_value, release_value) in telemetry_bindings.items():
            if not _published_matches(telemetry_value, release_value):
                raise SyncError(
                    f"Route31 release {label} differs from authenticated telemetry"
                )

    judge = metadata.get("open_mos_judge_provenance")
    if not isinstance(judge, dict):
        raise SyncError("release metadata has no open_mos_judge_provenance")
    judge_revision = str(judge.get("revision") or "").strip()
    judge_fingerprint = str(judge.get("judge_fingerprint") or "").strip()
    judge_model = str(judge.get("model") or "").strip()
    if not judge_model:
        raise SyncError("release metadata has no Open-MOS judge model")
    if str(metadata.get("open_mos_judge") or "").strip() != judge_model:
        raise SyncError("release Open-MOS judge model disagrees with judge provenance")
    if IMMUTABLE_REVISION_PATTERN.fullmatch(judge_revision) is None:
        raise SyncError("release Open-MOS judge revision is not immutable")
    if SHA256_PATTERN.fullmatch(judge_fingerprint) is None:
        raise SyncError("release metadata has no valid judge_fingerprint")
    if schema == ROUTE31_SELECTION_SCHEMA:
        evaluation = _load_hashed_json(
            selection_artifacts["evaluation_contract"][0], "Route31 contract"
        )
        selection_judge_revision = str(evaluation.get("judge_revision") or "").strip()
    else:
        revisions = (
            provenance.get("revisions") if isinstance(provenance, dict) else None
        )
        selection_judge_revision = str(
            revisions.get("open_mos_judge") if isinstance(revisions, dict) else ""
        ).strip()
    if judge_revision != selection_judge_revision:
        raise SyncError("release judge revision does not match selection provenance")
    return verified_path


def validate_selection(
    selection: Mapping[str, Any],
    candidate: Mapping[str, Any],
    expected_samples: int,
) -> str:
    if selection.get("status") != "passed":
        raise SyncError("full selection status must be 'passed'")
    method = str(selection.get("selected_method") or "").strip()
    if not method or method != candidate.get("method"):
        raise SyncError(
            f"selection winner {method!r} does not match CSV method {candidate.get('method')!r}"
        )
    if (
        selection.get("metric") != "open_mos"
        or selection.get("secondary_metric") != "token_f1"
    ):
        raise SyncError(
            "full selection must gate on Open MOS with Token F1 as secondary metric"
        )
    sample_counts = selection.get("sample_counts")
    if (
        not isinstance(sample_counts, dict)
        or _integer(sample_counts.get(method), f"selection.sample_counts.{method}")
        != expected_samples
    ):
        raise SyncError(
            f"selection must record {expected_samples} samples for {method}"
        )
    selected_score = _decimal(
        selection.get("selected_score"), "selection.selected_score"
    )
    selected_secondary = _decimal(
        selection.get("selected_secondary_score"), "selection.selected_secondary_score"
    )
    selected_recompute = _decimal(
        selection.get("selected_recompute_ratio"), "selection.selected_recompute_ratio"
    )
    if not _published_matches(selected_score, candidate.get("open_mos")):
        raise SyncError("selection Open MOS does not match the published CSV value")
    if not _published_matches(selected_secondary, candidate.get("token_f1")):
        raise SyncError("selection Token F1 does not match the published CSV value")
    if not _published_matches(selected_recompute, candidate.get("recompute_ratio")):
        raise SyncError(
            "selection recompute ratio does not match the published CSV value"
        )
    selection_reference = selection.get(
        "selected_reference_patch_compute_ratio",
        selection.get("selected_reference_recompute_ratio"),
    )
    selected_reference = _decimal(
        selection_reference, "selection.selected_reference_patch_compute_ratio"
    )
    if not _published_matches(
        selected_reference, candidate.get("reference_recompute_ratio")
    ):
        raise SyncError(
            "selection reference compute ratio does not match the published CSV value"
        )
    passing = selection.get("passing_candidates")
    if not isinstance(passing, list) or method not in passing:
        raise SyncError("selected method is absent from selection.passing_candidates")
    required_score = _decimal(
        selection.get("required_score"), "selection.required_score"
    )
    secondary_required = _decimal(
        selection.get("secondary_required_score"), "selection.secondary_required_score"
    )
    max_recompute = _decimal(
        selection.get("max_recompute_ratio"), "selection.max_recompute_ratio"
    )
    max_reference = _decimal(
        selection.get("max_reference_recompute_ratio"),
        "selection.max_reference_recompute_ratio",
    )
    if selected_score <= required_score:
        raise SyncError("selected Open MOS does not clear the full-gate requirement")
    if selected_secondary <= secondary_required:
        raise SyncError("selected Token F1 does not clear the full-gate requirement")
    if selected_recompute > max_recompute:
        raise SyncError("selected recompute ratio exceeds the full-gate maximum")
    if selected_reference > max_reference:
        raise SyncError(
            "selected reference compute ratio exceeds the full-gate maximum"
        )
    provenance = selection.get("artifact_provenance")
    if (
        not isinstance(provenance, dict)
        or _integer(
            provenance.get("schema_version"), "artifact_provenance.schema_version"
        )
        < 1
    ):
        raise SyncError("full selection is missing versioned artifact_provenance")
    schema, _, _ = selection_identity(provenance)
    if schema == LEGACY_SELECTION_SCHEMA:
        revisions = provenance.get("revisions")
        for key in ("candidate_model", "dataset", "open_mos_judge"):
            if (
                not isinstance(revisions, dict)
                or not str(revisions.get(key, "")).strip()
            ):
                raise SyncError(
                    f"full selection provenance is missing revision {key!r}"
                )
    else:
        if method != ROUTE31_WINNER_METHOD:
            raise SyncError("Route31 campaign selected an unexpected method")
        if passing != [ROUTE31_WINNER_METHOD]:
            raise SyncError("Route31 campaign passing candidate set is invalid")
        if max_recompute != Decimal("0.98") or max_reference != Decimal("0.98"):
            raise SyncError(
                "Route31 selection recompute caps differ from the campaign contract"
            )
        if sample_counts != {item: 634 for item in ROUTE31_METHODS}:
            raise SyncError(
                "Route31 selection sample counts do not match all four methods"
            )
        candidate_scores = selection.get("candidate_scores")
        candidate_secondary = selection.get("candidate_secondary_scores")
        baseline_scores = selection.get("baseline_scores")
        baseline_secondary = selection.get("baseline_secondary_scores")
        if (
            not isinstance(candidate_scores, dict)
            or set(candidate_scores) != {method}
            or not isinstance(candidate_secondary, dict)
            or set(candidate_secondary) != {method}
            or not isinstance(baseline_scores, dict)
            or set(baseline_scores) != set(ROUTE31_METHODS[:-1])
            or not isinstance(baseline_secondary, dict)
            or set(baseline_secondary) != set(ROUTE31_METHODS[:-1])
        ):
            raise SyncError(
                "Route31 selection score maps do not match the campaign methods"
            )
        if not _published_matches(candidate_scores[method], selected_score):
            raise SyncError(
                "Route31 selected Open MOS differs from its candidate score map"
            )
        if not _published_matches(candidate_secondary[method], selected_secondary):
            raise SyncError(
                "Route31 selected Token F1 differs from its candidate score map"
            )
        for baseline in ROUTE31_METHODS[:-1]:
            if selected_score <= _decimal(
                baseline_scores[baseline], f"selection.baseline_scores.{baseline}"
            ):
                raise SyncError(
                    "Route31 winner does not strictly beat every Open-MOS floor"
                )
            if selected_secondary <= _decimal(
                baseline_secondary[baseline],
                f"selection.baseline_secondary_scores.{baseline}",
            ):
                raise SyncError(
                    "Route31 winner does not strictly beat every Token-F1 floor"
                )
    return method


def validate_strict_multifamily_selection(
    selection: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    family: str,
    base_method: str,
    expected_samples: int,
    completion_schema: str | None = None,
) -> tuple[str, tuple[str, ...]]:
    """Validate one raw full-gate JSON used by the multi-family release."""

    if expected_samples != 634:
        raise SyncError(
            "multi-family releases require the published 634-sample contract"
        )
    if (
        selection.get("status") != "passed"
        or selection.get("metric") != "open_mos"
        or selection.get("secondary_metric") != "token_f1"
    ):
        raise SyncError(f"{family} selection did not pass the OpenMOS + Token-F1 gate")
    selected = str(selection.get("selected_method") or "").strip()
    if not selected or selected != str(candidate.get("method") or "").strip():
        raise SyncError(f"{family} selected method does not match its leaderboard row")
    candidates = selection.get("candidate_scores")
    candidate_secondary = selection.get("candidate_secondary_scores")
    baselines = selection.get("baseline_scores")
    baseline_secondary = selection.get("baseline_secondary_scores")
    if (
        not isinstance(candidates, dict)
        or not isinstance(candidate_secondary, dict)
        or not isinstance(baselines, dict)
        or not isinstance(baseline_secondary, dict)
        or not candidates
        or not baselines
        or set(candidates) != set(candidate_secondary)
        or set(baselines) != set(baseline_secondary)
        or set(candidates).intersection(baselines)
        or selected not in candidates
        or base_method not in baselines
    ):
        raise SyncError(f"{family} selection candidate/control method set is invalid")
    if completion_schema == QWEN_DUAL_COMPLETION_SCHEMA:
        method_contract = MULTIFAMILY_DUAL_SELECTION_CONTRACTS.get(family)
    elif completion_schema == QWEN7_FLOOR_FULL_COMPLETION_SCHEMA:
        method_contract = (
            MULTIFAMILY_QWEN7_FLOOR_SELECTION_CONTRACT
            if family == "qwen7"
            else None
        )
    else:
        method_contract = MULTIFAMILY_SELECTION_CONTRACTS.get(family)
    if method_contract is not None:
        expected_baselines = set(method_contract["baselines"])
        expected_candidates = method_contract["candidates"]
        if set(baselines) != expected_baselines or (
            expected_candidates is not None
            and set(candidates) != set(expected_candidates)
        ):
            raise SyncError(
                f"{family} selection does not contain its exact control/candidate set"
            )
    passing = selection.get("passing_candidates")
    if (
        not isinstance(passing, list)
        or not passing
        or passing[0] != selected
        or len(set(passing)) != len(passing)
        or not set(passing) <= set(candidates)
    ):
        raise SyncError(f"{family} passing-candidate order/set is invalid")

    selected_score = _decimal(
        selection.get("selected_score"), f"{family}.selected_score"
    )
    selected_secondary = _decimal(
        selection.get("selected_secondary_score"), f"{family}.selected_secondary_score"
    )
    selected_recompute = _decimal(
        selection.get("selected_recompute_ratio"), f"{family}.selected_recompute_ratio"
    )
    selected_reference = _decimal(
        selection.get(
            "selected_reference_patch_compute_ratio",
            selection.get("selected_reference_recompute_ratio"),
        ),
        f"{family}.selected_reference_patch_compute_ratio",
    )
    required = _decimal(selection.get("required_score"), f"{family}.required_score")
    secondary_required = _decimal(
        selection.get("secondary_required_score"), f"{family}.secondary_required_score"
    )
    max_recompute = _decimal(
        selection.get("max_recompute_ratio"), f"{family}.max_recompute_ratio"
    )
    max_reference = _decimal(
        selection.get("max_reference_recompute_ratio"),
        f"{family}.max_reference_recompute_ratio",
    )
    assert None not in {
        selected_score,
        selected_secondary,
        selected_recompute,
        selected_reference,
        required,
        secondary_required,
        max_recompute,
        max_reference,
    }
    baseline_values = {
        method: _decimal(value, f"{family}.baseline_scores.{method}")
        for method, value in baselines.items()
    }
    baseline_secondary_values = {
        method: _decimal(value, f"{family}.baseline_secondary_scores.{method}")
        for method, value in baseline_secondary.items()
    }
    candidate_value = _decimal(
        candidates[selected], f"{family}.candidate_scores.{selected}"
    )
    candidate_secondary_value = _decimal(
        candidate_secondary[selected], f"{family}.candidate_secondary_scores.{selected}"
    )
    if (
        selected_score != candidate_value
        or selected_secondary != candidate_secondary_value
        or required != max(baseline_values.values())
        or secondary_required != max(baseline_secondary_values.values())
        or selected_score <= required
        or selected_secondary <= secondary_required
    ):
        raise SyncError(f"{family} winner does not strictly beat every control")
    if (
        max_recompute != Decimal("0.98")
        or max_reference != Decimal("0.98")
        or not Decimal(0) <= selected_recompute <= max_recompute
        or not Decimal(0) <= selected_reference <= max_reference
    ):
        raise SyncError(f"{family} winner violates the recompute ceilings")
    if not _published_matches(selected_score, candidate.get("open_mos")):
        raise SyncError(f"{family} selected Open MOS differs from the release CSV")
    if not _published_matches(selected_secondary, candidate.get("token_f1")):
        raise SyncError(f"{family} selected Token F1 differs from the release CSV")
    if not _published_matches(selected_recompute, candidate.get("recompute_ratio")):
        raise SyncError(
            f"{family} selected recompute ratio differs from the release CSV"
        )
    if not _published_matches(
        selected_reference, candidate.get("reference_recompute_ratio")
    ):
        raise SyncError(
            f"{family} selected reference ratio differs from the release CSV"
        )

    methods = tuple(sorted({*candidates, *baselines}))
    counts = selection.get("sample_counts")
    if (
        not isinstance(counts, dict)
        or set(counts) != set(methods)
        or any(
            _integer(value, f"{family}.sample_counts") != expected_samples
            for value in counts.values()
        )
    ):
        raise SyncError(f"{family} selection does not bind every method to 634 samples")
    return selected, methods


def _validate_nested_descriptors(
    payload: Any,
    *,
    base_directory: Path,
    label: str,
) -> set[tuple[Path, str]]:
    validated: set[tuple[Path, str]] = set()

    def visit(value: Any, location: str) -> None:
        if isinstance(value, dict):
            has_path = "path" in value
            has_sha = "sha256" in value
            if has_path != has_sha:
                raise SyncError(f"{location} has an incomplete path/SHA descriptor")
            if has_path:
                path, digest = validate_artifact_descriptor(
                    {"path": value["path"], "sha256": value["sha256"]},
                    base_directory=base_directory,
                    label=location,
                )
                validated.add((path, digest))
            for key, child in value.items():
                if key not in {"path", "sha256"}:
                    visit(child, f"{location}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{location}[{index}]")

    visit(payload, label)
    return validated


def _inspect_multifamily_matrix(
    path: Path,
    *,
    family: str,
    expected_methods: set[str],
    expected_samples: int,
    judge: Mapping[str, Any],
) -> dict[str, Decimal]:
    rows, fields = _read_csv(path)
    required = {
        "method",
        "open_mos_score",
        "mos_judge_model",
        "mos_judge_revision",
        "judge_fingerprint",
        "error",
    }
    if not required <= set(fields) or not {"sample_id", "doc_id"}.intersection(fields):
        raise SyncError(f"{family} Open-MOS matrix is missing release columns")
    rows_by_method: dict[str, list[Mapping[str, str]]] = {}
    for row in rows:
        method = str(row.get("method") or "").strip()
        rows_by_method.setdefault(method, []).append(row)
    if set(rows_by_method) != expected_methods:
        raise SyncError(f"{family} matrix methods differ from its selection")
    means: dict[str, Decimal] = {}
    common_doc_ids: set[str] | None = None
    common_text_by_doc: dict[str, tuple[str, ...]] | None = None
    text_columns = tuple(
        column
        for column in ("question", "answer", "reference", "target")
        if column in fields
    )
    for method, method_rows in rows_by_method.items():
        identities = [
            str(row.get("sample_id") or row.get("doc_id") or "").strip()
            for row in method_rows
        ]
        if (
            len(method_rows) != expected_samples
            or len(set(identities)) != expected_samples
            or "" in identities
        ):
            raise SyncError(
                f"{family} matrix method {method} is not an exact 634-row set"
            )
        doc_ids = [str(row.get("doc_id") or "").strip() for row in method_rows]
        if (
            "doc_id" not in fields
            or "" in doc_ids
            or len(set(doc_ids)) != expected_samples
        ):
            raise SyncError(f"{family} matrix method {method} has no exact doc_id set")
        method_doc_ids = set(doc_ids)
        if common_doc_ids is None:
            common_doc_ids = method_doc_ids
        elif method_doc_ids != common_doc_ids:
            raise SyncError(f"{family} matrix doc_id sets differ across methods")
        if text_columns:
            method_text = {
                str(row.get("doc_id") or "").strip(): tuple(
                    str(row.get(column) or "") for column in text_columns
                )
                for row in method_rows
            }
            if common_text_by_doc is None:
                common_text_by_doc = method_text
            elif method_text != common_text_by_doc:
                raise SyncError(
                    f"{family} matrix question/reference fields differ across methods"
                )
        scores: list[Decimal] = []
        for row in method_rows:
            if (
                str(row.get("error") or "").strip()
                or str(row.get("mos_judge_model") or "").strip() != judge["model"]
                or str(row.get("mos_judge_revision") or "").strip() != judge["revision"]
                or str(row.get("judge_fingerprint") or "").strip()
                != judge["fingerprint"]
            ):
                raise SyncError(
                    f"{family} matrix judge identity/error contract changed"
                )
            score = _decimal(
                row.get("open_mos_score"), f"{family}.{method}.open_mos_score"
            )
            assert score is not None
            if not Decimal(0) <= score <= Decimal(5):
                raise SyncError(f"{family} matrix Open-MOS score is outside [0,5]")
            scores.append(score)
        if not scores:
            raise SyncError(f"{family} matrix method {method} has no scores")
        means[method] = sum(scores, Decimal(0)) / Decimal(len(scores))
    return means


def _open_mos_means_match(
    matrix_means: Mapping[str, Decimal],
    score_maps: Mapping[str, Any],
    *,
    family: str,
) -> bool:
    return set(matrix_means) == set(score_maps) and all(
        math.isclose(
            float(matrix_means[method]),
            float(_decimal(score, f"{family}.{method}.selection_open_mos")),
            rel_tol=0,
            abs_tol=1e-12,
        )
        for method, score in score_maps.items()
    )


def _validate_multifamily_summary(
    path: Path,
    *,
    family: str,
    selected_method: str,
    expected_samples: int,
) -> None:
    rows, _ = _read_csv(path)
    matches = [
        row
        for row in rows
        if str(row.get("method") or "").strip() == selected_method
        and str(row.get("task") or "").strip() == LPM_TASK
        and str(row.get("run_status") or "").strip().lower() == "success"
    ]
    if (
        len(matches) != 1
        or _integer(matches[0].get("samples"), f"{family}.summary.samples")
        != expected_samples
        or not str(matches[0].get("result_json") or "").strip()
    ):
        raise SyncError(f"{family} winner summary is not one complete 634-sample row")


def _inspect_legacy_open_mos_method(
    path: Path,
    *,
    method: str,
    judge: Mapping[str, Any],
    expected_fields: tuple[str, ...] | None,
) -> tuple[tuple[str, ...], str]:
    digest = hashlib.sha256()
    count = 0
    identities: set[str] = set()
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = tuple(reader.fieldnames or ())
            required = {
                "sample_id",
                "method",
                "open_mos_score",
                "mos_judge_model",
                "mos_judge_revision",
                "judge_fingerprint",
                "error",
            }
            if not required.issubset(fields) or (
                expected_fields is not None and fields != expected_fields
            ):
                raise SyncError(f"legacy Open-MOS columns differ for {method}")
            for row in reader:
                identity = str(
                    row.get("sample_id") or row.get("doc_id") or ""
                ).strip()
                score = _decimal(
                    row.get("open_mos_score"), f"legacy.{method}.open_mos_score"
                )
                if (
                    not identity
                    or identity in identities
                    or str(row.get("method") or "") != method
                    or str(row.get("error") or "").strip()
                    or str(row.get("mos_judge_model") or "") != judge["model"]
                    or str(row.get("mos_judge_revision") or "")
                    != judge["revision"]
                    or str(row.get("judge_fingerprint") or "")
                    != judge["fingerprint"]
                    or score is None
                    or not Decimal(0) <= score <= Decimal(5)
                ):
                    raise SyncError(f"legacy Open-MOS row differs for {method}")
                identities.add(identity)
                count += 1
                digest.update(
                    json.dumps(
                        [row.get(field, "") for field in fields],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                )
                digest.update(b"\n")
    except (OSError, csv.Error) as exc:
        raise SyncError(
            f"could not inspect legacy Open-MOS method {method}: {exc}"
        ) from exc
    if count != MULTIFAMILY_EXPECTED_SAMPLES:
        raise SyncError(f"legacy Open-MOS method {method} is not 634 rows")
    return fields, digest.hexdigest()


def validate_legacy_open_mos_release_binding(
    binding: Any,
    *,
    metadata_directory: Path,
    judge: Mapping[str, Any],
    open_mos_sources: set[tuple[Path, str]],
    open_mos_by_method: Any,
    route31_entry: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-authenticate the nine-method legacy MOS completion at publish time."""

    if (
        not isinstance(binding, dict)
        or set(binding)
        != {
            "schema_version",
            "campaign",
            "expected_samples",
            "rows",
            "judge",
            "completion",
            "contract",
            "matrix_csv",
            "methods",
        }
        or binding.get("schema_version") != 1
        or binding.get("campaign") != LEGACY_OPEN_MOS_CAMPAIGN
        or _integer(binding.get("expected_samples"), "legacy MOS expected_samples")
        != MULTIFAMILY_EXPECTED_SAMPLES
        or _integer(binding.get("rows"), "legacy MOS rows")
        != LEGACY_OPEN_MOS_EXPECTED_ROWS
        or binding.get("judge")
        != {
            "model": judge["model"],
            "revision": judge["revision"],
            "judge_fingerprint": judge["fingerprint"],
        }
    ):
        raise SyncError("legacy Open-MOS release binding header/judge is invalid")

    def exact_descriptor(
        value: Any,
        label: str,
        *,
        expected_path: Path | None = None,
    ) -> tuple[Path, str]:
        if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
            raise SyncError(f"{label} must be an exact path/SHA object")
        pair = validate_artifact_descriptor(
            value,
            base_directory=metadata_directory,
            label=label,
            expected_path=expected_path,
        )
        if value != {"path": str(pair[0]), "sha256": pair[1]}:
            raise SyncError(f"{label} descriptor is not canonical")
        return pair

    completion_pair = exact_descriptor(
        binding.get("completion"), "legacy Open-MOS completion"
    )
    completion_path = completion_pair[0]
    if completion_path.name != "completed.json":
        raise SyncError("legacy Open-MOS completion path is not canonical")
    completion = _load_json(completion_path)
    if (
        set(completion)
        != {
            "schema_version",
            "campaign",
            "status",
            "expected_samples",
            "inventory_methods",
            "methods",
            "rows",
            "judge",
            "contract",
            "scorer_command",
            "artifacts",
            "release_builder",
        }
        or completion.get("schema_version") != 1
        or completion.get("campaign") != LEGACY_OPEN_MOS_CAMPAIGN
        or completion.get("status") != "passed"
        or _integer(
            completion.get("expected_samples"), "legacy completion expected_samples"
        )
        != MULTIFAMILY_EXPECTED_SAMPLES
        or _integer(completion.get("rows"), "legacy completion rows")
        != LEGACY_OPEN_MOS_EXPECTED_ROWS
        or completion.get("inventory_methods")
        != list(LEGACY_OPEN_MOS_INVENTORY_METHODS)
        or completion.get("methods") != list(LEGACY_OPEN_MOS_METHODS)
        or not isinstance(completion.get("scorer_command"), list)
        or not completion["scorer_command"]
    ):
        raise SyncError("legacy Open-MOS completion header/schema is invalid")
    completion_judge = completion.get("judge")
    if not isinstance(completion_judge, dict) or (
        completion_judge.get("model") != judge["model"]
        or completion_judge.get("revision") != judge["revision"]
        or completion_judge.get("judge_fingerprint") != judge["fingerprint"]
    ):
        raise SyncError("legacy Open-MOS completion judge differs from release")

    contract_pair = exact_descriptor(
        completion.get("contract"), "legacy Open-MOS contract"
    )
    if binding.get("contract") != {
        "path": str(contract_pair[0]),
        "sha256": contract_pair[1],
    }:
        raise SyncError("legacy Open-MOS metadata/contract descriptor differs")
    contract = _load_json(contract_pair[0])
    if (
        set(contract)
        != {
            "campaign",
            "expected_samples",
            "inputs",
            "judge",
            "methods",
            "release_merge",
            "schema_version",
            "scored_methods",
            "source_files",
        }
        or contract.get("schema_version") != 1
        or contract.get("campaign") != LEGACY_OPEN_MOS_CAMPAIGN
        or _integer(contract.get("expected_samples"), "legacy contract samples")
        != MULTIFAMILY_EXPECTED_SAMPLES
        or contract.get("methods") != list(LEGACY_OPEN_MOS_INVENTORY_METHODS)
        or contract.get("scored_methods") != list(LEGACY_OPEN_MOS_METHODS)
        or contract.get("judge") != completion_judge
        or not isinstance(contract.get("inputs"), list)
        or not isinstance(contract.get("source_files"), list)
    ):
        raise SyncError("legacy Open-MOS contract differs from completion")

    release_contract = contract.get("release_merge")
    release_binding = completion.get("release_builder")
    if (
        not isinstance(release_contract, dict)
        or set(release_contract)
        != {
            "explicit_per_method_methods",
            "omit_legacy_methods",
            "route31_open_mos_matrix",
            "route31_selection",
        }
        or release_contract.get("explicit_per_method_methods")
        != list(LEGACY_OPEN_MOS_METHODS)
        or release_contract.get("omit_legacy_methods")
        != list(LEGACY_OPEN_MOS_OMITTED_METHODS)
        or not isinstance(release_binding, dict)
        or set(release_binding)
        != {
            "route31_open_mos_matrix",
            "route31_selection",
            "explicit_legacy_mos",
            "omitted_legacy_methods",
        }
        or release_binding.get("omitted_legacy_methods")
        != list(LEGACY_OPEN_MOS_OMITTED_METHODS)
        or release_binding.get("route31_selection")
        != release_contract.get("route31_selection")
        or release_binding.get("route31_open_mos_matrix")
        != release_contract.get("route31_open_mos_matrix")
        or release_binding.get("route31_selection")
        != route31_entry.get("selection")
        or release_binding.get("route31_open_mos_matrix")
        != route31_entry.get("open_mos_matrix")
    ):
        raise SyncError("legacy Open-MOS Route31/release binding differs")
    exact_descriptor(
        release_binding["route31_selection"], "legacy Open-MOS Route31 selection"
    )
    exact_descriptor(
        release_binding["route31_open_mos_matrix"],
        "legacy Open-MOS Route31 matrix",
    )

    artifacts = completion.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "input_manifest",
        "matrix_csv",
        "matrix_jsonl",
        "matrix_markdown",
        "per_method",
    }:
        raise SyncError("legacy Open-MOS artifact inventory differs")
    output_root = completion_path.parent
    canonical = {
        "input_manifest": output_root / "provenance/input_manifest.json",
        "matrix_csv": output_root / "mos/matrix.csv",
        "matrix_jsonl": output_root / "mos/matrix.jsonl",
        "matrix_markdown": output_root / "mos/matrix.md",
    }
    artifact_pairs = {
        name: exact_descriptor(
            artifacts.get(name),
            f"legacy Open-MOS {name}",
            expected_path=expected_path,
        )
        for name, expected_path in canonical.items()
    }
    if binding.get("matrix_csv") != {
        "path": str(artifact_pairs["matrix_csv"][0]),
        "sha256": artifact_pairs["matrix_csv"][1],
    }:
        raise SyncError("legacy Open-MOS metadata/matrix descriptor differs")

    raw_methods = artifacts.get("per_method")
    metadata_methods = binding.get("methods")
    explicit_methods = release_binding.get("explicit_legacy_mos")
    if (
        not isinstance(raw_methods, dict)
        or set(raw_methods) != set(LEGACY_OPEN_MOS_METHODS)
        or raw_methods != explicit_methods
        or raw_methods != metadata_methods
        or not isinstance(open_mos_by_method, dict)
    ):
        raise SyncError("legacy Open-MOS exact nine-method binding differs")

    per_method_digests: dict[str, str] = {}
    expected_fields: tuple[str, ...] | None = None
    for method in LEGACY_OPEN_MOS_METHODS:
        raw = raw_methods[method]
        if (
            not isinstance(raw, dict)
            or set(raw) != {"path", "sha256", "rows"}
            or _integer(raw.get("rows"), f"legacy {method}.rows")
            != MULTIFAMILY_EXPECTED_SAMPLES
        ):
            raise SyncError(f"legacy Open-MOS descriptor differs for {method}")
        pair = exact_descriptor(
            {"path": raw.get("path"), "sha256": raw.get("sha256")},
            f"legacy Open-MOS method {method}",
            expected_path=output_root / "per_method" / f"{method}.csv",
        )
        if pair not in open_mos_sources:
            raise SyncError(f"legacy Open-MOS method {method} is not a release source")
        if _artifact_path(
            open_mos_by_method.get(method),
            metadata_directory,
            f"legacy Open-MOS mapping {method}",
        ) != pair[0]:
            raise SyncError(f"legacy Open-MOS method/path mapping differs for {method}")
        expected_fields, digest = _inspect_legacy_open_mos_method(
            pair[0],
            method=method,
            judge=judge,
            expected_fields=expected_fields,
        )
        per_method_digests[method] = digest

    matrix_digests = {method: hashlib.sha256() for method in LEGACY_OPEN_MOS_METHODS}
    matrix_counts = {method: 0 for method in LEGACY_OPEN_MOS_METHODS}
    matrix_ids: set[str] = set()
    required_matrix_fields = {
        "sample_id",
        "method",
        "open_mos_correctness",
        "open_mos_score",
        "mos_judge_model",
        "mos_judge_revision",
        "request_fingerprint",
        "judge_fingerprint",
        "error",
        "question",
        "answer",
        "pred",
        "open_mos_review",
    }
    try:
        with artifact_pairs["matrix_csv"][0].open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            reader = csv.DictReader(handle)
            fields = tuple(reader.fieldnames or ())
            if fields != expected_fields or not required_matrix_fields.issubset(fields):
                raise SyncError("legacy Open-MOS matrix header differs")
            for row in reader:
                method = str(row.get("method") or "")
                sample_id = str(row.get("sample_id") or "")
                score = _decimal(
                    row.get("open_mos_score"), "legacy matrix open_mos_score"
                )
                if (
                    method not in matrix_digests
                    or not sample_id
                    or sample_id in matrix_ids
                    or str(row.get("error") or "").strip()
                    or str(row.get("mos_judge_model") or "") != judge["model"]
                    or str(row.get("mos_judge_revision") or "")
                    != judge["revision"]
                    or str(row.get("judge_fingerprint") or "")
                    != judge["fingerprint"]
                    or score is None
                    or score != score.to_integral_value()
                    or not Decimal(0) <= score <= Decimal(5)
                ):
                    raise SyncError("legacy Open-MOS matrix row differs")
                matrix_ids.add(sample_id)
                matrix_counts[method] += 1
                matrix_digests[method].update(
                    json.dumps(
                        [row.get(field, "") for field in fields],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                )
                matrix_digests[method].update(b"\n")
    except (OSError, csv.Error) as exc:
        raise SyncError(f"could not inspect legacy Open-MOS matrix: {exc}") from exc
    if (
        matrix_counts
        != {method: MULTIFAMILY_EXPECTED_SAMPLES for method in LEGACY_OPEN_MOS_METHODS}
        or len(matrix_ids) != LEGACY_OPEN_MOS_EXPECTED_ROWS
        or any(
            matrix_digests[method].hexdigest() != per_method_digests[method]
            for method in LEGACY_OPEN_MOS_METHODS
        )
    ):
        raise SyncError("legacy Open-MOS matrix/per-method completion differs")
    return dict(binding)


def _canonical_fingerprint(payload: Mapping[str, Any]) -> str:
    def json_value(value: Any) -> Any:
        if isinstance(value, Decimal):
            return float(value)
        if isinstance(value, dict):
            return {key: json_value(child) for key, child in value.items()}
        if isinstance(value, list):
            return [json_value(child) for child in value]
        return value

    unsigned = json_value(dict(payload))
    unsigned.pop("campaign_fingerprint", None)
    encoded = json.dumps(
        unsigned,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _nested_descriptor(
    value: Any,
    *,
    base_directory: Path,
    label: str,
) -> tuple[Path, str]:
    if not isinstance(value, dict) or "path" not in value or "sha256" not in value:
        raise SyncError(f"{label} has no path/SHA descriptor")
    return validate_artifact_descriptor(
        {"path": value["path"], "sha256": value["sha256"]},
        base_directory=base_directory,
        label=label,
    )


def _validate_checksum_manifest(path: Path, label: str) -> list[tuple[Path, str]]:
    rows: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SyncError(f"could not read {label}: {exc}") from exc
    for index, line in enumerate(lines, start=1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise SyncError(f"{label}:{index} is not a strict sha256sum row")
        artifact = _artifact_path(match.group(2), path.parent, f"{label}:{index}")
        if artifact in seen or sha256_file(artifact) != match.group(1):
            raise SyncError(f"{label}:{index} failed checksum/path validation")
        seen.add(artifact)
        rows.append((artifact, match.group(1)))
    if not rows:
        raise SyncError(f"{label} is empty")
    return rows


def _sample_identity(
    path: Path,
    *,
    label: str,
    expected_samples: int,
) -> dict[str, tuple[str, str]]:
    rows: dict[str, tuple[str, str]] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = _strict_json_loads(line, f"{label}:{line_number}")
                if not isinstance(row, dict) or "doc_id" not in row:
                    raise SyncError(f"{label}:{line_number} has no object/doc_id")
                doc_id = json.dumps(
                    row["doc_id"],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if doc_id in rows or "input" not in row or "target" not in row:
                    raise SyncError(
                        f"{label}:{line_number} has duplicate/missing typed identity"
                    )
                rows[doc_id] = (
                    json.dumps(
                        row["input"],
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    json.dumps(
                        row["target"],
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                )
    except OSError as exc:
        raise SyncError(f"could not validate {label}: {exc}") from exc
    if len(rows) != expected_samples:
        raise SyncError(f"{label} has {len(rows)} rows; expected {expected_samples}")
    return rows


def _summary_row(
    path: Path,
    *,
    family: str,
    method: str,
    expected_samples: int,
    expected_result: Path | None = None,
) -> Mapping[str, str]:
    rows, _ = _read_csv(path)
    matches = [
        row
        for row in rows
        if str(row.get("method") or "").strip() == method
        and str(row.get("task") or "").strip() == LPM_TASK
        and str(row.get("run_status") or "").strip().lower() == "success"
    ]
    if len(matches) != 1:
        raise SyncError(
            f"{family} summary must contain exactly one successful row for {method}"
        )
    row = matches[0]
    if (
        _integer(row.get("samples"), f"{family}.{method}.summary.samples")
        != expected_samples
    ):
        raise SyncError(f"{family} summary for {method} is not complete")
    raw_result = str(row.get("result_json") or "").strip()
    if not raw_result:
        raise SyncError(f"{family} summary for {method} has no result_json")
    if (
        expected_result is not None
        and Path(raw_result).expanduser().resolve() != expected_result
    ):
        raise SyncError(f"{family} summary/result binding mismatch for {method}")
    return row


def _selection_secondary(selection: Mapping[str, Any], method: str, family: str) -> Any:
    for field in ("baseline_secondary_scores", "candidate_secondary_scores"):
        values = selection.get(field)
        if isinstance(values, dict) and method in values:
            return values[method]
    raise SyncError(f"{family} selection has no Token-F1 binding for {method}")


def _assert_secondary_score(row: Mapping[str, Any], expected: Any, label: str) -> None:
    actual = _decimal(row.get("token_f1"), f"{label}.summary.token_f1")
    bound = _decimal(expected, f"{label}.selection.token_f1")
    assert actual is not None and bound is not None
    if not math.isclose(float(actual), float(bound), rel_tol=0, abs_tol=1e-12):
        raise SyncError(f"{label} Token-F1 does not match its authenticated summary")


def _validate_result_and_sample(
    *,
    family: str,
    method: str,
    sample: Path,
    result: Path,
    expected_samples: int,
) -> dict[str, tuple[str, str]]:
    suffix = "_results.json"
    if not result.name.endswith(suffix):
        raise SyncError(f"{family}.{method} result filename is unsupported")
    prefix = result.name[: -len(suffix)]
    if result.with_name(f"{prefix}_samples_densevideo.jsonl").resolve() != sample:
        raise SyncError(f"{family}.{method} sample/result pair is inconsistent")
    payload = _load_hashed_json(result, f"{family}.{method} result")
    config = payload.get("config")
    if (
        not isinstance(config, dict)
        or config.get("gen_kwargs") != {"max_new_tokens": 128, "temperature": 0}
        or _integer(config.get("limit"), f"{family}.{method}.result.limit")
        != expected_samples
        or str(config.get("batch_size")) != "1"
    ):
        raise SyncError(f"{family}.{method} result generation contract mismatch")
    return _sample_identity(
        sample, label=f"{family}.{method}.sample", expected_samples=expected_samples
    )


def _validate_run_summary(
    path: Path,
    *,
    family: str,
    method: str,
    role: str,
    expected_result: Path,
    expected_secondary: Any,
) -> Mapping[str, str]:
    row = _summary_row(
        path,
        family=family,
        method=method,
        expected_samples=MULTIFAMILY_EXPECTED_SAMPLES,
        expected_result=expected_result,
    )
    _assert_secondary_score(row, expected_secondary, f"{family}.{method}")
    policy = {"base": "disabled", "exact": "all", "candidate": "motion"}[role]
    policy_counts = _strict_json_loads(
        str(row.get("gate_policy_counts") or ""),
        f"{family}.{method}.gate_policy_counts",
    )
    if row.get("gate_policy") != policy or policy_counts != {
        policy: MULTIFAMILY_EXPECTED_SAMPLES
    }:
        raise SyncError(f"{family}.{method} gate-policy telemetry mismatch")
    values = {
        key: _decimal(row.get(key), f"{family}.{method}.{key}")
        for key in (
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
    }
    assert all(value is not None for value in values.values())
    requested = values["mean_requested_frames"]
    reference = values["mean_reference_frames"]
    orig = values["total_orig_patches"]
    reference_orig = values["total_reference_orig_patches"]
    recomputed = values["total_recomputed_patches"]
    ratio = values["mean_recompute_ratio"]
    reference_ratio = values["reference_patch_compute_ratio"]
    if (
        requested != 8
        or reference != 8
        or orig <= 0
        or reference_orig <= 0
        or recomputed < 0
        or values["mean_effective_fps"] <= 0
        or values["mean_throughput_fps"] <= 0
        or values["mean_wall_time_s"] <= 0
        or not math.isclose(
            float(ratio), float(recomputed / orig), rel_tol=0, abs_tol=1e-12
        )
        or not math.isclose(
            float(reference_ratio),
            float(recomputed / reference_orig),
            rel_tol=0,
            abs_tol=1e-12,
        )
    ):
        raise SyncError(f"{family}.{method} telemetry totals/rates are inconsistent")
    if role in {"base", "exact"}:
        if ratio != 1 or reference_ratio != 1:
            raise SyncError(
                f"{family}.{method} control recompute ratios must equal one"
            )
    elif not (
        0 <= ratio <= Decimal("0.98") and 0 <= reference_ratio <= Decimal("0.98")
    ):
        raise SyncError(f"{family}.{method} candidate exceeds recompute caps")
    return row


def _canonical_recovery_fingerprint(payload: Mapping[str, Any]) -> str:
    def json_value(value: Any) -> Any:
        if isinstance(value, Decimal):
            return float(value)
        if isinstance(value, dict):
            return {key: json_value(child) for key, child in value.items()}
        if isinstance(value, list):
            return [json_value(child) for child in value]
        return value

    unsigned = json_value(dict(payload))
    unsigned.pop("recovery_fingerprint", None)
    return hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _recovery_descriptor(
    value: Any,
    *,
    base_directory: Path,
    label: str,
    exact: bool = True,
) -> tuple[Path, str]:
    if exact and (not isinstance(value, dict) or set(value) != {"path", "sha256"}):
        raise SyncError(f"{label} is not an exact path/SHA descriptor")
    return _nested_descriptor(value, base_directory=base_directory, label=label)


def _normalized_recovery_doc_id(value: Any, label: str) -> str:
    if value is None or isinstance(value, bool) or not isinstance(value, (str, int)):
        raise SyncError(f"{label} has no valid doc_id")
    normalized = str(value).strip()
    if not normalized:
        raise SyncError(f"{label} has an empty doc_id")
    return normalized


def _recovery_tagged_metrics(line: str) -> dict[str, str]:
    pairs = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)", line)
    if len({key for key, _ in pairs}) != len(pairs):
        raise SyncError("qwen3 recovery telemetry contains duplicate fields")
    return dict(pairs)


def _recovery_positive(row: Mapping[str, str], field: str, label: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise SyncError(f"{label} has no finite {field}") from exc
    if not math.isfinite(value) or value <= 0:
        raise SyncError(f"{label}.{field} must be positive")
    return value


def _validate_recovery_raw_telemetry(
    path: Path,
    *,
    method: str,
    summary: Mapping[str, str],
) -> None:
    dense_rows: list[dict[str, str]] = []
    fps_rows: list[dict[str, str]] = []
    run_status = ""
    return_code: int | None = None
    try:
        with path.open("r", encoding="utf-8", errors="strict") as handle:
            for line in handle:
                if "[DENSE_METRICS]" in line:
                    dense_rows.append(_recovery_tagged_metrics(line))
                elif "[FPS_STATS]" in line:
                    fps_rows.append(_recovery_tagged_metrics(line))
                elif "[RUN_META]" in line:
                    values = _recovery_tagged_metrics(line)
                    if "run_status" in values:
                        run_status = values["run_status"]
                    if "return_code" in values:
                        return_code = _integer(
                            values["return_code"], f"qwen3.{method}.return_code"
                        )
    except OSError as exc:
        raise SyncError(f"could not read qwen3 telemetry for {method}: {exc}") from exc
    if (
        len(dense_rows) != MULTIFAMILY_EXPECTED_SAMPLES
        or len(fps_rows) != MULTIFAMILY_EXPECTED_SAMPLES
        or run_status != "success"
        or return_code != 0
    ):
        raise SyncError(
            f"qwen3.{method} raw telemetry/status is not exactly 634 successful rows"
        )
    effective: list[float] = []
    wall: list[float] = []
    throughput: list[float] = []
    recomputed: list[int] = []
    original: list[int] = []
    reference: list[int] = []
    for index, (dense, fps) in enumerate(zip(dense_rows, fps_rows), start=1):
        label = f"qwen3.{method}.telemetry[{index}]"
        dense_effective = _recovery_positive(dense, "effective_fps", label)
        fps_effective = _recovery_positive(fps, "effective_fps", label)
        current_wall = _recovery_positive(dense, "wall_time_s", label)
        current_throughput = _recovery_positive(dense, "throughput_fps", label)
        sampled = _integer(dense.get("sampled_frames"), f"{label}.sampled_frames")
        fps_sampled = _integer(
            fps.get("sampled_frames"), f"{label}.fps_sampled_frames"
        )
        current_recomputed = _integer(
            dense.get("recomputed_patches"), f"{label}.recomputed_patches"
        )
        current_original = _integer(
            dense.get("orig_patches"), f"{label}.orig_patches"
        )
        current_reference = _integer(
            dense.get("reference_orig_patches"),
            f"{label}.reference_orig_patches",
        )
        if (
            sampled <= 0
            or fps_sampled != sampled
            or current_recomputed < 0
            or current_original <= 0
            or current_reference <= 0
            or current_recomputed > current_original
            or not math.isclose(
                dense_effective, fps_effective, rel_tol=0, abs_tol=1e-12
            )
        ):
            raise SyncError(f"{label} contains invalid count/effective-FPS values")
        actual_ratio = float(
            _decimal(dense.get("recompute_ratio"), f"{label}.recompute_ratio")
        )
        projected_ratio = float(
            _decimal(
                dense.get("patch_projection_recompute_ratio"),
                f"{label}.patch_projection_recompute_ratio",
            )
        )
        reference_ratio = float(
            _decimal(
                dense.get("patch_projection_compute_ratio_vs_reference"),
                f"{label}.reference_ratio",
            )
        )
        if (
            not math.isclose(
                actual_ratio,
                current_recomputed / current_original,
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
            or not math.isclose(
                projected_ratio,
                current_recomputed / current_original,
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
            or not math.isclose(
                reference_ratio,
                current_recomputed / current_reference,
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
        ):
            raise SyncError(f"{label} recompute formula differs")
        effective.append(dense_effective)
        wall.append(current_wall)
        throughput.append(current_throughput)
        recomputed.append(current_recomputed)
        original.append(current_original)
        reference.append(current_reference)
    total_recomputed = sum(recomputed)
    total_original = sum(original)
    total_reference = sum(reference)
    expected = {
        "mean_effective_fps": sum(effective) / MULTIFAMILY_EXPECTED_SAMPLES,
        "mean_wall_time_s": sum(wall) / MULTIFAMILY_EXPECTED_SAMPLES,
        "mean_throughput_fps": sum(throughput) / MULTIFAMILY_EXPECTED_SAMPLES,
        "mean_recomputed_patches": total_recomputed / MULTIFAMILY_EXPECTED_SAMPLES,
        "mean_orig_patches": total_original / MULTIFAMILY_EXPECTED_SAMPLES,
        "total_recomputed_patches": float(total_recomputed),
        "total_orig_patches": float(total_original),
        "total_reference_orig_patches": float(total_reference),
        "mean_recompute_ratio": total_recomputed / total_original,
        "reference_patch_compute_ratio": total_recomputed / total_reference,
        "mean_patch_projection_recompute_ratio": total_recomputed / total_original,
        "mean_patch_projection_compute_ratio_vs_reference": total_recomputed
        / total_reference,
    }
    for field, value in expected.items():
        actual = _decimal(summary.get(field), f"qwen3.{method}.summary.{field}")
        assert actual is not None
        if not math.isclose(float(actual), value, rel_tol=1e-9, abs_tol=1e-9):
            raise SyncError(f"qwen3.{method} raw/summary telemetry differs for {field}")
    if Path(str(summary.get("run_log") or "")).expanduser().resolve() != path:
        raise SyncError(f"qwen3.{method} summary does not bind its raw log")


def _validate_qwen3_recovery_contract(
    contract_path: Path,
    *,
    contract_sha: str,
    recovery_fingerprint: str,
    campaign_fingerprint: str,
    recovery_driver_sha: str,
) -> dict[str, Any]:
    contract = _load_hashed_json(contract_path, "qwen3 recovery contract")
    expected_fields = {
        "schema_version",
        "protocol",
        "family",
        "campaign_fingerprint",
        "recovery_fingerprint",
        "recovery_root",
        "expected_samples",
        "generation",
        "judge",
        "methods",
        "quality_gate",
        "recovery_driver",
        "recovery_sources",
        "legacy",
    }
    recovery_root = Path(str(contract.get("recovery_root") or "")).expanduser().resolve()
    if (
        set(contract) != expected_fields
        or contract.get("schema_version") != 1
        or contract.get("protocol") != QWEN3_RECOVERY_PROTOCOL
        or contract.get("family") != "qwen3"
        or contract.get("campaign_fingerprint") != campaign_fingerprint
        or contract.get("recovery_fingerprint") != recovery_fingerprint
        or _canonical_recovery_fingerprint(contract) != recovery_fingerprint
        or contract_path != recovery_root / "provenance/contract.json"
        or _integer(contract.get("expected_samples"), "qwen3.recovery.samples")
        != MULTIFAMILY_EXPECTED_SAMPLES
        or contract.get("generation") != {"max_new_tokens": 128, "temperature": 0}
        or contract.get("methods")
        != {
            "archived_public": QWEN3_RECOVERY_PUBLIC,
            "reused_from_original": list(QWEN3_RECOVERY_REUSED),
            "native_recovery": list(QWEN3_RECOVERY_NATIVE),
            "screen_passing_finalists": list(QWEN3_RECOVERY_FINALISTS),
            "full_evaluations": list(QWEN3_RECOVERY_METHODS),
        }
        or contract.get("quality_gate")
        != {
            "primary_metric": "open_mos",
            "secondary_metric": "token_f1",
            "strictly_greater": True,
            "public_and_fresh_controls": True,
            "max_recompute_ratio": Decimal("0.98"),
            "max_reference_recompute_ratio": Decimal("0.98"),
        }
    ):
        raise SyncError("qwen3 recovery contract core semantics/fingerprint differ")
    driver_pair = _recovery_descriptor(
        contract.get("recovery_driver"),
        base_directory=contract_path.parent,
        label="qwen3 recovery driver",
    )
    if driver_pair[1] != recovery_driver_sha:
        raise SyncError("qwen3 recovery contract/driver identity differs")
    sources = contract.get("recovery_sources")
    if not isinstance(sources, dict) or set(sources) != {"manifest", "entries"}:
        raise SyncError("qwen3 recovery source schema differs")
    source_manifest_pair = _recovery_descriptor(
        sources["manifest"],
        base_directory=contract_path.parent,
        label="qwen3 recovery source manifest",
    )
    source_rows = _validate_checksum_manifest(
        source_manifest_pair[0], "qwen3 recovery source manifest"
    )
    entries = sources.get("entries")
    if (
        len(source_rows) != 3
        or not isinstance(entries, list)
        or len(entries) != 3
        or {
            _recovery_descriptor(
                row,
                base_directory=contract_path.parent,
                label="qwen3 recovery source entry",
            )
            for row in entries
        }
        != set(source_rows)
        or {path.name for path, _ in source_rows}
        != {
            "qwen3_followup_recovery.py",
            "prepare_qwen3_followup_recovery.sh",
            "slurm_grt_followup_qwen3_recovery.sbatch",
        }
        or driver_pair not in source_rows
    ):
        raise SyncError("qwen3 recovery source manifest/inventory differs")
    legacy = contract.get("legacy")
    expected_legacy_fields = {
        "broken_screen_report",
        "campaign_contract",
        "campaign_source_snapshot",
        "config",
        "driver_manifest",
        "input_manifest",
        "original_driver",
        "reused_methods",
        "run_contract",
        "screen_mos_matrix",
        "screen_selection",
    }
    if not isinstance(legacy, dict) or set(legacy) != expected_legacy_fields:
        raise SyncError("qwen3 recovery legacy evidence schema differs")
    legacy_pairs = {
        name: _recovery_descriptor(
            legacy[name],
            base_directory=contract_path.parent,
            label=f"qwen3 legacy {name}",
        )
        for name in expected_legacy_fields - {"reused_methods"}
    }
    if any(
        legacy_pairs[name][1] != digest
        for name, digest in QWEN3_RECOVERY_FIXED_SHA256.items()
    ):
        raise SyncError("qwen3 fixed legacy evidence checksum differs")
    source_snapshot_rows = _validate_checksum_manifest(
        legacy_pairs["campaign_source_snapshot"][0],
        "qwen3 campaign source snapshot",
    )
    input_rows = _validate_checksum_manifest(
        legacy_pairs["input_manifest"][0], "qwen3 legacy input manifest"
    )
    driver_rows = _validate_checksum_manifest(
        legacy_pairs["driver_manifest"][0], "qwen3 legacy driver manifest"
    )
    if (
        len(source_snapshot_rows) != 28
        or len(input_rows) != 42
        or driver_rows != [legacy_pairs["original_driver"]]
        or input_rows[0] != legacy_pairs["original_driver"]
    ):
        raise SyncError("qwen3 legacy snapshot/manifest cardinality differs")
    run = _load_hashed_json(legacy_pairs["run_contract"][0], "qwen3 legacy run contract")
    if (
        run.get("schema_version") != 1
        or run.get("protocol") != "dive-grt-followup-v2-family-full-parallel-v1"
        or run.get("family") != "qwen3"
        or run.get("campaign_fingerprint") != campaign_fingerprint
        or run.get("driver_sha256") != QWEN3_RECOVERY_ORIGINAL_DRIVER_SHA256
        or run.get("expected_samples") != MULTIFAMILY_EXPECTED_SAMPLES
        or run.get("generation") != {"max_new_tokens": 128, "temperature": 0}
        or run.get("methods")
        != {
            "archived_public": QWEN3_RECOVERY_PUBLIC,
            "fresh_base": QWEN3_RECOVERY_REUSED[0],
            "all_patch_exact": QWEN3_RECOVERY_REUSED[1],
            "screen_passing_finalists": list(QWEN3_RECOVERY_FINALISTS),
            "full_evaluations": list(QWEN3_RECOVERY_METHODS),
        }
    ):
        raise SyncError("qwen3 legacy run contract semantics differ")
    for field, legacy_name in (
        ("campaign_contract", "campaign_contract"),
        ("screen_selection", "screen_selection"),
        ("screen_mos_matrix", "screen_mos_matrix"),
        ("config", "config"),
    ):
        pair = _recovery_descriptor(
            run.get(field),
            base_directory=legacy_pairs["run_contract"][0].parent,
            label=f"qwen3 run contract {field}",
        )
        if pair != legacy_pairs[legacy_name]:
            raise SyncError(f"qwen3 legacy run/contract {field} differs")
    screen = _load_hashed_json(
        legacy_pairs["screen_selection"][0], "qwen3 legacy screen selection"
    )
    if (
        screen.get("status") != "passed"
        or screen.get("passing_candidates") != list(QWEN3_RECOVERY_FINALISTS)
        or screen.get("selected_method") != QWEN3_RECOVERY_FINALISTS[0]
        or screen.get("metric") != "open_mos"
        or screen.get("secondary_metric") != "token_f1"
    ):
        raise SyncError("qwen3 legacy screen finalists differ")
    reused = legacy.get("reused_methods")
    if not isinstance(reused, dict) or set(reused) != set(QWEN3_RECOVERY_REUSED):
        raise SyncError("qwen3 legacy reusable marker set differs")
    for method in QWEN3_RECOVERY_REUSED:
        row = reused[method]
        if not isinstance(row, dict) or set(row) != {"marker", "artifacts"}:
            raise SyncError(f"qwen3 legacy reusable schema differs for {method}")
        marker_pair = _recovery_descriptor(
            row["marker"],
            base_directory=contract_path.parent,
            label=f"qwen3 legacy marker {method}",
        )
        if marker_pair[1] != QWEN3_RECOVERY_OLD_MARKER_SHA256[method]:
            raise SyncError(f"qwen3 legacy marker checksum differs for {method}")
        artifacts = row.get("artifacts")
        if not isinstance(artifacts, dict) or set(artifacts) != {
            "sample",
            "summary",
            "result",
            "driver_log",
        }:
            raise SyncError(f"qwen3 legacy artifact schema differs for {method}")
        for field in artifacts:
            _recovery_descriptor(
                artifacts[field],
                base_directory=contract_path.parent,
                label=f"qwen3 legacy {method}.{field}",
            )
    if sha256_file(contract_path) != contract_sha:
        raise SyncError("qwen3 recovery contract changed during validation")
    return contract


def _validate_qwen_parallel_completion(
    completed: Mapping[str, Any],
    *,
    path: Path,
    entry: Mapping[str, Any],
    selection: Mapping[str, Any],
    selection_pair: tuple[Path, str],
    matrix_pair: tuple[Path, str],
    summary_pair: tuple[Path, str],
    judge: Mapping[str, Any],
) -> Path:
    family = str(entry["family"])
    selected = str(entry["selected_method"])
    fingerprint = str(entry["campaign_fingerprint"])
    if set(completed) != {
        "schema_version",
        "status",
        "family",
        "campaign_fingerprint",
        "driver_sha256",
        "selected_method",
        "artifacts",
    }:
        raise SyncError(f"{family} parallel completion top-level schema is inexact")
    driver_sha = str(completed.get("driver_sha256") or "")
    if SHA256_PATTERN.fullmatch(driver_sha) is None:
        raise SyncError(f"{family} completion has no immutable driver identity")
    artifacts = completed.get("artifacts")
    if not isinstance(artifacts, dict):
        raise SyncError(f"{family} completion has no artifact map")
    run_contract_path, _ = _nested_descriptor(
        artifacts.get("run_contract"),
        base_directory=path.parent,
        label=f"{family} run contract",
    )
    run_contract = _load_hashed_json(run_contract_path, f"{family} run contract")
    if (
        set(run_contract)
        != {
            "schema_version",
            "protocol",
            "family",
            "campaign_fingerprint",
            "driver_sha256",
            "campaign_contract",
            "screen_selection",
            "screen_mos_matrix",
            "config",
            "archive_sample",
            "methods",
            "expected_samples",
            "generation",
            "judge",
            "quality_gate",
            "max_parallel_workers",
        }
        or run_contract.get("schema_version") != 1
        or run_contract.get("protocol")
        != "dive-grt-followup-v2-family-full-parallel-v1"
        or run_contract.get("family") != family
        or run_contract.get("campaign_fingerprint") != fingerprint
        or run_contract.get("driver_sha256") != driver_sha
        or run_contract.get("expected_samples") != MULTIFAMILY_EXPECTED_SAMPLES
        or run_contract.get("generation") != {"max_new_tokens": 128, "temperature": 0}
        or run_contract.get("judge")
        != {"model": judge["model"], "revision": judge["revision"]}
        or run_contract.get("quality_gate")
        != {
            "primary_metric": "open_mos",
            "secondary_metric": "token_f1",
            "strictly_greater": True,
            "public_and_fresh_controls": True,
            "max_recompute_ratio": Decimal("0.98"),
            "max_reference_recompute_ratio": Decimal("0.98"),
        }
        or run_contract.get("max_parallel_workers") != 4
    ):
        raise SyncError(f"{family} run contract semantics mismatch")
    _validate_nested_descriptors(
        run_contract,
        base_directory=run_contract_path.parent,
        label=f"{family} run contract",
    )
    contract_path, _ = _nested_descriptor(
        run_contract.get("campaign_contract"),
        base_directory=run_contract_path.parent,
        label=f"{family} campaign contract",
    )
    methods = run_contract.get("methods")
    if not isinstance(methods, dict) or set(methods) != {
        "archived_public",
        "fresh_base",
        "all_patch_exact",
        "screen_passing_finalists",
        "full_evaluations",
    }:
        raise SyncError(f"{family} run contract method-role map is inexact")
    public = str(methods.get("archived_public") or "")
    fresh = str(methods.get("fresh_base") or "")
    exact = str(methods.get("all_patch_exact") or "")
    finalists = methods.get("screen_passing_finalists")
    evaluations = methods.get("full_evaluations")
    expected_baselines = set(MULTIFAMILY_SELECTION_CONTRACTS[family]["baselines"])
    if (
        {public, fresh, exact} != expected_baselines
        or public != MULTIFAMILY_CONTRACTS[family]["base_method"]
        or not isinstance(finalists, list)
        or not finalists
        or len(set(finalists)) != len(finalists)
        or not isinstance(evaluations, list)
        or evaluations != [fresh, exact, *finalists]
        or set(selection.get("baseline_scores", {})) != expected_baselines
        or set(selection.get("candidate_scores", {})) != set(finalists)
        or selected not in finalists
    ):
        raise SyncError(f"{family} selection/run-contract method sets disagree")
    allowed_candidates = set(
        MULTIFAMILY_SELECTION_CONTRACTS[family]["allowed_candidates"]
    )
    if not set(finalists) <= allowed_candidates:
        raise SyncError(f"{family} run contract contains an undeclared finalist")
    screen_selection_path, _ = _nested_descriptor(
        run_contract.get("screen_selection"),
        base_directory=run_contract_path.parent,
        label=f"{family} screen selection",
    )
    screen_selection = _load_hashed_json(
        screen_selection_path, f"{family} screen selection"
    )
    if (
        screen_selection.get("status") != "passed"
        or screen_selection.get("metric") != "open_mos"
        or screen_selection.get("secondary_metric") != "token_f1"
        or screen_selection.get("selected_method") != finalists[0]
        or screen_selection.get("passing_candidates") != finalists
        or set(screen_selection.get("baseline_scores", {})) != expected_baselines
        or set(screen_selection.get("baseline_secondary_scores", {}))
        != expected_baselines
        or set(screen_selection.get("candidate_scores", {})) != allowed_candidates
        or set(screen_selection.get("candidate_secondary_scores", {}))
        != allowed_candidates
    ):
        raise SyncError(f"{family} finalists do not reproduce bound screen selection")
    campaign_contract = _load_hashed_json(contract_path, f"{family} campaign contract")
    campaign_methods = campaign_contract.get("methods")
    archived = campaign_contract.get("archived_artifacts")
    run_config = _nested_descriptor(
        run_contract.get("config"),
        base_directory=run_contract_path.parent,
        label=f"{family} run config",
    )
    campaign_config = _nested_descriptor(
        campaign_contract.get("config"),
        base_directory=contract_path.parent,
        label=f"{family} campaign config",
    )
    if (
        campaign_contract.get("task") != "densevideo"
        or campaign_contract.get("expected_screen_samples") != 128
        or campaign_contract.get("generation") != "max_new_tokens=128,temperature=0"
        or not isinstance(campaign_methods, list)
        or not {fresh, exact, *allowed_candidates} <= set(campaign_methods)
        or not isinstance(archived, dict)
        or not isinstance(archived.get(family), dict)
        or archived[family].get("method") != public
        or run_config != campaign_config
    ):
        raise SyncError(f"{family} run methods are not authorized by campaign contract")
    fixed = {
        "run_contract",
        "driver_manifest",
        "input_manifest",
        "family_validation",
        "typed_identity",
        "archive_marker",
        "mos_manifest",
        "mos_matrix",
        "gate_evidence",
        "combined_selection",
    }
    if set(artifacts) != fixed | {f"method_{method}" for method in evaluations}:
        raise SyncError(f"{family} completion artifact set is incomplete or expanded")
    combined_pair = _nested_descriptor(
        artifacts["combined_selection"],
        base_directory=path.parent,
        label=f"{family} selection",
    )
    completion_matrix_pair = _nested_descriptor(
        artifacts["mos_matrix"],
        base_directory=path.parent,
        label=f"{family} MOS matrix",
    )
    if combined_pair != selection_pair or completion_matrix_pair != matrix_pair:
        raise SyncError(f"{family} completion selection/matrix binding mismatch")
    driver_manifest, _ = _nested_descriptor(
        artifacts["driver_manifest"],
        base_directory=path.parent,
        label=f"{family} driver manifest",
    )
    driver_rows = _validate_checksum_manifest(
        driver_manifest, f"{family} driver manifest"
    )
    if len(driver_rows) != 1 or driver_rows[0][1] != driver_sha:
        raise SyncError(f"{family} driver manifest does not bind its driver")
    input_manifest, _ = _nested_descriptor(
        artifacts["input_manifest"],
        base_directory=path.parent,
        label=f"{family} input manifest",
    )
    _validate_checksum_manifest(input_manifest, f"{family} input manifest")

    roles = {
        fresh: "base",
        exact: "exact",
        **{method: "candidate" for method in finalists},
    }
    marker_pairs: dict[str, dict[str, tuple[Path, str]]] = {}
    identities: dict[str, dict[str, tuple[str, str]]] = {}
    for method in evaluations:
        marker_path, _ = _nested_descriptor(
            artifacts[f"method_{method}"],
            base_directory=path.parent,
            label=f"{family} {method} marker",
        )
        marker = _load_hashed_json(marker_path, f"{family} {method} marker")
        if (
            set(marker)
            != {
                "schema_version",
                "status",
                "family",
                "method",
                "role",
                "samples",
                "campaign_fingerprint",
                "driver_sha256",
                "sample",
                "summary",
                "result",
            }
            or marker.get("schema_version") != 1
            or marker.get("status") != "complete"
            or marker.get("family") != family
            or marker.get("method") != method
            or marker.get("role") != roles[method]
            or marker.get("samples") != MULTIFAMILY_EXPECTED_SAMPLES
            or marker.get("campaign_fingerprint") != fingerprint
            or marker.get("driver_sha256") != driver_sha
        ):
            raise SyncError(f"{family} marker identity/schema mismatch for {method}")
        pairs = {
            field: _nested_descriptor(
                marker[field],
                base_directory=marker_path.parent,
                label=f"{family}.{method}.{field}",
            )
            for field in ("sample", "summary", "result")
        }
        if method == selected and pairs["summary"] != summary_pair:
            raise SyncError(f"{family} winner marker does not bind promoted summary")
        identities[method] = _validate_result_and_sample(
            family=family,
            method=method,
            sample=pairs["sample"][0],
            result=pairs["result"][0],
            expected_samples=MULTIFAMILY_EXPECTED_SAMPLES,
        )
        _validate_run_summary(
            pairs["summary"][0],
            family=family,
            method=method,
            role=roles[method],
            expected_result=pairs["result"][0],
            expected_secondary=_selection_secondary(selection, method, family),
        )
        marker_pairs[method] = pairs

    validation_path, _ = _nested_descriptor(
        artifacts["family_validation"],
        base_directory=path.parent,
        label=f"{family} validation",
    )
    validation = _load_hashed_json(validation_path, f"{family} validation")
    validation_rows = validation.get("artifacts")
    if (
        validation.get("schema_version") != 1
        or validation.get("status") != "passed"
        or validation.get("family") != family
        or validation.get("campaign_fingerprint") != fingerprint
        or validation.get("expected_samples") != MULTIFAMILY_EXPECTED_SAMPLES
        or validation.get("prediction_policy") != "diagnostic_only"
        or not isinstance(validation_rows, list)
        or {
            str(row.get("method") or "")
            for row in validation_rows
            if isinstance(row, dict)
        }
        != set(evaluations)
    ):
        raise SyncError(f"{family} family validation identity/method set mismatch")
    for row in validation_rows:
        method = str(row.get("method") or "")
        if row.get("role") != roles[method]:
            raise SyncError(f"{family} validation role mismatch for {method}")
        for field in ("sample", "summary", "result"):
            pair = _nested_descriptor(
                row.get(field),
                base_directory=validation_path.parent,
                label=f"{family} validation {field}",
            )
            if pair != marker_pairs[method][field]:
                raise SyncError(f"{family} validation evidence differs for {method}")

    typed_path, _ = _nested_descriptor(
        artifacts["typed_identity"],
        base_directory=path.parent,
        label=f"{family} typed identity",
    )
    typed = _load_hashed_json(typed_path, f"{family} typed identity")
    reports = typed.get("reports")
    if (
        typed.get("schema_version") != 1
        or typed.get("status") != "passed"
        or typed.get("campaign_fingerprint") != fingerprint
        or typed.get("base_method") != fresh
        or typed.get("expected_samples") != MULTIFAMILY_EXPECTED_SAMPLES
        or not isinstance(reports, list)
        or {str(row.get("method") or "") for row in reports if isinstance(row, dict)}
        != {public, *evaluations}
        or any(
            row.get("identical") is not True
            or row.get("left_rows") != MULTIFAMILY_EXPECTED_SAMPLES
            or row.get("right_rows") != MULTIFAMILY_EXPECTED_SAMPLES
            for row in reports
        )
    ):
        raise SyncError(f"{family} typed-identity evidence mismatch")
    archive_path, _ = _nested_descriptor(
        artifacts["archive_marker"],
        base_directory=path.parent,
        label=f"{family} archive marker",
    )
    archive = _load_hashed_json(archive_path, f"{family} archive marker")
    if (
        archive.get("schema_version") != 1
        or archive.get("status") != "complete"
        or archive.get("family") != family
        or archive.get("method") != public
        or archive.get("campaign_fingerprint") != fingerprint
        or archive.get("driver_sha256") != driver_sha
    ):
        raise SyncError(f"{family} archive marker identity mismatch")
    archive_pairs = {
        field: _nested_descriptor(
            archive.get(field),
            base_directory=archive_path.parent,
            label=f"{family} archive {field}",
        )
        for field in (
            "archive_source",
            "fresh_base",
            "staged_sample",
            "summary",
            "identity_report",
        )
    }
    if archive_pairs["fresh_base"] != marker_pairs[fresh]["sample"]:
        raise SyncError(f"{family} archive marker is not bound to fresh base")
    identity_report = _load_hashed_json(
        archive_pairs["identity_report"][0], f"{family} archive identity"
    )
    if (
        identity_report.get("status") != "identical"
        or identity_report.get("family") != family
        or identity_report.get("method") != public
        or identity_report.get("samples") != MULTIFAMILY_EXPECTED_SAMPLES
        or identity_report.get("campaign_fingerprint") != fingerprint
    ):
        raise SyncError(f"{family} archive identity report mismatch")
    public_identity = _sample_identity(
        archive_pairs["staged_sample"][0],
        label=f"{family}.{public}.archive_sample",
        expected_samples=MULTIFAMILY_EXPECTED_SAMPLES,
    )
    archive_summary = _summary_row(
        archive_pairs["summary"][0],
        family=family,
        method=public,
        expected_samples=MULTIFAMILY_EXPECTED_SAMPLES,
        expected_result=archive_pairs["staged_sample"][0],
    )
    _assert_secondary_score(
        archive_summary,
        _selection_secondary(selection, public, family),
        f"{family}.{public}",
    )
    if any(
        identity != identities[fresh]
        for identity in [public_identity, *identities.values()]
    ):
        raise SyncError(f"{family} samples do not share typed identity")

    mos_manifest_path, _ = _nested_descriptor(
        artifacts["mos_manifest"],
        base_directory=path.parent,
        label=f"{family} MOS manifest",
    )
    mos_manifest = _load_hashed_json(mos_manifest_path, f"{family} MOS manifest")
    mos_rows = mos_manifest.get("samples")
    if (
        mos_manifest.get("schema_version") != 1
        or mos_manifest.get("family") != family
        or mos_manifest.get("campaign_fingerprint") != fingerprint
        or mos_manifest.get("expected_samples") != MULTIFAMILY_EXPECTED_SAMPLES
        or not isinstance(mos_rows, list)
        or {str(row.get("method") or "") for row in mos_rows if isinstance(row, dict)}
        != {public, *evaluations}
    ):
        raise SyncError(f"{family} MOS manifest method set mismatch")
    for row in mos_rows:
        source = _artifact_path(
            row.get("source"), mos_manifest_path.parent, f"{family} MOS source"
        )
        if (
            _integer(row.get("rows"), f"{family}.mos.rows")
            != MULTIFAMILY_EXPECTED_SAMPLES
            or SHA256_PATTERN.fullmatch(str(row.get("sha256") or "")) is None
            or sha256_file(source) != row.get("sha256")
            or not str(row.get("staged_relative_path") or "").startswith("runs/")
        ):
            raise SyncError(f"{family} MOS source entry mismatch")
    gate_path, _ = _nested_descriptor(
        artifacts["gate_evidence"],
        base_directory=path.parent,
        label=f"{family} gate evidence",
    )
    gate = _load_hashed_json(gate_path, f"{family} gate evidence")
    gates = gate.get("gates")
    if (
        gate.get("schema_version") != 1
        or gate.get("status") != "passed"
        or gate.get("family") != family
        or gate.get("campaign_fingerprint") != fingerprint
        or gate.get("driver_sha256") != driver_sha
        or gate.get("primary_metric") != "open_mos"
        or gate.get("secondary_metric") != "token_f1"
        or gate.get("max_recompute_ratio") != Decimal("0.98")
        or gate.get("max_reference_recompute_ratio") != Decimal("0.98")
        or gate.get("return_codes") != {"public": 0, "fresh": 0, "combined": 0}
        or not isinstance(gates, dict)
        or set(gates) != {"public", "fresh", "combined"}
        or any(
            not isinstance(row, dict)
            or row.get("status") != "passed"
            or row.get("selected_method") != selected
            for row in gates.values()
        )
        or gate.get("selected_method") != selected
    ):
        raise SyncError(f"{family} three-way gate evidence mismatch")
    if (
        _nested_descriptor(
            gates["combined"],
            base_directory=gate_path.parent,
            label=f"{family} combined gate",
        )
        != selection_pair
    ):
        raise SyncError(f"{family} gate evidence does not bind selection")
    return contract_path


def _validate_llava7_completion(
    completed: Mapping[str, Any],
    *,
    path: Path,
    entry: Mapping[str, Any],
    selection: Mapping[str, Any],
    selection_pair: tuple[Path, str],
    matrix_pair: tuple[Path, str],
    summary_pair: tuple[Path, str],
) -> Path:
    fingerprint = str(entry["campaign_fingerprint"])
    selected = str(entry["selected_method"])
    roles = {
        "llava_onevision_7b_dual_quality_base": "base",
        "llava_onevision_7b_dual_grt_all": "exact",
        "grt_llava_onevision_7b_hf_dual_s002_o005": "candidate",
    }
    if set(completed) != {
        "schema_version",
        "status",
        "stage",
        "campaign_fingerprint",
        "expected_samples",
        "output_root",
        "contract",
        "source_snapshot",
        "methods",
        "validation",
        "archive",
        "mos",
        "selection",
    }:
        raise SyncError("llava7 completion top-level schema is inexact")
    methods = completed.get("methods")
    if not isinstance(methods, dict) or set(methods) != set(roles):
        raise SyncError(
            "llava7 completion marker set is not exactly base/exact/candidate"
        )
    if (
        _nested_descriptor(
            completed.get("selection"),
            base_directory=path.parent,
            label="llava7 selection",
        )
        != selection_pair
    ):
        raise SyncError("llava7 completion selection binding mismatch")
    mos = completed.get("mos")
    if (
        not isinstance(mos, dict)
        or _nested_descriptor(
            mos.get("matrix_csv"), base_directory=path.parent, label="llava7 MOS matrix"
        )
        != matrix_pair
    ):
        raise SyncError("llava7 completion MOS matrix binding mismatch")
    all_mos_methods = {*roles, "llava_onevision_original"}
    if (
        mos.get("rows") != MULTIFAMILY_EXPECTED_SAMPLES * len(all_mos_methods)
        or mos.get("errors") != 0
        or mos.get("method_rows")
        != {method: MULTIFAMILY_EXPECTED_SAMPLES for method in all_mos_methods}
    ):
        raise SyncError("llava7 MOS row/method evidence mismatch")
    marker_pairs: dict[str, dict[str, tuple[Path, str]]] = {}
    identities: dict[str, dict[str, tuple[str, str]]] = {}
    for method, role in roles.items():
        evidence = methods[method]
        if not isinstance(evidence, dict) or set(evidence) != {
            "marker",
            "sample",
            "summary",
            "result",
            "run_log",
        }:
            raise SyncError(f"llava7 completion evidence schema mismatch for {method}")
        marker_path, _ = _nested_descriptor(
            evidence["marker"],
            base_directory=path.parent,
            label=f"llava7 {method} marker",
        )
        marker = _load_hashed_json(marker_path, f"llava7 {method} marker")
        if (
            marker.get("schema_version") != 1
            or marker.get("status") != "complete"
            or marker.get("method") != method
            or marker.get("samples") != MULTIFAMILY_EXPECTED_SAMPLES
            or marker.get("campaign_fingerprint") != fingerprint
        ):
            raise SyncError(f"llava7 marker identity mismatch for {method}")
        pairs = {
            field: _nested_descriptor(
                marker.get(field),
                base_directory=marker_path.parent,
                label=f"llava7 {method}.{field}",
            )
            for field in ("sample", "summary", "result", "run_log")
        }
        for field, pair in pairs.items():
            if (
                _nested_descriptor(
                    evidence[field],
                    base_directory=path.parent,
                    label=f"llava7 completion {method}.{field}",
                )
                != pair
            ):
                raise SyncError(
                    f"llava7 marker/completion binding mismatch for {method}"
                )
        if method == selected and pairs["summary"] != summary_pair:
            raise SyncError("llava7 winner marker does not bind promoted summary")
        identities[method] = _validate_result_and_sample(
            family="llava7",
            method=method,
            sample=pairs["sample"][0],
            result=pairs["result"][0],
            expected_samples=MULTIFAMILY_EXPECTED_SAMPLES,
        )
        _validate_run_summary(
            pairs["summary"][0],
            family="llava7",
            method=method,
            role=role,
            expected_result=pairs["result"][0],
            expected_secondary=_selection_secondary(selection, method, "llava7"),
        )
        marker_pairs[method] = pairs
    base_identity = identities["llava_onevision_7b_dual_quality_base"]
    if any(identity != base_identity for identity in identities.values()):
        raise SyncError("llava7 method samples do not share typed identity")
    validation_path, _ = _nested_descriptor(
        completed.get("validation"),
        base_directory=path.parent,
        label="llava7 validation",
    )
    validation = _load_hashed_json(validation_path, "llava7 validation")
    validation_artifacts = validation.get("artifacts")
    if (
        validation.get("schema_version") != 1
        or validation.get("status") != "passed"
        or validation.get("stage") != "full"
        or validation.get("campaign_fingerprint") != fingerprint
        or validation.get("expected_samples") != MULTIFAMILY_EXPECTED_SAMPLES
        or validation.get("reproduction") is not None
        or validation.get("quality_gate_inputs")
        != {
            "controls": [
                "llava_onevision_7b_dual_quality_base",
                "llava_onevision_7b_dual_grt_all",
            ],
            "candidate": selected,
            "reproduction_oracles_included": False,
        }
        or not isinstance(validation_artifacts, dict)
        or set(validation_artifacts) != set(roles)
    ):
        raise SyncError("llava7 validation identity/method set mismatch")
    for method, artifact_map in validation_artifacts.items():
        for field in ("sample", "summary", "result", "run_log"):
            if (
                _nested_descriptor(
                    artifact_map.get(field),
                    base_directory=validation_path.parent,
                    label=f"llava7 validation {method}.{field}",
                )
                != marker_pairs[method][field]
            ):
                raise SyncError(f"llava7 validation evidence differs for {method}")
    archive = completed.get("archive")
    if not isinstance(archive, dict) or set(archive) != {
        "report",
        "subset_sample",
        "summary",
    }:
        raise SyncError("llava7 archive evidence schema mismatch")
    archive_report_path, _ = _nested_descriptor(
        archive["report"], base_directory=path.parent, label="llava7 archive report"
    )
    archive_sample_path, _ = _nested_descriptor(
        archive["subset_sample"],
        base_directory=path.parent,
        label="llava7 archive sample",
    )
    archive_summary_path, _ = _nested_descriptor(
        archive["summary"], base_directory=path.parent, label="llava7 archive summary"
    )
    archive_report = _load_hashed_json(archive_report_path, "llava7 archive report")
    if (
        archive_report.get("schema_version") != 1
        or archive_report.get("status") != "identical"
        or archive_report.get("family") != "llava7"
        or archive_report.get("method") != "llava_onevision_original"
        or archive_report.get("samples") != MULTIFAMILY_EXPECTED_SAMPLES
        or archive_report.get("campaign_fingerprint") != fingerprint
    ):
        raise SyncError("llava7 archive report identity mismatch")
    if (
        _sample_identity(
            archive_sample_path,
            label="llava7 archived sample",
            expected_samples=MULTIFAMILY_EXPECTED_SAMPLES,
        )
        != base_identity
    ):
        raise SyncError("llava7 archived/public typed identity mismatch")
    archive_summary = _summary_row(
        archive_summary_path,
        family="llava7",
        method="llava_onevision_original",
        expected_samples=MULTIFAMILY_EXPECTED_SAMPLES,
        expected_result=archive_sample_path,
    )
    _assert_secondary_score(
        archive_summary,
        _selection_secondary(selection, "llava_onevision_original", "llava7"),
        "llava7.llava_onevision_original",
    )
    contract_path, _ = _nested_descriptor(
        completed.get("contract"),
        base_directory=path.parent,
        label="llava7 campaign contract",
    )
    return contract_path


def _validate_recovery_gate(
    payload: Mapping[str, Any],
    *,
    label: str,
    baselines: set[str],
    selected: str,
) -> None:
    baseline_scores = payload.get("baseline_scores")
    baseline_secondary = payload.get("baseline_secondary_scores")
    candidate_scores = payload.get("candidate_scores")
    candidate_secondary = payload.get("candidate_secondary_scores")
    counts = payload.get("sample_counts")
    passing = payload.get("passing_candidates")
    if (
        payload.get("status") != "passed"
        or payload.get("metric") != "open_mos"
        or payload.get("secondary_metric") != "token_f1"
        or payload.get("selected_method") != selected
        or not isinstance(baseline_scores, dict)
        or set(baseline_scores) != baselines
        or not isinstance(baseline_secondary, dict)
        or set(baseline_secondary) != baselines
        or not isinstance(candidate_scores, dict)
        or set(candidate_scores) != set(QWEN3_RECOVERY_FINALISTS)
        or not isinstance(candidate_secondary, dict)
        or set(candidate_secondary) != set(QWEN3_RECOVERY_FINALISTS)
        or not isinstance(passing, list)
        or not passing
        or passing[0] != selected
        or len(set(map(str, passing))) != len(passing)
        or not set(map(str, passing)) <= set(QWEN3_RECOVERY_FINALISTS)
        or not isinstance(counts, dict)
        or set(counts) != { *baselines, *QWEN3_RECOVERY_FINALISTS }
        or any(
            _integer(value, f"qwen3.{label}.sample_count")
            != MULTIFAMILY_EXPECTED_SAMPLES
            for value in counts.values()
        )
    ):
        raise SyncError(f"qwen3 {label} recovery gate schema/method set differs")
    selected_score = _decimal(
        payload.get("selected_score"), f"qwen3.{label}.selected_score"
    )
    selected_secondary = _decimal(
        payload.get("selected_secondary_score"),
        f"qwen3.{label}.selected_secondary",
    )
    required = _decimal(payload.get("required_score"), f"qwen3.{label}.required")
    secondary_required = _decimal(
        payload.get("secondary_required_score"),
        f"qwen3.{label}.secondary_required",
    )
    ratio = _decimal(
        payload.get("selected_recompute_ratio"), f"qwen3.{label}.ratio"
    )
    reference = _decimal(
        payload.get(
            "selected_reference_patch_compute_ratio",
            payload.get("selected_reference_recompute_ratio"),
        ),
        f"qwen3.{label}.reference_ratio",
    )
    if None in {selected_score, selected_secondary, required, secondary_required, ratio, reference}:
        raise SyncError(f"qwen3 {label} gate contains missing numeric values")
    assert (
        selected_score is not None
        and selected_secondary is not None
        and required is not None
        and secondary_required is not None
        and ratio is not None
        and reference is not None
    )
    candidate_score = _decimal(
        candidate_scores[selected], f"qwen3.{label}.candidate_score"
    )
    candidate_f1 = _decimal(
        candidate_secondary[selected], f"qwen3.{label}.candidate_f1"
    )
    baseline_values = [
        _decimal(value, f"qwen3.{label}.baseline_score")
        for value in baseline_scores.values()
    ]
    secondary_values = [
        _decimal(value, f"qwen3.{label}.baseline_f1")
        for value in baseline_secondary.values()
    ]
    if (
        candidate_score != selected_score
        or candidate_f1 != selected_secondary
        or required != max(value for value in baseline_values if value is not None)
        or secondary_required
        != max(value for value in secondary_values if value is not None)
        or selected_score <= required
        or selected_secondary <= secondary_required
        or _decimal(payload.get("max_recompute_ratio"), f"qwen3.{label}.max_ratio")
        != Decimal("0.98")
        or _decimal(
            payload.get("max_reference_recompute_ratio"),
            f"qwen3.{label}.max_reference",
        )
        != Decimal("0.98")
        or not Decimal(0) <= ratio <= Decimal("0.98")
        or not Decimal(0) <= reference <= Decimal("0.98")
    ):
        raise SyncError(f"qwen3 {label} gate strict inequalities differ")


def _validate_recovery_archive(
    marker_path: Path,
    *,
    recovery_root: Path,
    recovery_fingerprint: str,
    recovery_driver_sha: str,
    archive_source_pair: tuple[Path, str],
    expected_fresh_pair: tuple[Path, str] | None,
    samples: int,
    kind: str,
    label: str,
) -> tuple[tuple[Path, str], tuple[Path, str]]:
    marker = _load_hashed_json(marker_path, label)
    expected_fields = {
        "schema_version",
        "status",
        "provenance_kind",
        "family",
        "method",
        "samples",
        "recovery_fingerprint",
        "producer_driver_sha256",
        "archive_source",
        "fresh_sample",
        "staged_sample",
        "summary",
        "identity_report",
    }
    if (
        set(marker) != expected_fields
        or marker.get("schema_version") != 1
        or marker.get("status") != "complete"
        or marker.get("provenance_kind") != kind
        or marker.get("family") != "qwen3"
        or marker.get("method") != QWEN3_RECOVERY_PUBLIC
        or _integer(marker.get("samples"), f"{label}.samples") != samples
        or marker.get("recovery_fingerprint") != recovery_fingerprint
        or marker.get("producer_driver_sha256") != recovery_driver_sha
    ):
        raise SyncError(f"{label} identity/schema differs")
    pairs = {
        field: _recovery_descriptor(
            marker[field],
            base_directory=marker_path.parent,
            label=f"{label}.{field}",
        )
        for field in (
            "archive_source",
            "fresh_sample",
            "staged_sample",
            "summary",
            "identity_report",
        )
    }
    if pairs["archive_source"] != archive_source_pair or (
        expected_fresh_pair is not None
        and pairs["fresh_sample"] != expected_fresh_pair
    ):
        raise SyncError(f"{label} archive/fresh source binding differs")
    for field in ("staged_sample", "summary", "identity_report"):
        try:
            pairs[field][0].relative_to(recovery_root)
        except ValueError as exc:
            raise SyncError(f"{label}.{field} leaves recovery root") from exc
    fresh_identity = _sample_identity(
        pairs["fresh_sample"][0],
        label=f"{label}.fresh",
        expected_samples=samples,
    )
    staged_identity = _sample_identity(
        pairs["staged_sample"][0],
        label=f"{label}.staged",
        expected_samples=samples,
    )
    if fresh_identity != staged_identity:
        raise SyncError(f"{label} fresh/staged typed identity differs")
    report = _load_hashed_json(pairs["identity_report"][0], f"{label} report")
    if (
        report.get("schema_version") != 1
        or report.get("status") != "identical"
        or report.get("family") != "qwen3"
        or report.get("method") != QWEN3_RECOVERY_PUBLIC
        or report.get("campaign_fingerprint") != recovery_fingerprint
        or _integer(report.get("samples"), f"{label}.report.samples") != samples
        or Path(str(report.get("archive_sample") or "")).expanduser().resolve()
        != archive_source_pair[0]
        or report.get("archive_sample_sha256") != archive_source_pair[1]
        or Path(str(report.get("fresh_sample") or "")).expanduser().resolve()
        != pairs["fresh_sample"][0]
        or report.get("fresh_sample_sha256") != pairs["fresh_sample"][1]
        or Path(str(report.get("subset_sample") or "")).expanduser().resolve()
        != pairs["staged_sample"][0]
        or report.get("subset_sample_sha256") != pairs["staged_sample"][1]
        or Path(str(report.get("summary_csv") or "")).expanduser().resolve()
        != pairs["summary"][0]
        or report.get("summary_sha256") != pairs["summary"][1]
    ):
        raise SyncError(f"{label} identity report binding differs")
    archive_summary = _summary_row(
        pairs["summary"][0],
        family="qwen3",
        method=QWEN3_RECOVERY_PUBLIC,
        expected_samples=samples,
        expected_result=pairs["staged_sample"][0],
    )
    _assert_secondary_score(
        archive_summary,
        report.get("token_f1"),
        f"{label}.summary",
    )
    return pairs["staged_sample"], pairs["summary"]


def _validate_qwen3_recovery_completion(
    completed: Mapping[str, Any],
    *,
    path: Path,
    entry: Mapping[str, Any],
    selection: Mapping[str, Any],
    selection_pair: tuple[Path, str],
    matrix_pair: tuple[Path, str],
    summary_pair: tuple[Path, str],
    judge: Mapping[str, Any],
) -> Path:
    expected_top = {
        "schema_version",
        "status",
        "protocol",
        "family",
        "campaign_fingerprint",
        "recovery_fingerprint",
        "recovery_driver_sha256",
        "expected_samples",
        "selected_method",
        "gate_selected_methods",
        "method_order",
        "method_producer_drivers",
        "method_provenance_kinds",
        "artifacts",
    }
    fingerprint = str(entry["campaign_fingerprint"])
    recovery_fingerprint = str(completed.get("recovery_fingerprint") or "")
    recovery_driver_sha = str(completed.get("recovery_driver_sha256") or "")
    selected = str(entry["selected_method"])
    if (
        set(completed) != expected_top
        or completed.get("protocol") != QWEN3_RECOVERY_PROTOCOL
        or completed.get("family") != "qwen3"
        or completed.get("campaign_fingerprint") != fingerprint
        or SHA256_PATTERN.fullmatch(recovery_fingerprint) is None
        or SHA256_PATTERN.fullmatch(recovery_driver_sha) is None
        or _integer(completed.get("expected_samples"), "qwen3.recovery.samples")
        != MULTIFAMILY_EXPECTED_SAMPLES
        or completed.get("selected_method") != selected
        or completed.get("method_order") != list(QWEN3_RECOVERY_METHODS)
    ):
        raise SyncError("qwen3 recovery completion top-level schema differs")
    artifacts = completed.get("artifacts")
    fixed = {
        "contract",
        "source_manifest",
        "screen_archive",
        "family_validation",
        "typed_identity",
        "archive_marker",
        "mos_manifest",
        "mos_matrix_csv",
        "mos_matrix_jsonl",
        "public_selection",
        "fresh_selection",
        "combined_selection",
    }
    if not isinstance(artifacts, dict) or set(artifacts) != fixed | {
        f"method_{method}" for method in QWEN3_RECOVERY_METHODS
    }:
        raise SyncError("qwen3 recovery completion artifact inventory differs")
    contract_pair = _recovery_descriptor(
        artifacts["contract"],
        base_directory=path.parent,
        label="qwen3 recovery contract",
    )
    contract = _validate_qwen3_recovery_contract(
        contract_pair[0],
        contract_sha=contract_pair[1],
        recovery_fingerprint=recovery_fingerprint,
        campaign_fingerprint=fingerprint,
        recovery_driver_sha=recovery_driver_sha,
    )
    recovery_root = Path(str(contract["recovery_root"])).resolve()
    if path != recovery_root / "provenance/completed.json":
        raise SyncError("qwen3 recovery completion leaves its canonical namespace")
    source_manifest_pair = _recovery_descriptor(
        artifacts["source_manifest"],
        base_directory=path.parent,
        label="qwen3 recovery source manifest",
    )
    contracted_source_pair = _recovery_descriptor(
        contract["recovery_sources"]["manifest"],
        base_directory=contract_pair[0].parent,
        label="qwen3 contracted recovery source manifest",
    )
    if source_manifest_pair != contracted_source_pair:
        raise SyncError("qwen3 completion/contract source manifest differs")

    expected_roles = {
        QWEN3_RECOVERY_REUSED[0]: "base",
        QWEN3_RECOVERY_REUSED[1]: "exact",
        **{method: "candidate" for method in QWEN3_RECOVERY_FINALISTS},
    }
    expected_producers = {
        **{
            method: QWEN3_RECOVERY_ORIGINAL_DRIVER_SHA256
            for method in QWEN3_RECOVERY_REUSED
        },
        **{method: recovery_driver_sha for method in QWEN3_RECOVERY_NATIVE},
    }
    expected_kinds = {
        **{
            method: "adopted_original_full_parallel_v2"
            for method in QWEN3_RECOVERY_REUSED
        },
        **{method: "native_qwen3_recovery_v2" for method in QWEN3_RECOVERY_NATIVE},
    }
    if (
        completed.get("method_producer_drivers") != expected_producers
        or completed.get("method_provenance_kinds") != expected_kinds
    ):
        raise SyncError("qwen3 mixed adopted/native producer maps differ")
    marker_payloads: dict[str, Mapping[str, Any]] = {}
    artifact_pairs: dict[str, dict[str, tuple[Path, str]]] = {}
    identities: dict[str, dict[str, tuple[str, str]]] = {}
    legacy_reused = contract["legacy"]["reused_methods"]
    for method in QWEN3_RECOVERY_METHODS:
        marker_pair = _recovery_descriptor(
            artifacts[f"method_{method}"],
            base_directory=path.parent,
            label=f"qwen3 recovery marker {method}",
        )
        adopted = method in QWEN3_RECOVERY_REUSED
        marker_root = "adopted_methods" if adopted else "native_methods"
        if marker_pair[0] != recovery_root / "provenance" / marker_root / f"{method}.json":
            raise SyncError(f"qwen3 recovery marker path differs for {method}")
        marker = _load_hashed_json(marker_pair[0], f"qwen3 recovery marker {method}")
        common = {
            "schema_version": 1,
            "status": "complete",
            "family": "qwen3",
            "method": method,
            "role": expected_roles[method],
            "samples": MULTIFAMILY_EXPECTED_SAMPLES,
            "campaign_fingerprint": fingerprint,
            "recovery_fingerprint": recovery_fingerprint,
            "producer_driver_sha256": expected_producers[method],
        }
        if any(marker.get(key) != value for key, value in common.items()):
            raise SyncError(f"qwen3 recovery marker identity differs for {method}")
        expected_marker_fields = {
            *common,
            "provenance_kind",
            "recovery_contract",
            "sample",
            "summary",
            "result",
            "driver_log",
        }
        expected_marker_fields.add(
            "legacy_marker" if adopted else "attempt_id"
        )
        if adopted:
            expected_marker_fields.add("adopted_by_driver_sha256")
        if (
            set(marker) != expected_marker_fields
            or marker.get("provenance_kind") != expected_kinds[method]
            or _recovery_descriptor(
                marker.get("recovery_contract"),
                base_directory=marker_pair[0].parent,
                label=f"qwen3 {method} recovery contract",
            )
            != contract_pair
        ):
            raise SyncError(f"qwen3 recovery marker schema differs for {method}")
        pairs = {
            field: _recovery_descriptor(
                marker[field],
                base_directory=marker_pair[0].parent,
                label=f"qwen3 {method}.{field}",
            )
            for field in ("sample", "summary", "result", "driver_log")
        }
        if adopted:
            legacy = legacy_reused[method]
            legacy_marker_pair = _recovery_descriptor(
                marker.get("legacy_marker"),
                base_directory=marker_pair[0].parent,
                label=f"qwen3 adopted legacy marker {method}",
            )
            expected_legacy_marker = _recovery_descriptor(
                legacy["marker"],
                base_directory=contract_pair[0].parent,
                label=f"qwen3 contract legacy marker {method}",
            )
            expected_artifacts = {
                field: _recovery_descriptor(
                    legacy["artifacts"][field],
                    base_directory=contract_pair[0].parent,
                    label=f"qwen3 contract legacy {method}.{field}",
                )
                for field in ("sample", "summary", "result", "driver_log")
            }
            if (
                marker.get("adopted_by_driver_sha256") != recovery_driver_sha
                or legacy_marker_pair != expected_legacy_marker
                or pairs != expected_artifacts
            ):
                raise SyncError(f"qwen3 adopted/legacy SHA chain differs for {method}")
        else:
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(marker.get("attempt_id") or "")):
                raise SyncError(f"qwen3 native attempt identity differs for {method}")
            for pair in pairs.values():
                try:
                    pair[0].relative_to(recovery_root)
                except ValueError as exc:
                    raise SyncError(
                        f"qwen3 native artifact leaves recovery root for {method}"
                    ) from exc
        identities[method] = _validate_result_and_sample(
            family="qwen3",
            method=method,
            sample=pairs["sample"][0],
            result=pairs["result"][0],
            expected_samples=MULTIFAMILY_EXPECTED_SAMPLES,
        )
        summary = _validate_run_summary(
            pairs["summary"][0],
            family="qwen3",
            method=method,
            role=expected_roles[method],
            expected_result=pairs["result"][0],
            expected_secondary=_selection_secondary(selection, method, "qwen3"),
        )
        _validate_recovery_raw_telemetry(
            pairs["driver_log"][0], method=method, summary=summary
        )
        if method == selected and pairs["summary"] != summary_pair:
            raise SyncError("qwen3 recovery winner summary descriptor differs")
        marker_payloads[method] = marker
        artifact_pairs[method] = pairs

    family_validation_pair = _recovery_descriptor(
        artifacts["family_validation"],
        base_directory=path.parent,
        label="qwen3 recovery family validation",
    )
    validation = _load_hashed_json(
        family_validation_pair[0], "qwen3 recovery family validation"
    )
    validation_rows = validation.get("artifacts")
    if (
        validation.get("schema_version") != 1
        or validation.get("status") != "passed"
        or validation.get("family") != "qwen3"
        or validation.get("campaign_fingerprint") != fingerprint
        or validation.get("expected_samples") != MULTIFAMILY_EXPECTED_SAMPLES
        or validation.get("prediction_policy") != "diagnostic_only"
        or not isinstance(validation_rows, list)
        or [str(row.get("method") or "") for row in validation_rows]
        != list(QWEN3_RECOVERY_METHODS)
    ):
        raise SyncError("qwen3 recovery family validation differs")
    for row in validation_rows:
        method = str(row["method"])
        if row.get("role") != expected_roles[method]:
            raise SyncError(f"qwen3 recovery validation role differs for {method}")
        for field in ("sample", "summary", "result"):
            if _nested_descriptor(
                row.get(field),
                base_directory=family_validation_pair[0].parent,
                label=f"qwen3 validation {method}.{field}",
            ) != artifact_pairs[method][field]:
                raise SyncError(f"qwen3 validation/marker differs for {method}.{field}")

    legacy_run_pair = _recovery_descriptor(
        contract["legacy"]["run_contract"],
        base_directory=contract_pair[0].parent,
        label="qwen3 legacy run contract",
    )
    legacy_run = _load_hashed_json(legacy_run_pair[0], "qwen3 legacy run contract")
    archive_source_pair = _recovery_descriptor(
        legacy_run.get("archive_sample"),
        base_directory=legacy_run_pair[0].parent,
        label="qwen3 archived public source",
    )
    screen_archive_pair = _recovery_descriptor(
        artifacts["screen_archive"],
        base_directory=path.parent,
        label="qwen3 recovery screen archive",
    )
    _validate_recovery_archive(
        screen_archive_pair[0],
        recovery_root=recovery_root,
        recovery_fingerprint=recovery_fingerprint,
        recovery_driver_sha=recovery_driver_sha,
        archive_source_pair=archive_source_pair,
        expected_fresh_pair=None,
        samples=128,
        kind="stable_screen_archive_v2",
        label="qwen3 screen archive",
    )
    archive_pair = _recovery_descriptor(
        artifacts["archive_marker"],
        base_directory=path.parent,
        label="qwen3 recovery full archive",
    )
    public_sample_pair, public_summary_pair = _validate_recovery_archive(
        archive_pair[0],
        recovery_root=recovery_root,
        recovery_fingerprint=recovery_fingerprint,
        recovery_driver_sha=recovery_driver_sha,
        archive_source_pair=archive_source_pair,
        expected_fresh_pair=artifact_pairs[QWEN3_RECOVERY_REUSED[0]]["sample"],
        samples=MULTIFAMILY_EXPECTED_SAMPLES,
        kind="stable_full_archive_v2",
        label="qwen3 full archive",
    )
    public_identity = _sample_identity(
        public_sample_pair[0],
        label="qwen3 archived public sample",
        expected_samples=MULTIFAMILY_EXPECTED_SAMPLES,
    )
    if any(identity != identities[QWEN3_RECOVERY_REUSED[0]] for identity in [public_identity, *identities.values()]):
        raise SyncError("qwen3 recovery seven-method typed identity differs")
    public_summary = _summary_row(
        public_summary_pair[0],
        family="qwen3",
        method=QWEN3_RECOVERY_PUBLIC,
        expected_samples=MULTIFAMILY_EXPECTED_SAMPLES,
        expected_result=public_sample_pair[0],
    )
    _assert_secondary_score(
        public_summary,
        _selection_secondary(selection, QWEN3_RECOVERY_PUBLIC, "qwen3"),
        "qwen3.archived_public",
    )
    typed_pair = _recovery_descriptor(
        artifacts["typed_identity"],
        base_directory=path.parent,
        label="qwen3 recovery typed identity",
    )
    typed = _load_hashed_json(typed_pair[0], "qwen3 recovery typed identity")
    reports = typed.get("reports")
    if (
        typed.get("schema_version") != 1
        or typed.get("status") != "passed"
        or typed.get("recovery_fingerprint") != recovery_fingerprint
        or typed.get("base_method") != QWEN3_RECOVERY_REUSED[0]
        or typed.get("expected_samples") != MULTIFAMILY_EXPECTED_SAMPLES
        or not isinstance(reports, list)
        or [str(row.get("method") or "") for row in reports]
        != [QWEN3_RECOVERY_PUBLIC, *QWEN3_RECOVERY_METHODS]
        or any(
            row.get("identical") is not True
            or row.get("left_rows") != MULTIFAMILY_EXPECTED_SAMPLES
            or row.get("right_rows") != MULTIFAMILY_EXPECTED_SAMPLES
            for row in reports
        )
    ):
        raise SyncError("qwen3 recovery typed identity report differs")

    mos_manifest_pair = _recovery_descriptor(
        artifacts["mos_manifest"],
        base_directory=path.parent,
        label="qwen3 recovery MOS manifest",
    )
    mos_manifest = _load_hashed_json(
        mos_manifest_pair[0], "qwen3 recovery MOS manifest"
    )
    mos_rows = mos_manifest.get("samples")
    mos_methods = [QWEN3_RECOVERY_PUBLIC, *QWEN3_RECOVERY_METHODS]
    expected_samples = {
        QWEN3_RECOVERY_PUBLIC: public_sample_pair,
        **{method: artifact_pairs[method]["sample"] for method in QWEN3_RECOVERY_METHODS},
    }
    if (
        set(mos_manifest) != {
            "schema_version",
            "family",
            "recovery_fingerprint",
            "expected_samples",
            "samples",
        }
        or mos_manifest.get("schema_version") != 1
        or mos_manifest.get("family") != "qwen3"
        or mos_manifest.get("recovery_fingerprint") != recovery_fingerprint
        or mos_manifest.get("expected_samples") != MULTIFAMILY_EXPECTED_SAMPLES
        or not isinstance(mos_rows, list)
        or [str(row.get("method") or "") for row in mos_rows] != mos_methods
    ):
        raise SyncError("qwen3 recovery MOS manifest differs")
    mos_root = recovery_root / "mos_inputs"
    if not mos_root.is_dir() or mos_root.is_symlink():
        raise SyncError("qwen3 recovery MOS staging root is not an isolated directory")
    expected_inventory = set()
    for row in mos_rows:
        method = str(row["method"])
        source = _artifact_path(
            row.get("source"), mos_manifest_pair[0].parent, f"qwen3 MOS {method}"
        )
        relative = str(row.get("staged_relative_path") or "")
        if (
            (source, str(row.get("sha256") or "")) != expected_samples[method]
            or row.get("rows") != MULTIFAMILY_EXPECTED_SAMPLES
            or relative
            != f"runs/{method}/densevideo/authorized/authorized_samples_densevideo.jsonl"
        ):
            raise SyncError(f"qwen3 recovery MOS source differs for {method}")
        staged = mos_root / relative
        if (
            not staged.is_symlink()
            or staged.resolve() != source
            or sha256_file(staged) != row["sha256"]
        ):
            raise SyncError(f"qwen3 recovery MOS staging differs for {method}")
        expected_inventory.add(relative)
    actual_inventory = {
        str(item.relative_to(mos_root))
        for item in mos_root.rglob("*")
        if item.is_file() or item.is_symlink()
    }
    if actual_inventory != expected_inventory:
        raise SyncError("qwen3 recovery MOS staging inventory differs")
    completion_matrix_pair = _recovery_descriptor(
        artifacts["mos_matrix_csv"],
        base_directory=path.parent,
        label="qwen3 recovery MOS CSV",
    )
    matrix_jsonl_pair = _recovery_descriptor(
        artifacts["mos_matrix_jsonl"],
        base_directory=path.parent,
        label="qwen3 recovery MOS JSONL",
    )
    if (
        completion_matrix_pair != matrix_pair
        or completion_matrix_pair[0] != recovery_root / "mos/matrix.csv"
        or matrix_jsonl_pair[0] != recovery_root / "mos/matrix.jsonl"
    ):
        raise SyncError("qwen3 recovery MOS matrix path/binding differs")
    csv_rows, _ = _read_csv(completion_matrix_pair[0])
    try:
        jsonl_rows = [
            _strict_json_loads(line, "qwen3 recovery MOS JSONL")
            for line in matrix_jsonl_pair[0].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except OSError as exc:
        raise SyncError(f"qwen3 recovery MOS JSONL is invalid: {exc}") from exc

    def mos_key(row: Mapping[str, Any]) -> tuple[str, str]:
        return str(row.get("method") or ""), str(row.get("sample_id") or "")

    csv_map = {mos_key(row): row for row in csv_rows}
    jsonl_map = {mos_key(row): row for row in jsonl_rows}
    if (
        len(csv_map) != len(csv_rows)
        or len(jsonl_map) != len(jsonl_rows)
        or set(csv_map) != set(jsonl_map)
    ):
        raise SyncError("qwen3 recovery MOS CSV/JSONL identities differ")
    for key, csv_row in csv_map.items():
        jsonl_row = jsonl_map[key]
        if (
            _normalized_recovery_doc_id(
                csv_row.get("doc_id"), "qwen3 recovery MOS CSV row"
            )
            != _normalized_recovery_doc_id(
                jsonl_row.get("doc_id"), "qwen3 recovery MOS JSONL row"
            )
            or _decimal(csv_row.get("open_mos_score"), "qwen3 MOS CSV score")
            != _decimal(jsonl_row.get("open_mos_score"), "qwen3 MOS JSONL score")
            or any(
                str(csv_row.get(field) or "") != str(jsonl_row.get(field) or "")
                for field in (
                    "mos_judge_model",
                    "mos_judge_revision",
                    "judge_fingerprint",
                    "request_fingerprint",
                    "error",
                )
            )
        ):
            raise SyncError(f"qwen3 recovery MOS CSV/JSONL row differs at {key}")
    if any(
        str(row.get("mos_judge_model") or "") != str(judge["model"])
        or str(row.get("mos_judge_revision") or "") != str(judge["revision"])
        or str(row.get("judge_fingerprint") or "") != str(judge["fingerprint"])
        for row in jsonl_rows
    ):
        raise SyncError("qwen3 recovery MOS JSONL judge identity differs")

    gate_specs = {
        "public": {QWEN3_RECOVERY_PUBLIC},
        "fresh": set(QWEN3_RECOVERY_REUSED[:2]),
        "combined": {
            QWEN3_RECOVERY_PUBLIC,
            *QWEN3_RECOVERY_REUSED[:2],
        },
    }
    gate_payloads: dict[str, Mapping[str, Any]] = {}
    gate_pairs: dict[str, tuple[Path, str]] = {}
    for label, baselines in gate_specs.items():
        pair = _recovery_descriptor(
            artifacts[f"{label}_selection"],
            base_directory=path.parent,
            label=f"qwen3 recovery {label} selection",
        )
        if pair[0] != recovery_root / "provenance/gates" / f"{label}.json":
            raise SyncError(f"qwen3 recovery {label} gate path differs")
        payload = _load_hashed_json(pair[0], f"qwen3 recovery {label} selection")
        _validate_recovery_gate(
            payload, label=label, baselines=baselines, selected=selected
        )
        gate_pairs[label] = pair
        gate_payloads[label] = payload
    if gate_pairs["combined"] != selection_pair or completed.get(
        "gate_selected_methods"
    ) != {label: selected for label in gate_specs}:
        raise SyncError("qwen3 completion/combined gate binding differs")
    combined = gate_payloads["combined"]
    for label in ("public", "fresh"):
        payload = gate_payloads[label]
        if (
            payload.get("candidate_scores") != combined.get("candidate_scores")
            or payload.get("candidate_secondary_scores")
            != combined.get("candidate_secondary_scores")
            or any(
                payload[score_map].get(method) != combined[score_map].get(method)
                for score_map in ("baseline_scores", "baseline_secondary_scores")
                for method in gate_specs[label]
            )
        ):
            raise SyncError(f"qwen3 recovery {label}/combined gate scores differ")
    campaign_contract_pair = _recovery_descriptor(
        contract["legacy"]["campaign_contract"],
        base_directory=contract_pair[0].parent,
        label="qwen3 parent campaign contract",
    )
    return campaign_contract_pair[0]


def _qwen7_floor_full_raw_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.strip():
                    continue
                row = _strict_json_loads(raw, f"{label} line {line_number}")
                if not isinstance(row, dict):
                    raise SyncError(f"{label} line {line_number} is not an object")
                rows.append(row)
    except OSError as exc:
        raise SyncError(f"could not read {label}: {exc}") from exc
    return rows


def _qwen7_floor_full_raw_doc_ids(path: Path, label: str) -> None:
    rows = _qwen7_floor_full_raw_jsonl(path, label)
    if len(rows) != MULTIFAMILY_EXPECTED_SAMPLES:
        raise SyncError(f"{label} must contain exactly 634 JSON objects")
    doc_ids: list[int] = []
    for line_number, row in enumerate(rows, start=1):
        doc_id = row.get("doc_id")
        if type(doc_id) is not int:
            raise SyncError(
                f"{label} line {line_number} doc_id must be a JSON integer"
            )
        doc_ids.append(doc_id)
    if len(set(doc_ids)) != MULTIFAMILY_EXPECTED_SAMPLES or set(doc_ids) != set(
        range(MULTIFAMILY_EXPECTED_SAMPLES)
    ):
        raise SyncError(f"{label} doc_id domain must be exactly 0..633")


def _qwen7_floor_full_raw_summary_samples(
    path: Path,
    *,
    method: str,
    run_name: str | None,
    label: str,
) -> int:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if (
                reader.fieldnames is None
                or len(reader.fieldnames) != len(set(reader.fieldnames))
                or not QWEN7_FLOOR_FULL_SUMMARY_REQUIRED_FIELDS
                <= set(reader.fieldnames)
            ):
                raise SyncError(f"{label} summary header is not closed")
            raw_rows = []
            for row_number, row in enumerate(reader, start=2):
                if (
                    None in row
                    or set(row) != set(reader.fieldnames)
                    or any(value is None for value in row.values())
                ):
                    raise SyncError(
                        f"{label} row {row_number} does not match its summary header"
                    )
                raw_rows.append(row)
            rows = [
                row
                for row in raw_rows
                if row.get("method") == method
                and row.get("task") == LPM_TASK
                and str(row.get("run_status", "")).lower() == "success"
                and (run_name is None or row.get("run_name") == run_name)
            ]
    except (OSError, csv.Error) as exc:
        raise SyncError(f"could not read {label}: {exc}") from exc
    if len(rows) != 1:
        raise SyncError(f"{label} must have exactly one successful method row")
    raw_samples = rows[0].get("samples")
    if (
        type(raw_samples) is not str
        or re.fullmatch(r"(?:0|[1-9][0-9]*)", raw_samples) is None
    ):
        raise SyncError(
            f"{label}.samples must be an integer encoded as a canonical "
            "nonnegative decimal token"
        )
    samples = int(raw_samples)
    if samples != MULTIFAMILY_EXPECTED_SAMPLES:
        raise SyncError(f"{label}.samples must equal 634")
    return samples


def _qwen7_floor_full_raw_score(value: Any, label: str) -> Decimal:
    if isinstance(value, bool):
        raise SyncError(f"{label} must be a finite numeric score")
    score = _decimal(value, label)
    assert score is not None
    if not Decimal(0) <= score <= Decimal(5):
        raise SyncError(f"{label} must be in [0,5]")
    return score


def _validate_qwen7_floor_full_raw_artifacts(
    *,
    contract: Mapping[str, Any],
    completed: Mapping[str, Any],
    contract_base: Path,
    completion_base: Path,
) -> dict[str, Any]:
    """Preserve raw JSON types and exact CSV/JSONL parity at publish time."""

    archive = contract.get("archived_full_control")
    markers = completed.get("markers")
    if not isinstance(archive, dict) or not isinstance(markers, dict):
        raise SyncError("qwen7 floor-cap48 raw sample sources are missing")
    if set(markers) != set(QWEN7_FLOOR_FULL_METHODS):
        raise SyncError("qwen7 floor-cap48 raw marker method set differs")

    sample_sources: dict[str, tuple[Any, Path]] = {
        QWEN7_FLOOR_FULL_ARCHIVE: (archive.get("sample"), contract_base)
    }
    for method in QWEN7_FLOOR_FULL_METHODS:
        marker = markers.get(method)
        if not isinstance(marker, dict):
            raise SyncError(f"qwen7 floor-cap48 marker {method} is malformed")
        sample_sources[method] = (marker.get("sample"), completion_base)
    for method in QWEN7_FLOOR_FULL_MOS_METHODS:
        sample_path, _ = validate_artifact_descriptor(
            sample_sources[method][0],
            base_directory=sample_sources[method][1],
            label=f"qwen7 floor-cap48 raw sample {method}",
        )
        _qwen7_floor_full_raw_doc_ids(
            sample_path, f"qwen7 floor-cap48 raw sample {method}"
        )

    mos_jsonl_path, _ = validate_artifact_descriptor(
        completed.get("mos_matrix_jsonl"),
        base_directory=completion_base,
        label="qwen7 floor-cap48 raw MOS JSONL",
    )
    mos_csv_path, _ = validate_artifact_descriptor(
        completed.get("mos_matrix_csv"),
        base_directory=completion_base,
        label="qwen7 floor-cap48 raw MOS CSV",
    )
    jsonl_rows = _qwen7_floor_full_raw_jsonl(
        mos_jsonl_path, "qwen7 floor-cap48 raw MOS JSONL"
    )
    if len(jsonl_rows) != len(QWEN7_FLOOR_FULL_MOS_METHODS) * 634:
        raise SyncError("qwen7 floor-cap48 raw MOS JSONL is not 4x634")
    jsonl_by_sample: dict[str, dict[str, Any]] = {}
    jsonl_doc_ids: dict[str, set[int]] = defaultdict(set)
    for line_number, row in enumerate(jsonl_rows, start=1):
        if set(row) != set(QWEN7_FLOOR_FULL_MOS_FIELDS):
            raise SyncError(
                f"qwen7 floor-cap48 raw MOS JSONL line {line_number} "
                "does not have the exact MOS fields"
            )
        method = row.get("method")
        sample_id = row.get("sample_id")
        doc_id = row.get("doc_id")
        if method not in QWEN7_FLOOR_FULL_MOS_METHODS:
            raise SyncError(
                f"qwen7 floor-cap48 raw MOS JSONL line {line_number} method differs"
            )
        if type(sample_id) is not str or not sample_id or sample_id in jsonl_by_sample:
            raise SyncError(
                f"qwen7 floor-cap48 raw MOS JSONL line {line_number} sample_id differs"
            )
        if type(doc_id) is not int:
            raise SyncError(
                f"qwen7 floor-cap48 raw MOS JSONL line {line_number} doc_id "
                "must be a JSON integer"
            )
        if doc_id in jsonl_doc_ids[method]:
            raise SyncError(
                f"qwen7 floor-cap48 raw MOS JSONL line {line_number} duplicates doc_id"
            )
        for field in QWEN7_FLOOR_FULL_MOS_TEXT_FIELDS:
            if field not in row or type(row[field]) is not str:
                raise SyncError(
                    f"qwen7 floor-cap48 raw MOS JSONL line {line_number} "
                    f"{field} must be a string"
                )
        if "open_mos_score" not in row or type(row["open_mos_score"]) is not int:
            raise SyncError(
                f"qwen7 floor-cap48 raw MOS JSONL line {line_number} "
                "score must be a JSON integer"
            )
        _qwen7_floor_full_raw_score(
            row["open_mos_score"],
            f"qwen7 floor-cap48 raw MOS JSONL line {line_number} score",
        )
        jsonl_by_sample[sample_id] = row
        jsonl_doc_ids[method].add(doc_id)
    exact_doc_ids = set(range(MULTIFAMILY_EXPECTED_SAMPLES))
    if set(jsonl_doc_ids) != set(QWEN7_FLOOR_FULL_MOS_METHODS) or any(
        doc_ids != exact_doc_ids for doc_ids in jsonl_doc_ids.values()
    ):
        raise SyncError(
            "qwen7 floor-cap48 raw MOS JSONL doc_id domain must be 0..633 per method"
        )

    csv_doc_ids: dict[str, set[int]] = defaultdict(set)
    seen_csv_samples: set[str] = set()
    try:
        with mos_csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if (
                reader.fieldnames is None
                or len(reader.fieldnames) != len(QWEN7_FLOOR_FULL_MOS_FIELDS)
                or set(reader.fieldnames) != set(QWEN7_FLOOR_FULL_MOS_FIELDS)
            ):
                raise SyncError(
                    "qwen7 floor-cap48 raw MOS CSV must have the exact MOS columns"
                )
            for row_number, csv_row in enumerate(reader, start=2):
                if (
                    None in csv_row
                    or set(csv_row) != set(QWEN7_FLOOR_FULL_MOS_FIELDS)
                    or any(value is None for value in csv_row.values())
                ):
                    raise SyncError(
                        f"qwen7 floor-cap48 raw MOS CSV row {row_number} "
                        "does not have the exact MOS cells"
                    )
                sample_id = csv_row.get("sample_id", "")
                if not sample_id or sample_id in seen_csv_samples:
                    raise SyncError(
                        f"qwen7 floor-cap48 raw MOS CSV row {row_number} "
                        "sample_id differs"
                    )
                jsonl_row = jsonl_by_sample.get(sample_id)
                if jsonl_row is None:
                    raise SyncError(
                        f"qwen7 floor-cap48 raw MOS CSV row {row_number} "
                        "has no JSONL row"
                    )
                method = str(csv_row.get("method", ""))
                doc_id = jsonl_row["doc_id"]
                if csv_row.get("doc_id") != str(doc_id):
                    raise SyncError(
                        f"qwen7 floor-cap48 raw MOS CSV row {row_number} doc_id "
                        "is not the canonical integer string"
                    )
                for field in QWEN7_FLOOR_FULL_MOS_TEXT_FIELDS:
                    if csv_row.get(field) != jsonl_row[field]:
                        raise SyncError(
                            f"qwen7 floor-cap48 raw MOS CSV/JSONL {field} differs "
                            f"at row {row_number}"
                        )
                csv_score = _qwen7_floor_full_raw_score(
                    csv_row.get("open_mos_score"),
                    f"qwen7 floor-cap48 raw MOS CSV row {row_number} score",
                )
                jsonl_score = _qwen7_floor_full_raw_score(
                    jsonl_row["open_mos_score"],
                    f"qwen7 floor-cap48 raw MOS JSONL {sample_id} score",
                )
                if (
                    csv_row.get("open_mos_score")
                    != str(jsonl_row["open_mos_score"])
                    or csv_score != jsonl_score
                ):
                    raise SyncError(
                        "qwen7 floor-cap48 raw MOS CSV score is not the canonical "
                        "integer string or differs from JSONL "
                        f"at row {row_number}"
                    )
                if (
                    method not in QWEN7_FLOOR_FULL_MOS_METHODS
                    or doc_id in csv_doc_ids[method]
                ):
                    raise SyncError(
                        f"qwen7 floor-cap48 raw MOS CSV row {row_number} "
                        "identity differs"
                    )
                seen_csv_samples.add(sample_id)
                csv_doc_ids[method].add(doc_id)
    except (OSError, csv.Error) as exc:
        raise SyncError(f"could not read qwen7 floor-cap48 raw MOS CSV: {exc}") from exc
    if (
        seen_csv_samples != set(jsonl_by_sample)
        or set(csv_doc_ids) != set(QWEN7_FLOOR_FULL_MOS_METHODS)
        or any(doc_ids != exact_doc_ids for doc_ids in csv_doc_ids.values())
    ):
        raise SyncError("qwen7 floor-cap48 raw MOS CSV is not exact 4x634 parity")

    archive_summary, _ = validate_artifact_descriptor(
        archive.get("summary"),
        base_directory=contract_base,
        label="qwen7 floor-cap48 archived raw summary",
    )
    summary_samples = {
        QWEN7_FLOOR_FULL_ARCHIVE: _qwen7_floor_full_raw_summary_samples(
            archive_summary,
            method=QWEN7_FLOOR_FULL_ARCHIVE,
            run_name=str(archive.get("summary_run_name", "")),
            label="qwen7 floor-cap48 archived raw summary",
        )
    }
    for method in QWEN7_FLOOR_FULL_METHODS:
        summary, _ = validate_artifact_descriptor(
            markers[method].get("summary"),
            base_directory=completion_base,
            label=f"qwen7 floor-cap48 raw summary {method}",
        )
        summary_samples[method] = _qwen7_floor_full_raw_summary_samples(
            summary,
            method=method,
            run_name=None,
            label=f"qwen7 floor-cap48 raw summary {method}",
        )
    return {
        "sample_rows_per_method": {
            method: 634 for method in QWEN7_FLOOR_FULL_MOS_METHODS
        },
        "mos_rows_per_method": {
            method: len(jsonl_doc_ids[method])
            for method in QWEN7_FLOOR_FULL_MOS_METHODS
        },
        "summary_samples": summary_samples,
    }


def _load_qwen7_floor_native_verifier(driver: Path, driver_sha256: str) -> Any:
    """Load the exact source-snapshot-bound native full verifier."""

    module_name = f"_dive_qwen7_floor_full_{driver_sha256}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, driver)
    if spec is None or spec.loader is None:
        raise SyncError("could not load qwen7 floor-cap48 native verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise SyncError(
            f"could not import qwen7 floor-cap48 native verifier: {exc}"
        ) from exc
    return module


def _validate_qwen7_floor_full_completion(
    path: Path,
    *,
    completed: Mapping[str, Any],
    entry: Mapping[str, Any],
    selection_pair: tuple[Path, str],
    matrix_pair: tuple[Path, str],
    summary_pair: tuple[Path, str],
    judge: Mapping[str, Any],
) -> None:
    """Delegate the complete artifact graph to the campaign's native verifier.

    The loader itself first authenticates the canonical contract, driver path,
    driver bytes and source-snapshot membership.  This avoids reducing the
    full contract to a few top-level JSON fields at the website boundary.
    """

    if (
        completed.get("schema") != QWEN7_FLOOR_FULL_COMPLETION_SCHEMA
        or completed.get("status") != "complete"
        or completed.get("stage") != "full"
        or completed.get("publishable") is not True
        or _integer(completed.get("expected_samples"), "qwen7 full samples") != 634
        or completed.get("campaign_fingerprint") != entry["campaign_fingerprint"]
        or completed.get("selected_method") != entry["selected_method"]
    ):
        raise SyncError("qwen7 floor-cap48 completion identity/status mismatch")
    contract_pair = validate_artifact_descriptor(
        completed.get("contract"),
        base_directory=path.parent,
        label="qwen7 floor-cap48 full contract",
    )
    contract = _load_hashed_json(
        contract_pair[0], "qwen7 floor-cap48 full contract"
    )
    fingerprint = str(contract.get("campaign_fingerprint", ""))
    if (
        contract_pair[1] != QWEN7_FLOOR_FULL_CONTRACT_SHA256
        or fingerprint != QWEN7_FLOOR_FULL_CAMPAIGN_FINGERPRINT
        or fingerprint != entry["campaign_fingerprint"]
        or _canonical_fingerprint(contract) != fingerprint
        or contract.get("campaign_kind") != "qwen7_route_floor_cap48_full_v1"
        or contract.get("expected_full_samples") != 634
        or contract.get("generation") != {"max_new_tokens": 48, "temperature": 0}
    ):
        raise SyncError("qwen7 floor-cap48 contract identity is not canonical")
    try:
        project_root = Path(str(contract["project_root"])).resolve(strict=True)
    except (KeyError, OSError) as exc:
        raise SyncError("qwen7 floor-cap48 project root is unavailable") from exc
    expected_driver = (
        project_root / "tools/densevideo/qwen7_floor_cap48_full.py"
    ).resolve(strict=True)
    driver_pair = validate_artifact_descriptor(
        completed.get("driver"),
        base_directory=path.parent,
        label="qwen7 floor-cap48 native driver",
        expected_path=expected_driver,
    )
    snapshot = contract.get("source_snapshot")
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("files"), list):
        raise SyncError("qwen7 floor-cap48 contract has no source snapshot")
    snapshot_pair = validate_artifact_descriptor(
        snapshot,
        base_directory=contract_pair[0].parent,
        label="qwen7 floor-cap48 source snapshot",
    )
    source_pairs = {
        validate_artifact_descriptor(
            descriptor,
            base_directory=contract_pair[0].parent,
            label=f"qwen7 floor-cap48 source[{index}]",
        )
        for index, descriptor in enumerate(snapshot["files"])
    }
    if driver_pair not in source_pairs:
        raise SyncError("qwen7 floor-cap48 native driver is absent from its snapshot")
    snapshot_lines = "".join(
        f"{descriptor['sha256']}  {Path(str(descriptor['path'])).resolve()}\n"
        for descriptor in snapshot["files"]
    )
    if snapshot_pair[0].read_text(encoding="utf-8") != snapshot_lines:
        raise SyncError("qwen7 floor-cap48 source snapshot content differs")

    native = _load_qwen7_floor_native_verifier(*driver_pair)
    if (
        Path(native.PROJECT_ROOT).resolve() != project_root
        or native.FULL_SAMPLES != 634
        or native.CANDIDATE != entry["selected_method"]
        or tuple(native.METHODS) != QWEN7_FLOOR_FULL_METHODS
        or tuple(native.MOS_METHODS) != QWEN7_FLOOR_FULL_MOS_METHODS
    ):
        raise SyncError("qwen7 floor-cap48 native verifier constants differ")
    try:
        verified_path, verified_contract = native.verified_contract(
            contract_pair[0], contract_pair[1], fingerprint
        )
        native.verify_completion_file(
            path,
            contract_path=verified_path,
            contract=verified_contract,
            contract_sha256=contract_pair[1],
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SyncError(
            f"qwen7 floor-cap48 native full evidence is invalid: {exc}"
        ) from exc
    _validate_qwen7_floor_full_raw_artifacts(
        contract=verified_contract,
        completed=completed,
        contract_base=contract_pair[0].parent,
        completion_base=path.parent,
    )

    gates = completed.get("gates")
    markers = completed.get("markers")
    if (
        not isinstance(gates, dict)
        or set(gates) != {"public", "fresh", "combined"}
        or not isinstance(markers, dict)
        or set(markers) != set(native.METHODS)
    ):
        raise SyncError("qwen7 floor-cap48 gate/marker set differs")
    combined_pair = validate_artifact_descriptor(
        gates["combined"],
        base_directory=path.parent,
        label="qwen7 floor-cap48 combined full gate",
    )
    completion_matrix_pair = validate_artifact_descriptor(
        completed.get("mos_matrix_csv"),
        base_directory=path.parent,
        label="qwen7 floor-cap48 full MOS CSV",
    )
    winner = markers.get(entry["selected_method"])
    if not isinstance(winner, dict):
        raise SyncError("qwen7 floor-cap48 winner marker is missing")
    completion_summary_pair = validate_artifact_descriptor(
        winner.get("summary"),
        base_directory=path.parent,
        label="qwen7 floor-cap48 winner summary",
    )
    if (
        combined_pair != selection_pair
        or completion_matrix_pair != matrix_pair
        or completion_summary_pair != summary_pair
    ):
        raise SyncError(
            "qwen7 floor-cap48 completion differs from release selection/matrix/summary"
        )
    diagnostics_pair = validate_artifact_descriptor(
        completed.get("split_diagnostics"),
        base_directory=path.parent,
        label="qwen7 floor-cap48 split diagnostics",
    )
    diagnostics = _load_hashed_json(
        diagnostics_pair[0], "qwen7 floor-cap48 split diagnostics"
    )
    if (
        diagnostics.get("eligible_as_quality_gate_input") is not False
        or diagnostics.get("full_634_is_only_hard_gate") is not True
        or diagnostics.get("split_sizes")
        != {"screen_selected_128": 128, "screen_unselected_506": 506}
    ):
        raise SyncError("qwen7 floor-cap48 split diagnostics are not reporting-only")
    checkpoint = contract.get("checkpoint")
    judge_contract = contract.get("judge")
    revisions = entry["revisions"]
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("revision") != revisions["model"]
        or contract.get("dataset_revision") != revisions["dataset"]
        or not isinstance(judge_contract, dict)
        or judge_contract.get("model") != judge["model"]
        or judge_contract.get("revision") != judge["revision"]
        or judge_contract.get("judge_fingerprint") != judge["fingerprint"]
        or revisions["judge"] != judge["revision"]
    ):
        raise SyncError("qwen7 floor-cap48 model/dataset/judge revisions differ")


def _validate_multifamily_completion(
    path: Path,
    *,
    entry: Mapping[str, Any],
    selection: Mapping[str, Any],
    selection_pair: tuple[Path, str],
    matrix_pair: tuple[Path, str],
    summary_pair: tuple[Path, str],
    judge: Mapping[str, Any],
) -> None:
    family = str(entry["family"])
    selected_method = str(entry["selected_method"])
    completed = _load_hashed_json(path, f"{family} completion")
    if entry["completion_schema"] == QWEN7_FLOOR_FULL_COMPLETION_SCHEMA:
        if family != "qwen7":
            raise SyncError(
                "Qwen7 floor-cap48 full completion cannot authorize another family"
            )
        if completed.get("campaign_kind") == "qwen7_route_floor_cap48_full_v1":
            raise SyncError(
                "FullCompletionRequired: a signed qwen7 floor-cap48 contract is "
                "non-terminal; its exact 634-sample full_completed.json is required"
            )
        _validate_qwen7_floor_full_completion(
            path,
            completed=completed,
            entry=entry,
            selection_pair=selection_pair,
            matrix_pair=matrix_pair,
            summary_pair=summary_pair,
            judge=judge,
        )
        return
    if (
        completed.get("schema_version") != 1
        or completed.get("status") != "complete"
        or completed.get("campaign_fingerprint") != entry["campaign_fingerprint"]
    ):
        raise SyncError(f"{family} completion identity/status mismatch")
    descriptors = _validate_nested_descriptors(
        completed,
        base_directory=path.parent,
        label=f"{family} completion",
    )
    if not {selection_pair, matrix_pair} <= descriptors:
        raise SyncError(f"{family} completion does not bind its selection and matrix")

    completion_schema = entry["completion_schema"]
    if completion_schema == "followup_parallel_completed_v1":
        if (
            completed.get("family") != family
            or completed.get("selected_method") != selected_method
        ):
            raise SyncError(f"{family} completion selected method/family mismatch")
        contract_path = _validate_qwen_parallel_completion(
            completed,
            path=path,
            entry=entry,
            selection=selection,
            selection_pair=selection_pair,
            matrix_pair=matrix_pair,
            summary_pair=summary_pair,
            judge=judge,
        )
    elif completion_schema == "llava7_dual_completed_v1":
        if completed.get("stage") != "full" or completed.get("expected_samples") != 634:
            raise SyncError("llava7 completion is not its 634-sample full stage")
        contract_path = _validate_llava7_completion(
            completed,
            path=path,
            entry=entry,
            selection=selection,
            selection_pair=selection_pair,
            matrix_pair=matrix_pair,
            summary_pair=summary_pair,
        )
    elif completion_schema == QWEN_DUAL_COMPLETION_SCHEMA:
        try:
            evidence = validate_qwen_dual_completion(
                path,
                family=family,
                judge_model=str(judge["model"]),
                judge_revision=str(judge["revision"]),
                judge_fingerprint=str(judge["fingerprint"]),
            )
        except QwenDualReleaseError as exc:
            raise SyncError(str(exc)) from exc

        def evidence_pair(descriptor: Mapping[str, str]) -> tuple[Path, str]:
            return Path(descriptor["path"]).resolve(), descriptor["sha256"]

        revisions = entry["revisions"]
        expected_methods = set(selection.get("baseline_scores", {})) | set(
            selection.get("candidate_scores", {})
        )
        if (
            evidence["campaign_fingerprint"] != entry["campaign_fingerprint"]
            or evidence["selected_method"] != selected_method
            or set(evidence["methods"]) != expected_methods
            or evidence_pair(evidence["selection"]) != selection_pair
            or evidence_pair(evidence["open_mos_matrix"]) != matrix_pair
            or evidence_pair(evidence["summary_csv"]) != summary_pair
            or evidence["model_revision"] != revisions["model"]
            or evidence["dataset_revision"] != revisions["dataset"]
            or revisions["judge"] != judge["revision"]
        ):
            raise SyncError(f"{family} dual completion differs from release evidence")
        return
    elif completion_schema == QWEN3_RECOVERY_COMPLETION_SCHEMA:
        if family != "qwen3":
            raise SyncError("Qwen3 recovery completion cannot authorize another family")
        contract_path = _validate_qwen3_recovery_completion(
            completed,
            path=path,
            entry=entry,
            selection=selection,
            selection_pair=selection_pair,
            matrix_pair=matrix_pair,
            summary_pair=summary_pair,
            judge=judge,
        )
    else:
        raise SyncError(
            f"unsupported multi-family completion schema {completion_schema!r}"
        )

    contract = _load_hashed_json(contract_path, f"{family} campaign contract")
    revisions = entry["revisions"]
    if (
        contract.get("campaign_fingerprint") != entry["campaign_fingerprint"]
        or _canonical_fingerprint(contract) != entry["campaign_fingerprint"]
        or contract.get("benchmark") != "DIVE-Bench"
        or contract.get("dataset_revision") != revisions["dataset"]
        or contract.get("expected_full_samples") != 634
    ):
        raise SyncError(f"{family} campaign contract identity/revision mismatch")
    if family == "llava7":
        checkpoint = contract.get("checkpoint")
        judge_contract = contract.get("judge")
        model_revision = (
            checkpoint.get("revision") if isinstance(checkpoint, dict) else None
        )
        judge_model = (
            judge_contract.get("model") if isinstance(judge_contract, dict) else None
        )
        judge_revision = (
            judge_contract.get("revision") if isinstance(judge_contract, dict) else None
        )
    else:
        model_revisions = contract.get("model_revisions")
        model_id = {
            "qwen3": "Qwen/Qwen2.5-VL-3B-Instruct",
            "qwen7": "Qwen/Qwen2.5-VL-7B-Instruct",
        }.get(family)
        model_revision = (
            model_revisions.get(model_id)
            if isinstance(model_revisions, dict) and model_id
            else None
        )
        judge_model = "Qwen/Qwen3-VL-32B-Instruct"
        judge_revision = contract.get("judge_revision")
    if (
        model_revision != revisions["model"]
        or judge_model != judge["model"]
        or judge_revision != judge["revision"]
        or revisions["judge"] != judge["revision"]
    ):
        raise SyncError(f"{family} campaign model/judge revisions changed")
    _validate_nested_descriptors(
        contract,
        base_directory=contract_path.parent,
        label=f"{family} campaign contract",
    )


def _strict_jsonl_objects(path: Path, label: str) -> Iterable[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.strip():
                    continue
                row = _strict_json_loads(raw, f"{label} line {line_number}")
                if not isinstance(row, dict):
                    raise SyncError(f"{label} line {line_number} is not an object")
                yield row
    except OSError as exc:
        raise SyncError(f"could not read {label}: {exc}") from exc


def _inspect_failed_full_mos_pair(
    csv_path: Path,
    jsonl_path: Path,
    *,
    family: str,
    expected_methods: set[str],
    judge: Mapping[str, Any],
) -> dict[str, dict[str, Decimal | float]]:
    """Recompute failure-gate metrics from an exact paired 4x634 matrix."""

    counts: dict[str, int] = defaultdict(int)
    doc_ids: dict[str, set[int]] = defaultdict(set)
    score_values: dict[str, list[Decimal]] = defaultdict(list)
    token_f1_values: dict[str, list[float]] = defaultdict(list)
    identities: dict[str, dict[int, tuple[str, ...]]] = defaultdict(dict)
    sample_ids: set[str] = set()
    json_rows = iter(_strict_jsonl_objects(jsonl_path, f"{family} failure MOS JSONL"))
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if (
                reader.fieldnames is None
                or len(reader.fieldnames) != len(QWEN7_FLOOR_FULL_MOS_FIELDS)
                or set(reader.fieldnames) != set(QWEN7_FLOOR_FULL_MOS_FIELDS)
            ):
                raise SyncError(f"{family} failure MOS CSV schema differs")
            for row_number, csv_row in enumerate(reader, start=2):
                try:
                    json_row = next(json_rows)
                except StopIteration as exc:
                    raise SyncError(
                        f"{family} failure MOS JSONL ended before its CSV"
                    ) from exc
                if set(json_row) != set(QWEN7_FLOOR_FULL_MOS_FIELDS):
                    raise SyncError(
                        f"{family} failure MOS JSONL row schema differs at {row_number}"
                    )
                if (
                    None in csv_row
                    or set(csv_row) != set(QWEN7_FLOOR_FULL_MOS_FIELDS)
                    or any(value is None for value in csv_row.values())
                ):
                    raise SyncError(
                        f"{family} failure MOS CSV row schema differs at {row_number}"
                    )
                method = json_row.get("method")
                sample_id = json_row.get("sample_id")
                doc_id = json_row.get("doc_id")
                score = json_row.get("open_mos_score")
                if (
                    type(method) is not str
                    or method not in expected_methods
                    or type(sample_id) is not str
                    or not sample_id.startswith(f"{method}::")
                    or sample_id in sample_ids
                    or type(doc_id) is not int
                    or doc_id in doc_ids[method]
                    or type(score) is not int
                ):
                    raise SyncError(
                        f"{family} failure MOS typed identity differs at row {row_number}"
                    )
                if csv_row.get("doc_id") != str(doc_id):
                    raise SyncError(
                        f"{family} failure MOS CSV doc_id is not canonical at row "
                        f"{row_number}"
                    )
                for field in QWEN7_FLOOR_FULL_MOS_TEXT_FIELDS:
                    if type(json_row[field]) is not str or csv_row.get(field) != json_row[
                        field
                    ]:
                        raise SyncError(
                            f"{family} failure MOS CSV/JSONL {field} differs at row "
                            f"{row_number}"
                        )
                json_score = _decimal(score, f"{family}.failure_jsonl.score")
                csv_score = _decimal(
                    csv_row.get("open_mos_score"), f"{family}.failure_csv.score"
                )
                assert json_score is not None and csv_score is not None
                review = _strict_json_loads(
                    json_row["open_mos_review"],
                    f"{family} failure MOS review row {row_number}",
                )
                correctness = json_row["open_mos_correctness"].lower()
                if (
                    csv_row.get("open_mos_score") != str(score)
                    or json_score != csv_score
                    or json_score != json_score.to_integral_value()
                    or not Decimal(0) <= json_score <= Decimal(5)
                    or json_row["mos_judge_model"] != judge["model"]
                    or json_row["mos_judge_revision"] != judge["revision"]
                    or json_row["judge_fingerprint"] != judge["fingerprint"]
                    or SHA256_PATTERN.fullmatch(json_row["request_fingerprint"])
                    is None
                    or json_row["request_fingerprint"]
                    != _open_mos_request_fingerprint(
                        json_row["question"], json_row["answer"], json_row["pred"]
                    )
                    or correctness not in {"yes", "no"}
                    or not isinstance(review, dict)
                    or set(review) != {"pred", "score"}
                    or review.get("pred") != correctness
                    or type(review.get("score")) is not int
                    or Decimal(review["score"]) != json_score
                    or json_row["error"]
                ):
                    raise SyncError(
                        f"{family} failure MOS judge/score differs at row {row_number}"
                    )
                sample_ids.add(sample_id)
                doc_ids[method].add(doc_id)
                counts[method] += 1
                identities[method][doc_id] = tuple(
                    json_row[field]
                    for field in (
                        "video_name",
                        "question_id",
                        "type",
                        "question",
                        "answer",
                    )
                )
                score_values[method].append(json_score)
                token_f1_values[method].append(
                    _densevideo_token_f1(json_row["pred"], json_row["answer"])
                )
    except (OSError, csv.Error) as exc:
        raise SyncError(f"could not inspect {family} failure MOS CSV: {exc}") from exc
    try:
        next(json_rows)
    except StopIteration:
        pass
    else:
        raise SyncError(f"{family} failure MOS JSONL has rows absent from its CSV")
    exact_doc_ids = set(range(MULTIFAMILY_EXPECTED_SAMPLES))
    if (
        set(counts) != expected_methods
        or any(count != MULTIFAMILY_EXPECTED_SAMPLES for count in counts.values())
        or any(values != exact_doc_ids for values in doc_ids.values())
        or len(sample_ids) != len(expected_methods) * MULTIFAMILY_EXPECTED_SAMPLES
    ):
        raise SyncError(f"{family} failure MOS is not exact methods x 634")
    reference_method = sorted(expected_methods)[0]
    if any(
        identities[method] != identities[reference_method]
        for method in expected_methods
    ):
        raise SyncError(f"{family} failure MOS paired identities differ")
    return {
        method: {
            "open_mos": sum(score_values[method], Decimal(0))
            / Decimal(MULTIFAMILY_EXPECTED_SAMPLES),
            "token_f1": math.fsum(token_f1_values[method])
            / MULTIFAMILY_EXPECTED_SAMPLES,
        }
        for method in sorted(expected_methods)
    }


def _validate_multifamily_release_v1_binding(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    leaderboard_csv: Path,
    leaderboard_rows: Sequence[Mapping[str, str]],
    metadata: Mapping[str, Any],
    metadata_path: Path,
) -> list[dict[str, Any]]:
    expected_header = {
        "schema_version",
        "schema",
        "benchmark",
        "task",
        "expected_samples",
        "expected_lpm_rows",
        "expected_highmotion_rows",
        "judge",
        "families",
    }
    if (
        set(manifest) != expected_header
        or manifest.get("schema_version") != 1
        or manifest.get("schema") != MULTIFAMILY_RELEASE_SCHEMA
        or manifest.get("benchmark") != "DIVE-Bench"
        or manifest.get("task") != LPM_TASK
        or manifest.get("expected_samples") != MULTIFAMILY_EXPECTED_SAMPLES
        or _integer(manifest.get("expected_lpm_rows"), "manifest.expected_lpm_rows")
        != MULTIFAMILY_EXPECTED_LPM_ROWS
        or _integer(
            manifest.get("expected_highmotion_rows"),
            "manifest.expected_highmotion_rows",
        )
        != MULTIFAMILY_EXPECTED_HIGHMOTION_ROWS
    ):
        raise SyncError(
            "multi-family release manifest header/count contract is invalid"
        )
    judge = manifest.get("judge")
    if not isinstance(judge, dict) or set(judge) != {
        "model",
        "revision",
        "fingerprint",
    }:
        raise SyncError("multi-family release has no exact judge identity")
    if (
        IMMUTABLE_REVISION_PATTERN.fullmatch(str(judge["revision"])) is None
        or SHA256_PATTERN.fullmatch(str(judge["fingerprint"])) is None
    ):
        raise SyncError(
            "multi-family release judge revision/fingerprint is not immutable"
        )

    release = metadata.get("artifact_provenance")
    if (
        not isinstance(release, dict)
        or _integer(
            release.get("schema_version"), "release artifact_provenance.schema_version"
        )
        < 2
    ):
        raise SyncError(
            "multi-family release metadata requires artifact schema version 2"
        )
    metadata_directory = metadata_path.parent
    validate_artifact_descriptor(
        release.get("leaderboard_csv"),
        base_directory=metadata_directory,
        label="release leaderboard_csv",
        expected_path=leaderboard_csv,
    )
    validate_artifact_descriptor(
        release.get("grt_release_manifest"),
        base_directory=metadata_directory,
        label="release grt_release_manifest",
        expected_path=manifest_path,
    )
    source_groups = release.get("source_artifacts")
    if not isinstance(source_groups, dict):
        raise SyncError("release metadata has no source_artifacts object")
    validated_groups: dict[str, set[tuple[Path, str]]] = {}
    for group in ("summary_csv", "open_mos_artifacts", "api_summaries"):
        descriptors = source_groups.get(group)
        if not isinstance(descriptors, list):
            raise SyncError(f"release source_artifacts.{group} must be a list")
        validated_groups[group] = {
            validate_artifact_descriptor(
                descriptor,
                base_directory=metadata_directory,
                label=f"release source_artifacts.{group}[{index}]",
            )
            for index, descriptor in enumerate(descriptors)
        }
        paths = metadata.get(group)
        if not isinstance(paths, list):
            raise SyncError(f"release metadata {group} must be a path list")
        if {
            _artifact_path(item, metadata_directory, f"release {group} path")
            for item in paths
        } != {item[0] for item in validated_groups[group]}:
            raise SyncError(
                f"release metadata {group} paths differ from hashed descriptors"
            )

    judge_metadata = metadata.get("open_mos_judge_provenance")
    if not isinstance(judge_metadata, dict) or judge_metadata != {
        "model": judge["model"],
        "revision": judge["revision"],
        "judge_fingerprint": judge["fingerprint"],
    }:
        raise SyncError("release metadata judge identity differs from the manifest")
    if metadata.get("open_mos_judge") != judge["model"]:
        raise SyncError(
            "release metadata Open-MOS judge name differs from the manifest"
        )

    entries = manifest.get("families")
    if not isinstance(entries, list) or len(entries) != len(MULTIFAMILY_CONTRACTS):
        raise SyncError("multi-family release must contain exactly four entries")
    raw_by_family = {
        str(entry.get("family") or ""): entry
        for entry in entries
        if isinstance(entry, dict)
    }
    if len(raw_by_family) != len(entries) or set(raw_by_family) != set(
        MULTIFAMILY_CONTRACTS
    ):
        raise SyncError("multi-family release family names are missing or duplicated")
    validate_legacy_open_mos_release_binding(
        release.get("legacy_open_mos_completion"),
        metadata_directory=metadata_directory,
        judge=judge,
        open_mos_sources=validated_groups["open_mos_artifacts"],
        open_mos_by_method=metadata.get("open_mos"),
        route31_entry=raw_by_family["route31"],
    )
    metadata_entries = release.get("verified_grt_selections")
    if not isinstance(metadata_entries, list):
        raise SyncError("release metadata has no verified_grt_selections list")
    metadata_by_family = {
        str(entry.get("family") or ""): entry
        for entry in metadata_entries
        if isinstance(entry, dict)
    }
    if set(metadata_by_family) != set(MULTIFAMILY_CONTRACTS) or len(
        metadata_by_family
    ) != len(metadata_entries):
        raise SyncError("release verified_grt_selections family set is invalid")

    verified: list[dict[str, Any]] = []
    occupied: set[str] = set()
    for family, family_contract in MULTIFAMILY_CONTRACTS.items():
        entry = raw_by_family[family]
        allowed_completion = family_contract["completion_schema"]
        completion_schema = entry.get("completion_schema")
        completion_matches = (
            completion_schema in allowed_completion
            if isinstance(allowed_completion, frozenset)
            else completion_schema == allowed_completion
        )
        if set(entry) != {
            "family",
            "base_method",
            "replaces_method",
            "selection_schema",
            "completion_schema",
            "campaign_fingerprint",
            "revisions",
            "selection",
            "completion",
            "open_mos_matrix",
            "summary_csv",
        } or any(
            entry.get(key) != value
            for key, value in family_contract.items()
            if key != "completion_schema"
        ) or not completion_matches:
            raise SyncError(
                f"{family} release entry differs from its fixed family contract"
            )
        fingerprint = str(entry.get("campaign_fingerprint") or "")
        revisions = entry.get("revisions")
        if (
            SHA256_PATTERN.fullmatch(fingerprint) is None
            or not isinstance(revisions, dict)
            or set(revisions) != {"model", "dataset", "judge"}
            or any(
                IMMUTABLE_REVISION_PATTERN.fullmatch(str(value)) is None
                for value in revisions.values()
            )
            or revisions["judge"] != judge["revision"]
        ):
            raise SyncError(
                f"{family} release fingerprint/revision contract is invalid"
            )
        selection_pair = validate_artifact_descriptor(
            entry.get("selection"),
            base_directory=manifest_path.parent,
            label=f"{family} selection",
        )
        matrix_pair = validate_artifact_descriptor(
            entry.get("open_mos_matrix"),
            base_directory=manifest_path.parent,
            label=f"{family} Open-MOS matrix",
        )
        summary_pair = validate_artifact_descriptor(
            entry.get("summary_csv"),
            base_directory=manifest_path.parent,
            label=f"{family} winner summary",
        )
        selection = _load_hashed_json(selection_pair[0], f"{family} selection")
        selected_method = str(selection.get("selected_method") or "").strip()
        candidate = _site_row(
            _candidate_row(leaderboard_rows, selected_method),
            selected_method,
            634,
        )
        selected_method, methods = validate_strict_multifamily_selection(
            selection,
            candidate,
            family=family,
            base_method=str(family_contract["base_method"]),
            expected_samples=634,
            completion_schema=(
                str(completion_schema) if completion_schema is not None else None
            ),
        )
        if occupied.intersection(methods):
            raise SyncError(
                "multi-family selection method sets overlap: "
                + ", ".join(sorted(occupied.intersection(methods)))
            )
        occupied.update(methods)
        expected_matrix_methods = (
            set(qwen_dual_mos_methods())
            if completion_schema == QWEN_DUAL_COMPLETION_SCHEMA
            else set(methods)
        )
        matrix_means = _inspect_multifamily_matrix(
            matrix_pair[0],
            family=family,
            expected_methods=expected_matrix_methods,
            expected_samples=634,
            judge=judge,
        )
        score_maps = {
            **selection["baseline_scores"],
            **selection["candidate_scores"],
        }
        family_matrix_means = {
            method: matrix_means[method]
            for method in methods
            if method in matrix_means
        }
        if not _open_mos_means_match(
            family_matrix_means, score_maps, family=family
        ):
            raise SyncError(
                f"{family} selection OpenMOS scores do not reproduce its matrix"
            )
        _validate_multifamily_summary(
            summary_pair[0],
            family=family,
            selected_method=selected_method,
            expected_samples=634,
        )
        if matrix_pair not in validated_groups["open_mos_artifacts"]:
            raise SyncError(f"{family} matrix is absent from hashed release sources")
        if summary_pair not in validated_groups["summary_csv"]:
            raise SyncError(
                f"{family} winner summary is absent from hashed release sources"
            )

        completion_pair: tuple[Path, str] | None = None
        if completion_schema is None:
            if entry.get("completion") is not None:
                raise SyncError(
                    "Route31 must use its self-contained selection provenance"
                )
            artifacts = validate_selection_artifacts(
                selection, selection_path=selection_pair[0]
            )
            provenance = selection.get("artifact_provenance")
            _, _, selection_fingerprint = selection_identity(provenance)
            if selection_fingerprint != fingerprint:
                raise SyncError("Route31 selection/manifest fingerprint mismatch")
            if (
                artifacts["open_mos_matrix"] != matrix_pair
                or artifacts["summary_route31_candidate"] != summary_pair
            ):
                raise SyncError(
                    "Route31 manifest matrix/summary is not selection-bound"
                )
            telemetry = _load_hashed_json(
                artifacts["telemetry_contract"][0], "Route31 telemetry"
            )
            telemetry_methods = telemetry.get("methods")
            if (
                telemetry.get("schema_version") != 1
                or telemetry.get("status") != "passed"
                or telemetry.get("limits")
                != {
                    "max_recompute_ratio": Decimal("0.98"),
                    "max_reference_patch_compute_ratio": Decimal("0.98"),
                }
                or not isinstance(telemetry_methods, dict)
                or set(telemetry_methods) != set(ROUTE31_METHODS[1:])
            ):
                raise SyncError(
                    "Route31 telemetry artifact does not match the campaign"
                )
            winner_telemetry = telemetry_methods[ROUTE31_WINNER_METHOD]
            if (
                winner_telemetry.get("role") != "candidate"
                or winner_telemetry.get("passes_recompute_caps") is not True
                or winner_telemetry.get("effective_max_new_tokens_counts")
                != {"128": 317, "31": 317}
                or winner_telemetry.get("question_route_counts")
                != {"ocr": 317, "subtitle": 317}
            ):
                raise SyncError(
                    "Route31 winner telemetry route/cap contract is invalid"
                )
            for field, candidate_field in (
                ("mean_recompute_ratio", "recompute_ratio"),
                ("reference_patch_compute_ratio", "reference_recompute_ratio"),
                ("mean_effective_fps", "effective_fps"),
                ("mean_throughput_fps", "throughput_fps"),
            ):
                if not _published_matches(
                    winner_telemetry.get(field), candidate.get(candidate_field)
                ):
                    raise SyncError(
                        f"Route31 release {field} differs from authenticated telemetry"
                    )
            secondary = {
                **selection.get("baseline_secondary_scores", {}),
                **selection.get("candidate_secondary_scores", {}),
            }
            summary_names = {
                ROUTE31_PUBLIC_METHOD: "summary_archived_public",
                ROUTE31_BASE_METHOD: "summary_route31_base",
                ROUTE31_EXACT_METHOD: "summary_route31_exact",
                ROUTE31_WINNER_METHOD: "summary_route31_candidate",
            }
            if set(secondary) != set(summary_names):
                raise SyncError("Route31 Token-F1 selection method set is incomplete")
            for method, artifact_name in summary_names.items():
                row = _summary_row(
                    artifacts[artifact_name][0],
                    family="route31",
                    method=method,
                    expected_samples=MULTIFAMILY_EXPECTED_SAMPLES,
                )
                _assert_secondary_score(row, secondary[method], f"route31.{method}")
            evaluation = _load_hashed_json(
                artifacts["evaluation_contract"][0], "Route31 contract"
            )
            if (
                evaluation.get("model_revision") != revisions["model"]
                or evaluation.get("dataset_revision") != revisions["dataset"]
                or evaluation.get("judge_revision") != revisions["judge"]
            ):
                raise SyncError("Route31 manifest revisions differ from its contract")
        else:
            completion_pair = validate_artifact_descriptor(
                entry.get("completion"),
                base_directory=manifest_path.parent,
                label=f"{family} completion",
            )
            enriched_entry = {
                **entry,
                "selected_method": selected_method,
            }
            _validate_multifamily_completion(
                completion_pair[0],
                entry=enriched_entry,
                selection=selection,
                selection_pair=selection_pair,
                matrix_pair=matrix_pair,
                summary_pair=summary_pair,
                judge=judge,
            )

        expected_metadata = {
            "family": family,
            "base_method": family_contract["base_method"],
            "replaces_method": family_contract["replaces_method"],
            "selection_schema": family_contract["selection_schema"],
            "completion_schema": completion_schema,
            "campaign_fingerprint": fingerprint,
            "revisions": revisions,
            "selected_method": selected_method,
            "methods": list(methods),
            "selection": {"path": str(selection_pair[0]), "sha256": selection_pair[1]},
            "completion": (
                {"path": str(completion_pair[0]), "sha256": completion_pair[1]}
                if completion_pair
                else None
            ),
            "open_mos_matrix": {"path": str(matrix_pair[0]), "sha256": matrix_pair[1]},
            "summary_csv": {"path": str(summary_pair[0]), "sha256": summary_pair[1]},
        }
        if metadata_by_family[family] != expected_metadata:
            raise SyncError(
                f"{family} release metadata descriptor differs from verified evidence"
            )
        verified.append(
            {
                **expected_metadata,
                "selection_payload": selection,
                "candidate": candidate,
            }
        )
    return verified


def _exact_descriptor(
    value: Any,
    *,
    base_directory: Path,
    label: str,
    expected_path: Path | None = None,
) -> tuple[Path, str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise SyncError(f"{label} must be an exact path/SHA object")
    return validate_artifact_descriptor(
        value,
        base_directory=base_directory,
        label=label,
        expected_path=expected_path,
    )


def _descriptor_object(pair: tuple[Path, str]) -> dict[str, str]:
    return {"path": str(pair[0]), "sha256": pair[1]}


def _require_pinned_pair(
    pair: tuple[Path, str], expected_sha256: str, label: str
) -> None:
    """Reject substituted evidence even when the descriptor is re-signed."""

    if pair[1] != expected_sha256:
        raise SyncError(f"{label} is not the pinned artifact")


def _validate_failure_descriptor_tree(
    payload: Any,
    *,
    base_directory: Path,
    label: str,
) -> int:
    """Hash explicit descriptors and adjacent ``*_sha256`` path pairs."""

    validated = 0
    if isinstance(payload, dict):
        has_path = "path" in payload
        has_sha = "sha256" in payload
        if has_path != has_sha:
            raise SyncError(f"{label} has a partial path/SHA descriptor")
        if has_path:
            validate_artifact_descriptor(
                {"path": payload["path"], "sha256": payload["sha256"]},
                base_directory=base_directory,
                label=label,
            )
            validated += 1
        for key, value in payload.items():
            if key.endswith("_sha256"):
                path_key = key[: -len("_sha256")]
                raw_path = payload.get(path_key)
                if isinstance(raw_path, str):
                    validate_artifact_descriptor(
                        {"path": raw_path, "sha256": value},
                        base_directory=base_directory,
                        label=f"{label}.{path_key}",
                    )
                    validated += 1
            if key not in {"path", "sha256"}:
                validated += _validate_failure_descriptor_tree(
                    value,
                    base_directory=base_directory,
                    label=f"{label}.{key}",
                )
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            validated += _validate_failure_descriptor_tree(
                value,
                base_directory=base_directory,
                label=f"{label}[{index}]",
            )
    return validated


def _verify_historical_source_snapshot(
    value: Any,
    *,
    family: str,
    contract_base: Path,
) -> tuple[Path, str]:
    """Verify the frozen inventory bytes without rehashing today's source tree."""

    if not isinstance(value, dict) or set(value) != {"path", "sha256", "files"}:
        raise SyncError(f"{family} historical source snapshot schema differs")
    pair = validate_artifact_descriptor(
        {"path": value["path"], "sha256": value["sha256"]},
        base_directory=contract_base,
        label=f"{family} historical source snapshot",
    )
    rows = value.get("files")
    if not isinstance(rows, list) or not rows:
        raise SyncError(f"{family} historical source inventory is empty")
    paths: set[str] = set()
    lines: list[str] = []
    for index, row in enumerate(rows):
        if (
            not isinstance(row, dict)
            or set(row) != {"path", "sha256"}
            or SHA256_PATTERN.fullmatch(str(row.get("sha256") or "")) is None
        ):
            raise SyncError(f"{family} historical source row {index} differs")
        path = str(Path(str(row["path"])).expanduser().resolve())
        if path in paths:
            raise SyncError(f"{family} historical source snapshot duplicates a path")
        paths.add(path)
        lines.append(f"{row['sha256']}  {path}\n")
    try:
        content = pair[0].read_text(encoding="utf-8")
    except OSError as exc:
        raise SyncError(f"could not read {family} historical source snapshot") from exc
    if content != "".join(lines):
        raise SyncError(f"{family} historical source snapshot inventory differs")
    return pair


def _validate_failed_selection(
    selection: Mapping[str, Any], *, family: str
) -> None:
    if set(selection) != {
        "status",
        "reason",
        "metric",
        "selected_method",
        "max_recompute_ratio",
        "max_reference_recompute_ratio",
        "reference_ratio_overrides",
    }:
        raise SyncError(f"{family} failed selection closed schema differs")
    max_ratio = _decimal(
        selection.get("max_recompute_ratio"), f"{family}.failed.max_recompute_ratio"
    )
    max_reference = _decimal(
        selection.get("max_reference_recompute_ratio"),
        f"{family}.failed.max_reference_recompute_ratio",
    )
    if (
        selection.get("status") != "failed"
        or selection.get("metric") != "open_mos"
        or selection.get("selected_method") is not None
        or max_ratio != Decimal("0.98")
        or max_reference != Decimal("0.98")
        or selection.get("reference_ratio_overrides") not in ({}, [])
        or not str(selection.get("reason") or "").strip()
    ):
        raise SyncError(f"{family} failed selection disposition differs")


def _failure_first_value(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            nested = _failure_first_value(*value)
            if nested:
                return nested
        elif value != "":
            return str(value)
    return ""


def _failure_sample_fields(
    record: Mapping[str, Any], *, index: int
) -> dict[str, str]:
    doc = record.get("doc") if isinstance(record.get("doc"), dict) else {}
    if "filtered_resps" in record:
        prediction = _failure_first_value(record.get("filtered_resps"))
    elif "resps" in record:
        prediction = _failure_first_value(record.get("resps"))
    else:
        prediction = _failure_first_value(
            record.get("pred"),
            record.get("prediction_parsed"),
            record.get("prediction_raw"),
        )
    answer = _failure_first_value(
        doc.get("answer"), record.get("target"), record.get("ground_truth")
    )
    question = _failure_first_value(
        doc.get("question"),
        record.get("input"),
        record.get("question"),
        record.get("prompt"),
    )
    if not question and str(record.get("subtask") or "").strip().lower() == "lpm":
        video_id = _failure_first_value(
            record.get("video_id"), record.get("video_path")
        )
        if video_id:
            question = f"What subtitles appear in the entire video {video_id}?"
    doc_id_text = _failure_first_value(record.get("doc_id"), index)
    doc_hash = str(record.get("doc_hash") or "")[:12]
    suffix = f"doc_{doc_id_text}_{doc_hash}" if doc_hash else f"doc_{doc_id_text}"
    return {
        "sample_suffix": suffix,
        "video_name": _failure_first_value(
            doc.get("video"), doc.get("video_id"), record.get("video_id")
        ),
        "question_id": _failure_first_value(
            doc.get("qid"), record.get("question_id"), record.get("doc_id")
        ),
        "type": _failure_first_value(doc.get("type"), record.get("type")),
        "question": question,
        "answer": answer,
        "pred": prediction,
    }


def _validate_failure_matrix_sample_binding(
    matrix_path: Path,
    *,
    family: str,
    methods: Sequence[str],
    sample_descriptors: Mapping[str, Any],
    descriptor_bases: Mapping[str, Path],
    trim_limit: int,
) -> None:
    matrix_by_method: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    try:
        with matrix_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if (
                reader.fieldnames is None
                or len(reader.fieldnames) != len(QWEN7_FLOOR_FULL_MOS_FIELDS)
                or set(reader.fieldnames) != set(QWEN7_FLOOR_FULL_MOS_FIELDS)
            ):
                raise SyncError(f"{family} matrix/sample binding header differs")
            for row_number, row in enumerate(reader, start=2):
                if None in row or any(value is None for value in row.values()):
                    raise SyncError(
                        f"{family} matrix/sample binding row {row_number} differs"
                    )
                method = str(row["method"])
                raw_doc_id = row["doc_id"]
                if re.fullmatch(r"(?:0|[1-9][0-9]*)", raw_doc_id) is None:
                    raise SyncError(f"{family} matrix doc_id is not canonical")
                doc_id = int(raw_doc_id)
                if doc_id in matrix_by_method[method]:
                    raise SyncError(f"{family} matrix/sample binding duplicates doc_id")
                matrix_by_method[method][doc_id] = row
    except (OSError, csv.Error) as exc:
        raise SyncError(f"could not bind {family} matrix to raw samples") from exc
    if set(matrix_by_method) != set(methods) or set(sample_descriptors) != set(methods):
        raise SyncError(f"{family} matrix/raw sample method sets differ")
    for method in methods:
        sample_pair = _exact_descriptor(
            sample_descriptors[method],
            base_directory=descriptor_bases[method],
            label=f"{family}.{method}.raw_sample",
        )
        samples = list(
            _strict_jsonl_objects(sample_pair[0], f"{family}.{method}.raw_sample")
        )
        if len(samples) != MULTIFAMILY_EXPECTED_SAMPLES:
            raise SyncError(f"{family}.{method} raw sample coverage is not 634")
        for index, sample in enumerate(samples):
            doc_id = sample.get("doc_id")
            if type(doc_id) is not int or doc_id != index:
                raise SyncError(f"{family}.{method} raw sample order/doc_id differs")
            matrix_row = matrix_by_method[method].get(doc_id)
            if matrix_row is None:
                raise SyncError(f"{family}.{method} matrix/raw docset differs")
            fields = _failure_sample_fields(sample, index=index)
            expected = {
                "sample_id": f"{method}::{fields.pop('sample_suffix')}",
                **fields,
            }
            expected["request_fingerprint"] = _open_mos_request_fingerprint(
                expected["question"],
                expected["answer"],
                expected["pred"],
                trim_limit=trim_limit,
            )
            review = _strict_json_loads(
                matrix_row["open_mos_review"], f"{family} matrix review"
            )
            if not isinstance(review, dict):
                raise SyncError(f"{family} matrix review is not an object")
            expected["open_mos_correctness"] = str(review.get("pred") or "").lower()
            expected["open_mos_score"] = str(review.get("score"))
            if any(matrix_row[field] != value for field, value in expected.items()):
                raise SyncError(
                    f"{family} matrix/raw request binding differs for {method}:{doc_id}"
                )


def _validate_llava7_failed_evidence(
    entry: Mapping[str, Any],
    *,
    manifest_path: Path,
    judge: Mapping[str, Any],
) -> dict[str, Any]:
    family = "llava7"
    evidence = entry["evidence"]
    contract_pair = _exact_descriptor(
        evidence["contract"],
        base_directory=manifest_path.parent,
        label="llava7 failed contract",
    )
    contract = _load_hashed_json(contract_pair[0], "llava7 failed contract")
    root = contract_pair[0].parent.parent
    fingerprint = str(contract.get("campaign_fingerprint") or "")
    methods = {
        "base": "llava_onevision_7b_dual_quality_base",
        "exact_control": "llava_onevision_7b_dual_grt_all",
        "candidate": "grt_llava_onevision_7b_hf_dual_s002_o005",
    }
    quality_gate = contract.get("quality_gate")
    checkpoint = contract.get("checkpoint")
    contract_judge = contract.get("judge")
    if (
        contract_pair[1] != LLAVA7_FAILED_CONTRACT_SHA256
        or fingerprint != LLAVA7_FAILED_CAMPAIGN_FINGERPRINT
        or _canonical_fingerprint(contract) != fingerprint
        or fingerprint != entry["campaign_fingerprint"]
        or contract.get("benchmark") != "DIVE-Bench"
        or contract.get("task") != LPM_TASK
        or contract.get("expected_full_samples") != MULTIFAMILY_EXPECTED_SAMPLES
        or Path(str(contract.get("output_root") or "")).expanduser().resolve()
        != root
        or contract.get("methods") != methods
        or not isinstance(quality_gate, dict)
        or any(
            quality_gate.get(key) != expected
            for key, expected in {
                "primary_metric": "open_mos",
                "secondary_metric": "token_f1",
                "strictly_greater_than_all_controls": True,
                "max_recompute_ratio": Decimal("0.98"),
                "max_reference_patch_compute_ratio": Decimal("0.98"),
            }.items()
        )
        or not isinstance(checkpoint, dict)
        or checkpoint.get("model")
        != "llava-hf/llava-onevision-qwen2-7b-ov-hf"
        or checkpoint.get("revision") != entry["revisions"]["model"]
        or contract.get("dataset_revision") != entry["revisions"]["dataset"]
        or not isinstance(contract_judge, dict)
        or contract_judge.get("model") != judge["model"]
        or contract_judge.get("revision") != judge["revision"]
        or entry["revisions"]["judge"] != judge["revision"]
    ):
        raise SyncError("llava7 failed contract identity/gate differs")
    archived = contract.get("archived_quality_control")
    if (
        not isinstance(archived, dict)
        or archived.get("method") != "llava_onevision_original"
        or _validate_failure_descriptor_tree(
            archived,
            base_directory=contract_pair[0].parent,
            label="llava7 archived public control",
        )
        < 3
    ):
        raise SyncError("llava7 archived public control is not deeply bound")
    snapshot_pair = _verify_historical_source_snapshot(
        contract.get("source_snapshot"),
        family=family,
        contract_base=contract_pair[0].parent,
    )
    if (
        snapshot_pair[1] != LLAVA7_FAILED_SOURCE_SNAPSHOT_SHA256
        or _descriptor_object(snapshot_pair) != evidence["source_snapshot"]
    ):
        raise SyncError("llava7 failure snapshot descriptor differs from contract")

    validation_pair = _exact_descriptor(
        evidence["validation"],
        base_directory=manifest_path.parent,
        label="llava7 failed validation",
        expected_path=root / "full_validation.json",
    )
    _require_pinned_pair(
        validation_pair,
        LLAVA7_FAILED_VALIDATION_SHA256,
        "llava7 failed validation",
    )
    validation = _load_hashed_json(validation_pair[0], "llava7 failed validation")
    fresh_methods = {methods["base"], methods["exact_control"], methods["candidate"]}
    if (
        validation.get("schema_version") != 1
        or validation.get("status") != "passed"
        or validation.get("stage") != "full"
        or validation.get("campaign_fingerprint") != fingerprint
        or validation.get("expected_samples") != MULTIFAMILY_EXPECTED_SAMPLES
        or validation.get("reproduction") is not None
        or validation.get("quality_gate_inputs")
        != {
            "controls": [methods["base"], methods["exact_control"]],
            "candidate": methods["candidate"],
            "reproduction_oracles_included": False,
        }
        or not isinstance(validation.get("artifacts"), dict)
        or set(validation["artifacts"]) != fresh_methods
        or _validate_failure_descriptor_tree(
            validation,
            base_directory=validation_pair[0].parent,
            label="llava7 failed validation",
        )
        < 12
    ):
        raise SyncError("llava7 failed full validation contract differs")

    selection_pair = _exact_descriptor(
        evidence["selection"],
        base_directory=manifest_path.parent,
        label="llava7 failed selection",
        expected_path=root / "full_selection.json",
    )
    _require_pinned_pair(
        selection_pair,
        LLAVA7_FAILED_SELECTION_SHA256,
        "llava7 failed selection",
    )
    selection = _load_hashed_json(selection_pair[0], "llava7 failed selection")
    _validate_failed_selection(selection, family=family)
    matrix_csv_pair = _exact_descriptor(
        evidence["open_mos_matrix_csv"],
        base_directory=manifest_path.parent,
        label="llava7 failed MOS CSV",
        expected_path=root / "full_mos/matrix.csv",
    )
    matrix_jsonl_pair = _exact_descriptor(
        evidence["open_mos_matrix_jsonl"],
        base_directory=manifest_path.parent,
        label="llava7 failed MOS JSONL",
        expected_path=root / "full_mos/matrix.jsonl",
    )
    _require_pinned_pair(
        matrix_csv_pair,
        LLAVA7_FAILED_MATRIX_CSV_SHA256,
        "llava7 failed MOS CSV",
    )
    _require_pinned_pair(
        matrix_jsonl_pair,
        LLAVA7_FAILED_MATRIX_JSONL_SHA256,
        "llava7 failed MOS JSONL",
    )
    expected_methods = {
        "llava_onevision_original",
        methods["base"],
        methods["exact_control"],
        methods["candidate"],
    }
    metrics = _inspect_failed_full_mos_pair(
        matrix_csv_pair[0],
        matrix_jsonl_pair[0],
        family=family,
        expected_methods=expected_methods,
        judge=judge,
    )
    archive_sample_descriptor = {
        "path": archived.get("sample"),
        "sha256": archived.get("sample_sha256"),
    }
    validation_artifacts = validation["artifacts"]
    sample_descriptors = {
        "llava_onevision_original": archive_sample_descriptor,
        **{
            method: validation_artifacts[method]["sample"]
            for method in fresh_methods
        },
    }
    _validate_failure_matrix_sample_binding(
        matrix_csv_pair[0],
        family=family,
        methods=sorted(expected_methods),
        sample_descriptors=sample_descriptors,
        descriptor_bases={
            "llava_onevision_original": contract_pair[0].parent,
            **{method: validation_pair[0].parent for method in fresh_methods},
        },
        trim_limit=_integer(
            contract_judge.get("trim_char_limit"), "llava7.judge.trim_char_limit"
        ),
    )
    controls = expected_methods - {methods["candidate"]}
    candidate_metrics = metrics[methods["candidate"]]
    if (
        candidate_metrics["open_mos"]
        > max(metrics[method]["open_mos"] for method in controls)
        and float(candidate_metrics["token_f1"])
        > max(float(metrics[method]["token_f1"]) for method in controls)
    ):
        raise SyncError("llava7 failed evidence actually passes its strict dual gate")
    telemetry = validation.get("telemetry")
    if not isinstance(telemetry, dict) or set(telemetry) != fresh_methods:
        raise SyncError("llava7 failed telemetry method set differs")
    for method, row in telemetry.items():
        if not isinstance(row, dict):
            raise SyncError(f"llava7 failed telemetry row {method} is malformed")
        token_f1 = _decimal(row.get("token_f1"), f"llava7.{method}.token_f1")
        assert token_f1 is not None
        if not math.isclose(
            float(token_f1),
            float(metrics[method]["token_f1"]),
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise SyncError("llava7 failed Token-F1 does not reproduce")
    candidate_telemetry = telemetry[methods["candidate"]]
    for field in ("recompute_ratio", "reference_patch_compute_ratio"):
        ratio = _decimal(candidate_telemetry.get(field), f"llava7.candidate.{field}")
        assert ratio is not None
        if not Decimal(0) <= ratio <= Decimal("0.98"):
            raise SyncError("llava7 candidate does not satisfy compute ceilings")
    if entry["candidate_methods"] != [methods["candidate"]]:
        raise SyncError("llava7 failed attempted candidate set differs")
    return {
        **entry,
        "methods": sorted(expected_methods),
        "selection_payload": selection,
        "candidate": None,
    }


def _validate_qwen7_failed_evidence(
    entry: Mapping[str, Any],
    *,
    manifest_path: Path,
    judge: Mapping[str, Any],
) -> dict[str, Any]:
    family = "qwen7"
    evidence = entry["evidence"]
    contract_pair = _exact_descriptor(
        evidence["contract"],
        base_directory=manifest_path.parent,
        label="qwen7 failed contract",
    )
    contract = _load_hashed_json(contract_pair[0], "qwen7 failed contract")
    root = contract_pair[0].parent.parent
    fingerprint = str(contract.get("campaign_fingerprint") or "")
    expected_methods = {
        "archive": QWEN7_FLOOR_FULL_ARCHIVE,
        "fresh": QWEN7_FLOOR_FULL_METHODS[0],
        "all": QWEN7_FLOOR_FULL_METHODS[1],
        "candidate": QWEN7_FLOOR_FULL_METHODS[2],
    }
    checkpoint = contract.get("checkpoint")
    contract_judge = contract.get("judge")
    if (
        contract_pair[1] != QWEN7_FLOOR_FULL_CONTRACT_SHA256
        or fingerprint != QWEN7_FLOOR_FULL_CAMPAIGN_FINGERPRINT
        or fingerprint != entry["campaign_fingerprint"]
        or _canonical_fingerprint(contract) != fingerprint
        or contract.get("campaign_kind") != "qwen7_route_floor_cap48_full_v1"
        or contract.get("benchmark") != "DIVE-Bench"
        or contract.get("task") != LPM_TASK
        or contract.get("expected_full_samples") != MULTIFAMILY_EXPECTED_SAMPLES
        or Path(str(contract.get("output_root") or "")).expanduser().resolve()
        != root
        or contract.get("methods") != expected_methods
        or contract.get("generation") != {"max_new_tokens": 48, "temperature": 0}
        or not isinstance(checkpoint, dict)
        or checkpoint.get("name") != "Qwen/Qwen2.5-VL-7B-Instruct"
        or checkpoint.get("revision") != entry["revisions"]["model"]
        or contract.get("dataset_revision") != entry["revisions"]["dataset"]
        or not isinstance(contract_judge, dict)
        or contract_judge.get("model") != judge["model"]
        or contract_judge.get("revision") != judge["revision"]
        or contract_judge.get("judge_fingerprint") != judge["fingerprint"]
        or entry["revisions"]["judge"] != judge["revision"]
    ):
        raise SyncError("qwen7 failed contract identity/schema differs")
    quality_gate = contract.get("quality_gate")
    if (
        not isinstance(quality_gate, dict)
        or quality_gate.get("candidate") != expected_methods["candidate"]
        or quality_gate.get("primary_metric") != "open_mos"
        or quality_gate.get("secondary_metric") != "token_f1"
        or quality_gate.get("strictly_greater_than_every_control") is not True
        or quality_gate.get("max_recompute_ratio") != Decimal("0.98")
        or quality_gate.get("max_reference_patch_compute_ratio")
        != Decimal("0.98")
        or quality_gate.get("gates")
        != {
            "public": [expected_methods["archive"]],
            "fresh": [expected_methods["fresh"], expected_methods["all"]],
            "combined": [
                expected_methods["archive"],
                expected_methods["fresh"],
                expected_methods["all"],
            ],
        }
    ):
        raise SyncError("qwen7 failed quality-gate contract differs")
    snapshot_pair = _verify_historical_source_snapshot(
        contract.get("source_snapshot"),
        family=family,
        contract_base=contract_pair[0].parent,
    )
    if _descriptor_object(snapshot_pair) != evidence["source_snapshot"]:
        raise SyncError("qwen7 failure snapshot descriptor differs from contract")
    archived = contract.get("archived_full_control")
    screen_prerequisite = contract.get("screen_prerequisite")
    if (
        not isinstance(archived, dict)
        or archived.get("method") != expected_methods["archive"]
        or _validate_failure_descriptor_tree(
            archived,
            base_directory=contract_pair[0].parent,
            label="qwen7 archived public control",
        )
        < 2
        or not isinstance(screen_prerequisite, dict)
        or _validate_failure_descriptor_tree(
            screen_prerequisite,
            base_directory=contract_pair[0].parent,
            label="qwen7 screen prerequisite",
        )
        < 1
    ):
        raise SyncError("qwen7 failed prerequisite evidence differs")

    validation_pair = _exact_descriptor(
        evidence["validation"],
        base_directory=manifest_path.parent,
        label="qwen7 failed validation",
        expected_path=root / "full_validation.json",
    )
    validation = _load_hashed_json(validation_pair[0], "qwen7 failed validation")
    raw_artifacts = validation.get("artifacts")
    fresh_methods = set(QWEN7_FLOOR_FULL_METHODS)
    if (
        validation.get("schema") != "qwen7_floor_cap48_full_validation_v1"
        or validation.get("schema_version") != 1
        or validation.get("status") != "passed"
        or validation.get("stage") != "full"
        or validation.get("campaign_fingerprint") != fingerprint
        or validation.get("expected_samples") != MULTIFAMILY_EXPECTED_SAMPLES
        or validation.get("contract") != _descriptor_object(contract_pair)
        or validation.get("quality_gate_inputs")
        != {
            "public": [expected_methods["archive"]],
            "fresh": [expected_methods["fresh"], expected_methods["all"]],
            "combined": [
                expected_methods["archive"],
                expected_methods["fresh"],
                expected_methods["all"],
            ],
            "candidate": expected_methods["candidate"],
            "actual_only": True,
        }
        or not isinstance(raw_artifacts, dict)
        or set(raw_artifacts) != fresh_methods
        or _validate_failure_descriptor_tree(
            validation,
            base_directory=validation_pair[0].parent,
            label="qwen7 failed validation",
        )
        < 12
    ):
        raise SyncError("qwen7 failed full validation contract differs")
    candidate_telemetry = validation.get("candidate_telemetry")
    if not isinstance(candidate_telemetry, dict):
        raise SyncError("qwen7 failed candidate telemetry is missing")
    for field in ("recompute_ratio", "reference_patch_compute_ratio"):
        ratio = _decimal(candidate_telemetry.get(field), f"qwen7.candidate.{field}")
        assert ratio is not None
        if not Decimal(0) <= ratio <= Decimal("0.98"):
            raise SyncError("qwen7 failed candidate violates compute ceilings")

    matrix_csv_pair = _exact_descriptor(
        evidence["open_mos_matrix_csv"],
        base_directory=manifest_path.parent,
        label="qwen7 failed MOS CSV",
        expected_path=root / "full_mos/matrix.csv",
    )
    matrix_jsonl_pair = _exact_descriptor(
        evidence["open_mos_matrix_jsonl"],
        base_directory=manifest_path.parent,
        label="qwen7 failed MOS JSONL",
        expected_path=root / "full_mos/matrix.jsonl",
    )
    matrix_methods = set(QWEN7_FLOOR_FULL_MOS_METHODS)
    metrics = _inspect_failed_full_mos_pair(
        matrix_csv_pair[0],
        matrix_jsonl_pair[0],
        family=family,
        expected_methods=matrix_methods,
        judge=judge,
    )
    sample_descriptors = {
        expected_methods["archive"]: archived["sample"],
        **{method: raw_artifacts[method]["sample"] for method in fresh_methods},
    }
    _validate_failure_matrix_sample_binding(
        matrix_csv_pair[0],
        family=family,
        methods=QWEN7_FLOOR_FULL_MOS_METHODS,
        sample_descriptors=sample_descriptors,
        descriptor_bases={
            expected_methods["archive"]: contract_pair[0].parent,
            **{method: validation_pair[0].parent for method in fresh_methods},
        },
        trim_limit=_integer(
            contract_judge.get("trim_char_limit"), "qwen7.judge.trim_char_limit"
        ),
    )
    selection_pair = _exact_descriptor(
        evidence["selection"],
        base_directory=manifest_path.parent,
        label="qwen7 failed selection",
    )
    attempt_root = root / ".full_attempts" / f"attempt-{fingerprint[:16]}"
    labels = [
        label
        for label in ("public", "fresh", "combined")
        if selection_pair[0] == attempt_root / f".{label}.unsealed.json"
    ]
    if len(labels) != 1:
        raise SyncError("qwen7 failed selection is not a contracted gate path")
    selection = _load_hashed_json(selection_pair[0], "qwen7 failed selection")
    _validate_failed_selection(selection, family=family)
    controls = set(quality_gate["gates"][labels[0]])
    candidate = expected_methods["candidate"]
    if (
        metrics[candidate]["open_mos"]
        > max(metrics[method]["open_mos"] for method in controls)
        and float(metrics[candidate]["token_f1"])
        > max(float(metrics[method]["token_f1"]) for method in controls)
    ):
        raise SyncError("qwen7 failed evidence actually passes its selected gate")
    if entry["candidate_methods"] != [candidate]:
        raise SyncError("qwen7 failed attempted candidate set differs")
    return {
        **entry,
        "methods": sorted(matrix_methods),
        "selection_payload": selection,
        "candidate": None,
    }


def _validate_promoted_v2_entry(
    entry: Mapping[str, Any],
    *,
    manifest_path: Path,
    leaderboard_rows: Sequence[Mapping[str, str]],
    judge: Mapping[str, Any],
    validated_groups: Mapping[str, set[tuple[Path, str]]],
) -> dict[str, Any]:
    family = str(entry["family"])
    family_contract = MULTIFAMILY_CONTRACTS[family]
    evidence = entry["evidence"]
    completion_schema = entry["completion_schema"]
    selection_pair = _exact_descriptor(
        evidence["selection"],
        base_directory=manifest_path.parent,
        label=f"{family} promoted selection",
    )
    matrix_pair = _exact_descriptor(
        evidence["open_mos_matrix"],
        base_directory=manifest_path.parent,
        label=f"{family} promoted Open-MOS matrix",
    )
    summary_pair = _exact_descriptor(
        evidence["summary_csv"],
        base_directory=manifest_path.parent,
        label=f"{family} promoted summary",
    )
    selection = _load_hashed_json(selection_pair[0], f"{family} promoted selection")
    selected_method = str(selection.get("selected_method") or "").strip()
    candidate = _site_row(
        _candidate_row(leaderboard_rows, selected_method),
        selected_method,
        MULTIFAMILY_EXPECTED_SAMPLES,
    )
    selected_method, methods = validate_strict_multifamily_selection(
        selection,
        candidate,
        family=family,
        base_method=str(family_contract["base_method"]),
        expected_samples=MULTIFAMILY_EXPECTED_SAMPLES,
        completion_schema=(
            str(completion_schema) if completion_schema is not None else None
        ),
    )
    candidates = selection.get("candidate_scores")
    if (
        not isinstance(candidates, dict)
        or entry["candidate_methods"] != sorted(map(str, candidates))
        or entry["selected_method"] != selected_method
    ):
        raise SyncError(f"{family} promoted candidate ledger differs from selection")
    expected_matrix_methods = (
        set(qwen_dual_mos_methods())
        if completion_schema == QWEN_DUAL_COMPLETION_SCHEMA
        else set(methods)
    )
    matrix_means = _inspect_multifamily_matrix(
        matrix_pair[0],
        family=family,
        expected_methods=expected_matrix_methods,
        expected_samples=MULTIFAMILY_EXPECTED_SAMPLES,
        judge=judge,
    )
    score_maps = {
        **selection["baseline_scores"],
        **selection["candidate_scores"],
    }
    family_means = {
        method: matrix_means[method] for method in methods if method in matrix_means
    }
    if not _open_mos_means_match(family_means, score_maps, family=family):
        raise SyncError(f"{family} promoted OpenMOS scores do not reproduce")
    _validate_multifamily_summary(
        summary_pair[0],
        family=family,
        selected_method=selected_method,
        expected_samples=MULTIFAMILY_EXPECTED_SAMPLES,
    )
    if matrix_pair not in validated_groups["open_mos_artifacts"]:
        raise SyncError(f"{family} promoted matrix is absent from release sources")
    if summary_pair not in validated_groups["summary_csv"]:
        raise SyncError(f"{family} promoted summary is absent from release sources")

    completion_pair: tuple[Path, str] | None = None
    if completion_schema is None:
        if evidence.get("completion") is not None or family != "route31":
            raise SyncError("only Route31 may use null promoted completion")
        artifacts = validate_selection_artifacts(
            selection, selection_path=selection_pair[0]
        )
        _, _, fingerprint = selection_identity(selection.get("artifact_provenance"))
        if fingerprint != entry["campaign_fingerprint"]:
            raise SyncError("Route31 promoted fingerprint differs")
        if (
            artifacts["open_mos_matrix"] != matrix_pair
            or artifacts["summary_route31_candidate"] != summary_pair
        ):
            raise SyncError("Route31 promoted evidence is not selection-bound")
        telemetry = _load_hashed_json(
            artifacts["telemetry_contract"][0], "Route31 telemetry"
        )
        telemetry_methods = telemetry.get("methods")
        if (
            telemetry.get("schema_version") != 1
            or telemetry.get("status") != "passed"
            or telemetry.get("limits")
            != {
                "max_recompute_ratio": Decimal("0.98"),
                "max_reference_patch_compute_ratio": Decimal("0.98"),
            }
            or not isinstance(telemetry_methods, dict)
            or set(telemetry_methods) != set(ROUTE31_METHODS[1:])
        ):
            raise SyncError("Route31 promoted telemetry differs")
        winner_telemetry = telemetry_methods[ROUTE31_WINNER_METHOD]
        if (
            winner_telemetry.get("role") != "candidate"
            or winner_telemetry.get("passes_recompute_caps") is not True
            or winner_telemetry.get("effective_max_new_tokens_counts")
            != {"128": 317, "31": 317}
            or winner_telemetry.get("question_route_counts")
            != {"ocr": 317, "subtitle": 317}
        ):
            raise SyncError("Route31 promoted route/cap telemetry differs")
        for field, candidate_field in (
            ("mean_recompute_ratio", "recompute_ratio"),
            ("reference_patch_compute_ratio", "reference_recompute_ratio"),
            ("mean_effective_fps", "effective_fps"),
            ("mean_throughput_fps", "throughput_fps"),
        ):
            if not _published_matches(
                winner_telemetry.get(field), candidate.get(candidate_field)
            ):
                raise SyncError(f"Route31 promoted {field} differs from CSV")
        secondary = {
            **selection.get("baseline_secondary_scores", {}),
            **selection.get("candidate_secondary_scores", {}),
        }
        summary_names = {
            ROUTE31_PUBLIC_METHOD: "summary_archived_public",
            ROUTE31_BASE_METHOD: "summary_route31_base",
            ROUTE31_EXACT_METHOD: "summary_route31_exact",
            ROUTE31_WINNER_METHOD: "summary_route31_candidate",
        }
        if set(secondary) != set(summary_names):
            raise SyncError("Route31 promoted Token-F1 method set differs")
        for method, artifact_name in summary_names.items():
            row = _summary_row(
                artifacts[artifact_name][0],
                family="route31",
                method=method,
                expected_samples=MULTIFAMILY_EXPECTED_SAMPLES,
            )
            _assert_secondary_score(row, secondary[method], f"route31.{method}")
        evaluation = _load_hashed_json(
            artifacts["evaluation_contract"][0], "Route31 contract"
        )
        revisions = entry["revisions"]
        if (
            evaluation.get("model_revision") != revisions["model"]
            or evaluation.get("dataset_revision") != revisions["dataset"]
            or evaluation.get("judge_revision") != revisions["judge"]
        ):
            raise SyncError("Route31 promoted revisions differ")
    else:
        completion_pair = _exact_descriptor(
            evidence["completion"],
            base_directory=manifest_path.parent,
            label=f"{family} promoted completion",
        )
        _validate_multifamily_completion(
            completion_pair[0],
            entry=entry,
            selection=selection,
            selection_pair=selection_pair,
            matrix_pair=matrix_pair,
            summary_pair=summary_pair,
            judge=judge,
        )
    return {
        **entry,
        "methods": list(methods),
        "selection_payload": selection,
        "candidate": candidate,
        "selection": _descriptor_object(selection_pair),
        "completion": (
            _descriptor_object(completion_pair) if completion_pair is not None else None
        ),
        "open_mos_matrix": _descriptor_object(matrix_pair),
        "summary_csv": _descriptor_object(summary_pair),
    }


def _validate_multifamily_release_v2_binding(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    leaderboard_csv: Path,
    leaderboard_rows: Sequence[Mapping[str, str]],
    metadata: Mapping[str, Any],
    metadata_path: Path,
) -> list[dict[str, Any]]:
    expected_header = {
        "schema_version",
        "schema",
        "benchmark",
        "task",
        "expected_samples",
        "baseline_lpm_rows",
        "expected_lpm_rows",
        "expected_highmotion_rows",
        "judge",
        "promotion_summary",
        "families",
    }
    if (
        set(manifest) != expected_header
        or manifest.get("schema_version") != 2
        or manifest.get("schema") != MULTIFAMILY_RELEASE_V2_SCHEMA
        or manifest.get("benchmark") != "DIVE-Bench"
        or manifest.get("task") != LPM_TASK
        or manifest.get("expected_samples") != MULTIFAMILY_EXPECTED_SAMPLES
        or manifest.get("baseline_lpm_rows") != MULTIFAMILY_BASELINE_LPM_ROWS
        or manifest.get("expected_highmotion_rows")
        != MULTIFAMILY_EXPECTED_HIGHMOTION_ROWS
        or manifest.get("judge") != MULTIFAMILY_JUDGE
    ):
        raise SyncError("v2 multi-family release header/schema differs")
    judge = manifest["judge"]
    entries = manifest.get("families")
    if (
        not isinstance(entries, list)
        or len(entries) != len(MULTIFAMILY_RELEASE_V2_FAMILIES)
        or [entry.get("family") for entry in entries if isinstance(entry, dict)]
        != list(MULTIFAMILY_RELEASE_V2_FAMILIES)
    ):
        raise SyncError("v2 release requires four ordered family audits")
    entry_fields = {
        "family",
        "status",
        "reason_code",
        "base_method",
        "replaces_method",
        "candidate_methods",
        "selected_method",
        "selection_schema",
        "completion_schema",
        "campaign_fingerprint",
        "revisions",
        "evidence",
    }
    passed_evidence_fields = {
        "kind",
        "selection",
        "completion",
        "open_mos_matrix",
        "summary_csv",
    }
    failed_evidence_fields = {
        "kind",
        "contract",
        "source_snapshot",
        "validation",
        "selection",
        "open_mos_matrix_csv",
        "open_mos_matrix_jsonl",
    }
    occupied_candidates: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != entry_fields:
            raise SyncError("v2 family audit closed schema differs")
        family = str(entry["family"])
        contract = MULTIFAMILY_CONTRACTS[family]
        candidates = entry.get("candidate_methods")
        revisions = entry.get("revisions")
        evidence = entry.get("evidence")
        if (
            entry.get("base_method") != contract["base_method"]
            or entry.get("replaces_method") != contract["replaces_method"]
            or entry.get("selection_schema") != contract["selection_schema"]
            or not isinstance(candidates, list)
            or not candidates
            or candidates != sorted(set(map(str, candidates)))
            or occupied_candidates.intersection(candidates)
            or SHA256_PATTERN.fullmatch(
                str(entry.get("campaign_fingerprint") or "")
            )
            is None
            or not isinstance(revisions, dict)
            or set(revisions) != {"model", "dataset", "judge"}
            or any(
                IMMUTABLE_REVISION_PATTERN.fullmatch(str(value)) is None
                for value in revisions.values()
            )
            or revisions["judge"] != judge["revision"]
            or not isinstance(evidence, dict)
        ):
            raise SyncError(f"{family} v2 audit identity differs")
        occupied_candidates.update(candidates)
        if entry["status"] == MULTIFAMILY_RELEASE_V2_PROMOTED:
            allowed = contract["completion_schema"]
            completion_schema = entry["completion_schema"]
            completion_allowed = (
                completion_schema in allowed
                if isinstance(allowed, frozenset)
                else completion_schema == allowed
            )
            if (
                entry["reason_code"] != MULTIFAMILY_RELEASE_V2_PASSED_REASON
                or entry["selected_method"] not in candidates
                or not completion_allowed
                or set(evidence) != passed_evidence_fields
                or evidence.get("kind")
                != MULTIFAMILY_RELEASE_V2_PASSED_EVIDENCE
            ):
                raise SyncError(f"{family} promoted audit differs")
            for key in ("selection", "open_mos_matrix", "summary_csv"):
                _exact_descriptor(
                    evidence[key],
                    base_directory=manifest_path.parent,
                    label=f"{family}.{key}",
                )
            if completion_schema is None:
                if evidence.get("completion") is not None:
                    raise SyncError("Route31 promoted completion must be null")
            else:
                _exact_descriptor(
                    evidence["completion"],
                    base_directory=manifest_path.parent,
                    label=f"{family}.completion",
                )
        elif entry["status"] == MULTIFAMILY_RELEASE_V2_NOT_PROMOTED:
            if (
                family not in {"llava7", "qwen7"}
                or entry["reason_code"] != MULTIFAMILY_RELEASE_V2_FAILED_REASON
                or entry["selected_method"] is not None
                or entry["completion_schema"] is not None
                or set(evidence) != failed_evidence_fields
                or evidence.get("kind")
                != MULTIFAMILY_RELEASE_V2_FAILED_EVIDENCE
            ):
                raise SyncError(f"{family} not-promoted audit differs")
            for key in failed_evidence_fields - {"kind"}:
                _exact_descriptor(
                    evidence[key],
                    base_directory=manifest_path.parent,
                    label=f"{family}.{key}",
                )
        else:
            raise SyncError(f"{family} has unknown v2 qualification status")

    promoted = [
        entry
        for entry in entries
        if entry["status"] == MULTIFAMILY_RELEASE_V2_PROMOTED
    ]
    failed = [
        entry
        for entry in entries
        if entry["status"] == MULTIFAMILY_RELEASE_V2_NOT_PROMOTED
    ]
    replacements = [
        entry["replaces_method"]
        for entry in promoted
        if entry["replaces_method"] is not None
    ]
    expected_summary = {
        "promoted_families": [entry["family"] for entry in promoted],
        "not_promoted_families": [entry["family"] for entry in failed],
        "published_grt_methods": [entry["selected_method"] for entry in promoted],
        "replacement_methods": replacements,
        "promoted_count": len(promoted),
        "replacement_count": len(set(replacements)),
    }
    summary = manifest.get("promotion_summary")
    expected_lpm = (
        MULTIFAMILY_BASELINE_LPM_ROWS
        - len(set(replacements))
        + len(promoted)
    )
    if (
        not isinstance(summary, dict)
        or set(summary) != set(expected_summary)
        or summary != expected_summary
        or manifest.get("expected_lpm_rows") != expected_lpm
        or expected_lpm not in {28, 29}
        or entries[0]["status"] != MULTIFAMILY_RELEASE_V2_PROMOTED
        or entries[1]["status"] != MULTIFAMILY_RELEASE_V2_NOT_PROMOTED
        or entries[2]["status"] != MULTIFAMILY_RELEASE_V2_PROMOTED
    ):
        raise SyncError("v2 promotion ledger/count does not recompute")

    release = metadata.get("artifact_provenance")
    release_fields = {
        "schema_version",
        "leaderboard_csv",
        "leaderboard_markdown",
        "source_artifacts",
        "verified_grt_selection",
        "grt_release_manifest",
        "legacy_open_mos_completion",
        "grt_family_audits",
        "verified_grt_selections",
        "not_promoted_grt_families",
    }
    if (
        not isinstance(release, dict)
        or set(release) != release_fields
        or release.get("schema_version") != 3
        or release.get("verified_grt_selection") is not None
        or release.get("grt_family_audits") != entries
        or release.get("verified_grt_selections") != promoted
        or release.get("not_promoted_grt_families") != failed
    ):
        raise SyncError("v2 release metadata audit partition differs")
    metadata_directory = metadata_path.parent
    _exact_descriptor(
        release["leaderboard_csv"],
        base_directory=metadata_directory,
        label="v2 release leaderboard_csv",
        expected_path=leaderboard_csv,
    )
    _exact_descriptor(
        release["leaderboard_markdown"],
        base_directory=metadata_directory,
        label="v2 release leaderboard_markdown",
    )
    _exact_descriptor(
        release["grt_release_manifest"],
        base_directory=metadata_directory,
        label="v2 release manifest",
        expected_path=manifest_path,
    )
    source_groups = release.get("source_artifacts")
    if not isinstance(source_groups, dict) or set(source_groups) != {
        "summary_csv",
        "open_mos_artifacts",
        "api_summaries",
    }:
        raise SyncError("v2 release source_artifacts schema differs")
    validated_groups: dict[str, set[tuple[Path, str]]] = {}
    for group in ("summary_csv", "open_mos_artifacts", "api_summaries"):
        descriptors = source_groups[group]
        if not isinstance(descriptors, list):
            raise SyncError(f"v2 source_artifacts.{group} must be a list")
        pairs = [
            _exact_descriptor(
                descriptor,
                base_directory=metadata_directory,
                label=f"v2 source_artifacts.{group}[{index}]",
            )
            for index, descriptor in enumerate(descriptors)
        ]
        if len(set(pairs)) != len(pairs):
            raise SyncError(f"v2 source_artifacts.{group} contains duplicates")
        validated_groups[group] = set(pairs)
        raw_paths = metadata.get(group)
        if not isinstance(raw_paths, list) or {
            _artifact_path(path, metadata_directory, f"v2 metadata {group}")
            for path in raw_paths
        } != {pair[0] for pair in pairs}:
            raise SyncError(f"v2 metadata {group} paths differ from descriptors")
    failed_evidence_pairs = {
        _exact_descriptor(
            entry["evidence"][key],
            base_directory=manifest_path.parent,
            label=f"{entry['family']}.failure_source_exclusion.{key}",
        )
        for entry in failed
        for key in (
            "contract",
            "source_snapshot",
            "validation",
            "selection",
            "open_mos_matrix_csv",
            "open_mos_matrix_jsonl",
        )
    }
    if failed_evidence_pairs.intersection(
        set().union(*validated_groups.values())
    ):
        raise SyncError("v2 failure evidence must remain outside MOS source groups")
    if (
        metadata.get("open_mos_judge") != judge["model"]
        or metadata.get("open_mos_judge_provenance")
        != {
            "model": judge["model"],
            "revision": judge["revision"],
            "judge_fingerprint": judge["fingerprint"],
        }
    ):
        raise SyncError("v2 metadata judge identity differs")
    route31_evidence = entries[0]["evidence"]
    validate_legacy_open_mos_release_binding(
        release.get("legacy_open_mos_completion"),
        metadata_directory=metadata_directory,
        judge=judge,
        open_mos_sources=validated_groups["open_mos_artifacts"],
        open_mos_by_method=metadata.get("open_mos"),
        route31_entry={
            "selection": route31_evidence["selection"],
            "open_mos_matrix": route31_evidence["open_mos_matrix"],
        },
    )

    verified: list[dict[str, Any]] = []
    occupied_methods: set[str] = set()
    for entry in entries:
        family = str(entry["family"])
        if entry["status"] == MULTIFAMILY_RELEASE_V2_PROMOTED:
            item = _validate_promoted_v2_entry(
                entry,
                manifest_path=manifest_path,
                leaderboard_rows=leaderboard_rows,
                judge=judge,
                validated_groups=validated_groups,
            )
        elif family == "llava7":
            item = _validate_llava7_failed_evidence(
                entry,
                manifest_path=manifest_path,
                judge=judge,
            )
        else:
            item = _validate_qwen7_failed_evidence(
                entry,
                manifest_path=manifest_path,
                judge=judge,
            )
        methods = set(item["methods"])
        if occupied_methods.intersection(methods):
            raise SyncError("v2 family evidence method sets overlap")
        occupied_methods.update(methods)
        verified.append(item)
    if len({item["revisions"]["dataset"] for item in verified}) != 1:
        raise SyncError("v2 family dataset revisions differ")
    return verified


def validate_multifamily_release_binding(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    leaderboard_csv: Path,
    leaderboard_rows: Sequence[Mapping[str, str]],
    metadata: Mapping[str, Any],
    metadata_path: Path,
) -> list[dict[str, Any]]:
    if manifest.get("schema_version") == 2 or manifest.get("schema") == (
        MULTIFAMILY_RELEASE_V2_SCHEMA
    ):
        return _validate_multifamily_release_v2_binding(
            manifest,
            manifest_path=manifest_path,
            leaderboard_csv=leaderboard_csv,
            leaderboard_rows=leaderboard_rows,
            metadata=metadata,
            metadata_path=metadata_path,
        )
    return _validate_multifamily_release_v1_binding(
        manifest,
        manifest_path=manifest_path,
        leaderboard_csv=leaderboard_csv,
        leaderboard_rows=leaderboard_rows,
        metadata=metadata,
        metadata_path=metadata_path,
    )


def _rank_key(row: Mapping[str, Any], primary: str) -> tuple[Any, Any, str]:
    primary_value = _decimal(
        row.get(primary), f"{row.get('method')}.{primary}", required=False
    )
    secondary = _decimal(
        row.get("token_f1"), f"{row.get('method')}.token_f1", required=False
    )
    return (
        -(primary_value if primary_value is not None else Decimal(-1)),
        -(secondary if secondary is not None else Decimal(-1)),
        str(row.get("model") or row.get("method") or ""),
    )


def rerank(rows: Sequence[Mapping[str, Any]], primary: str) -> list[dict[str, Any]]:
    ranked = [
        dict(row) for row in sorted(rows, key=lambda row: _rank_key(row, primary))
    ]
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def _validate_track(
    rows: Sequence[Mapping[str, Any]],
    *,
    track: str,
    expected_count: int,
    expected_samples: int,
    primary: str,
) -> None:
    if len(rows) != expected_count:
        raise SyncError(f"{track} contains {len(rows)} rows; expected {expected_count}")
    methods: set[str] = set()
    for row in rows:
        method = str(row.get("method") or "").strip()
        if not method:
            raise SyncError(f"{track} contains a row with no method")
        if method in methods:
            raise SyncError(f"duplicate method/task pair: {method}:{track}")
        methods.add(method)
        if _integer(row.get("samples"), f"{method}.samples") != expected_samples:
            raise SyncError(f"{method}:{track} has an unexpected sample count")
    expected_ranking = rerank(rows, primary)
    actual_methods = [str(row.get("method")) for row in rows]
    expected_methods = [str(row.get("method")) for row in expected_ranking]
    actual_ranks = [
        _integer(row.get("rank"), f"{row.get('method')}.rank") for row in rows
    ]
    if actual_methods != expected_methods or actual_ranks != list(
        range(1, expected_count + 1)
    ):
        raise SyncError(
            f"{track} rows are not contiguously ranked by {primary}, then token_f1"
        )


def _metadata_values(
    metadata: Mapping[str, Any] | None,
    *,
    generated_at: str | None,
    source_artifacts: int | None,
    open_mos_judge: str | None,
    existing_judge: str,
) -> tuple[str, int, str]:
    metadata = metadata or {}
    timestamp = (
        generated_at
        or str(
            metadata.get("generated_at") or metadata.get("generatedAt") or ""
        ).strip()
    )
    if not timestamp:
        raise SyncError("generated_at is required via metadata or --generated-at")
    try:
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SyncError(
            f"generated_at is not an ISO-8601 timestamp: {timestamp!r}"
        ) from exc

    count_value: Any = source_artifacts
    if count_value is None:
        for key in ("source_artifact_count", "sourceArtifacts", "source_artifacts"):
            if isinstance(metadata.get(key), (int, Decimal, str)):
                count_value = metadata[key]
                break
            if isinstance(metadata.get(key), list):
                count_value = len(
                    dict.fromkeys(str(value) for value in metadata[key] if str(value))
                )
                break
    if count_value is None:
        artifact_paths: list[str] = []
        for key in ("summary_csv", "open_mos_artifacts", "api_summaries"):
            values = metadata.get(key, [])
            if isinstance(values, list):
                artifact_paths.extend(str(value) for value in values if str(value))
        count_value = len(dict.fromkeys(artifact_paths))
    count = _integer(count_value, "sourceArtifacts")
    if count <= 0:
        raise SyncError("sourceArtifacts must be positive")

    judge_value: Any = open_mos_judge or metadata.get("open_mos_judge")
    if not judge_value:
        judge_value = metadata.get("open_mos_judges")
    if isinstance(judge_value, list):
        judge_value = ", ".join(str(value) for value in judge_value if str(value))
    judge = str(judge_value or existing_judge).strip()
    if not judge:
        raise SyncError("Open MOS judge must be declared")
    return timestamp, count, judge


def build_release_payload(
    site_payload: Mapping[str, Any],
    leaderboard_rows: Sequence[Mapping[str, str]],
    selection: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None = None,
    expected_lpm_count: int = 27,
    expected_highmotion_count: int = 3,
    expected_lpm_samples: int = 634,
    expected_highmotion_samples: int = 1000,
    base_method: str = DEFAULT_BASE_METHOD,
    generated_at: str | None = None,
    source_artifacts: int | None = None,
    open_mos_judge: str | None = None,
) -> dict[str, Any]:
    payload = deepcopy(dict(site_payload))
    tracks = payload.get("tracks")
    if not isinstance(tracks, dict):
        raise SyncError("site payload has no tracks object")
    lpm_before = tracks.get("lpm")
    highmotion_before = deepcopy(tracks.get("highmotion"))
    if not isinstance(lpm_before, list) or not isinstance(highmotion_before, list):
        raise SyncError("site payload must contain lpm and highmotion arrays")
    _validate_track(
        highmotion_before,
        track="highmotion",
        expected_count=expected_highmotion_count,
        expected_samples=expected_highmotion_samples,
        primary="grid_acc",
    )

    selected_method = str(selection.get("selected_method") or "").strip()
    source = _candidate_row(leaderboard_rows, selected_method)
    verified_candidate = _site_row(source, selected_method, expected_lpm_samples)
    validate_selection(selection, verified_candidate, expected_lpm_samples)
    updated_lpm = _release_lpm_rows(
        leaderboard_rows,
        expected_count=expected_lpm_count,
        expected_samples=expected_lpm_samples,
        strict_methods=(selected_method,),
    )
    candidate = next(row for row in updated_lpm if row["method"] == selected_method)
    verified_candidate["rank"] = candidate["rank"]
    if candidate != verified_candidate:
        raise SyncError("selected winner differs from its verified release CSV row")

    base_rows = [row for row in updated_lpm if row.get("method") == base_method]
    if len(base_rows) != 1:
        raise SyncError(
            f"release CSV must contain exactly one base LPM row {base_method!r}"
        )
    base = base_rows[0]
    if candidate["open_mos"] <= _decimal(
        base.get("open_mos"), f"{base_method}.open_mos"
    ):
        raise SyncError(
            "verified GRT winner does not strictly exceed the base Open MOS"
        )
    if candidate["token_f1"] <= _decimal(
        base.get("token_f1"), f"{base_method}.token_f1"
    ):
        raise SyncError(
            "verified GRT winner does not strictly exceed the base Token F1"
        )

    published_candidate = candidate

    timestamp, count, judge = _metadata_values(
        metadata,
        generated_at=generated_at,
        source_artifacts=source_artifacts,
        open_mos_judge=open_mos_judge,
        existing_judge=str(payload.get("openMosJudge") or ""),
    )
    tracks["lpm"] = updated_lpm
    payload["primaryGrtMethod"] = selected_method
    payload["generatedAt"] = timestamp
    payload["sourceArtifacts"] = count
    payload["openMosJudge"] = judge

    if tracks.get("highmotion") != highmotion_before:
        raise SyncError("High-Motion snapshot changed during LPM synchronization")
    _validate_track(
        tracks["lpm"],
        track="lpm",
        expected_count=expected_lpm_count,
        expected_samples=expected_lpm_samples,
        primary="open_mos",
    )
    _validate_track(
        tracks["highmotion"],
        track="highmotion",
        expected_count=expected_highmotion_count,
        expected_samples=expected_highmotion_samples,
        primary="grid_acc",
    )
    for field in LPM_NUMERIC_FIELDS:
        if published_candidate[field] != verified_candidate[field]:
            raise SyncError(f"numeric parity failure for selected row field {field}")
    return payload


def build_multifamily_release_payload(
    site_payload: Mapping[str, Any],
    leaderboard_rows: Sequence[Mapping[str, str]],
    verified_families: Sequence[Mapping[str, Any]],
    *,
    metadata: Mapping[str, Any],
    expected_lpm_count: int,
    expected_highmotion_count: int,
    expected_lpm_samples: int = 634,
    expected_highmotion_samples: int = 1000,
    generated_at: str | None = None,
    source_artifacts: int | None = None,
    open_mos_judge: str | None = None,
) -> dict[str, Any]:
    is_v2 = all(
        item.get("status")
        in {
            MULTIFAMILY_RELEASE_V2_PROMOTED,
            MULTIFAMILY_RELEASE_V2_NOT_PROMOTED,
        }
        for item in verified_families
    )
    promoted_families = [
        item
        for item in verified_families
        if not is_v2 or item.get("status") == MULTIFAMILY_RELEASE_V2_PROMOTED
    ]
    expected_v2_count = (
        MULTIFAMILY_BASELINE_LPM_ROWS
        - len(
            {
                str(item["replaces_method"])
                for item in promoted_families
                if item.get("replaces_method") is not None
            }
        )
        + len(promoted_families)
    )
    if (
        expected_lpm_count
        != (expected_v2_count if is_v2 else MULTIFAMILY_EXPECTED_LPM_ROWS)
        or (is_v2 and expected_lpm_count not in {28, 29})
        or expected_highmotion_count != MULTIFAMILY_EXPECTED_HIGHMOTION_ROWS
        or expected_lpm_samples != MULTIFAMILY_EXPECTED_SAMPLES
        or expected_highmotion_samples != MULTIFAMILY_EXPECTED_HIGHMOTION_SAMPLES
    ):
        raise SyncError(
            "multi-family payload row/sample counts differ from its release schema"
        )
    if (
        [str(item.get("family")) for item in verified_families]
        != list(MULTIFAMILY_CONTRACTS)
        or (is_v2 and len(promoted_families) not in {2, 3})
    ):
        raise SyncError(
            "multi-family payload requires four ordered contracted families"
        )
    payload = deepcopy(dict(site_payload))
    tracks = payload.get("tracks")
    if not isinstance(tracks, dict):
        raise SyncError("site payload has no tracks object")
    lpm_before = tracks.get("lpm")
    highmotion_before = deepcopy(tracks.get("highmotion"))
    if not isinstance(lpm_before, list) or not isinstance(highmotion_before, list):
        raise SyncError("site payload must contain lpm and highmotion arrays")
    replacements = {
        str(item["replaces_method"])
        for item in promoted_families
        if item.get("replaces_method") is not None
    }
    _validate_track(
        highmotion_before,
        track="highmotion",
        expected_count=expected_highmotion_count,
        expected_samples=expected_highmotion_samples,
        primary="grid_acc",
    )
    if (
        sum(
            str(row.get("method") or "") == LEGACY_GRT_METHOD
            for row in highmotion_before
        )
        != 1
    ):
        raise SyncError("legacy 0.5B GRT must remain exactly once in High-Motion")
    selected_methods = [str(item["selected_method"]) for item in promoted_families]
    updated_lpm = _release_lpm_rows(
        leaderboard_rows,
        expected_count=expected_lpm_count,
        expected_samples=expected_lpm_samples,
        strict_methods=selected_methods,
    )
    release_by_method = {str(row["method"]): row for row in updated_lpm}
    family_metadata: dict[str, dict[str, Any]] = {}
    for item in verified_families:
        family = str(item["family"])
        base_method = str(item["base_method"])
        base = release_by_method.get(base_method)
        if base is None:
            raise SyncError(
                f"{family} base method {base_method!r} is absent from release CSV"
            )
        promoted = not is_v2 or (
            item.get("status") == MULTIFAMILY_RELEASE_V2_PROMOTED
        )
        if promoted:
            selected = str(item["selected_method"])
            candidate = release_by_method.get(selected)
            verified_candidate = dict(item["candidate"])
            if candidate is not None:
                verified_candidate["rank"] = candidate["rank"]
            if candidate is None or candidate != verified_candidate:
                raise SyncError(
                    f"{family} winner differs from its verified release CSV row"
                )
            if candidate["open_mos"] <= _decimal(
                base.get("open_mos"), f"{base_method}.open_mos"
            ):
                raise SyncError(
                    f"{family} winner does not strictly exceed base Open MOS"
                )
            if candidate["token_f1"] <= _decimal(
                base.get("token_f1"), f"{base_method}.token_f1"
            ):
                raise SyncError(
                    f"{family} winner does not strictly exceed base Token F1"
                )
        else:
            selected = None
            leaked = set(map(str, item.get("candidate_methods", []))).intersection(
                release_by_method
            )
            if leaked:
                raise SyncError(
                    f"{family} not-promoted candidates leaked into leaderboard rows"
                )
        if is_v2:
            family_metadata[family] = {
                "status": str(item["status"]),
                "reasonCode": str(item["reason_code"]),
                "method": selected,
                "baseMethod": base_method,
                "campaignFingerprint": str(item["campaign_fingerprint"]),
            }
        else:
            family_metadata[family] = {
                "method": selected,
                "baseMethod": base_method,
                "campaignFingerprint": str(item["campaign_fingerprint"]),
            }

    timestamp, count, judge = _metadata_values(
        metadata,
        generated_at=generated_at,
        source_artifacts=source_artifacts,
        open_mos_judge=open_mos_judge,
        existing_judge=str(payload.get("openMosJudge") or ""),
    )
    tracks["lpm"] = updated_lpm
    payload["primaryGrtMethod"] = family_metadata["route31"]["method"]
    payload["grtMethods"] = [
        family_metadata[str(item["family"])]["method"]
        for item in promoted_families
    ]
    payload["grtFamilies"] = (
        [
            {"family": family, **family_metadata[family]}
            for family in MULTIFAMILY_CONTRACTS
        ]
        if is_v2
        else family_metadata
    )
    payload["generatedAt"] = timestamp
    payload["sourceArtifacts"] = count
    payload["openMosJudge"] = judge

    if tracks.get("highmotion") != highmotion_before:
        raise SyncError(
            "High-Motion snapshot changed during multi-family synchronization"
        )
    _validate_track(
        tracks["lpm"],
        track="lpm",
        expected_count=expected_lpm_count,
        expected_samples=expected_lpm_samples,
        primary="open_mos",
    )
    _validate_track(
        tracks["highmotion"],
        track="highmotion",
        expected_count=expected_highmotion_count,
        expected_samples=expected_highmotion_samples,
        primary="grid_acc",
    )
    final_methods = {str(row.get("method") or "") for row in tracks["lpm"]}
    for family, details in family_metadata.items():
        if details["baseMethod"] not in final_methods:
            raise SyncError(f"{family} base was lost during synchronization")
        if details["method"] is not None and details["method"] not in final_methods:
            raise SyncError(f"{family} winner was lost during synchronization")
    if replacements.intersection(final_methods):
        raise SyncError("a replaced legacy GRT row remains in the final LPM track")
    return payload


def write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--site-data", type=Path, default=Path("data/leaderboard.js"))
    command.add_argument("--leaderboard-csv", type=Path, required=True)
    command.add_argument(
        "--metadata-json",
        type=Path,
        required=True,
        help="Hashed leaderboard_sources.json emitted by the release builder.",
    )
    command.add_argument(
        "--selection-json",
        type=Path,
        help=(
            "Verified legacy hf_full/selection.json or Route31 hf_route31_full/selection.json; "
            "may instead be embedded in metadata."
        ),
    )
    command.add_argument(
        "--grt-release-manifest",
        type=Path,
        help=(
            "Four-family hashed release manifest emitted alongside artifact schema v2. "
            "Mutually exclusive with --selection-json."
        ),
    )
    command.add_argument("--generated-at")
    command.add_argument("--source-artifacts", type=int)
    command.add_argument("--open-mos-judge")
    command.add_argument("--base-method", default=DEFAULT_BASE_METHOD)
    command.add_argument("--expected-lpm-count", type=int)
    command.add_argument("--expected-highmotion-count", type=int)
    command.add_argument("--expected-lpm-samples", type=int, default=634)
    command.add_argument("--expected-highmotion-samples", type=int, default=1000)
    command.add_argument("--output", type=Path, required=True)
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.selection_json and args.grt_release_manifest:
            raise SyncError(
                "use either --selection-json or --grt-release-manifest, not both"
            )
        site_payload = load_site_payload(args.site_data)
        leaderboard_rows, _ = _read_csv(args.leaderboard_csv)
        metadata = _load_json(args.metadata_json)
        if args.grt_release_manifest:
            manifest = _load_json(args.grt_release_manifest)
            manifest_lpm = _integer(
                manifest.get("expected_lpm_rows"), "manifest.expected_lpm_rows"
            )
            manifest_highmotion = _integer(
                manifest.get("expected_highmotion_rows"),
                "manifest.expected_highmotion_rows",
            )
            if (
                args.expected_lpm_count is not None
                and args.expected_lpm_count != manifest_lpm
            ):
                raise SyncError(
                    "--expected-lpm-count differs from the release manifest"
                )
            if (
                args.expected_highmotion_count is not None
                and args.expected_highmotion_count != manifest_highmotion
            ):
                raise SyncError(
                    "--expected-highmotion-count differs from the release manifest"
                )
            if args.expected_lpm_samples != 634:
                raise SyncError(
                    "multi-family release requires --expected-lpm-samples=634"
                )
            if (
                args.expected_highmotion_samples
                != MULTIFAMILY_EXPECTED_HIGHMOTION_SAMPLES
            ):
                raise SyncError(
                    "multi-family release requires --expected-highmotion-samples=1000"
                )
            verified_families = validate_multifamily_release_binding(
                manifest,
                manifest_path=args.grt_release_manifest,
                leaderboard_csv=args.leaderboard_csv,
                leaderboard_rows=leaderboard_rows,
                metadata=metadata,
                metadata_path=args.metadata_json,
            )
            payload = build_multifamily_release_payload(
                site_payload,
                leaderboard_rows,
                verified_families,
                metadata=metadata,
                expected_lpm_count=manifest_lpm,
                expected_highmotion_count=manifest_highmotion,
                expected_lpm_samples=args.expected_lpm_samples,
                expected_highmotion_samples=args.expected_highmotion_samples,
                generated_at=args.generated_at,
                source_artifacts=args.source_artifacts,
                open_mos_judge=args.open_mos_judge,
            )
        else:
            selection = _load_json(args.selection_json) if args.selection_json else None
            if selection is None:
                selection = _selection_from_metadata(metadata)
            if selection is None:
                raise SyncError(
                    "a full selection is required via --selection-json or metadata"
                )
            validate_release_binding(
                selection,
                selection_path=args.selection_json,
                leaderboard_csv=args.leaderboard_csv,
                metadata=metadata,
                metadata_path=args.metadata_json,
            )
            payload = build_release_payload(
                site_payload,
                leaderboard_rows,
                selection,
                metadata=metadata,
                expected_lpm_count=args.expected_lpm_count or 27,
                expected_highmotion_count=args.expected_highmotion_count or 3,
                expected_lpm_samples=args.expected_lpm_samples,
                expected_highmotion_samples=args.expected_highmotion_samples,
                base_method=args.base_method,
                generated_at=args.generated_at,
                source_artifacts=args.source_artifacts,
                open_mos_judge=args.open_mos_judge,
            )
        rendered = render_site_payload(payload)
        # Reload before publish so serializer regressions fail without touching output.
        with tempfile.TemporaryDirectory(
            prefix="dive-leaderboard-validate-"
        ) as directory:
            validation_path = Path(directory) / "leaderboard.js"
            validation_path.write_text(rendered, encoding="utf-8")
            if load_site_payload(validation_path) != payload:
                raise SyncError("rendered leaderboard did not round-trip exactly")
        write_atomic(args.output, rendered)
    except SyncError as exc:
        print(f"[DIVE_SITE_SYNC] status=failed reason={exc}", file=sys.stderr)
        return 1
    print(
        "[DIVE_SITE_SYNC] status=passed "
        f"method={payload['primaryGrtMethod']} "
        f"grt={len(payload.get('grtMethods', [payload['primaryGrtMethod']]))} "
        f"lpm={len(payload['tracks']['lpm'])} "
        f"highmotion={len(payload['tracks']['highmotion'])} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
