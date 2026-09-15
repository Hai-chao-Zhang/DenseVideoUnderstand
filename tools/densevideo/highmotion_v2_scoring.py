"""Position-preserving scoring of versioned High-Motion reference corrections.

No model inference, benchmark selection, annotation mutation, or publication is
performed here. The same corrected references score every cached/new prediction.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import re


VERSION = "highmotion-right-ring-v2"
TARGET_JOINT = "rightRingFingerMetacarpal"
MASK_POLICY = "finite-positive-depth-in-frame-positive-confidence-v1"
SCORER_VERSION = "position-preserving-masked-grid-v1"
METRICS = ("grid_acc", "grid_ade", "grid_fde", "grid_transition_acc", "token_f1")
NAMES = ("topleft", "top", "topright", "left", "middle", "right",
         "bottomleft", "bottom", "bottomright")
COORDINATES = {name: ((index % 3) / 2, (index // 3) / 2)
               for index, name in enumerate(NAMES)}
ALIASES = {f"r{row + 1}c{column + 1}": NAMES[row * 3 + column]
           for row in range(3) for column in range(3)}
MAX_DISTANCE = math.sqrt(2)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def text_sha256(text):
    require(isinstance(text, str), "Expected text")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sample_indices(frame_count):
    """Exact floor of endpoint-inclusive uniform sampling, never mask-first."""
    require(type(frame_count) is int and frame_count > 0, "Invalid frame_count")
    count = min(8, frame_count)
    if count == 1:
        return [0]
    return [index * (frame_count - 1) // (count - 1) for index in range(count)]


def canonical_label(value):
    if not isinstance(value, str):
        return None
    token = re.sub(r"[\s_-]+", "", value.strip().strip("'\"").lower())
    return token if token in COORDINATES else ALIASES.get(token)


def parse_prediction(value):
    """Preserve unknown/empty entries as None instead of shifting later slots.

    Accepted formats are comma-separated labels or one JSON list. Arbitrary
    explanatory prose is not mined for labels. Surplus slots remain observable.
    """
    require(isinstance(value, str), "Prediction must be a string")
    text = value.strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            parsed = None
        if isinstance(parsed, list):
            return [canonical_label(item) for item in parsed]
    return [canonical_label(item) for item in text.split(",")]


def score_sample(prediction, reference, valid_mask):
    """Score existing sampled slots; missing GT never becomes a correct answer.

    FDE concerns the original final slot, and transitions require adjacent valid
    slots. They are undefined (None) when those respective references are absent.
    F1 uses canonical label bags on valid positions. Missing/malformed predictions
    remain wrong tokens; surplus output slots only enlarge its denominator.
    """
    require(isinstance(reference, list) and isinstance(valid_mask, list)
            and 0 < len(reference) == len(valid_mask) <= 8, "Invalid sampled references")
    for label, valid in zip(reference, valid_mask):
        require(type(valid) is bool, "Mask values must be booleans")
        require((valid and label in COORDINATES) or (not valid and label is None),
                "Reference label and validity mask disagree")
    parsed = parse_prediction(prediction)
    aligned = (parsed + [None] * len(reference))[:len(reference)]
    eligible = [index for index, valid in enumerate(valid_mask) if valid]
    adjacent = [index for index in range(len(reference) - 1)
                if valid_mask[index] and valid_mask[index + 1]]
    metrics = dict.fromkeys(METRICS)

    def distance(index):
        label = aligned[index]
        return (math.dist(COORDINATES[label], COORDINATES[reference[index]])
                if label in COORDINATES else MAX_DISTANCE)

    if eligible:
        metrics["grid_acc"] = sum(aligned[i] == reference[i] for i in eligible) / len(eligible)
        metrics["grid_ade"] = math.fsum(distance(i) for i in eligible) / len(eligible)
        predicted_bag = Counter(aligned[i] for i in eligible if aligned[i] is not None)
        reference_bag = Counter(reference[i] for i in eligible)
        overlap = sum((predicted_bag & reference_bag).values())
        metrics["token_f1"] = 2 * overlap / (2 * len(eligible) + max(0, len(parsed) - len(reference)))
    if valid_mask[-1]:
        metrics["grid_fde"] = distance(len(reference) - 1)
    if adjacent:
        correct = 0
        for index in adjacent:
            first, second = aligned[index:index + 2]
            if first not in COORDINATES or second not in COORDINATES:
                continue
            actual = tuple(b - a for a, b in zip(COORDINATES[first], COORDINATES[second]))
            expected = tuple(b - a for a, b in
                             zip(COORDINATES[reference[index]], COORDINATES[reference[index + 1]]))
            correct += actual == expected
        metrics["grid_transition_acc"] = correct / len(adjacent)
    return {
        "metrics": metrics,
        "denominators": {"grid_acc": len(eligible), "grid_ade": len(eligible),
                         "grid_fde": int(valid_mask[-1]), "grid_transition_acc": len(adjacent),
                         "token_f1": len(eligible)},
        "sampled_slots": len(reference), "valid_slots": len(eligible),
        "prediction_slots": len(parsed),
        "malformed_or_missing_scored_slots": sum(aligned[i] is None for i in eligible),
        "surplus_prediction_slots": max(0, len(parsed) - len(reference)),
    }


def reference_sequence(doc):
    require(doc.get("benchmark_version") == VERSION and doc.get("target_joint") == TARGET_JOINT
            and doc.get("reference_policy") == MASK_POLICY, "Wrong reference version or target")
    frame_count = doc.get("frame_count")
    indices = sample_indices(frame_count)
    require(text_sha256(doc.get("question")) == doc.get("legacy_question_sha256"),
            "GT-only revision changed question text")
    labels = json.loads(doc["answer"])
    masks = json.loads(doc["reference_valid"])
    require(isinstance(labels, list) and isinstance(masks, list)
            and len(labels) == len(masks) == frame_count, "Reference length mismatch")
    for label, valid in zip(labels, masks):
        require(type(valid) is bool and ((valid and label in COORDINATES)
                                        or (not valid and label is None)),
                "Invalid full-frame reference or mask")
    return [labels[index] for index in indices], [masks[index] for index in indices]


def score_doc(doc, prediction):
    reference, masks = reference_sequence(doc)
    result = score_sample(prediction, reference, masks)
    result["source_frame_indices"] = sample_indices(doc["frame_count"])
    return result


def aggregate_scores(rows):
    require(bool(rows), "No scored records supplied")
    averages, row_counts, slot_counts = {}, {}, {}
    for metric in METRICS:
        values = []
        total_slots = 0
        for row in rows:
            value = row["metrics"][metric]
            count = row["denominators"][metric]
            require(type(count) is int and count >= 0, "Invalid metric denominator")
            require((value is None) == (count == 0), "Missing metric/denominator mismatch")
            if value is not None:
                require(type(value) in (int, float) and math.isfinite(value), "Nonfinite metric")
                values.append(value)
            total_slots += count
        averages[metric] = math.fsum(values) / len(values) if values else None
        row_counts[metric], slot_counts[metric] = len(values), total_slots
    return {
        "benchmark_version": VERSION, "scorer_version": SCORER_VERSION,
        "target_joint": TARGET_JOINT, "reference_policy": MASK_POLICY,
        "aggregation": "macro mean of defined per-row metrics; metric-specific denominators",
        "records": len(rows), "records_with_scored_slots": sum(row["valid_slots"] > 0 for row in rows),
        "records_without_scored_slots": sum(row["valid_slots"] == 0 for row in rows),
        "sampled_slots": sum(row["sampled_slots"] for row in rows),
        "valid_slots": sum(row["valid_slots"] for row in rows),
        "metrics": averages, "metric_scored_records": row_counts,
        "metric_scored_slots_or_edges": slot_counts,
    }


def rescore_records(references, records, *, expected_count, expected_inputs=None):
    """Bind cached predictions to the original questions/identities before scoring.

    References retain all source rows; comparisons explicitly use the same fixed
    prefix as the cached benchmark. Input strings can additionally be checked
    against a separately verified common prompt sequence.
    """
    require(type(expected_count) is int and 0 < expected_count <= len(references),
            "Invalid expected prediction count")
    require(len(records) == expected_count, "Incomplete prediction coverage")
    if expected_inputs is not None:
        require(len(expected_inputs) == expected_count, "Incomplete prompt coverage")
    scored, input_bindings = [], []
    for index, record in enumerate(records):
        require(type(record.get("doc_id")) is int and record["doc_id"] == index,
                "Missing, duplicate, or reordered prediction IDs")
        reference, original = references[index], record.get("doc")
        require(isinstance(original, dict), "Missing source document")
        for key in ("qid", "video_path", "question", "frame_count"):
            require(original.get(key) == reference.get(key), "Cached input identity changed: " + key)
        require(text_sha256(original.get("answer")) == reference.get("legacy_answer_sha256"),
                "Cached prediction belongs to a different original reference release")
        prompt = record.get("input")
        require(isinstance(prompt, str) and bool(prompt), "Missing effective input prompt")
        if expected_inputs is not None:
            require(prompt == expected_inputs[index], "Effective prompt differs across compared methods")
        outputs = record.get("filtered_resps")
        require(isinstance(outputs, list) and len(outputs) == 1
                and isinstance(outputs[0], str) and bool(outputs[0].strip()),
                "Missing or ambiguous cached prediction")
        result = score_doc(reference, outputs[0])
        result["doc_id"] = index
        result["identity_sha256"] = text_sha256(json.dumps(
            [reference["video_path"], reference["qid"]], ensure_ascii=False, separators=(",", ":")))
        result["prompt_sha256"] = text_sha256(prompt)
        scored.append(result)
        input_bindings.append([index, reference["video_path"], reference["qid"], prompt])
    report = aggregate_scores(scored)
    report.update({
        "status": "rescored_cached_predictions", "inference_performed": False,
        "scope": f"first-{expected_count}-source-rows",
        "input_sequence_sha256": text_sha256(json.dumps(input_bindings, ensure_ascii=False,
                                                         separators=(",", ":"))),
        "automatic_publication": False, "grt_superiority_verified": False,
        "records_detail": scored,
    })
    return report
