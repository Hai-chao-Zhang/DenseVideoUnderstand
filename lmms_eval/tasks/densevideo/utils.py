# Adopted from lmms-eval (Apache-2.0); original copyright: 2024 LMMs-Lab.
# https://github.com/EvolvingLMMs-Lab/lmms-eval ; see LICENSE.
# DIVE-Bench release changes: optional decoder imports, fixed preview, strict paths.
import ast
import datetime
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import openai
import requests
import yaml
from loguru import logger as eval_logger
from openai import OpenAI

import lmms_eval.tasks._task_utils.file_utils as file_utils

with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

NUM_SECONDS_TO_SLEEP = 5

GPT_EVAL_MODEL_NAME = config["metadata"]["gpt_eval_model_name"]

API_TYPE = os.getenv("API_TYPE", "openai")
ENABLE_GPT_EVAL = os.getenv("DENSEVIDEO_ENABLE_GPT_EVAL", "0").lower() in {"1", "true", "yes"}
FAST_TEXT_METRICS = os.getenv("DENSEVIDEO_FAST_TEXT_METRICS", "0").lower() in {"1", "true", "yes"}
MAX_EDIT_DISTANCE_CELLS = int(os.getenv("DENSEVIDEO_MAX_EDIT_DISTANCE_CELLS", "25000000"))

if API_TYPE == "openai":
    API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

# Unzip all the zip files to HF HOME cache dir
HF_HOME = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
cache_dir = config["dataset_kwargs"]["cache_dir"]
cache_dir = os.path.join(HF_HOME, cache_dir)
cache_dir = os.path.join(cache_dir, "all_test")


def highmotion_select_1000(dataset):
    max_examples = os.getenv("DENSEVIDEO_HIGHMOTION_MAX_EXAMPLES", "1000").strip()
    if max_examples in {"", "0", "-1", "none", "None"}:
        return dataset
    max_examples_int = int(max_examples)
    return dataset.select(range(min(max_examples_int, len(dataset))))


HIGHMOTION_CONTENT_SHA256 = "90ee915016105f6a709f391e8a03a6d0e99bc5c908f945cdf7b80d0cb289e789"


def validate_highmotion_release(dataset):
    """Pin ordered task content while the public Hub revision remains unavailable."""
    if len(dataset) != 3243:
        raise ValueError("DIVE-Bench High-Motion release requires exactly 3243 source examples")
    rows = [[str(row["video_path"]), str(row["qid"]), str(row["question"]),
             str(row["answer"]), int(row["frame_count"])] for row in dataset]
    digest = hashlib.sha256(json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    if digest != HIGHMOTION_CONTENT_SHA256:
        raise ValueError("DIVE-Bench High-Motion annotation content/order differs from the audited release")
    return dataset


def highmotion_preview_1000(dataset):
    """Canonical preview is fixed; legacy environment overrides do not change its split."""
    validate_highmotion_release(dataset)
    return dataset.select(range(1000))


def _highmotion_frame_budget():
    value = os.getenv("DENSEVIDEO_HIGHMOTION_NUM_FRAMES", "8").strip()
    try:
        budget = int(value)
    except ValueError as exc:
        raise ValueError(f"DENSEVIDEO_HIGHMOTION_NUM_FRAMES must be an integer, got {value!r}") from exc
    if budget <= 0:
        raise ValueError(f"DENSEVIDEO_HIGHMOTION_NUM_FRAMES must be positive, got {budget}")
    return budget


def _uniform_subsample(sequence, sample_count):
    """Match the wrappers' endpoint-inclusive uniform frame sampling."""

    sequence = list(sequence)
    if not sequence:
        return []
    sample_count = min(int(sample_count), len(sequence))
    indices = np.linspace(0, len(sequence) - 1, sample_count, dtype=int)
    return [sequence[int(index)] for index in indices]


def _normalize_text(text):
    if text is None:
        return ""
    text = str(text).strip().lower()
    return " ".join(text.split())


def _levenshtein_distance(seq_a, seq_b):
    len_a = len(seq_a)
    len_b = len(seq_b)
    if len_a == 0:
        return len_b
    if len_b == 0:
        return len_a

    prev = list(range(len_b + 1))
    for i in range(1, len_a + 1):
        curr = [i] + [0] * len_b
        for j in range(1, len_b + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[len_b]


def _compute_cer(pred, ref):
    pred_chars = list(_normalize_text(pred))
    ref_chars = list(_normalize_text(ref))
    if FAST_TEXT_METRICS and len(pred_chars) * len(ref_chars) > MAX_EDIT_DISTANCE_CELLS:
        return float("nan")
    return _levenshtein_distance(pred_chars, ref_chars) / max(len(ref_chars), 1)


def _compute_wer(pred, ref):
    pred_words = _normalize_text(pred).split()
    ref_words = _normalize_text(ref).split()
    if FAST_TEXT_METRICS and len(pred_words) * len(ref_words) > MAX_EDIT_DISTANCE_CELLS:
        return float("nan")
    return _levenshtein_distance(pred_words, ref_words) / max(len(ref_words), 1)


def _compute_exact_match(pred, ref):
    return float(_normalize_text(pred) == _normalize_text(ref))


def _compute_token_f1(pred, ref):
    pred_tokens = _normalize_text(pred).split()
    ref_tokens = _normalize_text(ref).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts = {}
    ref_counts = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1

    overlap = 0
    for token, count in pred_counts.items():
        if token in ref_counts:
            overlap += min(count, ref_counts[token])

    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


GRID_LABELS = {
    "r1c1": (1.0 / 6.0, 1.0 / 6.0),
    "r1c2": (3.0 / 6.0, 1.0 / 6.0),
    "r1c3": (5.0 / 6.0, 1.0 / 6.0),
    "r2c1": (1.0 / 6.0, 3.0 / 6.0),
    "r2c2": (3.0 / 6.0, 3.0 / 6.0),
    "r2c3": (5.0 / 6.0, 3.0 / 6.0),
    "r3c1": (1.0 / 6.0, 5.0 / 6.0),
    "r3c2": (3.0 / 6.0, 5.0 / 6.0),
    "r3c3": (5.0 / 6.0, 5.0 / 6.0),
}

GRID_ALIASES = {
    "1": "r1c1",
    "2": "r1c2",
    "3": "r1c3",
    "4": "r2c1",
    "5": "r2c2",
    "6": "r2c3",
    "7": "r3c1",
    "8": "r3c2",
    "9": "r3c3",
    "topleft": "r1c1",
    "top_left": "r1c1",
    "top left": "r1c1",
    "upperleft": "r1c1",
    "upper_left": "r1c1",
    "upper left": "r1c1",
    "lefttop": "r1c1",
    "left_top": "r1c1",
    "top": "r1c2",
    "topcenter": "r1c2",
    "top_center": "r1c2",
    "top center": "r1c2",
    "topmiddle": "r1c2",
    "top_middle": "r1c2",
    "top middle": "r1c2",
    "upper": "r1c2",
    "uppercenter": "r1c2",
    "upper_center": "r1c2",
    "upper center": "r1c2",
    "topright": "r1c3",
    "top_right": "r1c3",
    "top right": "r1c3",
    "upperright": "r1c3",
    "upper_right": "r1c3",
    "upper right": "r1c3",
    "righttop": "r1c3",
    "right_top": "r1c3",
    "left": "r2c1",
    "middleleft": "r2c1",
    "middle_left": "r2c1",
    "middle left": "r2c1",
    "centerleft": "r2c1",
    "center_left": "r2c1",
    "center left": "r2c1",
    "middle": "r2c2",
    "center": "r2c2",
    "centre": "r2c2",
    "r2c2": "r2c2",
    "right": "r2c3",
    "middleright": "r2c3",
    "middle_right": "r2c3",
    "middle right": "r2c3",
    "centerright": "r2c3",
    "center_right": "r2c3",
    "center right": "r2c3",
    "bottomleft": "r3c1",
    "bottom_left": "r3c1",
    "bottom left": "r3c1",
    "lowerleft": "r3c1",
    "lower_left": "r3c1",
    "lower left": "r3c1",
    "leftbottom": "r3c1",
    "left_bottom": "r3c1",
    "bottom": "r3c2",
    "bottomcenter": "r3c2",
    "bottom_center": "r3c2",
    "bottom center": "r3c2",
    "bottommiddle": "r3c2",
    "bottom_middle": "r3c2",
    "bottom middle": "r3c2",
    "lower": "r3c2",
    "lowercenter": "r3c2",
    "lower_center": "r3c2",
    "lower center": "r3c2",
    "bottomright": "r3c3",
    "bottom_right": "r3c3",
    "bottom right": "r3c3",
    "lowerright": "r3c3",
    "lower_right": "r3c3",
    "lower right": "r3c3",
    "rightbottom": "r3c3",
    "right_bottom": "r3c3",
}

_GRID_MAX_DISTANCE = math.sqrt(2.0)


def _canonical_grid_label(label):
    if label is None:
        return None
    text = str(label).strip().lower()
    text = text.strip("`'\".,;:()[]{}")
    text = re.sub(r"[\s\-]+", " ", text)
    compact = text.replace(" ", "")
    text_us = text.replace(" ", "_")
    for candidate in (text, compact, text_us):
        if candidate in GRID_LABELS:
            return candidate
        if candidate in GRID_ALIASES:
            return GRID_ALIASES[candidate]
    match = re.fullmatch(r"r\s*([1-3])\s*c\s*([1-3])", text)
    if match:
        return f"r{match.group(1)}c{match.group(2)}"
    match = re.fullmatch(r"row\s*([1-3])\s*(?:col|column)\s*([1-3])", text)
    if match:
        return f"r{match.group(1)}c{match.group(2)}"
    return None


def parse_grid_sequence(text):
    """Parse 3x3 grid labels from JSON lists, comma text, or natural language."""
    if text is None:
        return []

    if isinstance(text, (list, tuple)):
        out = []
        for item in text:
            nested = parse_grid_sequence(item)
            if nested:
                out.extend(nested)
            else:
                label = _canonical_grid_label(item)
                if label:
                    out.append(label)
        return out

    if isinstance(text, dict):
        for key in ("answer", "trajectory", "traj", "sequence", "grid", "positions", "labels"):
            if key in text:
                parsed = parse_grid_sequence(text[key])
                if parsed:
                    return parsed
        out = []
        for value in text.values():
            out.extend(parse_grid_sequence(value))
        return out

    raw = str(text).strip()
    if not raw:
        return []

    for parser in (json.loads, ast.literal_eval):
        if raw[:1] in "[{\"'(":
            try:
                parsed = parser(raw)
                if parsed is not raw:
                    seq = parse_grid_sequence(parsed)
                    if seq:
                        return seq
            except Exception:
                pass

    lowered = raw.lower()
    lowered = lowered.replace("_", " ").replace("-", " ")

    matches = []

    for match in re.finditer(r"\br\s*([1-3])\s*c\s*([1-3])\b", lowered):
        matches.append((match.start(), match.end(), f"r{match.group(1)}c{match.group(2)}"))
    for match in re.finditer(r"\brow\s*([1-3])\s*(?:col|column)\s*([1-3])\b", lowered):
        matches.append((match.start(), match.end(), f"r{match.group(1)}c{match.group(2)}"))

    alias_items = sorted(
        ((alias.replace("_", " ").replace("-", " "), canonical) for alias, canonical in GRID_ALIASES.items() if not alias.isdigit()),
        key=lambda x: len(x[0]),
        reverse=True,
    )
    for alias, canonical in alias_items:
        pattern = r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])"
        for match in re.finditer(pattern, lowered):
            matches.append((match.start(), match.end(), canonical))

    for match in re.finditer(r"(?<!\d)([1-9])(?!\d)", lowered):
        matches.append((match.start(), match.end(), GRID_ALIASES[match.group(1)]))

    if matches:
        chosen = []
        occupied = []
        for start, end, label in sorted(matches, key=lambda x: (x[0], -(x[1] - x[0]))):
            if any(not (end <= s or start >= e) for s, e in occupied):
                continue
            chosen.append((start, label))
            occupied.append((start, end))
        return [label for _, label in sorted(chosen, key=lambda x: x[0])]

    out = []
    for chunk in re.split(r"[,;/\n]+|\s+then\s+|\s*->\s*", lowered):
        label = _canonical_grid_label(chunk)
        if label:
            out.append(label)
    return out


def grid_label_to_xy(label):
    canonical = _canonical_grid_label(label)
    if canonical is None:
        return None
    return GRID_LABELS.get(canonical)


def _mean_finite(values, default=0.0):
    vals = []
    for value in values:
        try:
            value = float(value)
        except Exception:
            continue
        if math.isfinite(value):
            vals.append(value)
    return float(np.mean(vals)) if vals else default


def _grid_sequence_metrics(pred_seq, ref_seq):
    if not ref_seq:
        return {
            "grid_acc": 0.0,
            "grid_ade": _GRID_MAX_DISTANCE,
            "grid_fde": _GRID_MAX_DISTANCE,
            "grid_transition_acc": 0.0,
        }

    correct = 0
    distances = []
    for idx, ref_label in enumerate(ref_seq):
        pred_label = pred_seq[idx] if idx < len(pred_seq) else None
        if pred_label == ref_label:
            correct += 1
        pred_xy = grid_label_to_xy(pred_label)
        ref_xy = grid_label_to_xy(ref_label)
        if pred_xy is None or ref_xy is None:
            distances.append(_GRID_MAX_DISTANCE)
        else:
            distances.append(math.dist(pred_xy, ref_xy))

    last_pred = pred_seq[len(ref_seq) - 1] if len(pred_seq) >= len(ref_seq) else None
    last_ref = ref_seq[-1]
    last_pred_xy = grid_label_to_xy(last_pred)
    last_ref_xy = grid_label_to_xy(last_ref)
    if last_pred_xy is None or last_ref_xy is None:
        fde = _GRID_MAX_DISTANCE
    else:
        fde = math.dist(last_pred_xy, last_ref_xy)

    if len(ref_seq) <= 1:
        transition_acc = 1.0
    else:
        trans_correct = 0
        for idx in range(len(ref_seq) - 1):
            if idx + 1 >= len(pred_seq):
                continue
            ref_a = grid_label_to_xy(ref_seq[idx])
            ref_b = grid_label_to_xy(ref_seq[idx + 1])
            pred_a = grid_label_to_xy(pred_seq[idx])
            pred_b = grid_label_to_xy(pred_seq[idx + 1])
            if None in (ref_a, ref_b, pred_a, pred_b):
                continue
            ref_delta = (round(ref_b[0] - ref_a[0], 6), round(ref_b[1] - ref_a[1], 6))
            pred_delta = (round(pred_b[0] - pred_a[0], 6), round(pred_b[1] - pred_a[1], 6))
            if pred_delta == ref_delta:
                trans_correct += 1
        transition_acc = trans_correct / float(len(ref_seq) - 1)

    return {
        "grid_acc": correct / float(len(ref_seq)),
        "grid_ade": float(np.mean(distances)) if distances else _GRID_MAX_DISTANCE,
        "grid_fde": float(fde),
        "grid_transition_acc": float(transition_acc),
    }





def _resolve_lpm_video_path(video_path):
    video_path = str(video_path)
    if os.path.exists(video_path):
        return video_path

    basename = os.path.basename(video_path)
    relative_path = video_path.lstrip(os.sep)
    roots = [
        os.getenv("DENSEVIDEO_DATA_ROOT"),
        os.getenv("HF_HOME"),
        os.path.expanduser("~/.cache/huggingface"),
        os.getcwd(),
        os.path.join(os.getcwd(), "hf_cache"),
    ]
    seen_roots = []
    for root in roots:
        if root and root not in seen_roots:
            seen_roots.append(root)

    for root in seen_roots:
        candidates = [
            os.path.join(root, relative_path),
            os.path.join(root, "DenseVideoEvaluation", relative_path),
            os.path.join(root, "highmotion_densevideounderstand", relative_path),
            os.path.join(root, "highmotion_densevideounderstand", "all_test", relative_path),
        ]
        # EgoDex repeats numeric basenames across actions. Never drop those directories.
        if not relative_path.startswith("egodex/"):
            candidates.extend([
                os.path.join(root, "DenseVideoEvaluation", "videos", basename),
                os.path.join(root, "DenseVideo-LPM", "videos", basename),
            ])
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
    raise FileNotFoundError(
        f"DIVE-Bench video not found: {video_path}. Set DENSEVIDEO_DATA_ROOT to the "
        "extracted video root and preserve egodex/<action>/<clip>.mp4 directories."
    )


# --- 1) 视频路径 loader ---
def lpm_doc_to_visual(doc):
    # `doc["video_path"]` 在 Parquet 里已经存了绝对或相对路径
    return [_resolve_lpm_video_path(doc["video_path"])]

# --- 2) 问题拼装 ---
def lpm_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """
    If the doc was generated from slide-level subtitle QA,
    doc['question'] 已含完整问题。
    """
    lmms_eval_specific_kwargs = lmms_eval_specific_kwargs or {}
    pre = lmms_eval_specific_kwargs.get("pre_prompt","")
    post = lmms_eval_specific_kwargs.get("post_prompt","")
    return f"{pre}{doc['question']}{post}"


def lpm_doc_to_text_structured(doc, lmms_eval_specific_kwargs=None):
    """Prompt LPM text QA with a compact, type-aware answer format."""
    question = str(doc["question"])
    lowered = question.lower()
    if "subtitle" in lowered:
        instruction = (
            " Answer with one line beginning with 'Subtitles:'. "
            "Transcribe any visible subtitle text exactly when it can be read. "
            "If speech text is not visible in the frames, write a concise visual caption of the scene and any readable on-screen text instead. "
            "Do not answer with an empty subtitle, placeholder text, angle brackets, the video id, or an explanation."
        )
    elif "ocr" in lowered or "text is extracted" in lowered:
        instruction = (
            " Answer with one line beginning with 'OCR:' followed by readable text from the video frames. "
            "If no text is readable, write a short visual caption after 'OCR:' instead of returning an empty answer. "
            "Do not output OCR coordinates, bounding boxes, confidence scores, JSON metadata, placeholder text, the video id, or an explanation."
        )
    else:
        instruction = (
            " Answer directly with the actual video content. Do not output placeholder text, the video id, or an explanation."
        )
    return f"{question}{instruction}"

# --- 3) Ground-truth answer extractor ---
def lpm_doc_to_answer(doc):
    # doc["answer"] 存了我们之前脚本里生成的 subtitles 或 OCR 文本
    return doc["answer"]


_HIGHMOTION_GRID_NAMES = {
    "r1c1": "topleft",
    "r1c2": "top",
    "r1c3": "topright",
    "r2c1": "left",
    "r2c2": "middle",
    "r2c3": "right",
    "r3c1": "bottomleft",
    "r3c2": "bottom",
    "r3c3": "bottomright",
}


def _highmotion_sampled_grid_sequence(doc):
    full_sequence = parse_grid_sequence(doc.get("answer", ""))
    return _uniform_subsample(full_sequence, _highmotion_frame_budget())


def _highmotion_count_text(value):
    words = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
        11: "eleven",
        12: "twelve",
    }
    return words.get(int(value), str(value))


def highmotion_doc_to_answer(doc):
    """Return ground truth aligned with the uniformly sampled model frames."""

    sampled = _highmotion_sampled_grid_sequence(doc)
    return ",".join(_HIGHMOTION_GRID_NAMES.get(label, label) for label in sampled)


def highmotion_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Ask only for positions observable in the model's sampled video frames."""

    sampled_count = len(_highmotion_sampled_grid_sequence(doc))
    sampled_count_text = _highmotion_count_text(sampled_count)
    comma_count_text = _highmotion_count_text(max(sampled_count - 1, 0))
    original_question = str(doc.get("question", "")).strip()
    action = re.split(r"\n\nWe consider all\s+\d+\s+frames", original_question, maxsplit=1)[0].strip()
    if not action:
        action = "Track the visible right-hand palm center in the video."
    return (
        f"{action}\n\n"
        f"The video input contains exactly {sampled_count_text} frames uniformly sampled in temporal order from the original clip, including its first and last frames. "
        "On each sampled frame, divide the image into top, middle, and bottom rows and left, center, and right columns, then locate the visible right-hand palm center. "
        "Use topleft, top, or topright for the top row; left, middle, or right for the middle row; and bottomleft, bottom, or bottomright for the bottom row. "
        "For every sampled frame in order, select one region; repeat a name when the hand stays in the same region.\n\n"
        f"Answer with exactly {sampled_count_text} labels in sampled-frame order (exactly {comma_count_text} commas) and no extra text. "
        "Return one item per input frame—not one item per possible region—and stop after the final frame's label. "
        f"Do not stop early: if uncertain, repeat your best region choice until all {sampled_count_text} frame slots are filled. "
        "Before answering, verify the comma count. Use commas only; do not use vertical bars or brackets."
    )

# --- 4) GPT-based 评估流程（几乎同 ActivityNetQA） ---
def lpm_process_results(doc, result):
    """
    Args:
      doc: one record from LPM_slides.parquet
      result: [predicted_string]
    Returns:
      dict with gpt_eval_score & gpt_eval_accuracy formats
    """
    question = doc["question"]
    answer   = doc["answer"]
    pred     = result[0]
    cer = _compute_cer(pred, answer)
    wer = _compute_wer(pred, answer)
    token_f1 = _compute_token_f1(pred, answer)
    exact_match = _compute_exact_match(pred, answer)
    # code复用 get_eval & parse_score
    if ENABLE_GPT_EVAL:
        try:
            review, _ = get_eval(question, answer, pred, max_tokens=64)
            pred_label, score = parse_score(review)
        except Exception:
            pred_label, score = "no", 0
    else:
        pred_label, score = "no", 0
    common = {
      "video_name": doc["video"],    # slide tag 或 video id
      "question":  question,
      "answer":    answer,
      "pred":      pred,
      "question_id": doc["qid"],
      "type":       doc["type"]
    }
    return {
      "cer": cer,
      "wer": wer,
      "token_f1": token_f1,
      "exact_match": exact_match,
      "gpt_eval_score":    {**common, "Correctness": pred_label, "score": score},
      "gpt_eval_accuracy": {**common, "Correctness": pred_label, "score": score}
    }


def highmotion_process_results(doc, result):
    """Evaluate high-motion grid trajectories with sequence-aware metrics."""
    question = highmotion_doc_to_text(doc)
    answer = highmotion_doc_to_answer(doc)
    pred = result[0] if result else ""

    ref_seq = parse_grid_sequence(answer)
    pred_seq = parse_grid_sequence(pred)
    metrics = _grid_sequence_metrics(pred_seq, ref_seq)
    # Score the parsed canonical labels so harmless formatting differences
    # (spaces after commas or rNcN aliases) do not corrupt token_f1.
    token_f1 = _compute_token_f1(" ".join(pred_seq), " ".join(ref_seq))

    common = {
        "video_name": doc.get("video", doc.get("video_id", "")),
        "question": question,
        "answer": answer,
        "pred": pred,
        "question_id": doc.get("qid", doc.get("question_id", "")),
        "type": doc.get("type", ""),
        "ref_grid_len": len(ref_seq),
        "pred_grid_len": len(pred_seq),
    }

    return {
        "grid_acc": {**common, "score": metrics["grid_acc"]},
        "grid_ade": {**common, "score": metrics["grid_ade"]},
        "grid_fde": {**common, "score": metrics["grid_fde"]},
        "grid_transition_acc": {**common, "score": metrics["grid_transition_acc"]},
        "token_f1": token_f1,
    }

# --- 5) 聚合函数（同样复用 ActivityNetQA 的） ---
def lpm_aggregate_score(results, args):
    yes = sum(1 for r in results if r["Correctness"]=="yes")
    no  = sum(1 for r in results if r["Correctness"]=="no")
    total = sum(r["score"] for r in results)
    return total/len(results) if results else 0

def lpm_aggregate_accuracy(results, args):
    yes = sum(1 for r in results if r["Correctness"]=="yes")
    no  = sum(1 for r in results if r["Correctness"]=="no")
    return yes/(yes+no) if (yes+no)>0 else 0


def lpm_aggregate_cer(results, args):
    values = [x for x in results if not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(values)) if values else float("nan")


def lpm_aggregate_wer(results, args):
    values = [x for x in results if not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(values)) if values else float("nan")


def lpm_aggregate_token_f1(results, args):
    return float(np.mean(results)) if results else 0.0


def lpm_aggregate_exact_match(results, args):
    return float(np.mean(results)) if results else 0.0


def _aggregate_score_field(results, default=0.0):
    values = []
    for item in results:
        if isinstance(item, dict):
            values.append(item.get("score"))
        else:
            values.append(item)
    return _mean_finite(values, default=default)


def highmotion_aggregate_grid_acc(results, args):
    return _aggregate_score_field(results)


def highmotion_aggregate_grid_ade(results, args):
    return _aggregate_score_field(results, default=_GRID_MAX_DISTANCE)


def highmotion_aggregate_grid_fde(results, args):
    return _aggregate_score_field(results, default=_GRID_MAX_DISTANCE)


def highmotion_aggregate_grid_transition_acc(results, args):
    return _aggregate_score_field(results)


TRIM_CHAR_LIMIT = 30_000


def get_eval(question, answer, pred, max_tokens: int, retries: int = 5):
    answer = (answer[:TRIM_CHAR_LIMIT] + '...') if len(answer) > TRIM_CHAR_LIMIT else answer
    pred   = (pred[:TRIM_CHAR_LIMIT]   + '...') if len(pred)   > TRIM_CHAR_LIMIT else pred

    messages = [
        {
            "role": "system",
            "content": (
                "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:\n"
                "## INSTRUCTIONS:\n"
                "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                "- Consider synonyms or paraphrases as valid matches.\n"
                "- Evaluate the correctness of the prediction compared to the answer."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Please evaluate the following video-based question-answer pair:\n\n"
                f"Question: {question}\n"
                f"Correct Answer: {answer}\n"
                f"Predicted Answer: {pred}\n\n"
                "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, "
                "with 5 indicating the highest meaningful match. "
                "Please generate the response only as a Python dictionary string with keys 'pred' and 'score', "
                "where the value of 'pred' is 'yes' or 'no' and the value of 'score' is an INTEGER, not a STRING. "
                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION."
            ),
        },
    ]

    payload = {
        "model": GPT_EVAL_MODEL_NAME,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }

    for attempt in range(retries):
        backoff = NUM_SECONDS_TO_SLEEP * (2 ** attempt)
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            # 如果是 429 或者其他 5xx/4xx，response.raise_for_status() 会抛出 HTTPError
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()
            if content:
                return content, data.get("model", "")
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            # 专门处理 429
            if status == 429:
                retry_after = e.response.headers.get("Retry-After")
                sleep_time = int(retry_after) if retry_after and retry_after.isdigit() else backoff
                eval_logger.warning(f"Rate limit hit (429). Sleeping for {sleep_time}s (attempt {attempt+1}/{retries})")
                time.sleep(sleep_time)
                continue
            else:
                eval_logger.error(f"HTTP error {status} on attempt {attempt+1}/{retries}: {e}. Backing off {backoff}s.")
                time.sleep(backoff)
                continue
        except requests.exceptions.RequestException as e:
            eval_logger.error(f"Request exception on attempt {attempt+1}/{retries}: {e}. Backing off {backoff}s.")
            time.sleep(backoff)
            continue
        except ValueError as e:
            # JSON decode error
            eval_logger.error(f"JSON decode error on attempt {attempt+1}/{retries}: {e}. Response text: {response.text}")
            time.sleep(backoff)
            continue
        except Exception as e:
            eval_logger.error(f"Unexpected error on attempt {attempt+1}/{retries}: {e}. Backing off {backoff}s.")
            time.sleep(backoff)
            continue

    eval_logger.error(f"All {retries} attempts failed.")
    return "", ""



# def get_eval(question, answer, pred, max_tokens: int, retries: int = 5):
#     global headers

#     answer = (answer[:TRIM_CHAR_LIMIT] + '...') if len(answer) > TRIM_CHAR_LIMIT else answer
#     pred   = (pred[:TRIM_CHAR_LIMIT]   + '...') if len(pred)   > TRIM_CHAR_LIMIT else pred

#     messages = [
#         {
#             "role": "system",
#             "content": "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
#             "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
#             "------"
#             "##INSTRUCTIONS: "
#             "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
#             "- Consider synonyms or paraphrases as valid matches.\n"
#             "- Evaluate the correctness of the prediction compared to the answer.",
#         },
#         {
#             "role": "user",
#             "content": f"Please evaluate the following video-based question-answer pair:\n\n"
#             f"Question: {question}\n"
#             f"Correct Answer: {answer}\n"
#             f"Predicted Answer: {pred}\n\n"
#             "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
#             "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
#             "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
#             "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}.",
#         },
#     ]

#     payload = {
#         "model": GPT_EVAL_MODEL_NAME,
#         "messages": messages,
#         "temperature": 0,
#         "max_tokens": max_tokens,
#     }

#     for attempt in range(retries):
#         try:
#             response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
#             response.raise_for_status()  # Raises HTTPError for bad responses
#             try:
#                 response_data = response.json()  # Attempt to parse JSON
#             except requests.exceptions.JSONDecodeError:
#                 eval_logger.error(f"JSON decode error on attempt {attempt + 1}. Response text: {response.text}")
#                 continue  # Skip to next retry
#             content = response_data["choices"][0]["message"]["content"].strip()
#             if content != "":
#                 return content, response_data["model"]
#         # Handle HTTP errors separately
#         except requests.exceptions.HTTPError as e:
#             eval_logger.error(f"HTTP error on attempt {attempt + 1}: {e}")
#         # Handle other requests-related errors
#         except requests.exceptions.RequestException as e:
#             eval_logger.error(f"Request exception on attempt {attempt + 1}: {e}")
#         except Exception as e:
#             eval_logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")

#         # Handle other unexpected errors
#         if attempt < retries - 1:
#             time.sleep(NUM_SECONDS_TO_SLEEP)
#         else:  # If this was the last attempt, log and return empty
#             eval_logger.error(f"All {retries} attempts failed. Last error message: {e}")
#             return "", ""

#     return "", ""


def parse_score(review):
    try:
        # Convert the string representation of a dictionary to an actual dictionary
        review = "{" + review.split("{")[1].split("}")[0] + "}"
        review_dict = ast.literal_eval(review)
        # import pdb;pdb.set_trace()
        score_match = review_dict["score"]
        score = int(score_match)
        pred = review_dict["pred"]
        if "yes" in pred.lower():
            pred = "yes"
        elif "no" in pred.lower():
            pred = "no"
        # pred = review_dict.get("pred", "no")
        # score = review_dict.get("score", 0)
        return [pred, score]
    except SyntaxError as e:
        eval_logger.error(f"Syntax error parsing the review string: {e}. Review content: {review}")
    except ValueError as e:
        eval_logger.error(f"Value error parsing the review string: {e}. Review content: {review}")
    except Exception as e:
        eval_logger.error(f"Unexpected error parsing the review string: {e}. Review content: {review}")


def activitynetqa_process_results(doc, result):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary
    """
    try:
        question = doc["question"]
        answer = doc["answer"]
        pred = result[0]

        # Assume get_eval returns a review and the model name, and parse_score parses this review
        review, model_name = get_eval(question, answer, pred, 64)
        scores = parse_score(review)
    except Exception as e:
        eval_logger.error(f"Error for Question ID: {doc.get('question_id', 'Unknown')}: {e}")
        review = "Failed to Get a Proper Review."
        model_name = "Failed Request"
        scores = ["no", 0]

    return {
        "gpt_eval_score": {"video_name": doc["video_name"], "question": doc["question"], "answer": doc["answer"], "pred": pred, "question_id": doc["question_id"], "type": doc["type"], "Correctness": scores[0], "score": scores[1]},
        "gpt_eval_accuracy": {"video_name": doc["video_name"], "question": doc["question"], "answer": doc["answer"], "pred": pred, "question_id": doc["question_id"], "type": doc["type"], "Correctness": scores[0], "score": scores[1]},
    }


def activitynetqa_gpt_eval(results, args):
    """
    Process the result file containing predictions, score them using GPT,
    and save the results with added scores and correctness fields to a new file.

    Args:
        result_file_path: path to the JSON file with results to be evaluated
        eval_file_path: path to save the JSON file with evaluated results
    """

    evaluated_results = []

    # Process each result to generate scores
    for data_dict in results:
        try:
            question = data_dict.get("Q", "")
            answer = data_dict.get("A", "")
            pred = data_dict.get("pred", "")

            # Assume get_eval returns a review and the model name, and parse_score parses this review
            review, model_name = get_eval(question, answer, pred, 64)
            scores = parse_score(review)
        except Exception as e:
            eval_logger.error(f"Error for Question ID: {data_dict.get('question_id', 'Unknown')}: {e}")
            review = "Failed to Get a Proper Review."
            model_name = "Failed Request"
            scores = ["no", 0]

        # Update the dictionary with the new entries
        updated_dict = {"video_name": data_dict["video_name"], "Correctness": scores[0], "score": scores[1], "Q": question, "A": answer, "pred": pred, "question_id": data_dict.get("question_id"), "type": data_dict.get("type")}
        evaluated_results.append(updated_dict)

    return evaluated_results


# Factory into different aggregate
def activitynetqa_aggregate_score(results, args):
    yes_count = 0
    no_count = 0
    total_score = 0

    # Iterate over the results to count correctness and sum scores
    for result_dict in results:
        if "yes" in result_dict["Correctness"].lower():
            yes_count += 1
        elif "no" in result_dict["Correctness"].lower():
            no_count += 1

        total_score += int(result_dict["score"])

    # Calculate accuracy and average score
    accuracy = yes_count / (yes_count + no_count) if (yes_count + no_count) > 0 else 0
    average_score = total_score / len(results) if results else 0
    eval_logger.info(f"Accuracy: {accuracy}")
    eval_logger.info(f"Average Score: {average_score}")
    return average_score


def activitynetqa_aggregate_accuracy(results, args):
    yes_count = 0
    no_count = 0
    total_score = 0

    # Iterate over the results to count correctness and sum scores
    for result_dict in results:
        if "yes" in result_dict["Correctness"].lower():
            yes_count += 1
        elif "no" in result_dict["Correctness"].lower():
            no_count += 1

        total_score += int(result_dict["score"])

    # Calculate accuracy and average score
    accuracy = yes_count / (yes_count + no_count) if (yes_count + no_count) > 0 else 0
    average_score = total_score / len(results) if results else 0
    eval_logger.info(f"Accuracy: {accuracy}")
    eval_logger.info(f"Average Score: {average_score}")
    return accuracy * 100
