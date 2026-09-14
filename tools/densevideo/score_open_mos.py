#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import requests


DEFAULT_MODEL = "Qwen/Qwen3-VL-32B-Instruct"
DEFAULT_TRIM_CHAR_LIMIT = 12000
OPEN_MOS_PROMPT_VERSION = "densevideo-open-mos-v1"
OPEN_MOS_RESPONSE_SCHEMA_VERSION = "strict-json-pred-score-v1"
OPEN_MOS_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["pred", "score"],
    "properties": {
        "pred": {"type": "string", "enum": ["yes", "no"]},
        "score": {"type": "integer", "minimum": 0, "maximum": 5},
    },
}

csv.field_size_limit(sys.maxsize)


def first_value(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            nested = first_value(*value)
            if nested:
                return nested
        elif value != "":
            return str(value)
    return ""


def normalize_text(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().split())


def prediction_from_record(record: Dict[str, Any]) -> str:
    if "filtered_resps" in record:
        return first_value(record.get("filtered_resps"))
    if "resps" in record:
        return first_value(record.get("resps"))
    return first_value(record.get("pred"), record.get("prediction_parsed"), record.get("prediction_raw"))


def ground_truth_from_record(record: Dict[str, Any]) -> str:
    doc = record.get("doc") if isinstance(record.get("doc"), dict) else {}
    return first_value(doc.get("answer"), record.get("target"), record.get("ground_truth"))


def question_from_record(record: Dict[str, Any]) -> str:
    doc = record.get("doc") if isinstance(record.get("doc"), dict) else {}
    question = first_value(
        doc.get("question"),
        record.get("input"),
        record.get("question"),
        record.get("prompt"),
    )
    if question:
        return question
    if str(record.get("subtask", "")).strip().lower() == "lpm":
        video_id = first_value(record.get("video_id"), record.get("video_path"))
        if video_id:
            return f"What subtitles appear in the entire video {video_id}?"
    return ""


def video_name_from_record(record: Dict[str, Any]) -> str:
    doc = record.get("doc") if isinstance(record.get("doc"), dict) else {}
    return first_value(doc.get("video"), doc.get("video_id"), record.get("video_id"))


def question_id_from_record(record: Dict[str, Any]) -> str:
    doc = record.get("doc") if isinstance(record.get("doc"), dict) else {}
    return first_value(doc.get("qid"), record.get("question_id"), record.get("doc_id"))


def task_type_from_record(record: Dict[str, Any]) -> str:
    doc = record.get("doc") if isinstance(record.get("doc"), dict) else {}
    return first_value(doc.get("type"), record.get("type"))


def trim_text(text: str, limit: int) -> str:
    text = "" if text is None else str(text)
    return (text[:limit] + "...") if len(text) > limit else text


def build_messages(question: str, answer: str, pred: str, trim_char_limit: int) -> List[Dict[str, str]]:
    answer = trim_text(answer, trim_char_limit)
    pred = trim_text(pred, trim_char_limit)
    return [
        {
            "role": "system",
            "content": (
                "You are an expert evaluator for DIVE-Bench dense video understanding answers. "
                "Compare a model prediction with the reference answer and assign a MOS-style semantic match score.\n"
                "Use score 0 for no useful match and score 5 for a near-complete meaningful match. "
                "Consider paraphrases valid, but penalize omissions, hallucinations, wrong temporal order, and generic refusals."
            ),
        },
        {
            "role": "user",
            "content": (
                "Evaluate this video question-answer pair.\n\n"
                f"Question: {question}\n"
                f"Reference answer: {answer}\n"
                f"Predicted answer: {pred}\n\n"
                "Return only strict JSON with keys: "
                "{\"pred\": \"yes\" or \"no\", \"score\": integer 0-5}. "
                "Do not include explanation or markdown."
            ),
        },
    ]


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def request_fingerprint(request: "JudgeRequest") -> str:
    """Hash the exact, method-independent input presented to a judge."""

    return _canonical_sha256(
        {
            "messages": request.messages,
            # DryRunJudge consumes these fields directly. Keeping them in the
            # identity also protects against a future prompt that omits one.
            "answer": request.answer,
            "pred": request.pred,
            "prompt_version": OPEN_MOS_PROMPT_VERSION,
            "response_schema_version": OPEN_MOS_RESPONSE_SCHEMA_VERSION,
        }
    )


def judge_fingerprint(
    *,
    model: str,
    revision: Optional[str] = None,
    backend: str,
    dtype: str,
    max_new_tokens: int,
    trim_char_limit: int,
    trust_remote_code: bool = True,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Hash every judge setting that can change a cached MOS decision."""

    payload: Dict[str, Any] = {
        "model": str(model),
        "revision": str(revision or "").strip(),
        "backend": str(backend),
        "dtype": str(dtype),
        "max_new_tokens": int(max_new_tokens),
        "trim_char_limit": int(trim_char_limit),
        "trust_remote_code": bool(trust_remote_code),
        "temperature": 0,
        "prompt_version": OPEN_MOS_PROMPT_VERSION,
        "prompt_template": build_messages(
            "<QUESTION>",
            "<REFERENCE_ANSWER>",
            "<PREDICTED_ANSWER>",
            trim_char_limit,
        ),
        "response_schema_version": OPEN_MOS_RESPONSE_SCHEMA_VERSION,
        "response_schema": OPEN_MOS_RESPONSE_SCHEMA,
    }
    if extra:
        payload["extra"] = extra
    return _canonical_sha256(payload)


def parse_mos_response(review: str) -> Tuple[str, int]:
    """Parse the advertised strict JSON schema or raise ``ValueError``.

    A valid score of zero must remain distinguishable from an invalid judge
    response so failed rows can be retried safely with ``--resume``.
    """

    text = str(review or "").strip()
    if not text:
        raise ValueError("empty response")
    def unique_object(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        parsed = json.loads(text, object_pairs_hook=unique_object)
    except json.JSONDecodeError as exc:
        raise ValueError(f"response is not strict JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("response must be a JSON object")
    if set(parsed) != {"pred", "score"}:
        raise ValueError("response must contain exactly the keys 'pred' and 'score'")

    pred = parsed["pred"]
    if not isinstance(pred, str) or pred.strip().lower() not in {"yes", "no"}:
        raise ValueError("pred must be the string 'yes' or 'no'")
    score = parsed["score"]
    if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 5:
        raise ValueError("score must be an integer between 0 and 5")
    return pred.strip().lower(), score


def reusable_result(row: Optional[Dict[str, Any]], request_hash: str, judge_hash: str) -> bool:
    """Return whether a completed row is safe to reuse for this exact run."""

    if not row or str(row.get("error", "")).strip():
        return False
    if row.get("request_fingerprint") != request_hash or row.get("judge_fingerprint") != judge_hash:
        return False
    try:
        parse_mos_response(str(row.get("open_mos_review", "")))
    except ValueError:
        return False
    return True


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj


def read_done(path: Path) -> Dict[str, Dict[str, Any]]:
    done: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            sample_id = str(obj.get("sample_id", ""))
            if sample_id:
                done[sample_id] = obj
    return done


def sample_id_for_record(record: Dict[str, Any], idx: int) -> str:
    doc_id = first_value(record.get("doc_id"), idx)
    doc_hash = str(record.get("doc_hash") or "")[:12]
    return f"doc_{doc_id}_{doc_hash}" if doc_hash else f"doc_{doc_id}"


@dataclass
class JudgeResponse:
    text: str
    model: str


@dataclass
class JudgeRequest:
    messages: List[Dict[str, str]]
    answer: str
    pred: str


class DryRunJudge:
    def __init__(self, model: str) -> None:
        self.model = model

    def score(self, messages: List[Dict[str, str]], *, answer: str, pred: str) -> JudgeResponse:
        score = 5 if normalize_text(answer) == normalize_text(pred) and answer else 0
        correctness = "yes" if score == 5 else "no"
        return JudgeResponse(text=json.dumps({"pred": correctness, "score": score}), model=self.model)

    def score_many(self, requests: Sequence[JudgeRequest]) -> List[JudgeResponse]:
        return [self.score(request.messages, answer=request.answer, pred=request.pred) for request in requests]


class OpenAICompatibleJudge:
    def __init__(
        self,
        model: str,
        api_base: str,
        api_key: str,
        retries: int,
        sleep_seconds: float,
        timeout: int,
        max_new_tokens: int,
    ) -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.retries = retries
        self.sleep_seconds = sleep_seconds
        self.timeout = timeout
        self.max_new_tokens = max_new_tokens

    def score(self, messages: List[Dict[str, str]], *, answer: str, pred: str) -> JudgeResponse:
        endpoint = self.api_base
        if not endpoint.endswith("/chat/completions"):
            endpoint = endpoint + "/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self.max_new_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        for attempt in range(self.retries):
            try:
                response = requests.post(endpoint, headers=headers, json=payload, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                text = data["choices"][0]["message"]["content"].strip()
                return JudgeResponse(text=text, model=data.get("model", self.model))
            except Exception:
                if attempt >= self.retries - 1:
                    raise
                time.sleep(self.sleep_seconds * (2**attempt))
        raise RuntimeError("openai-compatible judge failed without response")

    def score_many(self, requests: Sequence[JudgeRequest]) -> List[JudgeResponse]:
        return [self.score(request.messages, answer=request.answer, pred=request.pred) for request in requests]


class TransformersJudge:
    def __init__(
        self,
        model: str,
        dtype: str,
        trust_remote_code: bool,
        max_new_tokens: int,
        revision: Optional[str] = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        self.model_name = model
        self.revision = str(revision).strip() if revision else None
        self.max_new_tokens = max_new_tokens
        torch_dtype = dtype if dtype == "auto" else getattr(torch, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            revision=self.revision,
            trust_remote_code=trust_remote_code,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.processor = None
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model,
                revision=self.revision,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=trust_remote_code,
            ).eval()
        except Exception as causal_exc:
            model_cls = self._multimodal_model_class()
            self.processor = AutoProcessor.from_pretrained(
                model,
                revision=self.revision,
                trust_remote_code=trust_remote_code,
            )
            if getattr(self.processor, "tokenizer", None) is not None:
                self.processor.tokenizer.padding_side = "left"
                self.tokenizer = self.processor.tokenizer
            try:
                self.model = model_cls.from_pretrained(
                    model,
                    revision=self.revision,
                    torch_dtype=torch_dtype,
                    device_map="auto",
                    trust_remote_code=trust_remote_code,
                ).eval()
            except Exception as multimodal_exc:
                raise RuntimeError(
                    f"Failed to load {model} as either a causal LM or multimodal chat model. "
                    f"causal_error={causal_exc}; multimodal_error={multimodal_exc}"
                ) from multimodal_exc

    @staticmethod
    def _multimodal_model_class():
        try:
            from transformers import Qwen3VLForConditionalGeneration

            return Qwen3VLForConditionalGeneration
        except Exception:
            pass
        for class_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
            try:
                import transformers

                return getattr(transformers, class_name)
            except Exception:
                continue
        raise RuntimeError("No compatible multimodal chat model class is available in this transformers install.")

    def _prompt(self, messages: List[Dict[str, str]]) -> str:
        template_source = self.processor if self.processor is not None else self.tokenizer
        try:
            return template_source.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return template_source.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def score(self, messages: List[Dict[str, str]], *, answer: str, pred: str) -> JudgeResponse:
        return self.score_many([JudgeRequest(messages=messages, answer=answer, pred=pred)])[0]

    def score_many(self, requests: Sequence[JudgeRequest]) -> List[JudgeResponse]:
        import torch

        prompts = [self._prompt(request.messages) for request in requests]
        if self.processor is not None:
            inputs = self.processor(text=prompts, padding=True, return_tensors="pt")
            inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            input_length = inputs["input_ids"].shape[1]
            trimmed = output[:, input_length:]
            decoder = self.processor if hasattr(self.processor, "batch_decode") else self.tokenizer
            texts = decoder.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        else:
            inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.model.device)
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            new_tokens = output[:, inputs["input_ids"].shape[1] :]
            texts = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        return [JudgeResponse(text=text.strip(), model=self.model_name) for text in texts]


class VllmJudge:
    def __init__(
        self,
        model: str,
        dtype: str,
        trust_remote_code: bool,
        tensor_parallel_size: int,
        max_new_tokens: int,
        revision: Optional[str] = None,
    ) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self.model_name = model
        self.revision = str(revision).strip() if revision else None
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            revision=self.revision,
            trust_remote_code=trust_remote_code,
        )
        self.sampling_params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        self.llm = LLM(
            model=model,
            revision=self.revision,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
        )

    def _prompt(self, messages: List[Dict[str, str]]) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def score(self, messages: List[Dict[str, str]], *, answer: str, pred: str) -> JudgeResponse:
        return self.score_many([JudgeRequest(messages=messages, answer=answer, pred=pred)])[0]

    def score_many(self, requests: Sequence[JudgeRequest]) -> List[JudgeResponse]:
        prompts = [self._prompt(request.messages) for request in requests]
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [JudgeResponse(text=output.outputs[0].text.strip(), model=self.model_name) for output in outputs]


def judge_backend_name(judge: Any) -> str:
    if isinstance(judge, DryRunJudge):
        return "dry-run"
    if isinstance(judge, OpenAICompatibleJudge):
        return "openai-compatible"
    if isinstance(judge, TransformersJudge):
        return "transformers"
    if isinstance(judge, VllmJudge):
        return "vllm"
    return type(judge).__name__


def score_batch_resilient(
    judge: Any, requests: Sequence[JudgeRequest]
) -> List[Tuple[Optional[JudgeResponse], str]]:
    if not requests:
        return []
    try:
        responses = judge.score_many(requests)
        if len(responses) != len(requests):
            raise RuntimeError(f"Judge returned {len(responses)} responses for {len(requests)} requests")
        return [(response, "") for response in responses]
    except Exception as exc:
        if len(requests) == 1:
            return [(None, str(exc))]
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        midpoint = len(requests) // 2
        return score_batch_resilient(judge, requests[:midpoint]) + score_batch_resilient(judge, requests[midpoint:])


def build_judge(args: argparse.Namespace):
    if args.dry_run or args.backend == "dry-run":
        return DryRunJudge(args.model)
    if args.backend == "openai-compatible":
        return OpenAICompatibleJudge(
            model=args.model,
            api_base=args.api_base,
            api_key=args.api_key,
            retries=args.retries,
            sleep_seconds=args.sleep_seconds,
            timeout=args.timeout,
            max_new_tokens=args.max_new_tokens,
        )
    if args.backend == "transformers":
        return TransformersJudge(
            args.model,
            args.dtype,
            args.trust_remote_code,
            args.max_new_tokens,
            args.judge_revision,
        )
    if args.backend == "vllm":
        return VllmJudge(
            args.model,
            args.dtype,
            args.trust_remote_code,
            args.tensor_parallel_size,
            args.max_new_tokens,
            args.judge_revision,
        )
    if args.backend == "auto":
        try:
            return VllmJudge(
                args.model,
                args.dtype,
                args.trust_remote_code,
                args.tensor_parallel_size,
                args.max_new_tokens,
                args.judge_revision,
            )
        except Exception as exc:
            print(f"[OPEN_MOS] vLLM unavailable, falling back to transformers: {exc}", file=sys.stderr)
            return TransformersJudge(
                args.model,
                args.dtype,
                args.trust_remote_code,
                args.max_new_tokens,
                args.judge_revision,
            )
    raise ValueError(f"Unsupported backend: {args.backend}")


def make_result(
    record: Dict[str, Any],
    sample_id: str,
    review: str,
    judge_model: str,
    error: str = "",
    method: str = "",
    run_name: str = "",
    request_fingerprint_value: str = "",
    judge_fingerprint_value: str = "",
    judge_revision: str = "",
) -> Dict[str, Any]:
    question = question_from_record(record)
    answer = ground_truth_from_record(record)
    pred = prediction_from_record(record)
    correctness = ""
    score: Union[int, str] = ""
    if not error:
        try:
            correctness, score = parse_mos_response(review)
        except ValueError as exc:
            error = f"invalid_mos_response: {exc}"
    record_method = first_value(record.get("method"), record.get("model"), record.get("run_name"))
    method = first_value(method, record_method)
    run_name = first_value(run_name, record.get("run_name"), method)
    return {
        "sample_id": sample_id,
        "doc_id": record.get("doc_id", ""),
        "run_name": run_name,
        "method": method,
        "model": first_value(record.get("model"), method, run_name),
        "video_name": video_name_from_record(record),
        "question_id": question_id_from_record(record),
        "type": task_type_from_record(record),
        "question": question,
        "answer": answer,
        "pred": pred,
        "open_mos_correctness": correctness,
        "open_mos_score": score,
        "open_mos_review": review,
        "mos_judge_model": judge_model,
        "mos_judge_revision": str(judge_revision or "").strip(),
        "request_fingerprint": request_fingerprint_value,
        "judge_fingerprint": judge_fingerprint_value,
        "error": error,
    }


def write_summary(rows: List[Dict[str, Any]], out_csv: Path, out_md: Path, results_path: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    scored = [
        r
        for r in rows
        if not r.get("error") and r.get("open_mos_score") not in (None, "")
    ]
    scores = [float(r.get("open_mos_score", 0)) for r in scored]
    yes = sum(1 for r in scored if str(r.get("open_mos_correctness", "")).lower() == "yes")
    total = len(scored)
    avg = sum(scores) / total if total else 0.0
    acc = yes / total if total else 0.0
    judge_models = sorted({str(r.get("mos_judge_model", "")) for r in scored if r.get("mos_judge_model")})
    judge_revisions = sorted(
        {str(r.get("mos_judge_revision", "")) for r in scored if r.get("mos_judge_revision")}
    )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(
        "\n".join(
            [
                "# DIVE-Bench Open MOS",
                "",
                "| samples | scored | errors | avg_open_mos | open_mos_accuracy |",
                "| ---: | ---: | ---: | ---: | ---: |",
                f"| {len(rows)} | {total} | {len(rows) - total} | {avg:.6g} | {acc:.6g} |",
                "",
                f"- judge_model: `{', '.join(judge_models) if judge_models else ''}`",
                f"- judge_revision: `{', '.join(judge_revisions) if judge_revisions else ''}`",
                f"- results_jsonl: `{results_path}`",
                f"- results_csv: `{out_csv}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Score DIVE-Bench lmms-eval sample JSONL with an open MOS judge.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--model", default=os.getenv("OPEN_MOS_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--judge-revision",
        default=os.getenv("OPEN_MOS_REVISION", ""),
        help="Optional immutable Hugging Face judge revision; passed to local model loaders and fingerprints.",
    )
    parser.add_argument("--method", default="", help="Method id to attach to scored rows for leaderboard merging.")
    parser.add_argument("--run-name", default="", help="Run name to attach to scored rows.")
    parser.add_argument("--backend", choices=["auto", "vllm", "transformers", "openai-compatible", "dry-run"], default=os.getenv("OPEN_MOS_BACKEND", "auto"))
    parser.add_argument("--dry-run", action="store_true", help="Use deterministic exact-match scoring without loading a model.")
    parser.add_argument("--api-base", default=os.getenv("OPENAI_COMPATIBLE_API_URL", "http://localhost:8000/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_COMPATIBLE_API_KEY", "EMPTY"))
    parser.add_argument("--dtype", default=os.getenv("OPEN_MOS_DTYPE", "auto"))
    parser.add_argument("--tensor-parallel-size", type=int, default=int(os.getenv("OPEN_MOS_TP", "1")))
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("OPEN_MOS_BATCH_SIZE", "1")))
    parser.add_argument("--trim-char-limit", type=int, default=int(os.getenv("OPEN_MOS_TRIM_CHAR_LIMIT", DEFAULT_TRIM_CHAR_LIMIT)))
    parser.add_argument("--limit", type=int, default=0, help="Score at most N new rows. 0 means all rows.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.judge_revision = str(args.judge_revision or "").strip()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    input_jsonl = Path(args.input_jsonl)
    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out_csv) if args.out_csv else out_jsonl.with_suffix(".csv")
    out_md = Path(args.out_md) if args.out_md else out_jsonl.with_suffix(".md")

    judge = build_judge(args)
    judge_hash = judge_fingerprint(
        model=args.model,
        revision=args.judge_revision,
        backend=judge_backend_name(judge),
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        trim_char_limit=args.trim_char_limit,
        trust_remote_code=args.trust_remote_code,
        extra={
            "tensor_parallel_size": args.tensor_parallel_size,
            "batch_size": args.batch_size,
            "api_base": args.api_base if judge_backend_name(judge) == "openai-compatible" else "",
        },
    )
    done = read_done(out_jsonl) if args.resume else {}
    completed: Dict[str, Dict[str, Any]] = {}
    written = 0

    pending: List[Tuple[Dict[str, Any], str, JudgeRequest, str]] = []
    for idx, record in enumerate(iter_jsonl(input_jsonl)):
        sample_id = sample_id_for_record(record, idx)
        question = question_from_record(record)
        answer = ground_truth_from_record(record)
        pred = prediction_from_record(record)
        request = JudgeRequest(
            messages=build_messages(question, answer, pred, args.trim_char_limit),
            answer=answer,
            pred=pred,
        )
        request_hash = request_fingerprint(request)
        if reusable_result(done.get(sample_id), request_hash, judge_hash):
            completed[sample_id] = dict(done[sample_id])
            continue
        if idx < args.offset or (args.limit and len(pending) >= args.limit):
            continue
        pending.append(
            (
                record,
                sample_id,
                request,
                request_hash,
            )
        )

    with out_jsonl.open("w", encoding="utf-8") as out_f:
        for row in completed.values():
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            outcomes = score_batch_resilient(judge, [item[2] for item in batch])
            for (record, sample_id, _, request_hash), (response, error) in zip(batch, outcomes):
                if response is not None:
                    result = make_result(
                        record,
                        sample_id,
                        response.text,
                        response.model,
                        "",
                        args.method,
                        args.run_name,
                        request_hash,
                        judge_hash,
                        args.judge_revision,
                    )
                else:
                    result = make_result(
                        record,
                        sample_id,
                        "",
                        args.model,
                        error,
                        args.method,
                        args.run_name,
                        request_hash,
                        judge_hash,
                        args.judge_revision,
                    )
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
                written += 1
            print(
                f"[OPEN_MOS] progress={written}/{len(pending)} total_completed={len(completed) + written}",
                flush=True,
            )

    rows = list(iter_jsonl(out_jsonl))
    write_summary(rows, out_csv, out_md, out_jsonl)
    errors = [row for row in rows if row.get("error")]
    if errors:
        raise RuntimeError(f"Open-MOS scoring produced {len(errors)} invalid or failed rows; rerun with --resume")
    print(f"Wrote {written} new open MOS rows to {out_jsonl}")
    print(f"Summary: {out_md}")


if __name__ == "__main__":
    main()
