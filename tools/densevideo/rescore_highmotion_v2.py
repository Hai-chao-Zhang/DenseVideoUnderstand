"""CPU-only rescoring; never reruns a model or edits an existing benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from tools.densevideo.highmotion_v2_scoring import require, rescore_records


def checked_bytes(path, expected):
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, "Input SHA-256 mismatch: " + path.name)
    return raw


def records_from_jsonl(raw):
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    require(all(isinstance(row, dict) for row in rows), "Prediction records must be objects")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--references-sha256", required=True)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--predictions-sha256", required=True)
    parser.add_argument("--prompt-cache", required=True, type=Path)
    parser.add_argument("--prompt-cache-sha256", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--prediction-source", choices=("archived_baseline", "new_grt"), required=True)
    parser.add_argument("--expected-count", type=int, choices=(1000, 3243), default=1000)
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing paths are refused")
    args = parser.parse_args(argv)
    require(not args.output.exists() and not args.output.is_symlink(), "Output already exists")

    import pyarrow as pa
    import pyarrow.parquet as pq

    references = pq.read_table(pa.BufferReader(checked_bytes(args.references, args.references_sha256))).to_pylist()
    require(len(references) == 3243, "Expected all 3,243 source records in the corrected reference")
    records = records_from_jsonl(checked_bytes(args.predictions, args.predictions_sha256))
    prompt_records = records_from_jsonl(checked_bytes(args.prompt_cache, args.prompt_cache_sha256))
    require(len(prompt_records) == args.expected_count, "Wrong prompt-cache sample count")
    for index, row in enumerate(prompt_records):
        require(type(row.get("doc_id")) is int and row["doc_id"] == index, "Wrong prompt-cache ordering")
    report = rescore_records(references, records, expected_count=args.expected_count,
                             expected_inputs=[row.get("input") for row in prompt_records])
    report.update({
        "method": args.method, "prediction_source": args.prediction_source,
        "references_sha256": args.references_sha256, "predictions_sha256": args.predictions_sha256,
        "prompt_cache_sha256": args.prompt_cache_sha256,
        "source_reference_records": len(references),
        "full_source_coverage": args.expected_count == len(references),
        "comparison_caveat": "Archived runs have no consumed-tensor hashes or pinned weight revisions; configuration/prompt checks are not a fresh byte-identical paired reproduction",
    })
    os.umask(0o077)
    args.output.mkdir(mode=0o700)
    with (args.output / "rescoring.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({key: value for key, value in report.items() if key != "records_detail"},
                     indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
