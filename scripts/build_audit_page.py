#!/usr/bin/env python3
"""Publish outputs from the minimal release's canonical, source-bound generator.

Install the minimal package or supply --minimal-root /path/to/DIVE-Bench.
DIVE_MINIMAL_ROOT is the equivalent environment setting for tests and CI.
This wrapper contains no independent numeric inputs or ranking logic.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_outputs(minimal_root=None):
    location = minimal_root or os.environ.get("DIVE_MINIMAL_ROOT")
    root = Path(location).expanduser().resolve() if location else None
    if root is not None:
        module_path = root / "tools/densevideo/build_complete_leaderboard.py"
        if not module_path.is_file():
            raise ValueError("--minimal-root does not contain the canonical generator")
        sys.path.insert(0, str(root))
    try:
        canonical = importlib.import_module("tools.densevideo.build_complete_leaderboard")
    except ImportError as error:
        raise ValueError(
            "Install the minimal release package or supply --minimal-root /path/to/DIVE-Bench"
        ) from error
    if root is not None and not Path(canonical.__file__).resolve().is_relative_to(root):
        raise ValueError("Imported generator does not belong to the requested --minimal-root")
    return canonical.build_outputs()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check all generated assets without writes")
    parser.add_argument("--minimal-root", type=Path, help="Explicit minimal release source checkout")
    args = parser.parse_args(argv)
    try:
        outputs = build_outputs(args.minimal_root)
        for relative, text in outputs.items():
            path = ROOT / relative
            if args.check:
                if not path.exists() or path.read_text(encoding="utf-8") != text:
                    raise ValueError("Generated file is stale: " + relative)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Canonical leaderboard build failed: {error}\n")
    print("Canonical screened HTML, CSV and evidence assets are consistent.")


if __name__ == "__main__":
    main()
