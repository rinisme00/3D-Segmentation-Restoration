"""Lightweight verifier for restoration partial-to-complete NPZ pairs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_KEYS = ("partial_points", "complete_points")
OPTIONAL_KEYS = ("partial_features", "complete_features")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify restoration completion-pair preprocessing outputs.")
    parser.add_argument("--sample-index", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--code-root", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--expected-feature-mode", default=None)
    parser.add_argument("--expected-points", type=int, default=None)
    return parser.parse_args()


def resolve_output_path(raw_path: str, sample_index: Path, project_root: Path, code_root: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    candidates = [
        (sample_index.parent / path).resolve(),
        (project_root / path).resolve(),
        (code_root / path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def load_rows(sample_index: Path) -> list[dict[str, str]]:
    with sample_index.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_npz(path: Path, row: dict[str, str], expected_points: int | None) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"missing_npz:{path}"]
    try:
        payload = np.load(path, allow_pickle=True)
    except Exception as exc:  # pragma: no cover - message is the useful output in CLI use.
        return [f"npz_load_error:{path}:{exc}"]

    for key in REQUIRED_KEYS:
        if key not in payload.files:
            errors.append(f"missing_key:{key}")
            continue
        array = payload[key]
        if array.ndim != 2 or array.shape[1] != 3:
            errors.append(f"bad_shape:{key}:{array.shape}")
        if expected_points is not None and array.shape[0] != expected_points:
            errors.append(f"bad_point_count:{key}:{array.shape[0]}")

    for key in OPTIONAL_KEYS:
        if key in payload.files:
            array = payload[key]
            if array.ndim != 2:
                errors.append(f"bad_shape:{key}:{array.shape}")
            if expected_points is not None and array.shape[0] != expected_points:
                errors.append(f"bad_point_count:{key}:{array.shape[0]}")

    feature_mode = row.get("feature_mode")
    if feature_mode and "partial_features" in payload.files:
        feature_dim = payload["partial_features"].shape[1]
        expected_dim = 9 if feature_mode == "9d" else 3
        if feature_dim != expected_dim:
            errors.append(f"bad_feature_dim:{feature_dim}:expected_{expected_dim}")
    return errors


def main() -> int:
    args = parse_args()
    sample_index = args.sample_index.resolve()
    project_root = args.project_root.resolve()
    code_root = args.code_root.resolve() if args.code_root else project_root / "3D-Segmentation-Restoration"

    rows = load_rows(sample_index)
    checked = 0
    missing_sources = 0
    failures: list[dict[str, Any]] = []
    feature_modes: dict[str, int] = {}

    for row in rows[: max(args.limit, 0)]:
        checked += 1
        feature_mode = row.get("feature_mode", "")
        feature_modes[feature_mode] = feature_modes.get(feature_mode, 0) + 1
        if args.expected_feature_mode and feature_mode != args.expected_feature_mode:
            failures.append({"sample": row.get("sample_name"), "errors": [f"feature_mode:{feature_mode}"]})
            continue
        for source_key in ("partial_path", "complete_path"):
            source = row.get(source_key)
            if source and not Path(source).exists():
                missing_sources += 1
                failures.append({"sample": row.get("sample_name"), "errors": [f"missing_{source_key}:{source}"]})
        output_path = resolve_output_path(row.get("output_path", ""), sample_index, project_root, code_root)
        errors = verify_npz(output_path, row, args.expected_points)
        if errors:
            failures.append({"sample": row.get("sample_name"), "output_path": str(output_path), "errors": errors})

    summary = {
        "sample_index": str(sample_index),
        "total_rows": len(rows),
        "checked_rows": checked,
        "feature_modes_checked": feature_modes,
        "missing_sources": missing_sources,
        "failure_count": len(failures),
        "failures": failures[:20],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
