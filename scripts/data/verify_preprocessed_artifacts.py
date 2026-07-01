#!/usr/bin/env python3
"""Verify preprocessed artifact indexes and summaries without loading NPZ payloads."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_OUTPUTS = [
    {
        "name": "segmentation/pointnext_3d",
        "relative": "preprocessed/segmentation/pointnext_3d",
        "expected_feature_mode": "3d",
        "expected_num_points": 8192,
        "requires_label_quality": True,
    },
    {
        "name": "segmentation/pointnext_9d",
        "relative": "preprocessed/segmentation/pointnext_9d",
        "expected_feature_mode": "9d",
        "expected_num_points": 8192,
        "requires_label_quality": True,
    },
    {
        "name": "restoration/completion_pairs_3d",
        "relative": "preprocessed/restoration/completion_pairs_3d",
        "expected_feature_mode": "3d",
        "expected_num_points_partial": 8192,
        "expected_num_points_complete": 8192,
        "requires_label_quality": False,
    },
    {
        "name": "restoration/completion_pairs_9d",
        "relative": "preprocessed/restoration/completion_pairs_9d",
        "expected_feature_mode": "9d",
        "expected_num_points_partial": 8192,
        "expected_num_points_complete": 8192,
        "requires_label_quality": False,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd().parent,
        help="Project root containing preprocessed/. Defaults to parent of cwd.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a text summary.",
    )
    return parser.parse_args()


def count_npz(samples_dir: Path) -> int:
    if not samples_dir.exists():
        return 0
    total = 0
    for _, _, files in os.walk(samples_dir):
        total += sum(1 for name in files if name.endswith(".npz"))
    return total


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {"_value": data}


def counter_from_rows(rows: list[dict[str, str]], column: str) -> dict[str, int]:
    return dict(Counter(row.get(column, "") for row in rows))


def aggregate_count_rows(rows: list[dict[str, str]], key_column: str, count_column: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        key = row.get(key_column, "")
        try:
            count = int(row.get(count_column, "1"))
        except ValueError:
            count = 1
        counts[key] += count
    return dict(counts)


def first_present_key(row: dict[str, str], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key, "")
        if value:
            return value
    return ""


def check_path_layout(index_rows: list[dict[str, str]]) -> dict[str, Any]:
    outputs = [first_present_key(row, ("output_path", "partial_output_path")) for row in index_rows]
    outputs = [path for path in outputs if path]
    bb_artifact = [path for path in outputs if "BreakingBad/artifact/" in path]
    bb_everyday = [path for path in outputs if "BreakingBad/everyday/" in path]
    fb = [path for path in outputs if "FantasticBreaks/" in path or "Fantastic_Breaks" in path]
    return {
        "breakingbad_artifact_paths": len(bb_artifact),
        "breakingbad_everyday_paths": len(bb_everyday),
        "fantastic_breaks_paths": len(fb),
        "breakingbad_everyday_has_category": all(
            len(Path(path).parts) >= 5 and "everyday" in Path(path).parts for path in bb_everyday[:100]
        ),
        "preview_paths": outputs[:3],
    }


def verify_one(project_root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    output_dir = project_root / spec["relative"]
    samples_dir = output_dir / "samples"
    index_path = output_dir / "sample_index.csv"
    summary_path = output_dir / "preprocess_summary.json"
    label_quality_path = output_dir / "label_quality.csv"
    result: dict[str, Any] = {
        "name": spec["name"],
        "path": str(output_dir),
        "exists": output_dir.exists(),
        "samples_exists": samples_dir.exists(),
        "index_exists": index_path.exists(),
        "summary_exists": summary_path.exists(),
        "label_quality_exists": label_quality_path.exists(),
        "npz_files": count_npz(samples_dir),
        "index_rows": None,
        "summary_written_samples": None,
        "summary_planned_samples": None,
        "summary_feature_mode": None,
        "summary_num_points": None,
        "summary_num_points_partial": None,
        "summary_num_points_complete": None,
        "feature_mode_counts": {},
        "num_points_counts": {},
        "num_points_partial_counts": {},
        "num_points_complete_counts": {},
        "label_quality_counts": {},
        "path_layout": {},
        "status": "incomplete",
        "issues": [],
    }

    if not output_dir.exists():
        result["issues"].append("output directory is missing")
        return result
    if not samples_dir.exists():
        result["issues"].append("samples directory is missing")
    if not index_path.exists():
        result["issues"].append("sample_index.csv is missing")
    if not summary_path.exists():
        result["issues"].append("preprocess_summary.json is missing")
    if spec.get("requires_label_quality") and not label_quality_path.exists():
        result["issues"].append("label_quality.csv is missing")

    index_rows: list[dict[str, str]] = []
    if index_path.exists():
        index_rows = read_csv_rows(index_path)
        result["index_rows"] = len(index_rows)
        result["feature_mode_counts"] = counter_from_rows(index_rows, "feature_mode")
        result["num_points_counts"] = counter_from_rows(index_rows, "num_points")
        result["num_points_partial_counts"] = counter_from_rows(index_rows, "num_points_partial")
        result["num_points_complete_counts"] = counter_from_rows(index_rows, "num_points_complete")
        result["path_layout"] = check_path_layout(index_rows)

        if result["npz_files"] != len(index_rows):
            result["issues"].append(f"npz file count {result['npz_files']} != index rows {len(index_rows)}")
        expected_feature_mode = spec.get("expected_feature_mode")
        if expected_feature_mode and set(result["feature_mode_counts"]) != {expected_feature_mode}:
            result["issues"].append(
                f"feature_mode counts do not match expected {expected_feature_mode}: "
                f"{result['feature_mode_counts']}"
            )
        expected_num_points = spec.get("expected_num_points")
        if expected_num_points is not None and set(result["num_points_counts"]) != {str(expected_num_points)}:
            result["issues"].append(
                f"num_points counts do not match expected {expected_num_points}: {result['num_points_counts']}"
            )
        expected_partial = spec.get("expected_num_points_partial")
        if expected_partial is not None and set(result["num_points_partial_counts"]) != {str(expected_partial)}:
            result["issues"].append(
                "num_points_partial counts do not match expected "
                f"{expected_partial}: {result['num_points_partial_counts']}"
            )
        expected_complete = spec.get("expected_num_points_complete")
        if expected_complete is not None and set(result["num_points_complete_counts"]) != {str(expected_complete)}:
            result["issues"].append(
                "num_points_complete counts do not match expected "
                f"{expected_complete}: {result['num_points_complete_counts']}"
            )

    if summary_path.exists():
        summary = read_json(summary_path)
        result["summary_written_samples"] = summary.get("written_samples")
        result["summary_planned_samples"] = summary.get("planned_samples")
        result["summary_feature_mode"] = summary.get("feature_mode")
        result["summary_num_points"] = summary.get("num_points")
        result["summary_num_points_partial"] = summary.get("num_points_partial")
        result["summary_num_points_complete"] = summary.get("num_points_complete")

        if index_rows and summary.get("written_samples") != len(index_rows):
            result["issues"].append(
                f"summary written_samples {summary.get('written_samples')} != index rows {len(index_rows)}"
            )
        if summary.get("feature_mode") != spec.get("expected_feature_mode"):
            result["issues"].append(
                f"summary feature_mode {summary.get('feature_mode')} != expected {spec.get('expected_feature_mode')}"
            )
        if "expected_num_points" in spec and summary.get("num_points") != spec["expected_num_points"]:
            result["issues"].append(
                f"summary num_points {summary.get('num_points')} != expected {spec['expected_num_points']}"
            )
        if (
            "expected_num_points_partial" in spec
            and summary.get("num_points_partial") != spec["expected_num_points_partial"]
        ):
            result["issues"].append(
                "summary num_points_partial "
                f"{summary.get('num_points_partial')} != expected {spec['expected_num_points_partial']}"
            )
        if (
            "expected_num_points_complete" in spec
            and summary.get("num_points_complete") != spec["expected_num_points_complete"]
        ):
            result["issues"].append(
                "summary num_points_complete "
                f"{summary.get('num_points_complete')} != expected {spec['expected_num_points_complete']}"
            )

    if label_quality_path.exists():
        rows = read_csv_rows(label_quality_path)
        if rows and "count" in rows[0]:
            result["label_quality_counts"] = aggregate_count_rows(rows, "label_quality", "count")
        else:
            result["label_quality_counts"] = counter_from_rows(rows, "label_quality")

    if not result["issues"]:
        result["status"] = "pass"
    elif result["npz_files"] > 0 and (not index_path.exists() or not summary_path.exists()):
        result["status"] = "interrupted_or_incomplete"
    else:
        result["status"] = "fail"
    return result


def print_text_report(results: list[dict[str, Any]]) -> None:
    print("# Preprocessed Artifact Verification")
    print()
    for result in results:
        print(f"## {result['name']}")
        print(f"status: {result['status']}")
        print(f"path: {result['path']}")
        print(f"npz_files: {result['npz_files']}")
        print(f"index_rows: {result['index_rows']}")
        print(f"summary_written_samples: {result['summary_written_samples']}")
        print(f"feature_mode_counts: {result['feature_mode_counts']}")
        if result["num_points_counts"] and set(result["num_points_counts"]) != {""}:
            print(f"num_points_counts: {result['num_points_counts']}")
        if result["num_points_partial_counts"] and set(result["num_points_partial_counts"]) != {""}:
            print(f"num_points_partial_counts: {result['num_points_partial_counts']}")
        if result["num_points_complete_counts"] and set(result["num_points_complete_counts"]) != {""}:
            print(f"num_points_complete_counts: {result['num_points_complete_counts']}")
        if result["label_quality_counts"]:
            print(f"label_quality_counts: {result['label_quality_counts']}")
        if result["path_layout"]:
            print(f"path_layout: {result['path_layout']}")
        if result["issues"]:
            print("issues:")
            for issue in result["issues"]:
                print(f"- {issue}")
        print()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    results = [verify_one(project_root, spec) for spec in EXPECTED_OUTPUTS]
    if args.json:
        print(json.dumps({"project_root": str(project_root), "outputs": results}, indent=2, sort_keys=True))
    else:
        print_text_report(results)


if __name__ == "__main__":
    main()
