"""Unified manifest schema for segmentation and restoration."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

STAGES = {"segmentation", "restoration"}
FEATURE_MODES = {"3d", "9d"}
SPLITS = {"train", "val", "test", ""}
ANNOTATION_SOURCE_STATUSES = {"none", "blender_pending", "converted", "invalid", "review"}

MANIFEST_COLUMNS = [
    "dataset_name",
    "stage",
    "object_id",
    "category",
    "sample_id",
    "source_paths",
    "label_paths",
    "class_label",
    "point_labels_available",
    "partial_path",
    "complete_path",
    "feature_mode",
    "split",
    "metadata",
]


class ManifestValidationError(ValueError):
    """Raised when manifest rows fail validation."""


@dataclass
class ManifestRow:
    dataset_name: str
    stage: str
    object_id: str
    category: str
    sample_id: str
    source_paths: dict[str, str] = field(default_factory=dict)
    label_paths: dict[str, str] = field(default_factory=dict)
    class_label: int | None = None
    point_labels_available: bool = False
    partial_path: str = ""
    complete_path: str = ""
    feature_mode: str = "3d"
    split: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_csv_row(self) -> dict[str, str]:
        return {
            "dataset_name": self.dataset_name,
            "stage": self.stage,
            "object_id": self.object_id,
            "category": self.category,
            "sample_id": self.sample_id,
            "source_paths": json.dumps(self.source_paths, sort_keys=True),
            "label_paths": json.dumps(self.label_paths, sort_keys=True),
            "class_label": "" if self.class_label is None else str(int(self.class_label)),
            "point_labels_available": "true" if self.point_labels_available else "false",
            "partial_path": self.partial_path,
            "complete_path": self.complete_path,
            "feature_mode": self.feature_mode,
            "split": self.split,
            "metadata": json.dumps(self.metadata, sort_keys=True),
        }

    @classmethod
    def from_csv_row(cls, row: dict[str, str], row_number: int | None = None) -> "ManifestRow":
        missing = [column for column in MANIFEST_COLUMNS if column not in row]
        if missing:
            prefix = f"row {row_number}: " if row_number is not None else ""
            raise ManifestValidationError(f"{prefix}missing columns: {missing}")
        try:
            source_paths = _parse_json_object(row["source_paths"], "source_paths", row_number)
            label_paths = _parse_json_object(row["label_paths"], "label_paths", row_number)
            metadata = _parse_json_object(row["metadata"], "metadata", row_number)
        except json.JSONDecodeError as exc:
            prefix = f"row {row_number}: " if row_number is not None else ""
            raise ManifestValidationError(f"{prefix}invalid JSON: {exc}") from exc
        class_label = row["class_label"].strip()
        return cls(
            dataset_name=row["dataset_name"].strip(),
            stage=row["stage"].strip(),
            object_id=row["object_id"].strip(),
            category=row["category"].strip(),
            sample_id=row["sample_id"].strip(),
            source_paths=source_paths,
            label_paths=label_paths,
            class_label=None if class_label == "" else int(class_label),
            point_labels_available=_parse_bool(row["point_labels_available"], row_number),
            partial_path=row["partial_path"].strip(),
            complete_path=row["complete_path"].strip(),
            feature_mode=row["feature_mode"].strip(),
            split=row["split"].strip(),
            metadata=metadata,
        )


def _parse_json_object(value: str, field_name: str, row_number: int | None) -> dict[str, Any]:
    if value.strip() == "":
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        prefix = f"row {row_number}: " if row_number is not None else ""
        raise ManifestValidationError(f"{prefix}{field_name} must be a JSON object")
    return parsed


def _parse_bool(value: str, row_number: int | None) -> bool:
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    prefix = f"row {row_number}: " if row_number is not None else ""
    raise ManifestValidationError(f"{prefix}point_labels_available must be true or false")


def read_manifest_csv(path: str | Path, limit: int | None = None) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=2):
            rows.append(ManifestRow.from_csv_row(row, row_number=idx))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_manifest_csv(path: str | Path, rows: list[ManifestRow]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_row())


def validate_manifest_rows(rows: list[ManifestRow], stage: str | None = None) -> dict[str, Any]:
    errors: list[str] = []
    stage_counts: dict[str, int] = {}
    annotation_counts: dict[str, int] = {}
    for idx, row in enumerate(rows, start=1):
        row_errors = validate_manifest_row(row, expected_stage=stage)
        errors.extend(f"row {idx}: {error}" for error in row_errors)
        stage_counts[row.stage] = stage_counts.get(row.stage, 0) + 1
        status = str(row.metadata.get("annotation_source_status", "none"))
        annotation_counts[status] = annotation_counts.get(status, 0) + 1
    if errors:
        raise ManifestValidationError("\n".join(errors))
    return {
        "row_count": len(rows),
        "stage_counts": stage_counts,
        "annotation_source_status_counts": annotation_counts,
    }


def validate_manifest_row(row: ManifestRow, expected_stage: str | None = None) -> list[str]:
    errors: list[str] = []
    for field_name in ("dataset_name", "stage", "object_id", "sample_id", "feature_mode"):
        if not str(getattr(row, field_name)).strip():
            errors.append(f"{field_name} is required")
    if row.stage not in STAGES:
        errors.append(f"stage must be one of {sorted(STAGES)}, got {row.stage!r}")
    if expected_stage is not None and row.stage != expected_stage:
        errors.append(f"expected stage {expected_stage!r}, got {row.stage!r}")
    if row.feature_mode not in FEATURE_MODES:
        errors.append(f"feature_mode must be one of {sorted(FEATURE_MODES)}, got {row.feature_mode!r}")
    if row.split not in SPLITS:
        errors.append(f"split must be train, val, test, or empty; got {row.split!r}")
    annotation_status = str(row.metadata.get("annotation_source_status", "none"))
    if annotation_status not in ANNOTATION_SOURCE_STATUSES:
        errors.append(
            "metadata.annotation_source_status must be one of "
            f"{sorted(ANNOTATION_SOURCE_STATUSES)}, got {annotation_status!r}"
        )
    if row.stage == "segmentation":
        errors.extend(_validate_segmentation_row(row, annotation_status))
    elif row.stage == "restoration":
        errors.extend(_validate_restoration_row(row))
    return errors


def _validate_segmentation_row(row: ManifestRow, annotation_status: str) -> list[str]:
    errors: list[str] = []
    if row.class_label is not None:
        errors.append("segmentation rows must leave class_label empty")
    if row.point_labels_available and not row.label_paths:
        errors.append("point_labels_available=true requires label_paths")
    if annotation_status == "blender_pending":
        if row.point_labels_available:
            errors.append("blender_pending rows cannot set point_labels_available=true")
        if row.label_paths:
            errors.append("blender_pending rows cannot include converted label_paths")
    return errors


def _validate_restoration_row(row: ManifestRow) -> list[str]:
    errors: list[str] = []
    if row.class_label is not None:
        errors.append("restoration rows must leave class_label empty")
    if row.point_labels_available:
        errors.append("restoration rows must set point_labels_available=false")
    if not row.partial_path:
        errors.append("restoration rows require partial_path")
    if not row.complete_path:
        errors.append("restoration rows require complete_path")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a unified manifest CSV.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage", choices=sorted(STAGES), default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_manifest_csv(args.manifest, limit=args.limit)
    summary = validate_manifest_rows(rows, stage=args.stage)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except ManifestValidationError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
