"""Shared utilities for lightweight preprocessing builders."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from src.common.geometry.pointcloud import (
    build_features,
    load_point_cloud,
    normalize_points,
    pointcloud_summary,
    save_json_summary,
)
from src.common.manifests import MANIFEST_COLUMNS, ManifestRow, validate_manifest_row


@dataclass
class BuildArgs:
    manifest: Path
    output_dir: Path
    feature_mode: str
    limit: int | None
    dry_run: bool
    seed: int
    num_points: int | None = None
    num_points_partial: int | None = None
    num_points_complete: int | None = None
    model_family: str = ""
    config: Path | None = None
    debug: bool = False
    overwrite: bool = False
    resume: bool = False
    verbose: bool = False


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON/YAML config file.")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--feature-mode", choices=["3d", "9d"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    parser.add_argument("--resume", action="store_true", help="Skip already-written samples; continue a partial run.")
    parser.add_argument("--verbose", action="store_true", help="Print per-mesh W/S status and pre-run existing count.")


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML config support requires PyYAML; use JSON or install PyYAML") from exc
        data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must contain a mapping: {path}")
    return data


def config_value(cli_value: Any, config: dict[str, Any], key: str, default: Any = None) -> Any:
    return cli_value if cli_value is not None else config.get(key, default)


def require_path(value: Any, name: str) -> Path:
    if value is None or str(value).strip() == "":
        raise ValueError(f"{name} is required")
    return Path(value)


def iter_manifest_rows(path: Path, expected_stage: str, limit: int | None = None) -> Iterator[ManifestRow]:
    yielded = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = [column for column in MANIFEST_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"manifest missing required columns: {missing}")
        for row_number, raw in enumerate(reader, start=2):
            row = ManifestRow.from_csv_row(raw, row_number=row_number)
            errors = validate_manifest_row(row, expected_stage=expected_stage)
            if errors:
                raise ValueError(f"row {row_number}: " + "; ".join(errors))
            yield row
            yielded += 1
            if limit is not None and yielded >= limit:
                break


def ensure_output_dir(path: Path, dry_run: bool, overwrite: bool = False, resume: bool = False) -> None:
    if not dry_run:
        if path.exists() and any(path.iterdir()):
            if resume:
                pass  # keep existing partial data; per-sample skip handled by callers
            elif not overwrite:
                raise FileExistsError(f"output_dir exists and is not empty; use --overwrite to replace it: {path}")
            else:
                shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "samples").mkdir(parents=True, exist_ok=True)


def matching_feature_rows(path: Path, expected_stage: str, feature_mode: str) -> Iterator[ManifestRow]:
    for row in iter_manifest_rows(path, expected_stage=expected_stage):
        if row.feature_mode == feature_mode:
            yield row


def safe_part(value: Any, default: str = "unknown") -> str:
    text = str(value if value not in {None, ""} else default)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return text.strip("._") or default


def variant_from_row(row: ManifestRow) -> str:
    metadata = row.metadata
    for key in ("variant", "partial_variant", "partial_role", "mesh_role", "file_role", "complete_role"):
        if metadata.get(key):
            return safe_part(metadata[key])
    return safe_part(row.sample_id)


def sample_output_parts(row: ManifestRow) -> list[str]:
    parts = [safe_part(row.dataset_name)]
    subset = row.metadata.get("subset")
    if row.dataset_name == "BreakingBad" and subset:
        parts.append(safe_part(subset))
        if str(subset) != "artifact":
            parts.append(safe_part(row.category))
        parts.append(safe_part(row.object_id))
        return parts
    parts.extend([safe_part(row.category), safe_part(row.object_id)])
    return parts


def sample_output_path(output_dir: Path, row: ManifestRow, leaf_name: str) -> Path:
    path = output_dir / "samples"
    for part in sample_output_parts(row):
        path /= part
    return path / variant_from_row(row) / safe_part(row.feature_mode) / f"{safe_part(leaf_name)}.npz"


def source_meshes_from_row(row: ManifestRow) -> list[Path]:
    source_paths = row.source_paths
    if source_paths.get("mesh"):
        return [Path(source_paths["mesh"])]
    if source_paths.get("pieces_dir"):
        pieces_dir = Path(source_paths["pieces_dir"])
        piece_glob = source_paths.get("piece_glob", "piece_*.obj")
        return sorted(pieces_dir.glob(piece_glob))
    return []


def path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def process_point_file(path: Path, num_points: int, feature_mode: str, seed: int) -> dict[str, Any]:
    cloud = load_point_cloud(path, sample_points=num_points, seed=seed)
    normalized, norm_meta = normalize_points(cloud.points)
    features, feature_meta = build_features(normalized, feature_mode=feature_mode, normals=cloud.normals)
    return {
        "points": normalized.astype(np.float32),
        "features": features.astype(np.float32),
        "normalization": norm_meta,
        "features_meta": feature_meta,
        "pointcloud": pointcloud_summary(normalized, features),
        "load_metadata": cloud.metadata or {},
    }


def write_npz_sample(path: Path, payload: dict[str, Any]) -> None:
    array_payload = {k: v for k, v in payload.items() if isinstance(v, np.ndarray)}
    metadata_payload = {k: json_safe(v) for k, v in payload.items() if not isinstance(v, np.ndarray)}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **array_payload, metadata=json.dumps(metadata_payload, sort_keys=True))


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items() if not isinstance(v, np.ndarray)}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value if not isinstance(v, np.ndarray)]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_common(
    *,
    builder: str,
    args: BuildArgs,
    planned: int,
    written: int,
    skipped: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "builder": builder,
        "dry_run": args.dry_run,
        "manifest": str(args.manifest),
        "output_dir": str(args.output_dir),
        "feature_mode": args.feature_mode,
        "limit": args.limit,
        "seed": args.seed,
        "planned_samples": planned,
        "written_samples": written,
        "skipped_samples": skipped,
    }
    if args.num_points is not None:
        summary["num_points"] = args.num_points
    if args.num_points_partial is not None:
        summary["num_points_partial"] = args.num_points_partial
    if args.num_points_complete is not None:
        summary["num_points_complete"] = args.num_points_complete
    if args.model_family:
        summary["model_family"] = args.model_family
    summary["overwrite"] = args.overwrite
    summary["resume"] = args.resume
    if extra:
        summary.update(extra)
    return summary


def print_or_write_summary(args: BuildArgs, summary: dict[str, Any]) -> None:
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.dry_run:
        print(text)
    else:
        save_json_summary(args.output_dir / "preprocess_summary.json", summary)
        print(text)


def limited(iterable: Iterable[Any], limit: int | None) -> Iterator[Any]:
    count = 0
    for item in iterable:
        yield item
        count += 1
        if limit is not None and count >= limit:
            break


def count_existing_npz(output_dir: Path) -> int:
    """Count .npz files already on disk under output_dir (for pre-run progress reporting)."""
    samples_dir = output_dir / "samples"
    if not samples_dir.exists():
        return 0
    return sum(1 for _ in samples_dir.rglob("*.npz"))
