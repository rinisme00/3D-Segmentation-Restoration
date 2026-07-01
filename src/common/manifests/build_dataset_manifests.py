"""Build lightweight dataset manifests without point-cloud preprocessing."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from src.common.manifests import MANIFEST_COLUMNS, ManifestRow, validate_manifest_row

FEATURE_MODES = ("3d", "9d")


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def limited(items: Iterable[Path], limit: int | None) -> list[Path]:
    result = sorted(items)
    return result if limit is None else result[:limit]


def path_exists(path: Path) -> str:
    return str(path.resolve()) if path.exists() else ""


def breakingbad_variant_class(variant: str) -> tuple[int, str, str]:
    if variant == "mode_0":
        return 0, "completed", "completed_canonical"
    if variant.startswith("mode_"):
        return 1, "broken", "broken_synthetic_mode"
    if variant.startswith("fractured_"):
        return 1, "broken", "broken_fractured"
    return 1, "broken", "broken_unknown_variant"


def build_fantastic_segmentation(data_root: Path, limit: int | None) -> Iterable[ManifestRow]:
    fb_root = data_root / "Fantastic_Breaks_v1"
    object_dirs = limited((p for p in fb_root.glob("*/*") if p.is_dir()), limit)
    for obj_dir in object_dirs:
        category = obj_dir.parent.name
        object_id = obj_dir.name
        meta_paths = sorted(obj_dir.glob("meta_*.npz"))
        mask_path = meta_paths[0] if meta_paths else None
        for stem in ("model_b_0", "model_r_0"):
            mesh_path = obj_dir / f"{stem}.ply"
            if not mesh_path.exists():
                continue
            for feature_mode in FEATURE_MODES:
                yield ManifestRow(
                    dataset_name="FantasticBreaks",
                    stage="segmentation",
                    object_id=object_id,
                    category=category,
                    sample_id=f"{category}_{object_id}_{stem}_{feature_mode}",
                    source_paths={"mesh": str(mesh_path.resolve())},
                    label_paths={"mask_candidate": str(mask_path.resolve())} if mask_path else {},
                    point_labels_available=bool(mask_path),
                    feature_mode=feature_mode,
                    metadata={
                        "mesh_role": stem,
                        "mask_source": "meta_npz_mask_candidate" if mask_path else "missing",
                        "mask_alignment_status": "needs_loader_verification" if mask_path else "missing",
                        "split_status": "unassigned",
                        "annotation_source_status": "none",
                    },
                )


def build_breakingbad_raw_segmentation(data_root: Path, limit: int | None) -> Iterable[ManifestRow]:
    bb_root = data_root / "BreakingBad"
    variant_dirs = sorted(p for p in (bb_root / "artifact").glob("*/*") if p.is_dir())
    variant_dirs.extend(sorted(p for p in (bb_root / "everyday").glob("*/*/*") if p.is_dir()))
    if limit is not None:
        variant_dirs = variant_dirs[:limit]
    for variant_dir in variant_dirs:
        piece_count = sum(1 for _ in variant_dir.glob("piece_*.obj"))
        if piece_count == 0:
            continue
        rel_parts = variant_dir.relative_to(bb_root).parts
        subset = rel_parts[0]
        if subset == "artifact":
            category = "artifact"
            object_id = rel_parts[1]
            variant = rel_parts[2]
        else:
            category = rel_parts[1]
            object_id = rel_parts[2]
            variant = rel_parts[3]
        _, _, variant_role = breakingbad_variant_class(variant)
        for feature_mode in FEATURE_MODES:
            yield ManifestRow(
                dataset_name="BreakingBad",
                stage="segmentation",
                object_id=object_id,
                category=category,
                sample_id=f"{subset}_{category}_{object_id}_{variant}_{feature_mode}",
                source_paths={"pieces_dir": str(variant_dir.resolve()), "piece_glob": "piece_*.obj"},
                point_labels_available=False,
                feature_mode=feature_mode,
                metadata={
                    "subset": subset,
                    "variant": variant,
                    "variant_role": variant_role,
                    "piece_count": piece_count,
                    "label_status": "unlabeled_raw",
                    "manifest_granularity": "variant",
                    "split_status": "unassigned",
                    "annotation_source_status": "none",
                },
            )


def observed_material_names(mtl_path: Path) -> list[str]:
    names: list[str] = []
    try:
        with mtl_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("newmtl "):
                    names.append(line.split(None, 1)[1].strip())
    except OSError:
        pass
    return names


def canonical_annotation_pair(path: Path) -> tuple[Path, str]:
    name = path.name
    if name.endswith("_annotated.obj"):
        return path.with_suffix(".mtl"), "canonical"
    if name.endswith("_annotated.mtl.obj"):
        return Path(str(path).removesuffix(".obj") + ".mtl"), "filename_repair_needed"
    return path.with_suffix(".mtl"), "unknown"


def build_breakingbad_annotation_sources(project_root: Path, data_root: Path, limit: int | None) -> Iterable[ManifestRow]:
    annotations_root = project_root / "annotations" / "BreakingBad"
    obj_paths = sorted(annotations_root.glob("**/*_annotated.obj"))
    obj_paths.extend(sorted(annotations_root.glob("**/*_annotated.mtl.obj")))
    if limit is not None:
        obj_paths = obj_paths[:limit]
    for annotated_obj in obj_paths:
        parts = annotated_obj.relative_to(annotations_root).parts
        subset = parts[0]
        if subset == "artifact":
            category = "artifact"
            object_id = parts[1]
            variant = parts[2]
            piece_name = annotated_obj.name.split("_annotated", 1)[0]
            raw_path = data_root / "BreakingBad" / subset / object_id / variant / f"{piece_name}.obj"
        else:
            category = parts[1]
            object_id = parts[2]
            variant = parts[3]
            piece_name = annotated_obj.name.split("_annotated", 1)[0]
            raw_path = data_root / "BreakingBad" / subset / category / object_id / variant / f"{piece_name}.obj"
        annotated_mtl, filename_status = canonical_annotation_pair(annotated_obj)
        material_names = observed_material_names(annotated_mtl)
        material_alias_repair_needed = any(
            name.startswith(("intact_surface", "fracture_surface", "fractured_surface"))
            for name in material_names
        )
        for feature_mode in FEATURE_MODES:
            yield ManifestRow(
                dataset_name="BreakingBad",
                stage="segmentation",
                object_id=object_id,
                category=category,
                sample_id=f"{subset}_{category}_{object_id}_{variant}_{piece_name}_annotation_{feature_mode}",
                source_paths={
                    "raw_mesh": path_exists(raw_path),
                    "annotated_obj": str(annotated_obj.resolve()),
                    "annotated_mtl": path_exists(annotated_mtl),
                },
                point_labels_available=False,
                feature_mode=feature_mode,
                metadata={
                    "subset": subset,
                    "variant": variant,
                    "piece_id": piece_name,
                    "annotation_source_status": "blender_pending",
                    "filename_repair_needed": filename_status != "canonical",
                    "filename_status": filename_status,
                    "material_vocab_observed": material_names,
                    "material_alias_repair_needed": material_alias_repair_needed,
                    "raw_path_exists": raw_path.exists(),
                    "split_status": "unassigned",
                },
            )


def build_garf_segmentation(data_root: Path) -> Iterable[ManifestRow]:
    mapping = {
        "bone_synthetic": "bone_synthetic.hdf5",
        "bone_real": "bone_real.hdf5",
        "fractura_real": "fractura_real.hdf5",
    }
    for dataset_name, file_name in mapping.items():
        path = data_root / "garf" / file_name
        if not path.exists():
            continue
        for feature_mode in FEATURE_MODES:
            yield ManifestRow(
                dataset_name=dataset_name,
                stage="segmentation",
                object_id=path.stem,
                category="garf",
                sample_id=f"{path.stem}_{feature_mode}",
                source_paths={"hdf5": str(path.resolve())},
                point_labels_available=False,
                feature_mode=feature_mode,
                metadata={
                    "source_format": "hdf5",
                    "label_status": "unknown_schema_inventory_only",
                    "quantitative_eval_allowed": False,
                    "split_status": "unassigned",
                    "annotation_source_status": "none",
                },
            )


def build_fantastic_restoration(data_root: Path, limit: int | None) -> Iterable[ManifestRow]:
    fb_root = data_root / "Fantastic_Breaks_v1"
    object_dirs = limited((p for p in fb_root.glob("*/*") if p.is_dir()), limit)
    for obj_dir in object_dirs:
        category = obj_dir.parent.name
        object_id = obj_dir.name
        complete = obj_dir / "model_c.ply"
        if not complete.exists():
            continue
        for stem in ("model_b_0", "model_r_0"):
            partial = obj_dir / f"{stem}.ply"
            if not partial.exists():
                continue
            for feature_mode in FEATURE_MODES:
                yield ManifestRow(
                    dataset_name="FantasticBreaks",
                    stage="restoration",
                    object_id=object_id,
                    category=category,
                    sample_id=f"{category}_{object_id}_{stem}_to_model_c_{feature_mode}",
                    source_paths={"partial": str(partial.resolve()), "complete": str(complete.resolve())},
                    partial_path=str(partial.resolve()),
                    complete_path=str(complete.resolve()),
                    feature_mode=feature_mode,
                    metadata={
                        "partial_role": stem,
                        "complete_role": "model_c",
                        "pair_status": "direct",
                        "split_status": "unassigned",
                        "annotation_source_status": "none",
                    },
                )


def build_breakingbad_restoration(data_root: Path, limit: int | None) -> Iterable[ManifestRow]:
    bb_root = data_root / "BreakingBad"
    object_dirs = list(limited((p for p in (bb_root / "artifact").glob("*") if p.is_dir()), limit))
    everyday_dirs = sorted(p for p in (bb_root / "everyday").glob("*/*") if p.is_dir())
    if limit is not None:
        everyday_dirs = everyday_dirs[:limit]
    object_dirs.extend(everyday_dirs)
    for obj_dir in object_dirs:
        subset = obj_dir.relative_to(bb_root).parts[0]
        category = "artifact" if subset == "artifact" else obj_dir.parent.name
        object_id = obj_dir.name
        complete_dir = obj_dir / "mode_0"
        complete_piece_count = sum(1 for _ in complete_dir.glob("piece_*.obj")) if complete_dir.is_dir() else 0
        if complete_piece_count == 0:
            continue
        partial_dirs = [
            p
            for p in sorted(obj_dir.glob("mode_*"))
            if p.is_dir() and p.name != "mode_0"
        ]
        partial_dirs.extend(sorted(p for p in obj_dir.glob("fractured_*") if p.is_dir()))
        if limit is not None:
            partial_dirs = partial_dirs[:limit]
        for partial_dir in partial_dirs:
            partial_piece_count = sum(1 for _ in partial_dir.glob("piece_*.obj"))
            if partial_piece_count == 0:
                continue
            variant = partial_dir.name
            _, _, variant_role = breakingbad_variant_class(variant)
            for feature_mode in FEATURE_MODES:
                yield ManifestRow(
                    dataset_name="BreakingBad",
                    stage="restoration",
                    object_id=object_id,
                    category=category,
                    sample_id=f"{subset}_{category}_{object_id}_{variant}_to_mode_0_{feature_mode}",
                    source_paths={
                        "partial_dir": str(partial_dir.resolve()),
                        "partial_piece_glob": "piece_*.obj",
                        "complete_dir": str(complete_dir.resolve()),
                        "complete_piece_glob": "piece_*.obj",
                    },
                    partial_path=str(partial_dir.resolve()),
                    complete_path=str(complete_dir.resolve()),
                    feature_mode=feature_mode,
                    metadata={
                        "subset": subset,
                        "partial_variant": variant,
                        "partial_variant_role": variant_role,
                        "partial_piece_count": partial_piece_count,
                        "complete_piece_count": complete_piece_count,
                        "complete_variant": "mode_0",
                        "pair_status": "direct_mode_0",
                        "manifest_granularity": "variant",
                        "split_status": "unassigned",
                        "annotation_source_status": "none",
                    },
                )


def write_and_validate(path: Path, rows: Iterable[ManifestRow], stage: str | None = None) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    stage_counts: dict[str, int] = defaultdict(int)
    annotation_counts: dict[str, int] = defaultdict(int)
    errors: list[str] = []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            row_count += 1
            row_errors = validate_manifest_row(row, expected_stage=stage)
            if row_errors:
                errors.extend(f"row {row_count}: {error}" for error in row_errors)
                if len(errors) >= 20:
                    break
            stage_counts[row.stage] += 1
            annotation_counts[str(row.metadata.get("annotation_source_status", "none"))] += 1
            writer.writerow(row.to_csv_row())
    if errors:
        raise ValueError("\n".join(errors))
    summary = {
        "row_count": row_count,
        "stage_counts": dict(stage_counts),
        "annotation_source_status_counts": dict(annotation_counts),
    }
    summary["path"] = str(path.resolve())
    return summary


def combine(*row_groups: Iterable[ManifestRow]) -> Iterable[ManifestRow]:
    for rows in row_groups:
        yield from rows


def build_all(project_root: Path, data_root: Path, limit_per_dataset: int | None) -> dict[str, object]:
    seg_dir = project_root / "manifests" / "segmentation"
    rest_dir = project_root / "manifests" / "restoration"
    for directory in (seg_dir, rest_dir):
        directory.mkdir(parents=True, exist_ok=True)

    outputs = {
        "segmentation/fantastic_breaks_segmentation_manifest.csv": (
            seg_dir / "fantastic_breaks_segmentation_manifest.csv",
            lambda: build_fantastic_segmentation(data_root, limit_per_dataset),
            "segmentation",
        ),
        "segmentation/breakingbad_segmentation_raw_manifest.csv": (
            seg_dir / "breakingbad_segmentation_raw_manifest.csv",
            lambda: build_breakingbad_raw_segmentation(data_root, limit_per_dataset),
            "segmentation",
        ),
        "segmentation/breakingbad_blender_annotation_sources.csv": (
            seg_dir / "breakingbad_blender_annotation_sources.csv",
            lambda: build_breakingbad_annotation_sources(project_root, data_root, limit_per_dataset),
            "segmentation",
        ),
        "segmentation/garf_segmentation_inventory_manifest.csv": (
            seg_dir / "garf_segmentation_inventory_manifest.csv",
            lambda: build_garf_segmentation(data_root),
            "segmentation",
        ),
        "segmentation/all_segmentation_manifest.csv": (
            seg_dir / "all_segmentation_manifest.csv",
            lambda: combine(
                build_fantastic_segmentation(data_root, limit_per_dataset),
                build_breakingbad_raw_segmentation(data_root, limit_per_dataset),
                build_breakingbad_annotation_sources(project_root, data_root, limit_per_dataset),
                build_garf_segmentation(data_root),
            ),
            "segmentation",
        ),
        "restoration/fantastic_breaks_restoration_pairs_manifest.csv": (
            rest_dir / "fantastic_breaks_restoration_pairs_manifest.csv",
            lambda: build_fantastic_restoration(data_root, limit_per_dataset),
            "restoration",
        ),
        "restoration/breakingbad_restoration_pairs_manifest.csv": (
            rest_dir / "breakingbad_restoration_pairs_manifest.csv",
            lambda: build_breakingbad_restoration(data_root, limit_per_dataset),
            "restoration",
        ),
        "restoration/all_restoration_manifest.csv": (
            rest_dir / "all_restoration_manifest.csv",
            lambda: combine(
                build_fantastic_restoration(data_root, limit_per_dataset),
                build_breakingbad_restoration(data_root, limit_per_dataset),
            ),
            "restoration",
        ),
    }
    summaries: dict[str, object] = {}
    for name, (path, row_factory, stage) in outputs.items():
        summaries[name] = write_and_validate(path, row_factory(), stage=stage)
    summaries["garf_restoration_inventory"] = {
        "row_count": 0,
        "reason": "skipped because restoration rows require known partial_path and complete_path",
    }
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build lightweight unified dataset manifests.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--limit-per-dataset", type=int, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root
    data_root = args.data_root or project_root / "data"
    summary = build_all(project_root, data_root, args.limit_per_dataset)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(1)
