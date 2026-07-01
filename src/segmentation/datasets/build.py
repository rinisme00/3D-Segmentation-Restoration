"""Build or dry-run segmentation point-cloud preprocessing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.common.preprocessing.builders import (
    BuildArgs,
    add_common_args,
    config_value,
    count_existing_npz,
    ensure_output_dir,
    load_config,
    matching_feature_rows,
    path_exists,
    print_or_write_summary,
    process_point_file,
    require_path,
    sample_output_path,
    source_meshes_from_row,
    summarize_common,
    write_csv,
    write_npz_sample,
)


def parse_args() -> BuildArgs:
    parser = argparse.ArgumentParser(description="Build segmentation preprocessed point-cloud samples.")
    add_common_args(parser)
    parser.add_argument("--model-family", choices=["pointnext"], default=None)
    parser.add_argument("--num-points", type=int, default=None)
    ns = parser.parse_args()
    config = load_config(ns.config)
    return BuildArgs(
        manifest=require_path(config_value(ns.manifest, config, "manifest"), "--manifest"),
        output_dir=require_path(config_value(ns.output_dir, config, "output_dir"), "--output-dir"),
        feature_mode=config_value(ns.feature_mode, config, "feature_mode", "9d"),
        limit=config_value(ns.limit, config, "limit"),
        dry_run=bool(ns.dry_run or config.get("dry_run", False)),
        seed=int(config_value(ns.seed, config, "seed", 0)),
        num_points=int(config_value(ns.num_points, config, "num_points", 8192)),
        model_family=config_value(ns.model_family, config, "model_family", "pointnext"),
        config=ns.config,
        debug=bool(ns.debug or config.get("debug", False)),
        overwrite=bool(ns.overwrite or config.get("overwrite", False)),
        resume=bool(ns.resume or config.get("resume", False)),
        verbose=bool(ns.verbose or config.get("verbose", False)),
    )


def label_quality(row) -> str:
    status = str(row.metadata.get("annotation_source_status", "none"))
    if status == "blender_pending":
        return "blender_pending"
    label_status = str(row.metadata.get("label_status", ""))
    if label_status == "unknown_schema_inventory_only":
        return "unknown_schema_inventory_only"
    if row.point_labels_available and row.label_paths:
        return "candidate_needs_verification"
    return "unlabeled"


def build(args: BuildArgs) -> dict[str, object]:
    ensure_output_dir(args.output_dir, args.dry_run, overwrite=args.overwrite, resume=args.resume)

    if args.verbose and not args.dry_run:
        existing = count_existing_npz(args.output_dir)
        print(f"=== Pre-run: {existing:,} existing npz in {args.output_dir} ===", flush=True)
        print(f"=== Manifest: {args.manifest.name}  feature_mode={args.feature_mode} ===", flush=True)

    index_rows: list[dict[str, object]] = []
    planned = written = skipped = seen = 0
    quality_counts: dict[str, int] = {}
    for row in matching_feature_rows(args.manifest, expected_stage="segmentation", feature_mode=args.feature_mode):
        quality = label_quality(row)
        quality_counts[quality] = quality_counts.get(quality, 0) + 1
        if quality in {"blender_pending", "unknown_schema_inventory_only"}:
            skipped += 1
            continue
        meshes = source_meshes_from_row(row)
        if not meshes:
            skipped += 1
            continue
        for mesh_idx, mesh_path in enumerate(meshes):
            if args.limit is not None and planned >= args.limit:
                break
            if not path_exists(mesh_path):
                skipped += 1
                continue
            planned += 1
            sample_name = f"{row.sample_id}_{mesh_path.stem}_{mesh_idx:04d}"
            output_path = sample_output_path(args.output_dir, row, f"{mesh_path.stem}_{mesh_idx:04d}")
            index_rows.append(
                {
                    "sample_name": sample_name,
                    "manifest_sample_id": row.sample_id,
                    "source_path": str(mesh_path),
                    "label_quality": quality,
                    "feature_mode": args.feature_mode,
                    "manifest_feature_mode": row.feature_mode,
                    "num_points": args.num_points,
                    "output_path": str(output_path),
                }
            )
            if not args.dry_run:
                seen += 1
                if args.resume and output_path.exists():
                    skipped += 1
                    if args.verbose:
                        rel = output_path.relative_to(args.output_dir / "samples")
                        print(f"S [{seen:7d} | W:{written:6d} S:{skipped:6d}] {rel}", flush=True)
                    continue
                payload = process_point_file(mesh_path, args.num_points or 8192, args.feature_mode, args.seed + planned)
                payload["manifest_metadata"] = row.metadata
                payload["label_quality"] = quality
                write_npz_sample(output_path, payload)
                written += 1
                if args.verbose:
                    rel = output_path.relative_to(args.output_dir / "samples")
                    print(f"W [{seen:7d} | W:{written:6d} S:{skipped:6d}] {mesh_path.name}  ->  {rel}", flush=True)
        if args.limit is not None and planned >= args.limit:
            break
    if not args.dry_run:
        write_csv(args.output_dir / "sample_index.csv", index_rows)
        write_csv(args.output_dir / "label_quality.csv", [{"label_quality": k, "count": v} for k, v in sorted(quality_counts.items())])
    summary = summarize_common(
        builder="segmentation",
        args=args,
        planned=planned,
        written=written,
        skipped=skipped,
        extra={"index_rows": len(index_rows), "label_quality_counts": quality_counts, "preview_rows": index_rows[:3]},
    )
    print_or_write_summary(args, summary)
    return summary


def main() -> None:
    args = parse_args()
    build(args)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(1)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
