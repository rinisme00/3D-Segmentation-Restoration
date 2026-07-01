"""Build or dry-run restoration partial-to-complete preprocessing pairs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.common.preprocessing.builders import (
    BuildArgs,
    add_common_args,
    config_value,
    ensure_output_dir,
    load_config,
    matching_feature_rows,
    path_exists,
    print_or_write_summary,
    process_point_file,
    require_path,
    sample_output_path,
    summarize_common,
    write_csv,
    write_npz_sample,
)


def parse_args() -> BuildArgs:
    parser = argparse.ArgumentParser(description="Build restoration partial-to-complete point-cloud pairs.")
    add_common_args(parser)
    parser.add_argument("--num-points-partial", type=int, default=None)
    parser.add_argument("--num-points-complete", type=int, default=None)
    ns = parser.parse_args()
    config = load_config(ns.config)
    return BuildArgs(
        manifest=require_path(config_value(ns.manifest, config, "manifest"), "--manifest"),
        output_dir=require_path(config_value(ns.output_dir, config, "output_dir"), "--output-dir"),
        feature_mode=config_value(ns.feature_mode, config, "feature_mode", "9d"),
        limit=config_value(ns.limit, config, "limit"),
        dry_run=bool(ns.dry_run or config.get("dry_run", False)),
        seed=int(config_value(ns.seed, config, "seed", 0)),
        num_points_partial=int(config_value(ns.num_points_partial, config, "num_points_partial", 8192)),
        num_points_complete=int(config_value(ns.num_points_complete, config, "num_points_complete", 8192)),
        config=ns.config,
        debug=bool(ns.debug or config.get("debug", False)),
        overwrite=bool(ns.overwrite or config.get("overwrite", False)),
        resume=bool(ns.resume or config.get("resume", False)),
    )


def first_piece_or_path(path: Path) -> Path | None:
    if path.is_file():
        return path
    if path.is_dir():
        candidates = sorted(path.glob("piece_*.obj"))
        return candidates[0] if candidates else None
    return None


def build(args: BuildArgs) -> dict[str, object]:
    ensure_output_dir(args.output_dir, args.dry_run, overwrite=args.overwrite, resume=args.resume)
    index_rows: list[dict[str, object]] = []
    planned = written = skipped = 0
    for row in matching_feature_rows(args.manifest, expected_stage="restoration", feature_mode=args.feature_mode):
        if args.limit is not None and planned >= args.limit:
            break
        partial = first_piece_or_path(Path(row.partial_path))
        complete = first_piece_or_path(Path(row.complete_path))
        if partial is None or complete is None or not path_exists(partial) or not path_exists(complete):
            skipped += 1
            continue
        planned += 1
        sample_name = row.sample_id
        output_path = sample_output_path(args.output_dir, row, f"{partial.stem}_to_{complete.stem}")
        index_rows.append(
            {
                "sample_name": sample_name,
                "manifest_sample_id": row.sample_id,
                "partial_path": str(partial),
                "complete_path": str(complete),
                "feature_mode": args.feature_mode,
                "manifest_feature_mode": row.feature_mode,
                "num_points_partial": args.num_points_partial,
                "num_points_complete": args.num_points_complete,
                "output_path": str(output_path),
            }
        )
        if not args.dry_run:
            if args.resume and output_path.exists():
                skipped += 1
                continue
            partial_payload = process_point_file(
                partial,
                args.num_points_partial or 8192,
                args.feature_mode,
                args.seed + planned,
            )
            complete_payload = process_point_file(
                complete,
                args.num_points_complete or 8192,
                args.feature_mode,
                args.seed + 100000 + planned,
            )
            payload = {
                "partial_points": partial_payload["points"],
                "partial_features": partial_payload["features"],
                "complete_points": complete_payload["points"],
                "complete_features": complete_payload["features"],
                "partial_metadata": partial_payload,
                "complete_metadata": complete_payload,
                "manifest_metadata": row.metadata,
            }
            write_npz_sample(output_path, payload)
            written += 1
    if not args.dry_run:
        write_csv(args.output_dir / "sample_index.csv", index_rows)
    summary = summarize_common(
        builder="restoration",
        args=args,
        planned=planned,
        written=written,
        skipped=skipped,
        extra={"index_rows": len(index_rows), "preview_rows": index_rows[:3]},
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
