"""Attach aligned fracture-probability sidecars to restoration split CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-split", required=True, type=Path)
    parser.add_argument("--output-split", required=True, type=Path)
    parser.add_argument("--guidance-root", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--code-root", type=Path, default=None)
    parser.add_argument("--strict", action="store_true", help="Fail if any row has no valid aligned guidance sidecar.")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve(raw: str, project_root: Path, code_root: Path | None) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    for base in (project_root, code_root):
        if base is None:
            continue
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (project_root / path).resolve()


def guidance_candidate(row: dict[str, str], pair_path: Path, guidance_root: Path) -> Path:
    rel_parts = pair_path.with_suffix(".npz").parts
    if "samples" in rel_parts:
        suffix = Path(*rel_parts[rel_parts.index("samples") + 1 :])
        return guidance_root / suffix
    return guidance_root / f"{row.get('sample_name', pair_path.stem)}.npz"


def validate_guidance(pair_path: Path, guidance_path: Path) -> str | None:
    if not guidance_path.exists():
        return "missing_guidance"
    pair = np.load(pair_path, allow_pickle=False)
    guidance = np.load(guidance_path, allow_pickle=False)
    if "partial_points" not in pair.files:
        return "pair_missing_partial_points"
    if "fracture_prob" not in guidance.files:
        return "guidance_missing_fracture_prob"
    n_points = pair["partial_points"].shape[0]
    if guidance["fracture_prob"].shape != (n_points,):
        return f"fracture_prob_shape:{guidance['fracture_prob'].shape}:expected_{n_points}"
    if "partial_points" in guidance.files and not np.allclose(pair["partial_points"], guidance["partial_points"], atol=1e-6):
        return "partial_points_not_aligned"
    return None


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    code_root = args.code_root.resolve() if args.code_root else None
    rows = read_rows(args.input_split)
    out_rows: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    for row in rows:
        pair_path = resolve(row["output_path"], args.project_root, code_root)
        guidance_path = guidance_candidate(row, pair_path, args.guidance_root)
        error = validate_guidance(pair_path, guidance_path)
        out = dict(row)
        if error is None:
            out["guidance_path"] = str(guidance_path)
        else:
            out["guidance_path"] = ""
            failures.append({"sample_name": row.get("sample_name", ""), "guidance_path": str(guidance_path), "error": error})
        out_rows.append(out)

    if args.strict and failures:
        print(json.dumps({"failure_count": len(failures), "failures": failures[:20]}, indent=2, sort_keys=True))
        return 1
    write_csv(args.output_split, out_rows)
    summary = {
        "input_split": str(args.input_split),
        "output_split": str(args.output_split),
        "rows": len(out_rows),
        "attached": sum(1 for row in out_rows if row.get("guidance_path")),
        "failure_count": len(failures),
        "failures": failures[:20],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
