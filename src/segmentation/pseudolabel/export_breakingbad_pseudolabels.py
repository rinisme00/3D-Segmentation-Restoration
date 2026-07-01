"""Export high-confidence pseudo-labels for unlabeled Breaking Bad samples."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


CODE_SRC = Path(__file__).resolve().parents[2]
POINTNEXT_ROOT = CODE_SRC / "pointnext"
if str(POINTNEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(POINTNEXT_ROOT))

from openpoints.dataset import get_features_by_keys  # noqa: E402
from openpoints.models import build_model_from_cfg  # noqa: E402
from openpoints.transforms import build_transforms_from_cfg  # noqa: E402
from openpoints.utils import EasyConfig, load_checkpoint  # noqa: E402


def _norm_path_string(path: str) -> str:
    return str(Path(path).expanduser())


def _resolve_existing_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.exists():
        return candidate.resolve()
    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate.resolve()
    return candidate


def _featureless_sample_id(sample_id: str) -> str:
    for suffix in ("_3d", "_9d"):
        if sample_id.endswith(suffix):
            return sample_id[: -len(suffix)]
    return sample_id


def _load_cfg(config_path: Path, opts: list[str]) -> EasyConfig:
    cfg = EasyConfig()
    cfg.load(str(config_path), recursive=True)
    if opts:
        cfg.update(opts)
    if cfg.model.get("in_channels", None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels
    cfg.rank = 0
    cfg.world_size = 1
    cfg.distributed = False
    cfg.mp = False
    cfg.sync_bn = False
    cfg.dataloader.num_workers = 0
    return cfg


def _load_json_cell(value: str) -> dict[str, Any]:
    if not value:
        return {}
    return json.loads(value)


def _source_mesh(row: dict[str, str]) -> str | None:
    source_paths = _load_json_cell(row.get("source_paths", ""))
    mesh = source_paths.get("mesh")
    return _norm_path_string(mesh) if mesh else None


def _pieces_dir(row: dict[str, str]) -> str | None:
    source_paths = _load_json_cell(row.get("source_paths", ""))
    pieces_dir = source_paths.get("pieces_dir")
    return _norm_path_string(pieces_dir) if pieces_dir else None


def _manifest_unlabeled_rows(manifest: Path, split: str | None) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with manifest.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("dataset_name") != "BreakingBad":
                continue
            metadata = _load_json_cell(row.get("metadata", ""))
            if metadata.get("label_status") != "unlabeled":
                continue
            if split is not None and row.get("split", "") != split:
                continue
            rows[row["sample_id"]] = row
    return rows


def _sample_index_rows(
    sample_index: Path,
    manifest_rows: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    manifest_ids = set(manifest_rows)
    manifest_by_featureless_id = {
        _featureless_sample_id(sample_id): sample_id
        for sample_id in manifest_rows
    }
    manifest_by_source = {
        source: sample_id
        for sample_id, row in manifest_rows.items()
        if (source := _source_mesh(row)) is not None
    }
    manifest_by_pieces_dir = {
        pieces_dir: sample_id
        for sample_id, row in manifest_rows.items()
        if (pieces_dir := _pieces_dir(row)) is not None
    }
    rows: list[dict[str, str]] = []
    with sample_index.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            manifest_sample_id = row.get("manifest_sample_id", "")
            if manifest_sample_id in manifest_ids:
                rows.append(row)
                continue
            featureless_id = _featureless_sample_id(manifest_sample_id)
            if featureless_id in manifest_by_featureless_id:
                row["manifest_sample_id"] = manifest_by_featureless_id[featureless_id]
                rows.append(row)
                continue
            # 3D segmentation sample indexes may use object/variant-level manifest ids
            # while the balanced 9D manifest is piece-level. Source mesh path is stable.
            source_path = _norm_path_string(row.get("source_path", ""))
            if source_path in manifest_by_source:
                row["manifest_sample_id"] = manifest_by_source[source_path]
                rows.append(row)
                continue
            if "/BreakingBad/" in source_path:
                for pieces_dir, sample_id in manifest_by_pieces_dir.items():
                    if source_path.startswith(f"{pieces_dir}/"):
                        row["manifest_sample_id"] = sample_id
                        rows.append(row)
                        break
    return rows


def _relative_breakingbad_path(npz_path: Path) -> Path:
    parts = npz_path.parts
    try:
        idx = parts.index("BreakingBad")
    except ValueError as exc:
        raise ValueError(f"Expected 'BreakingBad' in NPZ path: {npz_path}") from exc
    return Path(*parts[idx + 1 :])


def _build_item(npz_path: Path, transform, in_channels: int) -> dict[str, torch.Tensor]:
    raw = np.load(npz_path, allow_pickle=True)
    pts_norm = raw["points"].astype(np.float32)
    feat = raw["features"].astype(np.float32)
    metadata = json.loads(str(raw["metadata"]))
    centroid = np.array(metadata["normalization"]["centroid"], dtype=np.float32)
    scale = float(metadata["normalization"]["scale"])
    pts = pts_norm * scale + centroid

    if in_channels > 3:
        non_xyz = torch.from_numpy(feat[:, 3:in_channels].copy())
    data: dict[str, Any] = {"pos": pts[:, :3]}
    if transform is not None:
        data = transform(data)
    pos_t = data["pos"]
    if not torch.is_tensor(pos_t):
        pos_t = torch.from_numpy(np.asarray(pos_t, dtype=np.float32))
    data["x"] = pos_t if in_channels <= 3 else torch.cat([pos_t, non_xyz], dim=-1)
    return {key: value.unsqueeze(0) for key, value in data.items()}


def _to_device(data: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in data.items()}


def _build_model(cfg: EasyConfig, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = build_model_from_cfg(cfg.model).to(device)
    load_checkpoint(model, str(checkpoint))
    model.eval()
    return model


def export(args: argparse.Namespace) -> None:
    cfg = _load_cfg(Path(args.config), args.opts)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _build_model(cfg, Path(args.checkpoint), device)
    transform = build_transforms_from_cfg("val", cfg.datatransforms) if cfg.get("datatransforms") else None

    manifest_rows = _manifest_unlabeled_rows(Path(args.manifest), args.split)
    index_rows = _sample_index_rows(Path(args.sample_index), manifest_rows)
    index_rows = sorted(index_rows, key=lambda row: (row["manifest_sample_id"], row["output_path"]))
    if args.limit is not None:
        index_rows = index_rows[: int(args.limit)]
    if not index_rows:
        raise RuntimeError("No unlabeled Breaking Bad rows matched the balanced manifest and sample index.")

    output_root = Path(args.output_dir)
    samples_root = output_root / "samples" / "BreakingBad"
    samples_root.mkdir(parents=True, exist_ok=True)
    index_path = output_root / "pseudo_index.csv"
    rows_out: list[dict[str, Any]] = []

    for row in index_rows:
        npz_path = _resolve_existing_path(row["output_path"])
        manifest_row = manifest_rows[row["manifest_sample_id"]]
        item = _to_device(_build_item(npz_path, transform, int(cfg.model.encoder_args.in_channels)), device)
        item["x"] = get_features_by_keys(item, cfg.feature_keys)
        with torch.no_grad():
            logits = model(item)
            prob = torch.softmax(logits, dim=1)[:, 1, :].squeeze(0).detach().cpu().numpy().astype(np.float32)

        pseudo = np.full(prob.shape, int(args.ignore_label), dtype=np.uint8)
        pseudo[prob > float(args.fracture_threshold)] = 1
        pseudo[prob < float(args.intact_threshold)] = 0
        valid = (pseudo != int(args.ignore_label)).astype(np.float32)
        rel = _relative_breakingbad_path(npz_path)
        out_path = samples_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "dataset_name": "BreakingBad",
            "source_npz": str(npz_path),
            "manifest_sample_id": row["manifest_sample_id"],
            "teacher_config": str(Path(args.config)),
            "teacher_checkpoint": str(Path(args.checkpoint)),
            "fracture_threshold": float(args.fracture_threshold),
            "intact_threshold": float(args.intact_threshold),
            "ignore_label": int(args.ignore_label),
            "label_status": "pseudo_labeled",
            "weak_label": True,
            "split": manifest_row.get("split", ""),
            "metadata": _load_json_cell(manifest_row.get("metadata", "")),
        }
        np.savez_compressed(
            out_path,
            fracture_prob=prob,
            pseudo_label=pseudo,
            valid_mask=valid,
            metadata=json.dumps(metadata, sort_keys=True),
        )
        rows_out.append(
            {
                "manifest_sample_id": row["manifest_sample_id"],
                "split": manifest_row.get("split", ""),
                "source_npz": str(npz_path),
                "pseudo_path": str(out_path),
                "num_points": int(prob.shape[0]),
                "valid_points": int(valid.sum()),
                "fracture_points": int((pseudo == 1).sum()),
                "intact_points": int((pseudo == 0).sum()),
                "ignored_points": int((pseudo == int(args.ignore_label)).sum()),
                "mean_fracture_prob": float(prob.mean()),
            }
        )

    with index_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    summary = {
        "config": str(Path(args.config)),
        "checkpoint": str(Path(args.checkpoint)),
        "manifest": str(Path(args.manifest)),
        "sample_index": str(Path(args.sample_index)),
        "output_dir": str(output_root),
        "rows": len(rows_out),
        "fracture_threshold": float(args.fracture_threshold),
        "intact_threshold": float(args.intact_threshold),
        "ignore_label": int(args.ignore_label),
        "split": args.split,
        "total_valid_points": int(sum(row["valid_points"] for row in rows_out)),
        "total_ignored_points": int(sum(row["ignored_points"] for row in rows_out)),
        "total_fracture_points": int(sum(row["fracture_points"] for row in rows_out)),
        "total_intact_points": int(sum(row["intact_points"] for row in rows_out)),
    }
    (output_root / "pseudo_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Teacher PointNeXt segmentation config.")
    parser.add_argument("--checkpoint", required=True, help="Teacher checkpoint path.")
    parser.add_argument("--manifest", required=True, help="Balanced segmentation manifest CSV.")
    parser.add_argument("--sample-index", required=True, help="pointnext_9d/sample_index.csv.")
    parser.add_argument("--output-dir", required=True, help="Pseudo-label output root.")
    parser.add_argument("--fracture-threshold", type=float, default=0.8)
    parser.add_argument("--intact-threshold", type=float, default=0.2)
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", default="train", choices=["train", "val", "test", "all"])
    parser.add_argument("--device", default=None)
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Optional config overrides.")
    args = parser.parse_args()
    if args.split == "all":
        args.split = None
    return args


def main() -> None:
    export(parse_args())


if __name__ == "__main__":
    main()
