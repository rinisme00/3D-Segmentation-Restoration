"""Export PointNeXt fracture-surface segmentation predictions.

The exporter uses the same OpenPoints config, model, and dataloader path as
training, then writes thesis-friendly qualitative artifacts and per-object
metrics for binary labels: 0=intact, 1=fracture surface.
"""

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

from openpoints.dataset import build_dataloader_from_cfg, get_features_by_keys  # noqa: E402
from openpoints.models import build_model_from_cfg  # noqa: E402
from openpoints.utils import EasyConfig, load_checkpoint  # noqa: E402


CLASS_COLORS = np.array(
    [
        [128, 128, 128],  # intact
        [255, 100, 0],    # fracture
    ],
    dtype=np.uint8,
)
ERROR_COLORS = {
    "tn": np.array([128, 128, 128], dtype=np.uint8),
    "tp": np.array([255, 100, 0], dtype=np.uint8),
    "fp": np.array([0, 100, 255], dtype=np.uint8),
    "fn": np.array([220, 0, 0], dtype=np.uint8),
}


def _load_cfg(config_path: Path, opts: list[str]) -> EasyConfig:
    cfg = EasyConfig()
    cfg.load(str(config_path), recursive=True)
    if opts:
        cfg.update(opts)
    for root_key in ("npz_root", "labeled_npz_root", "pseudo_label_root"):
        root_value = cfg.dataset.common.get(root_key, None)
        if root_value is not None and not Path(root_value).is_absolute():
            cfg.dataset.common[root_key] = str((POINTNEXT_ROOT / root_value).resolve())
    if cfg.model.get("in_channels", None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels
    cfg.rank = 0
    cfg.world_size = 1
    cfg.distributed = False
    cfg.mp = False
    cfg.sync_bn = False
    cfg.dataloader.num_workers = 0
    return cfg


def _build_model(cfg: EasyConfig, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = build_model_from_cfg(cfg.model).to(device)
    load_checkpoint(model, str(checkpoint))
    model.eval()
    return model


def _to_device(data: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in data.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def _sample_name(npz_path: Path, dataset_root: Path) -> str:
    rel = npz_path.relative_to(dataset_root)
    stem = rel.name.replace(".npz", "")
    parts = rel.parts
    if parts[0] == "FantasticBreaks":
        return f"{parts[2]}__{parts[3]}__{stem}"
    if parts[0] == "BreakingBad":
        if parts[1] == "artifact":
            return f"artifact__{parts[2]}__{parts[3]}__{stem}"
        if parts[1] == "everyday":
            return f"everyday__{parts[2]}__{parts[3]}__{parts[4]}__{stem}"
    return "__".join(parts).replace(".npz", "")


def _dataset_npz_paths(dataset: Any) -> list[Path]:
    if hasattr(dataset, "_npz_paths"):
        return [Path(p) for p in getattr(dataset, "_npz_paths")]
    if hasattr(dataset, "_samples"):
        return [Path(sample["npz_path"]) for sample in getattr(dataset, "_samples")]
    return []


def _binary_metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    tp = int(np.logical_and(gt, pred).sum())
    tn = int(np.logical_and(~gt, ~pred).sum())
    fp = int(np.logical_and(~gt, pred).sum())
    fn = int(np.logical_and(gt, ~pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou_fracture = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    iou_intact = tn / (tn + fp + fn) if tn + fp + fn else 0.0
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float((tp + tn) / max(1, tp + tn + fp + fn)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou_fracture": float(iou_fracture),
        "iou_intact": float(iou_intact),
        "miou": float((iou_fracture + iou_intact) / 2.0),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {key: int(sum(int(row[key]) for row in rows)) for key in ("tp", "tn", "fp", "fn")}
    aggregate = _binary_metrics(
        np.array([1] * counts["tp"] + [0] * counts["tn"] + [0] * counts["fp"] + [1] * counts["fn"]),
        np.array([1] * counts["tp"] + [0] * counts["tn"] + [1] * counts["fp"] + [0] * counts["fn"]),
    )
    aggregate["num_samples"] = len(rows)
    aggregate["mean_confidence"] = float(np.mean([row["mean_confidence"] for row in rows])) if rows else 0.0
    return aggregate


def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray | None = None, scalar: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        if scalar is not None:
            f.write("property float fracture_probability\n")
        f.write("end_header\n")
        for idx, point in enumerate(points):
            values: list[str] = [f"{float(point[0]):.8f}", f"{float(point[1]):.8f}", f"{float(point[2]):.8f}"]
            if colors is not None:
                values.extend(str(int(c)) for c in colors[idx])
            if scalar is not None:
                values.append(f"{float(scalar[idx]):.8f}")
            f.write(" ".join(values) + "\n")


def _error_colors(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    colors = np.zeros((len(gt), 3), dtype=np.uint8)
    colors[(gt == 0) & (pred == 0)] = ERROR_COLORS["tn"]
    colors[(gt == 1) & (pred == 1)] = ERROR_COLORS["tp"]
    colors[(gt == 0) & (pred == 1)] = ERROR_COLORS["fp"]
    colors[(gt == 1) & (pred == 0)] = ERROR_COLORS["fn"]
    return colors


def _write_panel_png(
    path: Path,
    points: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    metrics: dict[str, Any],
    *,
    sample_id: str,
    experiment_id: str,
    threshold: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    gt_colors = CLASS_COLORS[gt]
    pred_colors = CLASS_COLORS[pred]
    fig = plt.figure(figsize=(11, 5), dpi=160)
    title = (
        f"{experiment_id} | {sample_id} | "
        f"F1={metrics['f1']:.3f} IoU={metrics['iou_fracture']:.3f} "
        f"conf={metrics['mean_confidence']:.3f} thr={threshold:.2f}"
    )
    fig.suptitle(title, fontsize=9)
    for idx, (label, colors) in enumerate((("Ground truth", gt_colors), ("Prediction", pred_colors)), start=1):
        ax = fig.add_subplot(1, 2, idx, projection="3d")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors / 255.0, s=1, depthshade=False)
        ax.set_title(label, fontsize=10)
        ax.set_axis_off()
        _set_equal_axes(ax, points)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _set_equal_axes(ax: Any, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) / 2.0, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def export_predictions(args: argparse.Namespace) -> None:
    cfg = _load_cfg(Path(args.config), args.opts)
    cfg.dataset.test.split = args.split
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _build_model(cfg, Path(args.checkpoint), device)
    loader = build_dataloader_from_cfg(
        1,
        cfg.dataset,
        cfg.dataloader,
        datatransforms_cfg=cfg.datatransforms,
        split=args.split,
        distributed=False,
    )
    dataset = loader.dataset
    npz_paths = _dataset_npz_paths(dataset)
    if len(npz_paths) != len(dataset):
        raise RuntimeError("export_predictions currently requires NPZ-mode datasets with source NPZ paths.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    root_value = cfg.dataset.common.get("npz_root", None) or cfg.dataset.common.get("labeled_npz_root", None)
    if root_value is None:
        raise RuntimeError("export_predictions requires cfg.dataset.common.npz_root or labeled_npz_root.")
    dataset_root = Path(root_value)
    dataset_root = dataset_root / "samples"

    rows: list[dict[str, Any]] = []
    for idx, data in enumerate(loader):
        npz_path = npz_paths[idx]
        sample_id = _sample_name(npz_path, dataset_root)
        data = _to_device(data, device)
        target = data["y"].squeeze(0).detach().cpu().numpy().astype(np.uint8)
        points = data["pos"].squeeze(0).detach().cpu().numpy().astype(np.float32)
        data["x"] = get_features_by_keys(data, cfg.feature_keys)
        with torch.no_grad():
            logits = model(data)
            probs = torch.softmax(logits, dim=1)[:, 1, :].squeeze(0).detach().cpu().numpy()
        pred = (probs >= args.threshold).astype(np.uint8)
        confidence = np.maximum(probs, 1.0 - probs)
        metrics = _binary_metrics(target, pred)
        metrics["mean_confidence"] = float(confidence.mean())
        metrics["sample_id"] = sample_id
        metrics["source_npz"] = str(npz_path)
        sample_dir = output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        if args.export_npy:
            np.save(sample_dir / f"{sample_id}_pred_mask.npy", pred.astype(np.uint8))
            np.save(sample_dir / f"{sample_id}_pred_prob.npy", probs.astype(np.float32))
        if args.export_ply:
            _write_ply(sample_dir / f"{sample_id}_gt.ply", points, CLASS_COLORS[target])
            _write_ply(sample_dir / f"{sample_id}_pred.ply", points, CLASS_COLORS[pred])
            _write_ply(sample_dir / f"{sample_id}_error.ply", points, _error_colors(target, pred))
            _write_ply(sample_dir / f"{sample_id}_prob.ply", points, scalar=probs)
        if args.export_panel_png:
            _write_panel_png(
                sample_dir / f"{sample_id}_panel.png",
                points,
                target,
                pred,
                metrics,
                sample_id=sample_id,
                experiment_id=args.experiment_id,
                threshold=args.threshold,
            )

        metadata = {
            "dataset": cfg.dataset.common.get("dataset_name", "FantasticBreaks"),
            "sample_id": sample_id,
            "source_npz": str(npz_path),
            "checkpoint_path": str(Path(args.checkpoint).resolve()),
            "experiment_id": args.experiment_id,
            "prediction_threshold": args.threshold,
            "feature_mode": f"{cfg.dataset.common.in_channels}d",
            "metrics": metrics,
        }
        (sample_dir / f"{sample_id}_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        rows.append(metrics)

    fieldnames = [
        "sample_id",
        "source_npz",
        "tp",
        "tn",
        "fp",
        "fn",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "iou_fracture",
        "iou_intact",
        "miou",
        "mean_confidence",
    ]
    with (output_dir / "per_object_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fieldnames} for row in rows)
    summary = _aggregate(rows)
    summary.update(
        {
            "experiment_id": args.experiment_id,
            "checkpoint_path": str(Path(args.checkpoint).resolve()),
            "config_path": str(Path(args.config).resolve()),
            "split": args.split,
            "threshold": args.threshold,
        }
    )
    (output_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--experiment-id", default="segmentation_export")
    parser.add_argument("--device", default=None)
    parser.add_argument("--export-ply", action="store_true")
    parser.add_argument("--export-npy", action="store_true")
    parser.add_argument("--export-panel-png", action="store_true")
    args, opts = parser.parse_known_args()
    args.opts = opts
    return args


def main() -> None:
    export_predictions(parse_args())


if __name__ == "__main__":
    main()
