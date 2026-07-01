"""Tune binary fracture prediction threshold on a validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .export_predictions import _binary_metrics, _build_model, _load_cfg, _to_device

from openpoints.dataset import build_dataloader_from_cfg, get_features_by_keys


def tune_threshold(args: argparse.Namespace) -> None:
    cfg = _load_cfg(Path(args.config), args.opts)
    cfg.dataset.val.split = args.split
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

    all_gt: list[np.ndarray] = []
    all_prob: list[np.ndarray] = []
    for data in loader:
        data = _to_device(data, device)
        all_gt.append(data["y"].squeeze(0).detach().cpu().numpy().astype(np.uint8))
        data["x"] = get_features_by_keys(data, cfg.feature_keys)
        with torch.no_grad():
            logits = model(data)
            all_prob.append(torch.softmax(logits, dim=1)[:, 1, :].squeeze(0).detach().cpu().numpy())

    gt = np.concatenate(all_gt)
    prob = np.concatenate(all_prob)
    rows = []
    for threshold in args.thresholds:
        pred = (prob >= threshold).astype(np.uint8)
        metrics = _binary_metrics(gt, pred)
        metrics["threshold"] = float(threshold)
        rows.append(metrics)
    best = max(rows, key=lambda row: row[args.metric])
    result = {
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "config_path": str(Path(args.config).resolve()),
        "split": args.split,
        "metric": args.metric,
        "best_threshold": best["threshold"],
        "best_metrics": best,
        "thresholds": rows,
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
    print(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--metric", default="f1", choices=["f1", "iou_fracture", "miou", "recall", "precision"])
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    args, opts = parser.parse_known_args()
    args.opts = opts
    return args


def main() -> None:
    tune_threshold(parse_args())


if __name__ == "__main__":
    main()
