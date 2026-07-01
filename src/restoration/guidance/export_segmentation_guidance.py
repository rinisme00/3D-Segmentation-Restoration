import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any
import numpy as np
import torch

# Add PointNeXt to path
CODE_SRC = Path(__file__).resolve().parents[2]
POINTNEXT_ROOT = CODE_SRC / "pointnext"
if str(POINTNEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(POINTNEXT_ROOT))

# Also add CODE_SRC to sys.path so we can import from src.*
if str(CODE_SRC.parent) not in sys.path:
    sys.path.insert(0, str(CODE_SRC.parent))

from openpoints.dataset import get_features_by_keys
from openpoints.transforms import build_transforms_from_cfg
from openpoints.models import build_model_from_cfg
from openpoints.utils import EasyConfig, load_checkpoint

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PointNeXt segmentation predictions as restoration guidance.")
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--seg-checkpoint", type=Path, required=True)
    parser.add_argument("--feature-mode", choices=["3d", "9d"], default="3d")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()

def _load_cfg(checkpoint_path: Path) -> EasyConfig:
    # cfg.yaml is in the parent-parent directory of the checkpoint
    config_path = checkpoint_path.parent.parent / "cfg.yaml"
    if not config_path.exists():
        # try parent
        config_path = checkpoint_path.parent / "cfg.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find cfg.yaml near checkpoint: {checkpoint_path}")
    
    import yaml
    with open(config_path, "r") as f:
        cfg = yaml.unsafe_load(f)
    
    # Resolve relative paths in config to be absolute under PointNeXt root
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

def main() -> None:
    args = parse_args()
    
    # Load config and model
    cfg = _load_cfg(args.seg_checkpoint)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _build_model(cfg, args.seg_checkpoint, device)
    transform = build_transforms_from_cfg("val", cfg.datatransforms) if cfg.get("datatransforms") else None
    
    # Read rows from split CSV
    with args.split_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        
    if args.limit is not None:
        rows = rows[:args.limit]
        
    args.output_root.mkdir(parents=True, exist_ok=True)
    
    print(f"Exporting guidance for {len(rows)} samples...")
    
    for i, row in enumerate(rows):
        # Resolve output_path (pointing to completion pair NPZ)
        rel_path = Path(row["output_path"])
        if rel_path.is_absolute():
            npz_path = rel_path
        else:
            candidates = [args.project_root / rel_path, args.code_root / rel_path]
            npz_path = None
            for cand in candidates:
                if cand.resolve().exists():
                    npz_path = cand.resolve()
                    break
            if npz_path is None:
                raise FileNotFoundError(f"Could not find npz path: {rel_path}")
                
        # Load completion pair NPZ
        payload = np.load(npz_path, allow_pickle=True)
        partial_points = payload["partial_points"].astype(np.float32)
        
        # Build input dict for model
        data = {"pos": partial_points[:, :3]}
        
        if args.feature_mode == "9d":
            partial_features = payload["partial_features"].astype(np.float32)
            data["x"] = partial_features
        else:
            data["x"] = partial_points[:, :3]
            
        # Apply PointNeXt val transform (normalizations etc.)
        if transform is not None:
            data = transform(data)
            
        # Add batch dimension and move to device
        item = {}
        for k, v in data.items():
            if torch.is_tensor(v):
                item[k] = v.unsqueeze(0).to(device)
            elif isinstance(v, np.ndarray):
                item[k] = torch.from_numpy(v).unsqueeze(0).to(device)
            else:
                item[k] = v
                
        # build feature keys as requested by config
        item["x"] = get_features_by_keys(item, cfg.feature_keys)
        
        # Run inference
        with torch.no_grad():
            logits = model(item)
            prob = torch.softmax(logits, dim=1)[:, 1, :].squeeze(0).detach().cpu().numpy().astype(np.float32)
            
        # Replicate the path structure under samples/ to be matched by attach_guidance_to_splits.py
        pair_parts = npz_path.parts
        if "samples" in pair_parts:
            rel_suffix = Path(*pair_parts[pair_parts.index("samples") + 1 :])
        else:
            dataset_name = row.get("dataset_name", "BreakingBad")
            category = row.get("category", "")
            object_id = row.get("object_id", "")
            rel_suffix = Path(dataset_name) / category / object_id / npz_path.name
        
        guidance_path = args.output_root / rel_suffix
        guidance_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save guidance (fracture probability, valid mask, and original partial points)
        valid_mask = np.ones_like(prob)
        np.savez_compressed(
            guidance_path,
            fracture_prob=prob,
            valid_mask=valid_mask,
            partial_points=partial_points
        )
        
        if (i + 1) % 100 == 0 or (i + 1) == len(rows):
            print(f"Processed {i + 1}/{len(rows)} samples.")
            
    print("Guidance export completed successfully!")

if __name__ == "__main__":
    main()
