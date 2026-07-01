import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import csv
from scipy.spatial import cKDTree
import trimesh
from skimage import measure

CODE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(CODE_ROOT))
sys.path.append(os.path.join(CODE_ROOT, "models", "deepmend_native", "python"))
sys.path.append(os.path.join(CODE_ROOT, "src"))

from restoration.datasets.deepmend_10d_dataset import DeepMend10DDataset
from train_deepmend_10d import DeepMend10DPipeline
from restoration.evaluation.completion_metrics import chamfer_distance

def make_3d_grid(bb_min, bb_max, resolution):
    """Generates a 3D grid of points."""
    x = np.linspace(bb_min[0], bb_max[0], resolution)
    y = np.linspace(bb_min[1], bb_max[1], resolution)
    z = np.linspace(bb_min[2], bb_max[2], resolution)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    return grid_points, xx.shape

def compute_iou(occ_pred, occ_gt):
    intersection = np.sum((occ_pred > 0) & (occ_gt > 0))
    union = np.sum((occ_pred > 0) | (occ_gt > 0))
    if union == 0:
        return 1.0
    return intersection / union

def main():
    parser = argparse.ArgumentParser(description="Evaluate DeepMend 10D")
    parser.add_argument('--model-path', type=str, required=True, help="Path to best_model.pth")
    parser.add_argument('--csv-path', type=str, required=True, help="Path to sample_index.csv")
    parser.add_argument(
        '--output-dir',
        type=str,
        default=os.path.join(PROJECT_ROOT, "results", "restoration", "eval_output"),
    )
    parser.add_argument('--grid-res', type=int, default=128, help="Marching cubes grid resolution")
    parser.add_argument('--limit', type=int, default=None, help='Limit number of samples to EXPORT (meshes)')
    parser.add_argument('--eval-limit', type=int, default=None, help='Limit number of samples to EVALUATE (metrics)')
    parser.add_argument('--seed', type=int, default=42, help="Random seed for shuffling the test set exports")
    parser.add_argument('--save-meshes', action='store_true', help="Export PLY meshes")
    parser.add_argument('--smoke-test', action='store_true', help="Run 1 sample quickly")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_meshes:
        os.makedirs(os.path.join(args.output_dir, 'meshes'), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.smoke_test else 'cpu')
    print(f"Evaluating on {device}...")

    # Load Model
    model = DeepMend10DPipeline().to(device)
    if args.model_path != "dummy":
        try:
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            print("Model weights loaded successfully.")
        except Exception as e:
            print(f"Could not load weights, checking if state dict has module prefix: {e}")
            # Sometimes DataParallel adds 'module.', we can strip it
            state_dict = torch.load(args.model_path, map_location=device)
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)

    model.eval()

    # Load full dataset paths without limit first
    dataset = DeepMend10DDataset(csv_path=args.csv_path, is_train=False, limit=None)
    paths = dataset.npz_paths
    
    scale_factor = dataset.scale_factor
    occ_threshold = dataset.occ_threshold

    # Filter BB paths to only include samples with 2-6 pieces
    import glob
    filtered_paths = []
    print("Filtering Breaking Bad samples to keep only 2-6 piece fractures...")
    for p in tqdm(paths, desc="Filtering paths"):
        if 'BreakingBad' in p:
            raw_dir = p.split('/9d/')[0].replace(
                'preprocessed/restoration/completion_pairs_9d/samples',
                os.path.join(PROJECT_ROOT, 'data'),
            )
            if os.path.exists(raw_dir):
                num_pieces = len(glob.glob(os.path.join(raw_dir, "piece_*.obj")))
                if 2 <= num_pieces <= 6:
                    filtered_paths.append(p)
        else:
            filtered_paths.append(p)
    paths = filtered_paths
    print(f"Filtered down to {len(paths)} valid samples.")

    import random
    random.seed(args.seed)
    random.shuffle(paths)
    
    if args.eval_limit is not None:
        paths = paths[:args.eval_limit]
        print(f"Truncated evaluation set to {args.eval_limit} samples.")
    
    # Path truncation based on limit is removed. We evaluate the full set.
    # The limit is now applied to mesh saving.
        
    if args.smoke_test:
        paths = paths[:1]
        args.grid_res = 32  # fast grid for smoke test

    grid_pts, grid_shape = make_3d_grid([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5], args.grid_res)
    grid_pts_tensor = torch.from_numpy(grid_pts).float().to(device)
    
    # Pre-split grid into batches to avoid VRAM OOM
    query_batch_size = 100000
    
    total_iou = 0.0
    total_cd = 0.0
    valid_samples = 0
    num_exports = 0

    print(f"Starting evaluation on {len(paths)} samples...")
    pbar = tqdm(paths)
    for idx, path in enumerate(pbar):
        try:
            d = np.load(path)
            partial_features = d['partial_features'].astype(np.float32)
            complete_points = d['complete_points'].astype(np.float32)
        except Exception as e:
            print(f"Skipping corrupted file {path}: {e}")
            continue

        if partial_features.shape[1] == 9:
            p_frac = np.zeros((partial_features.shape[0], 1), dtype=np.float32)
            x_10d = np.hstack([partial_features, p_frac])
        else:
            print(f"Skipping {path}: invalid feature dim")
            continue

        # Scale coordinates
        x_10d[:, :3] = x_10d[:, :3] / scale_factor
        scaled_complete_points = complete_points / scale_factor

        x_10d_tensor = torch.from_numpy(x_10d).transpose(1, 0).unsqueeze(0).to(device) # (1, 10, N)

        # 1. Forward pass for latent
        with torch.no_grad():
            z_both = model.encoder(x_10d_tensor)
            z = z_both[:, :model.latent_size]
            z_tool = z_both[:, model.latent_size:]

            # 2. Query the entire 3D grid in batches
            c_preds = []
            for i in range(0, grid_pts_tensor.shape[0], query_batch_size):
                batch_pts = grid_pts_tensor[i:i+query_batch_size].unsqueeze(0) # (1, N_q, 3)
                N_q = batch_pts.shape[1]
                z_exp = z.unsqueeze(1).expand(-1, N_q, -1).reshape(-1, model.latent_size)
                z_tool_exp = z_tool.unsqueeze(1).expand(-1, N_q, -1).reshape(-1, model.decoder.tool_latent_size)
                pts_flat = batch_pts.reshape(-1, 3)
                
                net_input = torch.cat([z_exp, z_tool_exp, pts_flat], dim=-1)
                c_x, _, _, _ = model.decoder(net_input)
                # Apply sigmoid to get probabilities
                c_probs = torch.sigmoid(c_x)
                c_preds.append(c_probs.squeeze().cpu().numpy())
            
            c_preds = np.concatenate(c_preds, axis=0)
            vol_pred = c_preds.reshape(grid_shape)
        
        # 3. Calculate Volumetric IoU
        # GT Occupancy for the grid
        tree_c = cKDTree(scaled_complete_points)
        dist_c, _ = tree_c.query(grid_pts)
        c_gt = (dist_c < occ_threshold).astype(np.int32)
        
        iou = compute_iou((c_preds > 0.5).astype(np.int32), c_gt)
        total_iou += iou

        # 4. Marching Cubes
        try:
            verts, faces, normals, values = measure.marching_cubes(vol_pred, level=0.5)
            # Map voxel indices back to [-0.5, 0.5] space
            verts = (verts / (args.grid_res - 1)) - 0.5
            
            # Reverse Scaling!
            verts_real = verts * scale_factor
            
            # Create Trimesh to sample points
            mesh = trimesh.Trimesh(vertices=verts_real, faces=faces)
            if len(mesh.faces) > 0:
                sampled_pts, _ = trimesh.sample.sample_surface(mesh, 8192)
            else:
                sampled_pts = verts_real # fallback
                
        except Exception as e:
            # Marching cubes fails if no surface exists at level=0.5
            print(f"Marching cubes failed for {path}: {e}")
            sampled_pts = np.zeros((8192, 3))
            verts_real = np.zeros((3,3))
            faces = np.zeros((1,3))

        # 5. Chamfer Distance
        # Compare sampled_pts vs original unscaled complete_points
        # Make sure both have shapes (1, N, 3)
        # sample complete_points down to 8192 if it's too large, or just use as is
        cp = complete_points
        if cp.shape[0] > 8192:
            idx_choice = np.random.choice(cp.shape[0], 8192, replace=False)
            cp = cp[idx_choice]
        
        pred_tensor = torch.from_numpy(sampled_pts).float().unsqueeze(0).cuda() if not args.smoke_test else torch.from_numpy(sampled_pts).float().unsqueeze(0)
        gt_tensor = torch.from_numpy(cp).float().unsqueeze(0).cuda() if not args.smoke_test else torch.from_numpy(cp).float().unsqueeze(0)
        
        # chamfer_distance returns shape (1,)
        cd_val = chamfer_distance(pred_tensor, gt_tensor).mean().item()
        total_cd += cd_val
        
        # 6. Save Mesh (limited by args.limit)
        if args.save_meshes and (args.limit is None or num_exports < args.limit) and len(verts_real) > 3:
            # Create a descriptive filename from the npz path
            # Example path: .../FantasticBreaks/00/00002/model_r_0/9d/...
            # This captures the object_id and the fractured variant name
            parts = path.split('/')
            object_id = parts[-4]
            variant_id = parts[-3]
            
            if 'BreakingBad' in path:
                if '/everyday/' in path:
                    subset = 'everyday'
                    category = parts[-5]
                else:
                    subset = 'artifact'
                    category = 'artifact'
                
                mesh_basename = f"{subset}_{category}_{object_id}_{variant_id}_to_mode_0_9d"
                mesh_filename = os.path.join(args.output_dir, 'meshes', f"{mesh_basename}.ply")
            else:
                mesh_basename = f"{object_id}_{variant_id}"
                mesh_filename = os.path.join(args.output_dir, 'meshes', f"{mesh_basename}.ply")
            
            print(f"Exporting mesh: {mesh_basename} | IoU: {iou:.4f}, CD: {cd_val:.4f}")
            
            # Apply dynamic vertex colors based on distance to original broken mesh
            # - Original intact points -> Light Gray
            # - Predicted shape-completion filled points -> Light Blue
            tree_partial = cKDTree(partial_features[:, :3])
            distances, _ = tree_partial.query(verts_real)
            
            # Use the occ_threshold scaled back to real coordinates
            threshold = occ_threshold * scale_factor * 1.5
            
            # Start with all points as Light Blue (Prediction)
            colors = np.tile([100, 150, 255, 255], (len(verts_real), 1))
            
            # Any points very close to the original partial mesh are marked as Light Gray
            is_original = distances < threshold
            colors[is_original] = [200, 200, 200, 255]
            
            mesh.visual.vertex_colors = colors
            
            mesh.export(mesh_filename)
            num_exports += 1

        valid_samples += 1
        pbar.set_postfix({'IoU': iou, 'CD': cd_val})

    # Summary
    if valid_samples > 0:
        avg_iou = total_iou / valid_samples
        avg_cd = total_cd / valid_samples
        print(f"\n--- Evaluation Results ({valid_samples} samples) ---")
        print(f"Average Volumetric IoU: {avg_iou:.4f}")
        print(f"Average Chamfer Distance: {avg_cd:.6f}")
        
        # Save results to txt
        with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
            f.write(f"Evaluated Samples: {valid_samples}\n")
            f.write(f"Average Volumetric IoU: {avg_iou:.4f}\n")
            f.write(f"Average Chamfer Distance: {avg_cd:.6f}\n")
    else:
        print("No valid samples were evaluated.")

if __name__ == "__main__":
    main()
