import sys
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import yaml
import numpy as np
from scipy.spatial import cKDTree
from skimage import measure
import trimesh

CODE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(CODE_ROOT))
sys.path.append(os.path.join(CODE_ROOT, "models", "deepmend_native", "python"))
sys.path.append(os.path.join(CODE_ROOT, "src"))

from restoration.datasets.deepmend_10d_dataset import DeepMend10DDataset
from networks.encoder import DeepMendEncoder10D
from networks.decoder_z_lb_occ_leaky import Decoder as DeepMendDecoder
from restoration.evaluation.completion_metrics import chamfer_distance

def compute_iou(occ_pred, occ_gt):
    intersection = np.sum(occ_pred & occ_gt)
    union = np.sum(occ_pred | occ_gt)
    if union == 0:
        return 1.0
    return intersection / union

class DeepMend10DPipeline(nn.Module):
    def __init__(self, in_channels=10, latent_size=256, tool_latent_size=256):
        super().__init__()
        self.encoder = DeepMendEncoder10D(in_channels, latent_size, tool_latent_size)
        self.decoder = DeepMendDecoder(
            latent_size=latent_size,
            tool_latent_size=tool_latent_size,
            dims=[512, 512, 512, 512, 512, 512, 512, 512],
            num_dims=3,
            do_code_regularization=True,
            use_occ=True,
            subnet_dims=[512, 512, 512, 512, 512],
            subnet_xyz=True,
            subnet_latent_in_inflate=False,
            subnet_norm=[0, 1, 2, 3, 4],
            latent_in=[4],
            norm_layers=[0, 1, 2, 3, 4, 5, 6, 7],
            weight_norm=True
        )
        self.latent_size = latent_size

def make_3d_grid(bb_min, bb_max, shape):
    size = shape[0] * shape[1] * shape[2]
    pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
    pys = torch.linspace(bb_min[1], bb_max[1], shape[1])
    pzs = torch.linspace(bb_min[2], bb_max[2], shape[2])
    pxs = pxs.view(-1, 1, 1).expand(*shape).contiguous().view(size)
    pys = pys.view(1, -1, 1).expand(*shape).contiguous().view(size)
    pzs = pzs.view(1, 1, -1).expand(*shape).contiguous().view(size)
    p = torch.stack([pxs, pys, pzs], dim=1)
    return p.numpy(), shape

def compute_f1_score(pred_pts, gt_pts, threshold=0.02):
    if len(pred_pts) == 0 or len(gt_pts) == 0: return 0.0
    tree_gt = cKDTree(gt_pts)
    dist_pred_to_gt, _ = tree_gt.query(pred_pts)
    precision = np.mean(dist_pred_to_gt < threshold)
    
    tree_pred = cKDTree(pred_pts)
    dist_gt_to_pred, _ = tree_pred.query(gt_pts)
    recall = np.mean(dist_gt_to_pred < threshold)
    
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config")
    parser.add_argument('--csv-path', type=str, default=None, help="Override YAML dataset path")
    parser.add_argument('--output-dir', type=str, default=None, help="Override output directory")
    parser.add_argument('--save-limit', type=int, default=15, help="Number of random samples to save as PLY")
    parser.add_argument('--eval-limit', type=int, default=None, help="Limit number of samples to evaluate")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for selection")
    parser.add_argument('--smoke-test', action='store_true', help="Run fast test")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    csv_path = args.csv_path if args.csv_path is not None else config['csv_path']
        
    device = torch.device('cuda' if torch.cuda.is_available() and not args.smoke_test else 'cpu')
    model = DeepMend10DPipeline().to(device)
    
    model_path = os.path.join(config['save_dir'], 'best_model.pth')
    print(f"Loading best model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    output_dir = args.output_dir if args.output_dir is not None else os.path.join(config['save_dir'], 'eval')
    os.makedirs(os.path.join(output_dir, 'meshes'), exist_ok=True)
    
    dataset = DeepMend10DDataset(csv_path=csv_path, is_train=False, limit=None)
    
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

    # Optional truncation for evaluation speed
    if args.eval_limit is not None and args.eval_limit < len(paths):
        import random
        random.seed(args.seed)
        random.shuffle(paths)
        paths = paths[:args.eval_limit]
        print(f"Truncated evaluation to {args.eval_limit} samples.")
    
    import random
    random.seed(args.seed)
    random.shuffle(paths)
    
    if args.smoke_test:
        paths = paths[:2]
        grid_res = 32
    else:
        grid_res = 128
        
    grid_pts, grid_shape = make_3d_grid([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5], (grid_res, grid_res, grid_res))
    grid_pts_tensor = torch.from_numpy(grid_pts).float().to(device)
    query_batch_size = 100000
    
    total_iou = 0.0
    total_cd = 0.0
    total_f1 = 0.0
    valid_samples = 0
    num_exports = 0

    print(f"Starting evaluation on {len(paths)} samples...")
    pbar = tqdm(paths)
    
    for path in pbar:
        try:
            d = np.load(path)
            partial_features = d['partial_features'].astype(np.float32)
            complete_points = d['complete_points'].astype(np.float32)
        except:
            continue
            
        p_frac = np.zeros((partial_features.shape[0], 1), dtype=np.float32)
        x_10d = np.hstack([partial_features, p_frac])
        x_10d[:, :3] = x_10d[:, :3] / scale_factor
        scaled_complete_points = complete_points / scale_factor
        
        x_tensor = torch.from_numpy(x_10d).transpose(1, 0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            z_both = model.encoder(x_tensor)
            z = z_both[:, :model.latent_size]
            z_tool = z_both[:, model.latent_size:]
            
            c_preds = []
            for i in range(0, grid_pts_tensor.shape[0], query_batch_size):
                batch_pts = grid_pts_tensor[i:i+query_batch_size].unsqueeze(0)
                N_q = batch_pts.shape[1]
                z_exp = z.unsqueeze(1).expand(-1, N_q, -1).reshape(-1, model.latent_size)
                z_tool_exp = z_tool.unsqueeze(1).expand(-1, N_q, -1).reshape(-1, model.decoder.tool_latent_size)
                pts_flat = batch_pts.reshape(-1, 3)
                
                net_input = torch.cat([z_exp, z_tool_exp, pts_flat], dim=-1)
                c_x, _, _, _ = model.decoder(net_input)
                c_probs = torch.sigmoid(c_x)
                c_preds.append(c_probs.squeeze().cpu().numpy())
            
            c_preds = np.concatenate(c_preds, axis=0)
            vol_pred = c_preds.reshape(grid_shape)
        
        tree_c = cKDTree(scaled_complete_points)
        dist_c, _ = tree_c.query(grid_pts)
        c_gt = (dist_c < occ_threshold).astype(np.int32)
        
        iou = compute_iou((c_preds > 0.5).astype(np.int32), c_gt)
        total_iou += iou

        try:
            verts, faces, normals, values = measure.marching_cubes(vol_pred, level=0.5)
            verts = (verts / (grid_res - 1)) - 0.5
            verts_real = verts * scale_factor
            
            mesh = trimesh.Trimesh(vertices=verts_real, faces=faces)
            if len(mesh.faces) > 0:
                sampled_pts, _ = trimesh.sample.sample_surface(mesh, 8192)
            else:
                sampled_pts = verts_real 
                
        except Exception as e:
            sampled_pts = np.zeros((8192, 3))
            verts_real = np.zeros((3,3))

        cp = complete_points
        if cp.shape[0] > 8192:
            idx_choice = np.random.choice(cp.shape[0], 8192, replace=False)
            cp = cp[idx_choice]
        
        pred_tensor = torch.from_numpy(sampled_pts).float().unsqueeze(0).to(device)
        gt_tensor = torch.from_numpy(cp).float().unsqueeze(0).to(device)
        
        cd_val = chamfer_distance(pred_tensor, gt_tensor).mean().item()
        f1_val = compute_f1_score(sampled_pts, complete_points, threshold=0.02)
        
        total_cd += cd_val
        total_f1 += f1_val
        valid_samples += 1
        
        if num_exports < args.save_limit and len(verts_real) > 3:
            parts = path.split('/')
            object_id = parts[-4]
            variant_id = parts[-3]
            mesh_basename = f"{object_id}_{variant_id}"
            mesh_filename = os.path.join(output_dir, 'meshes', f"{mesh_basename}.ply")
            
            tree_partial = cKDTree(partial_features[:, :3])
            distances, _ = tree_partial.query(verts_real)
            threshold = occ_threshold * scale_factor * 1.5
            colors = np.tile([100, 150, 255, 255], (len(verts_real), 1))
            is_original = distances < threshold
            colors[is_original] = [200, 200, 200, 255]
            mesh.visual.vertex_colors = colors
            mesh.export(mesh_filename)
            num_exports += 1

        pbar.set_postfix({'IoU': f"{total_iou/valid_samples:.3f}", 'CD': f"{total_cd/valid_samples:.4f}", 'F1': f"{total_f1/valid_samples:.3f}"})

    final_iou = total_iou / max(valid_samples, 1)
    final_cd = total_cd / max(valid_samples, 1)
    final_f1 = total_f1 / max(valid_samples, 1)

    print(f"\nEvaluation Complete! [{valid_samples} samples]")
    print(f"Mean Volumetric IoU: {final_iou:.4f}")
    print(f"Mean Chamfer Distance: {final_cd:.4f}")
    print(f"Mean F1-Score@0.02: {final_f1:.4f}")
    print(f"Saved {num_exports} PLY meshes to: {os.path.join(output_dir, 'meshes')}")

    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Mean Volumetric IoU: {final_iou:.4f}\n")
        f.write(f"Mean Chamfer Distance: {final_cd:.4f}\n")
        f.write(f"Mean F1-Score@0.02: {final_f1:.4f}\n")
        f.write(f"Evaluated Samples: {valid_samples}\n")

if __name__ == "__main__":
    main()
