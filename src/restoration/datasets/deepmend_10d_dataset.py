import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial import cKDTree

class DeepMend10DDataset(Dataset):
    """
    Dynamic 10D Dataset for DeepMend Overfitting.
    Loads a .npz file with partial and complete point clouds, 
    dynamically computes normalized coordinates, and samples query points
    with approximate occupancy ground truth.
    """
    def __init__(self, npz_paths=None, csv_path=None, is_train=True, num_query_points=2048, occ_threshold=0.02, scale_factor=2.0, limit=None):
        super().__init__()
        import csv
        import hashlib
        
        self.npz_paths = npz_paths if npz_paths is not None else []
        self.num_query_points = num_query_points
        self.occ_threshold = occ_threshold
        self.scale_factor = scale_factor
        self.is_train = is_train
        
        if csv_path is not None:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sample_name = row['sample_name']
                    npz_rel_path = row['output_path']
                    if npz_rel_path.startswith('../preprocessed'):
                        csv_root = Path(csv_path).resolve().parent
                        npz_full_path = str((csv_root / npz_rel_path).resolve())
                    else:
                        npz_full_path = os.path.normpath(os.path.join(os.path.dirname(csv_path), npz_rel_path))
                    
                    hash_val = int(hashlib.md5(sample_name.encode('utf-8')).hexdigest(), 16) % 100
                    if is_train and hash_val < 80:
                        self.npz_paths.append(npz_full_path)
                    elif not is_train and hash_val >= 80:
                        self.npz_paths.append(npz_full_path)
                        
        if limit is not None:
            self.npz_paths = self.npz_paths[:limit]
            
        print(f"Loaded {len(self.npz_paths)} sample paths for {'Training' if is_train else 'Validation'}")

    def __len__(self):
        return len(self.npz_paths)

    def __getitem__(self, idx):
        path = self.npz_paths[idx]
        try:
            d = np.load(path)
            partial_points = d['partial_points'].astype(np.float32)
            partial_features = d['partial_features'].astype(np.float32)
            complete_points = d['complete_points'].astype(np.float32)
        except Exception as e:
            print(f"Warning: Skipping corrupted file {path}: {e}")
            import random
            return self.__getitem__(random.randint(0, len(self.npz_paths) - 1))
        
        if partial_features.shape[1] == 9:
            p_frac = np.zeros((partial_features.shape[0], 1), dtype=np.float32)
            x_10d = np.hstack([partial_features, p_frac])
        else:
            raise ValueError(f"Expected 9D features, got {partial_features.shape[1]}D")
            
        # Apply Safe Geometric Augmentation (Jitter & Point Dropout) ONLY during training
        if self.is_train:
            # 1. Jittering (Add small Gaussian noise to XYZ coordinates)
            jitter = np.random.normal(0, 0.01, size=(x_10d.shape[0], 3)).astype(np.float32)
            jitter = np.clip(jitter, -0.05, 0.05)
            x_10d[:, :3] += jitter
            
            # 2. Point Dropout (Randomly replace 0-20% of points with duplicates of other points)
            # This simulates varying sparsity without changing the fixed tensor shape (8192, 10).
            drop_ratio = np.random.uniform(0, 0.2)
            num_drop = int(drop_ratio * x_10d.shape[0])
            if num_drop > 0:
                drop_indices = np.random.choice(x_10d.shape[0], num_drop, replace=False)
                replace_indices = np.random.choice(x_10d.shape[0], num_drop, replace=True)
                x_10d[drop_indices] = x_10d[replace_indices]
                
        complete_points = d['complete_points'].astype(np.float32)
        
        x_10d[:, :3] = x_10d[:, :3] / self.scale_factor
        complete_points = complete_points / self.scale_factor
        partial_points = x_10d[:, :3]
        
        # Sample query points in [-0.5, 0.5]
        # Mix of uniform random and near-surface
        n_uniform = self.num_query_points // 2
        n_surface = self.num_query_points - n_uniform
        
        query_uniform = np.random.uniform(-0.5, 0.5, size=(n_uniform, 3)).astype(np.float32)
        
        # Near surface (perturb complete points)
        surface_idx = np.random.choice(complete_points.shape[0], n_surface, replace=True)
        query_surface = complete_points[surface_idx] + np.random.normal(scale=0.05, size=(n_surface, 3)).astype(np.float32)
        
        query_pts = np.vstack([query_uniform, query_surface])
        
        # Compute Occupancy using KDTree
        # c_gt (complete occupancy)
        tree_c = cKDTree(complete_points)
        dist_c, _ = tree_c.query(query_pts)
        c_gt = (dist_c < self.occ_threshold).astype(np.float32)
        
        # b_gt (broken occupancy)
        tree_b = cKDTree(partial_points)
        dist_b, _ = tree_b.query(query_pts)
        b_gt = (dist_b < self.occ_threshold).astype(np.float32)
        
        # r_gt (restoration occupancy)
        r_gt = (c_gt > 0) & (b_gt == 0)
        r_gt = r_gt.astype(np.float32)
        
        # t_gt (tool occupancy) -> approx same as b_gt for basic test
        t_gt = b_gt.copy()
        
        # Transpose x_10d for PointNet: (10, N)
        x_10d_tensor = torch.from_numpy(x_10d).transpose(1, 0)
        
        return (
            x_10d_tensor, 
            torch.from_numpy(query_pts), 
            torch.from_numpy(c_gt).unsqueeze(-1), 
            torch.from_numpy(b_gt).unsqueeze(-1), 
            torch.from_numpy(r_gt).unsqueeze(-1), 
            torch.from_numpy(t_gt).unsqueeze(-1)
        )
