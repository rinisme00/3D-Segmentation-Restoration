import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from restoration.utils.point_alignment import align_fracture_probabilities_scipy, align_fracture_probabilities_torch

def test_knn_alignment():
    print("--- Testing K-NN Alignment Interface ---")
    
    # 1. Generate a mock 9D original partial point cloud (N=1000)
    N = 1000
    x_orig_9d = np.random.randn(N, 9).astype(np.float32)
    xyz_orig = x_orig_9d[:, :3]
    
    # 2. Simulate PointNeXt output which is scrambled and possibly downsampled (M=800)
    # We will pick 800 random points from original, add slight noise, and attach mock probabilities
    M = 800
    indices = np.random.choice(N, M, replace=False)
    xyz_pred = xyz_orig[indices] + np.random.randn(M, 3).astype(np.float32) * 1e-4
    
    # Assign clear probabilities based on Z coordinate (just for verification)
    p_frac_pred = (xyz_pred[:, 2] > 0).astype(np.float32) 
    
    x_pred_4d = np.hstack([xyz_pred, p_frac_pred[:, np.newaxis]])
    
    # 3. Test SciPy Implementation
    print("Running SciPy cKDTree alignment...")
    x_10d_np = align_fracture_probabilities_scipy(x_orig_9d, x_pred_4d)
    
    assert x_10d_np.shape == (N, 10), f"Expected shape {(N, 10)}, got {x_10d_np.shape}"
    assert np.allclose(x_10d_np[:, :9], x_orig_9d), "First 9 dims should be identical to orig"
    
    # 4. Test PyTorch Implementation
    print("Running PyTorch cdist alignment...")
    x_orig_9d_tensor = torch.from_numpy(x_orig_9d).cuda()
    x_pred_4d_tensor = torch.from_numpy(x_pred_4d).cuda()
    
    x_10d_torch = align_fracture_probabilities_torch(x_orig_9d_tensor, x_pred_4d_tensor)
    
    assert x_10d_torch.shape == (N, 10), f"Expected shape {(N, 10)}, got {x_10d_torch.shape}"
    assert torch.allclose(x_10d_torch[:, :9], x_orig_9d_tensor), "First 9 dims should be identical to orig"
    
    # Ensure Torch and SciPy match
    p_frac_np = x_10d_np[:, 9]
    p_frac_torch = x_10d_torch[:, 9].cpu().numpy()
    
    mismatches = np.sum(p_frac_np != p_frac_torch)
    assert mismatches == 0, f"Torch and SciPy disagree on {mismatches} points!"
    
    print("SUCCESS: K-NN Alignment logic perfectly maps P_frac to the unscrambled 9D coordinate space.")

if __name__ == "__main__":
    test_knn_alignment()
