import torch
from scipy.spatial import cKDTree
import numpy as np

def align_fracture_probabilities_scipy(x_orig_9d: np.ndarray, x_pred_4d: np.ndarray) -> np.ndarray:
    """
    Aligns fracture probabilities from a potentially scrambled prediction point cloud
    (e.g., from PointNeXt voxel/FPS downsampling) to the original 9D coordinates.

    Args:
        x_orig_9d (np.ndarray): Original 9D partial point cloud (N x 9). First 3 cols are XYZ.
        x_pred_4d (np.ndarray): Predicted fracture probabilities (M x 4). First 3 cols are XYZ, 4th is P_frac.

    Returns:
        np.ndarray: Strict 10D tensor (N x 10) combining original 9D features with aligned P_frac.
    """
    if not isinstance(x_orig_9d, np.ndarray) or not isinstance(x_pred_4d, np.ndarray):
        raise TypeError("Inputs must be numpy arrays.")

    xyz_orig = x_orig_9d[:, :3]
    xyz_pred = x_pred_4d[:, :3]
    p_frac_pred = x_pred_4d[:, 3]

    # Build k-d tree on the predicted XYZ coordinates
    tree = cKDTree(xyz_pred)

    # For each original point, find the nearest predicted point
    # k=1 returns distances and indices of the nearest neighbor
    distances, indices = tree.query(xyz_orig, k=1)

    # Extract the corresponding P_frac
    p_frac_aligned = p_frac_pred[indices]

    # Concatenate to form the 10D output
    x_10d = np.hstack([x_orig_9d, p_frac_aligned[:, np.newaxis]])

    return x_10d

def align_fracture_probabilities_torch(x_orig_9d: torch.Tensor, x_pred_4d: torch.Tensor) -> torch.Tensor:
    """
    PyTorch native implementation of K-NN alignment for GPU efficiency.
    
    Args:
        x_orig_9d (torch.Tensor): Original 9D partial point cloud (N x 9).
        x_pred_4d (torch.Tensor): Predicted point cloud (M x 4).
        
    Returns:
        torch.Tensor: Aligned 10D tensor (N x 10).
    """
    xyz_orig = x_orig_9d[:, :3]
    xyz_pred = x_pred_4d[:, :3]
    p_frac_pred = x_pred_4d[:, 3]

    # Compute pairwise Euclidean distances (N x M)
    # Using cdist for efficiency
    distances = torch.cdist(xyz_orig, xyz_pred)

    # Find the index of the nearest neighbor in prediction for each original point
    _, indices = torch.min(distances, dim=1)

    # Extract aligned probabilities
    p_frac_aligned = p_frac_pred[indices]

    # Concatenate
    x_10d = torch.cat([x_orig_9d, p_frac_aligned.unsqueeze(1)], dim=1)

    return x_10d
