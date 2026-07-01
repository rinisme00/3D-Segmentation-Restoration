import torch

def chamfer_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute mean squared Chamfer distance with standard PyTorch ops.
    
    pred: [B, N, 3]
    target: [B, M, 3]
    returns: [B] average squared distance
    """
    dist = torch.cdist(pred, target, p=2).pow(2)
    dist1 = dist.min(dim=2).values
    dist2 = dist.min(dim=1).values
    return dist1.mean(dim=-1) + dist2.mean(dim=-1)

def f_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.01) -> float:
    """Computes average F-Score at threshold.
    
    pred: [B, N, 3]
    target: [B, M, 3]
    threshold: float
    returns: float (average F-Score across batch)
    """
    with torch.no_grad():
        dist = torch.cdist(pred, target, p=2)
        d1 = dist.min(dim=2).values
        d2 = dist.min(dim=1).values
        
        precision = (d1 < threshold).float().mean(dim=-1)
        recall = (d2 < threshold).float().mean(dim=-1)
        f_score_val = 2 * precision * recall / (precision + recall + 1e-8)
        return f_score_val.mean().item()

def fracture_region_cd(
    pred: torch.Tensor,
    target: torch.Tensor,
    fracture_mask: torch.Tensor,
    threshold: float = 0.3,
) -> torch.Tensor:
    """Compute Chamfer Distance restricted to fracture-region target points.

    Selects target points where ``fracture_mask > threshold`` (or True for
    boolean masks), then computes the one-directional CD from those target
    points to the nearest points in ``pred``.  This measures how well the
    generated shape covers the missing/fractured region specifically.

    Args:
        pred: [B, N, 3] predicted complete point cloud.
        target: [B, M, 3] ground-truth complete point cloud.
        fracture_mask: [B, M] fracture probability or binary mask
            aligned to target.
        threshold: probability threshold for selecting fracture points
            (ignored for boolean masks).

    Returns:
        frac_cd: [B] mean squared distance from fracture-region target
            points to their nearest neighbours in pred.
    """
    B = pred.shape[0]
    frac_cd_list = []

    for b in range(B):
        mask = fracture_mask[b]
        if mask.dtype == torch.bool:
            frac_idx = mask
        else:
            frac_idx = mask > threshold

        frac_target = target[b][frac_idx]  # [K, 3]
        if frac_target.shape[0] == 0:
            frac_cd_list.append(torch.tensor(0.0, device=pred.device))
            continue

        dist = torch.cdist(frac_target.unsqueeze(0), pred[b].unsqueeze(0), p=2).pow(2)
        frac_cd_list.append(dist.min(dim=2).values.mean())

    return torch.stack(frac_cd_list)
