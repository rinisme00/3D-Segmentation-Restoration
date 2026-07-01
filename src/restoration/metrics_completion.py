import torch
import torch.nn as nn
from extensions.chamfer_dist.chamfer import ChamferFunction, ChamferDistanceL1, ChamferDistanceL2


class CompletionMetrics:
    def __init__(self):
        self.chamfer_l1 = ChamferDistanceL1()
        self.chamfer_l2 = ChamferDistanceL2()

    def compute_cd(self, pred, gt):
        # pred, gt: [B, N, 3]
        loss_l1 = self.chamfer_l1(pred, gt).item()
        loss_l2 = self.chamfer_l2(pred, gt).item()
        return loss_l1, loss_l2

    def compute_f1(self, pred, gt, thresholds=[0.01, 0.02]):
        dist1, dist2 = ChamferFunction.apply(pred, gt)
        dist1 = torch.sqrt(dist1)
        dist2 = torch.sqrt(dist2)

        f1_scores = {}
        for th in thresholds:
            precision = (dist1 < th).float().mean(dim=-1)
            recall = (dist2 < th).float().mean(dim=-1)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
            f1_scores[th] = f1.mean().item()
        
        return f1_scores

    def compute_frac_cd(self, pred, gt, fracture_prob=None, fracture_mask_gt=None, threshold=0.5):
        """
        Compute CD only on fracture regions.
        """
        if fracture_mask_gt is not None:
            mask = fracture_mask_gt > threshold
        elif fracture_prob is not None:
            mask = fracture_prob > threshold
        else:
            return None # Cannot compute Frac-CD without a mask

        b_size = pred.shape[0]
        frac_cd_total = 0.0
        valid_batches = 0

        for b in range(b_size):
            b_mask = mask[b].squeeze()
            if b_mask.sum() == 0:
                continue

            gt_frac = gt[b][b_mask].unsqueeze(0)
            if gt_frac.shape[1] > 0:
                cd_val = self.chamfer_l1(pred[b].unsqueeze(0), gt_frac).item()
                frac_cd_total += cd_val
                valid_batches += 1
                
        if valid_batches == 0:
            return None
            
        return frac_cd_total / valid_batches
