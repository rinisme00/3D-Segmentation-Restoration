import sys
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import yaml

CODE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(CODE_ROOT, "models", "deepmend_native", "python"))
sys.path.append(os.path.join(CODE_ROOT, "src"))

from restoration.datasets.deepmend_10d_dataset import DeepMend10DDataset
from networks.encoder import DeepMendEncoder10D
from networks.decoder_z_lb_occ_leaky import Decoder as DeepMendDecoder
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

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

    def forward(self, x, query_pts):
        z_both = self.encoder(x)
        z = z_both[:, :self.latent_size]
        z_tool = z_both[:, self.latent_size:]
        
        B, N_q, _ = query_pts.shape
        z_expanded = z.unsqueeze(1).expand(-1, N_q, -1).reshape(-1, self.latent_size)
        z_tool_expanded = z_tool.unsqueeze(1).expand(-1, N_q, -1).reshape(-1, self.decoder.tool_latent_size)
        pts_flat = query_pts.reshape(-1, 3)
        
        net_input = torch.cat([z_expanded, z_tool_expanded, pts_flat], dim=-1)
        c_x, b_x, r_x, t_x = self.decoder(net_input)
        
        c_x = c_x.view(B, N_q, 1)
        b_x = b_x.view(B, N_q, 1)
        r_x = r_x.view(B, N_q, 1)
        t_x = t_x.view(B, N_q, 1)
        
        return c_x, b_x, r_x, t_x

def apply_se3_augmentation(x_10d, query_pts, config):
    B, _, N = x_10d.shape
    device = x_10d.device
    
    x = x_10d.transpose(1, 2).clone()
    q = query_pts.clone()
    
    cfg = config['augmentations']['rigid_se3']
    
    for i in range(B):
        # 1. Mirror XYZ
        if torch.rand(1).item() < cfg['mirror_xyz_prob']:
            if torch.rand(1).item() < 0.5: # Mirror X
                x[i, :, 0] *= -1
                x[i, :, 3] *= -1
                q[i, :, 0] *= -1
            if torch.rand(1).item() < 0.5: # Mirror Y
                x[i, :, 1] *= -1
                x[i, :, 4] *= -1
                q[i, :, 1] *= -1
            if torch.rand(1).item() < 0.5: # Mirror Z
                x[i, :, 2] *= -1
                x[i, :, 5] *= -1
                q[i, :, 2] *= -1
                
        # 2. Isotropic Scale (REMOVED)
        # Isotropic scaling decoupled the pre-computed Local Density feature (Column 6), 
        # causing catastrophic geometric desynchronization. Allowed only isometries.
        
        # 3. Y-Rotation
        if cfg['random_y_rotation']:
            theta = torch.empty(1).uniform_(0, 2 * math.pi).item()
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            R = torch.tensor([
                [cos_t, 0, sin_t],
                [0, 1, 0],
                [-sin_t, 0, cos_t]
            ], device=device)
            x[i, :, :3] = torch.matmul(x[i, :, :3], R.T)
            x[i, :, 3:6] = torch.matmul(x[i, :, 3:6], R.T)
            
            # Interrogation 3: Normal Vector Unit Integrity
            # Enforce strict L2 normalization after SE(3) rotation to prevent float drift
            x[i, :, 3:6] = torch.nn.functional.normalize(x[i, :, 3:6], p=2, dim=-1)
            
            q[i] = torch.matmul(q[i], R.T)
            
        # 4. Cuboid Cutout (REMOVED)
        # Cuboid cutout decoupled the pre-computed Surface Variation and Eigenentropy (Columns 7, 8) 
        # by altering the K-Nearest Neighbor topology without re-computing PCA on the GPU.
                
    return x.transpose(1, 2), q

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config")
    parser.add_argument('--smoke-test', action='store_true', help="Run fast test")
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    os.makedirs(config['save_dir'], exist_ok=True)
    
    if args.smoke_test:
        device = torch.device('cpu')
        print("Smoke test forced to CPU.")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    limit = 2 if args.smoke_test else None
    
    train_dataset = DeepMend10DDataset(csv_path=config['csv_path'], is_train=True, limit=limit)
    val_dataset = DeepMend10DDataset(csv_path=config['csv_path'], is_train=False, limit=limit)
    
    # Disable internal dataloader jitter/dropout as requested by config
    train_dataset.is_train = False 
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=args.num_workers)
    
    model = DeepMend10DPipeline().to(device)
    
    if config.get('pretrained_checkpoint'):
        print(f"Loading pretrained weights: {config['pretrained_checkpoint']}")
        model.load_state_dict(torch.load(config['pretrained_checkpoint'], map_location=device))
        
    # Split Optimizer
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': float(config['encoder_lr'])},
        {'params': model.decoder.parameters(), 'lr': float(config['decoder_lr'])}
    ], weight_decay=float(config['weight_decay']))
    
    # SequentialLR Scheduler
    warmup_epochs = config['scheduler']['warmup_epochs']
    total_epochs = config['epochs']
    
    scheduler1 = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    scheduler2 = CosineAnnealingLR(optimizer, T_max=(total_epochs - warmup_epochs), eta_min=float(config['scheduler']['min_lr']))
    scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_epochs])
    
    loss_bce = torch.nn.BCELoss()
    
    use_wandb = not args.smoke_test and os.environ.get('WANDB_DISABLED', 'false').lower() != 'true'
    if use_wandb:
        try:
            import wandb

            wandb.init(project="DeepMend-Restoration", name="finetune_fb", config=config)
        except ImportError:
            use_wandb = False
            print("W&B is not installed; continuing without online logging.")
        
    best_val_loss = float('inf')
    
    for epoch in range(total_epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]")
        for batch in pbar:
            x_10d, query_pts, c_gt, b_gt, r_gt, t_gt = [b.to(device) for b in batch]
            
            # Apply GPU Augmentations
            x_10d, query_pts = apply_se3_augmentation(x_10d, query_pts, config)
            
            optimizer.zero_grad()
            c_pred, b_pred, r_pred, t_pred = model(x_10d, query_pts)
            
            loss_c = loss_bce(torch.sigmoid(c_pred), c_gt)
            loss_t = loss_bce(torch.sigmoid(t_pred), t_gt)
            loss = loss_c + loss_t
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            if args.smoke_test: break
            
        train_loss /= len(train_loader)
        scheduler.step()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Val]"):
                x_10d, query_pts, c_gt, b_gt, r_gt, t_gt = [b.to(device) for b in batch]
                c_pred, b_pred, r_pred, t_pred = model(x_10d, query_pts)
                
                loss_c = loss_bce(torch.sigmoid(c_pred), c_gt)
                loss_t = loss_bce(torch.sigmoid(t_pred), t_gt)
                val_loss += (loss_c + loss_t).item()
                if args.smoke_test: break
                
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR Encoder: {optimizer.param_groups[0]['lr']:.2e}")
        
        if use_wandb:
            wandb.log({
                "train_loss": train_loss, 
                "val_loss": val_loss, 
                "encoder_lr": optimizer.param_groups[0]['lr'],
                "decoder_lr": optimizer.param_groups[1]['lr'],
                "epoch": epoch
            })
            
        torch.save(model.state_dict(), os.path.join(config['save_dir'], 'last_model.pth'))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(config['save_dir'], 'best_model.pth'))
            print(f"Saved new best model (Val Loss: {best_val_loss:.4f})")
            
        if args.smoke_test: break

if __name__ == "__main__":
    main()
