import sys
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

CODE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(CODE_ROOT))
sys.path.append(os.path.join(CODE_ROOT, "models", "deepmend_native", "python"))
sys.path.append(os.path.join(CODE_ROOT, "src"))

from restoration.datasets.deepmend_10d_dataset import DeepMend10DDataset
from networks.encoder import DeepMendEncoder10D
from networks.decoder_z_lb_occ_leaky import Decoder as DeepMendDecoder

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

def main():
    parser = argparse.ArgumentParser(description="DeepMend 10D Full Training")
    parser.add_argument(
        '--csv-path',
        type=str,
        default=os.environ.get(
            "DEEPMEND_CSV_PATH",
            os.path.join(PROJECT_ROOT, "preprocessed", "restoration", "completion_pairs_9d", "sample_index.csv"),
        ),
    )
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument(
        '--output-dir',
        type=str,
        default=os.environ.get(
            "DEEPMEND_OUTPUT_DIR",
            os.path.join(PROJECT_ROOT, "results", "restoration", "deepmend_10d"),
        ),
    )
    parser.add_argument('--smoke-test', action='store_true', help='Run quickly on 1 batch for verification')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--pretrained-weights', type=str, default=None, help='Path to pre-trained model weights for fine-tuning')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.smoke_test:
        device = torch.device('cpu')
        print(f"Smoke test forced to CPU to bypass CUDA capability sm_120 mismatch.")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
    
    limit = 2 if args.smoke_test else None
    
    train_dataset = DeepMend10DDataset(csv_path=args.csv_path, is_train=True, limit=limit)
    val_dataset = DeepMend10DDataset(csv_path=args.csv_path, is_train=False, limit=limit)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    model = DeepMend10DPipeline().to(device)
    
    if args.pretrained_weights:
        print(f"Loading pre-trained weights from {args.pretrained_weights}")
        model.load_state_dict(torch.load(args.pretrained_weights, map_location=device))
        
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    loss_bce_with_logits = torch.nn.BCEWithLogitsLoss()
    loss_bce = torch.nn.BCELoss()
    
    # WandB
    use_wandb = os.environ.get('WANDB_DISABLED', 'false').lower() != 'true'
    if use_wandb:
        try:
            import wandb
            wandb.init(project="3D-Segmentation-Restoration", name="DeepMend-10D", config=vars(args))
        except ImportError:
            use_wandb = False
            print("WandB not installed. Skipping logging.")
            
    best_val_loss = math.inf
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            x_10d, query_pts, c_gt, b_gt, r_gt, t_gt = [t.to(device) for t in batch]
            
            optimizer.zero_grad()
            c_x, b_x, r_x, t_x = model(x_10d, query_pts)
            
            loss_c = loss_bce_with_logits(c_x, c_gt)
            loss_b = loss_bce(b_x, b_gt)
            loss_r = loss_bce(r_x, r_gt)
            loss_t = loss_bce_with_logits(t_x, t_gt)
            
            total_loss = loss_c + loss_b + loss_r + loss_t
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            pbar.set_postfix({'loss': total_loss.item()})
            
            if args.smoke_test and batch_idx == 0:
                break
                
        train_loss /= len(train_loader) if not args.smoke_test else 1
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
                x_10d, query_pts, c_gt, b_gt, r_gt, t_gt = [t.to(device) for t in batch]
                c_x, b_x, r_x, t_x = model(x_10d, query_pts)
                
                loss_c = loss_bce_with_logits(c_x, c_gt)
                loss_b = loss_bce(b_x, b_gt)
                loss_r = loss_bce(r_x, r_gt)
                loss_t = loss_bce_with_logits(t_x, t_gt)
                total_loss = loss_c + loss_b + loss_r + loss_t
                val_loss += total_loss.item()
                
                if args.smoke_test and batch_idx == 0:
                    break
        
        val_loss /= len(val_loader) if not args.smoke_test else 1
        
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if use_wandb:
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch+1})
            
        if val_loss < best_val_loss and not args.smoke_test:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))
            
        if not args.smoke_test and (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pth"))
            
    print("Training finished.")

if __name__ == "__main__":
    main()
