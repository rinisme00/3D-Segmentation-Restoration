import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add the native deepmend to path
sys.path.append(os.path.join(
    os.path.dirname(__file__), 
    "../models/deepmend_native/python"
))

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
        """
        x: (B, 10, N)
        query_pts: (B, N_q, 3)
        """
        z_both = self.encoder(x) # (B, latent_size + tool_latent_size)
        
        # Split into z and z_tool
        z = z_both[:, :self.latent_size]
        z_tool = z_both[:, self.latent_size:]
        
        # Prepare input for decoder
        # The decoder expects net_input as N x (|z| + |bz| + num_dims)
        # where N is the total number of query points across the batch.
        # But wait, decoder doesn't handle batched queries nicely if we don't reshape properly.
        # Let's reshape things so it works. 
        B, N_q, _ = query_pts.shape
        z_expanded = z.unsqueeze(1).expand(-1, N_q, -1).reshape(-1, self.latent_size)
        z_tool_expanded = z_tool.unsqueeze(1).expand(-1, N_q, -1).reshape(-1, self.decoder.tool_latent_size)
        pts_flat = query_pts.reshape(-1, 3)
        
        net_input = torch.cat([z_expanded, z_tool_expanded, pts_flat], dim=-1)
        
        # Decoder forward
        c_x, b_x, r_x, t_x = self.decoder(net_input)
        
        return c_x, b_x, r_x, t_x

def test_pipeline():
    print("Testing DeepMend 10D Pipeline with Gradients...")
    
    device = "cpu" # Test on CPU due to CUDA sm_120 compatibility issues in the current environment
    model = DeepMend10DPipeline().to(device)
    
    # Dummy inputs
    B = 2
    N = 1024
    N_q = 500
    
    # 10D geometric input tensor
    # Ensure requires_grad=True to test gradient flow back to the input
    x = torch.randn(B, 10, N, device=device, requires_grad=True)
    
    # Query points
    query_pts = torch.randn(B, N_q, 3, device=device)
    
    # Dummy GT
    gt_c = torch.empty(B * N_q, 1, device=device).random_(2).float()
    
    print(f"Input shape: {x.shape}, Query points shape: {query_pts.shape}")
    
    # Forward pass
    c_x, b_x, r_x, t_x = model(x, query_pts)
    
    print(f"Output shape (c_x): {c_x.shape}")
    
    # Dummy BCE Loss
    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(c_x, gt_c)
    
    print(f"Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Assert gradients flow back to x
    assert x.grad is not None, "Gradients did not flow back to input x!"
    assert torch.sum(torch.abs(x.grad)) > 0, "Gradients at x are zero!"
    
    print("SUCCESS: Gradients successfully flowed from Occupancy loss back to the 10D geometric input tensor.")

if __name__ == "__main__":
    test_pipeline()
