import torch
import torch.nn as nn 
from einops import rearrange
import math

class PatchEmbed(nn.Module):
    def __init__(self, in_channels=8, patch_size=2, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.patch_dim = self.patch_size ** 3 * self.in_channels
        self.projection = nn.Linear(self.patch_dim, self.embed_dim)

    def get_pos_embed(self, embed_dim, grid_size_h, grid_size_w, num_frames):
        spatial_dim = embed_dim // 2
        temporal_dim = embed_dim - spatial_dim

        pos_h = torch.arange(grid_size_h, dtype=torch.float32).unsqueeze(1)
        pos_w = torch.arange(grid_size_w, dtype=torch.float32).unsqueeze(1)
        div_term_spatial = torch.exp(torch.arange(0, spatial_dim//2, 2).float() * -(math.log(10000.0) / (spatial_dim//2)))
        pe_h = torch.sin(pos_h * div_term_spatial)
        pe_w = torch.cos(pos_w * div_term_spatial)
        spatial_pe = torch.cat([pe_h, pe_w], dim=1).flatten()

        pos_t = torch.arange(num_frames, dtype=torch.float32).unsqueeze(1)
        div_term_temporal = torch.exp(torch.arange(0, temporal_dim//2, 2).float() * -(math.log(10000.0) / (temporal_dim//2)))
        pe_t_sin = torch.sin(pos_t * div_term_temporal)
        pe_t_cos = torch.cos(pos_t * div_term_temporal)
        temporal_pe = torch.cat([pe_t_sin, pe_t_cos], dim=1).flatten()

        pos_emb = torch.cat([spatial_pe, temporal_pe], dim=0)
        pos_emb = pos_emb[:embed_dim]
        pos_emb = pos_emb.unsqueeze(0).repeat(grid_size_h * grid_size_w * num_frames, 1)
        return pos_emb
    
    def forward(self, latent):
        if latent.dim() == 4:
            B, C, H, W = latent.shape
            T = 1
            latent = latent.unsqueeze(2)
        else:
            B, C, T, H, W = latent.shape

        num_patches_t = T // self.patch_size
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        
        if T % self.patch_size != 0:
            T = num_patches_t * self.patch_size
            latent = latent[:, :, :T, :, :]
        if H % self.patch_size != 0:
            H = num_patches_h * self.patch_size
            latent = latent[:, :, :, :H, :]
        if W % self.patch_size != 0:
            W = num_patches_w * self.patch_size
            latent = latent[:, :, :, :, :W]

        patches = rearrange(latent, 'b c (t pt) (h p1) (w p2) -> b (t h w) (pt p1 p2 c)', 
                          pt=self.patch_size, p1=self.patch_size, p2=self.patch_size, 
                          h=num_patches_h, w=num_patches_w, t=num_patches_t)
        emb = self.projection(patches)

        pos_emb = self.get_pos_embed(self.embed_dim, num_patches_h, num_patches_w, num_patches_t)
        emb = emb + pos_emb.unsqueeze(0)
        return emb



if __name__ == "__main__":
    latent_path = "../latents_test/437_How_To_Ride_A_Bike_ride_bike_f_cm_np1_ba_med_0.pt"
    latent = torch.load(latent_path)
    
    print(f"Loaded latent shape: {latent.shape}")
    
    if latent.dim() == 4:
        latent = latent.unsqueeze(0)
    
    print(f"Input shape: {latent.shape}")
    
    patch_embed = PatchEmbed(in_channels=8, patch_size=2, embed_dim=768)
    output = patch_embed(latent)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected: [batch_size, num_patches, embed_dim]")
