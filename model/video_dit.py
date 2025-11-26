import torch 
import torch.nn as nn
from einops import rearrange
from model.patch_embed import PatchEmbed
from model.dit import VideoDiTBlock
from model.Blocks import get_time_embedding 

class VideoDiT(nn.Module):
    def __init__(self, in_channels=4, T=16, H=32, W=32, patch_size=2, dim=768, depth=6, heads=8):
        super().__init__()
        self.in_channels = in_channels
        self.dim = dim
        self.patch_embed = PatchEmbed(in_channels, patch_size, dim)
        self.t_embed = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.SiLU(),
            nn.Linear(dim*4, dim),
        )
        num_frames = T // patch_size
        spatial_size = H // patch_size

        self.blocks = nn.ModuleList([
            VideoDiTBlock(dim, heads, num_frames=num_frames, spatial_size=spatial_size) 
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, in_channels * (patch_size ** 3))
        self.T,self.H,self.W,self.patch_size=T,H,W,patch_size
        


    def forward(self, latent, t):
        tokens = self.patch_embed(latent)
        t_freq = get_time_embedding(t, self.dim)
        t_emb = self.t_embed(t_freq)
        x = tokens + t_emb.unsqueeze(1)

        for block in self.blocks:
            x = block(x,t_emb)
        x = self.norm(x)
        x = self.out_proj(x)

        b,n,_ = x.shape
        t_patches = self.T // self.patch_size
        h_patches = self.H // self.patch_size
        w_patches = self.W // self.patch_size
        noise_pred = rearrange(x,'b (t h w) (c pt ph pw) -> b c (t pt) (h ph) (w pw)',
        t=t_patches,h=h_patches,w=w_patches,c=self.in_channels,pt=self.patch_size,ph=self.patch_size,pw=self.patch_size)
        return noise_pred