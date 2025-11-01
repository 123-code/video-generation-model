import torch 
import torch.nn as nn
from model.attention import SpatialAttention, TemporalAttention
from model.patch_embed import PatchEmbed

class VideoDiTBlock(nn.Module):
    def __init__(self,dim,heads=8,mlp_ratio=4):
        super().__init__()
        self.spatial_attn = SpatialAttention(dim,heads)
        self.temporal_attn = TemporalAttention(dim,heads)
        self.ffn = nn.Sequential(
            nn.Linear(dim,dim*mlp_ratio),
            nn.GELU(),
            nn.Linear(dim*mlp_ratio,dim),)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.t_proj = nn.Linear(dim,dim*2)
    def forward(self,x,t_emb):
        x_spatial = rearrange(x,'b (t h w) d -> b t (h w) d',t=8,h=8,w=8)
        x_spatial = self.spatial_attn(self.norm1(x_spatial)) + x_spatial
        x_temporal = rearrange(x_spatial, 'b t (h w) d -> b t (h w) d', t=8, h=8, w=8)
        x_temporal = self.temporal_attn(self.norm2(x_temporal)) + x_temporal
        x = rearrange(x_temporal, 'b t (h w) d -> b (t h w) d', t=8, h=8, w=8)
        x = self.ffn(self.norm2(x)) + x
        scale,shift = self.t_proj(t_emb).chunk(2,dim=-1)
        scale,shift = scale.unsqueeze(1),shift.unsqueeze(1)
        x = x * (1 + scale) + shift
        return x

class VideoDiT(nn.Module):
    def __init__(self,latent_dim=1024,dim=768,depth=12,heads=12):
        super().__init__()
        self.patch_embed=PatchEmbed(in_channels=8,patch_size=2,embed_dim=dim)
        self.t_embed=nn.Sequential(
            nn.Linear(1,dim*4),
            nn.SiLU(),
            nn.Linear(dim*4,dim)
        )
        self.blocks = nn.ModuleList([VideoDiTBlock(dim,heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim*(64//2*64//2*16//2),latent_dim)

    def forward(self,latent,t):
        tokens = self.patch_embed(latent)
        t_emb = self.t_embed(t.float().unsqueeze(1))
        x = tokens + t_emb.unsqueeze(1)
        for block in self.blocks:
            x = block(x,t_emb)
        x = self.norm(x)
        x = self.out_proj(x.flatten(1))
        return x

if __name__ == "__main__":
    latent = torch.randn(1, 8, 16, 64, 64)  
    dit = VideoDiT(latent_dim=8192, dim=768)  
    t = torch.tensor([500]).to(device)  
    out = dit(latent, t)
    print(f"Output shape: {out.shape}")