import torch 
import torch.nn as nn
from model.attention import SpatialAttention, TemporalAttention

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

