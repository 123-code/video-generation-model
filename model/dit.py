import torch 
import torch.nn as nn
from einops import rearrange
from model.attention import SpatialAttention, TemporalAttention

class VideoDiTBlock(nn.Module):
    def __init__(self,dim,heads=8,mlp_ratio=4,num_frames=8,spatial_size=32):
        super().__init__()
        self.spatial_attn = SpatialAttention(dim,heads)
        self.temporal_attn = TemporalAttention(dim,heads)
        self.ffn = nn.Sequential(
            nn.Linear(dim,dim*mlp_ratio),
            nn.GELU(),
            nn.Linear(dim*mlp_ratio,dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.t_proj = nn.Linear(dim,dim*2)
        self.num_frames = num_frames
        self.spatial_size = spatial_size
        
    def forward(self,x,t_emb):
        b, n, d = x.shape
        t = self.num_frames
        hw = n // t
        h = w = int(hw ** 0.5)
        
        x_spatial = rearrange(x,'b (t h w) d -> (b t) (h w) d',t=t,h=h,w=w)
        x_spatial = self.spatial_attn(self.norm1(x_spatial))
        x_spatial = rearrange(x_spatial,'(b t) (h w) d -> b (t h w) d',b=b,t=t,h=h,w=w)
        x = x + x_spatial
        
        x_temporal = rearrange(x, 'b (t h w) d -> (b h w) t d', t=t, h=h, w=w)
        x_temporal = self.temporal_attn(self.norm2(x_temporal))
        x_temporal = rearrange(x_temporal, '(b h w) t d -> b (t h w) d', b=b, t=t, h=h, w=w)
        x = x + x_temporal
        
        x = x + self.ffn(self.norm3(x))
        
        scale,shift = self.t_proj(t_emb).chunk(2,dim=-1)
        scale,shift = scale.unsqueeze(1),shift.unsqueeze(1)
        x = x * (1 + scale) + shift
        return x
