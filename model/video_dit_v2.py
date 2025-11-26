import torch 
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=4096):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
        
    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, rope_cos=None, rope_sin=None):
        b, n, d = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        if rope_cos is not None and rope_sin is not None:
            q, k = apply_rotary_pos_emb(q, k, rope_cos.unsqueeze(0).unsqueeze(0), rope_sin.unsqueeze(0).unsqueeze(0))
        
        attn = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(attn, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class DiTBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, mlp_mult=4, dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, heads, dim_head, dropout)
        self.norm2 = RMSNorm(dim)
        self.ff = FeedForward(dim, mlp_mult, dropout)
        
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6)
        )
        
    def forward(self, x, c, rope_cos=None, rope_sin=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(c).chunk(6, dim=-1)
        
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope_cos, rope_sin)
        x = x + gate_mlp.unsqueeze(1) * self.ff(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class SpatioTemporalDiTBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, mlp_mult=4, dropout=0.0, num_frames=8, spatial_size=16):
        super().__init__()
        self.num_frames = num_frames
        self.spatial_size = spatial_size
        
        self.spatial_block = DiTBlock(dim, heads, dim_head, mlp_mult, dropout)
        self.temporal_block = DiTBlock(dim, heads, dim_head, mlp_mult, dropout)
        
        self.spatial_rope = RotaryEmbedding(dim_head, max_seq_len=spatial_size*spatial_size)
        self.temporal_rope = RotaryEmbedding(dim_head, max_seq_len=num_frames)
        
    def forward(self, x, c):
        b, n, d = x.shape
        t = self.num_frames
        hw = self.spatial_size * self.spatial_size
        
        x_spatial = rearrange(x, 'b (t hw) d -> (b t) hw d', t=t, hw=hw)
        cos_s, sin_s = self.spatial_rope(hw, x.device)
        x_spatial = self.spatial_block(x_spatial, c.repeat_interleave(t, dim=0), cos_s, sin_s)
        x = rearrange(x_spatial, '(b t) hw d -> b (t hw) d', b=b, t=t)
        
        x_temporal = rearrange(x, 'b (t hw) d -> (b hw) t d', t=t, hw=hw)
        cos_t, sin_t = self.temporal_rope(t, x.device)
        x_temporal = self.temporal_block(x_temporal, c.repeat_interleave(hw, dim=0), cos_t, sin_t)
        x = rearrange(x_temporal, '(b hw) t d -> b (t hw) d', b=b, hw=hw)
        
        return x

class FinalLayer(nn.Module):
    def __init__(self, dim, patch_size, out_channels):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.linear = nn.Linear(dim, patch_size**3 * out_channels)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 2)
        )
        
    def forward(self, x, c):
        shift, scale = self.adaLN(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)

class VideoDiTV2(nn.Module):
    def __init__(
        self,
        in_channels=4,
        T=16,
        H=32,
        W=32,
        patch_size=2,
        dim=1024,
        depth=16,
        heads=16,
        dim_head=64,
        mlp_mult=4,
        dropout=0.0,
        num_classes=0  # 0 = unconditional
    ):
        super().__init__()
        self.in_channels = in_channels
        self.T = T
        self.H = H
        self.W = W
        self.patch_size = patch_size
        self.dim = dim
        self.num_classes = num_classes
        
        self.t_patches = T // patch_size
        self.h_patches = H // patch_size
        self.w_patches = W // patch_size
        self.num_patches = self.t_patches * self.h_patches * self.w_patches
        
        patch_dim = in_channels * (patch_size ** 3)
        self.patch_embed = nn.Linear(patch_dim, dim)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.t_embedder = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )
        
        if num_classes > 0:
            self.class_embed = nn.Embedding(num_classes + 1, dim)  # +1 for null class (CFG)
        else:
            self.class_embed = None
        
        self.blocks = nn.ModuleList([
            SpatioTemporalDiTBlock(
                dim, heads, dim_head, mlp_mult, dropout,
                num_frames=self.t_patches,
                spatial_size=self.h_patches
            )
            for _ in range(depth)
        ])
        
        self.final_layer = FinalLayer(dim, patch_size, in_channels)
        
        self.initialize_weights()
        
    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.apply(_basic_init)
        
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)
        nn.init.zeros_(self.final_layer.adaLN[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN[-1].bias)
        
        for block in self.blocks:
            nn.init.zeros_(block.spatial_block.adaLN[-1].weight)
            nn.init.zeros_(block.spatial_block.adaLN[-1].bias)
            nn.init.zeros_(block.temporal_block.adaLN[-1].weight)
            nn.init.zeros_(block.temporal_block.adaLN[-1].bias)
    
    def get_timestep_embedding(self, t, dim):
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb
    
    def patchify(self, x):
        p = self.patch_size
        x = rearrange(x, 'b c (t pt) (h ph) (w pw) -> b (t h w) (pt ph pw c)',
                     pt=p, ph=p, pw=p, t=self.t_patches, h=self.h_patches, w=self.w_patches)
        return x
    
    def unpatchify(self, x):
        p = self.patch_size
        x = rearrange(x, 'b (t h w) (pt ph pw c) -> b c (t pt) (h ph) (w pw)',
                     t=self.t_patches, h=self.h_patches, w=self.w_patches,
                     pt=p, ph=p, pw=p, c=self.in_channels)
        return x
    
    def forward(self, x, t, y=None):
        x = self.patchify(x)
        x = self.patch_embed(x)
        x = x + self.pos_embed
        
        t_emb = self.get_timestep_embedding(t, self.dim)
        c = self.t_embedder(t_emb)
        
        if self.class_embed is not None and y is not None:
            c = c + self.class_embed(y)
        
        for block in self.blocks:
            x = block(x, c)
        
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale=4.0):
        half = x.shape[0] // 2
        combined = self.forward(x, t, y)
        cond, uncond = combined[:half], combined[half:]
        return uncond + cfg_scale * (cond - uncond)

