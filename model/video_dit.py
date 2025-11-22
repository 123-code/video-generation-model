import torch 
import torch.nn as nn
from einops import rearrange
from model.patch_embed import PatchEmbed
from model.dit import VideoDiTBlock


def modulate(x,shift,scale):
    return x * (1+scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    def __init__(self,hidden_size,num_heads,mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size,elementwise_affine=False,eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size*mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_size*mlp_ratio), hidden_size),
        )
        

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6*hidden_size, bias=True)
        )
    def forward(self,x,c):
        shift_msa,scale_msa,gate_msa,shift_mlp,scale_mlp,gate_mlp = self.adaLN_modulation(c).chunk(6,dim=-1)
        modulated_msa = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulated_msa, modulated_msa, modulated_msa)[0]
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class VideoDiT(nn.Module):
    def __init__(self,in_channels=4,T=16,H=32,W=32,patch_size=2,dim=768,depth=12,heads=8):
        super().__init__()
        self.in_channels = in_channels
        self.patch_embed = PatchEmbed(in_channels=in_channels,patch_size=patch_size,embed_dim=dim)
        self.t_embed = nn.Sequential(
            nn.Linear(1,dim*4),
            nn.SiLU(),
            nn.Linear(dim*4,dim),
        )
        self.blocks = nn.ModuleList([DiTBlock(dim,heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim,in_channels * (patch_size**3))
        self.T,self.H,self.W,self.patch_size = T,H,W,patch_size
    def forward(self,latent,t):
        tokens = self.patch_embed(latent)
        t_emb = self.t_embed(t.float().unsqueeze(1))
        x = tokens

        for block in self.blocks:
            x = block(x,t_emb)
        x = self.norm(x)
        x = self.out_proj(x)

        b, n, _ = x.shape
        t_patches = self.T // self.patch_size
        h_patches = self.H // self.patch_size
        w_patches = self.W // self.patch_size

        noise_pred = rearrange(x,'b (t h w) (c pt ph pw) -> b c (t pt) (h ph) (w pw)',
                              t=t_patches,h=h_patches,w=w_patches,
                              c=self.in_channels,pt=self.patch_size,ph=self.patch_size,pw=self.patch_size)
        return noise_pred