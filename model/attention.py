import torch.nn as nn
import math 
from einops import rearrange
from patch_embed import PatchEmbed

class SpatialAttention(nn.Module):
    def __init__(self,dim,heads=8):
        super().__init__()
        self.heads = heads
        self.scale = math.sqrt(dim//heads)
        self.to_qkv = nn.Linear(dim,dim*3,bias=False)
        self.to_out = nn.Linear(dim,dim)
    def forward(self,x):
        #batch size, number of patches, dimension of the patch
        b,n,d = x.shape
        #projection to qkv
        qkv = self.to_qkv(x).chunk(3,dim=-1)
        q,k,v = map(lambda t:rearrange(t,'b n (h d) -> bh n d',h=self.heads),qkv)
        dots = torch.matmul(q,k.transpose(-1,-2)) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.matmul(attn,v)
        out = rearrange(out,'b h n d -> b n (h d)')
        out = self.to_out(out) + x

        return out

class TemporalAttention(nn.Module):
    def __init__(self,dim,heads=8):
        self.heads = heads
        self.scale = math.sqrt(dim//heads)
        self.to_qkv = nn.Linear(dim,dim*3,bias=False)
        self.to_out = nn.Linear(dim,dim)
    def forward(self,x):
        b,t,n,d = x.shape
        qkv = self.to_qkv(x).chunk(3,dim=-1)
        q,k,v = map(lambda t:rearrange(t,'b t n (h d) -> b h t n d',h=self.heads),qkv)
        dots = torch.matmul(q,k.transpose(-1,-2)) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.matmul(attn,v)
        out = rearrange(out,'b h t n d -> b t n (h d)')
        out = self.to_out(out) + x
        return out


