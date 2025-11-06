import torch
import torch.nn as nn
import math 
from einops import rearrange

class SpatialAttention(nn.Module):
    def __init__(self,dim,heads=8):
        super().__init__()
        self.heads = heads
        self.scale = 1.0 / math.sqrt(dim//heads)
        self.to_qkv = nn.Linear(dim,dim*3,bias=False)
        self.to_out = nn.Linear(dim,dim)
    def forward(self,x):
        b,n,d = x.shape
        qkv = self.to_qkv(x).chunk(3,dim=-1)
        q,k,v = map(lambda t:rearrange(t,'b n (h d) -> b h n d',h=self.heads),qkv)
        dots = torch.matmul(q,k.transpose(-1,-2)) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.matmul(attn,v)
        out = rearrange(out,'b h n d -> b n (h d)')
        return self.to_out(out)

class TemporalAttention(nn.Module):
    def __init__(self,dim,heads=8):
        super().__init__()
        self.heads = heads
        self.scale = 1.0 / math.sqrt(dim//heads)
        self.to_qkv = nn.Linear(dim,dim*3,bias=False)
        self.to_out = nn.Linear(dim,dim)
    def forward(self,x):
        b,t,d = x.shape
        qkv = self.to_qkv(x).chunk(3,dim=-1)
        q,k,v = map(lambda t:rearrange(t,'b t (h d) -> b h t d',h=self.heads),qkv)
        dots = torch.matmul(q,k.transpose(-1,-2)) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.matmul(attn,v)
        out = rearrange(out,'b h t d -> b t (h d)')
        return self.to_out(out)
