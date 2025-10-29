import os 
import torch 
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
import numpy as np 
from torch.optim import Adam 


class DiTBlock(nn.Module):
    def __init__(self,channels,num_heads=4,dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(4,channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels,num_heads=num_heads,dropout=dropout,batch_first=True)
        self.norm2 = nn.GroupNorm(4,channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels,channels*4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels*4,channels),
        )
        self.dropout = nn.Dropout(dropout)
    def self_attention(self,x):
        qkv = self.to_qkv(x).chunk(3,dim=-1)
        q,k,v = map(lambda t:rearrange(t,'b n (h d) -> b h n d',h=num_heads),qkv)
        dots = torch.matmul(q,k.transpose(-1,-2)) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.matmul(attn,v)
        out = rearrange(out,'b h n d -> b n (h d)')
        return out
    def forward(self,x):
        residual = x
        x = self.norm1(x)
        batch_size,seq_len,embed_dim = x.shape
        x,_ = self.attn(x,x,x)
        x = self.dropout(x+ residual)
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.dropout(x+ residual)
        return x
class DiT(nn.Module):
    def __init__(self,embed_dim=768,depth=4,num_heads=4,dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([DiTBlock(embed_dim,num_heads,dropout)for _ in range(depth)])
        self.final_norm = nn.GroupNorm(4,embed_dim)
        self.time_projection = nn.Sequential(
            nn.Linear(1,embed_dim//4),
            nn.SiLU(),
            nn.Linear(embed_dim//4,embed_dim),
        )
        self.input_proj = nn.Linear(embed_dim,embed_dim)
        self.output_proj = nn.Linear(embed_dim,embed_dim)
    
    def forward(self,tokens,t):
        seq_len,embed_dim = tokens.shape
        B = t.shape[0]

        t_emb = self.time_projection(t.float().unsqueeze(-1))
        t_emb = t_emb.unsqueeze(1).repeat(1,seq_len,1)
        tokens = tokens.unsqueeze(0) + t_emb
        tokens = self.input_proj(tokens)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.final_norm(tokens)
        tokens = self.output_proj(tokens)
        tokens = tokens.squeeze(0)  
        return tokens
        