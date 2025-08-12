import torch 
import torch.nn as nn 
from einops import rearrange
import numpy as np
from diffusers.models.embeddings import PatchEmbed


class SpatioTemporalTokenizer(nn.Module):
    def __init__(self, in_channels, latent_size, patch_size, hidden_size):
        super().__init__()
        
        # Use the pre-built PatchEmbed from the diffusers library
        self.patch_embedder = PatchEmbed(
            in_channels=in_channels,
            patch_size=patch_size,
            embed_dim=hidden_size
        )
        
        self.num_patches = (latent_size // patch_size) ** 2
        grid_size = latent_size // patch_size
        
        pos_embed_np = get_2d_sincos_pos_embed(hidden_size, grid_size)
        self.register_buffer('pos_embed', torch.from_numpy(pos_embed_np).float())

    def forward(self, x):
  
        tokens = self.patch_embedder(x)
        final_tokens = tokens + self.pos_embed
        return final_tokens


class PatchEmbedder2D(nn.Module):
    def __init__(self,in_channels,patch_size,hidden_size):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size

        self.projection = nn.Conv2d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self,x:torch.Tensor)->torch.Tensor:
        #cambio de figura de tensor 
        # (b,c,h,w) -> (b,d,h/p,w/p)
        x = self.projection(x)
        tokens = rearrange(x,'b d h w -> b(h w) d')
        return tokens
    

def get_1d_sincos_pos_embed_from_grid(embed_dim,pos):
    assert embed_dim % 2 == 0 
    #crear un array omega de longitud (embed_dim//2)
    omega = np.arange(embed_dim//2,dtype=np.float64)
    #dividir omega por embed_dim/2
    omega /= embed_dim/2
    #frecuencias 1/10000**omega
    omega = 1./10000**omega 
    #reshape pos para que sea un vector
    pos = pos.reshape(-1)
    #multiplicar pos por omega
    out = np.einsum('m,d->md',pos,omega)
    #seno y coseno de out
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    #concatenar seno y coseno
    emb = np.concatenate([emb_sin,emb_cos],axis=1)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim,grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb

def get_2d_sincos_pos_embed(embed_dim,grid_size):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed[np.newaxis, ...]

def get_1d_sincos_temp_embed(embed_dim,length):
    pos = np.arange(0,length,dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim,pos)
    return pos_embed[np.newaxis,...]








    
#test
if __name__ == "__main__":
    B = 4   
    C = 4   
    H = 32  
    W = 32  
    P = 2  
    D = 768 

    dummy_latents = torch.randn(B, C, H, W)
    print(f"Input shape: {dummy_latents.shape}")

    patch_embedder = PatchEmbedder2D(C,P,D)
    tokens = patch_embedder(dummy_latents)
    print(f"Output shape: {tokens.shape}")


