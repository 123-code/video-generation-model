import torch 
import torch.nn as nn 
from einops import rearrange
import numpy as np

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


