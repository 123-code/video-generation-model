import torch
import torch.nn as nn 
from einops import rearrange
import math
class PatchEmbed(nn.Module):
    def __init__(self,latent,in_channels = 3):

        super().__init__()
        self.latent = latent
        self.patch_size = 4
        self.embed_dim = 768
        self.in_channels = in_channels
        self.patch_dim = self.patch_size * self.patch_size * self.in_channels
        self.projection = nn.Linear(self.patch_dim,self.embed_dim)

    def get_pos_embed(self,embed_dim,seq_len):
        #crear tensor de logitud seq len, agregar una dimension, dtype=float
        position = torch.arange(seq_len).unsqueeze(1).float()
        #decay factors para frecuencias
        div_term = torch.exp(torch.arange(0,embed_dim,2).float() * -(math.log(10000.0) / embed_dim))
        #tensor de zeros
        pe = torch.zeros(seq_len,embed_dim)
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        return pe

    


    
    def forward(self,latent):
        
        if latent.dim() == 4:
            B,C,H,W = latent.shape
            T = 1
            latent = latent.unsqueeze(2) # agrega una dimension 
        else:
            B,C,T,H,W = latent.shape

# crea patches 
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        num_patches = num_patches_h * num_patches_w

        patches = rearrange(latent,'b c t (h p1) (w p2) -> (b t) (h w) (p1 p2 c)',p1=self.patch_size,p2=self.patch_size,h=num_patches_h,w=num_patches_w)
        emb = self.projection(patches)

        pos_emb = self.get_pos_embed(emb.shape[-1],emb.shape[0]*emb.shape[1])
        emb += pos_emb.unsqueeze(0)
        return emb



if __name__ == "__main__":
    #carga un tensor de forma: [t,c,h,w]
    latent = torch.randn(1, 3, 32, 32)
    if latent.shape[0] == 16:
        #cambia la forma a [c,t,h,w]
        latent = latent.permute(1,0,2,3)
    patch_embed = PatchEmbed(latent)
    print(patch_embed(latent).shape)
