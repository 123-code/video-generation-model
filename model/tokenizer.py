import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

# --- Import the necessary component from diffusers ---
from diffusers.models.embeddings import PatchEmbed

# --- Positional Embedding Helper Functions (These are your custom functions, they are perfect) ---

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    Creates a 1D sinusoidal positional embedding from a grid of positions.
    
    Args:
        embed_dim (int): The embedding dimension. Must be even.
        pos (np.ndarray): A numpy array of positions to be encoded.
        
    Returns:
        np.ndarray: The positional embeddings of shape (num_positions, embed_dim).
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb

def get_2d_sincos_pos_embed(embed_dim, grid_size, add_batch_dim=True):
    """
    Creates a 2D sinusoidal positional embedding.
    
    Args:
        embed_dim (int): The embedding dimension (hidden_size).
        grid_size (int): The height and width of the patch grid.
        add_batch_dim (bool): If True, adds a leading dimension for easy broadcasting.
        
    Returns:
        np.ndarray: A (1, num_patches, embed_dim) or (num_patches, embed_dim) positional embedding.
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # Note: meshgrid order is (x, y)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])

    # Get the embedding from the grid
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    
    if add_batch_dim:
        pos_embed = pos_embed[np.newaxis, ...]
        
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    # Use half of dimensions to encode grid_h (y-axis, from grid[1])
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    # Use the other half to encode grid_w (x-axis, from grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb

# --- The Complete Tokenizer Module using Diffusers' PatchEmbed ---

class SpatioTemporalTokenizer(nn.Module):
    """
    Performs the complete tokenization for a batch of frames, including
    patch embedding and adding the 2D spatial positional embedding.
    """
    def __init__(self, in_channels, latent_size, patch_size, hidden_size):
        super().__init__()
        
        # 1. Use the pre-built PatchEmbed from the diffusers library.
        # It handles the Conv2d projection and reshaping internally.
        self.patch_embedder = PatchEmbed(
            in_channels=in_channels,
            patch_size=patch_size,
            embed_dim=hidden_size
        )
        
        self.num_patches = (latent_size // patch_size) ** 2
        grid_size = latent_size // patch_size
        
        # 2. Create and register the positional embedding as a non-trainable buffer.
        pos_embed_np = get_2d_sincos_pos_embed(hidden_size, grid_size)
        self.register_buffer('pos_embed', torch.from_numpy(pos_embed_np).float())

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Channels, Height, Width).
                              For video, this will be (B*F, C, H, W).
        
        Returns:
            torch.Tensor: Spatially aware tokens of shape (Batch, Num_Patches, Hidden_Size).
        """
        # The diffusers PatchEmbed takes a 4D tensor and returns tokens.
        # It performs the Conv2d and reshape in one step.
        tokens = self.patch_embedder(x)
        
        # Add the spatial positional embedding.
        # `self.pos_embed` of shape (1, T, D) is broadcasted across the batch dimension.
        final_tokens = tokens + self.pos_embed
        
        return final_tokens

# --- Test ---
if __name__ == "__main__":
    B_F = 8   # Combined batch and frame dimension (e.g., 2 videos * 4 frames)
    C = 4     # Latent channels
    H = 32    # Latent height
    W = 32    # Latent width
    P = 2     # Patch size
    D = 768   # Hidden size

    dummy_latents_batch = torch.randn(B_F, C, H, W)
    print(f"Input shape (a batch of frames): {dummy_latents_batch.shape}")

    # Initialize the complete tokenizer
    tokenizer = SpatioTemporalTokenizer(
        in_channels=C,
        latent_size=H,
        patch_size=P,
        hidden_size=D
    )
    
    print("\nUsing `diffusers.models.embeddings.PatchEmbed` for tokenization.")
    
    # Get the final tokens
    final_tokens = tokenizer(dummy_latents_batch)
    
    expected_num_patches = (H // P) * (W // P)
    print(f"\nFinal tokens shape: {final_tokens.shape}")
    print(f"Expected final tokens shape: ({B_F}, {expected_num_patches}, {D})")
    
    assert final_tokens.shape == (B_F, expected_num_patches, D)
    print("\nTokenizer test passed!")