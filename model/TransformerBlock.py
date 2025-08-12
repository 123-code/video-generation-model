import torch
import torch.nn as nn
from diffusers.models.attention import BasicTransformerBlock
from einops import rearrange

# Corrected relative import path assuming patch_embed.py is in a 'model' subdir
from patch_embed import SpatioTemporalTokenizer, get_1d_sincos_temp_embed

class LatteTransformer(nn.Module):
    def __init__(self, in_channels, latent_size, patch_size, hidden_size, num_heads, depth=14, num_frames=16):
        super().__init__()
        # Store key dimensions
        self.in_channels = in_channels
        self.out_channels = in_channels # Output channels should match input
        self.latent_size = latent_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.depth = depth
        self.num_frames = num_frames
        
        # 1. Tokenizer
        self.tokenizer = SpatioTemporalTokenizer(in_channels, latent_size, patch_size, hidden_size)
        self.num_patches = self.tokenizer.num_patches

        # 2. Transformer Blocks
        attention_head_dim = hidden_size // num_heads
        self.spatial_blocks = nn.ModuleList([
            BasicTransformerBlock(
                dim=hidden_size,
                num_attention_heads=num_heads,
                attention_head_dim=attention_head_dim,
            ) for _ in range(depth)
        ])
        self.temporal_blocks = nn.ModuleList([
            BasicTransformerBlock(
                dim=hidden_size,
                num_attention_heads=num_heads,
                attention_head_dim=attention_head_dim,
            ) for _ in range(depth)
        ])

        # 3. Temporal Positional Embedding
        temp_embed_np = get_1d_sincos_temp_embed(hidden_size, self.num_frames)
        self.register_buffer('temp_embed', torch.from_numpy(temp_embed_np).float())
        
        # 4. Final Layer (Unpatching)
        self.norm_out = nn.LayerNorm(hidden_size)
        self.proj_out = nn.Linear(hidden_size, patch_size * patch_size * self.out_channels)
        
        # Internal flag for adding temporal embedding once
        self._temp_embed_added = False

    def unpatchify(self, x):
        """
        Reshapes tokens back into a video latent.
        x shape: (B*F, T, P*P*C) -> (B, C, F, H, W)
        """
        B_F, T, _ = x.shape
        # Use stored values to ensure correctness
        B = B_F // self.num_frames
        C = self.out_channels
        P = self.patch_size
        H_W_patches = self.latent_size // P # Height/Width of the patch grid
        
        # Reshape sequence of patches back into a grid of patches, then into an image
        x = x.reshape(B_F, H_W_patches, H_W_patches, P, P, C)
        x = rearrange(x, '(b f) h w p1 p2 c -> b c f (h p1) (w p2)', b=B, f=self.num_frames)
        return x

    def forward(self, x):
        B, C, F, H, W = x.shape
        
        # Reshape for per-frame tokenization and apply tokenizer
        x = rearrange(x, 'b c f h w -> (b f) c h w')
        x = self.tokenizer(x)

        # Main Transformer Loop
        for spatial_block, temporal_block in zip(self.spatial_blocks, self.temporal_blocks):
            # Spatial processing
            x = spatial_block(x)
            
            # Reshape for temporal processing
            x = rearrange(x, '(b f) t d -> (b t) f d', b=B)
            
            # Add temporal embedding only on the first pass
            if not self._temp_embed_added:
                x = x + self.temp_embed
                self._temp_embed_added = True
            
            # Temporal processing
            x = temporal_block(x)
            
            # Reshape back for spatial processing
            x = rearrange(x, '(b t) f d -> (b f) t d', b=B)
        
        # Reset the flag after the loop for the next forward call
        self._temp_embed_added = False
        
        # Final projection and unpatching
        x = self.norm_out(x)
        x = self.proj_out(x)
        x = self.unpatchify(x)
        
        return x

# --- Test ---
if __name__ == "__main__":
    # Config
    B, F, C, H, W = 2, 16, 4, 32, 32
    P, D, H_heads = 2, 768, 12
    
    # Create the complete model
    model = LatteTransformer(
        in_channels=C,
        latent_size=H,
        patch_size=P,
        hidden_size=D,
        num_heads=H_heads,
        num_frames=F
    )
    
    # Create a dummy input tensor representing a batch of video latents
    dummy_video_latents = torch.randn(B, C, F, H, W)
    
    # --- Run the forward pass ---
    print(f"Input shape: {dummy_video_latents.shape}")
    output = model(dummy_video_latents)
    print(f"Output shape: {output.shape}")
    
    # --- Verify the output shape matches the input shape ---
    assert dummy_video_latents.shape == output.shape
    print("\nModel forward pass successful! Output shape matches input shape.")
