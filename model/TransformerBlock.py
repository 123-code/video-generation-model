import torch
import torch.nn as nn
from einops import rearrange

# Local modules (same folder)
from patch_embed import SpatioTemporalTokenizer, get_1d_sincos_temp_embed
from conditioning import TimestepEmbedder


class AdaLayerNormZero(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        # Produce shift/scale/gate for MSA and MLP: 6 * hidden_size
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 6, bias=True),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, T, D), cond: (B, D)
        x_norm = self.norm(x)
        m = self.modulation(cond)  # (B, 6*D)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(m, 6, dim=-1)
        return x_norm, shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp


class ConditionedTransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.ada_norm = AdaLayerNormZero(hidden_size)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.dropout_attn = nn.Dropout(dropout)

        mlp_hidden = int(hidden_size * mlp_ratio)
        self.ln_mlp = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self.dropout_mlp = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # AdaLN-Zero for attention
        x_norm, shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ada_norm(x, cond)
        x_msa_in = x_norm * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        attn_out, _ = self.attn(x_msa_in, x_msa_in, x_msa_in, need_weights=False)
        x = x + self.dropout_attn(gate_msa[:, None, :] * attn_out)

        # AdaLN-Zero for MLP
        x_mlp_in = self.ln_mlp(x)
        x_mlp_in = x_mlp_in * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        mlp_out = self.mlp(x_mlp_in)
        x = x + self.dropout_mlp(gate_mlp[:, None, :] * mlp_out)
        return x

class LatteTransformer(nn.Module):
    def __init__(self, in_channels, latent_size, patch_size, hidden_size, num_heads, depth=14, num_frames=16, time_embed_dim: int = 256):
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
        self.time_embed_dim = time_embed_dim
        
        # 1. Tokenizer
        self.tokenizer = SpatioTemporalTokenizer(in_channels, latent_size, patch_size, hidden_size)
        self.num_patches = self.tokenizer.num_patches

        # 2. Time conditioning
        self.t_embedder = TimestepEmbedder(hidden_size=self.hidden_size, frequency_embedding_size=self.time_embed_dim)

        # 3. Transformer Blocks (conditioned)
        self.spatial_blocks = nn.ModuleList([
            ConditionedTransformerBlock(hidden_size=hidden_size, num_heads=num_heads)
            for _ in range(depth)
        ])
        self.temporal_blocks = nn.ModuleList([
            ConditionedTransformerBlock(hidden_size=hidden_size, num_heads=num_heads)
            for _ in range(depth)
        ])

        # 4. Temporal Positional Embedding
        temp_embed_np = get_1d_sincos_temp_embed(hidden_size, self.num_frames)
        self.register_buffer('temp_embed', torch.from_numpy(temp_embed_np).float())
        
        # 5. Final Layer (Unpatching)
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

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        B, C, F, H, W = x.shape
        
        # Reshape for per-frame tokenization and apply tokenizer
        x = rearrange(x, 'b c f h w -> (b f) c h w')
        x = self.tokenizer(x)

        # Time embedding
        t_emb = self.t_embedder(t)  # (B, D)
        # Expand conditioning per block usage
        cond_spatial = t_emb.repeat_interleave(self.num_frames, dim=0)  # (B*F, D)

        # Main Transformer Loop
        for spatial_block, temporal_block in zip(self.spatial_blocks, self.temporal_blocks):
            # Spatial processing
            x = spatial_block(x, cond=cond_spatial)
            
            # Reshape for temporal processing
            x = rearrange(x, '(b f) t d -> (b t) f d', b=B)
            # For temporal, repeat cond across patches (tokens)
            cond_temporal = t_emb.repeat_interleave(self.num_patches, dim=0)  # (B*T, D)
            
            # Add temporal embedding only on the first pass
            if not self._temp_embed_added:
                x = x + self.temp_embed
                self._temp_embed_added = True
            
            # Temporal processing
            x = temporal_block(x, cond=cond_temporal)
            
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
    dummy_timesteps = torch.randint(0, 1000, (B,))
    
    # --- Run the forward pass ---
    print(f"Input shape: {dummy_video_latents.shape}")
    output = model(dummy_video_latents, dummy_timesteps)
    print(f"Output shape: {output.shape}")
    
    # --- Verify the output shape matches the input shape ---
    assert dummy_video_latents.shape == output.shape
    print("\nModel forward pass successful! Output shape matches input shape.")
