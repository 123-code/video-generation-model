"""VideoDiTV2: Diffusion Transformer for class-conditional video generation."""

import jax
import jax.numpy as jnp
from flax import nnx
import math

from model.layers import (
    RMSNorm, Attention, FeedForward, modulate,
    build_rope_cache, apply_rotary_pos_emb,
)


class DiTBlock(nnx.Module):
    """Diffusion Transformer block with adaptive layer norm (adaLN)."""

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64,
                 mlp_mult: int = 4, dropout: float = 0.0, *, rngs: nnx.Rngs):
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, heads, dim_head, dropout, rngs=rngs)
        self.norm2 = RMSNorm(dim)
        self.ff = FeedForward(dim, mlp_mult, dropout, rngs=rngs)

        # adaLN: SiLU + Linear -> 6 * dim (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        self.adaLN_linear = nnx.Linear(dim, dim * 6, rngs=rngs)

    def __call__(self, x, c, rope_cos=None, rope_sin=None):
        ada_out = self.adaLN_linear(jax.nn.silu(c))  # (B, 6*dim)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(ada_out, 6, axis=-1)

        # Attention branch with adaLN modulation
        x = x + gate_msa[:, None, :] * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), rope_cos, rope_sin
        )
        # FFN branch with adaLN modulation
        x = x + gate_mlp[:, None, :] * self.ff(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class SpatioTemporalDiTBlock(nnx.Module):
    """Spatiotemporal DiT block: spatial attention over patches within each frame,
    then temporal attention over frames at each spatial position."""

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64,
                 mlp_mult: int = 4, dropout: float = 0.0,
                 num_frames: int = 8, spatial_size: int = 16, *, rngs: nnx.Rngs):
        self.num_frames = num_frames
        self.spatial_size = spatial_size

        self.spatial_block = DiTBlock(dim, heads, dim_head, mlp_mult, dropout, rngs=rngs)
        self.temporal_block = DiTBlock(dim, heads, dim_head, mlp_mult, dropout, rngs=rngs)

        # Precompute RoPE tables (stored as nnx.Variable for proper state tracking)
        hw = spatial_size * spatial_size
        s_cos, s_sin = build_rope_cache(hw, dim_head)
        t_cos, t_sin = build_rope_cache(num_frames, dim_head)
        self.spatial_rope_cos = nnx.Variable(s_cos)
        self.spatial_rope_sin = nnx.Variable(s_sin)
        self.temporal_rope_cos = nnx.Variable(t_cos)
        self.temporal_rope_sin = nnx.Variable(t_sin)

    def __call__(self, x, c):
        b, n, d = x.shape
        t = self.num_frames
        hw = self.spatial_size * self.spatial_size

        # --- Spatial attention: attend within each frame ---
        # (B, T*HW, D) -> (B*T, HW, D)
        x_spatial = x.reshape(b, t, hw, d).reshape(b * t, hw, d)
        # Repeat conditioning for each frame: (B, D) -> (B*T, D)
        c_spatial = jnp.repeat(c, t, axis=0)
        x_spatial = self.spatial_block(x_spatial, c_spatial,
                                       self.spatial_rope_cos.value, self.spatial_rope_sin.value)
        x = x_spatial.reshape(b, t, hw, d).reshape(b, t * hw, d)

        # --- Temporal attention: attend across frames at each spatial position ---
        # (B, T*HW, D) -> (B*HW, T, D)
        x_temporal = x.reshape(b, t, hw, d).transpose(0, 2, 1, 3).reshape(b * hw, t, d)
        # Repeat conditioning for each spatial position: (B, D) -> (B*HW, D)
        c_temporal = jnp.repeat(c, hw, axis=0)
        x_temporal = self.temporal_block(x_temporal, c_temporal,
                                         self.temporal_rope_cos.value, self.temporal_rope_sin.value)
        x = x_temporal.reshape(b, hw, t, d).transpose(0, 2, 1, 3).reshape(b, t * hw, d)

        return x


class FinalLayer(nnx.Module):
    """Final projection layer with adaLN modulation."""

    def __init__(self, dim: int, patch_size: int, out_channels: int, *, rngs: nnx.Rngs):
        self.norm = RMSNorm(dim)
        self.linear = nnx.Linear(dim, patch_size ** 3 * out_channels, rngs=rngs)
        self.adaLN_linear = nnx.Linear(dim, dim * 2, rngs=rngs)

    def __call__(self, x, c):
        ada_out = self.adaLN_linear(jax.nn.silu(c))
        shift, scale = jnp.split(ada_out, 2, axis=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


class VideoDiTV2(nnx.Module):
    """Video Diffusion Transformer V2 for class-conditional video generation.

    Operates on VAE latent tensors of shape (B, C, T, H, W).
    """

    def __init__(
        self,
        in_channels: int = 4,
        T: int = 16,
        H: int = 32,
        W: int = 32,
        patch_size: int = 2,
        dim: int = 1024,
        depth: int = 16,
        heads: int = 16,
        dim_head: int = 64,
        mlp_mult: int = 4,
        dropout: float = 0.0,
        num_classes: int = 0,
        *,
        rngs: nnx.Rngs,
    ):
        self.in_channels = in_channels
        self.T = T
        self.H = H
        self.W = W
        self.patch_size = patch_size
        self.dim = dim
        self.num_classes = num_classes

        self.t_patches = T // patch_size
        self.h_patches = H // patch_size
        self.w_patches = W // patch_size
        self.num_patches = self.t_patches * self.h_patches * self.w_patches

        patch_dim = in_channels * (patch_size ** 3)
        self.patch_embed = nnx.Linear(patch_dim, dim, rngs=rngs)

        # Learned positional embedding
        self.pos_embed = nnx.Param(
            jax.random.truncated_normal(rngs.params(), -2.0, 2.0,
                                        shape=(1, self.num_patches, dim)) * 0.02
        )

        # Timestep embedder: Linear -> SiLU -> Linear
        self.t_embed_linear1 = nnx.Linear(dim, dim * 4, rngs=rngs)
        self.t_embed_linear2 = nnx.Linear(dim * 4, dim, rngs=rngs)

        # Class embedding (optional)
        if num_classes > 0:
            self.class_embed = nnx.Embed(num_classes + 1, dim, rngs=rngs)  # +1 for null class
        else:
            self.class_embed = None

        # Transformer blocks
        self.blocks = [
            SpatioTemporalDiTBlock(
                dim, heads, dim_head, mlp_mult, dropout,
                num_frames=self.t_patches,
                spatial_size=self.h_patches,
                rngs=rngs,
            )
            for _ in range(depth)
        ]

        # Final projection
        self.final_layer = FinalLayer(dim, patch_size, in_channels, rngs=rngs)

        # Apply weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Zero-init final layer and all adaLN output projections."""
        # Zero-init final layer linear
        self.final_layer.linear.kernel.value = jnp.zeros_like(self.final_layer.linear.kernel.value)
        self.final_layer.linear.bias.value = jnp.zeros_like(self.final_layer.linear.bias.value)

        # Zero-init final layer adaLN
        self.final_layer.adaLN_linear.kernel.value = jnp.zeros_like(
            self.final_layer.adaLN_linear.kernel.value
        )
        self.final_layer.adaLN_linear.bias.value = jnp.zeros_like(
            self.final_layer.adaLN_linear.bias.value
        )

        # Zero-init all adaLN projections in transformer blocks
        for block in self.blocks:
            for sub_block in [block.spatial_block, block.temporal_block]:
                sub_block.adaLN_linear.kernel.value = jnp.zeros_like(
                    sub_block.adaLN_linear.kernel.value
                )
                sub_block.adaLN_linear.bias.value = jnp.zeros_like(
                    sub_block.adaLN_linear.bias.value
                )

    def get_timestep_embedding(self, t, dim):
        """Sinusoidal timestep embedding.

        Args:
            t: Integer timesteps of shape (B,).
            dim: Embedding dimension.

        Returns:
            Embedding of shape (B, dim).
        """
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)
        emb = t.astype(jnp.float32)[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        if dim % 2 == 1:
            emb = jnp.pad(emb, ((0, 0), (0, 1)))
        return emb

    def patchify(self, x):
        """Convert (B, C, T, H, W) to (B, num_patches, patch_dim).

        Reshapes the 5D tensor into 3D patches using pure reshape/transpose.
        """
        b, c, t, h, w = x.shape
        p = self.patch_size
        tp, hp, wp = self.t_patches, self.h_patches, self.w_patches

        # (B, C, T, H, W) -> (B, C, tp, p, hp, p, wp, p)
        x = x.reshape(b, c, tp, p, hp, p, wp, p)
        # -> (B, tp, hp, wp, p, p, p, C)
        x = x.transpose(0, 2, 4, 6, 3, 5, 7, 1)
        # -> (B, num_patches, patch_dim)
        x = x.reshape(b, tp * hp * wp, p * p * p * c)
        return x

    def unpatchify(self, x):
        """Convert (B, num_patches, patch_dim) back to (B, C, T, H, W)."""
        b = x.shape[0]
        p = self.patch_size
        c = self.in_channels
        tp, hp, wp = self.t_patches, self.h_patches, self.w_patches

        # (B, num_patches, patch_dim) -> (B, tp, hp, wp, p, p, p, C)
        x = x.reshape(b, tp, hp, wp, p, p, p, c)
        # -> (B, C, tp, p, hp, p, wp, p)
        x = x.transpose(0, 7, 1, 4, 2, 5, 3, 6)
        # -> (B, C, T, H, W)
        x = x.reshape(b, c, tp * p, hp * p, wp * p)
        return x

    def __call__(self, x, t, y=None):
        """Forward pass.

        Args:
            x: Noisy latents (B, C, T, H, W).
            t: Timesteps (B,) integer tensor.
            y: Class labels (B,) integer tensor, optional.

        Returns:
            Predicted noise (B, C, T, H, W).
        """
        # Patchify and embed
        x = self.patchify(x)
        x = self.patch_embed(x)
        x = x + self.pos_embed.value

        # Timestep conditioning
        t_emb = self.get_timestep_embedding(t, self.dim)
        c = self.t_embed_linear2(jax.nn.silu(self.t_embed_linear1(t_emb)))

        # Class conditioning
        if self.class_embed is not None and y is not None:
            c = c + self.class_embed(y)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, c)

        # Final projection and unpatchify
        x = self.final_layer(x, c)
        x = self.unpatchify(x)

        return x

    def forward_with_cfg(self, x, t, y, cfg_scale=4.0):
        """Forward pass with Classifier-Free Guidance.

        Expects x to be doubled: first half conditional, second half unconditional.
        """
        half = x.shape[0] // 2
        combined = self(x, t, y)
        cond, uncond = combined[:half], combined[half:]
        return uncond + cfg_scale * (cond - uncond)
