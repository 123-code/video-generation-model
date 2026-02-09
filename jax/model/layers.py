"""Core model layers: RMSNorm, RoPE, Attention, FeedForward."""

import jax
import jax.numpy as jnp
from flax import nnx
import math


def modulate(x, shift, scale):
    """Adaptive layer norm modulation: x * (1 + scale) + shift.

    x: (B, N, D), shift/scale: (B, D) -> broadcast over N.
    """
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


class RMSNorm(nnx.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        self.eps = eps
        self.weight = nnx.Param(jnp.ones((dim,)))

    def __call__(self, x):
        return x * jnp.rsqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.eps) * self.weight.value


# --- Rotary Position Embedding (pure functions) ---

def build_rope_cache(seq_len: int, dim: int, dtype=jnp.float32):
    """Precompute cos/sin tables for RoPE.

    Args:
        seq_len: Maximum sequence length.
        dim: Head dimension (must be even).

    Returns:
        cos, sin: Arrays of shape (seq_len, dim).
    """
    inv_freq = 1.0 / (10000 ** (jnp.arange(0, dim, 2, dtype=dtype) / dim))
    t = jnp.arange(seq_len, dtype=dtype)
    freqs = jnp.outer(t, inv_freq)  # (seq_len, dim/2)
    emb = jnp.concatenate([freqs, freqs], axis=-1)  # (seq_len, dim)
    return jnp.cos(emb), jnp.sin(emb)


def rotate_half(x):
    """Rotate half of the hidden dims of x."""
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embeddings to query and key tensors.

    q, k: (B, H, N, D)
    cos, sin: (N, D) -> broadcast to (1, 1, N, D)
    """
    cos = cos[None, None, :, :]  # (1, 1, N, D)
    sin = sin[None, None, :, :]
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


class Attention(nnx.Module):
    """Multi-head attention with optional RoPE."""

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64,
                 dropout: float = 0.0, *, rngs: nnx.Rngs):
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head

        self.to_qkv = nnx.Linear(dim, inner_dim * 3, use_bias=False, rngs=rngs)
        self.to_out = nnx.Linear(inner_dim, dim, rngs=rngs)
        self.dropout_rate = dropout

    def __call__(self, x, rope_cos=None, rope_sin=None):
        b, n, d = x.shape
        qkv = self.to_qkv(x)  # (B, N, 3*inner_dim)
        q, k, v = jnp.split(qkv, 3, axis=-1)  # each (B, N, inner_dim)

        # Reshape to multi-head: (B, N, H*D) -> (B, H, N, D)
        q = q.reshape(b, n, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        k = k.reshape(b, n, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        v = v.reshape(b, n, self.heads, self.dim_head).transpose(0, 2, 1, 3)

        if rope_cos is not None and rope_sin is not None:
            q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)

        # Scaled dot-product attention
        out = jax.nn.dot_product_attention(q, k, v)  # (B, H, N, D)

        # Merge heads: (B, H, N, D) -> (B, N, H*D)
        out = out.transpose(0, 2, 1, 3).reshape(b, n, self.heads * self.dim_head)
        return self.to_out(out)


class FeedForward(nnx.Module):
    """Two-layer MLP with GELU activation."""

    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0, *, rngs: nnx.Rngs):
        inner_dim = dim * mult
        self.linear1 = nnx.Linear(dim, inner_dim, rngs=rngs)
        self.linear2 = nnx.Linear(inner_dim, dim, rngs=rngs)
        self.dropout_rate = dropout

    def __call__(self, x):
        x = self.linear1(x)
        x = jax.nn.gelu(x)
        x = self.linear2(x)
        return x
