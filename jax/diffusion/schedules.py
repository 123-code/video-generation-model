"""Diffusion schedules and core diffusion functions (pure JAX)."""

import jax.numpy as jnp
import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DiffusionSchedule:
    """Precomputed diffusion schedule arrays (all float64 for precision)."""
    timesteps: int
    betas: jnp.ndarray
    alphas: jnp.ndarray
    alphas_cumprod: jnp.ndarray
    alphas_cumprod_prev: jnp.ndarray
    sqrt_alphas_cumprod: jnp.ndarray
    sqrt_one_minus_alphas_cumprod: jnp.ndarray
    log_one_minus_alphas_cumprod: jnp.ndarray
    sqrt_recip_alphas_cumprod: jnp.ndarray
    sqrt_recipm1_alphas_cumprod: jnp.ndarray
    posterior_variance: jnp.ndarray
    posterior_log_variance_clipped: jnp.ndarray
    posterior_mean_coef1: jnp.ndarray
    posterior_mean_coef2: jnp.ndarray


def _build_schedule(betas: jnp.ndarray, timesteps: int) -> DiffusionSchedule:
    """Build a DiffusionSchedule from a beta array."""
    betas = betas.astype(jnp.float64)
    alphas = 1.0 - betas
    alphas_cumprod = jnp.cumprod(alphas)
    alphas_cumprod_prev = jnp.concatenate([jnp.ones(1, dtype=jnp.float64), alphas_cumprod[:-1]])

    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    return DiffusionSchedule(
        timesteps=timesteps,
        betas=betas,
        alphas=alphas,
        alphas_cumprod=alphas_cumprod,
        alphas_cumprod_prev=alphas_cumprod_prev,
        sqrt_alphas_cumprod=jnp.sqrt(alphas_cumprod),
        sqrt_one_minus_alphas_cumprod=jnp.sqrt(1.0 - alphas_cumprod),
        log_one_minus_alphas_cumprod=jnp.log(1.0 - alphas_cumprod),
        sqrt_recip_alphas_cumprod=jnp.sqrt(1.0 / alphas_cumprod),
        sqrt_recipm1_alphas_cumprod=jnp.sqrt(1.0 / alphas_cumprod - 1),
        posterior_variance=posterior_variance,
        posterior_log_variance_clipped=jnp.log(jnp.clip(posterior_variance, min=1e-20)),
        posterior_mean_coef1=betas * jnp.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        posterior_mean_coef2=(1.0 - alphas_cumprod_prev) * jnp.sqrt(alphas) / (1.0 - alphas_cumprod),
    )


def create_cosine_schedule(timesteps: int = 1000, s: float = 0.008) -> DiffusionSchedule:
    """Create a cosine beta schedule (Nichol & Dhariwal, 2021)."""
    steps = timesteps + 1
    x = jnp.linspace(0, timesteps, steps)
    alphas_cumprod = jnp.cos((x / timesteps + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = jnp.clip(betas, 0.0001, 0.9999)
    return _build_schedule(betas, timesteps)


def create_linear_schedule(timesteps: int = 1000) -> DiffusionSchedule:
    """Create a linear beta schedule."""
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    betas = jnp.linspace(beta_start, beta_end, timesteps, dtype=jnp.float64)
    return _build_schedule(betas, timesteps)


def _extract(a, t, x_shape):
    """Extract values from schedule array a at indices t, reshaped for broadcasting.

    Args:
        a: 1D schedule array of shape (timesteps,).
        t: Batch of timestep indices, shape (B,).
        x_shape: Shape of the tensor to broadcast against.

    Returns:
        Extracted values reshaped to (B, 1, 1, ...) for broadcasting with x_shape.
    """
    out = a[t].astype(jnp.float32)
    # Reshape to (B, 1, 1, ...) with as many trailing 1s as needed
    return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))


def q_sample(schedule: DiffusionSchedule, x_start, t, noise):
    """Forward diffusion: add noise to x_start at timestep t.

    x_t = sqrt(alpha_cumprod_t) * x_start + sqrt(1 - alpha_cumprod_t) * noise
    """
    sqrt_alpha = _extract(schedule.sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alpha = _extract(schedule.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise


def predict_start_from_noise(schedule: DiffusionSchedule, x_t, t, noise):
    """Predict x_0 from x_t and predicted noise."""
    return (
        _extract(schedule.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
        _extract(schedule.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
    )


def q_posterior(schedule: DiffusionSchedule, x_start, x_t, t):
    """Compute the posterior q(x_{t-1} | x_t, x_0)."""
    posterior_mean = (
        _extract(schedule.posterior_mean_coef1, t, x_t.shape) * x_start +
        _extract(schedule.posterior_mean_coef2, t, x_t.shape) * x_t
    )
    posterior_variance = _extract(schedule.posterior_variance, t, x_t.shape)
    posterior_log_variance = _extract(schedule.posterior_log_variance_clipped, t, x_t.shape)
    return posterior_mean, posterior_variance, posterior_log_variance


def training_loss(schedule: DiffusionSchedule, model, model_params, x_start, t, y=None, noise=None, rng=None):
    """Compute MSE training loss between predicted and actual noise.

    This is a pure function suitable for use with jax.value_and_grad.

    Args:
        schedule: Precomputed diffusion schedule.
        model: The model (nnx.Module).
        model_params: Not used directly - model is called as a closure.
        x_start: Clean latents (B, C, T, H, W).
        t: Timesteps (B,).
        y: Class labels (B,), optional.
        noise: Pre-sampled noise, same shape as x_start.
        rng: Not used (noise must be pre-sampled).

    Returns:
        Scalar MSE loss.
    """
    x_t = q_sample(schedule, x_start, t, noise)
    noise_pred = model(x_t, t, y)
    loss = jnp.mean((noise_pred - noise) ** 2)
    return loss
