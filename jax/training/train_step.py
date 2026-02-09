"""JIT-compiled training step."""

import jax
import jax.numpy as jnp
from flax import nnx

from training.ema import ema_update


def compute_loss(model, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
                 x_start, t, y, noise):
    """Compute MSE loss between predicted and actual noise.

    Args:
        model: VideoDiTV2 model (nnx.Module).
        sqrt_alphas_cumprod: Schedule array (T,).
        sqrt_one_minus_alphas_cumprod: Schedule array (T,).
        x_start: Clean latents (B, C, T, H, W) in bf16.
        t: Timesteps (B,) int32.
        y: Class labels (B,) int32.
        noise: Pre-sampled noise, same shape as x_start, in bf16.

    Returns:
        Scalar loss in float32.
    """
    # Inline q_sample to avoid passing full schedule dataclass
    batch_size = t.shape[0]
    ndim = len(x_start.shape)
    sqrt_alpha = sqrt_alphas_cumprod[t].astype(jnp.float32).reshape(
        batch_size, *((1,) * (ndim - 1)))
    sqrt_one_minus_alpha = sqrt_one_minus_alphas_cumprod[t].astype(jnp.float32).reshape(
        batch_size, *((1,) * (ndim - 1)))
    x_t = sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

    noise_pred = model(x_t, t, y)
    # Compute loss in float32 for stability
    loss = jnp.mean((noise_pred.astype(jnp.float32) - noise.astype(jnp.float32)) ** 2)
    return loss


@nnx.jit
def train_step(model, optimizer, ema_params,
               sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
               timesteps, x_start, labels, rng_key, step,
               cfg_dropout_prob, ema_decay, ema_warmup_steps):
    """Single JIT-compiled training step.

    Uses nnx.Optimizer which handles param extraction and optax updates internally.
    Schedule arrays are passed as plain JAX arrays (not a dataclass) for traceability.

    Args:
        model: VideoDiTV2 model (nnx.Module).
        optimizer: nnx.Optimizer wrapping optax.
        ema_params: EMA parameter pytree (nnx.State).
        sqrt_alphas_cumprod: Schedule array (T,) float32.
        sqrt_one_minus_alphas_cumprod: Schedule array (T,) float32.
        timesteps: Total number of diffusion timesteps (int).
        x_start: Batch of clean latents (B, C, T, H, W).
        labels: Class labels (B,).
        rng_key: PRNG key.
        step: Current global step (int scalar).
        cfg_dropout_prob: Probability of dropping labels for CFG.
        ema_decay: EMA decay rate.
        ema_warmup_steps: EMA warmup steps.

    Returns:
        loss: Scalar training loss.
        ema_params: Updated EMA params.
        new_rng: New PRNG key for next step.
    """
    batch_size = x_start.shape[0]

    # Split RNG for timesteps, noise, and CFG dropout
    rng_key, t_rng, noise_rng, cfg_rng = jax.random.split(rng_key, 4)

    # Sample random timesteps
    t = jax.random.randint(t_rng, (batch_size,), 0, timesteps)

    # Sample noise
    noise = jax.random.normal(noise_rng, x_start.shape)

    # Cast inputs to bf16 for TPU efficiency
    x_start_bf16 = x_start.astype(jnp.bfloat16)
    noise_bf16 = noise.astype(jnp.bfloat16)

    # CFG dropout: per-sample masking
    cfg_mask = jax.random.uniform(cfg_rng, (batch_size,)) < cfg_dropout_prob
    num_classes = model.num_classes
    y = jnp.where(cfg_mask, num_classes, labels)

    # Compute loss and gradients, then update via nnx.Optimizer
    loss, grads = nnx.value_and_grad(compute_loss)(
        model, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
        x_start_bf16, t, y, noise_bf16,
    )
    optimizer.update(grads)

    # Update EMA
    model_params = nnx.state(model, nnx.Param)
    ema_params = ema_update(ema_params, model_params, step, ema_decay, ema_warmup_steps)

    return loss, ema_params, rng_key
