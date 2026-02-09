"""DDPM and DDIM sampling loops using jax.lax for TPU efficiency."""

import jax
import jax.numpy as jnp

from diffusion.schedules import DiffusionSchedule, _extract, predict_start_from_noise, q_posterior


def _p_mean_variance(schedule, model, x, t_batch, y=None, cfg_scale=1.0, clip_denoised=True):
    """Compute predicted mean and variance for a single denoising step.

    Args:
        schedule: DiffusionSchedule.
        model: VideoDiTV2 model.
        x: Current noisy sample (B, C, T, H, W).
        t_batch: Timestep tensor (B,) of identical values.
        y: Class labels (B,).
        cfg_scale: Classifier-free guidance scale.
        clip_denoised: Whether to clip predicted x_0.

    Returns:
        model_mean, posterior_variance, posterior_log_variance, x_start
    """
    if cfg_scale > 1.0 and y is not None:
        # Double the batch for CFG
        x_double = jnp.concatenate([x, x], axis=0)
        t_double = jnp.concatenate([t_batch, t_batch], axis=0)
        num_classes = model.num_classes
        y_null = jnp.full_like(y, num_classes)
        y_double = jnp.concatenate([y, y_null], axis=0)

        noise_pred = model(x_double, t_double, y_double)
        noise_cond, noise_uncond = jnp.split(noise_pred, 2, axis=0)
        noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
    else:
        noise_pred = model(x, t_batch, y)

    x_start = predict_start_from_noise(schedule, x, t_batch, noise_pred)

    if clip_denoised:
        x_start = jnp.clip(x_start, -1.0, 1.0)

    model_mean, posterior_variance, posterior_log_variance = q_posterior(
        schedule, x_start, x, t_batch
    )
    return model_mean, posterior_variance, posterior_log_variance, x_start


def ddpm_sample(schedule, model, shape, y=None, cfg_scale=1.0, rng=None):
    """Full DDPM sampling loop (1000 steps).

    Uses jax.lax.fori_loop for TPU efficiency.

    Args:
        schedule: DiffusionSchedule.
        model: VideoDiTV2 model.
        shape: Output shape (B, C, T, H, W).
        y: Class labels (B,).
        cfg_scale: CFG scale.
        rng: JAX PRNG key.

    Returns:
        Denoised samples (B, C, T, H, W).
    """
    b = shape[0]
    rng, init_rng = jax.random.split(rng)
    x = jax.random.normal(init_rng, shape)

    # We iterate from timestep T-1 down to 0
    # fori_loop iterates i from 0 to T-1, so we use t = T-1-i
    def body_fn(i, carry):
        x, rng = carry
        t = schedule.timesteps - 1 - i
        t_batch = jnp.full((b,), t, dtype=jnp.int32)

        model_mean, _, model_log_variance, _ = _p_mean_variance(
            schedule, model, x, t_batch, y, cfg_scale
        )

        rng, noise_rng = jax.random.split(rng)
        noise = jax.random.normal(noise_rng, x.shape)
        # No noise at t=0
        nonzero_mask = (t > 0).astype(jnp.float32)
        x = model_mean + nonzero_mask * jnp.exp(0.5 * model_log_variance) * noise
        return x, rng

    x, _ = jax.lax.fori_loop(0, schedule.timesteps, body_fn, (x, rng))
    return x


def ddim_sample(schedule, model, shape, y=None, cfg_scale=1.0,
                ddim_steps=50, eta=0.0, rng=None):
    """DDIM accelerated sampling.

    Args:
        schedule: DiffusionSchedule.
        model: VideoDiTV2 model.
        shape: Output shape (B, C, T, H, W).
        y: Class labels (B,).
        cfg_scale: CFG scale.
        ddim_steps: Number of DDIM steps.
        eta: Stochasticity parameter (0 = deterministic DDIM).
        rng: JAX PRNG key.

    Returns:
        Denoised samples (B, C, T, H, W).
    """
    b = shape[0]
    rng, init_rng = jax.random.split(rng)
    x = jax.random.normal(init_rng, shape)

    # Build time schedule: evenly spaced from -1 to timesteps-1, reversed
    times = jnp.linspace(-1, schedule.timesteps - 1, ddim_steps + 1).astype(jnp.int32)
    # Reverse to go from high timestep to low
    times = times[::-1]
    # time_pairs[i] = (times[i], times[i+1])

    def body_fn(i, carry):
        x, rng = carry
        time_now = times[i]
        time_next = times[i + 1]

        t_batch = jnp.full((b,), time_now, dtype=jnp.int32)

        # Predict noise with optional CFG
        if cfg_scale > 1.0 and y is not None:
            x_double = jnp.concatenate([x, x], axis=0)
            t_double = jnp.concatenate([t_batch, t_batch], axis=0)
            num_classes = model.num_classes
            y_null = jnp.full_like(y, num_classes)
            y_double = jnp.concatenate([y, y_null], axis=0)
            noise_pred = model(x_double, t_double, y_double)
            noise_cond, noise_uncond = jnp.split(noise_pred, 2, axis=0)
            noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
        else:
            noise_pred = model(x, t_batch, y)

        x_start = predict_start_from_noise(schedule, x, t_batch, noise_pred)
        x_start = jnp.clip(x_start, -1.0, 1.0)

        # If time_next < 0, we're done -> return x_start
        # Otherwise, compute DDIM update
        alpha = _extract(schedule.alphas_cumprod, t_batch, x.shape)
        t_next_batch = jnp.full((b,), jnp.maximum(time_next, 0), dtype=jnp.int32)
        alpha_next = _extract(schedule.alphas_cumprod, t_next_batch, x.shape)

        sigma = eta * jnp.sqrt((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha))
        c = jnp.sqrt(1 - alpha_next - sigma ** 2)

        rng, noise_rng = jax.random.split(rng)
        noise = jax.random.normal(noise_rng, x.shape)
        noise = noise * (eta > 0).astype(jnp.float32)  # Zero noise if eta=0

        x_ddim = x_start * jnp.sqrt(alpha_next) + c * noise_pred + sigma * noise

        # Use x_start when time_next < 0, otherwise x_ddim
        x = jnp.where(time_next < 0, x_start, x_ddim)
        return x, rng

    x, _ = jax.lax.fori_loop(0, ddim_steps, body_fn, (x, rng))
    return x
