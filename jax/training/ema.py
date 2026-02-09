"""Exponential Moving Average (EMA) for model parameters."""

import jax
import jax.numpy as jnp


def ema_update(ema_params, model_params, step, decay=0.9999, warmup_steps=2000):
    """Update EMA parameters.

    During warmup (step < warmup_steps), EMA params are simply copied from model.
    After warmup, applies exponential moving average with ramp-up decay.

    Args:
        ema_params: Current EMA parameter pytree.
        model_params: Current model parameter pytree.
        step: Current training step (int).
        decay: Target EMA decay rate.
        warmup_steps: Number of steps before EMA kicks in.

    Returns:
        Updated EMA parameter pytree.
    """
    # Ramp up decay: min(target_decay, (1 + step) / (10 + step))
    effective_decay = jnp.minimum(decay, (1.0 + step) / (10.0 + step))
    # During warmup, just copy model params (decay = 0)
    effective_decay = jnp.where(step < warmup_steps, 0.0, effective_decay)

    return jax.tree.map(
        lambda ema, param: effective_decay * ema + (1.0 - effective_decay) * param,
        ema_params, model_params,
    )
