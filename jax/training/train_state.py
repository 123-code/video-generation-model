"""Training state and optimizer setup."""

import optax
from flax import nnx


def create_optimizer(model, lr: float, min_lr: float, total_steps: int,
                     weight_decay: float = 0.01, grad_clip_norm: float = 1.0):
    """Create an nnx.Optimizer wrapping optax chain: grad clipping + AdamW + cosine decay.

    Args:
        model: VideoDiTV2 nnx.Module (needed by nnx.Optimizer).
        lr: Peak learning rate.
        min_lr: Minimum learning rate at end of cosine decay.
        total_steps: Total number of training steps for cosine decay.
        weight_decay: AdamW weight decay.
        grad_clip_norm: Maximum gradient norm for clipping.

    Returns:
        nnx.Optimizer wrapping the optax chain.
    """
    schedule_fn = optax.cosine_decay_schedule(
        init_value=lr,
        decay_steps=total_steps,
        alpha=min_lr / lr,
    )

    tx = optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adamw(learning_rate=schedule_fn, weight_decay=weight_decay),
    )

    return nnx.Optimizer(model, tx)
