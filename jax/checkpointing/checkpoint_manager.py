"""Orbax checkpoint manager for async saves."""

import os
import jax
import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp


def create_checkpoint_manager(checkpoint_dir: str, max_to_keep: int = 5):
    """Create an Orbax CheckpointManager.

    Args:
        checkpoint_dir: Directory to store checkpoints.
        max_to_keep: Maximum number of checkpoints to retain.

    Returns:
        ocp.CheckpointManager instance.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    options = ocp.CheckpointManagerOptions(
        max_to_keep=max_to_keep,
        save_interval_steps=1,
    )
    manager = ocp.CheckpointManager(
        os.path.abspath(checkpoint_dir),
        options=options,
    )
    return manager


def save_checkpoint(manager, model, optimizer, ema_params,
                    epoch, global_step, rng):
    """Save a checkpoint with all training state.

    Args:
        manager: Orbax CheckpointManager.
        model: VideoDiTV2 model.
        optimizer: nnx.Optimizer.
        ema_params: EMA parameter pytree.
        epoch: Current epoch.
        global_step: Current global training step.
        rng: Current PRNG key.
    """
    model_params = nnx.state(model, nnx.Param)
    opt_state = nnx.state(optimizer)
    state = {
        "model_params": model_params,
        "opt_state": opt_state,
        "ema_params": ema_params,
        "epoch": jnp.int32(epoch),
        "global_step": jnp.int32(global_step),
        "rng": rng,
    }
    manager.save(global_step, args=ocp.args.StandardSave(state))
    print(f"  Checkpoint saved at step {global_step}")


def restore_checkpoint(manager, model, optimizer, ema_params):
    """Restore from the latest checkpoint if available.

    Args:
        manager: Orbax CheckpointManager.
        model: VideoDiTV2 model (for param structure).
        optimizer: nnx.Optimizer (for state structure).
        ema_params: EMA params (for structure).

    Returns:
        Tuple of (model, optimizer, ema_params, epoch, global_step, rng) or None.
    """
    latest_step = manager.latest_step()
    if latest_step is None:
        return None

    print(f"Restoring checkpoint from step {latest_step}...")

    # Build abstract restore target from current state
    model_params = nnx.state(model, nnx.Param)
    opt_state = nnx.state(optimizer)
    abstract_state = {
        "model_params": model_params,
        "opt_state": opt_state,
        "ema_params": ema_params,
        "epoch": jnp.int32(0),
        "global_step": jnp.int32(0),
        "rng": jax.random.key(0),
    }

    restored = manager.restore(
        latest_step,
        args=ocp.args.StandardRestore(abstract_state),
    )

    # Update model and optimizer with restored state
    nnx.update(model, restored["model_params"])
    nnx.update(optimizer, restored["opt_state"])

    return (
        model,
        optimizer,
        restored["ema_params"],
        int(restored["epoch"]),
        int(restored["global_step"]),
        restored["rng"],
    )
