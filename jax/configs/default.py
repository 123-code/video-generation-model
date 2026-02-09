"""Default configuration for VideoDiTV2 JAX training."""

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # --- Model ---
    config.model = ml_collections.ConfigDict()
    config.model.in_channels = 4
    config.model.T = 16
    config.model.H = 32
    config.model.W = 32
    config.model.patch_size = 2
    config.model.dim = 1024
    config.model.depth = 16
    config.model.heads = 16
    config.model.dim_head = 64
    config.model.mlp_mult = 4
    config.model.dropout = 0.0
    config.model.num_classes = 5  # 0 = unconditional

    # --- Diffusion ---
    config.diffusion = ml_collections.ConfigDict()
    config.diffusion.timesteps = 1000
    config.diffusion.beta_schedule = "cosine"

    # --- Training ---
    config.training = ml_collections.ConfigDict()
    config.training.epochs = 2000
    config.training.batch_size = 32  # Global batch size across all TPU chips
    config.training.lr = 1e-4
    config.training.min_lr = 1e-6
    config.training.weight_decay = 0.01
    config.training.grad_clip_norm = 1.0
    config.training.cfg_dropout_prob = 0.1
    config.training.seed = 42

    # --- EMA ---
    config.ema = ml_collections.ConfigDict()
    config.ema.decay = 0.9999
    config.ema.warmup_steps = 2000

    # --- Data ---
    config.data = ml_collections.ConfigDict()
    config.data.latent_dir = "latents_npy"
    config.data.num_workers = 4

    # --- Checkpointing ---
    config.checkpoint = ml_collections.ConfigDict()
    config.checkpoint.save_every_steps = 500
    config.checkpoint.save_every_epochs = 5
    config.checkpoint.sample_every_epochs = 50
    config.checkpoint.max_to_keep = 5
    config.checkpoint.checkpoint_dir = "checkpoints"
    config.checkpoint.sample_dir = "samples"
    config.checkpoint.hf_repo_id = "Jnaranjo/video-dit-jax"
    config.checkpoint.hf_checkpoint_repo_id = "Jnaranjo/video-generation-epoch90-checkpoint"

    # --- Sampling ---
    config.sampling = ml_collections.ConfigDict()
    config.sampling.ddim_steps = 50
    config.sampling.cfg_scale = 4.0
    config.sampling.eta = 0.0

    return config
