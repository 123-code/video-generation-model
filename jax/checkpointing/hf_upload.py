"""Upload checkpoints to HuggingFace Hub."""

import os
from huggingface_hub import HfApi


def upload_checkpoint_to_hf(checkpoint_dir: str, repo_id: str,
                            step: int, epoch: int):
    """Upload the latest Orbax checkpoint folder to HuggingFace Hub.

    Args:
        checkpoint_dir: Local Orbax checkpoint directory.
        repo_id: HuggingFace repository ID (e.g. "user/model-name").
        step: Current training step (used to find the checkpoint subdirectory).
        epoch: Current epoch (for the commit message).
    """
    api = HfApi()

    # Orbax saves checkpoints in subdirectories named by step
    ckpt_path = os.path.join(os.path.abspath(checkpoint_dir), str(step))

    if not os.path.exists(ckpt_path):
        print(f"  Checkpoint path not found: {ckpt_path}")
        return

    api.upload_folder(
        folder_path=ckpt_path,
        path_in_repo=f"checkpoint-step{step}-epoch{epoch}",
        repo_id=repo_id,
        commit_message=f"Step {step}, epoch {epoch}",
    )
    print(f"  Uploaded checkpoint to HF: {repo_id}")
