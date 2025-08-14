import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.cuda.amp as amp

FILE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from TransformerBlock import LatteTransformer
from diffusion import DiffusionProcess
from latent_dataset import LatentDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_PATCH_SIZE = 2
DEFAULT_HIDDEN_SIZE = 768
DEFAULT_NUM_HEADS = 12
DEFAULT_DEPTH = 14

DEFAULT_TIMESTEPS = 1000
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 4
DEFAULT_LEARNING_RATE = 1e-4

def main():
    parser = argparse.ArgumentParser(description="Train DiT on pre-computed video latents")
    parser.add_argument('--latent_dir', type=str, default=os.path.join(PROJECT_ROOT, 'latents'))
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument('--timesteps', type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--patch_size', type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument('--hidden_size', type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument('--num_heads', type=int, default=DEFAULT_NUM_HEADS)
    parser.add_argument('--depth', type=int, default=DEFAULT_DEPTH)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--out_dir', type=str, default=PROJECT_ROOT)
    parser.add_argument('--amp', action='store_true', help='Enable mixed precision training (autocast+GradScaler)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading latent dataset from:", args.latent_dir)
    dataset = LatentDataset(latent_dir=args.latent_dir)

    # Inspect one sample to infer channel/frame/spatial dims
    sample = dataset[0]
    if sample.ndim != 4:
        raise RuntimeError(f"Expected latent sample to have shape [C,F,H,W], got {tuple(sample.shape)}")
    in_channels = sample.shape[0]
    num_frames = sample.shape[1]
    latent_size = sample.shape[2]

    print(f"Inferred latent dims -> C:{in_channels} F:{num_frames} H:W:{latent_size}x{latent_size}")

    model = LatteTransformer(
        in_channels=in_channels,
        latent_size=latent_size,
        patch_size=args.patch_size,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        depth=args.depth,
        num_frames=num_frames,
    ).to(device)

    diffusion = DiffusionProcess(timesteps=args.timesteps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = amp.GradScaler(enabled=args.amp and torch.cuda.is_available())

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    LATENT_SCALE_FACTOR = 2.874741

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training on {len(dataset)} pre-computed latents. Batch size: {args.batch_size}")
    print(f"Applying latent scaling factor: {LATENT_SCALE_FACTOR}")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    for epoch in range(args.epochs):
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, clean_latents in enumerate(progress_bar):
            clean_latents = clean_latents.to(device)
            clean_latents = clean_latents / LATENT_SCALE_FACTOR
            t = torch.randint(0, args.timesteps, (clean_latents.shape[0],), device=device).long()
            with amp.autocast(enabled=args.amp and torch.cuda.is_available()):
                loss = diffusion.training_loss(model, clean_latents, t)
            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        print(f"Epoch {epoch+1}/{args.epochs} completed")
        if (epoch + 1) % args.save_every == 0:
            os.makedirs(args.out_dir, exist_ok=True)
            ckpt_path = os.path.join(args.out_dir, f"latte_checkpoint_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    print("Training complete!")


if __name__ == "__main__":
    main()
