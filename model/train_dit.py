import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from TransformerBlock import LatteTransformer
from diffusion import DiffusionProcess
from latent_dataset import LatentDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
LATENT_SIZE = 32
IN_CHANNELS = 32
NUM_FRAMES = 16
PATCH_SIZE = 2
HIDDEN_SIZE = 768
NUM_HEADS = 12
DEPTH = 14

TIMESTEPS = 1000
EPOCHS = 100
BATCH_SIZE = 16 
LEARNING_RATE = 1e-4

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LatteTransformer(
        in_channels=IN_CHANNELS,
        latent_size=LATENT_SIZE,
        patch_size=PATCH_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        depth=DEPTH,
        num_frames=NUM_FRAMES,
    ).to(device)

    diffusion = DiffusionProcess(timesteps=TIMESTEPS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    print("Loading latent dataset...")
   
    latent_data_dir = "latents" 
    dataset = LatentDataset(latent_dir=latent_data_dir)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training on {len(dataset)} pre-computed latents.")

    for epoch in range(EPOCHS):
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for step,clean_latents in enumerate(progress_bar):
            clean_latents = clean_latents.to(device)
            t = torch.randint(0, TIMESTEPS, (clean_latents.shape[0],), device=device).long()
            loss = diffusion.training_loss(model,clean_latents,t)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            progress_bar.set_postfix(loss=loss.item())
        print(f"Epoch {epoch+1}/{EPOCHS} completed")
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"latte_checkpoint_epoch_{epoch+1}.pth")
    print("Training complete!")
    
if __name__ == "__main__":
    main()
