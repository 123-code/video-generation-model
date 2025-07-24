import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import lpips
from einops import rearrange
import os
import cv2
import numpy as np
from model.vae import VAE3D
import torch.cuda.amp as amp

# Custom Video Dataset for HMDB51
class VideoDataset(Dataset):
    def __init__(self, data_dir, num_frames=16, frame_size=(128, 128), transform=None):
        self.data_dir = data_dir
        self.video_files = []
        for action_folder in os.listdir(data_dir):
            action_path = os.path.join(data_dir, action_folder)
            if os.path.isdir(action_path):
                self.video_files.extend([os.path.join(action_path, f) for f in os.listdir(action_path) if f.endswith('.avi')])
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.transform = transform

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < self.num_frames:
            indices = np.arange(frame_count)
            indices = np.repeat(indices, self.num_frames // frame_count + 1)[:self.num_frames]
        else:
            indices = np.linspace(0, frame_count - 1, self.num_frames, dtype=int)

        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.frame_size)
                frame = frame / 255.0
                frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1)
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
        cap.release()

        video = torch.stack(frames, dim=1)  # Shape: [C, T, H, W]
        return video

# Training Configuration (unchanged from previous script)
model_config = {
    'down_channels': [32, 64, 128],
    'mid_channels': [128, 128],
    'down_sample': [True, False],
    'num_down_layers': 1,
    'num_mid_layers': 1,
    'num_up_layers': 1,
    'attn_down': [False, False],
    'z_channels': 8,
    'norm_channels': 4,
    'num_heads': 1
}

# Hyperparameters (unchanged)
batch_size = 2
num_epochs = 50
learning_rate = 1e-4
gradient_accumulation_steps = 4
kl_weight = 0.01
lpips_weight = 0.1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize Model, Loss, and Optimizer
vae = VAE3D(im_channels=3, model_config=model_config).to(device)
lpips_loss = lpips.LPIPS(net='vgg').to(device)
mse_loss = nn.MSELoss()
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
scaler = amp.GradScaler()

def train():
    # Training Loop (unchanged)
    vae.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, videos in enumerate(dataloader):
            videos = videos.to(device)  # Shape: [B, C, T, H, W]
            
            optimizer.zero_grad()
            with amp.autocast():
                recon_videos, encoder_output = vae(videos)
                mean, logvar = torch.chunk(encoder_output, 2, dim=1)
                
                # Compute Losses
                recon_loss = mse_loss(recon_videos, videos)
                lpips_val = lpips_loss(recon_videos, videos).mean()
                kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
                loss = recon_loss + lpips_weight * lpips_val + kl_weight * kl_loss

            # Backward Pass with Gradient Accumulation
            scaler.scale(loss / gradient_accumulation_steps).backward()
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], '
                      f'Loss: {loss.item():.4f}, Recon: {recon_loss.item():.4f}, '
                      f'LPIPS: {lpips_val.item():.4f}, KL: {kl_loss.item():.4f}')

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

        # Save Checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(vae.state_dict(), f'vae_checkpoint_epoch_{epoch+1}.pth')

    # Save Final Model
    torch.save(vae.state_dict(), 'vae_final.pth')
    print("Training completed and model saved.")

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = VideoDataset(
        data_dir='../data/hmdb51_org',  
        num_frames=16,
        frame_size=(128, 128),
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    train()
