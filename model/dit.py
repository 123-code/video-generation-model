import os 
import torch 
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
import numpy as np 
from torch.optim import Adam 
from torch.cuda.amp import autocast,GradScaler 

class LatentDataset(Dataset):
    def __init__(self,latent_dir):
        self.latent_files = [os.path.join(latent_dir,f) for f in os.listdir(latent_dir) if f.endswith('.pt')]
        self.latent_files.sort()

    def __len__(self):
        return len(self.latent_files)
    
    def __getitem__(self,idx):
        latent = torch.load(self.latent_files[idx])
        if latent.dim() == 5 and latent.shape[0] == 1:
            latent = latent.squeeze(0)
        elif latent.dim() == 6:
            latent = latent.squeeze(0)
        # Downsample to (8, 8, 8) for memory efficiency
        import torch.nn.functional as F
        latent = latent.unsqueeze(0)  # (1, C, T, H, W)
        latent = F.interpolate(latent, size=(8, 8, 8), mode='trilinear', align_corners=False)
        latent = latent.squeeze(0)  # (C, 8, 8, 8)
        return latent
    
class DiTBlock(nn.Module):
    def __init__(self,channels,num_heads=4,dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(4, channels)
        self.attn = nn.MultiheadAttention(channels,num_heads,dropout=dropout,batch_first=True)
        self.norm2 = nn.GroupNorm(4, channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels,channels*4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels*4,channels),
        )
        self.dropout = nn.Dropout(dropout)


    def forward(self,x):
        residual = x 
        x = self.norm1(x)
        batch_size,channels,t,h,w = x.shape
        x = x.reshape(batch_size,channels,t*h*w).transpose(1,2)
        x,_ = self.attn(x,x,x)
        x = x.transpose(1,2).reshape(batch_size,channels,t,h,w)
        x = self.dropout(x + residual)

        residual = x
        x = self.norm2(x)
        x = x.reshape(batch_size,channels,t*h*w).transpose(1,2)
        x = self.mlp(x)
        x = x.transpose(1,2).reshape(batch_size,channels,t,h,w)
        x = self.dropout(x + residual)
        return x 
    
class DiT(nn.Module):
    def __init__(self,channels=8,depth=4,num_heads=4,dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([DiTBlock(channels,num_heads,dropout) for _ in range(depth)])
        self.final_norm = nn.GroupNorm(4,channels)

    def forward(self,x,t):
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return x 
    
def get_noise_schedule(timesteps = 100):
    betas = torch.linspace(0.0001, 0.02, timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas,dim=0)
    return alphas_cumprod

def train_dit(latent_dir ,output_dir,num_epochs=100,batch_size=1,learning_rate=1e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir,exist_ok=True)
    dataset = LatentDataset(latent_dir)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
    model = DiT(channels=8,depth=4).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()
    alphas_cumprod = get_noise_schedule().to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            batch_size = batch.shape[0]
            t = torch.randint(0,len(alphas_cumprod),(batch_size,),device=device).long()
            noise = torch.randn_like(batch)
            noisy_batch = torch.sqrt(alphas_cumprod[t])[:,None,None,None,None] * batch + torch.sqrt(1.0 - alphas_cumprod[t])[:,None,None,None,None] * noise
            with autocast():
                predicted_noise = model(noisy_batch,t)
                loss = nn.MSELoss()(predicted_noise,noise)
            # NaN/Inf check
            if torch.isnan(loss) or torch.isinf(loss):
                print('Warning: NaN or Inf loss encountered, skipping optimizer step.')
                optimizer.zero_grad()
                continue
            scaler.scale(loss).backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(),os.path.join(output_dir,f"model_epoch_{epoch+1}.pth"))

    torch.save(model.state_dict(), os.path.join(output_dir, 'dit_final.pth'))
    print("Training completed and model saved.")

if __name__ == "__main__":
    latent_dir = "latents"
    output_dir = "dit_checkpoints"
    train_dit(latent_dir,output_dir)



