import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import lpips
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


class VideoDataset(Dataset):
    def __init__(self, data_dir, num_frames=16, frame_size=(64, 64)):
        self.samples = []
        for root, _, files in os.walk(data_dir):
            for fname in files:
                if fname.endswith('.avi'):
                    self.samples.append(os.path.join(root, fname))
        self.num_frames = num_frames
        self.frame_size = frame_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.samples[idx])
        frames = []
        while len(frames) < self.num_frames and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, self.frame_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        if len(frames) < self.num_frames:
            frames.extend([np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)] * (self.num_frames - len(frames)))
        elif len(frames) > self.num_frames:
            frames = frames[:self.num_frames]
        video = np.stack(frames, axis=0)
        video = torch.tensor(video, dtype=torch.float32).permute(3, 0, 1, 2) / 255.0
        return video


class VAE3D(nn.Module):
    def __init__(self, im_channels, model_config):
        super().__init__()
        self.down_channels = model_config['down_channels']
        self.mid_channels = model_config['mid_channels']
        self.down_sample = model_config['down_sample']
        self.num_down_layers = model_config['num_down_layers']
        self.num_mid_layers = model_config['num_mid_layers']
        self.num_up_layers = model_config['num_up_layers']
        self.attns = model_config['attn_down']
        self.z_channels = model_config['z_channels']
        self.norm_channels = model_config['norm_channels']
        self.num_heads = model_config['num_heads']

        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-1]
        assert len(self.down_sample) == len(self.down_channels) - 1
        assert len(self.attns) == len(self.down_channels) - 1
        self.up_sample = list(reversed(self.down_sample))

        self.encoder_conv_in = nn.Conv3d(im_channels, self.down_channels[0], kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.encoder_layers = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.encoder_layers.append(DownBlock3D(
                self.down_channels[i],
                self.down_channels[i + 1],
                t_emb_dim=None,
                down_sample=self.down_sample[i],
                num_heads=self.num_heads,
                num_layers=self.num_down_layers,
                attn=self.attns[i],
                norm_channels=self.norm_channels
            ))
        self.encoder_mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.encoder_mids.append(MidBlock3D(self.mid_channels[i], self.mid_channels[i + 1],
                                               t_emb_dim=None,
                                               num_heads=self.num_heads,
                                               num_layers=self.num_mid_layers,
                                               norm_channels=self.norm_channels))

        self.encoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[-1])
        self.encoder_conv_out = nn.Conv3d(self.down_channels[-1], 2 * self.z_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pre_quant_conv = nn.Conv3d(2 * self.z_channels, 2 * self.z_channels, kernel_size=1)
        self.post_quant_conv = nn.Conv3d(self.z_channels, self.z_channels, kernel_size=1)
        self.decoder_conv_in = nn.Conv3d(self.z_channels, self.mid_channels[-1], kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.decoder_mids = nn.ModuleList([])
        for i in reversed(range(1, len(self.mid_channels))):
            self.decoder_mids.append(MidBlock3D(self.mid_channels[i], self.mid_channels[i - 1],
                                               t_emb_dim=None,
                                               num_heads=self.num_heads,
                                               num_layers=self.num_mid_layers,
                                               norm_channels=self.norm_channels))

        self.decoder_layers = nn.ModuleList([])
        for i in reversed(range(1, len(self.down_channels))):
            self.decoder_layers.append(UpBlock3D(self.down_channels[i], self.down_channels[i - 1],
                                                t_emb_dim=None, up_sample=self.down_sample[i - 1],
                                                num_heads=self.num_heads,
                                                num_layers=self.num_up_layers,
                                                attn=self.attns[i - 1],
                                                norm_channels=self.norm_channels))

        self.decoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[0])
        self.decoder_conv_out = nn.Conv3d(self.down_channels[0], im_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

    def encode(self, x):
        out = self.encoder_conv_in(x)
        for idx, down in enumerate(self.encoder_layers):
            out = down(out)
        for mid in self.encoder_mids:
            out = mid(out)
        out = self.encoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.encoder_conv_out(out)
        out = self.pre_quant_conv(out)
        mean, logvar = torch.chunk(out, 2, dim=1)
        std = torch.exp(0.5 * logvar)
        sample = mean + std * torch.randn(mean.shape).to(device=x.device)
        return sample, out

    def decode(self, z):
        out = z
        out = self.post_quant_conv(out)
        out = self.decoder_conv_in(out)
        for mid in self.decoder_mids:
            out = mid(out)
        for idx, up in enumerate(self.decoder_layers):
            out = up(out)
        out = self.decoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.decoder_conv_out(out)
        return out

    def forward(self, x):
        z, encoder_output = self.encode(x)
        out = self.decode(z)
        return out, encoder_output


def get_time_embedding(time_steps, temb_dim):
    assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"
    factor = 10000 ** ((torch.arange(start=0, end=temb_dim // 2, dtype=torch.float32, device=time_steps.device) / (temb_dim // 2)))
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb

class DownBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, down_sample, num_heads, num_layers, attn, norm_channels, cross_attn=False, context_dim=None):
        super().__init__()
        self.num_layers = num_layers
        self.down_sample = down_sample
        self.attn = attn
        self.context_dim = context_dim
        self.cross_attn = cross_attn
        self.t_emb_dim = t_emb_dim
        self.resnet_conv_first = nn.ModuleList([nn.Sequential(nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels), nn.SiLU(), nn.Conv3d(in_channels if i == 0 else out_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))) for i in range(num_layers)])
        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList([nn.Sequential(nn.SiLU(), nn.Linear(self.t_emb_dim, out_channels)) for _ in range(num_layers)])
        self.resnet_conv_second = nn.ModuleList([nn.Sequential(nn.GroupNorm(norm_channels, out_channels), nn.SiLU(), nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))) for _ in range(num_layers)])
        if self.attn:
            self.attention_norms = nn.ModuleList([nn.GroupNorm(norm_channels, out_channels) for _ in range(num_layers)])
            self.attentions = nn.ModuleList([nn.MultiheadAttention(out_channels, num_heads, batch_first=True) for _ in range(num_layers)])
        if self.cross_attn:
            assert context_dim is not None
            self.cross_attention_norms = nn.ModuleList([nn.GroupNorm(norm_channels, out_channels) for _ in range(num_layers)])
            self.cross_attentions = nn.ModuleList([nn.MultiheadAttention(out_channels, num_heads, batch_first=True) for _ in range(num_layers)])
            self.context_proj = nn.ModuleList([nn.Linear(context_dim, out_channels) for _ in range(num_layers)])
        self.residual_input_conv = nn.ModuleList([nn.Conv3d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1) for i in range(num_layers)])
        self.down_sample_conv = nn.Conv3d(out_channels, out_channels, kernel_size=(2, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)) if self.down_sample else nn.Identity()

    def forward(self, x, t_emb=None, context=None):
        out = x
        for i in range(self.num_layers):
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            if self.t_emb_dim is not None:
                out = out + self.t_emb_layers[i](t_emb)[:, :, None, None, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)
            if self.attn:
                batch_size, channels, t, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, t * h * w)
                in_attn = self.attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, t, h, w)
                out = out + out_attn
            if self.cross_attn and context is not None:
                batch_size, channels, t, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, t * h * w)
                in_attn = self.cross_attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                context_proj = self.context_proj[i](context)
                out_attn, _ = self.cross_attentions[i](in_attn, context_proj, context_proj)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, t, h, w)
                out = out + out_attn
        out = self.down_sample_conv(out)
        return out

class MidBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, num_heads, num_layers, norm_channels, cross_attn=None, context_dim=None):
        super().__init__()
        self.num_layers = num_layers
        self.t_emb_dim = t_emb_dim
        self.context_dim = context_dim
        self.cross_attn = cross_attn
        self.resnet_conv_first = nn.ModuleList([nn.Sequential(nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels), nn.SiLU(), nn.Conv3d(in_channels if i == 0 else out_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))) for i in range(num_layers + 1)])
        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList([nn.Sequential(nn.SiLU(), nn.Linear(t_emb_dim, out_channels)) for _ in range(num_layers + 1)])
        self.resnet_conv_second = nn.ModuleList([nn.Sequential(nn.GroupNorm(norm_channels, out_channels), nn.SiLU(), nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))) for _ in range(num_layers + 1)])
        self.attention_norms = nn.ModuleList([nn.GroupNorm(norm_channels, out_channels) for _ in range(num_layers)])
        self.attentions = nn.ModuleList([nn.MultiheadAttention(out_channels, num_heads, batch_first=True) for _ in range(num_layers)])
        if self.cross_attn:
            assert context_dim is not None
            self.cross_attention_norms = nn.ModuleList([nn.GroupNorm(norm_channels, out_channels) for _ in range(num_layers)])
            self.cross_attentions = nn.ModuleList([nn.MultiheadAttention(out_channels, num_heads, batch_first=True) for _ in range(num_layers)])
            self.context_proj = nn.ModuleList([nn.Linear(context_dim, out_channels) for _ in range(num_layers)])
        self.residual_input_conv = nn.ModuleList([nn.Conv3d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1) for i in range(num_layers + 1)])

    def forward(self, x, t_emb=None, context=None):
        out = x
        resnet_input = out
        out = self.resnet_conv_first[0](out)
        if self.t_emb_dim is not None:
            out = out + self.t_emb_layers[0](t_emb)[:, :, None, None, None]
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](resnet_input)
        for i in range(self.num_layers):
            batch_size, channels, t, h, w = out.shape
            in_attn = out.reshape(batch_size, channels, t * h * w)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2)
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, t, h, w)
            out = out + out_attn
            if self.cross_attn and context is not None:
                batch_size, channels, t, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, t * h * w)
                in_attn = self.cross_attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                context_proj = self.context_proj[i](context)
                out_attn, _ = self.cross_attentions[i](in_attn, context_proj, context_proj)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, t, h, w)
                out = out + out_attn
            resnet_input = out
            out = self.resnet_conv_first[i + 1](out)
            if self.t_emb_dim is not None:
                out = out + self.t_emb_layers[i + 1](t_emb)[:, :, None, None, None]
            out = self.resnet_conv_second[i + 1](out)
            out = out + self.residual_input_conv[i + 1](resnet_input)
        return out

class UpBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, up_sample, num_heads, num_layers, attn, norm_channels):
        super().__init__()
        self.num_layers = num_layers
        self.up_sample = up_sample
        self.t_emb_dim = t_emb_dim
        self.attn = attn
        self.resnet_conv_first = nn.ModuleList([nn.Sequential(nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels), nn.SiLU(), nn.Conv3d(in_channels if i == 0 else out_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))) for i in range(num_layers)])
        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList([nn.Sequential(nn.SiLU(), nn.Linear(t_emb_dim, out_channels)) for _ in range(num_layers)])
        self.resnet_conv_second = nn.ModuleList([nn.Sequential(nn.GroupNorm(norm_channels, out_channels), nn.SiLU(), nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))) for _ in range(num_layers)])
        if self.attn:
            self.attention_norms = nn.ModuleList([nn.GroupNorm(norm_channels, out_channels) for _ in range(num_layers)])
            self.attentions = nn.ModuleList([nn.MultiheadAttention(out_channels, num_heads, batch_first=True) for _ in range(num_layers)])
        self.residual_input_conv = nn.ModuleList([nn.Conv3d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1) for i in range(num_layers)])
        self.up_sample_conv = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=(2, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)) if self.up_sample else nn.Identity()

    def forward(self, x, out_down=None, t_emb=None):
        x = self.up_sample_conv(x)
        if out_down is not None:
            x = torch.cat([x, out_down], dim=1)
        out = x
        for i in range(self.num_layers):
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            if self.t_emb_dim is not None:
                out = out + self.t_emb_layers[i](t_emb)[:, :, None, None, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)
            if self.attn:
                batch_size, channels, t, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, t * h * w)
                in_attn = self.attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, t, h, w)
                out = out + out_attn
        return out


def visualize_reconstructions(checkpoint_path, data_dir, output_dir, num_videos=5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


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

   
    model = VAE3D(im_channels=3, model_config=model_config).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    model.eval()


    dataset = VideoDataset(data_dir, num_frames=8, frame_size=(96, 96))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


    os.makedirs(output_dir, exist_ok=True)

  
    with torch.no_grad():
        all_originals = []
        all_recons = []
        lpips_loss = lpips.LPIPS(net='alex').to(device)
        for i, video in enumerate(dataloader):
            if i >= num_videos:
                break
            video = video.to(device)
            recon, _ = model(video)

       
            all_originals.append(video.cpu())
            all_recons.append(recon.cpu())

        
            video_frames = (video.cpu().permute(0, 2, 3, 4, 1).squeeze(0).numpy() * 255).astype(np.uint8)
            recon_frames = (recon.cpu().permute(0, 2, 3, 4, 1).squeeze(0).numpy() * 255).astype(np.uint8)

      
            orig_path = os.path.join(output_dir, f'original_video_{i}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_orig = cv2.VideoWriter(orig_path, fourcc, 10.0, (96, 96))
            for frame in video_frames:
                out_orig.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            out_orig.release()

         
            recon_path = os.path.join(output_dir, f'recon_video_{i}.mp4')
            out_recon = cv2.VideoWriter(recon_path, fourcc, 10.0, (96, 96))
            for frame in recon_frames:
                out_recon.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            out_recon.release()

            print(f"Saved original to {orig_path} and reconstruction to {recon_path}")

      
        all_originals = torch.cat(all_originals, dim=0).to(device)
        all_recons = torch.cat(all_recons, dim=0).to(device)
        mse_loss = nn.MSELoss()
        mse = mse_loss(all_recons, all_originals)
        lpips_vals = []
        for t in range(all_recons.shape[2]):
            recon_frame = all_recons[:, :, t, :, :]
            orig_frame = all_originals[:, :, t, :, :]
            lpips_vals.append(lpips_loss(recon_frame, orig_frame))
        lpips_val = torch.stack(lpips_vals).mean()
        psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        psnr_score = psnr(all_recons, all_originals)
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        ssim_score = ssim(all_recons, all_originals)

        print(f"Average Metrics - MSE: {mse.item():.4f}, LPIPS: {lpips_val.item():.4f}, PSNR: {psnr_score.item():.4f}, SSIM: {ssim_score.item():.4f}")

if __name__ == "__main__":
    checkpoint_path = "video-generation-model/vae_checkpoint_epoch_24.pth"  
    data_dir = "."  
    output_dir = "reconstructed_videos"
    visualize_reconstructions(checkpoint_path, data_dir, output_dir, num_videos=15)
