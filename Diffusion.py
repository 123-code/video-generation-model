import torch 
import torch.nn as nn
import torch.nn.functional as F
import math 
from tqdm import tqdm

def cosine_beta_schedule(timestamps,s=0.008):
    steps = timestamps + 1
    #crea un vector de pasos de 0 a timestamps
    x = torch.linspace(0,timestamps,steps)
    #alphas_cumprod es un array que indica cuanta "senal" queda de la altente original 
    alphas_cumprod = torch.cos((x / timestamps + s) / (1 + s) * math.pi * 0.5) ** 2
    #normaliza el array para que el primer elemento sea 1
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    # how much of the remaining signal did we just eliminate? caculate with 1- signal left as t/signal left at t-1
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clip(betas,0.0001,0.9999)

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

class GaussianDiffusion:
    def __init__(self, timesteps=1000, beta_schedule='cosine'):
        self.timesteps = timesteps
        
        if beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            betas = linear_beta_schedule(timesteps)
        
        betas = betas.double()
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)
        
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
    
    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).float()
    
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
    
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance
    
    def p_mean_variance(self, model, x, t, y=None, cfg_scale=1.0, clip_denoised=True):
        if cfg_scale > 1.0 and y is not None:
            x_double = torch.cat([x, x], dim=0)
            t_double = torch.cat([t, t], dim=0)
            y_null = torch.full_like(y, model.num_classes)  # null class
            y_double = torch.cat([y, y_null], dim=0)
            noise_pred = model(x_double, t_double, y_double)
            noise_cond, noise_uncond = noise_pred.chunk(2, dim=0)
            noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
        else:
            noise_pred = model(x, t, y)
        
        x_start = self.predict_start_from_noise(x, t, noise_pred)
        
        if clip_denoised:
            x_start = torch.clamp(x_start, -1.0, 1.0)
        
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start, x, t)
        return model_mean, posterior_variance, posterior_log_variance, x_start
    
    @torch.no_grad()
    def p_sample(self, model, x, t, y=None, cfg_scale=1.0, clip_denoised=True):
        b = x.shape[0]
        t_batch = torch.full((b,), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            model, x, t_batch, y, cfg_scale, clip_denoised
        )
        noise = torch.randn_like(x) if t > 0 else 0
        return model_mean + (0.5 * model_log_variance).exp() * noise, x_start
    
    @torch.no_grad()
    def ddpm_sample(self, model, shape, y=None, cfg_scale=1.0, device='cuda', progress=True):
        b = shape[0]
        x = torch.randn(shape, device=device)
        
        iterator = reversed(range(self.timesteps))
        if progress:
            iterator = tqdm(iterator, desc='DDPM Sampling', total=self.timesteps)
        
        for t in iterator:
            x, _ = self.p_sample(model, x, t, y, cfg_scale)
        
        return x
    
    @torch.no_grad()
    def ddim_sample(self, model, shape, y=None, cfg_scale=1.0, ddim_steps=50, eta=0.0, device='cuda', progress=True):
        b = shape[0]
        x = torch.randn(shape, device=device)
        
        times = torch.linspace(-1, self.timesteps - 1, ddim_steps + 1).long()
        times = list(reversed(times.tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        
        iterator = time_pairs
        if progress:
            iterator = tqdm(iterator, desc='DDIM Sampling')
        
        for time, time_next in iterator:
            t = torch.full((b,), time, device=device, dtype=torch.long)
            
            if cfg_scale > 1.0 and y is not None:
                x_double = torch.cat([x, x], dim=0)
                t_double = torch.cat([t, t], dim=0)
                y_null = torch.full_like(y, model.num_classes)
                y_double = torch.cat([y, y_null], dim=0)
                noise_pred = model(x_double, t_double, y_double)
                noise_cond, noise_uncond = noise_pred.chunk(2, dim=0)
                noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
            else:
                noise_pred = model(x, t, y)
            
            x_start = self.predict_start_from_noise(x, t, noise_pred)
            x_start = torch.clamp(x_start, -1.0, 1.0)
            
            if time_next < 0:
                x = x_start
                continue
            
            alpha = self._extract(self.alphas_cumprod, t, x.shape)
            alpha_next = self._extract(self.alphas_cumprod, torch.full((b,), time_next, device=device, dtype=torch.long), x.shape)
            
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            
            noise = torch.randn_like(x) if eta > 0 else 0
            x = x_start * alpha_next.sqrt() + c * noise_pred + sigma * noise
        
        return x
    
    def training_losses(self, model, x_start, t, y=None, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_t = self.q_sample(x_start, t, noise)
        noise_pred = model(x_t, t, y)
        
        loss = F.mse_loss(noise_pred, noise)
        return loss


class EMA:
    def __init__(self, model, decay=0.9999, warmup_steps=2000):
        self.model = model
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step = 0
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        self.step += 1
        decay = min(self.decay, (1 + self.step) / (10 + self.step))
        if self.step < self.warmup_steps:
            decay = 0.0
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = decay * self.shadow[name] + (1 - decay) * param.data
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self):
        return {
            'shadow': self.shadow,
            'step': self.step
        }
    
    def load_state_dict(self, state_dict):
        self.shadow = state_dict['shadow']
        self.step = state_dict['step']