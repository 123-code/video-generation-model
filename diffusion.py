import torch
import torch.nn.functional as F 
import numpy as np 

def extract(a,t,x_shape):
    batch_size = t.shape[0]
    out = a.to(t.device).gather(0,t)
    return out.reshape(batch_size,*((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps 
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start,beta_end,timesteps,dtype=torch.float64) 


class DiffusionProcess:
    def __init__(self,timesteps=1000):
        self.timesteps = timesteps 
        #cuanto noise se agrega a los datos en cada timestep
        betas = linear_beta_schedule(timesteps)
        #cuanta informacion de la imagen orgia=nal queda en cada timestep 
        alphas = 1. - betas
         #producto cumulativo del tensor alphas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        #tensor cumprod de alphas, pero movido un timestep, esto ayuda en el proceso de reconstruccion 
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

    def register_buffer(self,name,attr):
        setattr(self,name,attr)

    def q_sample(self,x_start,t,noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod,t,x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod,t,x_start.shape)
        noisy_x = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return noisy_x
    
    def training_loss(self,model,x_start,t):
        #retorna el error entre el noise predicho y el verdadero (mse loss)
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start,t,noise=noise)
        predicted_noise = model(x_t,t)
        loss = F.mse_loss(predicted_noise,noise)
        return loss




    