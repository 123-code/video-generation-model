# Video Diffusion Transformer (VideoDiT)

https://github.com/user-attachments/assets/22e77f09-dfd2-4f9d-b0ae-0a9825411132



A class-conditional video generation model using a Diffusion Transformer architecture operating in VAE latent space. The model generates 16-frame videos at 256x256 resolution through iterative denoising.

## Architecture Overview

```
Input Noise (4, 16, 32, 32)
         │
         ▼
   ┌─────────────┐
   │  Patchify   │  3D patches (2x2x2)
   └─────────────┘
         │
         ▼
   ┌─────────────┐
   │ Patch Embed │  Linear projection to dim=1024
   └─────────────┘
         │
         ▼
   ┌─────────────┐
   │  + PosEmbed │  Learned positional embeddings
   └─────────────┘
         │
         ▼
  ┌──────────────────┐
  │  SpatioTemporal  │ ×16 blocks
  │    DiT Blocks    │
  │  (with adaLN)    │
  └──────────────────┘
         │
         ▼
   ┌─────────────┐
   │ Final Layer │  Project back to patch dim
   └─────────────┘
         │
         ▼
   ┌─────────────┐
   │ Unpatchify  │  Reconstruct latent tensor
   └─────────────┘
         │
         ▼
  Predicted Noise (4, 16, 32, 32)
```


https://github.com/user-attachments/assets/71797ead-1b1a-4eec-9ac9-2dea3b05a3b3

https://github.com/user-attachments/assets/7abc5521-6873-4963-8ae1-4aa0bf096fdc

