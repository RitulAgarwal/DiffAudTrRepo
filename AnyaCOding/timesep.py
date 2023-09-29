import torch
import torch.nn as nn 
import numpy as np
sample = torch.randn((4,3,80,800))
timesteps = torch.randint(low =1,high = 1000,size = (4,))
print(timesteps)
timesteps = timesteps.expand(sample.shape[0])
flip_sin_to_cos = True
class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(
        self, embedding_size: int = 256, scale: float = 1.0, set_W_to_weight=True, log=True, flip_sin_to_cos=False
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)
        self.log = log
        self.flip_sin_to_cos = flip_sin_to_cos

        if set_W_to_weight:
            # to delete later
            self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

            self.weight = self.W

    def forward(self, x):
        print(x)
        if self.log:
            x = torch.log(x)
        print(x)

        x_proj = x[:, None] * self.weight[None, :] * 2 * np.pi
        print(x_proj,x_proj.shape)
        
        if self.flip_sin_to_cos:
            out = torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)
        else:
            out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return out

# time_proj = GaussianFourierProjection(
#                 set_W_to_weight=False, log=False, flip_sin_to_cos=flip_sin_to_cos
#             )

print(timesteps,timesteps.shape)
x=timesteps
weight = nn.Parameter(torch.randn(256) * 1.0, requires_grad=False)
# W = nn.Parameter(torch.randn(256) * 1.0, requires_grad=False)

x_proj = x[:, None] * weight[None, :] * 2 * np.pi
print(x_proj,x_proj.shape)
# t_emb = time_proj(timesteps)

if flip_sin_to_cos:
            out = torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)
else:
    out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

print(out,out.shape)