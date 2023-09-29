import torch.nn as nn 
import torch
import numpy as np 
import torch.nn.functional as F



class StatsPool(nn.Module):
    def __init__(self,
                 channels:int=256,
                 latent_dim:int=512) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.conv_1 = nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=1)
        self.mean_ = nn.AdaptiveAvgPool1d(output_size=latent_dim)
        
    def forward(self, x):
        print(x.shape)
        o = self.conv_1(x)
        print(o.shape)
        mean = self.mean_(o)
        print(mean.shape,'************')
        std = torch.sqrt((self.mean_(o**2)-self.mean_(o)**2))
        mean_std = torch.cat([mean, std],dim=2).squeeze()
        latent_vec = torch.zeros(self.latent_dim)
        for i in range(self.latent_dim):
            q = torch.distributions.Normal(mean_std[0][i].item(),mean_std[0][i+self.latent_dim].item())
            z = q.rsample()
            latent_vec[i] = z
        latent_vec = latent_vec[None,:]
        return latent_vec

class VariationalAutoEncoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels= 4, kernel_size=13),
            nn.ReLU(),
            nn.Conv2d(4,16,11),
            nn.ReLU(),
            nn.Conv2d(16,64,7),
            nn.ReLU(),
            nn.Conv2d(64,128,5),
            nn.ReLU(),
            nn.Conv2d(128,256,3),
            nn.ReLU() ,
            StatsPool(latent_dim=latent_dim)   
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1,128,29,dilation=4,stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64,39,dilation=5,stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 16,41,dilation=5,stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16,4,43,dilation=5,stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(4,1,45,dilation=5,stride=3),
            nn.ReLU(),
        )
    
        
    def forward(self, x):
        print(x.shape,'input')
        eo = self.encoder(x)
        print(eo.unsqueeze(1).shape,'encoder out')
        d = self.decoder(eo.unsqueeze(1))
        return d
                 
                 
                 
                 
                 
    # def forward(self, x):
    #     print(x[0].shape,'input')
    #     x = self.encoder(x[0])
    #     print(x.shape,'encoder')  
    #     x = torch.permute(x,(1,0))  # 31972,256    
    #     y = torch.zeros(x.shape[0])
    #     print(y.shape)
    #     for i in range(len(x)):
    #         y[i] = torch.mean(x[i])
    #     print(y.shape,'shape of avg y ')
    #     y = y[None,:]
    #     mu = self.mean(y).squeeze() 
    #     print(mu.shape, 'mean')
    #     var = (self.mean(y**2) - self.mean(y)**2).squeeze() 
    #     std = torch.sqrt(var) 
    #     print(std.shape,'std')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
    #     mu_std = torch.cat((mu,std))
    #     print(mu_std.shape,'latent embedding stacked mu and sigma ki shape ')
    #     latent_vec = torch.zeros(self.latent_dim)
    #     for i in range(self.latent_dim):
    #         q = torch.distributions.Normal(mu_std[i],mu_std[i+self.latent_dim])
    #         z = q.rsample()
    #         latent_vec[i] = z
    #     latent_vec = latent_vec[None,:]
    #     print(latent_vec.shape,'sample ki shape')
    #     z = self.decoder(latent_vec)
    #     print(z.shape,'decoded ouptu')   
    #     return z
    
    
vae = VariationalAutoEncoder()
o = vae.forward((torch.randn(4,1,80,880)))
print(o.shape)
from torchinfo import summary
summary(vae)

