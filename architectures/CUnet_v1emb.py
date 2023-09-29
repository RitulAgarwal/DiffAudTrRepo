import math

import torch

import torch.nn as nn 

import torch.nn.functional as F 

""" ONLY ENCODER DECODER WITH TIMESTEPS """

class Encoder(nn.Module):
    
    def __init__(self,
                 num_input_channels : int,
                 base_channel_size : int,
                 latent_dim : int,
                 act_fn : object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.l1 = nn.Conv1d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2) # 32x32 => 16x16
        self.re = act_fn()
        self.l2 = nn.Conv1d(c_hid, c_hid, kernel_size=3, padding=1)
        self.l3 = nn.BatchNorm1d(c_hid)
        self.l4 = nn.Conv1d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2) # 16x16 => 8x8
        self.l5 = nn.Conv1d(2*c_hid, 4*c_hid, kernel_size=3, padding=1)
        self.l6 = nn.BatchNorm1d(4*c_hid)
        self.l7 = nn.Conv1d(4*c_hid, 6*c_hid, kernel_size=3, padding=1, stride=2)# 8x8 => 4x4
    
        self.l10 = nn.Flatten()
        self.net = nn.Sequential(
            nn.Conv1d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2), # 32x32 => 16x16
            act_fn(),
            nn.Conv1d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.BatchNorm1d(c_hid),
            nn.Conv1d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 16x16 => 8x8
            act_fn(),
            nn.Conv1d(2*c_hid, 4*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.BatchNorm1d(4*c_hid),
            nn.Conv1d(4*c_hid, 6*c_hid, kernel_size=3, padding=1, stride=2), # 8x8 => 4x4
            act_fn(),
            nn.Flatten(), # Image grid to single feature vector
           
        )

    def forward(self, x): 
        # print(x.shape)
        # print(self.l1(x).shape)
        # print(self.l2(self.l1(x)).shape)
        # print(self.l3(self.l2(self.l1(x))).shape)
        # print(self.l4(self.l3(self.l2(self.l1(x)))).shape)
        # print(self.l5(self.l4(self.l3(self.l2(self.l1(x))))).shape)
        # print(self.l6(self.l5(self.l4(self.l3(self.l2(self.l1(x)))))).shape)
        # print(self.l7(self.l6(self.l5(self.l4(self.l3(self.l2(self.l1(x))))))).shape)
       
        return self.net(x)
    
class Decoder(nn.Module):

    def __init__(self,
                 num_input_channels : int,
                 base_channel_size : int,
                 latent_dim : int,
                 act_fn : object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        c_hid = 80
        print(c_hid,'chewiuhd')
        # self.linear = nn.Linear(latent_dim,27)
        self.net = nn.Sequential(
            nn.ConvTranspose1d(256, 3*c_hid, kernel_size=1,padding= 27),
            act_fn(),
            nn.ConvTranspose1d(3*c_hid, 3*c_hid, kernel_size=1,padding= 27),
            act_fn(),
            # nn.BatchNorm1d(4*c_hid),
            nn.ConvTranspose1d(3*c_hid, 3*c_hid, kernel_size=1,padding= 27), # 8x8 => 16x16
            act_fn(),
            nn.ConvTranspose1d(3*c_hid, 3*c_hid, kernel_size=1,padding= 27),
            act_fn(),
            # nn.BatchNorm1d(c_hid),
            # nn.ConvTranspose1d(c_hid, c_hid, kernel_size=1,padding= 27),
            # act_fn(),3
            nn.ConvTranspose1d(3*c_hid, c_hid, kernel_size=1,padding= 26), # 16x16 => 32x32
            # nn.Tanh() # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        # x = self.linear(x)
        # print(x.shape)
        # x = x.reshape(x.shape[0], -1, 4, 4)
   
        x = self.net(x)
       
        return x

class Autoencoder(nn.Module):
    """ ONLY ENCODER DECODER WITH TIMESTEPS """
    def __init__(self,
                 base_channel_size: int = 3 ,
                 latent_dim: int = 256,
                 encoder_class : object = Encoder,
                 decoder_class : object = Decoder,
                 num_input_channels: int = 3):
        super().__init__()
        
        self.encoder = encoder_class(num_input_channels, base_channel_size, latent_dim)
        self.decoder = decoder_class(num_input_channels, base_channel_size, latent_dim)
        self.act_fn = nn.SiLU()
        self.sin = PositionalEncoding(d_model=512)
        self.dense_1 = nn.Conv1d(1, 3, 1)
        self.dense_2 = nn.Linear(in_features=512, out_features=1)
        self.out = nn.Conv1d(1, 256, 1)
        
    def forward(self, x, timesteps):
        # print(x.shape,'unet input') #5,512
        # print(timesteps.shape,'timesteps') #5
        sinee = self.sin(timesteps) # 5,512
        #####ADDING POSITIONAL EMB IF U WANT
        # x += self.act_fn(sinee)[:, :]
        # print(x.shape,'here')
        x = self.dense_1(x.unsqueeze(1))
        # print(x.shape,'her1')
        z = self.encoder(x)
        # print(z.shape,'pp')
        
        #####ADDING POSITIONAL EMB IF U WANT
        zt = z #+ self.dense_2(self.act_fn(sinee))
        zt = self.out(zt.unsqueeze(1))
        # print(zt.shape,'oo')
    #    
       
        x_hat = F.pad((self.decoder(zt)),(0,1))
        # print(x_hat.shape,'//')
        return z,x_hat
    
# 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model= 256, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pe = torch.zeros(max_len, d_model)
       
        self.pe[:, 0::2] = torch.sin(position * div_term)#for even terms sin 
        self.pe[:, 1::2] = torch.cos(position * div_term)#for odd terms cos
       
        self.b = nn.Linear(in_features=d_model, out_features=d_model)
        self.c = nn.Linear(in_features=d_model, out_features=d_model)

    def forward(self, t) :
    
        return self.c(self.b(self.pe[:t.size(0)]))
    
