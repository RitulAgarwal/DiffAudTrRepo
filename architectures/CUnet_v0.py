import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F 
#only encoder decoder paradigm with spectrogram input 

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
        self.l1 = nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2) # 32x32 => 16x16
        self.re = act_fn()
        self.l2 = nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1)
        self.l3 = nn.BatchNorm2d(c_hid)
        self.l4 = nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2) # 16x16 => 8x8
        self.l5 = nn.Conv2d(2*c_hid, 4*c_hid, kernel_size=3, padding=1)
        self.l6 = nn.BatchNorm2d(4*c_hid)
        self.l7 = nn.Conv2d(4*c_hid, 6*c_hid, kernel_size=3, padding=1, stride=2)# 8x8 => 4x4
    
        self.l10 = nn.Flatten()
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2), # 32x32 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.BatchNorm2d(c_hid),
            nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 16x16 => 8x8
            act_fn(),
            nn.Conv2d(2*c_hid, 4*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.BatchNorm2d(4*c_hid),
            nn.Conv2d(4*c_hid, 6*c_hid, kernel_size=3, padding=1, stride=2), # 8x8 => 4x4
            act_fn(),
            nn.Flatten(), # Image grid to single feature vector
            nn.Linear(73728,latent_dim)
        )

    def forward(self, x):       
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
        c_hid = base_channel_size
        self.linear = nn.Sequential(
            nn.Linear(latent_dim,16*2*c_hid),
            act_fn()
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 6*c_hid, kernel_size=(3,7), padding=(1,0),stride = 2),
            act_fn(),
            nn.ConvTranspose2d(6*c_hid, 4*c_hid, kernel_size=(3,7)),
            act_fn(),
            nn.BatchNorm2d(4*c_hid),
            nn.ConvTranspose2d(4*c_hid, 2*c_hid, kernel_size=(3,7), output_padding=1, padding=(1,0), stride=2,dilation = (2,10)), # 8x8 => 16x16
            act_fn(),
            nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=(3,10), padding=(1,0)),
            act_fn(),
            nn.BatchNorm2d(c_hid),
            nn.ConvTranspose2d(c_hid, c_hid, kernel_size=(3,10), padding=(1,0)),
            act_fn(),
            nn.ConvTranspose2d(c_hid, 1, kernel_size=(3,9), output_padding=1, padding=(1,0), stride=(4,7),dilation = (2,9)), # 16x16 => 32x32
            nn.Tanh() # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = F.pad((self.net(x)),(2,2))
        return x

class Autoencoder(nn.Module):
    """ 
    only encoder decoder paradigm with spectrogram input 
    """
    def __init__(self,
                 base_channel_size: int,
                 latent_dim: int,
                 encoder_class : object = Encoder,
                 decoder_class : object = Decoder,
                 num_input_channels: int = 3,
                 width: int = 32,
                 height: int = 32):
        super().__init__()
        
        self.encoder = encoder_class(num_input_channels, base_channel_size, latent_dim)
        self.decoder = decoder_class(num_input_channels, base_channel_size, latent_dim)
       
    def forward(self, x):
       
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z,x_hat
    