import sys; sys.path.append("./"), sys.path.append("../")
import os
import gc
import time
import torch
import torchaudio
import torch.nn as nn 
from typing import Tuple
import torch.nn.functional as F
from torch.distributions import normal
from Architectures.HuggingFace import * 
from Architectures.CUnet_V1EMB import Autoencoder
from Architectures.DiffusionProcess import Diffusion
from Architectures.SmallArch import SmallArch

class Architecture(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.AudioEncoder = AudioEncoder().audioEncoder
        self.Unet = Autoencoder()
        # self.vocoder = Vocoder()
        self.Diffusion = Diffusion(timesteps = 10)
        self.smallArch = SmallArch()
        
    def forward(self,Eaudio):
        print(Eaudio.shape,'/')
        EaudioEmbedding = self.AudioEncoder.get_audio_embedding_from_data(x = Eaudio, use_tensor=True).to('cpu')
        print(EaudioEmbedding.shape,'//')
        LR,a = self.Unet(EaudioEmbedding,torch.randn((EaudioEmbedding.shape[0])))
        print(LR.shape,'..')
        print(a.shape,'!!')
        SpecSize = a.shape
        mu,sigma = torch.mean(LR),torch.std(LR)
        print(mu,sigma)
        normal_distribution = normal.Normal(mu,sigma)
        sample = normal_distribution.rsample(SpecSize)
        print(sample.shape,'lll')
        L1R = self.smallArch(sample)
        print(L1R.shape,'>')     
        denoisedSpec = self.Diffusion.calcs(L1R)
        print(denoisedSpec.shape,'aert')
        HAudio = self.vocoder(denoisedSpec)
        print(HAudio.shape,'kkk')
        
        return HAudio
    

a = Architecture()
input = torch.randn((5,192000))
out = a(input)       