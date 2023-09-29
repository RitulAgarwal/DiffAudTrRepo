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
from torch.utils.data import DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class SimpleDiffusion:
    """
    initialises the basic alpha,beta noise schedule (0<beta1<beta2<----betan<1),
    all preprocessing of underroot 1-alpha and all that's required before forward and reverse diffusion process takes place
    """
    def __init__(self,
                 num_diffusion_timesteps:int=1000,
                 device:str="cpu"):
        
        self.num_diffusion_timesteps = num_diffusion_timesteps
    
        self.device = device

        self.initialize()

    def initialize(self):
        # BETAs & ALPHAs required at different places in the Algorithm.
        self.beta  = self.get_betas()
        self.alpha = 1 - self.beta
        
        self_sqrt_beta                       = torch.sqrt(self.beta)
        self.alpha_cumulative                = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_cumulative           = torch.sqrt(self.alpha_cumulative)
        self.one_by_sqrt_alpha               = 1. / torch.sqrt(self.alpha)
        self.sqrt_one_minus_alpha_cumulative = torch.sqrt(1 - self.alpha_cumulative)
         
    def get_betas(self):
        """linear schedule, proposed in original ddpm paper"""
        scale = 1000 / self.num_diffusion_timesteps
        beta_start = scale * 1e-4
        beta_end = scale * 0.02
        return torch.linspace(
            beta_start,
            beta_end,
            self.num_diffusion_timesteps,
            dtype=torch.float32,
        )

class ReverseDiffusion:
    """takes inputs and returns a converted spectrogram and tries to minimimse the loss 
    """
    def __init__(self,
                 experiment_name,
                mean,
                std,                 
                 config:dict) -> None:
        
      
        self.config = config
        self.simple_diffusion = SimpleDiffusion(self.config["timesteps"])
        self.mean = mean
        self.std = std 
        self.experiment_name = experiment_name
        
        
    
    def get(self,element: torch.Tensor, t: torch.Tensor):
        """
        Get value at index position "t" in "element" and
            reshape it to have the same dimension as a batch of images.
        """
        ele = element.gather(-1, t)#extract the value from the input tensor along with the specified dimension that we want.
        return ele.reshape(-1, 1, 1)
    
    
    def reverse_diffusion(self,epoch,num_images):
        
        normal_distribution = normal.Normal(self.mean,self.std)
        
        shape = (1, self.config["image_shape"][-2], self.config["image_shape"][-1])
        
        sample = normal_distribution.rsample(shape)
        
        reconst_spec = self.smallArchitecture(sample) 

        shape = (1, self.config["image_shape"][-2], self.config["image_shape"][-1])
        
        sample = reconst_spec.to('cpu')
        random_noise = torch.randn(shape)
        
        path = os.path.join(os.path.expanduser("~"), "RITUL_NSUT/Experiment")
        if not os.path.exists(path):
            os.makedirs(path)
       
        shape = (1, self.config["image_shape"][-2], self.config["image_shape"][-1])
        random_noise = torch.randn(shape)

        for time_step in reversed(range(self.config["timesteps"])):

            ts = torch.ones(num_images, dtype=torch.long) * time_step
            
            z = torch.randn_like(random_noise) if time_step > 1 else torch.zeros_like(random_noise)
        
            beta_t = self.get(self.simple_diffusion.beta, ts) # 4,1,1,1

            one_by_sqrt_alpha_t = self.get(self.simple_diffusion.one_by_sqrt_alpha, ts) # 4,1,1,1

            sqrt_one_minus_alpha_cumulative_t = self.get(self.simple_diffusion.sqrt_one_minus_alpha_cumulative, ts)
            
            a = beta_t / sqrt_one_minus_alpha_cumulative_t #([4, 1, 1, 1])
        
            random_noise = (
                one_by_sqrt_alpha_t
                * (random_noise - a * sample)
                + torch.sqrt(beta_t) * z
            )
            random_noise = torch.squeeze(random_noise)
           
        
        count = 0 
    
        path = os.path.join(os.path.expanduser("~"), ("RITUL_NSUT/DataGenerated/"+ self.experiment_name))
        if not os.path.exists(path):
            os.makedirs(path)
            
        for i in random_noise :
    
            out = self.vocoder.convert_spectrogram_to_audio(spec = i.unsqueeze(0).to(self.config['device']))
            
            torchaudio.save(('/home/earth/RITUL_NSUT/DataGenerated/' + self.experiment_name+ '/HifiGan_MEL_Aud_' + str(count) +'_Epoch' + str(epoch)+ '.wav'),out.to('cpu'),self.config['sr'])
            count+=1 
            
       
                    