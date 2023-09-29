import sys
sys.path.append("./"), sys.path.append("../")
import torch
import argparse
import warnings
import torch.nn as nn 
warnings.filterwarnings('ignore')
from Architectures.HuggingFace import * 
from Dataloaders.HinEngDL import AudioDatasetEng
# from Architectures.UNET import UNet
from Architectures.LDMmodels import Unet
from Dataloaders.newDLTE import AudioDatasetTamil
from Diffusion.BASICpipeline import TrainingPipe
# from hifigan_vocoder import hifigan
from transformers import SpeechT5HifiGan


import logging
logging.disable(logging.CRITICAL)

experiment_name = 'GaussianNoiseonENgS_LDM_HinLossMSE'

language = 'tamil'

if language.lower() == 'tamil':
    AudioDataSet = AudioDatasetTamil()
else :
    AudioDataSet = AudioDatasetEng()
    
CONFIG = {'image_shape':(1,80,885),
          'BASE_CH' : 64,
          'TIME_EMB_MULT' : 2 ,
          'sr' : 48000,
          'LR' : 2e-4,
          'timesteps' : 300,
          'batch_size': 16,#20,
          'epochs' : 24,
          "device": 'cuda:1',
          'num_images' : 15}

# model = UNet(enc_chs=(1,64,128,256,512,1024), dec_chs=(1024, 512, 256, 128, 64))
image_size = 4
channels = 1
batch_size = CONFIG['batch_size']
device = CONFIG['device']
model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(1, 2, 4)
)
# model.to(device)

Trainer = TrainingPipe(experiment_name = experiment_name,
                       unet = model,
                    #    vocoder = hifigan(),
                       vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan"), 
                       dataset = AudioDataSet, 
                       criterion = nn.MSELoss(),
                       audioEncoder = AudioEncoder().audioEncoder,
                       optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['LR']),
                       config=CONFIG)


for i in range(CONFIG["epochs"]):
    print(i,'epoch number STARTED')
    reconst = Trainer.train_1_epoch(i)
    print(i,'epoch number ENDED')
  


    
