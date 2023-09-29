import sys
sys.path.append("./"), sys.path.append("../")
import torch
import argparse
import warnings
import torch.nn as nn 
warnings.filterwarnings('ignore')
from Diffusion.ForwardDiffusion1 import DDPM_Trainer
from Architectures.SmallArch import SmallArch
from Architectures.HuggingFace import * 
from Dataloaders.HinEngDL import AudioDatasetEng
from Dataloaders.TamilEngDL import AudioDatasetTamil
from Architectures.CUnet_V6 import Unet


import logging
logging.disable(logging.CRITICAL)


########################################################################################
parser = argparse.ArgumentParser(description="SpeechToSpeech")
parser.add_argument("--a", type=str)
parser.add_argument("--device", type=int)
parser.add_argument("--language",type=str)
arguments = parser.parse_args()
########################################################################################


device = torch.device(f"cuda:{arguments.device}" if arguments.device < torch.cuda.device_count() else "cpu")

experiment_name = arguments.a

language = arguments.language

if language.lower() == 'tamil':
    AudioDataSet = AudioDatasetTamil()
else :
    AudioDataSet = AudioDatasetEng()
    
CONFIG = {'image_shape':(1,80,885),
          'BASE_CH' : 64,
          'TIME_EMB_MULT' : 2 ,
          'sr' : 48000,
          'LR' : 2e-4,
          'timesteps' : 1000,
          'batch_size': 24,#20,
          'epochs' : 24,
          "device": device,
          'num_images' : 15}

model = Unet(seq_length=512,KernelSizeList=[13,9,5])

Trainer = DDPM_Trainer(experiment_name = experiment_name,
                       architecture =  model ,
                       dataset = AudioDataSet, 
                       criterion = nn.MSELoss(),
                       audioEncoder =AudioEncoder().audioEncoder,
                       vae=vae().vae,
                       optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['LR']),
                       smallArchitecture= SmallArch(),
                    #    vocoder=Vocoder().vocoder,
                       text_tokenizer = TextTokenizer().Texttokenizer, 
                       text_encoder =TextEncoder().Textencoder,
                       config=CONFIG)


for i in range(CONFIG["epochs"]):
    print(i,'epoch number STARTED')
    reconst = Trainer.forwardProcess(i)
    print(i,'epoch number ENDED')
    Trainer.reverseProcess(i,reconst,CONFIG['num_images'])
    print(i,'reverse process ended')