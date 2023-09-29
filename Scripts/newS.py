import sys
sys.path.append("./"), sys.path.append("../")
import torch
import argparse
import warnings
import torch.nn as nn 
warnings.filterwarnings('ignore')
from torch.distributions import normal
from Architectures.HuggingFace import * 
from Architectures.SmallArch import SmallArch
from Dataloaders.HinEngDL import AudioDatasetEng
from Architectures.CUnet_V1EMB import Autoencoder
from Dataloaders.TamilEngDL import AudioDatasetTamil
from Architectures.DiffusionProcess import Diffusion
from Diffusion.newTrainingPipeline import TrainingPipe

import logging
logging.disable(logging.CRITICAL)


# ########################################################################################
# parser = argparse.ArgumentParser(description="SpeechToSpeech")
# parser.add_argument("--a", type=str)
# parser.add_argument("--device", type=int)
# parser.add_argument("--language",type=str)
# arguments = parser.parse_args()
# ########################################################################################


# device = torch.device(f"cuda:{arguments.device}" if arguments.device < torch.cuda.device_count() else "cpu")

experiment_name = 'tryAsli'

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
          'timesteps' : 10,
          'batch_size': 24,#20,
          'epochs' : 24,
          "device": 'cpu',
          'num_images' : 15}

model = Autoencoder()


Trainer = TrainingPipe(experiment_name = experiment_name,
                       unet = model,
                       diffusion = Diffusion(timesteps = CONFIG['timesteps']),
                       vocoder = Vocoder(),
                       dataset = AudioDataSet, 
                       criterion = nn.MSELoss(),
                       audioEncoder = AudioEncoder().audioEncoder,
                       vae =Vae(),
                       optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['LR']),
                       smallArchitecture= SmallArch(),
                       config=CONFIG)


for i in range(CONFIG["epochs"]):
    print(i,'epoch number STARTED')
    reconst = Trainer.train_1_epoch(i)
    print(i,'epoch number ENDED')
    # Trainer.eval_1_epoch(i,reconst,CONFIG['num_images'])
    # print(i,'Eval and saved audios')



    
        
    
   

