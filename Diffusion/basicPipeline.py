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
from Architectures.LDMmodels import q_sample

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class TrainingPipe:
  
    def __init__(self,
                 experiment_name,
                 dataset,
                 criterion,
                 audioEncoder,
                 optimizer,
                 unet,
                 vocoder,
                 config:dict) -> None:
        
        self.experiment_name = experiment_name
        path = os.path.join(os.path.expanduser("~"), "RITUL_NSUT/Logs")
        if not os.path.exists(path):
            os.makedirs(path)
        self.log_path = os.path.join(path, f"{experiment_name}.txt")
        self.config = config
        self.device = self.config['device']
        self.unet = unet
        self.dataset = dataset
        # self.Clayer = nn.ConvTranspose1d(80,80,60,dilation=6)
        self.audioEncoder = audioEncoder
        # self.flatten = nn.Flatten()
        self.criterion = criterion
        self.criterion1 = criterion
        self.optimizer = optimizer
        # self.extrapolate = nn.Linear(512,885)
        self.vocoder = vocoder #.to(self.config["device"])
        
        
    def log(self, *args):
        with open(self.log_path, "a") as F:
            F.write(" ".join([str(i) for i in args]))
            F.write("\n")
        F.close()    
    
    def p_losses(self,original, predicted, loss_type:str="l1"):

            if loss_type == 'l1':
                loss = F.l1_loss(original, predicted)
            elif loss_type == 'l2':
                loss = F.mse_loss(original, predicted)
            elif loss_type == "huber":
                loss = F.smooth_l1_loss(original, predicted)
            else:
                raise NotImplementedError()

            return loss
    
    def pipeline(self, minibatch):
        
        Hdata,Edata,Haudio,Eaudio, _ = minibatch
        print(Hdata.shape,Edata.shape,Eaudio.shape,Haudio.shape)
        Hdata = Hdata.unsqueeze(1)
        Edata = Edata.unsqueeze(1)
        
        # EngAudioEmb = self.audioEncoder.get_audio_embedding_from_data(x = Eaudio, use_tensor=True).to('cpu')
        # EngAudioEmb = F.pad(EngAudioEmb,(0,4))
        # print(EngAudioEmb.shape)
        # fusedSpecEMb = torch.mul(Edata.permute(1,0,2),EngAudioEmb).permute(1,0,2)
        # print(fusedSpecEMb[0].shape)
        # GeneratedHinSPec = self.unet(fusedSpecEMb.unsqueeze(1)).squeeze()#timestep # replace unet with ldm
        # print(Edata.shape[0])
        
        t = torch.randint(0, self.config['timesteps'], (Edata.shape[0],)).long()
        print(t.shape,'time')
        
        # Architectures/hindiNoiseonENgS_LDM_HinLossL1
        # HindinoisedEng = q_sample(x_start=Edata, t=t, noise=Hdata)#noising the english spectrogam with hindi spectrogram acc to random timestep 
        # print(HindinoisedEng.shape,t.shape)
        # Output = self.unet(HindinoisedEng,t)
        # print(Output.shape)
        # l1 =  self.p_losses(Hdata,Output,'l2')
        # print(l1)
        
        # # Architectures/GaussianNoiseonENgS_LDM_HinLossMSE
        # GaussianNoisedEng = q_sample(x_start=Edata, t=t, noise=torch.randn_like(Edata))#noising the english spectrogam with hindi spectrogram acc to random timestep 
        # print(GaussianNoisedEng.shape,t.shape)
        # Output = self.unet(GaussianNoisedEng,t)
        # print(Output.shape)
        # l1 =  self.p_losses(Hdata,Output,'l2')
        # print(l1)
        
        # Architectures/HindiNoiseonENgS_removeHinNoise_huber
        HindiNoisedEng = q_sample(x_start=Edata, t=t, noise=Hdata)#noising the english spectrogam with hindi spectrogram acc to random timestep 
        print(HindiNoisedEng.shape,t.shape)
        PredictedHinNoise = self.unet(HindiNoisedEng,t)
        # print(Output.shape)
        l1 =  self.p_losses(Hdata,PredictedHinNoise,'l2')
        print(l1)
        # OutputNoise = PredictedHinNoise        
        
        
        # audio = self.vocoder(Output.squeeze().permute(0,2,1))
        
        # l2 = self.criterion1(audio, Haudio)
        # print(l2)
        # loss = l1+l2
        # print(loss)
        loss = l1
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
        
        return loss#,audio
        
        
    def dataloader(self):
        return DataLoader(dataset=self.dataset,
                          batch_size=self.config["batch_size"],
                          shuffle=True,
                          num_workers=4,
                          pin_memory=True)
        

    def train_1_epoch(self, epoch):

        self.unet.train()
     
        self.criterion.train()
        self.criterion1.train()
        dataloader = self.dataloader()
        
        start_time = time.time()
        epoch_loss = 0 
        
        for minibatch_idx, minibatch in enumerate(dataloader):
            
            loss,audio = self.pipeline(minibatch=minibatch)
            epoch_loss += loss.detach().item()
                
            
            torch.cuda.empty_cache()
            gc.collect()
            
            self.log(f"Epoch - {epoch}",
                    f"minibatch idx - {minibatch_idx}",
                    f"Loss - {(epoch_loss/len(dataloader)):.4f}",
                    f"Time Taken - {((time.time()-start_time)/36000):.4f}")       

            print(f"Epoch - {epoch} minibatch idx - {minibatch_idx} Loss - {(epoch_loss/len(dataloader)):.4f} Training - {(100 * (minibatch_idx/dataloader.__len__())):.4f}")
            
        print(audio.shape,'////')
        for sno,i in enumerate(audio.unsqueeze(1)) : 
            path = '/home/earth/RITUL_NSUT/Architectures/'+ self.experiment_name + '/' +str(epoch)+ '/'
            if not os.path.exists(path):
                os.makedirs(path)
            print(i.shape,'{{{}}}')
            torchaudio.save((path+str(sno)+'out.wav'), i, 22000)
        
        # self.save_checkpoints(epoch)     
        # return reconst_spec

    