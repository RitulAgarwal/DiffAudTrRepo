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

class TrainingPipe:
  
    def __init__(self,
                 experiment_name,
                 dataset,
                 criterion,
                 audioEncoder,
                 optimizer,
                 smallArchitecture,
                 unet,
                 vae ,
                 diffusion,
                 vocoder,
                 config:dict) -> None:
        
        self.experiment_name = experiment_name
        path = os.path.join(os.path.expanduser("~"), "RITUL_NSUT/Logs")
        if not os.path.exists(path):
            os.makedirs(path)
        self.log_path = os.path.join(path, f"{experiment_name}.txt")
        self.config = config
        
        self.unet = unet
        self.dataset = dataset
        self.Clayer = torch.nn.ConvTranspose1d(80,80,60,dilation=6)
        self.audioEncoder = audioEncoder
        self.flatten = nn.Flatten()
        self.criterion = criterion
        self.criterion1 = criterion
        self.criterion2 = criterion
        self.criterion3 = criterion
        self.vaeEncoder = vae.encoder
        self.vaeDecoder = vae.decoder
        self.optimizer = optimizer
        self.optimizer1 = optimizer
        self.smallArchitecture = smallArchitecture
        self.diffusion = diffusion
        self.vocoder = vocoder #.to(self.config["device"])
        
    def log(self, *args):
        with open(self.log_path, "a") as F:
            F.write(" ".join([str(i) for i in args]))
            F.write("\n")
        F.close()    
    
    def pipeline(self, minibatch):
        
        Hdata,Edata,Eaudio, _ = minibatch
        
        audio_embeddings = self.audioEncoder.get_audio_embedding_from_data(x = Eaudio, use_tensor=True).to('cpu')
        # print(audio_embeddings.shape,'audio Embedding')
        # print(torch.unsqueeze(Edata,dim=1).shape,'////////////////////////////////')
        print(Edata.shape,'vae input')
        latentSpaceVAE = self.vaeEncoder(torch.unsqueeze(Edata,dim=1)).squeeze()#.to(self.config["device"])
        print(latentSpaceVAE.shape,'latent space of VAE')
        
        latent_representation,model_out = self.unet(audio_embeddings,torch.randn((24)))
        # print(latent_representation.shape,'latent rep of UNET')
        
        loss = self.criterion(latentSpaceVAE, latent_representation)    
        
        # print(latentSpaceVAE,latent_representation,'check if ithsi isnt nan 1111')
        # self.optimizer.zero_grad(set_to_none=True)
        # loss.backward(retain_graph=True)
        # self.optimizer.step()
        
        # print(model_out.shape,'unet output')
        loss1 = self.criterion1(Hdata, model_out)    
        # print(model_out,latent_representation,'check if ithsi isnt nan ******')
        
     
        latent_representation=torch.unsqueeze(latent_representation,dim=(1))
        latent_representation=torch.unsqueeze(latent_representation,dim=(2))
        print(latent_representation.shape,'vae decoder ka input')
        outputSpec = F.pad(self.vaeDecoder(latent_representation),(0,1)).squeeze()#.clone())
        print(outputSpec.shape,'lr and vae output spec')
        torch.autograd.set_detect_anomaly(True)

        loss2 = self.criterion2(outputSpec, Hdata)    
        
        # L = loss+loss1 + loss2
        # print(L,'accum loss')
        # self.optimizer.zero_grad()
        # L.backward(retain_graph=True)
        # self.optimizer.step()
    
        # print(outputSpec, Hdata,'------------------')
        
        # print(latent_representation.shape,'latent representation')
        latent_representation = latent_representation.squeeze()
        # print(latent_representation.shape,'latent representation')
        mean = torch.mean(latent_representation)
        std = torch.std(latent_representation)
        # print(mean,std,'mu,signa')
        normal_distribution = normal.Normal(mean,std)
        shape = outputSpec.shape
        sample = normal_distribution.rsample(shape)
        # print(sample.shape,'sample')
        reconst_spec = self.smallArchitecture(sample) 
        # print(reconst_spec,'samlscr')
        
        loss3 = self.criterion3(reconst_spec, Hdata)   
        
        # eps = 1e-6
        # if loss.isnan(): loss=eps
        # else: loss = loss.item() 
        
        # print(loss3,' loss finally')
        # self.optimizer1.zero_grad()
        # loss3.backward()
        # self.optimizer1.step()

        L = loss+loss1 + loss2 + loss3
        print(loss,loss1,loss2, loss3,L)
        self.optimizer.zero_grad()
        L.backward(retain_graph=True)
        self.optimizer.step()
        
        # specOut = self.diffusion.calcs(reconst_spec)
        # print(specOut.shape,'afer dif')
        

        # print(reconst_spec,'*********---------------------------------------')
        
        return reconst_spec,L

    
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
            
            latentRep,loss = self.pipeline(minibatch=minibatch)
            epoch_loss += loss.detach().item()
            
            torch.cuda.empty_cache()
            gc.collect()
            
            self.log(f"Epoch - {epoch}",
                    f"minibatch idx - {minibatch_idx}",
                    f"Loss - {(epoch_loss/len(dataloader)):.4f}",
                    f"Time Taken - {((time.time()-start_time)/36000):.4f}")       
            
            print(f"Epoch - {epoch} minibatch idx - {minibatch_idx} Loss - {(epoch_loss/len(dataloader)):.4f} Training - {(100 * (minibatch_idx/dataloader.__len__())):.4f}")
    
        # self.save_checkpoints(epoch)     
        # return reconst_spec

    
    def save_checkpoints(self, epoch):
        path = os.path.join(os.path.expanduser("~"), "RITUL_NSUT/Experiment")#checkpointing
        if not os.path.exists(path):
            os.makedirs(path)
            
        chkpt = {"architecture": self.architecture.state_dict(),
                 "optimizer": self.optimizer.state_dict(),
                 "epoch": epoch}
        
        torch.save(chkpt, os.path.join(path, f"{self.experiment_name}_checkpoint_{epoch}.pth"))
    
    
    def eval_1_epoch(self, epoch,reconstructed_spectrogram , num_images):
        path = os.path.join(os.path.expanduser("~"), "RITUL_NSUT/Experiment")
        if not os.path.exists(path):
            os.makedirs(path)
       
        shape = (1, self.config["image_shape"][-2], self.config["image_shape"][-1])
        sample = reconstructed_spectrogram.to('cpu')
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
     
        # Log10MEL = torch.log10(random_noise+1e-9)
        # SPECout = self.vocoder.convert_spectrogram_to_audio(spec = Log10MEL.to(self.config['device']))
        # print(SPECout)
        path = os.path.join(os.path.expanduser("~"), ("RITUL_NSUT/DataGenerated/"+ self.experiment_name))
        if not os.path.exists(path):
            os.makedirs(path)
            
        for i in random_noise :
            # print(i,'original')
            # out = self.vocoder.convert_spectrogram_to_audio(spec = i.unsqueeze(0).to(self.config['device']))
            #  self.vocoder.generate_wav(numpy.absolute(mel_spectrogram)) VAKYANSH 
            # print(out,'after vocoder')
            torchaudio.save(('/home/earth/RITUL_NSUT/DataGenerated/' + self.experiment_name+ '/HifiGan_MEL_Aud_' + str(count) +'_Epoch' + str(epoch)+ '.wav'),out.to('cpu'),self.config['sr'])
            count+=1 
            
        # count = 0
        # for i in Log10MEL :
        #     print(i,'originalLOGMEL')
        #     out = self.vocoder.convert_spectrogram_to_audio(spec = i.unsqueeze(0).to(self.config['device']))
        #     print(out,'after vocoderLOGMEL')
        #     torchaudio.save(('/home/earth/RITUL_NSUT/DATAgenerated/' + self.experiment_name+ '/HifiGan_MELLOG10_Aud_' + str(count) +'_Epoch' + str(epoch)+ '.wav'),out.to('cpu'),self.config['sr'])
        #     count+=1  
            
                    