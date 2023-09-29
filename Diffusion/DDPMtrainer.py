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

class DDPM_Trainer:
    """takes inputs and returns a converted spectrogram and tries to minimimse the loss 
    """
    def __init__(self,
                 experiment_name,
                 architecture,
                 dataset,
                 criterion,
                 audioEncoder,
                 optimizer,
                 smallArchitecture,
                 text_tokenizer,
                 vae,
                 text_encoder,
                #  vocoder,
                 config:dict) -> None:
        
        self.experiment_name = experiment_name
        path = os.path.join(os.path.expanduser("~"), "RITUL_NSUT/Logs")
        if not os.path.exists(path):
            os.makedirs(path)
        self.log_path = os.path.join(path, f"{experiment_name}.txt")
        self.config = config
        
        self.architecture = architecture
        self.dataset = dataset
        self.Clayer = torch.nn.ConvTranspose1d(80,80,60,dilation=6)
        self.audioEncoder = audioEncoder
        self.flatten = nn.Flatten()
        self.criterion = criterion
        self.vae = vae
        self.optimizer = optimizer
        self.smallArchitecture = smallArchitecture
        self.tokenizer = text_tokenizer
        self.Textencoder = text_encoder
        # self.vocoder = vocoder#.to(self.config["device"])
        self.simple_diffusion = SimpleDiffusion(self.config["timesteps"])
        
    def log(self, *args):
        with open(self.log_path, "a") as F:
            F.write(" ".join([str(i) for i in args]))
            F.write("\n")
        F.close()
        
    def get(self,element: torch.Tensor, t: torch.Tensor):
        """
        Get value at index position "t" in "element" and
            reshape it to have the same dimension as a batch of images.
        """
        ele = element.gather(-1, t)#extract the value from the input tensor along with the specified dimension that we want.
        return ele.reshape(-1, 1, 1)
    
    def forward_diffusion(self, minibatch):
        """for every minibatch forward difussion tkaes place inwhich loss,mean,std is returned"""
        Hdata,Edata,Eaudio, label = minibatch
       
        text_embeddings = self.Textencoder.get_text_features(**self.tokenizer(label, padding=True, return_tensors="pt"))
        # audio_embeddings = torch.zeros(1,512)
        # for i in Eaudio :
        #     a = self.audioEncoder.get_audio_features(**self.AudioFeatureExtractor(i, return_tensors="pt"))
        #     audio_embeddings = torch.cat((audio_embeddings,a),dim=0)
        # audio_embeddings = audio_embeddings[1:]
      
        audio_embeddings = self.audioEncoder.get_audio_embedding_from_data(x = Eaudio, use_tensor=True).to('cpu')
       
        # if text_embeddings.shape[-1] > audio_embeddings.shape[-1]:
        #     audio_embeddings = F.pad(audio_embeddings, (0, text_embeddings.shape[-1]-audio_embeddings.shape[-1]), value=1e5)
        # elif audio_embeddings.shape[-1] > text_embeddings.shape[-1]:
        #     text_embeddings = F.pad(text_embeddings, (0, audio_embeddings.shape[-1]-text_embeddings.shape[-1]), value=1e5)
     
        out = self.vae(torch.unsqueeze(Edata,dim=1))#.to(self.config["device"])
       
        #############_________________FOR UNET AUTOENCODER ONE ENcDecT
        #model_input1 = F.normalize(torch.cat([audio_embeddings.unsqueeze(1), text_embeddings.unsqueeze(1),out], dim=1)).unsqueeze(dim=2)
        #print(model_input1.shape)
        # ts = torch.randint(low = 1,high = self.config["timesteps"],size=(len(Edata),))
        #print(ts.shape)
        #latent_representation,model_out = self.architecture(model_input1,ts)#.to(self.config['device']).to(dtype=torch.float32))
        #model_out = F.pad((model_out),(0,4))
        ############________________FOR UNET TRANSFOMERWS ONE 
        modelInput = F.normalize(torch.cat([audio_embeddings.unsqueeze(1),out.unsqueeze(1)],dim=1))
        print(modelInput.shape)
        latent_representation,model_out = self.architecture(modelInput,text_embeddings)
        model_out = self.Clayer(model_out)
        
        model_out = F.pad(model_out,(0,885-866))
  
        loss = self.criterion(model_out, Hdata)    
        print(loss)   
        self.optimizer.zero_grad(set_to_none=True)
        # print(loss)
        loss.backward()
        self.optimizer.step()

        mean = torch.mean(latent_representation.flatten())
        std = torch.std(latent_representation.flatten())
        
        return loss,mean,std

    
    def dataloader(self):
        return DataLoader(dataset=self.dataset,
                          batch_size=self.config["batch_size"],
                          shuffle=True,
                          num_workers=4,
                          pin_memory=True)
        

    def train_1_epoch(self, epoch):

        self.architecture.train()
     
        self.criterion.train()
        
        dataloader = self.dataloader()
        
        epoch_loss, epoch_mean, epoch_std = 0, 0, 0
        
        start_time = time.time()
        
        for minibatch_idx, minibatch in enumerate(dataloader):
            
            loss,mean,std = self.forward_diffusion(minibatch=minibatch)
            epoch_loss += loss.detach().item()
            epoch_mean += mean
            epoch_std += std
            
            torch.cuda.empty_cache()
            gc.collect()
            
            self.log(f"Epoch - {epoch}",
                    f"minibatch idx - {minibatch_idx}",
                    f"Loss - {(epoch_loss/len(dataloader)):.4f}",
                    f"Mean - {(epoch_mean/len(dataloader)):.4f}",
                    f"Standarad Deviation - {(epoch_std/len(dataloader)):.4f}",
                    f"Time Taken - {((time.time()-start_time)/36000):.4f}")       
            
            print(f"Epoch - {epoch} minibatch idx - {minibatch_idx} Loss - {(epoch_loss/len(dataloader)):.4f} Training - {(100 * (minibatch_idx/dataloader.__len__())):.4f}")
                  
        normal_distribution = normal.Normal(epoch_mean/len(dataloader), epoch_std/len(dataloader))
        shape = (1, self.config["image_shape"][-2], self.config["image_shape"][-1])
        sample = normal_distribution.rsample(shape)
        
        reconst_spec = self.smallArchitecture(sample) 
                
    
        self.save_checkpoints(epoch)     
        return reconst_spec

    
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
            
                    