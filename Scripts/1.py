import sys
import torch
import laion_clap
sys.path.append("./"), sys.path.append("../")
import torch.nn as nn 
from diffusers import AudioLDMPipeline
from torch.utils.data import DataLoader
from diffusers.models import AutoencoderKL
from Dataloaders.HinEngDL import AudioDatasetEng
from transformers import ClapModel,AutoTokenizer
from Architectures.CUnet_V1EMB import Autoencoder

AudioDataSet = AudioDatasetEng()

dataloader =  DataLoader(dataset=AudioDataSet,
                          batch_size=6,
                          shuffle=True,
                          num_workers=4,
                          pin_memory=True)

# minibatch has HINspectrogram, ENGspectrogram,Engwaveform,EngText
# of shape (BS,80,885)H,E and (BS,192000) and text 


class VAE: 
    def __init__(self):
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        vae.encoder.conv_in = nn.Conv2d(1, 128, kernel_size=(3, 3))
        vae.encoder.conv_out = nn.Conv2d(512, 1, kernel_size=(3,1)) 
        self.vae = vae
 
class AudioEncoder:
    def __init__(self):
        self.audioEncoder = laion_clap.CLAP_Module(enable_fusion=False,device='cpu')

class TextTokenizer:
    def __init__(self):
        self.Texttokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")
         
class TextEncoder:
    def __init__(self):
        self.Textencoder = ClapModel.from_pretrained("laion/clap-htsat-unfused")
         

text_tokenizer = TextTokenizer().Texttokenizer
text_encoder =TextEncoder().Textencoder
audioEncoder =AudioEncoder().audioEncoder
vae = VAE().vae
ae = Autoencoder()
print(vae)

# repo_id = "cvssp/audioldm-s-full-v2"
# pipe = AudioLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
# unet = pipe.unet
# print(unet)
# for minibatch_idx, minibatch in enumerate(dataloader):
#     HS,ES,EW,ET = minibatch
#     textEMBEDDING = text_encoder.get_text_features(**text_tokenizer(ET, padding=True, return_tensors="pt"))
#     audioEMBEDDING = audioEncoder.get_audio_embedding_from_data(x = EW, use_tensor=True).to('cpu')    
#     print(textEMBEDDING.shape,audioEMBEDDING.shape)
   
   
#     lr,output = ae(audioEMBEDDING)#,torch.randn((6)))
#     print(lr.shape,output.shape)


#     # out = vae(torch.unsqueeze(HS,dim=1)).squeeze()
#     # print(out.shape) 
      
#     break

