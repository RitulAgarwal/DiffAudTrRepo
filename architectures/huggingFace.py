import laion_clap
import torch
import torchaudio
import torch.nn as nn 
from diffusers.models import AutoencoderKL
from transformers import ClapModel,AutoTokenizer
# from nemo.collections.tts.models import HifiGanModel

class AudioEncoder:
    def __init__(self):
        self.audioEncoder = laion_clap.CLAP_Module(enable_fusion=False,device='cpu')
    
class Vae: 
    def __init__(self):
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        
        vae.encoder.conv_in = nn.Conv2d(1, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        vae.encoder.conv_out = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=(4, 1),dilation = (3,1)),
            nn.Linear(110,1152)
)
        self.encoder = vae.encoder
        
        vae.decoder.conv_in = nn.Sequential(
            nn.Conv2d(1, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Linear(1152,110)
)
        vae.decoder.conv_out = nn.Conv2d(128,80, kernel_size=(7, 3),stride=(2,1),padding=(0,3))
  
        self.decoder = vae.decoder
                         
class TextTokenizer:
    def __init__(self):
        self.Texttokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")
         
class TextEncoder:
    def __init__(self):
        self.Textencoder = ClapModel.from_pretrained("laion/clap-htsat-unfused")
         
class Vocoder:
    def __init__(self):
        self.vocoder = HifiGanModel.from_pretrained(model_name="nvidia/tts_hifigan")
 
 
# class Vocoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(80,1,3),
#             nn.ReLU()
#         )
#         self.flatten = nn.Flatten()
        
#     def forward(self,x):
#         return self.flatten(self.net(x))
    
# from transformers import SpeechT5HifiGan
# vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
# speech = vocoder(torch.randn((80,885)))
# print(speech.shape)

# from waveglow_vocoder import WaveGlowVocoder

# WV = WaveGlowVocoder()
# wav = WV.mel2wav(torch.randn((1,80,885)))
# print(wav.shape)

# from hifigan_vocoder import hifigan

# model = hifigan() # dataset in ['uni', 'vctk']; device in ['cpu', 'cuda']; checkpoint will be downloaded from google driver.
# # print(model.model)
# audio = torch.tensor(model.model(torch.randn((5,80,885))))
# print(audio.shape)
# # print(torch.tensor(audio).shape) # (527872,)]

# # for sno,i in enumerate(audio) : 
# #     print(sno)
# #     torchaudio.save(('/home/earth/RITUL_NSUT/Architectures/'+str(sno)+'out.wav'), i, 48000)
   