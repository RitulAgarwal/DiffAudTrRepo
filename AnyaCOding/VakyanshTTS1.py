''' Example file to test tts_infer after installing it. Refer to section 1.1 in README.md for steps of installation. '''

from tts_infer.tts import MelToWav
from tts_infer.num_to_word_on_sent import normalize_nums
import torch
import re
import torchaudio
import math 
import random
import numpy as np
import torch.nn.functional as F
from scipy.io.wavfile import write


class StandardScaler:

    def __init__(self, mean=None, std=None, epsilon=1e-7):
        """Standard Scaler.
        The class can be used to normalize PyTorch Tensors using native functions. The module does not expect the
        tensors to be of any specific shape; as long as the features are the last dimension in the tensor, the module
        will work fine.
        :param mean: The mean of the features. The property will be set after a call to fit.
        :param std: The standard deviation of the features. The property will be set after a call to fit.
        :param epsilon: Used to avoid a Division-By-Zero exception.
        """
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def fit(self, values):
        dims = list(range(values.dim() - 1))
        self.mean = torch.mean(values, dim=dims)
        self.std = torch.std(values, dim=dims)
        return self.mean,self.std

    def transform(self, values):
        return (values - self.mean) / (self.std + self.epsilon)

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)
    
    def inverseTransform(self,oldMean,oldStd,Diffvalues):
        
        return Diffvalues*(oldStd+self.epsilon) + oldMean


device='cpu'
mel_to_wav = MelToWav(hifi_model_dir='/home/earth/RITUL_NSUT/vakyanshTts/checkpoints/hifi/hifi_F', device=device)

    
def run_tts(mel):

    print("||||||||||||||||||||||||||||||||||||||||",mel,mel.max(),mel.min())
    audio, sr = mel_to_wav.generate_wav(mel)
    print("]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]")
    
    print(type(audio),audio.shape,'ppppp')
    write(filename='temp212.wav', rate=sr, data=audio) # for saving wav file, if needed
    return (sr, audio)

def load_audio_new(path:str,
               duration:int=4,
               full:bool=False,
               sample_rate:int=16000,
               normalization:bool=True,
               audio_channel:str="mono",
               loading_backend:str="torchaudio",
               audio_concat_srategy:str="repeat") -> torch.Tensor:

    if loading_backend not in ["torchaudio", "librosa", "soundfile"]:
        raise Exception(f"Only implemented for (torchaudio, librosa, soundfile)")
    if audio_channel != "mono":
        raise Exception("Not Implemented")
    if audio_concat_srategy not in ["flip_n_join", "repeat"]:
        raise Exception(f"Only implemented for (random_concat, flip_n_join, repeat)")

    if loading_backend == "torchaudio":
        audio, sr = torchaudio.load(path)
  
    max_frames = duration * sample_rate

    if sample_rate != sr:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        audio = resampler(audio)
    else: pass
    
    if full: 
        if normalization:
            audio = torch.nn.functional.normalize(audio)
        else: pass
        return audio, sample_rate
    
    if audio.shape[1] < max_frames:
        if audio_concat_srategy == "flip_n_join":
            audio = torch.cat([audio, audio.flip((1,))]*int(max_frames/audio.shape[1]), dim=1)[0][:max_frames]
        if audio_concat_srategy == "repeat":
            audio = torch.tile(audio, (math.ceil(max_frames/audio.shape[1]),))[0][:max_frames]   
    else:
         start = random.randint(0, audio.shape[1]-max_frames + 1)
         audio = audio[0][start:start+max_frames]

    if audio.shape[-1] != max_frames:
        audio = F.pad(audio, (0, max_frames-audio.shape[-1]))
         
    if normalization:
        if len(audio.shape) == 1:
            audio = torch.nn.functional.normalize(audio.unsqueeze(0))
        else:
            audio = torch.nn.functional.normalize(audio)

    return audio, sample_rate


if __name__ == "__main__":
    # link = '/home/earth/DATASETS/DATASET_BPCC_AUDIO/Hindi/comparable_109.wav'	
    link = '/home/earth/DATASETS/DATASET_BPCC_AUDIO/Hindi/comparable_10.wav'
    # link = '/home/earth/RITUL_NSUT/DATASET/cvss_c_ta_en_v1.0/train/common_voice_ta_19093453.mp3.wav'	
#   
    # scaler = StandardScaler()
    # audio,sr = load_audio_new(link)
    # transform = torchaudio.transforms.MelSpectrogram(n_fft = 512,n_mels=80)
    # TAMmelSpectrogram = transform(audio)

    waveform, sample_rate = load_audio_new(link)
    transform = torchaudio.transforms.MelSpectrogram(n_mels = 80)
    # mel_specgram = torch.log10(transform(waveform) +1e-9)
    
    mel_specgram = torch.log10(torch.squeeze(transform(waveform))) 
    print(mel_specgram.shape,type(mel_specgram),'kya problem h bahi')
    # print(mel_specgram.shape)
    # mu,std = scaler.fit(mel_specgram)
    # mel_specgram = scaler.fit_transform(mel_specgram)
    # mel_specgram = scaler.inverseTransform(mu,std,mel_specgram)
    # _, audio = run_tts('mera naam kuch nhi hai', 'hi')
    sr,audio = run_tts(mel_specgram)
        
