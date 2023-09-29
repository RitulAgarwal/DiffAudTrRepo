# Load FastPitch
import torch
import random
import librosa
import torchaudio
import warnings
warnings.filterwarnings('ignore')
import soundfile as sf 

from nemo.collections.tts.models import HifiGanModel
# from nemo.collections.tts.models import FastPitchModel

def load_audio (audio,sample_rate, duration=3, full=False) :
    # print(sample_rate) 22050
    max_frames = duration * sample_rate
    audio = torch.unsqueeze(torch.from_numpy(audio),dim = 0)
    if full: 
        if audio.shape[1] < max_frames:
            audio = torch.cat([audio, audio.flip((1,))]*int(max_frames/audio.shape[1]), dim=1)[0][:max_frames]
        return audio, sample_rate
    
    if audio.shape[1] < max_frames:
        audio = torch.cat([audio, audio.flip((1,))]*int(max_frames/audio.shape[1]), dim=1)[0][:max_frames]
    else:
         start = random.randint(0, audio.shape[1]-max_frames)
         audio = audio[0][start:start+max_frames]
    return audio.unsqueeze(0), sample_rate


model = HifiGanModel.from_pretrained(model_name="nvidia/tts_hifigan")
audio,sr = torchaudio.load('/home/earth/DATASETS/DATASET_BPCC_AUDIO/English/comparable_0.wav')

transform = torchaudio.transforms.MelSpectrogram(n_mels=80)
mel_specgram = transform(audio) 
Log2MEL = torch.log2(mel_specgram)

audio = model.convert_spectrogram_to_audio(spec=Log2MEL.to('cuda'))
print(audio.shape)
torchaudio.save('HifiGanReconstLOG2.mp3', audio.to('cpu'), sr)

noise = torch.rand(mel_specgram.shape)#random numbers from a uniform distribution on the interval [0,1)
print(noise.shape)

noisySpec = mel_specgram + noise
print(noisySpec.shape)


audio = model.convert_spectrogram_to_audio(spec=mel_specgram.to('cuda'))
print(audio.shape)
torchaudio.save('HifiGanReconstMEL.mp3', audio.to('cpu'), sr)

##HIFIGAN WITH LOG2 GOOD RECONSTT