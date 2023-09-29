import torch
import random
import torchaudio
import warnings
warnings.filterwarnings('ignore')
import soundfile as sf 
from VakyanshTTS import MelToWav
import math 

# from nemo.collections.tts.models import HifiGanModel
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


audio,sr = torchaudio.load('/home/earth/DATASETS/DATASET_BPCC_AUDIO/English/comparable_0.wav')
print(audio.shape,sr)
transform = torchaudio.transforms.MelSpectrogram(n_mels=80)
mel_specgram = transform(audio) 
print(mel_specgram.shape)
Log2MEL = torch.log2(mel_specgram)
print(Log2MEL.shape)
Log10MEL = torch.log10(mel_specgram)
print(Log10MEL.shape)
# log_mel_spectrogram = torch.tensor(librosa.power_to_db(mel_specgram.to('cuda:0')))
# print(log_mel_spectrogram.shape)

# y, sr = librosa.load('/home/earth/DATASETS/DATASET_BPCC_AUDIO/English/comparable_0.wav')
# print(y.shape)
# Mspectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=158, hop_length=100)  
# transform = torchaudio.transforms.Spectrogram(n_fft=158,hop_length=100)
# spectrogram = torch.squeeze(transform(torch.tensor(y))).to('cuda')
# print('spec shape',spectrogram.shape)

# mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80)
# print(mel_spectrogram.shape)
# log_mel_spectrogram = torch.tensor(librosa.power_to_db(mel_spectrogram))
# print(log_mel_spectrogram.shape)

vocoder = MelToWav(hifi_model_dir='/home/earth/RITUL_NSUT/TTS/checkpoints/hinhifi/female', device=torch.device('cuda:0'))
o,sr = vocoder.generate_wav(mel_specgram)
# audio = model.convert_spectrogram_to_audio(spec=mel_specgram.to('cuda'))
print(o.shape,sr)
# speed_low = torchaudio.transforms.Speed(sr,0.5)
# audio = speed_low(audio.to('cpu'))

torchaudio.save('VAKYANSH_MELSPEC.wav', o.unsqueeze(0).to('cpu'), sr)


vocoder = MelToWav(hifi_model_dir='/home/earth/RITUL_NSUT/TTS/checkpoints/hinhifi/female', device=torch.device('cuda:0'))
o,sr = vocoder.generate_wav(mel_specgram)
# audio = model.convert_spectrogram_to_audio(spec=mel_specgram.to('cuda'))
print(o.shape,sr)
# speed_low = torchaudio.transforms.Speed(sr,0.5)
# audio = speed_low(audio.to('cpu'))

torchaudio.save('VAKYANSH_MELSPEC.wav', o.unsqueeze(0).to('cpu'), sr)


noise = torch.rand(mel_specgram.shape)#random numbers from a uniform distribution on the interval [0,1)
print(noise.shape)

noisySpec = mel_specgram + noise
o,sr = vocoder.generate_wav(noisySpec)
torchaudio.save('VAKYANSH_MELSPECNOISE.wav', o.unsqueeze(0).to('cpu'), sr)


noisySpecLOg2 = Log2MEL + noise 
o,sr = vocoder.generate_wav(noisySpecLOg2)
torchaudio.save('VAKYANSH_MELLOG2NOISE.wav', o.unsqueeze(0).to('cpu'), sr)

