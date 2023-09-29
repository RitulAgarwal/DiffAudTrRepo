# Load FastPitch
import torch
import random
import librosa
import torchaudio
import soundfile as sf 

from nemo.collections.tts.models import HifiGanModel

model = HifiGanModel.from_pretrained(model_name="nvidia/tts_hifigan")


y, sr = librosa.load('/home/earth/RITUL_NSUT/DDPM/b.mp3')
print(y.shape,sr)
# # waveform, s = load_audio(audio = y,sample_rate=sr, duration=3)#sr is 44100
#spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=158, hop_length=100)  
#y,sr = torchaudio.load('/home/earth/RITUL_NSUT/b.mp3')
transform = torchaudio.transforms.Spectrogram(n_fft=158,hop_length=100)
spectrogram = torch.unsqueeze(transform(torch.tensor(y)), dim = 0)
print('spec shape',spectrogram.shape)

# parsed = spec_generator.parse("You can type your sentence here to get nemo to produce speech.")
# spectrogram = spec_generator.generate_spectrogram(tokens=parsed)
audio = model.convert_spectrogram_to_audio(spec=spectrogram.to('cuda:0'))
print(audio.shape)
sf.write("test1.wav", audio, sr)
#torchaudio.save('path.mp3', audio, sr)