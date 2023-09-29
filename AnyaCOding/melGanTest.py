# import torch
# import librosa
# import torchaudio
# import torch.nn as nn 
# from torchinfo import summary

# vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
# y,sr = librosa.load('/home/earth/RITUL_NSUT/b.mp3')
# # transform = torchaudio.transforms.Spectrogram(n_fft=158,hop_length=100)
# # transform = torchaudio.transforms.MelSpectrogram(n_mels=80)
# # mel_specgram = transform(torch.tensor(y)) 
# # print(mel_specgram.shape)
# # out = vocoder.inverse(mel_specgram)  # audio (torch.tensor) -> (batch_size, 80, timesteps)
# # torchaudio.save('MelGanResMelSpec.wav',out.to('cpu'),sr)

# mel_spectrogram = librosa.feature.melspectrogram(y, sr, n_mels=80)
# print(mel_spectrogram.shape)
# log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
# print(log_mel_spectrogram.shape)

# out = vocoder.inverse(torch.tensor(log_mel_spectrogram))  # audio (torch.tensor) -> (batch_size, 80, timesteps)

# speed_low = torchaudio.transforms.Speed(sr, 0.5)
# out = speed_low(out.to('cpu'))

# torchaudio.save('MelGanResLogMelSpecSRSLOW.wav',out[0].to('cpu'),sr)



# # spectrogram = transform(torch.tensor(y))
# # print(spectrogram.shape)
# # out = vocoder.inverse(spectrogram)  # audio (torch.tensor) -> (batch_size, 80, timesteps)
# # torchaudio.save('MelGanRes.wav',out.to('cpu'),sr*2)

import torch
import torchaudio
# vocoder = torch.hub.load('descriptinc/melgan-neurips', model='load_melgan',force_reload=True)
# vocoder.inverse(audio)  # audio (torch.tensor) -> (batch_size, 80, timesteps)'
audio,sr = torchaudio.load('/home/earth/DATASETS/DATASET_BPCC_AUDIO/English/comparable_0.wav')

transform = torchaudio.transforms.MelSpectrogram(n_mels=80)
mel_specgram = transform(audio) 

Log2MEL = torch.log2(mel_specgram)

Log10MEL = torch.log10(mel_specgram)

vocoder = torch.hub.load('seungwonpark/melgan', 'melgan')
vocoder.eval()

if torch.cuda.is_available():
    vocoder = vocoder.cuda()
    mel = mel_specgram.cuda()
    log2mel = Log2MEL.cuda()
    log10mel = Log10MEL.cuda()

with torch.no_grad():
    audio = vocoder.inference(log2mel)
    audio10 = vocoder.inference(log10mel)
print(audio.shape)
torchaudio.save('MelGanLOG2MELSPEC.wav',audio.unsqueeze(0).to('cpu'),sr)

torchaudio.save('MelGanLOG10MELSPEC.wav',audio10.unsqueeze(0).to('cpu'),sr)