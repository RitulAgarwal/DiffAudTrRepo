import torch
import torchaudio 
import torch.nn as nn 
import matplotlib.pyplot as plt 

audio,sr = torchaudio.load('/home/earth/DATASETS/DATASET_BPCC_AUDIO/English/comparable_0.wav')
print(audio.shape,sr)
layer1 = nn.Conv1d(1,4,11,dilation=1)

audio_re = layer1(audio)
print(audio_re.shape,type(audio_re))
Summedaudio = (audio_re[0] + audio_re[1]+ audio_re[2]+ audio_re[3]).unsqueeze(0)
print(Summedaudio.shape)
torchaudio.save('AudiosAnalys/SumAud.wav',Summedaudio,sr)
AvgAudio = ((audio_re[0] + audio_re[1]+ audio_re[2]+ audio_re[3])/4).unsqueeze(0)
print(AvgAudio.shape)
torchaudio.save('AudiosAnalys/AvgAud.wav',AvgAudio,sr)
m = nn.MaxPool1d(2)
MaxPooledAud = m(audio)
print(MaxPooledAud.shape)
torchaudio.save('AudiosAnalys/MaxPoolAud.wav',MaxPooledAud,int(sr/2))

# audio = 0
# print(audio.shape)
# for i in audio_re:
#     print(i.shape)
#     audio+=i
# # audioF = audio_re
# print(audio.shape)

