# Load FastPitch
import torch
import random
import torchaudio
import warnings
warnings.filterwarnings('ignore')

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
print(audio.shape)
transform = torchaudio.transforms.MelSpectrogram(n_mels=80)
mel_specgram = transform(audio) 
print(mel_specgram)
Log2MEL = torch.log2(mel_specgram)
print(Log2MEL.shape)
Log10MEL = torch.log10(mel_specgram)
print(Log10MEL.shape)
# y, sr = librosa.load('/home/earth/RITUL_NSUT/b.mp3')
# # # spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=158, hop_length=100)  
# # transform = torchaudio.transforms.Spectrogram(n_fft=158,hop_length=100)
# # spectrogram = torch.squeeze(transform(torch.tensor(y))).to('cuda')
# # print('spec shape',spectrogram.shape)

# mel_spectrogram = librosa.feature.melspectrogram(y, sr, n_mels=80)
# print(mel_spectrogram.shape)
# log_mel_spectrogram = torch.tensor(librosa.power_to_db(mel_spectrogram))
# print(log_mel_spectrogram.shape)



audio = model.convert_spectrogram_to_audio(spec=mel_specgram.to('cuda'))
print(audio.shape)
torchaudio.save('HifiGanReconstMELLLL.mp3', audio.to('cpu'), sr)



audio = model.convert_spectrogram_to_audio(spec=Log2MEL.to('cuda'))
print(audio.shape)
torchaudio.save('HifiGanReconstLOG2.mp3', audio.to('cpu'), sr)



audio = model.convert_spectrogram_to_audio(spec=Log10MEL.to('cuda'))
print(audio.shape)
torchaudio.save('HifiGanReconstLOG10.mp3', audio.to('cpu'), sr)