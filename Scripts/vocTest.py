import torch,torchaudio,os,math,random
import torch.nn.functional as F
from transformers import SpeechT5HifiGan
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
# aud = '/home/earth/DATASETS/DATASET_BPCC_AUDIO/Hindi/massive_223.wav'
aud = '/home/earth/DATASETS/DATASET_BPCC_AUDIO/English/comparable_102.wav'

def load_audio_new(path:str,
               duration:int=2,
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
    # if loading_backend == "librosa":
    #     audio, sr = librosa.load(path)
    #     audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    # if loading_backend == "soundfile":
    #     audio, sr = soundfile.read(path)
    #     audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        

    max_frames = duration * sample_rate

    if sample_rate != sr:
        resampler = torchaudio.transforms.Resample(sr, sample_rate, dtype=audio.dtype)
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


path = '/home/earth/RITUL_NSUT/Architectures/TEST'
if not os.path.exists(path):
    os.makedirs(path)


wave, s2 = load_audio_new(path=aud.strip(), duration=5,sample_rate=16000,normalization=False)
torchaudio.save((path+'EEinput.wav'), wave.unsqueeze(0), 16000)

transform = torchaudio.transforms.MelSpectrogram(n_fft = 512,n_mels=80)
TAMmelSpectrogram = transform(wave).unsqueeze(0)
print(TAMmelSpectrogram.shape)

out = vocoder(TAMmelSpectrogram.permute(0,2,1))
print(out.shape)
 
torchaudio.save((path+'EEoutput.wav'), out, 16000)
# print(wave.shape)

