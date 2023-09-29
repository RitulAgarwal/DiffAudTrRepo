import torchaudio
import torch, random,math,librosa
import warnings 
import torch.nn.functional as F 
warnings.filterwarnings('ignore')
from torch.utils.data import Dataset,DataLoader

class AudioDatasetEng(Dataset):
    """
    Dataloader
    """
    def __init__(self,
                 Hinfile_path:str='/home/earth/RITUL_NSUT/DATASET/HindiCommonAudios',
                 Engfile_path:str='/home/earth/RITUL_NSUT/DATASET/EnglishCommonAudios1',
                 duration:int=4,
                 tag:str="melspectrogram"):
        if tag not in ["spectrogram", "melspectrogram"]:
            raise Exception(f"{tag} not implemented")
        self.duration = duration
        self.tag = tag   
       
        with open(Hinfile_path, "r") as F: H = F.readlines(); F.close()
        with open(Engfile_path, "r") as F: E = F.readlines(); F.close()
        self.HindiAudList = H 
        self.EnglishAudList = E 
        
    def __len__(self):
        return len(self.HindiAudList)
         
    def __getitem__(self, index):        
        Hinaudio_path = self.HindiAudList[index]
        EngAudioTrans = self.EnglishAudList[index]
        AudioTrans = EngAudioTrans.split("||")
        hinwaveform, _ = load_audio_new(path=Hinaudio_path.strip(), duration=self.duration,sample_rate=48000,normalization=False)#sr is 44100
        Engwaveform, _ = load_audio_new(path=AudioTrans[0].strip(), duration=self.duration,sample_rate=48000,normalization=False)
    
        if self.tag == "spectrogram":
            transform = torchaudio.transforms.Spectrogram(n_fft=158,hop_length=100)
            HINspectrogram = torch.log10(torch.squeeze(transform(hinwaveform))+1e-9)
            ENGspectrogram = torch.log10(torch.squeeze(transform(Engwaveform))+1e-9)
            return HINspectrogram, ENGspectrogram,Engwaveform,AudioTrans[1]
            
        elif self.tag == "melspectrogram":
            transform = torchaudio.transforms.MelSpectrogram( n_fft = 512,hop_length = 217,n_mels=80)
            HINmelSpectrogram = torch.log10(torch.squeeze(transform(hinwaveform))+1e-9)
            ENGmelSpectrogram = torch.log10(torch.squeeze(transform(Engwaveform))+1e-9)
            return HINmelSpectrogram, ENGmelSpectrogram,Engwaveform,AudioTrans[1]
            
        else:
            return hinwaveform, Engwaveform, AudioTrans[1]


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
    if loading_backend == "librosa":
        audio, sr = librosa.load(path)
        audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

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


# ad = AudioDatasetEng()
# dataloader = DataLoader(ad,5)
# for i in enumerate(dataloader):
#     print(i)
#     _,(a,b,c,d) = i 
#     print(a.shape,b.shape,c.shape)
#     break