import librosa
import torchaudio
import torch, random
import warnings, math
import torch.nn.functional as F
warnings.filterwarnings('ignore')
from torch.utils.data import Dataset,DataLoader
from torchaudio.transforms import Resample

class AudioDatasetTamil(Dataset):
    """
    Dataloader
    """
    def __init__(self,
                 path = '/home/earth/RITUL_NSUT/DATASET/cvss_c_ta_en_v1.0/train.tsv',
                 duration:int=6,
                 tag:str="melspectrogram"):
        if tag not in ["spectrogram", "melspectrogram"]:
            raise Exception(f"{tag} not implemented")
        self.duration = duration
        self.tag = tag   
        with open(path, "r") as F: H = F.readlines(); F.close()
        Audio_paths = []
        EngText = []
        for i in H :
            Audio_paths.append(i.split('	')[0])
            EngText.append(i.split('	')[1])
        self.AudioPaths = Audio_paths
        self.Text = EngText
        
    def __len__(self):
        return len(self.AudioPaths)
         
    def __getitem__(self, index):        
        ENGaudio_path = '/home/earth/RITUL_NSUT/DATASET/cvss_c_ta_en_v1.0/train/' + self.AudioPaths[index]+ '.wav'
        TAMaudio_path = '/home/earth/RITUL_NSUT/DATASET/ta/clips/' + self.AudioPaths[index] 

        Tamwaveform, s1 = load_audio_new(path=TAMaudio_path.strip(), duration=self.duration,sample_rate=22000,normalization=False)#sr is 44100
        Engwaveform, s2 = load_audio_new(path=ENGaudio_path.strip(), duration=self.duration,sample_rate=22000,normalization=False)
        if self.tag == "spectrogram":
            transform = torchaudio.transforms.Spectrogram(n_fft=158,hop_length=100)
            TAMspectrogram = torch.log10(torch.squeeze(transform(Tamwaveform))+1e-3)
            ENGspectrogram = torch.log10(torch.squeeze(transform(Engwaveform))+1e-3)
            return TAMspectrogram, ENGspectrogram,Tamwaveform,Engwaveform,self.Text[index]
            
        elif self.tag == "Fmelspectrogram":
            transform = torchaudio.transforms.MelSpectrogram( n_fft = 650,hop_length = 271,n_mels=80)
            TAMmelSpectrogram = torch.squeeze(transform(Tamwaveform))
            ENGmelSpectrogram = torch.squeeze(transform(Engwaveform))
            return TAMmelSpectrogram[:,:-1], ENGmelSpectrogram[:,:-1],Tamwaveform,Engwaveform,self.Text[index]
        
        
        elif self.tag == "melspectrogram":
            transform = torchaudio.transforms.MelSpectrogram(n_fft = 512,n_mels=80)
            TAMmelSpectrogram = torch.squeeze(transform(Tamwaveform))
            ENGmelSpectrogram = torch.squeeze(transform(Engwaveform))
            return TAMmelSpectrogram, ENGmelSpectrogram,F.pad(Tamwaveform,(0,96)),Engwaveform,self.Text[index]
                

        else:
            return Tamwaveform, Engwaveform, self.Text[index]


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

