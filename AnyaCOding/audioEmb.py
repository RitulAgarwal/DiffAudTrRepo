import torch
import torchaudio
from torchaudio.transforms import Resample
from transformers import RobertaTokenizer, RobertaModel
from transformers import Wav2Vec2FeatureExtractor, UniSpeechSatForXVector



feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/unispeech-sat-base-sv')
model = UniSpeechSatForXVector.from_pretrained('microsoft/unispeech-sat-base-sv')
text_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
text_encoder = RobertaModel.from_pretrained('roberta-base')
# sig,sr = torchaudio.load('audio1.wav')
sig,sr = torchaudio.load('/home/earth/RITUL_NSUT/DATASET/FSDKaggle2018.audio_train/ffd52400.wav')
print(sig.shape,sr)
text = 'dkwjfer oerihf'
target_sample_rate = 16000
resample_transform = Resample(orig_freq=sr, new_freq=target_sample_rate)
sig_new = resample_transform(sig)
print(sig.shape,target_sample_rate)
inputs = feature_extractor(sig_new,sampling_rate = target_sample_rate,return_tensors = 'pt')
inP = torch.squeeze(inputs['input_values'],dim=0) # used instead of **inputs
audio_embeddings = model(inP).embeddings
audio_embeddings = torch.nn.functional.normalize(audio_embeddings, dim=-1).cpu()
print(audio_embeddings.shape)
tokens = text_tokenizer(text, return_tensors="pt", padding=True)
text_embeddings = text_encoder(**tokens)[1]
print(text_embeddings.shape)