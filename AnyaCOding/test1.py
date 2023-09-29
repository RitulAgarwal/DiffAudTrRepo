from transformers import AutoFeatureExtractor, ClapModel
import torch

model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
feature_extractor = AutoFeatureExtractor.from_pretrained("laion/clap-htsat-unfused")

for i in range(4):
    random_audio = torch.rand((5,1,16000,1))
    inputs = feature_extractor(random_audio, return_tensors="pt",sampling_rate=48000)
    audio_features = model.get_audio_features(**inputs)
    print(audio_features.shape,type(audio_features))