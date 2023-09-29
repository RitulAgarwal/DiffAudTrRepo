from datasets import load_dataset
from transformers import AutoProcessor, ClapAudioModel

dataset = load_dataset("ashraq/esc50")
audio_sample = dataset["train"]["audio"][0]["array"]
print(audio_sample.shape,'input')
model = ClapAudioModel.from_pretrained("laion/clap-htsat-fused")
processor = AutoProcessor.from_pretrained("laion/clap-htsat-fused")

inputs = processor(audios=audio_sample, return_tensors="pt")
print(inputs['input_features'],'model in')
outputs = model(**inputs)
print(outputs['pooler_output'].shape,'model out')
last_hidden_state = outputs.last_hidden_state
print(last_hidden_state,type(last_hidden_state))
print(last_hidden_state.shape)