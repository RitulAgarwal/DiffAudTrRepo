''' Example file to test tts_infer after installing it. Refer to section 1.1 in README.md for steps of installation. '''

from tts_infer.tts import TextToMel, MelToWav
from tts_infer.transliterate import XlitEngine
from tts_infer.num_to_word_on_sent import normalize_nums
import torch
import re
import numpy as np
from scipy.io.wavfile import write

# from mosestokenizer import *
# from indicnlp.tokenize import sentence_tokenize

INDIC = ["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]

# def split_sentences(paragraph, language):
#     if language == "en":
#         with MosesSentenceSplitter(language) as splitter:
#             return splitter([paragraph])
#     elif language in INDIC:
#         return sentence_tokenize.sentence_split(paragraph, lang=language)

class StandardScaler:

    def __init__(self, mean=None, std=None, epsilon=1e-7):
        """Standard Scaler.
        The class can be used to normalize PyTorch Tensors using native functions. The module does not expect the
        tensors to be of any specific shape; as long as the features are the last dimension in the tensor, the module
        will work fine.
        :param mean: The mean of the features. The property will be set after a call to fit.
        :param std: The standard deviation of the features. The property will be set after a call to fit.
        :param epsilon: Used to avoid a Division-By-Zero exception.
        """
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def fit(self, values):
        dims = list(range(values.dim() - 1))
        self.mean = torch.mean(values, dim=dims)
        self.std = torch.std(values, dim=dims)

    def transform(self, values):
        return (values - self.mean) / (self.std + self.epsilon)

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

    def inverseTransform(self,oldMean,oldStd,Diffvalues):
        
        return Diffvalues*(oldStd+self.epsilon) + oldMean


device='cpu'
text_to_mel = TextToMel(glow_model_dir='/home/earth/RITUL_NSUT/vakyansh-tts/checkpoints/glow/glow_F', device=device)
mel_to_wav = MelToWav(hifi_model_dir='/home/earth/RITUL_NSUT/vakyansh-tts/checkpoints/hifi/hifi_F', device=device)
scaler = StandardScaler()

lang='hi' # transliteration from En to Hi
print('prob')
engine = XlitEngine(lang) # loading translit model globally

def translit(text, lang):
    reg = re.compile(r'[a-zA-Z]')
    words = [engine.translit_word(word, topk=1)[lang][0] if reg.match(word) else word for word in text.split()]
    updated_sent = ' '.join(words)
    return updated_sent
    
def run_tts(text, lang):
    text = text.replace('।', '.') # only for hindi models
    text_num_to_word = normalize_nums(text, lang) # converting numbers to words in lang
    text_num_to_word_and_transliterated = translit(text_num_to_word, lang) # transliterating english words to lang
    final_text = ' ' + text_num_to_word_and_transliterated
    mel = text_to_mel.generate_mel(final_text)
    print(mel,mel.max(),mel.min())
    
    dims = list(range(mel.dim() - 1))
    mean = torch.mean(mel, dim=dims)
    std = torch.std(mel, dim=dims)
    
    mel = scaler.fit_transform(mel)
    print(mel,mel.max(),mel.min())
    
    mel = scaler.inverseTransform(mean,std,mel)
    print(mel,mel.max(),mel.min())
    
    audio, sr = mel_to_wav.generate_wav(mel)
    write(filename='temp124444444444442.wav', rate=sr, data=audio) # for saving wav file, if needed
    return (sr, audio)

# def run_tts_paragraph(text, lang):
#     audio_list = []
#     split_sentences_list = split_sentences(text, language='hi')

#     for sent in split_sentences_list:
#         sr, audio = run_tts(sent, lang)
#         audio_list.append(audio)

#     concatenated_audio = np.concatenate([i for i in audio_list])
#     write(filename='temp_long.wav', rate=sr, data=concatenated_audio)
#     return (sr, concatenated_audio)

if __name__ == "__main__":
    # _, audio = run_tts('mera naam kuch nhi hai', 'hi')
    _,audio = run_tts('भारत मेरा देश है ','hi')
        
    # para = '''
    # भारत मेरा देश है और मुझे भारतीय होने पर गर्व है। ये विश्व का सातवाँ सबसे बड़ा और विश्व में दूसरा सबसे अधिक जनसंख्या वाला देश है।
    # इसे भारत, हिन्दुस्तान और आर्यव्रत के नाम से भी जाना जाता है। ये एक प्रायद्वीप है जो पूरब में बंगाल की खाड़ी, 
    # पश्चिम में अरेबियन सागर और दक्षिण में भारतीय महासागर जैसे तीन महासगरों से घिरा हुआ है। 
    # भारत का राष्ट्रीय पशु चीता, राष्ट्रीय पक्षी मोर, राष्ट्रीय फूल कमल, और राष्ट्रीय फल आम है। 
    # भारत मेरा देश है और मुझे भारतीय होने पर गर्व है। ये विश्व का सातवाँ सबसे बड़ा और विश्व में दूसरा सबसे अधिक जनसंख्या वाला देश है।
    # इसे भारत, हिन्दुस्तान और आर्यव्रत के नाम से भी जाना जाता है। ये एक प्रायद्वीप है जो पूरब में बंगाल की खाड़ी, 
    # पश्चिम में अरेबियन सागर और दक्षिण में भारतीय महासागर जैसे तीन महासगरों से घिरा हुआ है। 
    # भारत का राष्ट्रीय पशु चीता, राष्ट्रीय पक्षी मोर, राष्ट्रीय फूल कमल, और राष्ट्रीय फल आम है। 
    # भारत मेरा देश है और मुझे भारतीय होने पर गर्व है। ये विश्व का सातवाँ सबसे बड़ा और विश्व में दूसरा सबसे अधिक जनसंख्या वाला देश है।
    # इसे भारत, हिन्दुस्तान और आर्यव्रत के नाम से भी जाना जाता है। ये एक प्रायद्वीप है जो पूरब में बंगाल की खाड़ी, 
    # पश्चिम में अरेबियन सागर और दक्षिण में भारतीय महासागर जैसे तीन महासगरों से घिरा हुआ है। 
    # भारत का राष्ट्रीय पशु चीता, राष्ट्रीय पक्षी मोर, राष्ट्रीय फूल कमल, और राष्ट्रीय फल आम है। 
    # '''
    
    # print('Num chars in paragraph: ', len(para))
    # _, audio_long = run_tts_paragraph(para, 'hi')
