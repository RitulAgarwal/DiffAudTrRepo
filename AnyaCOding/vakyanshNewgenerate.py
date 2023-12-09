from tts_infer.tts import TextToMel, MelToWav
from tts_infer.transliterate import XlitEngine
from tts_infer.num_to_word_on_sent import normalize_nums

import re
from scipy.io.wavfile import write

text_to_mel = TextToMel(glow_model_dir='/home/earth/RITUL_NSUT/vakyansh-tts/checkpoints/glow/glow_F')
mel_to_wav = MelToWav(hifi_model_dir='/home/earth/RITUL_NSUT/vakyansh-tts/checkpoints/hifi/hifi_F')

def translit(text, lang):
    reg = re.compile(r'[a-zA-Z]')
    engine = XlitEngine(lang)
    words = [engine.translit_word(word, topk=1)[lang][0] if reg.match(word) else word for word in text.split()]
    updated_sent = ' '.join(words)
    return updated_sent
    
def run_tts(text, lang):
    text = text.replace('।', '.') # only for hindi models
    text_num_to_word = normalize_nums(text, lang) # converting numbers to words in lang
    text_num_to_word_and_transliterated = translit(text_num_to_word, lang) # transliterating english words to lang
    
    mel = text_to_mel.generate_mel(text_num_to_word_and_transliterated)
    audio, sr = mel_to_wav.generate_wav(mel)
    write(filename='/home/earth/RITUL_NSUT/vakyansh-tts/temp.wav', rate=sr, data=audio) # for saving wav file, if needed
    return (sr, audio)

print(run_tts('यह एक सिंथेटिक भाषण है','hi'))
