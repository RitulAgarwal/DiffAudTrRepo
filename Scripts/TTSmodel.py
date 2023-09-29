import torch 
import numpy as np
import os 
import sys 
sys.path.append("./")
import warnings
warnings.filterwarnings('ignore')
from VAKYANSH import utils
from VAKYANSH import models 
from VAKYANSH.Text import text_to_sequence

def check_directory(dir):
    if not os.path.exists(dir):
        sys.exit("Error: {} directory does not exist".format(dir))


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


class TextToMel:
    def __init__(self, glow_model_dir, device="cuda"):
        self.glow_model_dir = glow_model_dir
        check_directory(self.glow_model_dir)
        self.device = device
        self.hps, self.glow_tts_model = self.load_glow_tts()

    def load_glow_tts(self):
        hps = utils.get_hparams_from_dir(self.glow_model_dir)
        checkpoint_path = utils.latest_checkpoint_path(self.glow_model_dir)
        symbols = list(hps.data.punc) + list(hps.data.chars)
        
        glow_tts_model = models.FlowGenerator(
            len(symbols) + getattr(hps.data, "add_blank", False),
            out_channels=hps.data.n_mel_channels,
            **hps.model
        )  # .to(self.device)
        #summary(glow_tts_model)
        if self.device == "cuda":
            glow_tts_model.to("cuda")

        utils.load_checkpoint(checkpoint_path, glow_tts_model)
        glow_tts_model.decoder.store_inverse()
        _ = glow_tts_model.eval()

        return hps, glow_tts_model

    def generate_mel(self, text, noise_scale=0.667, length_scale=1.0):
        # print(f"Noise scale: {noise_scale} and Length scale: {length_scale}")
        symbols = list(self.hps.data.punc) + list(self.hps.data.chars)
        cleaner = self.hps.data.text_cleaners
        if getattr(self.hps.data, "add_blank", False):
            text_norm = text_to_sequence(text, symbols, cleaner)
            text_norm = intersperse(text_norm, len(symbols))
        else:  # If not using "add_blank" option during training, adding spaces at the beginning and the end of utterance improves quality
            text = " " + text.strip() + " "
            text_norm = text_to_sequence(text, symbols, cleaner)

        sequence = np.array(text_norm)[None, :]

        del symbols
        del cleaner
        del text
        del text_norm

        if self.device == "cuda":
            x_tst = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
            x_tst_lengths = torch.tensor([x_tst.shape[1]]).cuda()
        else:
            x_tst = torch.autograd.Variable(torch.from_numpy(sequence)).long()
            x_tst_lengths = torch.tensor([x_tst.shape[1]])

        with torch.no_grad():
            (y_gen_tst, *_), *_, (attn_gen, *_) = self.glow_tts_model(
                x_tst,
                x_tst_lengths,
                gen=True,
                noise_scale=noise_scale,
                length_scale=length_scale,
            )
        del x_tst
        del x_tst_lengths
        torch.cuda.empty_cache()
        return y_gen_tst.cpu().detach().numpy()



class VakyanshModel:
    def __init__(self,glow= '/home/earth/RITUL_NSUT/VAKYANSH/checkpoints/hinglow/female',hifi='/home/earth/RITUL_NSUT/VAKYANSH/checkpoints/hinhifi/female',device = 'cpu'):
        self.model = TextToMel(glow_model_dir=glow, device=device)
        
TTSmodel = VakyanshModel().model
print(TTSmodel.glow_tts_model)


# mel = TTSmodel.generate_mel(text, noiseScale, LengthScale)

# audio, sr = mel_to_wav.generate_wav(mel)
# pitch = torchaudio.transforms.PitchShift(sr, n_steps=random.randint(-6, 2))
# audio = pitch(audio.cpu().detach().unsqueeze(0))
# torchaudio.save(outputPath, audio.to(torch.int16), sr)

        