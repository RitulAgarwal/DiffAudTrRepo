import torch
import warnings
import argparse
from VakyanshTTS import TTS 


warnings.filterwarnings("ignore")

########################################################################################
parser = argparse.ArgumentParser(description="Text To Speech")
parser.add_argument("--file_number", type=int)
parser.add_argument("--device", type=int)
arguments = parser.parse_args()
########################################################################################

Hindi = "/home/earth/RITUL_NSUT/HindiDevnagri"
English = "/home/earth/RITUL_NSUT/EnglishLatin"


HindiPathGlow = '/home/earth/RITUL_NSUT/TTS/checkpoints/hinglow/female'
HindiPathHifi = '/home/earth/RITUL_NSUT/TTS/checkpoints/hinhifi/female'
EngPathGlow = '/home/earth/RITUL_NSUT/TTS/checkpoints/hinglowF/female'
EngPathHifi = '/home/earth/RITUL_NSUT/TTS/checkpoints/hinhigiF/female'

SavePath = "/home/earth/DATASETS/DATASET_BPCC_AUDIO"

Device = torch.device(f"cuda:{arguments.device}" if arguments.device < torch.cuda.device_count() else "cpu")



with open(Hindi, "r") as F: H = F.readlines(); F.close()
with open(English, "r") as F: E = F.readlines(); F.close()

H = [i.replace("\n", "") for i in H][arguments.file_number]
E = [i.replace("\n", "") for i in E][arguments.file_number]


def lines_generator(path):
    with open(path, "r") as F:
        L = F.readlines()
        for i in L:
            yield i


# hgen = lines_generator(H)
# hcounter = 0
# for i in hgen:
#     TTS(HindiPathGlow,
#         HindiPathHifi,
#         i,
#         outputPath=SavePath+"/Hindi/"+str(H.split("/")[-3])+f"_{hcounter}.wav",
#         noiseScale=0.5,
#         LengthScale=0.78,
#         device=Device)
#     hcounter += 1
    
    
egen = lines_generator(E)
ecounter = 0
for j in egen:
    TTS(EngPathGlow,
        EngPathHifi,
        j,
        outputPath=SavePath+"/English/"+str(E.split("/")[-3])+f"_{ecounter}.wav",
        noiseScale=0.3,
        LengthScale=0.7,
        device=Device)
    ecounter += 1
