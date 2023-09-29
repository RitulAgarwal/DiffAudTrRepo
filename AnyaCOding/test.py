import os 
import laion_clap
from torch.utils.data import DataLoader
from ENGHINDLIMPROVED import AudioDataset
from transformers import ClapModel,AutoFeatureExtractor, AutoProcessor, ClapAudioModel,ClapAudioModelWithProjection, ClapProcessor
import warnings
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"


model = laion_clap.CLAP_Module(enable_fusion=False)
# model.load_ckpt() # download the default pretrained checkpoint.

def testEMbAud(minibatch):
    _,_,E, _ = minibatch
    audio_embed = model.get_audio_embedding_from_data(x = E, use_tensor=True)

    print(audio_embed.shape)
   

def Ddataloader():
    return DataLoader(dataset=AudioDataset(),
                        batch_size=24,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=True)
     
def train_1_epoch(epoch):
    dataloader = Ddataloader()        
    for _, minibatch in enumerate(dataloader):
        testEMbAud(minibatch=minibatch)
        
for i in range(2):
    print(i)
    train_1_epoch(i)
    
print('done')


