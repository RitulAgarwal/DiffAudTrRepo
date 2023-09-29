import math 
import numpy
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from transformers import AutoTokenizer, RobertaModel
import warnings 
warnings.filterwarnings('ignore')
###FOR SPEECH TO TEXT TRASNFOER 
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained("roberta-base")
#decoder is autoregressively used for average number of words in sentences,

class PositionalEmbedding(nn.Module):
    def __init__(self,max_seq_len,embed_model_dim):
        """
        Args:
            seq_len: length of input sequence
            embed_model_dim: demension of embedding
        """
        super(PositionalEmbedding, self).__init__()
        self.embed_dim = embed_model_dim

        pe = torch.zeros(max_seq_len,self.embed_dim)
        for pos in range(max_seq_len):
            for i in range(0,self.embed_dim,2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/self.embed_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/self.embed_dim)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)


    def forward(self, x):
        """
        Args:
            x: input vector
        Returns:
            x: output
        """
      
        # make embeddings relatively larger
        x = x * math.sqrt(self.embed_dim)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.pe[:,:seq_len], requires_grad=False)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model= 256, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pe = torch.zeros(max_len, d_model)
       
        self.pe[:, 0::2] = torch.sin(position * div_term)#for even terms sin 
        self.pe[:, 1::2] = torch.cos(position * div_term)#for odd terms cos
       
        self.b = nn.Linear(in_features=d_model, out_features=d_model)
        self.c = nn.Linear(in_features=d_model, out_features=d_model)

    def forward(self, t) :
        return self.c(self.b(self.pe[:t.size(0)]))

class MHAVEc(nn.Module):
    def __init__(self,embVecSize,cross= False):
        super().__init__()
       
        self.emb = embVecSize
        self.state = nn.LazyLinear(self.emb)
        self.cross = cross
        
    def forward(self,InputEmbedding,encoderOutput=None):
        query = self.state(InputEmbedding)
        key = self.state(InputEmbedding)
        value = self.state(InputEmbedding)
        if self.cross : 
            assert encoderOutput != None
            key = encoderOutput
            value = encoderOutput
            
        multihead_attn = nn.MultiheadAttention(self.emb, 8)
        attn_output, _ = multihead_attn(query, key, value)
        return attn_output
      
class Encoder(nn.Module):
    def __init__(self,embeddingSize):
        super().__init__()
        self.mha = MHAVEc(embeddingSize)
        self.AddNorm = nn.LayerNorm(embeddingSize)
        self.FeedForward = nn.Sequential(
            nn.Linear(embeddingSize,80),
            nn.ReLU(),
            nn.Linear(80,embeddingSize)
        )
    def forward(self,input):
        o1 = self.mha(input)
        o2 = self.AddNorm(o1+input)
        o3 = self.FeedForward(o2)
        o4 = self.AddNorm(o3+o2)
        return o4

class Decoder(nn.Module):
    def __init__(self ,embeddingSize):
        super().__init__()
        self.mhaCross = MHAVEc(embeddingSize,cross = True)
        self.mha = MHAVEc(embeddingSize)
        self.AddNorm = nn.LayerNorm(embeddingSize)
        self.FeedForward = nn.Sequential(
            nn.Linear(embeddingSize,80),
            nn.ReLU(),
            nn.Linear(80,embeddingSize)
        )
    def forward(self,x,encoderOuput):
        o1 = self.mha(x)
        # print(o1.shape)
        # print(x.shape)
        o2 = self.AddNorm(o1+x)
        # print(o2.shape)
        o3 = self.mhaCross(o2,encoderOuput)
        # print(o3.shape)
        o4 = self.AddNorm(o2+o3)
        # print(o4.shape)
        o5 = self.FeedForward(o4)
        # print(o5.shape)
        o6 = self.AddNorm(o5+o4)
        # print(o6.shape)
        return o6
    
class TransFrameWise(nn.Module):
    """  """
    def __init__(self,embVecSize= 256,RobertaWordEmbSize = 768,Encoder=Encoder,Decoder=Decoder,training=True,inference = True):
        super().__init__()
        self.training = training 
        self.inference = inference
        self.embVecSize = embVecSize
        self.dense_1 = nn.LazyLinear(self.embVecSize)
        self.dense2 = nn.LazyLinear(self.embVecSize)
        self.dense_2 = nn.Linear(self.embVecSize,1)
        self.act_fn = nn.SiLU()
        self.RobertaWordEmbSize = RobertaWordEmbSize
        self.lossf = nn.MSELoss()
        self.EncSplitsReq = []
        self.dense_3 = nn.Linear(80,1)
        self.enc = Encoder(self.embVecSize)
        self.dec = Decoder(self.embVecSize)
        self.IPpositionalEncoding = PositionalEncoding(self.embVecSize)
        self.IPpositionalEncoding1 = PositionalEmbedding(256,256)
        
    def forward(self,TargetSequnce,input):
        decoderInput = []
        for i in TargetSequnce:
            self.EncSplitsReq.append(len(i.split()))
            inputs = tokenizer(i, return_tensors="pt")
            outputs = model(**inputs) 
            decoderInput.append(outputs[1])
        decIn = torch.stack(decoderInput)

        
        words = int(numpy.mean(self.EncSplitsReq))
 
        decoderBatchedinput = []
        for i in decIn:
            outputEmbs =  torch.split(i,int(self.RobertaWordEmbSize/words),dim=1)
            o = torch.cat(outputEmbs)
            o = self.dense_1(o)
            decoderBatchedinput.append(o)       
           
        DecoderText = torch.stack(decoderBatchedinput).permute(1,0,2)
     
        bs,C,_ = input.shape
        inputs = []
        for a in input:
            #diving input spec to frames
            i = self.dense2(a.view(words,C,-1))
         
            i = self.IPpositionalEncoding1(i)
            b = self.dense_3(self.enc(i).permute(0,2,1)).squeeze()
          
            inputs.append(b)
        EncoderHiddenStates = torch.stack(inputs).permute(1,0,2)
        print(EncoderHiddenStates.shape,'Encoder hidden state shape is') # 4,3,256
      
        dec = Decoder(self.embVecSize)
        
        for i in range(words):
            BatchedEncRep = EncoderHiddenStates[i] #BS,256            
            if self.training:
                BatchedDecText = DecoderText[i] #BS,256
                print(BatchedDecText.shape,']]333')
                predicted = dec(BatchedDecText,BatchedEncRep)
                print(predicted.shape)
                if (i+1 < words):
                    actual = DecoderText[i+1]
                    loss = self.lossf(actual,predicted)
                    print(loss)
                    # loss.backward()
            # # #for inference time 
            if self.inference:
                sos = torch.randn((1,256))
                pred1 = dec(sos,BatchedEncRep)  
                print(pred1.shape)
                subPreds = torch.cat((sos,pred1)) 
                print(subPreds.shape)
                # for i in range(words-1):
                # EncOUT = EncoderHiddenState[i]   
                # pred1 = dec(sos,EncOUT)    
                # subPreds = pred1 
                # print(subPreds.shape)
                # for i in range(words-1):
                    #     EncOUT = EncoderHiddenState[i+1] #3,256
                    #     print(subPreds.shape,'-*syb')
                    #     pred = dec(subPreds,EncOUT)
                    #     print(pred.shape,'pred')
                    #     subPreds = torch.cat((subPreds,pred))
                
BatchTargetSequence = ['my name is ritul and i live here','i study math here','hi hello']
input = torch.randn((3,80,800))
        
t = TransFrameWise()
a = t(BatchTargetSequence,input)
