import math
import numpy 
import torch
import torch.nn as nn
import torch.nn.functional as F
       
class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1,1,1)
        
    def forward(self,embeddingVector):       
        out = self.conv(embeddingVector)
        return out 

class MHAarchitecture(nn.Module):
    def __init__(self) :
        super().__init__()  
        self.layer1 = nn.Conv1d(1,1,5)
        self.relu = nn.ReLU()
        self.layer2 = nn.Conv1d(1,1,9)
        self.lastLinear = nn.LazyLinear(512)
        
    def forward(self,embeddingVector):        
        t = torch.mul(self.lastLinear(self.layer2(self.relu(self.layer1(embeddingVector)))),embeddingVector)          
        return t
        
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pe = torch.zeros(max_len, 1, d_model)
        self.pe[:, 0, 0::2] = torch.sin(position * div_term)#for even terms sin 
        self.pe[:, 0, 1::2] = torch.cos(position * div_term)#for odd terms cos

    def forward(self, x) :
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Transformers(nn.Module):
    def __init__(self,att_arch,attBlock,Attention,num_blocks,num_heads,embedding_size=512) :
        super().__init__()
        self.embedding_size = embedding_size
        self.AdAvgPool = nn.AdaptiveAvgPool1d(self.embedding_size)
        self.att_arch = att_arch
        self.num_heads = num_heads
        self.attBlock = attBlock
        self.Attention = Attention
        self.n = num_blocks
        self.AudioEmbedding = nn.LazyLinear(self.embedding_size)
        
    def forward(self,input):
        positionEncoding = PositionalEncoding(input.shape[-1])
        posEncs = positionEncoding(input)
        encIpData = input+posEncs
        adaptiveAvgAttfirst = self.AdAvgPool(encIpData)
        embedding = self.AudioEmbedding(input)
        SinglePointAttention = torch.mul(embedding,adaptiveAvgAttfirst)
        embeddingOut = SinglePointAttention
        bs = embeddingOut.shape[0]
        for _ in range(self.n) :
            if self.Attention == 'self':
                a = AttentionBlock()
                out_attention = a(embeddingOut) #5,1,512 
                embeddingOut = torch.mul(out_attention,embeddingOut)
            
            if self.Attention == 'multi':
                a = MHAarchitecture()
                prev = 0
                for _ in range(self.num_heads):
                    prev += a(embeddingOut)
                embeddingOut = prev
                
        final_embeddign = embeddingOut
        print(final_embeddign.shape)
     

input = torch.randn((5,1,32000))  
#t = Transformers(None,None,'multi',2,4,512)
t = Transformers(None,None,'self',2,0)
t(input)
  
        