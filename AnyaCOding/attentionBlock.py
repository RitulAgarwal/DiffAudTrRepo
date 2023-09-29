import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy
from DDPM.SpecEncoder import SpecEncoder

class SelfAttention(nn.Module):
    def __init__(self, input_dim,input):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.input = input
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)
        return weighted
  

#d_model is the size of the embedding vector    
class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model):# dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.linear = nn.Linear(d_model, d_model)
        # self.v_linear = nn.Linear(d_model, d_model)
        # self.k_linear = nn.Linear(d_model, d_model)
        #self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        # perform linear operation and split into h heads
        k = self.linear(k).view(bs, -1, self.h, self.d_k)
        q = self.linear(q).view(bs, -1, self.h, self.d_k)
        v = self.linear(v).view(bs, -1, self.h, self.d_k)
        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask)#, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output
    
def attention(q, k, v, d_k, mask=None, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)
        
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return output

# mha = MultiHeadAttention(5,512)
# o = mha()


class SMHAPppling(nn.Module):
    def __init__(self, heads,padded_enc_size):
        super().__init__()
        self.D = padded_enc_size
        self.K = heads 
        self.linear = nn.Linear( padded_enc_size, padded_enc_size )
        self.d_k = padded_enc_size // heads
         
    def forward(self,enc_hidden_states):
        bs = enc_hidden_states.size(0)
        h = self.linear(enc_hidden_states)
        print(h.shape)
        h = h.view(bs, -1, self.K, self.d_k).transpose(1, 2)
        print(h.shape)
        U = torch.randn((self.K))
        print(U.shape)
        d = int(numpy.sqrt(self.d_k))
        print(d)
        sum = 0 
        weight = []
        for i in h :
            i = torch.permute(i,(1,2,0))
            print(i.shape,U.shape)
            attention_score = torch.exp((torch.mul(i, U))/ d )
            print(attention_score.shape)
        #     sum += attention_score
        #     weight[i] = attention_score
        # weight[i] = weight[i]/sum
        # print(weight)
            
          
        
        # # calculate attention using function we will define next
        # scores = attention(q, k, v, self.d_k, mask)#, self.dropout)
        # # concatenate heads and put through final linear layer
        # concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        # output = self.out(concat)
        # return output
# smha = SMHAPppling(8,512)
# a = smha(torch.randn((10,20,512)))   # batch size, seq_len,embediing 

class AttentionBlocGalat(nn.Module):
    def __init__(self,AttentionType,num_heads):
        super().__init__()
        self.typeA = AttentionType        
        
        self.num_heads = num_heads
        
    def forward(self,embeddingVector):
        shape = embeddingVector.shape[-1]
        linear = nn.Linear(shape,shape)
       
        if self.typeA == 'self' and self.num_heads == 0 :
            out = linear(embeddingVector)
            return out
        
        elif self.typeA == 'self' and self.num_heads != 0 :
            Exception('self attention doest need any heads')
            
        elif self.typeA == 'multi':
          
            bs = embeddingVector.shape[0]
            d_k = shape // self.num_heads
            h = embeddingVector.view(bs, -1, self.num_heads, d_k).transpose(1, 2)
            return h 
            
        else:
            Exception('either enter self or multi for sha or mha resp.')
        
        

        
    
