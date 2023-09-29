import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import warnings
import matplotlib.pyplot as plt
warnings.simplefilter("ignore")
# 4 block of transfomer in encoder and decoder 

class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        """
        Args:
            vocab_size: size of vocabulary
            embed_dim: dimension of embeddings
        """
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
    def forward(self, x):
        """
        Args:
            x: input vector
        Returns:
            out: embedding vector
        """
        out = self.embed(x)
        return out
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
  
        if x.ndim == 2 :
            x = x.unsqueeze(0)
      
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.pe[:,:seq_len], requires_grad=False)
        return x       
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=512, n_heads=4):
        """
        Args:
            embed_dim: dimension of embeding vector output
            n_heads: number of self attention heads
        """
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim    #512 dim
        self.n_heads = n_heads   #8
        self.single_head_dim = int(self.embed_dim / self.n_heads)   #512/8 = 64  . each key,query, value will be of 64d
       
        #key,query and value matrixes    #64 x 64   
        self.query_matrix = nn.Linear(self.single_head_dim , self.single_head_dim ,bias=False)  # single key matrix for all 8 keys #512x512
        self.key_matrix = nn.Linear(self.single_head_dim  , self.single_head_dim, bias=False)
        self.value_matrix = nn.Linear(self.single_head_dim ,self.single_head_dim , bias=False)
        self.out = nn.Linear(self.n_heads*self.single_head_dim ,self.embed_dim) 

    def forward(self,key,query,value,mask=None):    #batch_size x sequence_length x embedding_dim    # 32 x 10 x 512
        
        """
        Args:
           key : key vector
           query : query vector
           value : value vector
           mask: mask for decoder
        
        Returns:
           output vector from multihead attention
        """
        batch_size = key.size(0)
        seq_length = key.size(1)
        
        # query dimension can change in decoder during inference. 
        # so we cant take general seq_length
        seq_length_query = query.size(1)
        
        # 32x10x512
        key = key.view(batch_size, seq_length, self.n_heads, self.single_head_dim)  #batch_size x sequence_length x n_heads x single_head_dim = (32x10x8x64)
        query = query.view(batch_size, seq_length_query, self.n_heads, self.single_head_dim) #(32x10x8x64)
        value = value.view(batch_size, seq_length, self.n_heads, self.single_head_dim) #(32x10x8x64)
       
        k = self.key_matrix(key)       # (32x10x8x64)
        q = self.query_matrix(query)   
        v = self.value_matrix(value)

        q = q.transpose(1,2)  # (batch_size, n_heads, seq_len, single_head_dim)    # (32 x 8 x 10 x 64)
        k = k.transpose(1,2)  # (batch_size, n_heads, seq_len, single_head_dim)
        v = v.transpose(1,2)  # (batch_size, n_heads, seq_len, single_head_dim)
       
        # computes attention
        # adjust key for matrix multiplication
        k_adjusted = k.transpose(-1,-2)  #(batch_size, n_heads, single_head_dim, seq_ken)  #(32 x 8 x 64 x 10)
        product = torch.matmul(q, k_adjusted)  #(32 x 8 x 10 x 64) x (32 x 8 x 64 x 10) = #(32x8x10x10)
      
        
        # fill those positions of product matrix as (-1e20) where mask positions are 0
        if mask is not None:
             product = product.masked_fill(mask == 0, float("-1e20"))

        #divising by square root of key dimension
        product = product / math.sqrt(self.single_head_dim) # / sqrt(64)

        #applying softmax
        scores = F.softmax(product, dim=-1)
 
        #mutiply with value matrix
        scores = torch.matmul(scores, v)  ##(32x8x 10x 10) x (32 x 8 x 10 x 64) = (32 x 8 x 10 x 64) 
        
        #concatenated output
        concat = scores.transpose(1,2).contiguous().view(batch_size, seq_length_query, self.single_head_dim*self.n_heads)  # (32x8x10x64) -> (32x10x8x64)  -> (32,10,512)
        
        output = self.out(concat) #(32,10,512) -> (32,10,512)
       
        return output
class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=2, n_heads=4):
        super(DecoderBlock, self).__init__()

        """
        Args:
           embed_dim: dimension of the embedding
           expansion_factor: fator ehich determines output dimension of linear layer
           n_heads: number of attention heads
        
        """
        self.attention = MultiHeadAttention(embed_dim, n_heads=8)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.2)
        self.transformer_block = TransformerBlock(embed_dim, expansion_factor, n_heads)
        
    
    def forward(self, key, query, x,mask):
        
        """
        Args:
           key: key vector
           query: query vector
           value: value vector
           mask: mask to be given for multi head attention 
        Returns:
           out: output of transformer block
    
        """
        
        #we need to pass mask mask only to fst attention
        attention = self.attention(x,x,x,mask=mask) #32x10x512
        value = self.dropout(self.norm(attention + x))
        
        out = self.transformer_block(key, query, value)

        
        return out
class TransformerDecoder(nn.Module):
    def __init__(self, target_vocab_size, embed_dim, seq_len, num_layers=2, expansion_factor=2, n_heads=4):
        super(TransformerDecoder, self).__init__()
        """  
        Args:
           target_vocab_size: vocabulary size of taget
           embed_dim: dimension of embedding
           seq_len : length of input sequence
           num_layers: number of encoder layers
           expansion_factor: factor which determines number of linear layers in feed forward layer
           n_heads: number of heads in multihead attention
        
        """
        self.word_embedding = nn.Linear(1, embed_dim)
        self.position_embedding = PositionalEmbedding(seq_len, embed_dim)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_dim, expansion_factor=2, n_heads=4) 
                for _ in range(num_layers)
            ]

        )
        self.fc_out = nn.Linear(embed_dim, target_vocab_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, enc_out, mask):
        
        """
        Args:
            x: input vector from target
            enc_out : output from encoder layer
            trg_mask: mask for decoder self attention
        Returns:
            out: output vector
        """
       
        x = x[:,:,None]
        x = self.word_embedding(x)  #32x10x512
       
        x = self.position_embedding(x) #32x10x512
        
        x = self.dropout(x)
       
        for layer in self.layers:
            x = layer(enc_out, x, enc_out, mask) 
          
        out = F.softmax(self.fc_out(x))
        
        return out
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=2, n_heads=4):
        super(TransformerBlock, self).__init__()
        
        """
        Args:
           embed_dim: dimension of the embedding
           expansion_factor: fator ehich determines output dimension of linear layer
           n_heads: number of attention heads
        
        """
        self.attention = MultiHeadAttention(embed_dim, n_heads)
        
        self.norm1 = nn.LayerNorm(embed_dim) 
    
        self.FF1 = nn.Linear(embed_dim, expansion_factor*embed_dim)
        self.FF2 = nn.ReLU()
        self.FF3 = nn.Linear(expansion_factor*embed_dim, embed_dim)
       
        self.dropout2 = nn.Dropout(0.2)

    def forward(self,key,query,value):
        
        """
        Args:
           key: key vector
           query: query vector
           value: value vector
           norm2_out: output of transformer block
        
    """
       
        
        attention_out = self.attention(key,query,value) 
        attention_residual_out = attention_out + value 
        norm1_out = self.dropout2(self.norm1(attention_residual_out))

        feed_fwd_out = self.FF3(self.FF2(self.FF1(norm1_out))) 
        feed_fwd_residual_out = feed_fwd_out + norm1_out
        norm2_out = self.dropout2(self.norm1(feed_fwd_residual_out))

        return norm2_out
class TransformerEncoder(nn.Module):
    """
    Args:
        seq_len : length of input sequence
        embed_dim: dimension of embedding
        num_layers: number of encoder layers
        expansion_factor: factor which determines number of linear layers in feed forward layer
        n_heads: number of heads in multihead attention
        
    Returns:
        out: output of the encoder
    """
    def __init__(self, seq_len, vocab_size, embed_dim, num_layers=2, expansion_factor=2, n_heads=4):
        super(TransformerEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_layer = nn.Linear(1, embed_dim)
        self.positional_encoder = PositionalEmbedding(seq_len, embed_dim)
        self.Lin = nn.Linear(vocab_size,1)
        self.layers = nn.ModuleList([TransformerBlock(embed_dim, expansion_factor, n_heads) for i in range(num_layers)])
    
    def forward(self, x):
      
        x = x[:,:,:,None]#.view(-1)
   
        embed_out = self.embedding_layer(x)
    
       
        embed_out = embed_out.permute(0,1,3,2)
        embed_out = self.Lin(embed_out).squeeze()
    
        out = self.positional_encoder(embed_out)
  
        for layer in self.layers:
            out = layer(out,out,out)
      
        return out  
class Transformer(nn.Module):
    def __init__(self, embed_dim, src_vocab_size, target_vocab_size, seq_length,num_layers=2, expansion_factor=2, n_heads=4):
        super(Transformer, self).__init__()
        
        """  
        Args:
           embed_dim:  dimension of embedding 
           src_vocab_size: vocabulary size of source
           target_vocab_size: vocabulary size of target
           seq_length : length of input sequence
           num_layers: number of encoder layers
           expansion_factor: factor which determines number of linear layers in feed forward layer
           n_heads: number of heads in multihead attention
        
        """
        
        self.target_vocab_size = target_vocab_size
        self.Lin = nn.Linear(target_vocab_size,1)
        self.encoder = TransformerEncoder(seq_length, src_vocab_size, embed_dim, num_layers=num_layers, expansion_factor=expansion_factor, n_heads=n_heads)
        self.decoder = TransformerDecoder(target_vocab_size, embed_dim, seq_length, num_layers=num_layers, expansion_factor=expansion_factor, n_heads=n_heads)
        
    
    def make_trg_mask(self, trg):
        """
        Args:
            trg: target sequence
        Returns:
            trg_mask: target mask
        """
        batch_size, trg_len = trg.shape
        # returns the lower triangular part of matrix filled with ones
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            batch_size, 1, trg_len, trg_len
        )
        return trg_mask    

    def decode(self,src,trg):
        """
        for inference
        Args:
            src: input to encoder 
            trg: input to decoder
        out:
            out_labels : returns final prediction of sequence
        """
        trg_mask = self.make_trg_mask(trg)
        enc_out = self.encoder(src)
        out_labels = []
        batch_size,seq_len = src.shape[0],src.shape[1]
        #outputs = torch.zeros(seq_len, batch_size, self.target_vocab_size)
        out = trg
        for i in range(seq_len): #10
            out = self.decoder(out,enc_out,trg_mask) #bs x seq_len x vocab_dim
            # taking the last token
            out = out[:,-1,:]
     
            out = out.argmax(-1)
            out_labels.append(out.item())
            out = torch.unsqueeze(out,axis=0)
          
        
        return out_labels
    
    def forward(self, src, trg):
        """
        Args:
            src: input to encoder 
            trg: input to decoder
        out:
            out: final vector which returns probabilities of each target word
        """
        src = src.permute(0,2,1)
   
        trg_mask = self.make_trg_mask(trg)
        enc_out = self.encoder(src)
    
        outputs = self.decoder(trg, enc_out, trg_mask).permute(0,2,1)
  
        return outputs
class Unet(nn.Module):
 
    def __init__(self, embed_dim=128, src_vocab_size=2, target_vocab_size=80, seq_length=800,num_layers=2, expansion_factor=2, n_heads=4,KernelSizeList = [13,9,5]):

        super().__init__()
        self.embed_dim = embed_dim
        self.src_vocab_size = src_vocab_size
        self.target_vocab_size = target_vocab_size
        self.seq_length = seq_length
        self.num_layers=num_layers 
        self.expansion_factor=expansion_factor
        self.n_heads=n_heads  
        self.KernelSizeList = KernelSizeList
        self.t1 = Transformer(self.embed_dim,self.src_vocab_size,self.src_vocab_size,self.seq_length,self.num_layers,self.expansion_factor,self.n_heads)
        self.seq_length1 = self.seq_length-self.KernelSizeList[0]+1 
        self.t2 = Transformer(self.embed_dim,self.src_vocab_size*4,self.src_vocab_size*4,self.seq_length1,self.num_layers,self.expansion_factor,self.n_heads)
        self.seq_length2 = self.seq_length-self.KernelSizeList[0]+1-self.KernelSizeList[1]+1
        self.t3 = Transformer(self.embed_dim,self.src_vocab_size*8,self.src_vocab_size*8,self.seq_length2,self.num_layers,self.expansion_factor,self.n_heads)
        self.seq_length3 = self.seq_length-self.KernelSizeList[0]+1-self.KernelSizeList[1]+1-self.KernelSizeList[2]+1
        self.t4 = Transformer(self.embed_dim,self.src_vocab_size*16,self.src_vocab_size*16,self.seq_length3,self.num_layers,self.expansion_factor,self.n_heads)
        self.t5 = Transformer(self.embed_dim,self.src_vocab_size*16,self.src_vocab_size*16,self.seq_length3,self.num_layers,self.expansion_factor,self.n_heads)
        self.t6 = Transformer(self.embed_dim,self.src_vocab_size*24,self.src_vocab_size*24,self.seq_length2,self.num_layers,self.expansion_factor,self.n_heads)
        self.t7 = Transformer(self.embed_dim,self.src_vocab_size*32,self.src_vocab_size*32,self.seq_length1,self.num_layers,self.expansion_factor,self.n_heads)
        self.t8 = Transformer(self.embed_dim,self.src_vocab_size*40,self.src_vocab_size*40,self.seq_length,self.num_layers,self.expansion_factor,self.n_heads)
        self.Conv1 = nn.Conv1d(src_vocab_size,src_vocab_size*4,self.KernelSizeList[0])
        self.Conv2 = nn.Conv1d(src_vocab_size*4,src_vocab_size*8,self.KernelSizeList[1])
        self.Conv3 = nn.Conv1d(src_vocab_size*8,src_vocab_size*16,self.KernelSizeList[2])
        self.Conv4 = nn.ConvTranspose1d(src_vocab_size*16,src_vocab_size*24,self.KernelSizeList[2])
        self.Conv5 = nn.ConvTranspose1d(src_vocab_size*24,src_vocab_size*32,self.KernelSizeList[1])
        self.Conv6 = nn.ConvTranspose1d(src_vocab_size*32,src_vocab_size*40,self.KernelSizeList[0])
        self.flat = nn.Flatten()
        self.lin = nn.Linear(15616,1024)
    def forward(self,x,text):
       
        trans1 = self.t1(x,text)
       
        trans1 = self.Conv1(trans1)
   
        trans1 = self.t2(F.normalize(trans1),text[:,:self.seq_length1])
      
        trans1 = self.Conv2(trans1)
       
        trans1 = self.t3(F.normalize(trans1),text[:,:self.seq_length2])
     
        trans1 = self.Conv3(trans1)
       
        trans1 = self.t4(F.normalize(trans1),text[:,:self.seq_length3])
       
        LatentREp = self.lin(self.flat(trans1))
   #decoder
       
        trans1 = self.t5(F.normalize(trans1),text[:,:self.seq_length3])
     
        trans1 = self.Conv4(trans1)
       
        trans1 = self.t6(F.normalize(trans1),text[:,:self.seq_length2])
    
        trans1 = self.Conv5(trans1)
      
        trans1 = self.t7(F.normalize(trans1),text[:,:self.seq_length1])
    
        trans1 = self.Conv6(trans1)
   
        trans1 = self.t8(F.normalize(trans1),text[:,:self.seq_length])
       
        return LatentREp,trans1       

    
       