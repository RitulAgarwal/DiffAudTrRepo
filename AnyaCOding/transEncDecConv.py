
from typing import Dict, Iterable, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Tuple, Union


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = nn.Conv1d(n_state, n_state,kernel_size=1)
        self.key = nn.Conv1d(n_state, n_state, bias=False,kernel_size=1)
        self.value = nn.Conv1d(n_state, n_state,kernel_size=1)
        self.out = nn.Conv1d(n_state, n_state,kernel_size=1)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x.permute(0, 2, 1)

        q = self.query(x)
     
    
        if kv_cache is None or xa is None or self.key not in kv_cache:
            k = self.key(x if xa is None else xa)
            
            v = self.value(x if xa is None else xa)
            
        else:
            k = kv_cache[self.key]
            v = kv_cache[self.value]
     
        wv, qk = self.qkv_attention(q, k, v, mask)
        wv = wv.permute(0,2,1)
        return self.out(wv), qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        _, n_state, n_ctx = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(q.shape[0],q.shape[2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(k.shape[0],k.shape[2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(v.shape[0],v.shape[2], self.n_head, -1).permute(0, 2, 1, 3).to(q.dtype)
 
        qk = q @ k

    
        
        if mask is not None:
          
            qk = qk + mask[:n_ctx, :n_ctx]
           
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)

        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = nn.LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = nn.LayerNorm(n_state) if cross_attention else None

       
    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        
        x = x.permute(0,2,1) + self.attn(self.attn_ln(x.float()), mask=mask, kv_cache=kv_cache)[0]
     
        if self.cross_attn:
           
            x=x.permute(0,2,1)
          
            x = x.permute(0,2,1) + self.cross_attn(self.cross_attn_ln(x.float()),  kv_cache=kv_cache)[0]
            
        return x


class AudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=1)
        self.positional_embedding = sinusoids(n_ctx, n_state)
        self.AttBlock = ResidualAttentionBlock(n_state, n_head)
        self.ln_post = nn.LayerNorm(n_state)
        self.conv3 = nn.Conv1d(n_ctx, n_ctx, kernel_size=1)

    def forward(self, x: Tensor):
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)
        x = self.AttBlock(x)
        x = self.ln_post(x.permute(0,2,1))
        x = self.conv3(x)
        return x

class Decoder(nn.Module):
    def __init__(
        self, n_ctx: int, n_state: int, n_head: int, n_mels: int
    ):
        super().__init__()
        self.AttBlock = ResidualAttentionBlock(n_state, n_head, cross_attention=True)
        self.ln = nn.LayerNorm(n_state)
        self.conv3 = nn.Conv1d(n_state, n_mels, kernel_size=1)
        self.mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)

    def forward(self, x: Tensor,  kv_cache: Optional[dict] = None):
        x = self.AttBlock(x, mask=self.mask, kv_cache=kv_cache)       
        x = self.ln(x.permute(0,2,1))
        x = x.permute(0, 2, 1)
        x = self.conv3(x)
        return x 
        

@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
  
#input data = BS,80,883


class Transformer(nn.Module):
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims
        if (self.dims.n_audio_state % self.dims.n_audio_head != 0) or (self.dims.n_text_state % self.dims.n_text_head != 0) :
            raise Exception('size of embedding vector must be divisible by number of attention heads ')
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
        )
        self.decoder = Decoder(
       
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_mels
          
        )
  
    def embed_audio(self, 
                    mel: torch.Tensor
                    ):
        
        return self.encoder(mel)

    def logits(self, 
               tokens: torch.Tensor, 
               audio_features: torch.Tensor
               ):
        
        return self.decoder(tokens, audio_features)

    def forward(
        self,
        mel: torch.Tensor
        ) -> Dict[str, torch.Tensor]:
        # print(self.encoder(mel).shape)
        return self.decoder(self.encoder(mel))

from torchinfo import summary


from torchinfo import summary

dims = ModelDimensions(80,883,128,4,883,128,4)

model = Transformer(dims)
o = model.forward(torch.randn((5,80,883)))
print(o.shape)
summary(model,input_data=torch.randn((5,80,883)),depth=8)