import base64
import gzip
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
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        print(x.dtype,'0')
        q = self.query(x)
        print(q.dtype,'1')
        #print(q.shape,'this is query shape')
    
        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            
            v = self.value(x if xa is None else xa)
            
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]
        print('done tiil her')   
        #print(k.shape,'this is key shape')
        #print(v.shape,'this is value shape')
        
        wv, qk = self.qkv_attention(q, k, v, mask)
        print('done tiil herww')  
        return self.out(wv), qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3).to(q.dtype)

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

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            nn.Linear(n_state, n_mlp), nn.GELU(), nn.Linear(n_mlp, n_state)
        )
        self.mlp_ln = nn.LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        print(x.dtype,'this is the type of x ')
     
        print(self.attn_ln(x.float()).dtype,'this is the type of x ')
        print(self.attn(self.attn_ln(x.float()), mask=mask, kv_cache=kv_cache)[0].dtype,'type os x ading')
        x = x + self.attn(self.attn_ln(x.float()), mask=mask, kv_cache=kv_cache)[0]
        print('yaha')
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x.float()), xa.float(), kv_cache=kv_cache)[0]
            print('ab yaha')
        x = x + self.mlp(self.mlp_ln(x))
        print('done')
        return x


class AudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        if n_state % n_head != 0 :
            raise Exception('size of embedding vector must be divisible by number of attention heads ')
        self.InitialLayer = nn.Conv1d(2, n_mels, kernel_size=1) # 32x32 => 16x16
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = nn.LayerNorm(n_state)

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, 3channels, embedding size of 768)
            the mel spectrogram of the audio
        """
        print(x.shape,'input')
        x = F.gelu(self.InitialLayer(x))
        print(x.shape,'enc 1 layer')
        x = F.gelu(self.conv1(x))
        print(x.shape,'enc 2 layer')
        x = F.gelu(self.conv2(x))
        print(x.shape,'enc 3 layer')
        x = x.permute(0, 2, 1)
        print(x.shape,'enc 4 layer')
        print(self.positional_embedding.shape,'positonal embeddng hsape')
        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)
        print(x.shape,'incorporated pe')
        for block in self.blocks:
            x = block(x)
            print(x.shape,'after residual block')

        x = self.ln_post(x)
        print(x.shape,'final enc ouptut')
        return x

ae = AudioEncoder(80,768,512,4,3)
# b = ae(torch.randn((5,2,768)))
#print(b.shape)
from torchinfo import summary
summary(ae,(5,2,768),device=torch.device('cuda:1'))

class TextDecoder(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [
                ResidualAttentionBlock(n_state, n_head, cross_attention=True)
                for _ in range(n_layer)
            ]
        )
        self.ln = nn.LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        #print(offset)
        #print(self.token_embedding(x).shape)
        #print(self.positional_embedding[offset : offset + x.shape[-1]].shape)
        print(self.token_embedding)
        print(type(x))
        x = (
            self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]]
        )
        print(x.shape,'text and position embediing ')
        
        x = x.to(xa.dtype)
        #print(x.shape,'work 5,768 ke text tokeniser encoder output')
        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)
            #print(x.shape,'not wokr')
            print(x.shape,'after residual block1')
        
        
        x = self.ln(x)
        
        print(x.shape,'after residual block final output')
        
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()
        print(logits.shape,'logits after decoding to produce word ')
        return logits
    
    
dec = TextDecoder(5000,768,512,4,3)
# out = dec(torch.randint(1000,(5,768)).type(torch.LongTensor),torch.randint(1000,(5,768,512)))

from torchinfo import summary
summary(model=dec,input_data=[torch.randint(100,(5,768)),torch.randint(100,(5,768,512))],device=torch.device("cuda:1"))

print(dec)
