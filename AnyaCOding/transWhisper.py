import base64
import gzip
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Tuple, Union



@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super()


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


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
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

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
            
        #print(k.shape,'this is key shape')
        #print(v.shape,'this is value shape')
        
        wv, qk = self.qkv_attention(q, k, v, mask)
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
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        # print(x.dtype,'this is the type of x ')
        # print(self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0].dtype,'type os x ading')
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        if n_state % n_head != 0 :
            raise Exception('size of embedding vector must be divisible by number of attention heads ')
        self.InitialLayer = nn.Conv1d(2, n_mels, kernel_size=1) # 32x32 => 16x16
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

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

# ae = AudioEncoder(80,768,512,4,3)
# b = ae(torch.randn((5,2,768)))
# #print(b.shape)


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
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)
###BASICALLY EMBEDDING LAYER MEI VOCAB WORDS KI EK MAPPING HOTI HAI JO DUSRE PARAM KE BARABBAR SIZE KI HOTI HAI 
###ABB HUM JAB EMBEDDING LAYER LGTA HAI INPUT PE TOH VO every INPUT BHI USSI SIZE KA HO JATA HAI 
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
out = dec(torch.randint(1000,(5,768)),torch.randint(1000,(5,768,512)))

from torchinfo import summary
summary(model=dec,input_data=[torch.randint(100,(5,768)).to(torch.int64),torch.randint(100,(5,768,512))],device=torch.device("cuda:1"))


class Transformer(nn.Module):
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )
        all_heads = torch.zeros(
            self.dims.n_text_layer, self.dims.n_text_head, dtype=torch.bool
        )
        all_heads[self.dims.n_text_layer // 2 :] = True
        self.register_buffer("alignment_heads", all_heads.to_sparse(), persistent=False)

    def set_alignment_heads(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool
        ).copy()
        mask = torch.from_numpy(array).reshape(
            self.dims.n_text_layer, self.dims.n_text_head
        )
        self.register_buffer("alignment_heads", mask.to_sparse(), persistent=False)

    def embed_audio(self, mel: torch.Tensor):#mel is mel spec of audio 
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features)

    def forward(
        self, mel: torch.Tensor, tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
       
        return self.decoder(tokens, self.encoder(mel))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab == 51865

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.dims.n_text_ctx:
                # save as-is, for the first token or cross attention
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    # detect_language = detect_language_function
    # transcribe = transcribe_function
    # decode = decode_function

dims = ModelDimensions(80,768,512,4,3,5000,768,512,4,3)

# model = Transformer(dims)
# o = model.forward(torch.randn((5,2,768)),torch.randint(100,(5,768)))


@dataclass(frozen=True)
class DecodingOptions:
    # language that the audio is in; uses detected language if None
    language: Optional[str] = None
    # sampling-related options
    temperature: float = 0.0
    sample_len: Optional[int] = None  # maximum number of tokens to sample
    best_of: Optional[int] = None  # number of independent sample trajectories, if t > 0
    beam_size: Optional[int] = None  # number of beams in beam search, if t == 0
    patience: Optional[float] = None  # patience in beam search (arxiv:2204.05424)
    # "alpha" in Google NMT, or None for length norm, when ranking generations
    # to select which to return among the beams or best-of-N samples
    length_penalty: Optional[float] = None
    # text or tokens to feed as the prompt or the prefix; for more info:
    # https://github.com/openai/whisper/discussions/117#discussioncomment-3727051
    prompt: Optional[Union[str, List[int]]] = None  # for the previous context
    prefix: Optional[Union[str, List[int]]] = None  # to prefix the current context
    # list of tokens ids (or comma-separated token ids) to suppress
    # "-1" will suppress a set of symbols as defined in `tokenizer.non_speech_tokens()`
    suppress_tokens: Optional[Union[str, Iterable[int]]] = "-1"
    suppress_blank: bool = True  # this will suppress blank outputs
    # timestamp sampling options
    without_timestamps: bool = False  # use <|notimestamps|> to sample text tokens only
    max_initial_timestamp: Optional[float] = 1.0
    # implementation details
    fp16: bool = True  # use fp16 for most of the calculation


@dataclass(frozen=True)
class DecodingResult:
    audio_features: Tensor
    language: str
    language_probs: Optional[Dict[str, float]] = None
    tokens: List[int] = field(default_factory=list)
    text: str = ""
    avg_logprob: float = np.nan
    no_speech_prob: float = np.nan
    temperature: float = np.nan
    compression_ratio: float = np.nan


class Inference:
    def logits(self, tokens: Tensor, audio_features: Tensor) -> Tensor:
        """Perform a forward pass on the decoder and return per-token logits"""
        raise NotImplementedError

    def rearrange_kv_cache(self, source_indices) -> None:
        """Update the key-value cache according to the updated beams"""
        raise NotImplementedError

    def cleanup_caching(self) -> None:
        """Clean up any resources or hooks after decoding is finished"""
        pass

class PyTorchInference(Inference):
    def __init__(self, model, initial_token_length: int):
        self.model= model
        self.initial_token_length = initial_token_length
        self.kv_cache = {}
        self.hooks = []

        key_modules = [block.attn.key for block in self.model.decoder.blocks]
        value_modules = [block.attn.value for block in self.model.decoder.blocks]
        self.kv_modules = key_modules + value_modules

    def logits(self, tokens: Tensor, audio_features: Tensor) -> Tensor:
        if not self.kv_cache:
            self.kv_cache, self.hooks = self.model.install_kv_cache_hooks()

        if tokens.shape[-1] > self.initial_token_length:
            # only need to use the last token except in the first forward pass
            tokens = tokens[:, -1:]

        return self.model.decoder(tokens, audio_features, kv_cache=self.kv_cache)

    def cleanup_caching(self):
        for hook in self.hooks:
            hook.remove()

        self.kv_cache = {}
        self.hooks = []

    def rearrange_kv_cache(self, source_indices):
        if source_indices != list(range(len(source_indices))):
            for module in self.kv_modules:
                # update the key/value cache to contain the selected sequences
                self.kv_cache[module] = self.kv_cache[module][source_indices].detach()

