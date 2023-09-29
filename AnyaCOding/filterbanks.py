import torch.nn as nn 
from asteroid_filterbanks import Encoder, ParamSincFB
import torch

C = 1024
m = nn.conv1 = Encoder(
            ParamSincFB(
                C // 4,
                1001,
                stride= 157,
            )
        )

input = torch.randn(1, 1, 32000)
output = m(input)
print(output.shape)

# Args:
#         n_filters (int): Number of filters. Half of `n_filters` (the real
#             parts) will have parameters, the other half will correspond to the
#             imaginary parts. `n_filters` should be even.
#         kernel_size (int): Length of the filters.
#         stride (int, optional): Stride of the convolution. If None (default),
#             set to ``kernel_size // 2``.
#         sample_rate (float, optional): The sample rate (used for initialization).
#         min_low_hz (int, optional): Lowest low frequency allowed (Hz).
#         min_band_hz (int, optional): Lowest band frequency allowed (Hz).










###CONV LENGTH CHANGING FB

import torch.nn as nn 
import torch

m = nn.Conv1d(1, 80,kernel_size=30000, stride=160,padding=0, dilation=1)
input = torch.randn(1, 1, 32000)
output = m(input)
print(output.shape)