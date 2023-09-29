import torch.nn as nn  
import torch
import numpy as np
import torch.nn.functional as F  
import random 
     
class DynamicConv(nn.Module):
    def __init__(self,filters=12,out_ch = 4,in_channel = 1,kernel_size=[3,7,5,9,11]):
        super().__init__()
        self.filters = filters 
        self.out_ch = out_ch 
        self.in_channel = in_channel   
        self.kernel_size = kernel_size    
        self.attention_module = nn.Sequential(
            nn.AdaptiveAvgPool1d((256)),
            nn.Linear(256, 128),
            nn.ReLU(),  
            nn.Linear(128,filters),
            ) 
        self.softmax = nn.Softmax(dim=2)
        kernel_size = random.choices(kernel_size, k=filters)
        kernel_size.sort()
        print(kernel_size)
        self.kernels = nn.ParameterList([(torch.randn((out_ch, in_channel,kernel_size[i]), dtype=torch.float32)) for i in range(filters)])
        self.BN = nn.BatchNorm1d(out_ch)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        y = self.softmax(self.attention_module(x)) # torch.Size([1, 1, 12])
        self.kernel = [self.kernels[i] * y[0][0][i] for i in range(len(self.kernels))] 
        print(self.kernel[0].shape)
        ouput = F.conv1d(x,self.kernel[0])
        max_seq_len = ouput.shape[-1]
        for i in self.kernel[1:]:
            print(i.shape)
            a = F.conv1d(x,i)
            a = F.pad(a,(0,max_seq_len-a.shape[-1]))
            ouput += a
        output = self.activation(self.BN(ouput))
        print(output.shape)
        return output
        
m = DynamicConv(filters = 2,kernel_size=[3])
from torchinfo import summary
summary(m, (1,1,32000))
m.forward(x=(torch.randn(1,1,32000)))